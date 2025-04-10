#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <algorithm>
#include <random>

const int NUM_EPOCHS = 10;
const int N_SAMPLES = 200000;
const int N_FEATURES = 4000;
const float SPARSE_DENSITY = 0.1f;
const int BATCH_SIZE = 256;
const float LR = 0.2f;
const int NNZ_PER_ROW = static_cast<int>(N_FEATURES * SPARSE_DENSITY);
const int NUM_IMPORTANT = 200; 
const float BETA = 0.9f; // momentum

struct SparseMatrix
{
    // host ptrs
    int* row_ptr;
    int* col_idx;
    float* values;

    // device ptrs
    int* d_row_ptr;
    int* d_col_idx;
    float* d_values;

    int nnz;
    int rows;
    int cols;
};

SparseMatrix convert_to_sparse(const float* dense, int rows, int cols, float density)
{
    SparseMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    
    mat.nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(dense[i]) > 1e-6) mat.nnz++;
    }
    
    mat.row_ptr = new int[rows + 1];
    mat.col_idx = new int[mat.nnz];
    mat.values = new float[mat.nnz];
    
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        mat.row_ptr[i] = idx;
        for (int j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (fabs(val) > 1e-6) {
                mat.col_idx[idx] = j;
                mat.values[idx] = val;
                idx++;
            }
        }
    }
    mat.row_ptr[rows] = idx;
    return mat;
}

void create_gpu_sparse_matrix(SparseMatrix& mat)
{
    cudaMalloc(&mat.d_row_ptr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&mat.d_col_idx, mat.nnz * sizeof(int));
    cudaMalloc(&mat.d_values, mat.nnz * sizeof(float));
    
    cudaMemcpy(mat.d_row_ptr, mat.row_ptr, (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat.d_col_idx, mat.col_idx, mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat.d_values, mat.values, mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
}

void free_gpu_sparse_matrix(SparseMatrix& mat)
{
    cudaFree(mat.d_row_ptr);
    cudaFree(mat.d_col_idx);
    cudaFree(mat.d_values);
}

__global__ void update_sgd_kernel(
    const int* row_ptr, const int* col_idx, const float* values,
    float* theta, float* velocity, 
    const float* y, float lr, float beta,
    int batch_size, int n_features)
{
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= n_features) return;

    float grad = 0.0f;
    int active_count = 0;

    for (int i = 0; i < batch_size; i++) {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
            if (col_idx[p] == fid) {
                float pred = 0.0f;
                for (int q = row_ptr[i]; q < row_ptr[i+1]; q++) {
                    pred += values[q] * theta[col_idx[q]];
                }
                grad += 2.0f * (pred - y[i]) * values[p];
                active_count++;
                break;
            }
        }
    }

    if (active_count > 0) {
        float eff_lr = lr / (1.0f + __log2f(active_count));
        velocity[fid] = beta * velocity[fid] + (1.0f - beta) * (grad / batch_size);
        theta[fid] -= eff_lr * velocity[fid];
    }
}

void run(int N, int d, int batch_size, float lr, float beta, float *theta_host,
        SparseMatrix &mat, float *y_dev, float *theta_dev, float *velocity_dev)
{
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int batch_start = 0; batch_start < N; batch_start += batch_size) {
            int threads_per_block = 256;
            int blocks = (d + threads_per_block - 1) / threads_per_block;
            int shared_mem_size = threads_per_block * sizeof(float);
            update_sgd_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                mat.d_row_ptr + batch_start,
                mat.d_col_idx,
                mat.d_values,
                theta_dev,
                velocity_dev,
                y_dev + batch_start,
                lr,
                beta,
                std::min(batch_size, N - batch_start),
                d
            );
        }
        
        cudaMemcpy(theta_host, theta_dev, d * sizeof(float), cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Epoch " << epoch + 1 << " | Time: " << elapsed.count() << "s\n";
    }
}

float compute_error(int N_FEATURES, float *true_theta, float *theta, int num_important)
{
    float err = 0.0f;
    for (int i = 0; i < N_FEATURES; i++) {
        if (true_theta[i]) err += std::abs((theta[i] - true_theta[i]) / true_theta[i]);
    }
    return err / num_important;
}

int main()
{
    std::cout << "Generating sparse data...\n";

    float* X_host = new float[N_SAMPLES * N_FEATURES]();
    float* y_host = new float[N_SAMPLES];
    float* theta_host = new float[N_FEATURES]();
    float* grad_host = new float[N_FEATURES];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);

    std::vector<float> true_theta(N_FEATURES, 0.0f);
    std::uniform_int_distribution<int> feat_dist(0, N_FEATURES - 1);
    std::uniform_real_distribution<float> theta_dist(-0.5f, 0.5f);
    
    for (int i = 0; i < NUM_IMPORTANT; i++) {
        int j = feat_dist(rng);
        true_theta[j] = theta_dist(rng);
    }

    std::vector<int> all_cols(N_FEATURES);
    std::iota(all_cols.begin(), all_cols.end(), 0);

    for (int i = 0; i < N_SAMPLES; i++) {
        std::shuffle(all_cols.begin(), all_cols.end(), rng);
        
        for (int k = 0; k < NNZ_PER_ROW; k++) {
            int j = all_cols[k];
            X_host[i * N_FEATURES + j] = value_dist(rng);
        }

        float dot = 0.0f;
        for (int j = 0; j < N_FEATURES; j++) {
            dot += X_host[i * N_FEATURES + j] * true_theta[j];
        }
        y_host[i] = dot + noise_dist(rng);
    }

    SparseMatrix sparse_mat = convert_to_sparse(X_host, N_SAMPLES, N_FEATURES, SPARSE_DENSITY);
    create_gpu_sparse_matrix(sparse_mat);

    float *y_dev, *theta_dev, *velocity_dev;
    cudaMalloc(&y_dev, N_SAMPLES * sizeof(float));
    cudaMalloc(&theta_dev, N_FEATURES * sizeof(float));
    cudaMalloc(&velocity_dev, N_FEATURES * sizeof(float));

    cudaMemcpy(y_dev, y_host, N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(theta_dev, 0, N_FEATURES * sizeof(float));
    cudaMemset(velocity_dev, 0, N_FEATURES * sizeof(float));

    std::cout << "Training sparse linear model...\n";
    auto start = std::chrono::high_resolution_clock::now();
    run(N_SAMPLES, N_FEATURES, BATCH_SIZE, LR, BETA,
            theta_host, sparse_mat, y_dev, theta_dev, velocity_dev);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nFinal results:\n";
    std::cout << "Avg epoch time: " << elapsed.count() / NUM_EPOCHS << "s\n";
    std::cout << "Accuracy: "
              << 1.0f - compute_error(N_FEATURES, true_theta.data(), theta_host, NUM_IMPORTANT)
              << "\n";

    free_gpu_sparse_matrix(sparse_mat);
    delete[] sparse_mat.row_ptr;
    delete[] sparse_mat.col_idx;
    delete[] sparse_mat.values;
    cudaFree(y_dev);
    cudaFree(theta_dev);
    cudaFree(velocity_dev);

    return 0;
}
