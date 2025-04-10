import time
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 10
N_SAMPLES = 200_000
N_FEATURES = 4_000
BATCH_SIZE = 256
LR = 0.2
SPARSE_DENSITY = 0.1
NUM_IMPORTANT_FEATURES = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X_sparse = X_sparse
        self.y = y
        
    def __len__(self):
        return self.X_sparse.shape[0]
    
    def __getitem__(self, idx):
        return self.X_sparse[idx], self.y[idx]

def sparse_collate_fn(batch):
    X_batch = scipy.sparse.vstack([item[0] for item in batch])
    y_batch = np.array([item[1] for item in batch])
    
    X_coo = X_batch.tocoo()
    indices = np.vstack((X_coo.row, X_coo.col))
    
    return (
        torch.sparse_coo_tensor(
            indices=torch.LongTensor(indices),
            values=torch.FloatTensor(X_coo.data),
            size=X_coo.shape
        ).to(DEVICE),
        torch.FloatTensor(y_batch).to(DEVICE)
    )

print("Generating sparse data...")
rng = np.random.RandomState(42)

true_theta = np.zeros(N_FEATURES, dtype=np.float32)
important_indices = rng.choice(N_FEATURES, NUM_IMPORTANT_FEATURES, replace=False)
true_theta[important_indices] = rng.uniform(-0.5, 0.5, size=NUM_IMPORTANT_FEATURES).astype(np.float32)

data = np.zeros(N_SAMPLES * int(N_FEATURES * SPARSE_DENSITY), dtype=np.float32)
indices = np.zeros_like(data, dtype=int)
indptr = np.zeros(N_SAMPLES + 1, dtype=int)

for i in range(N_SAMPLES):
    row_cols = rng.choice(N_FEATURES, int(N_FEATURES * SPARSE_DENSITY), replace=False)
    start_idx = i * int(N_FEATURES * SPARSE_DENSITY)
    end_idx = start_idx + int(N_FEATURES * SPARSE_DENSITY)
    
    indices[start_idx:end_idx] = row_cols
    data[start_idx:end_idx] = rng.uniform(-1, 1, size=int(N_FEATURES * SPARSE_DENSITY)).astype(np.float32)
    indptr[i + 1] = indptr[i] + int(N_FEATURES * SPARSE_DENSITY)

X_sparse = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_SAMPLES, N_FEATURES))
y = X_sparse.dot(true_theta) + 0.1 * rng.randn(N_SAMPLES).astype(np.float32)


dataset = SparseDataset(X_sparse, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                       shuffle=True, collate_fn=sparse_collate_fn)

model = torch.nn.Linear(N_FEATURES, 1).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

print("Training sparse linear model...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.3f}s")

total_time = time.time() - start_time
print(f"Avg epoch time: {(time.time()-start_time)/NUM_EPOCHS:.3f}s")

err = 0.0
for i in range(0, N_FEATURES):
    if true_theta[i] != 0:
        err += abs((model.weight[0, i] - true_theta[i]) / true_theta[i])
err /= NUM_IMPORTANT_FEATURES

print(f"Accuracy: {1.0 - err}")
