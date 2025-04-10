# SSGD
CUDA-based Stochastic Gradient Descent

## Hardware
CPU: Intel(R) Xeon(R) W-11955M CPU @ 2.60GHz

GPU: NVIDIA RTX A3000

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A3000 Laptop GPU    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   52C    P0             20W /   60W |      15MiB /   6144MiB |      7%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Benchmarks
```
My implementation (50k samples, 1k features)
Avg epoch time: 0.624942s
Accuracy: 0.980763

PyTorch (50k samples, 1k features)
Avg epoch time: 1.701s
Accuracy: 0.954012
```

```
My implementation (100k samples, 2k features)
Avg epoch time: 2.41948s
Accuracy: 0.988917

PyTorch (100k samples, 2k features)
Avg epoch time: 4.811s
Accuracy: 0.970017
```

```
My implementation (200k samples, 4k features)
Avg epoch time: 9.57413s
Accuracy: 0.990615

PyTorch (200k samples, 4k features)
Avg epoch time: 14.227s
Accuracy: 0.980243
```
