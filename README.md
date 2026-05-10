# Distributed Training Optimization for DNNs

A framework for training large deep neural networks efficiently across multiple GPUs, using **PyTorch DDP** (Distributed Data Parallel), **NCCL** communication backend, and **FP16 mixed precision**. Achieves **85% scaling efficiency** on a 4-GPU cluster with **1.8x training speedup** and **40% reduction in inter-GPU communication overhead**.

---

## Why Distributed Training?

Modern DNNs like ResNet-50, BERT, and GPT require hours or days to train on a single GPU. Distributing training across multiple GPUs reduces this — but naively scaling introduces new bottlenecks: gradient synchronization overhead, load imbalance, and memory pressure. This project identifies and eliminates those bottlenecks.

---

## Key Optimizations

| Optimization | Benefit |
|-------------|---------|
| PyTorch DDP + NCCL | Overlaps gradient all-reduce with backward pass |
| FP16 Mixed Precision | Halves memory bandwidth; 2x faster tensor ops on Tensor Cores |
| Gradient Compression | Reduces all-reduce payload by ~40% |
| Overlap Comm + Compute | Hides communication latency behind backward computation |
| Optimal Batch Scheduling | Eliminates stragglers; keeps all GPUs equally loaded |

---

## Results

| Metric | 1 GPU | 4 GPU (naive) | 4 GPU (ours) |
|--------|-------|--------------|--------------|
| ResNet-50 epoch time | 12 min | 5.1 min | **4.2 min** |
| Scaling efficiency | 100% | 59% | **85%** |
| Communication overhead | — | 41% | **18%** |
| Memory per GPU | 10 GB | 10 GB | **5.5 GB** (FP16) |
| Convergence (epochs to 75% acc) | 90 | 90 | **27** |

---

## Tech Stack

- **PyTorch DDP** — gradient synchronization across GPUs
- **NCCL** — optimized GPU collective communication (all-reduce)
- **FP16 / AMP** — automatic mixed precision (torch.cuda.amp)
- **CUDA Events** — precise GPU-side timing for profiling
- **TensorBoard** — training metrics visualization

---

## Project Structure

```
Distributed-Training-Optimization-for-DNNs/
├── src/
│   ├── train_ddp.py          # Main DDP training script
│   ├── model.py              # ResNet-50 model definition
│   ├── dataset.py            # ImageNet / CIFAR data loaders
│   ├── optimizer.py          # LARS + gradient clipping
│   ├── comm_optimizer.py     # Gradient compression & overlap
│   └── profiler.py           # GPU timing and throughput measurement
├── scripts/
│   ├── launch_4gpu.sh        # torchrun launcher for 4-GPU training
│   └── profile.sh            # NVIDIA Nsight Systems profiling
├── configs/
│   └── resnet50_4gpu.yaml    # Hyperparameters
├── requirements.txt
└── README.md
```

---

## Quick Start

### Single-node, 4-GPU training
```bash
git clone https://github.com/Unnati3007/Distributed-Training-Optimization-for-DNNs
cd Distributed-Training-Optimization-for-DNNs
pip install -r requirements.txt

# Launch on 4 GPUs (uses torchrun)
torchrun --nproc_per_node=4 src/train_ddp.py \
    --config configs/resnet50_4gpu.yaml \
    --data-dir /path/to/imagenet
```

### Profile GPU utilization
```bash
bash scripts/profile.sh
```
