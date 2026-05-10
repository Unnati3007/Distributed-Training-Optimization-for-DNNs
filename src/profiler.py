"""
profiler.py — GPU throughput and communication overhead measurement.
Run after training to get a breakdown of compute vs. communication time.
"""

import time
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
import torchvision.models as models


def measure_throughput(model, batch_size: int = 256, n_warmup: int = 5, n_iter: int = 50):
    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warmup
    for _ in range(n_warmup):
        with autocast():
            _ = model(dummy)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        with autocast():
            _ = model(dummy)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    images_per_sec = (batch_size * n_iter) / (elapsed_ms / 1000)
    return images_per_sec


def measure_allreduce_bandwidth(tensor_size_mb: float = 100.0):
    """Measure effective all-reduce bandwidth between GPUs."""
    if not dist.is_initialized():
        print("Process group not initialized — skipping all-reduce benchmark")
        return

    n_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # float32
    tensor = torch.ones(n_elements, device="cuda")

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # All-reduce bandwidth = 2 * (N-1)/N * size / time  (ring all-reduce formula)
    world_size = dist.get_world_size()
    bandwidth_gbps = (2 * (world_size - 1) / world_size * tensor_size_mb * 20) / (elapsed * 1024)
    print(f"All-reduce bandwidth: {bandwidth_gbps:.2f} GB/s ({tensor_size_mb} MB tensor)")
    return bandwidth_gbps


if __name__ == "__main__":
    model = models.resnet50().cuda().eval()
    imgs_per_sec = measure_throughput(model, batch_size=64)
    print(f"Single-GPU throughput: {imgs_per_sec:.0f} images/sec")
