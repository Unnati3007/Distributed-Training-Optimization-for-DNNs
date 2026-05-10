"""
train_ddp.py — Distributed Data Parallel Training with Mixed Precision

Launch with:
    torchrun --nproc_per_node=4 train_ddp.py --epochs 90 --batch-size 256
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as datasets


def setup_distributed():
    """Initialize the NCCL process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def build_dataloaders(args, rank, world_size):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), normalize,
    ])
    val_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   val_transform)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=8, pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, rank):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward (FP16 gradients)
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if rank == 0 and step % 100 == 0:
            print(f"  Epoch {epoch} step {step}/{len(loader)} "
                  f"loss={loss.item():.4f} acc={correct/total:.3f}")

    elapsed = time.time() - t0
    if rank == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f} "
              f"acc={correct/total:.4f} time={elapsed:.1f}s")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, rank):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    # Aggregate metrics across all ranks
    stats = torch.tensor([correct, total, loss_sum], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    correct, total, loss_sum = stats.tolist()

    acc = correct / total
    if rank == 0:
        print(f"  Val accuracy: {acc:.4f}  Val loss: {loss_sum/len(loader):.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="/data/imagenet")
    parser.add_argument("--epochs",     type=int,   default=90)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--momentum",   type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    # ── Model ──
    model = models.resnet50(weights=None).to(device)
    # Sync BatchNorm across GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Optimizer & Scheduler ──
    # Scale LR linearly with world size (linear scaling rule)
    lr = args.lr * world_size
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    # ── Data ──
    train_loader, val_loader, train_sampler = build_dataloaders(args, rank, world_size)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # ensures different shuffling per epoch
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, rank)
        val_acc = validate(model, val_loader, criterion, device, rank)
        scheduler.step()

        if rank == 0 and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.module.state_dict(), "checkpoints/best_resnet50.pt")
            print(f"  Saved best model (acc={val_acc:.4f})")

    if rank == 0:
        print(f"Training complete. Best val accuracy: {best_acc:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
