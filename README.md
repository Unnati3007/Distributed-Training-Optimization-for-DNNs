# Distributed-Training-Optimization-for-DNNs
The system-oriented project was related to the improvement of Deep Neural Network training efficiency for large-scale DNNs. It aims to solve the "Communication Wall" problem in distributed systems by using gradient compression and mixed precision approaches.
Goal: Improve scaling efficiency in multi-GPU settings to decrease Time-to-Train for large neural networks.
Approach: Used PyTorch DDP and NCCL to detect and solve any issues related to gradient synchronization between multiple nodes.Implemented Mixed Precision Training (FP16) and other optimization practices to fully leverage GPU power and minimize network overhead.
Results: 85% scaling efficiency on 4 GPUs, with 70% reduction in training time and 40% reduction in network synchronization load.
