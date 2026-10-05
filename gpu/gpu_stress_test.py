#!/usr/bin/env python3
"""GPU 压力测试（torch + DDP，死循环模式）。

三种模式：
  compute —— 纯大矩阵乘法打满算力
  mixed   —— 前向+反向+优化器，模拟深度学习训练
  memory  —— 尽量占满显存

用法：
  python gpu_stress_test.py --mode mixed --gpus 8
Ctrl+C 停止。
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import time
import argparse

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def stress_test_compute(rank, world_size, duration=None, batch_size=512):
    """
    纯计算压力测试 - 大矩阵乘法（死循环）
    Args:
        rank: GPU编号
        world_size: GPU总数
        duration: 运行时长(秒) - 忽略此参数，死循环运行
        batch_size: 批次大小
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    print(f"GPU {rank}: 开始计算压力测试（死循环模式，按Ctrl+C停止）")
    
    # 创建大矩阵
    size = 16384  # 16K x 16K 矩阵
    A = torch.randn(size, size, device=f'cuda:{rank}', dtype=torch.float32)
    B = torch.randn(size, size, device=f'cuda:{rank}', dtype=torch.float32)
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:  # 死循环
            # 矩阵乘法
            C = torch.matmul(A, B)
            # 额外的操作保持GPU忙碌
            C = torch.matmul(C, A)
            C = torch.nn.functional.relu(C)
            
            iteration += 1
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                print(f"GPU {rank}: 迭代 {iteration}, 已运行 {elapsed:.1f}秒")
    except KeyboardInterrupt:
        print(f"GPU {rank}: 收到停止信号")
    finally:
        cleanup()
        print(f"GPU {rank}: 完成，总迭代 {iteration} 次")

def stress_test_mixed(rank, world_size, duration=None):
    """
    混合压力测试 - 模拟深度学习训练（死循环）
    包含前向传播、反向传播、梯度计算
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    print(f"GPU {rank}: 开始混合压力测试（死循环模式，按Ctrl+C停止）")
    
    # 创建一个大型模型
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(8192, 8192) for _ in range(20)
            ])
            self.activation = nn.ReLU()
            
        def forward(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x
    
    model = LargeModel().to(f'cuda:{rank}')
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    batch_size = 1024
    start_time = time.time()
    iteration = 0
    
    try:
        while True:  # 死循环
            # 生成随机数据
            x = torch.randn(batch_size, 8192, device=f'cuda:{rank}')
            target = torch.randn(batch_size, 8192, device=f'cuda:{rank}')
            
            # 前向传播
            output = model(x)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iteration += 1
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                print(f"GPU {rank}: 迭代 {iteration}, Loss: {loss.item():.4f}, 已运行 {elapsed:.1f}秒")
    except KeyboardInterrupt:
        print(f"GPU {rank}: 收到停止信号")
    finally:
        cleanup()
        print(f"GPU {rank}: 完成，总迭代 {iteration} 次")

def stress_test_memory(rank, world_size, duration=None):
    """
    显存压力测试 - 尽可能占用显存（死循环）
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    print(f"GPU {rank}: 开始显存压力测试（死循环模式，按Ctrl+C停止）")
    
    # 获取可用显存
    total_memory = torch.cuda.get_device_properties(rank).total_memory
    print(f"GPU {rank}: 总显存 {total_memory / 1024**3:.2f} GB")
    
    tensors = []
    # 分配大量张量占用显存
    chunk_size = 1024 * 1024 * 256  # 每块1GB (256M float32)
    
    try:
        # 占用约90%显存
        target_memory = int(total_memory * 0.9)
        allocated = 0
        
        while allocated < target_memory:
            tensor = torch.randn(chunk_size, device=f'cuda:{rank}', dtype=torch.float32)
            tensors.append(tensor)
            allocated += chunk_size * 4  # float32 = 4 bytes
            print(f"GPU {rank}: 已分配 {allocated / 1024**3:.2f} GB")
    except RuntimeError as e:
        print(f"GPU {rank}: 显存分配达到极限")
    
    # 持续进行计算保持GPU忙碌（死循环）
    start_time = time.time()
    iteration = 0
    
    try:
        while True:  # 死循环
            for i in range(min(len(tensors), 10)):
                tensors[i] = torch.nn.functional.relu(tensors[i] * 1.0001)
            
            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"GPU {rank}: 迭代 {iteration}, 已运行 {elapsed:.1f}秒")
    except KeyboardInterrupt:
        print(f"GPU {rank}: 收到停止信号")
    finally:
        cleanup()
        print(f"GPU {rank}: 完成")

def main():
    parser = argparse.ArgumentParser(description='GPU压力测试工具（死循环版本）')
    parser.add_argument('--mode', type=str, default='mixed', 
                       choices=['compute', 'mixed', 'memory'],
                       help='测试模式: compute(纯计算), mixed(混合训练), memory(显存)')
    parser.add_argument('--gpus', type=int, default=8,
                       help='使用的GPU数量, 默认8')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA设备")
        return
    
    num_gpus = min(args.gpus, torch.cuda.device_count())
    print(f"检测到 {torch.cuda.device_count()} 个GPU, 将使用 {num_gpus} 个GPU")
    print(f"运行模式: {args.mode}")
    print(f"运行方式: 死循环（按Ctrl+C停止）")
    print("=" * 60)
    
    # 选择测试函数
    if args.mode == 'compute':
        test_func = stress_test_compute
    elif args.mode == 'mixed':
        test_func = stress_test_mixed
    else:
        test_func = stress_test_memory
    
    # 启动多进程
    try:
        mp.spawn(
            test_func,
            args=(num_gpus,),
            nprocs=num_gpus,
            join=True
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭所有GPU进程...")
    
    print("=" * 60)
    print("所有GPU测试完成!")

if __name__ == '__main__':
    main()
