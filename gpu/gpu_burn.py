#!/usr/bin/env python3
"""GPU 打满 / 压测脚本 (torch)。

每张卡起一个进程，循环跑大矩阵乘法把算力打满；可选按比例占满显存。
用法示例：
  python gpu_burn.py                      # 所有卡，100% 利用率，一直跑
  python gpu_burn.py --gpus 0,1,2         # 只压 0/1/2 三张卡
  python gpu_burn.py --util 80            # 占空比控制到约 80% 利用率
  python gpu_burn.py --mem-fraction 0.9   # 同时把 90% 显存占住
  python gpu_burn.py --minutes 30         # 跑 30 分钟自动退出
Ctrl+C 可随时优雅退出。
"""
import argparse
import time
import torch
import torch.multiprocessing as mp

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def worker(gpu_id, args, stop_time):
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    dtype = DTYPES[args.dtype]

    # 可选：按比例占满显存
    holders = []
    if args.mem_fraction > 0:
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        # 给矩阵乘法留 2GB 余量，避免 OOM
        target = int(total * args.mem_fraction) - 2 * 1024**3
        step = 512 * 1024**2  # 每块 512MB
        got = 0
        while got < target:
            try:
                holders.append(torch.empty(step // 2, dtype=torch.float16, device=dev))
                got += step
            except RuntimeError:
                break
        print(f"[gpu{gpu_id}] 显存占用约 {got / 1024**3:.1f} GiB", flush=True)

    n = args.matrix_size
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = torch.empty(n, n, device=dev, dtype=dtype)

    duty = max(0.01, min(1.0, args.util / 100.0))
    period = 0.1  # 占空比控制窗口 100ms
    iters = 0
    print(f"[gpu{gpu_id}] 开始压测: {n}x{n} {args.dtype}, 目标利用率 {args.util}%", flush=True)

    try:
        while True:
            if stop_time and time.time() > stop_time:
                break
            t0 = time.time()
            while time.time() - t0 < period * duty:
                for _ in range(args.batch):
                    torch.matmul(a, b, out=c)
                torch.cuda.synchronize(gpu_id)
                iters += 1
            if duty < 1.0:
                time.sleep(period * (1.0 - duty))
    except KeyboardInterrupt:
        pass
    print(f"[gpu{gpu_id}] 结束, 共 {iters} 轮", flush=True)


def main():
    p = argparse.ArgumentParser(description="用 torch 把 GPU 利用率打满")
    p.add_argument("--gpus", default="all", help="卡号, 逗号分隔, 如 0,1,3; 默认全部")
    p.add_argument("--util", type=float, default=100.0, help="目标利用率 %% (1-100), 默认 100")
    p.add_argument("--mem-fraction", type=float, default=0.0, help="额外占用的显存比例 0-1, 默认 0")
    p.add_argument("--matrix-size", type=int, default=8192, help="方阵边长, 默认 8192")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16", help="计算精度, 默认 bf16")
    p.add_argument("--batch", type=int, default=8, help="每次同步前的矩阵乘次数, 默认 8")
    p.add_argument("--minutes", type=float, default=0.0, help="运行时长(分钟), 0 表示一直跑")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    if args.gpus == "all":
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    stop_time = time.time() + args.minutes * 60 if args.minutes > 0 else 0
    dur = f"{args.minutes} 分钟" if args.minutes > 0 else "持续 (Ctrl+C 停止)"
    print(f"压测 GPU: {gpus} | 利用率 {args.util}% | 时长 {dur}", flush=True)

    mp.set_start_method("spawn", force=True)
    procs = []
    for g in gpus:
        proc = mp.Process(target=worker, args=(g, args, stop_time))
        proc.start()
        procs.append(proc)

    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("\n收到中断, 正在停止所有 worker...", flush=True)
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    main()
