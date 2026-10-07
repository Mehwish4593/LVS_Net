"""
benchmark.py -- inference cost of LVS-Net: latency, throughput and memory.

"""

import argparse
import time

import numpy as np
import torch

from lvs_net import LVSNet

SIZE = 512


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers() if b.dtype.is_floating_point)
    return trainable, buffers


def bench(device, batch, runs, warmup, threads=None):
    if device == "cpu" and threads:
        torch.set_num_threads(threads)

    model = LVSNet().to(device).eval()
    x = torch.randn(batch, 3, SIZE, SIZE, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    per_image = np.median(times) / batch
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return per_image, 1.0 / per_image, peak, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    model = LVSNet()
    trainable, buffers = count_params(model)
    total = trainable + buffers
    print("parameters : %s trainable + %s batch-norm statistics = %s"
          % (format(trainable, ","), format(buffers, ","), format(total, ",")))
    print("model size : %.2f MB (float32)" % (total * 4 / 2**20))

    devices = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("cuda", "both") and torch.cuda.is_available():
        devices.append("cuda")

    print("\ninput 1x3x%dx%d, batch %d, median of %d runs after %d warmup\n"
          % (SIZE, SIZE, args.batch, args.runs, args.warmup))
    print("%-34s %10s %9s %12s" % ("device", "ms/image", "FPS", "peak MB"))
    print("-" * 68)
    for d in devices:
        per_image, fps, peak, times = bench(d, args.batch, args.runs, args.warmup, args.threads)
        if d == "cuda":
            name = torch.cuda.get_device_name(0)[:32]
            mem = "%.1f" % peak
        else:
            name = "CPU (%d threads)" % torch.get_num_threads()
            mem = "-"
        print("%-34s %10.2f %9.2f %12s" % (name, 1000 * per_image, fps, mem))
        print("%-34s %10s min %.2f / max %.2f ms"
              % ("", "", 1000 * times.min() / args.batch, 1000 * times.max() / args.batch))


if __name__ == "__main__":
    main()
