"""Small multi-stage CUDA workload for the Nsight Systems appendix example."""

import argparse
from statistics import median

import torch


def make_workload(size: int):
    host_a = torch.randn((size, size), dtype=torch.bfloat16, pin_memory=True)
    a = torch.empty_like(host_a, device="cuda")
    b = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((size, size), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(c)

    def run():
        with torch.cuda.nvtx.range("H2D input"):
            a.copy_(host_a, non_blocking=True)
        with torch.cuda.nvtx.range("BF16 GEMM"):
            torch.mm(a, b, out=c)
        with torch.cuda.nvtx.range("ReLU"):
            torch.clamp_min(c, 0, out=output)

    return run, output


def run_once_for_profiler(run, *, warmup_calls: int):
    for _ in range(warmup_calls):
        run()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    with torch.cuda.nvtx.range("target operation"):
        run()
        torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def measure_event_us(run, *, warmup_calls: int, samples: int):
    for _ in range(warmup_calls):
        run()
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1e3)
    return values


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile-once", action="store_true")
    mode.add_argument("--event-samples", type=int, default=0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup-calls", type=int, default=5)
    args = parser.parse_args()

    run, output = make_workload(args.size)
    if args.profile_once:
        run_once_for_profiler(run, warmup_calls=args.warmup_calls)
    elif args.event_samples:
        values = measure_event_us(
            run,
            warmup_calls=args.warmup_calls,
            samples=args.event_samples,
        )
        print(
            f"median={median(values):.3f} us, "
            f"min={min(values):.3f} us, max={max(values):.3f} us"
        )
    else:
        run()
        torch.cuda.synchronize()

    if not torch.isfinite(output).all().item():
        raise RuntimeError("workload produced a non-finite output")


if __name__ == "__main__":
    main()
