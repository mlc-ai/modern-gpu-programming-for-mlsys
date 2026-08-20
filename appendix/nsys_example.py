"""Reusable CUDA workload for the benchmarking and profiling appendix."""

import argparse
from statistics import median

import torch


def make_workload(size: int):
    a = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((size, size), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(c)

    def run():
        with torch.cuda.nvtx.range("BF16 GEMM"):
            torch.mm(a, b, out=c)
        with torch.cuda.nvtx.range("ReLU"):
            torch.clamp_min(c, 0, out=output)

    def validate():
        torch.set_float32_matmul_precision("highest")
        expected = torch.clamp_min(torch.mm(a.float(), b.float()), 0).to(output.dtype)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)

    return run, validate


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


def collect_proton(run, *, warmup_calls: int, profile_calls: int, output: str):
    import triton.profiler as proton

    for _ in range(warmup_calls):
        run()
    torch.cuda.synchronize()

    session = proton.start(output, context="shadow", data="tree")
    if session is None:
        raise RuntimeError("Proton session could not be created")
    try:
        with proton.scope("target_operation"):
            for _ in range(profile_calls):
                run()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile-once", action="store_true")
    mode.add_argument("--event-samples", type=int, default=0)
    mode.add_argument("--proton-calls", type=int, default=0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup-calls", type=int, default=500)
    parser.add_argument("--proton-output", default="operator")
    args = parser.parse_args()

    run, validate = make_workload(args.size)
    run()
    torch.cuda.synchronize()
    validate()

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
    elif args.proton_calls:
        collect_proton(
            run,
            warmup_calls=args.warmup_calls,
            profile_calls=args.proton_calls,
            output=args.proton_output,
        )
    else:
        run()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
