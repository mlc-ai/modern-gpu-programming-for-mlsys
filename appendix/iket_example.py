"""Minimal TIRx workload with IKET ranges for two warp roles."""

from pathlib import Path

import numpy as np

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda import iket


N = 256
ELEMS_PER_LANE = 8


@T.function
def warp_role_example(inp: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=64))
    profiler = iket.IketProfiler()
    warp_id = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((N,), "float32")

    profiler.mark("kernel_start", warp_id)
    if warp_id == 0:
        profiler.range_push("producer_load")
        for i in T.serial(ELEMS_PER_LANE):
            index = i * 32 + lane
            shared[index] = inp[index]
        profiler.range_pop()

    profiler.range_push("wait_for_data")
    T.cuda.cta_sync()
    profiler.range_pop()

    if warp_id == 1:
        profiler.range_push("consumer_compute")
        for i in T.serial(ELEMS_PER_LANE):
            index = i * 32 + lane
            out[index] = shared[index] * T.float32(2) + T.float32(1)
        profiler.range_pop()


def profile_workload():
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    executable = tvm_tirx.tirx.compile(warp_role_example, target=target)
    module = executable.jit()

    input_numpy = np.arange(N, dtype=np.float32)
    inp = tvm_tirx.runtime.tensor(input_numpy, device=tvm_tirx.cuda())
    out = tvm_tirx.runtime.empty((N,), "float32", device=tvm_tirx.cuda())
    module.main(inp, out)
    tvm_tirx.cuda().sync()

    expected = input_numpy * 2 + 1
    np.testing.assert_array_equal(out.numpy(), expected)


def main():
    result = iket.run(
        profile_workload,
        output_dir=Path("reports/iket-warp-roles"),
        postprocess="all",
        clobber=True,
        timeout=600.0,
    )
    print(f"IKET output directory: {result.output_dir}")
    for path in (*result.json_traces, *result.perfetto_traces, *result.html_reports):
        print(f"IKET artifact: {path}")


if __name__ == "__main__":
    main()
