"""Minimal TIRx workload with IKET ranges for two warp roles."""

from pathlib import Path

import numpy as np

import tvm
from tvm.script import tirx as T
from tvm.tirx.cuda import iket


N = 256
ELEMS_PER_LANE = 8


@T.prim_func
def warp_role_example(inp: T.Buffer((N,), "float32"), out: T.Buffer((N,), "float32")):
    T.device_entry()
    profiler = iket.IketProfiler()
    warp_id = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((N,), "float32", scope="shared")

    profiler.mark("kernel_start", warp_id)
    if warp_id == 0:
        profiler.range_push("producer_load")
        for i in T.serial(ELEMS_PER_LANE, unroll=False):
            index = i * 32 + lane
            shared[index] = inp[index]
        profiler.range_pop()

    profiler.range_push("wait_for_data")
    T.cuda.cta_sync()
    profiler.range_pop()

    if warp_id == 1:
        profiler.range_push("consumer_compute")
        for i in T.serial(ELEMS_PER_LANE, unroll=False):
            index = i * 32 + lane
            out[index] = shared[index] * T.float32(2) + T.float32(1)
        profiler.range_pop()


def profile_workload():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    executable = tvm.compile(warp_role_example, target=target, tir_pipeline="tirx")
    module = executable.jit()

    input_numpy = np.arange(N, dtype=np.float32)
    inp = tvm.runtime.tensor(input_numpy, device=tvm.cuda())
    out = tvm.runtime.empty((N,), "float32", device=tvm.cuda())
    module.main(inp, out)
    tvm.cuda().sync()

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
