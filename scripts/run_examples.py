"""Run the book's complete examples directly from the English and Chinese sources.

Requires a Blackwell GPU and requirements-examples.txt. Each example runs in a
fresh interpreter so compiler state cannot leak between the GEMM steps.
"""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time


ROOT = Path(__file__).resolve().parents[1]


def python_blocks(path):
    if path.suffix == ".md":
        return re.findall(r"^```python\s*\n(.*?)^```", path.read_text(), re.M | re.S)
    lines = path.read_text().splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        end = index + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith(" ")):
            end += 1
        blocks.append(textwrap.dedent("\n".join(lines[index + 1:end])).strip())
    return blocks


def gemm_check(version, shape):
    return f'''
import torch
torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")
M, N, K = {shape!r}
target = tvm_tirx.target.Target("cuda")
with target:
    ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({{"main": hgemm_v{version}(M, N, K)}}),
                     target=target)
A = torch.empty(M, K, device="cuda", dtype=torch.float16)
B = torch.empty(N, K, device="cuda", dtype=torch.float16)
D = torch.empty(M, N, device="cuda", dtype=torch.float16)
for iteration in range(5):
    A.normal_()
    B.normal_()
    D.fill_(float("nan"))
    ex.mod(A, B, D)
    torch.cuda.synchronize()
    ref = (A.float() @ B.float().T).half()
    torch.testing.assert_close(D, ref, rtol=2e-2, atol=1e-2)
print("PASS GEMM v{version}", M, N, K)
'''


def kernel_cases(base, large, scheduler_stress=False):
    intro = python_blocks(base / "chapter_intro_tirx/index.md")
    yield "intro", "\n".join(intro[:3])
    for chapter in ("chapter_gemm_basics", "chapter_gemm_async", "chapter_gemm_advanced"):
        imports = ""
        for block in python_blocks(base / chapter / "index.md"):
            if "from tvm_tirx.script import" in block:
                imports = block
            match = re.search(r"^def hgemm_v(\d+)\(", block, re.M)
            if not match:
                continue
            version = int(match[1])
            source = block if "from tvm_tirx.script import" in block else imports + "\n" + block
            if version == 1:
                shapes = [(128, 128, 64)]
            elif version == 2:
                shapes = [(128, 128, 64), (128, 128, 320)]
            elif version == 6:
                # This example deliberately carries pipeline phase across tiles;
                # its documented contract requires K_TILES % (2 * PIPE_DEPTH) == 0.
                shapes = [(1024, 1024, 256), (1024, 1024, 512)]
                if large:
                    shapes.append((4096, 4096, 4096))
            else:
                shapes = [(1024, 1024, 64), (1024, 1024, 320)]
                if large:
                    shapes.append((4096, 4096, 4096))
            if scheduler_stress and version >= 6:
                yield f"gemm-v{version}-scheduler-stress", source + gemm_check(version, (5120, 5120, 256))
            for shape in shapes:
                suffix = "x".join(map(str, shape))
                yield f"gemm-v{version}-{suffix}", source + gemm_check(version, shape)
    attention = python_blocks(base / "chapter_flash_attention/index.md")[-1]
    yield "attention", attention


def reference_cases(base):
    yield "data-layout", "\n".join(python_blocks(base / "chapter_data_layout/index.md"))
    layouts = python_blocks(base / "chapter_tirx_layout_api/index.md")
    imports = next(block for block in layouts if "from tvm_tirx.tirx.layout import" in block)
    concrete = [block for block in layouts
                if block.startswith(("layout =", "frag =", "tile =", "scale ="))
                and "pool." not in block and "T." not in block]
    yield "layouts", "\n".join([imports, "sf_per_mma = 4", *concrete, 'print("PASS layouts")'])
    header = '''import tvm_tirx
import torch
from tvm_tirx.script import tirx as T
target = tvm_tirx.target.Target("cuda")
'''
    reference = base / "tirx_guide/language_reference/cuda"
    examples = [
        ("buffers.rst", "smem_demo", 128, "ex.mod(a, b)", "a * 2"),
        ("data_types.rst", "dtypes", 256, "ex.mod(a, b)", "a"),
        ("threads_sync.rst", "warp_reduce", 32, "ex.mod(b)", "torch.full_like(a, 496)"),
        ("threads_sync.rst", "k", 256, "ex.mod(a, b)", "a.clamp_min(0)"),
    ]
    for filename, name, size, launch, expected in examples:
        source = next(b for b in python_blocks(reference / filename) if f"def {name}(" in b)
        yield name, header + source + f'''
with target:
    ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({{"main": {name}}}), target=target)
a = torch.arange({size}, dtype=torch.float32, device="cuda") - 16
b = torch.empty_like(a)
{launch}
torch.cuda.synchronize()
torch.testing.assert_close(b, {expected})
print("PASS {name}")
'''
    arch = python_blocks(base / "tirx_guide/arch/lowering_pipeline.rst")
    yield "lowering", "\n".join([arch[0], arch[1], arch[2], '''import torch
a = torch.arange(1024, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
exe.mod(a, b)
torch.cuda.synchronize()
torch.testing.assert_close(b, a * 2)
print("PASS lowering")
'''])


    # These architecture pages were added to the Chinese edition on main.
    architecture = base / "tirx_guide/arch"
    for filename, function, invocation, expected in (
        ("ir_representation.rst", "ir_copy", "ex.mod(a, b)", "a"),
        ("tile_primitive_layout_lowering.rst", "layout_fill", "ex.mod(b)", "torch.full_like(a, 3)"),
    ):
        path = architecture / filename
        if not path.exists():
            continue
        source = "\n".join(python_blocks(path))
        yield function, source + f'''
import torch
a = torch.randn((128, 4), dtype=torch.float32, device="cuda")
b = torch.full_like(a, float("nan"))
ex = tvm_tirx.tirx.compile({function}, target="cuda")
{invocation}
torch.cuda.synchronize()
torch.testing.assert_close(b, {expected})
print("PASS {function}")
'''


def benchmark_cases(base):
    blocks = python_blocks(base / "appendix/benchmarking_gpu_kernels.md")
    yield "events", "\n".join(blocks[1:3])
    yield "host-timer", "\n".join([blocks[1], blocks[3]])
    yield "multistream", blocks[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("kernels", "reference", "benchmarks", "all"), default="all")
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--match", default="", help="Run only cases whose names contain this text")
    parser.add_argument("--large", action="store_true", help="Also check 4096-cubed GEMMs (steps 3-9)")
    parser.add_argument("--scheduler-stress", action="store_true",
                        help="Also check partial L2 groups and repeated persistent scheduling")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/examples")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("apache-tvm-tirx", "apache-tvm-ffi", "apache-tvm-ffi-orcjit", "cuda-bindings", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    (output / "environment.json").write_text(json.dumps({
        "python": sys.version, "executable": sys.executable, "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["TVM_BACKTRACE"] = "1"
    languages = ("en", "zh") if args.language == "both" else (args.language,)
    results = []
    for language in languages:
        base = ROOT / "zh" if language == "zh" else ROOT
        cases = []
        if args.suite in ("all", "kernels"):
            cases.extend(kernel_cases(base, args.large, args.scheduler_stress))
        if args.suite in ("all", "reference"):
            cases.extend(reference_cases(base))
        if args.suite in ("all", "benchmarks"):
            cases.extend(benchmark_cases(base))
        for short_name, source in cases:
            name = f"{language}-{short_name}"
            if args.match not in name:
                continue
            script = output / f"{name}.py"
            script.write_text(source)
            log_path = output / f"{name}.log"
            started = time.monotonic()
            try:
                with log_path.open("w") as log:
                    process = subprocess.run(
                        [sys.executable, "-c",
                         "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
                         "runpy.run_path(sys.argv[2], run_name='__main__')",
                         str(ROOT), str(script)],
                        cwd=output, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                    )
                status = "PASS" if process.returncode == 0 else "FAIL"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
            results.append({"name": name, "status": status,
                            "seconds": round(time.monotonic() - started, 2), "log": str(log_path)})
            print(f"{status}: {name}", flush=True)
            if status != "PASS":
                print("\n".join(log_path.read_text().splitlines()[-12:]), flush=True)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if not results:
        parser.error("No examples matched")
    passed = sum(result["status"] == "PASS" for result in results)
    print(f"{passed}/{len(results)} passed; logs: {output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
