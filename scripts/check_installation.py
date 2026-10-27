"""Check native LLVM execution and, optionally, ARM/Wasm object generation."""

import argparse
from pathlib import Path

import numpy as np
import tvm_tirx
from tvm_tirx.script import tirx as T


@T.function
def add_one(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
    for i in T.serial(32):
        B[i] = A[i] + T.float32(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", action="store_true", help="Requires apache-tvm-tirx[all-targets]")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/installation"))
    args = parser.parse_args()
    executable = tvm_tirx.tirx.compile(add_one, target="llvm")
    values = np.arange(32, dtype=np.float32)
    a = tvm_tirx.runtime.tensor(values)
    b = tvm_tirx.runtime.empty((32,), "float32")
    executable(a, b)
    np.testing.assert_array_equal(b.numpy(), values + 1)
    print("PASS: native LLVM execution")
    if args.cross:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, triple in (("aarch64", "aarch64-linux-gnu"), ("wasm", "wasm32-unknown-unknown")):
            target = tvm_tirx.target.Target({"kind": "llvm", "mtriple": triple})
            module = tvm_tirx.tirx.compile(add_one, target=target)
            path = args.output_dir / f"{name}.o"
            module.mod.write_to_file(str(path), "o")
            data = path.read_bytes()
            if name == "aarch64":
                assert data[:4] == b"\x7fELF" and int.from_bytes(data[18:20], "little") == 183
            else:
                assert data[:8] == b"\x00asm\x01\x00\x00\x00"
            print(f"PASS: {name} object generation ({path})")


if __name__ == "__main__":
    main()
