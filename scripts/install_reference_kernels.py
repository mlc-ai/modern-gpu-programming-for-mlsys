"""Install the book's pinned FA4 reference with its TIRx API compatibility patch."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REVISION = "9d48e43336d4eca43ad852fee9dc224ca9ac9527"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "validation/reference-kernels")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = Path(tempfile.mkdtemp(prefix="tirx-kernels-", dir=args.work_dir)).resolve()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                    "https://github.com/mlc-ai/TIRx-kernels.git"], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", REVISION], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    patch = ROOT / "patches/tirx-kernels-api.patch"
    subprocess.run(["git", "-C", str(source), "apply", "--unidiff-zero", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(source), "apply", "--unidiff-zero", str(patch)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", str(source)], check=True)
    print(f"Installed FA4 reference from {REVISION}; patched source: {source}")


if __name__ == "__main__":
    main()
