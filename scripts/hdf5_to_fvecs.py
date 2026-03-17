# /// script
# dependencies = [
#   "h5py"
# ]
# ///

"""Convert ann-benchmarks HDF5 dataset to fvecs/ivecs format.

Usage:
    uv run scripts/hdf5_to_fvecs.py \\
        --input http://ann-benchmarks.com/deep-image-96-angular.hdf5 \\
        --output data/deep10M --name deep10M

    or extract from local file:

    uv run scripts/hdf5_to_fvecs.py \\
        --input /path/to/deep-image-96-angular.hdf5 \\
        --output data/deep10M --name deep10M

This will produce:
    data/deep10M/deep10M_base.fvecs
    data/deep10M/deep10M_query.fvecs
    data/deep10M/deep10M_groundtruth.ivecs
"""

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(filepath: Path, data: np.ndarray) -> None:
    """Write float32 vectors in fvecs format: [dim(i32)][vec(f32 x dim)] per row."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, dim = data.shape
    dim_bytes = struct.pack("<i", dim)
    with open(filepath, "wb") as f:
        for i in range(n):
            f.write(dim_bytes)
            f.write(data[i].tobytes())
    print(f"  wrote {n} vectors (dim={dim}) -> {filepath}")


def write_ivecs(filepath: Path, data: np.ndarray) -> None:
    """Write int32 vectors in ivecs format: [dim(i32)][vec(i32 x dim)] per row."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    n, dim = data.shape
    dim_bytes = struct.pack("<i", dim)
    with open(filepath, "wb") as f:
        for i in range(n):
            f.write(dim_bytes)
            f.write(data[i].tobytes())
    print(f"  wrote {n} vectors (dim={dim}) -> {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ann-benchmarks HDF5 to fvecs/ivecs")
    parser.add_argument("--input", required=True, help="URL or local path to .hdf5 file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--name", default="deep10M", help="Dataset name prefix (default: deep10M)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.input
    if source.startswith(("http://", "https://")):
        local_path = out_dir / Path(source).name
        if not local_path.exists():
            print(f"Downloading {source} -> {local_path} ...")
            wget = shutil.which("wget")
            curl = shutil.which("curl")
            if wget:
                subprocess.run([wget, "-O", str(local_path), source], check=True)
            elif curl:
                subprocess.run([curl, "-L", "-o", str(local_path), source], check=True)
            else:
                print("Error: wget or curl not found", file=sys.stderr)
                sys.exit(1)
            print("  download complete.")
        else:
            print(f"Using cached {local_path}")
        hdf5_path = local_path
    else:
        hdf5_path = Path(source)
        if not hdf5_path.exists():
            print(f"Error: {hdf5_path} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Reading {hdf5_path} ...")
    with h5py.File(hdf5_path, "r") as f:
        print(f"  datasets: {list(f.keys())}")

        train = f["train"][:]  # base vectors
        test = f["test"][:]  # query vectors
        neighbors = f["neighbors"][:]  # ground truth ids

        print(f"  train:     {train.shape} {train.dtype}")
        print(f"  test:      {test.shape} {test.dtype}")
        print(f"  neighbors: {neighbors.shape} {neighbors.dtype}")

    prefix = args.name
    write_fvecs(out_dir / f"{prefix}_base.fvecs", train)
    write_fvecs(out_dir / f"{prefix}_query.fvecs", test)
    write_ivecs(out_dir / f"{prefix}_groundtruth.ivecs", neighbors)

    print("Done.")


if __name__ == "__main__":
    main()
