#!/usr/bin/env python3
"""Record canonical module size and verify that retired native symbols stay absent."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SYMBOL_PATTERNS = (
    "alaya::BasePyIndex",
    "alaya::PyIndex<",
    "alaya::IndexFactory",
    "alaya::disk::PyDiskCollection",
    "alaya::laser::bindings",
    "alaya::vamana::bindings",
)


def _text_bytes(binary: Path) -> int:
    output = subprocess.check_output(["size", "-A", str(binary)], text=True)
    return sum(
        int(parts[1])
        for line in output.splitlines()
        if (parts := line.split()) and parts[0].startswith(".text")
    )


def _demangled_symbols(binary: Path) -> list[str]:
    output = subprocess.check_output(["nm", "--demangle", str(binary)], text=True)
    return [line.split(maxsplit=2)[-1] for line in output.splitlines() if line.strip()]


def _identity_rows() -> int:
    config = yaml.safe_load((ROOT / "tools/codegen/dispatch.yaml").read_text(encoding="utf-8"))
    return len(config["combinations"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/Release")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/size_map/baseline.json")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    obj = build / "python/CMakeFiles/_alayalitepy.dir/src/pybind.cpp.o"
    module = next((build / "python").glob("_alayalitepy*.so"))
    symbols = _demangled_symbols(module)
    leaked = {
        pattern: [name for name in symbols if pattern in name]
        for pattern in LEGACY_SYMBOL_PATTERNS
    }
    leaked = {pattern: names for pattern, names in leaked.items() if names}
    if leaked:
        raise RuntimeError(f"retired native symbols leaked into the extension: {sorted(leaked)}")

    payload = {
        "schema_version": 2,
        "measurement": "GNU size + demangled symbol absence audit",
        "build_type": "Release",
        "canonical_identity_rows": _identity_rows(),
        "legacy_dispatch_rows_linked": 0,
        "legacy_symbol_matches": 0,
        "pybind_object": {"bytes": obj.stat().st_size, "text_bytes": _text_bytes(obj)},
        "extension_module": {
            "bytes": module.stat().st_size,
            "text_bytes": _text_bytes(module),
        },
        "wheel": None
        if args.wheel is None
        else {"bytes": args.wheel.stat().st_size, "name": args.wheel.name},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
