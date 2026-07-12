#!/usr/bin/env python3
"""Record wheel/module size and attributable .text symbols for 33 dispatch rows."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[tuple[str, str, str, str]]:
    config = yaml.safe_load((ROOT / "tools/codegen/dispatch.yaml").read_text(encoding="utf-8"))
    return [(row["data"], row["id"], row["quant"], row["index"]) for row in config["combinations"]]


def _text_bytes(obj: Path) -> int:
    output = subprocess.check_output(["size", "-A", str(obj)], text=True)
    return sum(
        int(parts[1]) for line in output.splitlines() if (parts := line.split()) and parts[0].startswith(".text")
    )


def _symbols(obj: Path) -> list[tuple[int, str]]:
    output = subprocess.check_output(["nm", "-S", "--size-sort", "--demangle", str(obj)], text=True)
    found = []
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[2].lower() in {"t", "w"}:
            found.append((int(parts[1], 16), parts[3]))
    return found


def _tokens(row: tuple[str, str, str, str]) -> tuple[str, str, str]:
    data, ident, quant, index = row
    data_token = {"float": "float", "int8_t": "signed char", "uint8_t": "unsigned char"}[data]
    id_token = {"uint32_t": "unsigned int", "uint64_t": "unsigned long"}[ident]
    builder_token = {"HNSW": "HnswSegment<", "NSG": "NSGBuilder<", "FUSION": "FusionGraphBuilder<"}[index]
    space_token = {"NONE": "RawSpace<", "SQ8": "SQ8Space<", "SQ4": "SQ4Space<", "RABITQ": "RaBitQSpace<"}[quant]
    return builder_token, f"{space_token}{data_token}, float, {id_token}", id_token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/Release")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/size_map/baseline.json")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    obj = build / "python/CMakeFiles/_alayalitepy.dir/src/index_factory.cpp.o"
    module = next((build / "python").glob("_alayalitepy*.so"))
    symbols = _symbols(obj)
    combinations = []
    for row in _rows():
        builder, space, _ = _tokens(row)
        matched = [(size, name) for size, name in symbols if builder in name and space in name]
        combinations.append(
            {
                "data": row[0],
                "id": row[1],
                "quant": row[2],
                "index": row[3],
                # An attribution map, not a sum of disjoint sections: template symbols may be
                # shared by scalar-on/off and nested Fusion builders.
                "attributed_text_symbol_bytes": sum(size for size, _ in matched),
                "matched_text_symbols": len(matched),
            }
        )
    payload = {
        "schema_version": 1,
        "measurement": "GNU nm symbol attribution; entries may overlap",
        "build_type": "Release",
        "dispatch_combinations": combinations,
        "index_factory_object": {"bytes": obj.stat().st_size, "text_bytes": _text_bytes(obj)},
        "extension_module": {"bytes": module.stat().st_size},
        "wheel": None if args.wheel is None else {"bytes": args.wheel.stat().st_size, "name": args.wheel.name},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
