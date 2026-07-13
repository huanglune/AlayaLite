#!/usr/bin/env python3
"""Compare Gate 6 artifact inventories with the documented ISA policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FLAT_FAMILIES = ("disk_flat", "disk_flat_segment")
DISK_VAMANA_FAMILIES = ("disk_vamana", "disk_vamana_segment")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifacts(inventory: dict[str, object]) -> dict[str, object]:
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("inventory has no artifact family map")
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-repeat", type=Path, required=True)
    parser.add_argument("--avx2", type=Path, required=True)
    parser.add_argument("--avx2-repeat", type=Path, required=True)
    args = parser.parse_args()

    baseline = _artifacts(_load(args.baseline))
    native_inventory = _load(args.native)
    native_repeat = _load(args.native_repeat)
    avx2_inventory = _load(args.avx2)
    avx2_repeat = _load(args.avx2_repeat)
    native = _artifacts(native_inventory)
    avx2 = _artifacts(avx2_inventory)
    failed = False

    for lane, first, second in (
        ("native", native_inventory, native_repeat),
        ("avx2", avx2_inventory, avx2_repeat),
    ):
        stable = first == second
        print(f"self_consistency lane={lane} result={'PASS' if stable else 'FAIL'}")
        failed |= not stable

    for lane, actual in (("native", native), ("avx2", avx2)):
        for family in sorted(baseline):
            matches = actual.get(family) == baseline[family]
            if family in FLAT_FAMILIES:
                policy = "strict"
            elif family in DISK_VAMANA_FAMILIES or lane == "native":
                policy = "report"
            else:
                # The fixed AVX2 lane is the checked 14-family golden lane.
                # Host-native builds can legitimately select wider ISA in
                # unrelated families; those remain visible but do not change
                # Gate 6's Flat/Vamana acceptance rule.
                policy = "strict"
            print(
                f"baseline lane={lane} family={family} "
                f"result={'MATCH' if matches else 'DRIFT'} policy={policy} "
                f"sha={_digest(actual.get(family))}"
            )
            if not matches and policy == "strict":
                failed = True

    for family in FLAT_FAMILIES:
        matches = native.get(family) == avx2.get(family)
        print(
            f"cross_lane family={family} result={'MATCH' if matches else 'DRIFT'} "
            f"policy=strict native_sha={_digest(native.get(family))} "
            f"avx2_sha={_digest(avx2.get(family))}"
        )
        failed |= not matches

    for family in DISK_VAMANA_FAMILIES:
        matches = native.get(family) == avx2.get(family)
        print(
            f"cross_lane family={family} result={'MATCH' if matches else 'DRIFT'} "
            f"policy=report native_sha={_digest(native.get(family))} "
            f"avx2_sha={_digest(avx2.get(family))}"
        )

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
