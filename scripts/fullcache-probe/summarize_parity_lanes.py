# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Validate a parity-lanes CSV and print balanced iso-recall Markdown tables."""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--targets", default="0.90,0.95,0.99")
    return parser.parse_args()


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def load_points(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("measurement CSV is empty")

    multiplicity = Counter((row["arm"], int(row["lanes"]), int(row["ef"]), row["order"]) for row in rows)
    if len(set(multiplicity.values())) != 1:
        raise ValueError("arm/C/ef/order cells have unequal repeat multiplicity")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"], int(row["lanes"]), int(row["ef"])].append(row)

    points = {}
    for key, group in grouped.items():
        recalls = {float(row["recall"]) for row in group}
        checksums = {row["checksum"] for row in group}
        if len(recalls) != 1 or len(checksums) != 1:
            raise ValueError(f"recall/checksum changed for {key}")
        by_order = defaultdict(list)
        for row in group:
            by_order[row["order"]].append(float(row["qps"]))
        if len(by_order) != 2:
            raise ValueError(f"expected two paired orders for {key}")
        order_medians = [statistics.median(values) for values in by_order.values()]
        points[key] = (recalls.pop(), math.sqrt(math.prod(order_medians)))
    return rows, points, multiplicity


def interpolate(points, arm, lanes, target):
    curve = sorted(
        (recall, ef, qps)
        for (point_arm, point_lanes, ef), (recall, qps) in points.items()
        if point_arm == arm and point_lanes == lanes
    )
    for lower, upper in zip(curve, curve[1:]):
        r0, ef0, qps0 = lower
        r1, ef1, qps1 = upper
        if r0 <= target <= r1:
            fraction = (target - r0) / (r1 - r0)
            return (
                ef0 + fraction * (ef1 - ef0),
                qps0 + fraction * (qps1 - qps0),
            )
    raise ValueError(f"target {target} is outside {arm}/C={lanes} recall curve")


def main():
    args = parse_args()
    targets = [float(value) for value in args.targets.split(",")]
    rows, points, multiplicity = load_points(args.csv)
    arms = sorted({row["arm"] for row in rows})
    if "memqg" in arms:
        arms = ["memqg", *(arm for arm in arms if arm != "memqg")]
    lanes = sorted({int(row["lanes"]) for row in rows})
    if len(arms) != 2:
        raise ValueError("iso-recall summary requires exactly two arms")

    print(f"records={len(rows)} cells={len(multiplicity)} repeats_per_order={next(iter(multiplicity.values()))}")
    print("\n| Arm | ef | recall | " + " | ".join(f"C={lane} QPS" for lane in lanes) + " |")
    print("|---|---:|---:|" + "---:|" * len(lanes))
    for arm in arms:
        efs = sorted({ef for point_arm, _, ef in points if point_arm == arm})
        for ef in efs:
            recall = points[arm, lanes[0], ef][0]
            qps = [points[arm, lane, ef][1] for lane in lanes]
            print(f"| {arm} | {ef} | {recall:.8f} | " + " | ".join(f"{value:.3f}" for value in qps) + " |")

    ratios = []
    by_lane = defaultdict(list)
    print(f"\n| C | target | {arms[0]} ef* / QPS | {arms[1]} ef* / QPS | {arms[1]} / {arms[0]} |")
    print("|---:|---:|---:|---:|---:|")
    for lane in lanes:
        for target in targets:
            ef0, qps0 = interpolate(points, arms[0], lane, target)
            ef1, qps1 = interpolate(points, arms[1], lane, target)
            ratio = qps1 / qps0
            ratios.append(ratio)
            by_lane[lane].append(ratio)
            print(f"| {lane} | {target:.2f} | {ef0:.3f} / {qps0:.3f} | {ef1:.3f} / {qps1:.3f} | {ratio * 100:.3f}% |")
    print("\n| Aggregate | Geometric mean |")
    print("|---|---:|")
    for lane in lanes:
        print(f"| C={lane} | {geometric_mean(by_lane[lane]) * 100:.3f}% |")
    print(f"| All cells | {geometric_mean(ratios) * 100:.3f}% |")


if __name__ == "__main__":
    main()
