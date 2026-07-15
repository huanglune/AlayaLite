#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
"""Enforce include-layer boundaries across the codebase.

Rules:
  core/       → may only include core/ and standard library
  space/      → may not include storage/rocksdb_storage, index/collection/, index/disk/
  index/graph/ → may not include index/collection/, index/disk/ (graph kernels are leaf)

Violations are printed to stderr and cause a non-zero exit.
"""

import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')

RULES: list[tuple[str, list[str], str]] = [
    # (source prefix, forbidden include patterns, reason)
    (
        "include/core/",
        [
            "index/",
            "space/",
            "storage/",
            "recovery/",
            "executor/",
            "utils/",
            "python/",
            "sdk/",
        ],
        "core/ must not depend on higher layers",
    ),
    (
        "include/space/",
        [
            "storage/rocksdb_storage",
            "storage/detail/rocksdb_storage",
            "index/collection/",
            "index/disk/",
        ],
        "space/ must not depend on storage backends or collection internals",
    ),
    (
        "include/index/graph/",
        [
            "index/collection/",
            "index/disk/",
        ],
        "graph kernels must not depend on collection or disk segment internals",
    ),
    (
        "include/platform/",
        [
            "core/",
            "index/",
            "space/",
            "storage/",
            "utils/",
        ],
        "platform/ is the lowest layer and must not depend on any project code",
    ),
]

EXCEPTIONS: set[str] = {
    # graph/detail/search_runtime may include space (by design)
}


def check_file(path: Path, root: Path) -> list[str]:
    violations = []
    rel = str(path.relative_to(root)).replace("\\", "/")

    if rel in EXCEPTIONS:
        return []

    applicable_rules = [(forbidden, reason) for prefix, forbidden, reason in RULES if rel.startswith(prefix)]
    if not applicable_rules:
        return []

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for lineno, line in enumerate(lines, 1):
        m = INCLUDE_RE.match(line)
        if not m:
            continue
        included = m.group(1)
        for forbidden_list, reason in applicable_rules:
            for pattern in forbidden_list:
                if included.startswith(pattern):
                    violations.append(f"{rel}:{lineno}: includes '{included}' — {reason}")

    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    include_dir = root / "include"

    if not include_dir.is_dir():
        print(f"error: {include_dir} not found", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for hpp in sorted(include_dir.rglob("*.hpp")):
        all_violations.extend(check_file(hpp, root))

    if all_violations:
        print(f"Layer boundary violations ({len(all_violations)}):", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
