# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Expose the repository namespace to focused example tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
