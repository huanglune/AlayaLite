# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify third-party license materials are shipped in the wheel."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FILES = ("LICENSE", "LICENSES/*.txt", "REUSE.toml", "THIRD_PARTY_NOTICES.md")


def test_python_distributions_include_license_materials():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    expected = f"wheel.license-files = {list(REQUIRED_FILES)!r}".replace("'", '"')
    assert expected in pyproject
    expected = f"sdist.include = {list(REQUIRED_FILES)!r}".replace("'", '"')
    assert expected in pyproject


def test_conan_package_includes_license_materials():
    recipe = (ROOT / "conanfile.py").read_text(encoding="utf-8")

    for filename in ("LICENSE", "*.txt", "REUSE.toml", "THIRD_PARTY_NOTICES.md"):
        assert f'"{filename}"' in recipe
    assert 'os.path.join(self.package_folder, "licenses")' in recipe
