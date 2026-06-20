#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
#
# Bump the project version across all source-of-truth files, update uv.lock,
# commit, and create a git tag.
#
# Usage:
#   ./scripts/bump_version.sh 1.0.2          # bump + commit + tag
#   ./scripts/bump_version.sh 1.0.2 --dry    # preview changes, don't write
#
# Files updated:
#   pyproject.toml                       version = "X.Y.Z"
#   python/src/alayalite/__init__.py     __version__ = "X.Y.Z"
#   conanfile.py                         version = "X.Y.Z"

set -euo pipefail

VERSION="${1:-}"
DRY="${2:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version> [--dry]"
  echo "Example: $0 1.0.2"
  exit 1
fi

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([ab][0-9]+)?$ ]]; then
  echo "Error: version '$VERSION' does not match X.Y.Z or X.Y.Za1/b1 pattern"
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"

PYPROJECT="$ROOT/pyproject.toml"
INIT_PY="$ROOT/python/src/alayalite/__init__.py"
CONANFILE="$ROOT/conanfile.py"

for f in "$PYPROJECT" "$INIT_PY" "$CONANFILE"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: $f not found"
    exit 1
  fi
done

OLD_VERSION=$(grep -oP '(?<=^version = ")[^"]+' "$PYPROJECT" | head -1)
if [[ -z "$OLD_VERSION" ]]; then
  echo "Error: cannot read current version from $PYPROJECT"
  exit 1
fi

if [[ "$OLD_VERSION" == "$VERSION" ]]; then
  echo "Already at version $VERSION, nothing to do."
  exit 0
fi

echo "Bumping: $OLD_VERSION → $VERSION"
echo ""
echo "Files:"
echo "  $PYPROJECT"
echo "  $INIT_PY"
echo "  $CONANFILE"

if [[ "$DRY" == "--dry" ]]; then
  echo ""
  echo "(dry run — no files changed)"
  exit 0
fi

sed -i "s/^version = \"$OLD_VERSION\"/version = \"$VERSION\"/" "$PYPROJECT"
sed -i "s/^__version__ = \"$OLD_VERSION\"/__version__ = \"$VERSION\"/" "$INIT_PY"
sed -i "s/version = \"$OLD_VERSION\"/version = \"$VERSION\"/" "$CONANFILE"

for f in "$PYPROJECT" "$INIT_PY" "$CONANFILE"; do
  if ! grep -q "$VERSION" "$f"; then
    echo "Error: version not updated in $f — check manually"
    exit 1
  fi
done

echo "Syncing uv.lock..."
(cd "$ROOT" && uv lock --quiet 2>/dev/null || true)

echo ""
echo "Version updated to $VERSION in all files."
echo ""
echo "Next steps:"
echo "  1. Update CHANGELOG.md with the new [${VERSION}] section"
echo "  2. git add -A && git commit -m 'chore(release): bump version to ${VERSION}'"
echo "  3. git tag v${VERSION}"
echo "  4. git push upstream main && git push upstream v${VERSION}"
