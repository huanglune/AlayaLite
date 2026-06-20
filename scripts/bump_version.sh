#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
#
# Bump the project version in pyproject.toml (the single source of truth)
# and sync uv.lock. Other files (conanfile.py, __init__.py) read the version
# dynamically at build/runtime — no manual sync needed.
#
# Usage:
#   ./scripts/bump_version.sh 1.0.2          # bump + sync lockfile
#   ./scripts/bump_version.sh 1.0.2 --dry    # preview changes, don't write

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

# Portable sed -i: GNU sed uses -i without arg, BSD sed requires -i ''
_sed_i() {
  if sed --version >/dev/null 2>&1; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

OLD_VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$PYPROJECT" | head -1)
if [[ -z "$OLD_VERSION" ]]; then
  echo "Error: cannot read current version from $PYPROJECT"
  exit 1
fi

if [[ "$OLD_VERSION" == "$VERSION" ]]; then
  echo "Already at version $VERSION, nothing to do."
  exit 0
fi

echo "Bumping: $OLD_VERSION → $VERSION"
echo "  pyproject.toml (single source of truth)"

if [[ "$DRY" == "--dry" ]]; then
  echo ""
  echo "(dry run — no files changed)"
  exit 0
fi

_sed_i "s/^version = \"$OLD_VERSION\"/version = \"$VERSION\"/" "$PYPROJECT"

if ! grep -q "version = \"$VERSION\"" "$PYPROJECT"; then
  echo "Error: version not updated in pyproject.toml"
  exit 1
fi

echo "Syncing uv.lock..."
if ! (cd "$ROOT" && uv lock --quiet); then
  echo "Error: uv lock failed — fix pyproject.toml and retry"
  exit 1
fi

echo ""
echo "Done. Files changed:"
echo "  pyproject.toml"
echo "  uv.lock"
echo ""
echo "Next steps:"
echo "  1. Update CHANGELOG.md with the new [${VERSION}] section"
echo "  2. git add pyproject.toml uv.lock CHANGELOG.md"
echo "  3. git commit -m 'chore(release): bump version to ${VERSION}'"
echo "  4. git tag v${VERSION}"
echo "  5. git push upstream main && git push upstream v${VERSION}"
echo ""
echo "Or use: GitHub Actions → Prepare Release → version=${VERSION}"
