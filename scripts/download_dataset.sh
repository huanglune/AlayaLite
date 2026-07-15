#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
#
# Download benchmark datasets.
#
# Usage:
#   ./scripts/download_dataset.sh <dataset> [output_dir]
#
# Examples:
#   ./scripts/download_dataset.sh siftsmall
#   ./scripts/download_dataset.sh deep1m /path/to/data
#   ./scripts/download_dataset.sh --list

set -euo pipefail

declare -A DATASETS=(
  [siftsmall]="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
  [deep1m]="http://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz"
)

list_datasets() {
  echo "Available datasets:"
  for name in "${!DATASETS[@]}"; do
    echo "  $name  ${DATASETS[$name]}"
  done
}

download_one() {
  local name="$1"
  local output_dir="$2"
  local url="${DATASETS[$name]:-}"

  if [[ -z "$url" ]]; then
    echo "ERROR: unknown dataset '$name'" >&2
    list_datasets >&2
    return 1
  fi

  local dest="$output_dir/$name"
  local marker="$dest/.ready"

  if [[ -f "$marker" ]]; then
    echo "$name: already downloaded at $dest"
    return 0
  fi

  echo "$name: downloading from $url ..."
  mkdir -p "$dest"

  local archive
  archive="$(mktemp)"
  trap 'rm -f "$archive"' RETURN

  wget -q --show-progress "$url" -O "$archive"
  tar -zxf "$archive" --strip-components=1 -C "$dest"
  touch "$marker"
  echo "$name: ready at $dest"
}

main() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset|--list|--all> [output_dir]" >&2
    return 1
  fi

  local cmd="$1"
  local output_dir="${2:-./data}"

  case "$cmd" in
    --list)
      list_datasets
      ;;
    --all)
      for name in "${!DATASETS[@]}"; do
        download_one "$name" "$output_dir"
      done
      ;;
    *)
      download_one "$cmd" "$output_dir"
      ;;
  esac
}

main "$@"
