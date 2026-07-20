#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
# Builds the hosted gcc-13 C++ coverage targets, runs every labeled test, and captures LCOV output.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_JOBS="${BUILD_JOBS:-4}"
CTEST_JOBS="${CTEST_JOBS:-4}"
CTEST_LABELS="${CTEST_LABELS:-.}"
CTEST_EXCLUDE_REGEX="${CTEST_EXCLUDE_REGEX:-}"
# Coverage stays hosted to preserve the established gcc/gcov-13 reporting baseline.
GCOV_TOOL="${GCOV_TOOL:-/usr/bin/gcov-13}"
CMAKE_LAUNCHER_ARGS=()
if [[ -n "${CMAKE_CXX_COMPILER_LAUNCHER:-}" ]]; then
  CMAKE_LAUNCHER_ARGS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
fi

# The configure step resolves C++ dependencies through the Conan dependency provider, which needs
# the `conan` CLI on PATH. Self-provision on CI runners (uv is always present there).
if ! command -v conan >/dev/null 2>&1; then
  uv tool install conan
  uv_tool_dir="$(uv tool dir)"
  export PATH="${uv_tool_dir}/conan/bin:$HOME/.local/bin:$PATH"
fi

# rebuild the project
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake .. \
  -DBUILD_TESTING=ON \
  -DBUILD_PYTHON=OFF \
  -DENABLE_COVERAGE=ON \
  -DALAYA_NATIVE_ARCH=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13 \
  "${CMAKE_LAUNCHER_ARGS[@]}"
# Build exactly the test executables the labelled coverage run will invoke. Deriving the
# target set from CTest (instead of hard-coding it) keeps the coverage build in lockstep
# with the tests as they are renamed, added, or removed -- a stale hard-coded list silently
# breaks the build ("No rule to make target ...") or the run ("Unable to find executable").
#
# `ctest --show-only` filters by label correctly pre-build, but its JSON `command` field is
# null until the targets are built; CTest instead resolves each $<TARGET_FILE:> into the
# generated CTestTestfile.cmake at configure time, so map the label-selected test names to
# their executables through those files.
mapfile -t COVERAGE_TARGETS < <(
  ctest --test-dir "${BUILD_DIR}" --show-only=json-v1 -L "${CTEST_LABELS}" \
    | BUILD_DIR="${BUILD_DIR}" python3 -c '
import json, os, re, sys

build = os.path.abspath(os.environ["BUILD_DIR"])
selected = {test["name"] for test in json.load(sys.stdin).get("tests", [])}
add_test = re.compile(r"add_test\(\s*\[=\[(?P<name>[^\]]+)\]=\]\s+\"(?P<exe>[^\"]+)\"")
targets = []
seen = set()
for root, _dirs, files in os.walk(build):
    if "CTestTestfile.cmake" not in files:
        continue
    with open(os.path.join(root, "CTestTestfile.cmake"), encoding="utf-8") as handle:
        testfile = handle.read()
    for match in add_test.finditer(testfile):
        if match.group("name") not in selected:
            continue
        executable = os.path.abspath(match.group("exe"))
        # Skip tests driven by a system tool (e.g. cmake -P header-boundary scripts);
        # keep only executables this build produces, whose basename is the CMake target.
        if not executable.startswith(build + os.sep):
            continue
        target = os.path.basename(executable)
        if target not in seen:
            seen.add(target)
            targets.append(target)
for target in targets:
    print(target)
'
)

if [[ "${#COVERAGE_TARGETS[@]}" -eq 0 ]]; then
  echo "error: no coverage targets resolved from CTest labels '${CTEST_LABELS}'" >&2
  exit 1
fi
echo "Building ${#COVERAGE_TARGETS[@]} coverage targets: ${COVERAGE_TARGETS[*]}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" --target "${COVERAGE_TARGETS[@]}"

# run the tests in parallel
ctest_cmd=(
  ctest
  --test-dir "${BUILD_DIR}"
  --progress
  --output-on-failure
  -L "${CTEST_LABELS}"
  -j"${CTEST_JOBS}"
)

if [[ -n "${CTEST_EXCLUDE_REGEX}" ]]; then
  ctest_cmd+=(-E "${CTEST_EXCLUDE_REGEX}")
fi

"${ctest_cmd[@]}"
lcov --capture \
     --quiet \
     --ignore-errors mismatch,mismatch,gcov,gcov \
     --directory "${BUILD_DIR}" \
     --base-directory "${ROOT_DIR}" \
     --include "${ROOT_DIR}/include/*" \
     --output-file "${ROOT_DIR}/coverage_c++.info" \
     --gcov-tool "${GCOV_TOOL}"

lcov --summary "${ROOT_DIR}/coverage_c++.info"
