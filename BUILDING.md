<!--
SPDX-FileCopyrightText: 2025 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Building AlayaLite

## Prerequisites

- A C++20 compiler: GCC ≥ 11, Clang ≥ 14, AppleClang (Xcode 15+), or MSVC 2022
- CMake 3.24–3.31 and a build tool (Ninja recommended; Make works)
- [uv](https://docs.astral.sh/uv/) — manages the Python side (venv, pytest)
- [Conan 2](https://conan.io) on PATH: `uv tool install conan` (once per machine)
- Linux only: `libaio-dev` for the LASER disk index (`sudo apt-get install libaio-dev`)

C++ dependencies (RocksDB, spdlog, Eigen, GTest, …) resolve through the official **Conan dependency provider**
(`cmake/vendor/conan_provider.cmake`, wired up in `cmake/AlayaConan.cmake`): every CMake configure runs
`conan install` with a profile derived from the actual toolchain state and drops the packages under
`<build dir>/conan/`. There is no separate install step, and PEP 517 builds (`uv sync`, `uv build`, cibuildwheel)
bring their own `conan` from `[build-system].requires`. Opt out with `-DALAYA_AUTO_CONAN=OFF` if you manage
dependencies yourself.

## Quick start

```bash
uv sync                        # create .venv (Python deps + editable extension build)
make build                     # configure + build the release preset (build/Release)
make test                      # C++ ctest suite + Python pytest suite
```

Prefer raw CMake? The same flavors are exposed as presets:

```bash
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset debug           # plain debug build           -> build/Debug
cmake --preset asan            # ASan + UBSan at -O1         -> build/San
cmake --preset coverage        # gcov instrumentation        -> build/Coverage
```

A plain `cmake -B build/Release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON` configure (what CI uses) works too;
presets are a convenience, not a requirement.

## Options

| Option                      | Default        | Meaning                                                       |
| --------------------------- | -------------- | ------------------------------------------------------------- |
| `BUILD_PYTHON`              | `ON`           | Build the `_alayalitepy` pybind11 module                      |
| `BUILD_TESTING`             | `OFF` (`ON` in presets) | Build the C++ test suite                             |
| `ALAYA_ENABLE_LASER`        | platform-gated | LASER disk index (Linux x86_64 / macOS / Windows x64)         |
| `ENABLE_COVERAGE`           | `OFF`          | Per-target gcov/OpenCppCoverage instrumentation               |
| `ALAYA_NATIVE_ARCH`         | `OFF`          | `-march=native` (refuses to combine with distributable wheels) |
| `ALAYA_AUTO_CONAN`          | `ON`           | Run `conan install` automatically when the toolchain is absent |
| `ALAYA_USE_CCACHE`          | `ON`           | Use ccache when present                                       |
| `ALAYA_USE_FAST_LINKER`     | `ON`           | Use mold/lld when present                                     |

## How Python is resolved

All of CMake uses one interpreter (FindPython / `Python_*` variables), chosen in this order: explicit
`-DPython_EXECUTABLE=…` → activated `$VIRTUAL_ENV` → the project-local `.venv` (from `uv sync`) → whatever is on
`PATH`. The uv-managed `.venv` ships the C API headers, so `uv sync` is the zero-thought path; a bare system
interpreter without `python3-dev` is rejected at configure time with instructions rather than deep inside pybind11.

## Troubleshooting

- **"The `conan` executable was not found on PATH"** — `uv tool install conan` (or `pipx install conan`), then
  reconfigure.
- **"Python interpreter … has no C API headers"** — run `uv sync`, or pass `-DPython_EXECUTABLE`, or install
  `python3-dev`, or configure with `-DBUILD_PYTHON=OFF`.
- **Conan install failed during configure** — the conan output is echoed above CMake's error. Iterate directly with
  `conan install . --build=missing -s build_type=Release -s compiler.cppstd=20`. Source-built packages need a
  compiler plus make/ninja on `PATH`.
- **After `conanfile.py` changes** — nothing special: the provider re-runs `conan install` on the next configure
  (`make build-release` reconfigures automatically).
- **Wheel builds** — driven by scikit-build-core + cibuildwheel (see `pyproject.toml`); locally, `make wheel`.
