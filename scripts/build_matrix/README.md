# Build matrix

`run_matrix.sh` is the repeatable compile boundary check for the header-only core and optional LASER/Python surfaces.
Each case gets a clean directory below `build/matrix` and performs a Release configure and build.

```sh
scripts/build_matrix/run_matrix.sh             # all local cases
scripts/build_matrix/run_matrix.sh core-only   # one case (used by CI)
```

The core/portable cases also run the small CPU-feature and SIMD tests. `portable-no-avx512` audits that no AVX-512
baseline flag is present; `portable-no-avx2` disables `ALAYA_X86_AVX2_BASELINE` and audits AVX2/FMA flags as well.
`core-only` verifies that CMake never discovers libaio and compiles representative core headers without LASER.
The Python rows are compile-only.

Cross-platform feasibility is covered without introducing a new toolchain: the existing `cibuildwheel.yaml` matrix
builds Linux aarch64, macOS arm64/x86_64, and Windows x64. LASER remains unsupported on Linux ARM and defaults off.
MSVC and ARM cannot be executed by this local script unless it is run natively with those toolchains; add native CI
rows invoking `core-only` when compile-only feedback faster than wheel builds is needed.
