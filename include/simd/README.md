# SIMD Module

`include/simd/` contains the low-level SIMD helpers used by AlayaLite distance kernels and transform routines.

## Files

| File | Purpose |
| --- | --- |
| `cpu_features.hpp` | Runtime CPU feature detection and SIMD level selection |
| `distance_l2.hpp` / `.ipp` | L2 squared distance kernels for FP32, SQ8, and SQ4 |
| `distance_ip.hpp` / `.ipp` | Negative inner-product kernels for FP32, SQ8, and SQ4 |
| `fht.hpp` / `.ipp` | Fast Walsh-Hadamard Transform helpers |

## Dispatch Model

The module chooses the best available implementation at runtime:

- `Generic` fallback on all platforms
- `SSE4.1`, `AVX2+FMA`, and `AVX-512` on supported x86 CPUs

You can inspect the selected level with:

```cpp
#include "simd/cpu_features.hpp"

auto name = alaya::simd::get_simd_level_name();
```

`get_simd_level()` returns one of:

- `alaya::simd::SimdLevel::kGeneric`
- `alaya::simd::SimdLevel::kSse4`
- `alaya::simd::SimdLevel::kAvx2`
- `alaya::simd::SimdLevel::kAvx512`

## Public APIs

### L2 distance

```cpp
#include "simd/distance_l2.hpp"

float dist = alaya::simd::l2_sqr(x, y, dim);
```

Available entrypoints:

- `l2_sqr(...)`
- `l2_sqr_sq8(...)`
- `l2_sqr_sq4(...)`
- `get_l2_sqr_func()`
- `get_l2_sqr_sq8_func()`
- `get_l2_sqr_sq4_func()`

### Inner product distance

```cpp
#include "simd/distance_ip.hpp"

float dist = alaya::simd::ip_sqr(x, y, dim);
```

Available entrypoints:

- `ip_sqr(...)`
- `ip_sqr_sq8(...)`
- `ip_sqr_sq4(...)`
- `get_ip_sqr_func()`
- `get_ip_sqr_sq8_func()`
- `get_ip_sqr_sq4_func()`

`ip_sqr` returns the negative inner product so smaller values mean more similar vectors.

### FHT helpers

```cpp
#include "simd/fht.hpp"

alaya::simd::fht_float(buffer, log_n);
```

The transform helpers cover `2^6` through `2^11` specialized paths, with generic fallback outside those optimized cases.

## Quantized Inputs

- `SQ8` stores one quantized value per byte.
- `SQ4` packs two values per byte.
- Quantized distance APIs require matching `min` and `max` arrays for dequantization.

## When to Use What

- Use the template wrappers such as `l2_sqr(...)` and `ip_sqr(...)` for normal call sites.
- Use `get_*_func()` if you need to cache a dispatched function pointer inside a hot loop.
- Use `get_simd_level_name()` when debugging performance or verifying feature detection on a target machine.
