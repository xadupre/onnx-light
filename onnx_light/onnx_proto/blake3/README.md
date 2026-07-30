# Vendored BLAKE3

This directory vendors the official BLAKE3 C implementation, used by
`TensorProto::ContentHash(true)` to hash tensor payloads.

- Upstream: https://github.com/BLAKE3-team/BLAKE3 (`c/` directory)
- Version: 1.8.5
- License: Apache-2.0 (see `LICENSE`); BLAKE3 is also available under CC0-1.0.

## Files

Verbatim upstream sources:

- `blake3.h`, `blake3_impl.h`
- `blake3.c`, `blake3_dispatch.c`, `blake3_portable.c`

The library is compiled **portable only** (no SIMD / assembly): the
`BLAKE3_NO_SSE2`, `BLAKE3_NO_SSE41`, `BLAKE3_NO_AVX2`, `BLAKE3_NO_AVX512` and
`BLAKE3_USE_NEON=0` macros are defined for every BLAKE3 translation unit so the
build stays free of architecture-specific object files while remaining bit-for-bit
compatible with the canonical BLAKE3 digest.

onnx-light additions (not part of upstream):

- `blake3_join.cc` implements `blake3_compress_subtree_wide_join_tbb`, the hook
  upstream calls when `BLAKE3_USE_TBB` is defined. Instead of oneTBB it uses
  `std::async`, so `blake3_hasher_update_tbb` hashes the two halves of a large
  subtree in parallel. The number of concurrently spawned threads is bounded by
  the hardware concurrency, and the result is identical to the single-threaded
  digest regardless of the thread count.
- `blake3_hash.h` / `blake3_hash.cc` expose the small `utils::Blake3Hasher`
  wrapper used by the rest of onnx-light.
