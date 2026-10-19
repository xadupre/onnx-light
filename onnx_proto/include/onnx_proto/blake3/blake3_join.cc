// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_threads/fork_join.h"

extern "C" {
#include "blake3_impl.h"
}

extern "C" void blake3_compress_subtree_wide_join_tbb(
    const uint32_t key[8], uint8_t flags, bool use_tbb,
    const uint8_t *l_input, size_t l_input_len, uint64_t l_chunk_counter, uint8_t *l_cvs,
    size_t *l_n, const uint8_t *r_input, size_t r_input_len, uint64_t r_chunk_counter,
    uint8_t *r_cvs, size_t *r_n) noexcept {
  struct Subtree {
    const uint8_t *input;
    size_t length;
    const uint32_t *key;
    uint64_t counter;
    uint8_t flags;
    uint8_t *cvs;
    bool parallel;
    size_t *result;
  };
  Subtree left{l_input, l_input_len, key, l_chunk_counter, flags, l_cvs, use_tbb, l_n};
  Subtree right{r_input, r_input_len, key, r_chunk_counter, flags, r_cvs, use_tbb, r_n};
  const auto hash = [](void *context) noexcept {
    const auto &subtree = *static_cast<Subtree *>(context);
    *subtree.result = blake3_compress_subtree_wide(
        subtree.input, subtree.length, subtree.key, subtree.counter, subtree.flags,
        subtree.cvs, subtree.parallel);
  };
  onnx_threads::ForkJoin(&left, hash, &right, hash,
                         use_tbb && l_input_len + r_input_len >= 256 * 1024);
}
