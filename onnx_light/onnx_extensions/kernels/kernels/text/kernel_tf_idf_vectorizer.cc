// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/text/include_text_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_helpers.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// N-gram trie node. ``id_`` is non-zero when this node represents the
// last token of an n-gram that should produce an output count at
// position ``ngram_indexes[id_ - 1]``. ``leaves_`` is the (token →
// child node) map for the next token of the n-gram; an empty map
// means the trie has no longer n-grams starting with this prefix.
template <typename Key> struct NgramNode {
  int64_t id = 0;
  std::unordered_map<Key, NgramNode> leaves;
};

// Decomposes ``x.shape`` into (B, C) following the TfIdfVectorizer
// convention: rank-1 input ``[C]`` reports ``B = 0`` so that the
// output collapses to ``[C']``; rank-2 input ``[B, C]`` keeps the
// batch dimension. Throws std::invalid_argument for any other rank.
std::pair<int64_t, int64_t> BatchAndChannels(const onnx_kernels::Shape &shape) {
  if (shape.size() == 1) {
    return {0, shape[0]};
  }
  if (shape.size() == 2) {
    return {shape[0], shape[1]};
  }
  EXT_THROW_INVALID("kernel::TfIdfVectorizer: input shape must have rank 1 or 2.");
}

// Populates the n-gram trie ``root`` with ``n_ngrams`` consecutive
// n-grams of length ``ngram_size`` taken from ``pool`` starting at
// position ``start_idx``. Returns the next available ``ngram_id``.
//
// This mirrors the upstream onnx ``populate_grams`` reference helper:
// the i-th n-gram (1-based) gets identifier ``next_id + i - 1``; the
// caller is responsible for skipping ranges whose ``ngram_size`` is
// outside the requested ``[min_gram_length, max_gram_length]`` band.
template <typename Key>
int64_t PopulateGrams(const std::vector<Key> &pool, size_t start_idx, size_t n_ngrams,
                      size_t ngram_size, int64_t next_id, NgramNode<Key> &root) {
  for (size_t g = 0; g < n_ngrams; ++g) {
    NgramNode<Key> *node = &root;
    for (size_t k = 0; k < ngram_size; ++k) {
      EXT_ENFORCE_INVALID(
          start_idx < pool.size(),
          "kernel::TfIdfVectorizer: pool overflow while populating the n-gram trie.");
      auto &child = node->leaves[pool[start_idx]];
      if (k + 1 == ngram_size) {
        child.id = next_id;
        ++next_id;
      }
      node = &child;
      ++start_idx;
    }
  }
  return next_id;
}

// Builds the n-gram trie used at evaluation time from the operator
// attributes (``pool``, ``ngram_counts``, ``min_gram_length``,
// ``max_gram_length``).
template <typename Key>
NgramNode<Key> BuildTrie(const std::vector<Key> &pool, const std::vector<int64_t> &ngram_counts,
                         int64_t min_gram_length, int64_t max_gram_length) {
  NgramNode<Key> root;
  if (pool.empty()) {
    return root;
  }
  const size_t total_items = pool.size();
  int64_t next_id = 1;
  size_t ngram_size = 1;
  for (size_t i = 0; i < ngram_counts.size(); ++i) {
    const int64_t raw_start = ngram_counts[i];
    const int64_t raw_end =
        (i + 1 < ngram_counts.size()) ? ngram_counts[i + 1] : static_cast<int64_t>(total_items);
    EXT_ENFORCE_INVALID(raw_start >= 0 && raw_end >= raw_start &&
                            static_cast<size_t>(raw_end) <= total_items,
                        "kernel::TfIdfVectorizer: ngram_counts is out of range with respect to "
                        "the pool.");
    const size_t start_idx = static_cast<size_t>(raw_start);
    const size_t end_idx = static_cast<size_t>(raw_end);
    const size_t items = end_idx - start_idx;
    if (items > 0) {
      EXT_ENFORCE_INVALID(
          ngram_size != 0 && items % ngram_size == 0,
          "kernel::TfIdfVectorizer: ngram_counts is not a multiple of the current n-gram size.");
      const size_t ngrams = items / ngram_size;
      const int64_t ngram_size_i = static_cast<int64_t>(ngram_size);
      if (ngram_size_i >= min_gram_length && ngram_size_i <= max_gram_length) {
        next_id = PopulateGrams(pool, start_idx, ngrams, ngram_size, next_id, root);
      } else {
        next_id += static_cast<int64_t>(ngrams);
      }
    }
    ++ngram_size;
  }
  return root;
}

// Reusable scratch buffer holding one input row's worth of ``int64_t``
// tokens. Storage is acquired from the runtime ``RawBufferAllocator``
// when one is attached (so no per-row heap allocation happens outside
// the runtime context) and falls back to an inline ``std::vector`` only
// when no allocator is available. Mirrors the ``ReadZeroPoints`` helper
// in kernel_matmul_integer.cc.
class Int64RowBuffer {
public:
  Int64RowBuffer(RawBufferAllocator *allocator, int64_t row_size) : allocator_(allocator) {
    const size_t n_bytes = static_cast<size_t>(row_size) * sizeof(int64_t);
    if (allocator_ != nullptr) {
      buffer_ = allocator_->Allocate(n_bytes);
      EXT_ENFORCE_INVALID(buffer_ != nullptr,
                          "kernel::TfIdfVectorizer: row allocator returned null.");
      // RawBufferAllocator::Allocate returns at least n_bytes.
      EXT_ENFORCE_INVALID(buffer_->size() >= n_bytes,
                          "kernel::TfIdfVectorizer: row allocator returned too small a buffer.");
      EXT_ENFORCE_INVALID(reinterpret_cast<std::uintptr_t>(buffer_->data()) % alignof(int64_t) == 0,
                          "kernel::TfIdfVectorizer: allocator returned misaligned row buffer.");
    } else {
      fallback_.resize(static_cast<size_t>(row_size));
    }
  }

  ~Int64RowBuffer() {
    if (buffer_ != nullptr) {
      allocator_->Free(buffer_);
    }
  }

  Int64RowBuffer(const Int64RowBuffer &) = delete;
  Int64RowBuffer &operator=(const Int64RowBuffer &) = delete;

  int64_t *data() noexcept {
    return buffer_ != nullptr ? reinterpret_cast<int64_t *>(buffer_->data()) : fallback_.data();
  }

private:
  RawBufferAllocator *allocator_ = nullptr;
  RawBuffer *buffer_ = nullptr;
  std::vector<int64_t> fallback_;
};

// Reads a single row's worth of integer tokens out of ``x`` into the
// caller-provided ``row`` buffer, allowing the allocation to be reused
// across rows. Values are widened to ``int64_t`` to match the trie keys.
void ReadIntRow(const Tensor &x, int64_t row_num, int64_t row_size, int64_t *row) {
  const size_t offset = static_cast<size_t>(row_num) * static_cast<size_t>(row_size);
  if (x.data_type == static_cast<int32_t>(DataType::INT32)) {
    const int32_t *data = x.As<int32_t>();
    for (int64_t i = 0; i < row_size; ++i) {
      row[static_cast<size_t>(i)] = static_cast<int64_t>(data[offset + static_cast<size_t>(i)]);
    }
  } else if (x.data_type == static_cast<int32_t>(DataType::INT64)) {
    const int64_t *data = x.As<int64_t>();
    for (int64_t i = 0; i < row_size; ++i) {
      row[static_cast<size_t>(i)] = data[offset + static_cast<size_t>(i)];
    }
  } else {
    EXT_THROW_INVALID(
        "kernel::TfIdfVectorizer: integer pool requires an INT32 or INT64 input tensor.");
  }
}

// Computes the per-row n-gram frequencies for one input row and
// accumulates them into ``frequencies`` (a flat ``[num_rows,
// output_size]`` buffer). Mirrors the upstream onnx
// ``compute_impl`` reference helper.
template <typename Key>
void AccumulateRow(const Key *row, int64_t row_size, int64_t row_num, int64_t output_size,
                   int64_t min_gram_length, int64_t max_gram_length, int64_t max_skip_count,
                   const std::vector<int64_t> &ngram_indexes, const NgramNode<Key> &root,
                   std::vector<int64_t> &frequencies) {
  const int64_t max_skip_distance = max_skip_count + 1;
  int64_t start_ngram_size = min_gram_length;

  for (int64_t skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
    int64_t ngram_start = 0;
    while (ngram_start < row_size) {
      // Bail out when no n-gram of the smallest requested size could
      // possibly fit starting at ``ngram_start``.
      const int64_t at_least_this = ngram_start + skip_distance * (start_ngram_size - 1);
      if (at_least_this >= row_size) {
        break;
      }

      int64_t ngram_item = ngram_start;
      const NgramNode<Key> *node = &root;
      int64_t ngram_size = 1;
      while (!node->leaves.empty() && ngram_size <= max_gram_length && ngram_item < row_size) {
        const auto it = node->leaves.find(row[static_cast<size_t>(ngram_item)]);
        if (it == node->leaves.end()) {
          break;
        }
        const NgramNode<Key> &child = it->second;
        if (ngram_size >= start_ngram_size && child.id != 0) {
          const int64_t hit = child.id - 1;
          const int64_t output_idx =
              row_num * output_size + ngram_indexes[static_cast<size_t>(hit)];
          frequencies[static_cast<size_t>(output_idx)] += 1;
        }
        node = &child;
        ++ngram_size;
        ngram_item += skip_distance;
      }

      ++ngram_start;
    }

    // Unigrams are not affected by skip_distance and are emitted on
    // the first iteration only.
    if (start_ngram_size == 1) {
      ++start_ngram_size;
      if (start_ngram_size > max_gram_length) {
        break;
      }
    }
  }
}

// Combines the per-row frequency table with the weighting criteria
// (TF / IDF / TFIDF) and writes the result into ``out``.
void ApplyMode(int64_t num_rows, int64_t output_size, TfIdfVectorizer::Mode mode,
               const std::vector<int64_t> &frequencies, const std::vector<float> &weights,
               float *out) {
  const int64_t total = num_rows * output_size;
  using Mode = TfIdfVectorizer::Mode;
  switch (mode) {
  case Mode::kTF:
    for (int64_t i = 0; i < total; ++i) {
      out[i] = static_cast<float>(frequencies[static_cast<size_t>(i)]);
    }
    break;
  case Mode::kIDF: {
    const bool has_weights = !weights.empty();
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t c = 0; c < output_size; ++c) {
        const int64_t p = r * output_size + c;
        const bool hit = frequencies[static_cast<size_t>(p)] > 0;
        if (has_weights) {
          out[p] = hit ? weights[static_cast<size_t>(c)] : 0.0f;
        } else {
          out[p] = hit ? 1.0f : 0.0f;
        }
      }
    }
    break;
  }
  case Mode::kTFIDF: {
    const bool has_weights = !weights.empty();
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t c = 0; c < output_size; ++c) {
        const int64_t p = r * output_size + c;
        const float f = static_cast<float>(frequencies[static_cast<size_t>(p)]);
        if (has_weights) {
          out[p] = weights[static_cast<size_t>(c)] * f;
        } else {
          out[p] = f;
        }
      }
    }
    break;
  }
  }
}

} // namespace

TfIdfVectorizer::Mode TfIdfVectorizer::ParseMode(const std::string &value) {
  if (value == "TF") {
    return Mode::kTF;
  }
  if (value == "IDF") {
    return Mode::kIDF;
  }
  if (value == "TFIDF") {
    return Mode::kTFIDF;
  }
  EXT_THROW_INVALID("kernel::TfIdfVectorizer: invalid mode '", value,
                    "'. Valid values are \"TF\", \"IDF\", \"TFIDF\".");
}

onnx_kernels::Shape TfIdfVectorizer::ComputeOutputShape(const onnx_kernels::Shape &input_shape,
                                                        int64_t output_size) {
  if (input_shape.size() == 1) {
    return {output_size};
  }
  if (input_shape.size() == 2) {
    return {input_shape[0], output_size};
  }
  EXT_THROW_INVALID("kernel::TfIdfVectorizer: input shape must have rank 1 or 2.");
}

Tensor TfIdfVectorizer::operator()(const Tensor &x, Mode mode, int64_t min_gram_length,
                                   int64_t max_gram_length, int64_t max_skip_count,
                                   const std::vector<int64_t> &ngram_counts,
                                   const ParamInts &ngram_indexes, const ParamInts &pool_int64s,
                                   const std::vector<std::string> &pool_strings,
                                   const std::vector<float> &weights, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(!ngram_indexes.empty(),
                      "kernel::TfIdfVectorizer: ngram_indexes must be non-empty.");
  EXT_ENFORCE_INVALID(min_gram_length >= 1,
                      "kernel::TfIdfVectorizer: min_gram_length must be >= 1.");
  EXT_ENFORCE_INVALID(max_gram_length >= min_gram_length,
                      "kernel::TfIdfVectorizer: max_gram_length must be >= min_gram_length.");
  EXT_ENFORCE_INVALID(max_skip_count >= 0, "kernel::TfIdfVectorizer: max_skip_count must be >= 0.");
  EXT_ENFORCE_INVALID(pool_int64s.empty() != pool_strings.empty(),
                      "kernel::TfIdfVectorizer: exactly one of pool_int64s and pool_strings "
                      "must be set.");

  const int64_t output_size = *std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend()) + 1;
  EXT_ENFORCE_INVALID(output_size > 0,
                      "kernel::TfIdfVectorizer: ngram_indexes must contain non-negative entries.");

  const auto bc = BatchAndChannels(x.shape);
  const int64_t b_dim = bc.first;
  const int64_t c_dim = bc.second;
  const int64_t num_rows = b_dim == 0 ? 1 : b_dim;
  EXT_ENFORCE_INVALID(b_dim >= 0, "kernel::TfIdfVectorizer: batch dimension must be non-negative.");
  EXT_ENFORCE_INVALID(c_dim >= 0, "kernel::TfIdfVectorizer: input row size must be non-negative.");

  std::vector<int64_t> frequencies(static_cast<size_t>(num_rows * output_size), 0);

  if (c_dim > 0) {
    if (!pool_int64s.empty()) {
      NgramNode<int64_t> root =
          BuildTrie<int64_t>(pool_int64s, ngram_counts, min_gram_length, max_gram_length);
      if (!root.leaves.empty()) {
        Int64RowBuffer row(ctx_.allocator, c_dim);
        int64_t *row_data = row.data();
        for (int64_t r = 0; r < num_rows; ++r) {
          ReadIntRow(x, r, c_dim, row_data);
          AccumulateRow<int64_t>(row_data, c_dim, r, output_size, min_gram_length, max_gram_length,
                                 max_skip_count, ngram_indexes, root, frequencies);
        }
      }
    } else {
      NgramNode<std::string> root =
          BuildTrie<std::string>(pool_strings, ngram_counts, min_gram_length, max_gram_length);
      if (!root.leaves.empty()) {
        EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                            "kernel::TfIdfVectorizer: string pool requires a STRING input tensor.");
        for (int64_t r = 0; r < num_rows; ++r) {
          // ``string_data`` is contiguous row-major, so the row can be
          // read in place without copying it into a scratch buffer.
          const std::string *row_data =
              x.string_data.data() + static_cast<size_t>(r) * static_cast<size_t>(c_dim);
          AccumulateRow<std::string>(row_data, c_dim, r, output_size, min_gram_length,
                                     max_gram_length, max_skip_count, ngram_indexes, root,
                                     frequencies);
        }
      }
    }
  }

  onnx_kernels::Shape out_shape = ComputeOutputShape(x.shape, output_size);
  std::vector<float> out_values(static_cast<size_t>(num_rows * output_size), 0.0f);
  ApplyMode(num_rows, output_size, mode, frequencies, weights, out_values.data());
  return Tensor::FromFloat("", out_shape, out_values, ctx_.allocator);
}

void TfIdfVectorizer::operator()(const Tensor &x, Mode mode, int64_t min_gram_length,
                                 int64_t max_gram_length, int64_t max_skip_count,
                                 const std::vector<int64_t> &ngram_counts,
                                 const ParamInts &ngram_indexes, const ParamInts &pool_int64s,
                                 const std::vector<std::string> &pool_strings,
                                 const std::vector<float> &weights, Tensor &output) const {
  Tensor computed = (*this)(x, mode, min_gram_length, max_gram_length, max_skip_count, ngram_counts,
                            ngram_indexes, pool_int64s, pool_strings, weights);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::TfIdfVectorizer preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == computed.shape,
                      "kernel::TfIdfVectorizer preallocated output shape must match the "
                      "computed output shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == computed.size_bytes(),
                      "kernel::TfIdfVectorizer preallocated output buffer has unexpected size.");
  output.data = std::move(computed.data);
}

void TfIdfVectorizer::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const std::string mode_attr = GetRequiredAttributeString(node, "mode");
  const int64_t min_gram_length = GetAttributeIntOrDefault(node, "min_gram_length", 1);
  const int64_t max_gram_length = GetAttributeIntOrDefault(node, "max_gram_length", 1);
  const int64_t max_skip_count = GetAttributeIntOrDefault(node, "max_skip_count", 0);
  const std::vector<int64_t> ngram_counts = GetAttributeIntsOrDefault(node, "ngram_counts", {});
  const ParamInts ngram_indexes = GetAttributeIntsOrDefault(node, "ngram_indexes", {});
  const ParamInts pool_int64s = GetAttributeIntsOrDefault(node, "pool_int64s", {});
  const std::vector<std::string> pool_strings =
      GetAttributeStringsOrDefault(node, "pool_strings", {});
  const std::vector<float> weights = GetAttributeFloatsOrDefault(node, "weights", {});
  onnx_kernels::kernel::TfIdfVectorizer k(rt.kernel_ctx());
  SetOutput(node, 0,
            k(x, onnx_kernels::kernel::TfIdfVectorizer::ParseMode(mode_attr), min_gram_length,
              max_gram_length, max_skip_count, ngram_counts, ngram_indexes, pool_int64s,
              pool_strings, weights),
            rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
