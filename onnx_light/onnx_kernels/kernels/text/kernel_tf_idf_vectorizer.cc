// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/text/include_text_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

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
std::pair<int64_t, int64_t> BatchAndChannels(const std::vector<int64_t> &shape) {
  if (shape.size() == 1) {
    return {0, shape[0]};
  }
  if (shape.size() == 2) {
    return {shape[0], shape[1]};
  }
  throw std::invalid_argument("kernel::TfIdfVectorizer: input shape must have rank 1 or 2.");
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
      if (start_idx >= pool.size()) {
        // Defensive — ``ngram_counts`` should already protect us, but
        // make this a hard error so malformed inputs surface clearly.
        throw std::invalid_argument(
            "kernel::TfIdfVectorizer: pool overflow while populating the n-gram trie.");
      }
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
    if (raw_start < 0 || raw_end < raw_start || static_cast<size_t>(raw_end) > total_items) {
      throw std::invalid_argument(
          "kernel::TfIdfVectorizer: ngram_counts is out of range with respect to the pool.");
    }
    const size_t start_idx = static_cast<size_t>(raw_start);
    const size_t end_idx = static_cast<size_t>(raw_end);
    const size_t items = end_idx - start_idx;
    if (items > 0) {
      if (ngram_size == 0 || items % ngram_size != 0) {
        throw std::invalid_argument(
            "kernel::TfIdfVectorizer: ngram_counts is not a multiple of the current n-gram size.");
      }
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

// Reads a single row's worth of tokens out of ``x``. For string
// tensors the values are copied into a vector of ``std::string``; for
// integer tensors they are converted to ``int64_t``.
template <typename Key>
std::vector<Key> ReadRow(const Tensor &x, int64_t row_num, int64_t row_size);

template <>
std::vector<int64_t> ReadRow<int64_t>(const Tensor &x, int64_t row_num, int64_t row_size) {
  std::vector<int64_t> row(static_cast<size_t>(row_size));
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
    throw std::invalid_argument(
        "kernel::TfIdfVectorizer: integer pool requires an INT32 or INT64 input tensor.");
  }
  return row;
}

template <>
std::vector<std::string> ReadRow<std::string>(const Tensor &x, int64_t row_num, int64_t row_size) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                      "kernel::TfIdfVectorizer: string pool requires a STRING input tensor.");
  std::vector<std::string> row(static_cast<size_t>(row_size));
  const size_t offset = static_cast<size_t>(row_num) * static_cast<size_t>(row_size);
  for (int64_t i = 0; i < row_size; ++i) {
    row[static_cast<size_t>(i)] = x.string_data[offset + static_cast<size_t>(i)];
  }
  return row;
}

// Computes the per-row n-gram frequencies for one input row and
// accumulates them into ``frequencies`` (a flat ``[num_rows,
// output_size]`` buffer). Mirrors the upstream onnx
// ``compute_impl`` reference helper.
template <typename Key>
void AccumulateRow(const std::vector<Key> &row, int64_t row_num, int64_t output_size,
                   int64_t min_gram_length, int64_t max_gram_length, int64_t max_skip_count,
                   const std::vector<int64_t> &ngram_indexes, const NgramNode<Key> &root,
                   std::vector<int64_t> &frequencies) {
  const int64_t row_size = static_cast<int64_t>(row.size());
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
  throw std::invalid_argument("kernel::TfIdfVectorizer: invalid mode '" + value +
                              "'. Valid values are \"TF\", \"IDF\", \"TFIDF\".");
}

std::vector<int64_t> TfIdfVectorizer::ComputeOutputShape(const std::vector<int64_t> &input_shape,
                                                         int64_t output_size) {
  if (input_shape.size() == 1) {
    return {output_size};
  }
  if (input_shape.size() == 2) {
    return {input_shape[0], output_size};
  }
  throw std::invalid_argument("kernel::TfIdfVectorizer: input shape must have rank 1 or 2.");
}

Tensor TfIdfVectorizer::operator()(const Tensor &x, Mode mode, int64_t min_gram_length,
                                   int64_t max_gram_length, int64_t max_skip_count,
                                   const std::vector<int64_t> &ngram_counts,
                                   const std::vector<int64_t> &ngram_indexes,
                                   const std::vector<int64_t> &pool_int64s,
                                   const std::vector<std::string> &pool_strings,
                                   const std::vector<float> &weights) const {
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
        for (int64_t r = 0; r < num_rows; ++r) {
          std::vector<int64_t> row = ReadRow<int64_t>(x, r, c_dim);
          AccumulateRow<int64_t>(row, r, output_size, min_gram_length, max_gram_length,
                                 max_skip_count, ngram_indexes, root, frequencies);
        }
      }
    } else {
      NgramNode<std::string> root =
          BuildTrie<std::string>(pool_strings, ngram_counts, min_gram_length, max_gram_length);
      if (!root.leaves.empty()) {
        for (int64_t r = 0; r < num_rows; ++r) {
          std::vector<std::string> row = ReadRow<std::string>(x, r, c_dim);
          AccumulateRow<std::string>(row, r, output_size, min_gram_length, max_gram_length,
                                     max_skip_count, ngram_indexes, root, frequencies);
        }
      }
    }
  }

  std::vector<int64_t> out_shape = ComputeOutputShape(x.shape, output_size);
  std::vector<float> out_values(static_cast<size_t>(num_rows * output_size), 0.0f);
  ApplyMode(num_rows, output_size, mode, frequencies, weights, out_values.data());
  return Tensor::FromFloat("", out_shape, out_values);
}

void TfIdfVectorizer::operator()(const Tensor &x, Mode mode, int64_t min_gram_length,
                                 int64_t max_gram_length, int64_t max_skip_count,
                                 const std::vector<int64_t> &ngram_counts,
                                 const std::vector<int64_t> &ngram_indexes,
                                 const std::vector<int64_t> &pool_int64s,
                                 const std::vector<std::string> &pool_strings,
                                 const std::vector<float> &weights, Tensor &output) const {
  Tensor computed = (*this)(x, mode, min_gram_length, max_gram_length, max_skip_count, ngram_counts,
                            ngram_indexes, pool_int64s, pool_strings, weights);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::TfIdfVectorizer preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == computed.shape,
                      "kernel::TfIdfVectorizer preallocated output shape must match the "
                      "computed output shape.");
  EXT_ENFORCE_INVALID(output.data.size() == computed.data.size(),
                      "kernel::TfIdfVectorizer preallocated output buffer has unexpected size.");
  output.data = std::move(computed.data);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
