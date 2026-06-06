// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  int64_t a = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(a > 0 && a < rank, "kernel::TensorScatter: 'axis' must designate a non-batch "
                                         "dimension in [-rank, -1] / [1, rank-1].");
  return a;
}

// Row-major strides in elements.
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  const std::size_t r = shape.size();
  std::vector<int64_t> strides(r, 1);
  for (std::size_t i = r; i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

} // namespace

Tensor TensorScatter::operator()(const Tensor &past_cache, const Tensor &update,
                                 const Tensor *write_indices,
                                 const TensorScatter::Attributes &attrs) const {
  Tensor output;
  output.name = "";
  output.data_type = past_cache.data_type;
  output.shape = past_cache.shape;
  if (past_cache.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data.assign(static_cast<std::size_t>(past_cache.element_count()), std::string());
  } else {
    output.data.assign(PackedByteSize(past_cache.data_type, past_cache.element_count()),
                       static_cast<uint8_t>(0));
  }
  (*this)(past_cache, update, write_indices, attrs, output);
  return output;
}

void TensorScatter::operator()(const Tensor &past_cache, const Tensor &update,
                               const Tensor *write_indices, const TensorScatter::Attributes &attrs,
                               Tensor &output) const {
  EXT_ENFORCE_INVALID(past_cache.data_type == update.data_type,
                      "kernel::TensorScatter: 'past_cache' and 'update' must share dtype.");
  EXT_ENFORCE_INVALID(output.data_type == past_cache.data_type,
                      "kernel::TensorScatter: preallocated output dtype must match 'past_cache'.");
  EXT_ENFORCE_INVALID(past_cache.shape.size() == update.shape.size(),
                      "kernel::TensorScatter: 'past_cache' and 'update' must have the same rank.");
  EXT_ENFORCE_INVALID(past_cache.shape.size() >= 2,
                      "kernel::TensorScatter: rank must be >= 2 (batch + sequence + ...).");

  EXT_ENFORCE_INVALID(attrs.mode == "linear" || attrs.mode == "circular",
                      "kernel::TensorScatter: 'mode' must be 'linear' or 'circular'.");

  const int64_t rank = static_cast<int64_t>(past_cache.shape.size());
  const int64_t axis = NormalizeAxis(attrs.axis, rank);

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      EXT_ENFORCE_INVALID(update.shape[static_cast<std::size_t>(i)] <=
                              past_cache.shape[static_cast<std::size_t>(i)],
                          "kernel::TensorScatter: update length on 'axis' must be <= "
                          "past_cache length on 'axis'.");
    } else {
      EXT_ENFORCE_INVALID(update.shape[static_cast<std::size_t>(i)] ==
                              past_cache.shape[static_cast<std::size_t>(i)],
                          "kernel::TensorScatter: 'past_cache' and 'update' must agree on "
                          "every dimension other than 'axis'.");
    }
  }
  EXT_ENFORCE_INVALID(output.shape == past_cache.shape,
                      "kernel::TensorScatter: preallocated output shape mismatch.");

  const int64_t batch_size = past_cache.shape[0];
  const int64_t max_seq = past_cache.shape[static_cast<std::size_t>(axis)];
  const int64_t seq_len = update.shape[static_cast<std::size_t>(axis)];

  // Read write_indices (or default to all zeros).
  std::vector<int64_t> writes(static_cast<std::size_t>(batch_size), 0);
  if (write_indices != nullptr) {
    EXT_ENFORCE_INVALID(write_indices->data_type == static_cast<int32_t>(DataType::INT64),
                        "kernel::TensorScatter: 'write_indices' must be a tensor(int64).");
    EXT_ENFORCE_INVALID(write_indices->shape.size() == 1 && write_indices->shape[0] == batch_size,
                        "kernel::TensorScatter: 'write_indices' must have shape (batch_size,).");
    const int64_t *src = write_indices->AsInt64();
    for (int64_t i = 0; i < batch_size; ++i) {
      writes[static_cast<std::size_t>(i)] = src[i];
    }
  }

  // Initialize output := past_cache (copy).
  const bool is_string = past_cache.data_type == static_cast<int32_t>(DataType::STRING);
  const std::size_t total = static_cast<std::size_t>(past_cache.element_count());
  if (is_string) {
    EXT_ENFORCE_INVALID(past_cache.string_data.size() == total,
                        "kernel::TensorScatter: 'past_cache' string_data size mismatch.");
    output.string_data.assign(past_cache.string_data.begin(), past_cache.string_data.end());
  } else {
    const std::size_t bytes = PackedByteSize(past_cache.data_type, past_cache.element_count());
    EXT_ENFORCE_INVALID(past_cache.data.size() == bytes,
                        "kernel::TensorScatter: 'past_cache' data size mismatch.");
    std::memcpy(output.data.data(), past_cache.data.data(), bytes);
  }

  // Strides for cache (output) and update tensors (in elements).
  const std::vector<int64_t> cache_strides = RowMajorStrides(past_cache.shape);
  const std::vector<int64_t> update_strides = RowMajorStrides(update.shape);

  // Product of dims [0..axis): the prefix indices we iterate over.
  int64_t prefix_total = 1;
  for (int64_t i = 0; i < axis; ++i) {
    prefix_total *= past_cache.shape[static_cast<std::size_t>(i)];
  }
  // Product of dims (axis..rank): the slice size in elements.
  int64_t slice_elems = 1;
  for (int64_t i = axis + 1; i < rank; ++i) {
    slice_elems *= past_cache.shape[static_cast<std::size_t>(i)];
  }

  const std::size_t elem_size = is_string ? 0 : ElementSize(past_cache.data_type);
  const std::size_t slice_bytes = is_string ? 0 : static_cast<std::size_t>(slice_elems) * elem_size;

  // Strides through the leading `axis` dims (decompose the linear prefix
  // index into per-dim coordinates so we can recover batch_idx = coord[0]).
  std::vector<int64_t> prefix_strides(static_cast<std::size_t>(axis), 1);
  for (int64_t k = axis - 2; k >= 0; --k) {
    prefix_strides[static_cast<std::size_t>(k)] = prefix_strides[static_cast<std::size_t>(k + 1)] *
                                                  past_cache.shape[static_cast<std::size_t>(k + 1)];
  }

  for (int64_t p = 0; p < prefix_total; ++p) {
    int64_t remaining = p;
    int64_t batch_idx = 0;
    int64_t cache_prefix_offset = 0;
    int64_t update_prefix_offset = 0;
    for (int64_t k = 0; k < axis; ++k) {
      const int64_t c = remaining / prefix_strides[static_cast<std::size_t>(k)];
      remaining %= prefix_strides[static_cast<std::size_t>(k)];
      if (k == 0) {
        batch_idx = c;
      }
      cache_prefix_offset += c * cache_strides[static_cast<std::size_t>(k)];
      update_prefix_offset += c * update_strides[static_cast<std::size_t>(k)];
    }

    const int64_t base_write = writes[static_cast<std::size_t>(batch_idx)];
    for (int64_t s = 0; s < seq_len; ++s) {
      int64_t cache_axis_idx = base_write + s;
      if (attrs.mode == "circular") {
        cache_axis_idx = ((cache_axis_idx % max_seq) + max_seq) % max_seq;
      } else {
        EXT_ENFORCE_INVALID(cache_axis_idx >= 0 && cache_axis_idx < max_seq,
                            "kernel::TensorScatter: linear mode requires "
                            "write_indices+sequence_length<=max_sequence_length and "
                            "write_indices>=0.");
      }
      const int64_t cache_offset =
          cache_prefix_offset + cache_axis_idx * cache_strides[static_cast<std::size_t>(axis)];
      const int64_t update_offset =
          update_prefix_offset + s * update_strides[static_cast<std::size_t>(axis)];

      if (is_string) {
        for (int64_t e = 0; e < slice_elems; ++e) {
          output.string_data[static_cast<std::size_t>(cache_offset + e)] =
              update.string_data[static_cast<std::size_t>(update_offset + e)];
        }
      } else {
        std::memcpy(output.data.data() + static_cast<std::size_t>(cache_offset) * elem_size,
                    update.data.data() + static_cast<std::size_t>(update_offset) * elem_size,
                    slice_bytes);
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
