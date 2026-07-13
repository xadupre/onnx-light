// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_kernels/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceMinMax: axis is out of range.");
  return resolved;
}

Shape ComputeOutputShape(const Shape &input_shape, const std::vector<bool> &is_reduced,
                         bool keepdims) {
  Shape out_shape;
  out_shape.reserve(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); ++d) {
    if (is_reduced[d]) {
      if (keepdims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input_shape[d]);
    }
  }
  return out_shape;
}

Shape RowMajorStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ReduceMinMax: ", name, " must be a FLOAT tensor.");
}

void ValidateFloatOrBool(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          t.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::ReduceMinMax: ", name, " must be a FLOAT or BOOL tensor.");
}

void MinMaxReduce(const Tensor &data, const std::vector<bool> &is_reduced,
                  const Shape &output_shape_noreduce, ReduceMinMax::Mode mode, Tensor &output) {
  // BOOL path: ReduceMax = OR, ReduceMin = AND.
  if (data.data_type == static_cast<int32_t>(DataType::BOOL)) {
    const Shape out_strides = RowMajorStrides(output_shape_noreduce);
    uint8_t *py = output.mutable_bytes();
    const int64_t out_count = output.element_count();
    const uint8_t init =
        mode == ReduceMinMax::Mode::kMax ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1);
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = init;
    }

    const uint8_t *px = data.AsBool();
    const int64_t rank = static_cast<int64_t>(data.shape.size());
    std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
    const int64_t total = data.element_count();
    for (int64_t i = 0; i < total; ++i) {
      int64_t out_offset = 0;
      size_t out_dim = 0;
      for (int64_t d = 0; d < rank; ++d) {
        if (!is_reduced[static_cast<size_t>(d)]) {
          out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
          ++out_dim;
        }
      }
      if (mode == ReduceMinMax::Mode::kMax) {
        py[out_offset] = py[out_offset] | px[i];
      } else {
        py[out_offset] = py[out_offset] & px[i];
      }
      for (int64_t d = rank - 1; d >= 0; --d) {
        ++idx[static_cast<size_t>(d)];
        if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
          break;
        }
        idx[static_cast<size_t>(d)] = 0;
      }
    }
    return;
  }

  // FLOAT path.
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);
  float *py = output.AsFloat();
  const int64_t out_count = output.element_count();
  const float init = mode == ReduceMinMax::Mode::kMax ? -std::numeric_limits<float>::infinity()
                                                      : std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = init;
  }

  const float *px = data.AsFloat();
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
  const int64_t total = data.element_count();
  for (int64_t i = 0; i < total; ++i) {
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    if (mode == ReduceMinMax::Mode::kMax) {
      py[out_offset] = std::max(py[out_offset], px[i]);
    } else {
      py[out_offset] = std::min(py[out_offset], px[i]);
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
}

} // namespace

Tensor ReduceMinMax::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                RuntimeContext *rt) const {
  ValidateFloatOrBool(data, "data");
  const bool is_bool = data.data_type == static_cast<int32_t>(DataType::BOOL);
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size = is_bool ? sizeof(uint8_t) : sizeof(float);
  const int32_t out_dtype =
      is_bool ? static_cast<int32_t>(DataType::BOOL) : static_cast<int32_t>(DataType::FLOAT);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = MakeOutputTensor(out_dtype, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMinMax::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                              Tensor &output) const {
  ValidateFloatOrBool(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceMinMax preallocated output shape does not match expected shape.");

  if (noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MinMaxReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

Tensor ReduceMinMax::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrBool(data, "data");
  const bool is_bool = data.data_type == static_cast<int32_t>(DataType::BOOL);
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMinMax: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), true);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = true;
    }
  }
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size = is_bool ? sizeof(uint8_t) : sizeof(float);
  const int32_t out_dtype =
      is_bool ? static_cast<int32_t>(DataType::BOOL) : static_cast<int32_t>(DataType::FLOAT);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = MakeOutputTensor(out_dtype, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMinMax::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloatOrBool(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMinMax: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());

  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), true);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = true;
    }
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceMinMax preallocated output shape does not match expected shape.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MinMaxReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
