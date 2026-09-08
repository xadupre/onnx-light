// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceMinMax: axis is out of range.");
  return resolved;
}

Shape ComputeOutputShape(const Shape &input_shape, const Shape &is_reduced, bool keepdims) {
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

void ValidateNumericOrBool(const Tensor &t, const char *name) {
  switch (t.data_type) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::BOOL:
  case DataType::INT8:
  case DataType::INT16:
  case DataType::INT32:
  case DataType::INT64:
  case DataType::UINT8:
  case DataType::UINT16:
  case DataType::UINT32:
  case DataType::UINT64:
    return;
  default:
    EXT_THROW_INVALID("kernel::ReduceMinMax: unsupported ", name, " dtype ", t.data_type);
  }
}

template <typename T>
void MinMaxReduceTyped(const Tensor &data, const Shape &is_reduced,
                       const Shape &output_shape_noreduce, ReduceMinMax::Mode mode,
                       Tensor &output) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  const int64_t out_count = output.element_count();
  const T lower = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity()
                                                       : std::numeric_limits<T>::lowest();
  const T upper = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                       : std::numeric_limits<T>::max();
  const T init =
      mode == ReduceMinMax::Mode::kMax ? lower : (data.data_type == DataType::BOOL ? T(1) : upper);
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = init;
  }

  const T *px = reinterpret_cast<const T *>(data.bytes());
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape idx;
  idx.assign(static_cast<size_t>(rank), 0);
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
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(px[i])) {
        py[out_offset] = px[i];
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

void MinMaxReduce(const Tensor &data, const Shape &is_reduced, const Shape &output_shape_noreduce,
                  ReduceMinMax::Mode mode, Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == data.data_type &&
                          output.size_bytes() == static_cast<size_t>(output.element_count()) *
                                                     ElementSize(data.data_type),
                      "kernel::ReduceMinMax: output dtype or buffer size mismatch.");
  if (core::runtime::IsHalfPrecision(data.data_type)) {
    Tensor promoted = core::runtime::PromoteToFloat32(data);
    Tensor reduced = MakeOutputTensor(DataType::FLOAT, output.shape,
                                      output.element_count() * sizeof(float), nullptr);
    MinMaxReduceTyped<float>(promoted, is_reduced, output_shape_noreduce, mode, reduced);
    Tensor demoted = core::runtime::DemoteFromFloat32(reduced, data.data_type);
    if (output.size_bytes() > 0) {
      std::memcpy(output.mutable_bytes(), demoted.bytes(), output.size_bytes());
    }
    return;
  }
#define REDUCE_CASE(ENUM, TYPE)                                                                    \
  case DataType::ENUM:                                                                             \
    return MinMaxReduceTyped<TYPE>(data, is_reduced, output_shape_noreduce, mode, output)
  switch (data.data_type) {
    REDUCE_CASE(FLOAT, float);
    REDUCE_CASE(DOUBLE, double);
    REDUCE_CASE(BOOL, uint8_t);
    REDUCE_CASE(INT8, int8_t);
    REDUCE_CASE(INT16, int16_t);
    REDUCE_CASE(INT32, int32_t);
    REDUCE_CASE(INT64, int64_t);
    REDUCE_CASE(UINT8, uint8_t);
    REDUCE_CASE(UINT16, uint16_t);
    REDUCE_CASE(UINT32, uint32_t);
    REDUCE_CASE(UINT64, uint64_t);
  default:
    EXT_THROW_INVALID("kernel::ReduceMinMax: unsupported dtype ", data.data_type);
  }
#undef REDUCE_CASE
}

} // namespace

Tensor ReduceMinMax::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                RuntimeContext *rt) const {
  ValidateNumericOrBool(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size = ElementSize(data.data_type);
  const int32_t out_dtype = data.data_type;
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, out_dtype, out_shape, out_n_bytes)
                  : MakeOutputTensor(out_dtype, out_shape, out_n_bytes, nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMinMax::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                              Tensor &output) const {
  ValidateNumericOrBool(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceMinMax preallocated output shape does not match expected shape.");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type &&
                          output.size_bytes() == static_cast<size_t>(output.element_count()) *
                                                     ElementSize(data.data_type),
                      "kernel::ReduceMinMax: output dtype or buffer size mismatch.");

  if (noop_with_empty_axes) {
    if (data.size_bytes() > 0) {
      std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    }
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MinMaxReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

Tensor ReduceMinMax::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateNumericOrBool(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMinMax: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), 1);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = 1;
    }
  }
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size = ElementSize(data.data_type);
  const int32_t out_dtype = data.data_type;
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, out_dtype, out_shape, out_n_bytes)
                  : MakeOutputTensor(out_dtype, out_shape, out_n_bytes, nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMinMax::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes, Tensor &output) const {
  ValidateNumericOrBool(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMinMax: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());

  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), 1);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = 1;
    }
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceMinMax preallocated output shape does not match expected shape.");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type &&
                          output.size_bytes() == static_cast<size_t>(output.element_count()) *
                                                     ElementSize(data.data_type),
                      "kernel::ReduceMinMax: output dtype or buffer size mismatch.");

  if (naxes == 0 && noop_with_empty_axes) {
    if (data.size_bytes() > 0) {
      std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    }
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MinMaxReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

void ReduceMax::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op '", node.op_type(),
                      "' expects at most 2 inputs.");
  RequireOutputCount(node, 1);
  const bool keepdims = GetAttributeIntOrDefault(node, "keepdims", 1) != 0;
  const bool noop_with_empty_axes = GetAttributeIntOrDefault(node, "noop_with_empty_axes", 0) != 0;
  const std::vector<int64_t> axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
  const bool has_axes_attr = !axes_attr.empty();
  const Tensor axes_attr_tensor =
      axes_attr.empty()
          ? Tensor()
          : Tensor::FromInt64("", {static_cast<int64_t>(axes_attr.size())}, axes_attr);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
  if (axes_input != nullptr) {
    SetOutput(node, 0, (*this)(data, *axes_input, keepdims, noop_with_empty_axes, &rt), rt);
    return;
  }
  if (has_axes_attr) {
    SetOutput(node, 0, (*this)(data, axes_attr_tensor, keepdims, noop_with_empty_axes, &rt), rt);
    return;
  }
  SetOutput(node, 0, (*this)(data, keepdims, noop_with_empty_axes, &rt), rt);
}

void ReduceMin::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op '", node.op_type(),
                      "' expects at most 2 inputs.");
  RequireOutputCount(node, 1);
  const bool keepdims = GetAttributeIntOrDefault(node, "keepdims", 1) != 0;
  const bool noop_with_empty_axes = GetAttributeIntOrDefault(node, "noop_with_empty_axes", 0) != 0;
  const std::vector<int64_t> axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
  const bool has_axes_attr = !axes_attr.empty();
  const Tensor axes_attr_tensor =
      axes_attr.empty()
          ? Tensor()
          : Tensor::FromInt64("", {static_cast<int64_t>(axes_attr.size())}, axes_attr);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
  if (axes_input != nullptr) {
    SetOutput(node, 0, (*this)(data, *axes_input, keepdims, noop_with_empty_axes, &rt), rt);
    return;
  }
  if (has_axes_attr) {
    SetOutput(node, 0, (*this)(data, axes_attr_tensor, keepdims, noop_with_empty_axes, &rt), rt);
    return;
  }
  SetOutput(node, 0, (*this)(data, keepdims, noop_with_empty_axes, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
