// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceLogSumOp: axis is out of range.");
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

void ValidateFloatOrDouble(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          t.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::ReduceLogSumOp: ", name, " must be a FLOAT or DOUBLE tensor.");
}

// Templated reduction core for both FLOAT and DOUBLE.
template <typename T>
void LogSumReduceT(const T *px, T *py, int64_t out_count, int64_t total, int64_t rank,
                   const Shape &data_shape, const Shape &is_reduced,
                   const Shape &output_shape_noreduce, ReduceLogSumOp::Mode mode,
                   RawBufferAllocator *allocator) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);

  int64_t reduced_count = 1;
  for (size_t d = 0; d < data_shape.size(); ++d) {
    if (is_reduced[d]) {
      reduced_count *= data_shape[d];
    }
  }
  if (reduced_count == 0) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = neg_inf;
    }
    return;
  }

  if (mode == ReduceLogSumOp::Mode::kLogSum) {
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = static_cast<T>(0);
    }
    Shape idx;
    idx.assign(static_cast<size_t>(rank), 0);
    for (int64_t i = 0; i < total; ++i) {
      int64_t out_offset = 0;
      size_t out_dim = 0;
      for (int64_t d = 0; d < rank; ++d) {
        if (!is_reduced[static_cast<size_t>(d)]) {
          out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
          ++out_dim;
        }
      }
      py[out_offset] += px[i];
      for (int64_t d = rank - 1; d >= 0; --d) {
        ++idx[static_cast<size_t>(d)];
        if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
          break;
        }
        idx[static_cast<size_t>(d)] = 0;
      }
    }
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = std::log(py[i]);
    }
    return;
  }

  // kLogSumExp: use the max-shift trick.
  const T neg_inf = -std::numeric_limits<T>::infinity();
  detail::TemporaryTypedBuffer<T> max_vals_buf(static_cast<size_t>(out_count), allocator,
                                               "kernel::ReduceLogSumOp max_vals");
  T *max_vals = max_vals_buf.data();
  for (int64_t i = 0; i < out_count; ++i) {
    max_vals[static_cast<size_t>(i)] = neg_inf;
  }
  Shape idx;
  idx.assign(static_cast<size_t>(rank), 0);
  for (int64_t i = 0; i < total; ++i) {
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    const T v = px[i];
    if (v > max_vals[static_cast<size_t>(out_offset)]) {
      max_vals[static_cast<size_t>(out_offset)] = v;
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = static_cast<T>(0);
  }
  std::fill(idx.begin(), idx.end(), 0);
  for (int64_t i = 0; i < total; ++i) {
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    const T m = max_vals[static_cast<size_t>(out_offset)];
    if (std::isfinite(m)) {
      py[out_offset] += std::exp(px[i] - m);
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
  for (int64_t i = 0; i < out_count; ++i) {
    const T m = max_vals[static_cast<size_t>(i)];
    if (!std::isfinite(m)) {
      py[i] = m;
    } else {
      py[i] = m + std::log(py[i]);
    }
  }
}

void LogSumReduce(const Tensor &data, const Shape &is_reduced, const Shape &output_shape_noreduce,
                  ReduceLogSumOp::Mode mode, Tensor &output, RawBufferAllocator *allocator) {
  const int64_t out_count = output.element_count();
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const int64_t total = data.element_count();
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    LogSumReduceT<double>(data.AsDouble(), output.AsDouble(), out_count, total, rank, data.shape,
                          is_reduced, output_shape_noreduce, mode, allocator);
  } else {
    LogSumReduceT<float>(data.AsFloat(), output.AsFloat(), out_count, total, rank, data.shape,
                         is_reduced, output_shape_noreduce, mode, allocator);
  }
}

template <typename T>
void LogSumNoopElementwiseT(const T *px, T *py, int64_t total, ReduceLogSumOp::Mode mode) {
  switch (mode) {
  case ReduceLogSumOp::Mode::kLogSum:
    for (int64_t i = 0; i < total; ++i) {
      py[i] = std::log(px[i]);
    }
    break;
  case ReduceLogSumOp::Mode::kLogSumExp:
    // identity: log(exp(x)) == x
    break;
  }
}

void LogSumNoopElementwise(const Tensor &data, ReduceLogSumOp::Mode mode, Tensor &output) {
  const int64_t total = data.element_count();
  if (mode == ReduceLogSumOp::Mode::kLogSumExp) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    LogSumNoopElementwiseT<double>(data.AsDouble(), output.AsDouble(), total, mode);
  } else {
    LogSumNoopElementwiseT<float>(data.AsFloat(), output.AsFloat(), total, mode);
  }
}

} // namespace

Tensor ReduceLogSumOp::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                  RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
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
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, out_n_bytes)
                  : MakeOutputTensor(data.data_type, out_shape, out_n_bytes, nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out, rt ? rt->execution_allocator() : nullptr);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                Tensor &output, RawBufferAllocator *temporary_allocator) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ReduceLogSumOp: output dtype must match data dtype.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output, temporary_allocator);
}

Tensor ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                  bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceLogSumOp: axes must be an INT64 tensor.");
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
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, out_n_bytes)
                  : MakeOutputTensor(data.data_type, out_shape, out_n_bytes, nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out,
          rt ? rt->execution_allocator() : nullptr);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                bool noop_with_empty_axes, Tensor &output,
                                RawBufferAllocator *temporary_allocator) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ReduceLogSumOp: output dtype must match data dtype.");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceLogSumOp: axes must be an INT64 tensor.");
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
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output, temporary_allocator);
}

void ReduceLogSum::Run(RuntimeContext &rt) {
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

void ReduceLogSumExp::Run(RuntimeContext &rt) {
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
