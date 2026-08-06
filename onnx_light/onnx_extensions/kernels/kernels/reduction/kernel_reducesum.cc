// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Resolves a possibly-negative axis (ONNX semantics: ``axis`` in
// ``[-rank, rank - 1]``) to a non-negative axis. Throws on out-of-range.
int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, "kernel::ReduceSum: axis is out of range.");
  return resolved;
}

// Computes the output shape of a ReduceSum: dimensions in ``reduce_axes`` are
// either dropped (when ``keepdims`` is false) or replaced by 1.
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

// Row-major strides for ``shape``. Each stride is the number of elements one
// must skip to advance by one along that dimension.
Shape RowMajorStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Sums ``data`` into ``output`` by reducing along the dimensions for which
// ``is_reduced[d]`` is true. ``output`` is laid out with the non-reduced
// dimensions only (i.e. ``keepdims == false``); callers that want the
// keepdims layout reshape the same byte buffer afterwards.
template <typename T>
void SumReduceT(const Tensor &data, const Shape &is_reduced, const Shape &output_shape_noreduce,
                Tensor &output) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);

  // Zero-initialize the output bytes so we can accumulate into it.
  std::memset(output.mutable_bytes(), 0, output.size_bytes());

  const T *px = reinterpret_cast<const T *>(data.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());

  // Iterate over every element of the input using a multi-dimensional index.
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape idx;
  idx.assign(static_cast<size_t>(rank), 0);
  const int64_t total = data.element_count();
  for (int64_t i = 0; i < total; ++i) {
    // Compute the output offset by walking through the non-reduced dims.
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    py[out_offset] += px[i];

    // Increment the multi-dimensional index (row-major / C order).
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
}

void SumReduce(const Tensor &data, const Shape &is_reduced, const Shape &output_shape_noreduce,
               Tensor &output) {
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    SumReduceT<double>(data, is_reduced, output_shape_noreduce, output);
  } else {
    SumReduceT<float>(data, is_reduced, output_shape_noreduce, output);
  }
}

void ValidateFloatOrDouble(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          t.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::ReduceSum: ", name, " must be a FLOAT or DOUBLE tensor.");
}

} // namespace

Tensor ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                             RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    // Reduce over all dimensions.
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }

  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  const int64_t out_count = out_shape.empty() ? 1 : [&out_shape]() {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return n;
  }();
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out =
      MakeOutputTensor(data.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                           Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  ValidateFloatOrDouble(output, "output");
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);
  const int64_t rank = static_cast<int64_t>(data.shape.size());

  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }

  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::ReduceSum preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }

  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                             bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

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
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out =
      MakeOutputTensor(data.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                           bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  ValidateFloatOrDouble(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

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
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::ReduceSum preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }

  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

void ReduceSum::Run(RuntimeContext &rt) {
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
