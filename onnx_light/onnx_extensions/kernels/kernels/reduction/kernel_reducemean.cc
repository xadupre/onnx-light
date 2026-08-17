// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceMean: axis is out of range.");
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

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ReduceMean: ", name, " must be a FLOAT tensor.");
}

// Computes the arithmetic mean of elements of ``data`` along the reduced
// dimensions and writes the result into the output buffer. The empty-set
// identity for ``mean`` is undefined (division by zero); ONNX leaves the
// behaviour unspecified in that case, but no upstream reference test
// exercises it so we simply produce ``0`` values like the other reductions
// do when no elements are aggregated.
void MeanReduce(const Tensor &data, const Shape &is_reduced, const Shape &output_shape_noreduce,
                Tensor &output) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);
  float *py = output.AsFloat();
  const int64_t out_count = output.element_count();
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = 0.0f;
  }

  // Number of input elements aggregated into each output element: product of
  // the sizes of the reduced dimensions.
  int64_t reduced_count = 1;
  for (size_t d = 0; d < data.shape.size(); ++d) {
    if (is_reduced[d]) {
      reduced_count *= data.shape[d];
    }
  }

  const float *px = data.AsFloat();
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
    py[out_offset] += px[i];
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  if (reduced_count > 0) {
    const float denom = static_cast<float>(reduced_count);
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] /= denom;
    }
  }
}

} // namespace

Tensor ReduceMean::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                              RuntimeContext *rt) const {
  ValidateFloat(data, "data");
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
  const size_t out_n_bytes = static_cast<size_t>(out_count) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMean::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                            Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  Shape is_reduced;
  is_reduced.assign(static_cast<size_t>(rank), 0);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), 1);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceMean preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceMean preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MeanReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceMean::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMean: axes must be an INT64 tensor.");
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
  const size_t out_n_bytes = static_cast<size_t>(out_count) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceMean::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                            bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMean: axes must be an INT64 tensor.");
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
      "kernel::ReduceMean preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceMean preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MeanReduce(data, is_reduced, out_shape_noreduce, output);
}

void ReduceMean::Run(RuntimeContext &rt) {
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
