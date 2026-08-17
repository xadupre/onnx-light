// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Resolves the ONNX ``Shape`` operator's ``start``/``end`` attributes against
// an input of rank ``rank``. Negative values count from the back; the
// returned indices are clamped to ``[0, rank]`` per the upstream spec.
void ResolveStartEnd(const Shape::Attributes &attrs, int64_t rank, int64_t &start, int64_t &end) {
  start = attrs.start;
  if (start < 0) {
    start += rank;
  }
  if (start < 0) {
    start = 0;
  }
  if (start > rank) {
    start = rank;
  }

  end = attrs.end.has_value() ? *attrs.end : rank;
  if (end < 0) {
    end += rank;
  }
  if (end < 0) {
    end = 0;
  }
  if (end > rank) {
    end = rank;
  }
}

onnx_kernels::Shape ComputeShapeSlice(const Tensor &data, const Shape::Attributes &attrs) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  int64_t start = 0;
  int64_t end = 0;
  ResolveStartEnd(attrs, rank, start, end);
  onnx_kernels::Shape values;
  if (end > start) {
    values.reserve(static_cast<std::size_t>(end - start));
    for (int64_t i = start; i < end; ++i) {
      values.push_back(data.shape[static_cast<std::size_t>(i)]);
    }
  }
  return values;
}

} // namespace

Tensor Shape::operator()(const Tensor &data, RuntimeContext *rt) const {
  return (*this)(data, Attributes{}, rt);
}

Tensor Shape::operator()(const Tensor &data, const Attributes &attrs, RuntimeContext *rt) const {
  const onnx_kernels::Shape values = ComputeShapeSlice(data, attrs);
  const onnx_kernels::Shape out_shape{static_cast<int64_t>(values.size())};
  Tensor out =
      rt ? rt->MakeOutputTensor(0, DataType::INT64, out_shape, values.size() * sizeof(int64_t))
         : Tensor::FromInt64("", out_shape, values, ctx_.allocator);
  if (rt != nullptr && !values.empty()) {
    std::copy(values.begin(), values.end(), out.AsInt64());
  }
  return out;
}

void Shape::operator()(const Tensor &data, const Attributes &attrs, Tensor &output) const {
  const onnx_kernels::Shape values = ComputeShapeSlice(data, attrs);
  const onnx_kernels::Shape out_shape{static_cast<int64_t>(values.size())};
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Shape: preallocated output dtype must be INT64.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Shape: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.size_bytes() == values.size() * sizeof(int64_t),
                      "kernel::Shape: preallocated output byte-size mismatch.");
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), reinterpret_cast<int64_t *>(output.mutable_bytes()));
  }
}

void Shape::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::Shape::Attributes shape_attrs;
  shape_attrs.start = GetAttributeIntOrDefault(node, "start", 0);
  const AttributeProto *end_attr = FindAttribute(node, "end");
  if (end_attr != nullptr) {
    shape_attrs.end = end_attr->i();
  }
  onnx_kernels::kernel::Shape shape_kernel(rt.kernel_ctx());
  SetOutput(node, 0, shape_kernel(data, shape_attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
