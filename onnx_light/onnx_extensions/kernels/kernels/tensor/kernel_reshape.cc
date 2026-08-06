// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ReadShapeTensor(const Tensor &shape) {
  EXT_ENFORCE_INVALID(shape.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Reshape: 'shape' input must be INT64.");
  EXT_ENFORCE_INVALID(shape.shape.size() <= 1, "kernel::Reshape: 'shape' input must be 1-D.");
  if (shape.shape.empty()) {
    return {};
  }
  const int64_t n = shape.shape[0];
  onnx_kernels::Shape out;
  out.assign(static_cast<std::size_t>(n), 0);
  if (n > 0) {
    std::memcpy(out.begin(), shape.bytes(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  return out;
}

onnx_kernels::Shape ComputeOutputShape(const Tensor &data, const onnx_kernels::Shape &target,
                                       int64_t allowzero) {
  const int64_t input_rank = static_cast<int64_t>(data.shape.size());
  onnx_kernels::Shape out;
  out.reserve(target.size());

  int64_t neg_one_axis = -1;
  int64_t known_product = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(target.size()); ++i) {
    const int64_t v = target[static_cast<std::size_t>(i)];
    if (v == -1) {
      EXT_ENFORCE_INVALID(neg_one_axis < 0,
                          "kernel::Reshape: target shape may not have multiple -1 dimensions.");
      neg_one_axis = i;
      out.push_back(-1);
    } else if (v == 0) {
      if (allowzero == 0) {
        EXT_ENFORCE_INVALID(i < input_rank, "kernel::Reshape: invalid position of 0 in target.");
        const int64_t copied = data.shape[static_cast<std::size_t>(i)];
        out.push_back(copied);
        known_product *= copied;
      } else {
        out.push_back(0);
        known_product *= 0;
      }
    } else {
      EXT_ENFORCE_INVALID(v > 0, "kernel::Reshape: invalid target dimension value.");
      out.push_back(v);
      known_product *= v;
    }
  }

  if (neg_one_axis >= 0) {
    EXT_ENFORCE_INVALID(known_product != 0,
                        "kernel::Reshape: invalid target shape product of 0 with -1.");
    int64_t input_product = 1;
    for (int64_t d : data.shape) {
      input_product *= d;
    }
    EXT_ENFORCE_INVALID(input_product % known_product == 0,
                        "kernel::Reshape: incompatible target shape.");
    out[static_cast<std::size_t>(neg_one_axis)] = input_product / known_product;
  }

  int64_t output_product = 1;
  for (int64_t d : out) {
    output_product *= d;
  }
  int64_t input_product = 1;
  for (int64_t d : data.shape) {
    input_product *= d;
  }
  EXT_ENFORCE_INVALID(output_product == input_product,
                      "kernel::Reshape: input and output must have the same number of elements.");
  return out;
}

} // namespace

Tensor Reshape::operator()(const Tensor &data, const Tensor &shape, int64_t allowzero,
                           RuntimeContext *rt) const {
  const onnx_kernels::Shape target = ReadShapeTensor(shape);
  const onnx_kernels::Shape out_shape = ComputeOutputShape(data, target, allowzero);
  const std::size_t elem_size = ElementSize(data.data_type);
  int64_t element_count = 1;
  for (int64_t d : out_shape) {
    element_count *= d;
  }
  const size_t out_n_bytes = static_cast<std::size_t>(element_count) * elem_size;
  Tensor out =
      MakeOutputTensor(data.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, shape, allowzero, out);
  return out;
}

void Reshape::operator()(const Tensor &data, const Tensor &shape, int64_t allowzero,
                         Tensor &output) const {
  const onnx_kernels::Shape target = ReadShapeTensor(shape);
  const onnx_kernels::Shape out_shape = ComputeOutputShape(data, target, allowzero);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Reshape: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Reshape: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.size_bytes() == data.size_bytes(),
                      "kernel::Reshape: preallocated output byte-size mismatch.");
  std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
}

void Reshape::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const Tensor &shape = GetInput(node, 1, rt.tensors());
  const int64_t allowzero = GetAttributeIntOrDefault(node, "allowzero", 0);
  onnx_kernels::kernel::Reshape k(rt.kernel_ctx());
  SetOutput(node, 0, k(data, shape, allowzero, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
