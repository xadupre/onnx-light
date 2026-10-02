// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kMeanName = "kernel::Mean";

constexpr const char *kSupportedMeanTypesMsg = " only supports FLOAT and DOUBLE inputs.";

template <typename T> T MeanAddOf(T a, T b) { return a + b; }

template <typename T>
void BroadcastCopyFirst(const char *dtype_name, int32_t dtype, const Tensor &input,
                        Tensor &output) {
  const detail::BroadcastInfo bi =
      detail::CheckBinaryBroadcast(kMeanName, dtype_name, dtype, input, output);
  const T *src = reinterpret_cast<const T *>(input.bytes());
  T *dst = reinterpret_cast<T *>(output.mutable_bytes());
  if (input.shape == output.shape) {
    std::memcpy(dst, src, static_cast<size_t>(bi.element_count) * sizeof(T));
    return;
  }
  if (bi.nx == 1) {
    std::fill_n(dst, bi.element_count, src[0]);
    return;
  }
  const size_t rank = bi.shape.size();
  Shape index;
  index.assign(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t offset = 0;
    for (size_t d = 0; d < rank; ++d) {
      offset += index[d] * bi.strides_x[d];
    }
    dst[flat] = src[offset];
    for (size_t d = rank; d-- > 0;) {
      if (++index[d] < bi.shape[d]) {
        break;
      }
      index[d] = 0;
    }
  }
}

// Computes the broadcast shape of every tensor in ``inputs``. ``inputs`` must
// be non-empty and all tensors must share ``expected_dtype``.
Shape ValidateAndBroadcastShape(const Tensors &inputs, const char *dtype_name,
                                int32_t expected_dtype) {
  EXT_ENFORCE_INVALID(!inputs.empty(), kMeanName, " requires at least one input.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == expected_dtype, kMeanName, " only supports ",
                        dtype_name, " tensors.");
  }
  Shape shape = inputs[0].shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    shape = detail::BroadcastShape(kMeanName, shape, inputs[i].shape);
  }
  return shape;
}

template <typename T>
void AccumulateAndScale(const char *dtype_name, int32_t dtype, const Tensors &inputs,
                        Tensor &output) {
  // Single input: copy verbatim. ``Mean`` of a single tensor is the tensor itself.
  if (inputs.size() == 1) {
    std::memcpy(output.mutable_bytes(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return;
  }
  const bool first_pair_fills_output =
      detail::BroadcastShape(kMeanName, inputs[0].shape, inputs[1].shape) == output.shape;
  size_t next_input = 1;
  if (first_pair_fills_output) {
    detail::BinaryElementwise<T, T>(kMeanName, dtype_name, dtype, inputs[0], inputs[1], output,
                                    MeanAddOf<T>);
    next_input = 2;
  } else {
    BroadcastCopyFirst<T>(dtype_name, dtype, inputs[0], output);
  }
  for (size_t i = next_input; i < inputs.size(); ++i) {
    detail::BinaryElementwise<T, T>(kMeanName, dtype_name, dtype, output, inputs[i], output,
                                    MeanAddOf<T>);
  }
  // Divide the accumulated sum by the input count to obtain the mean.
  const T inv_n = static_cast<T>(1) / static_cast<T>(inputs.size());
  T *out_ptr = reinterpret_cast<T *>(output.mutable_bytes());
  const int64_t n_elements = output.element_count();
  for (int64_t i = 0; i < n_elements; ++i) {
    out_ptr[i] *= inv_n;
  }
}

template <typename T>
Tensor MeanAlloc(const char *dtype_name, int32_t dtype, const Tensors &inputs,
                 RawBufferAllocator *allocator) {
  const Shape out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t z_n_bytes = static_cast<size_t>(out_count) * sizeof(T);
  Tensor z = MakeOutputTensor(dtype, out_shape, z_n_bytes, allocator);
  AccumulateAndScale<T>(dtype_name, dtype, inputs, z);
  return z;
}

template <typename T>
void MeanInPlace(const char *dtype_name, int32_t dtype, const Tensors &inputs, Tensor &output) {
  const Shape out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  const size_t expected_bytes = [&]() {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return static_cast<size_t>(n) * sizeof(T);
  }();
  detail::CheckPreallocatedOutput(kMeanName, dtype_name, dtype, out_shape, expected_bytes, output);
  AccumulateAndScale<T>(dtype_name, dtype, inputs, output);
}

} // namespace

Tensor Mean::operator()(const Tensors &inputs, RuntimeContext *rt) const {
  if (rt != nullptr) {
    const Shape out_shape = ValidateAndBroadcastShape(inputs, "", inputs[0].data_type);
    Tensor output = rt->MakeOutputTensor(0, inputs[0].data_type, out_shape,
                                         static_cast<size_t>(out_shape.product()) *
                                             ElementSize(inputs[0].data_type));
    (*this)(inputs, output);
    return output;
  }
  EXT_ENFORCE_INVALID(!inputs.empty(), kMeanName, " requires at least one input.");
  RawBufferAllocator *allocator = nullptr;
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return MeanAlloc<float>("FLOAT", DataType::FLOAT, inputs, allocator);
  case DataType::DOUBLE:
    return MeanAlloc<double>("DOUBLE", DataType::DOUBLE, inputs, allocator);
  default:
    EXT_THROW_INVALID(kMeanName, ": unsupported data type ", inputs[0].data_type,
                      kSupportedMeanTypesMsg);
  }
}

void Mean::operator()(const Tensors &inputs, Tensor &output) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), kMeanName, " requires at least one input.");
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return MeanInPlace<float>("FLOAT", DataType::FLOAT, inputs, output);
  case DataType::DOUBLE:
    return MeanInPlace<double>("DOUBLE", DataType::DOUBLE, inputs, output);
  default:
    EXT_THROW_INVALID(kMeanName, ": unsupported data type ", inputs[0].data_type,
                      kSupportedMeanTypesMsg);
  }
}

void Mean::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  RequireOutputCount(node, 1);
  Tensors inputs;
  inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    inputs.push_back(GetInput(node, i, rt.tensors()));
  }
  SetOutput(node, 0, (*this)(inputs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
