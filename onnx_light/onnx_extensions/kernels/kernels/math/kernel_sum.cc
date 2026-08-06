// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kSumName = "kernel::Sum";

constexpr const char *kSupportedSumTypesMsg = " only supports FLOAT and DOUBLE inputs.";

// Returns the multidirectional-broadcast output shape of ``a`` and ``b``.
Shape BroadcastShape(const Shape &a, const Shape &b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  Shape sa, sb, out;
  sa.assign(rank, 1);
  sb.assign(rank, 1);
  out.assign(rank, 1);
  for (size_t i = 0; i < a.size(); ++i) {
    sa[rank - a.size() + i] = a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    sb[rank - b.size() + i] = b[i];
  }
  for (size_t d = 0; d < rank; ++d) {
    if (sa[d] == sb[d] || sa[d] == 1 || sb[d] == 1) {
      out[d] = sa[d] >= sb[d] ? sa[d] : sb[d];
    } else {
      EXT_THROW_INVALID(kSumName, " input shapes are not multidirectional-broadcastable.");
    }
  }
  return out;
}

// Computes the broadcast shape of every tensor in ``inputs``. ``inputs`` must
// be non-empty and all tensors must share ``expected_dtype``.
Shape ValidateAndBroadcastShape(const Tensors &inputs, const char *dtype_name,
                                int32_t expected_dtype) {
  EXT_ENFORCE_INVALID(!inputs.empty(), kSumName, " requires at least one input.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == expected_dtype, kSumName, " only supports ",
                        dtype_name, " tensors.");
  }
  Shape shape = inputs[0].shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    shape = BroadcastShape(shape, inputs[i].shape);
  }
  return shape;
}

template <typename T>
Tensor SumAlloc(const char *dtype_name, int32_t dtype, const Tensors &inputs,
                RawBufferAllocator *allocator) {
  const Shape out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t z_n_bytes = static_cast<size_t>(out_count) * sizeof(T);
  Tensor z = MakeOutputTensor(dtype, out_shape, z_n_bytes, allocator);
  if (inputs.size() == 1) {
    // Single input: copy verbatim. We still go through the broadcast check
    // above so a malformed input shape would have already thrown.
    std::memcpy(z.mutable_bytes(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return z;
  }
  // First pair: accumulate into the output buffer.
  detail::BinaryElementwise<T, T>(kSumName, dtype_name, dtype, inputs[0], inputs[1], z,
                                  [](T a, T b) -> T { return a + b; });
  // Subsequent inputs: accumulate in place by re-running the binary
  // element-wise driver with ``z`` as both an input and the output. The
  // driver supports aliasing when the input is not broadcast-expanded; ``z``
  // has the full broadcast shape so its stride pattern matches the output.
  for (size_t i = 2; i < inputs.size(); ++i) {
    Tensor partial = z;
    detail::BinaryElementwise<T, T>(kSumName, dtype_name, dtype, partial, inputs[i], z,
                                    [](T a, T b) -> T { return a + b; });
  }
  return z;
}

template <typename T>
void SumInPlace(const char *dtype_name, int32_t dtype, const Tensors &inputs, Tensor &output) {
  const Shape out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  const size_t expected_bytes = [&]() {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return static_cast<size_t>(n) * sizeof(T);
  }();
  detail::CheckPreallocatedOutput(kSumName, dtype_name, dtype, out_shape, expected_bytes, output);
  if (inputs.size() == 1) {
    std::memcpy(output.mutable_bytes(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return;
  }
  detail::BinaryElementwise<T, T>(kSumName, dtype_name, dtype, inputs[0], inputs[1], output,
                                  [](T a, T b) -> T { return a + b; });
  for (size_t i = 2; i < inputs.size(); ++i) {
    Tensor partial = output;
    detail::BinaryElementwise<T, T>(kSumName, dtype_name, dtype, partial, inputs[i], output,
                                    [](T a, T b) -> T { return a + b; });
  }
}

} // namespace

Tensor Sum::operator()(const Tensors &inputs, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), kSumName, " requires at least one input.");
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return SumAlloc<float>("FLOAT", DataType::FLOAT, inputs, allocator);
  case DataType::DOUBLE:
    return SumAlloc<double>("DOUBLE", DataType::DOUBLE, inputs, allocator);
  default:
    EXT_THROW_INVALID(kSumName, ": unsupported data type ", inputs[0].data_type,
                      kSupportedSumTypesMsg);
  }
}

void Sum::operator()(const Tensors &inputs, Tensor &output) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), kSumName, " requires at least one input.");
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return SumInPlace<float>("FLOAT", DataType::FLOAT, inputs, output);
  case DataType::DOUBLE:
    return SumInPlace<double>("DOUBLE", DataType::DOUBLE, inputs, output);
  default:
    EXT_THROW_INVALID(kSumName, ": unsupported data type ", inputs[0].data_type,
                      kSupportedSumTypesMsg);
  }
}

void Sum::Run(RuntimeContext &rt) {
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
