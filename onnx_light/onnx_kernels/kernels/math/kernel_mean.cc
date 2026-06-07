// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/elementwise_helpers.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kMeanName = "kernel::Mean";

constexpr const char *kSupportedMeanTypesMsg = " only supports FLOAT and DOUBLE inputs.";

// Returns the multidirectional-broadcast output shape of ``a`` and ``b``.
std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  std::vector<int64_t> sa(rank, 1), sb(rank, 1), out(rank, 1);
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
      throw std::invalid_argument(std::string(kMeanName) +
                                  " input shapes are not multidirectional-broadcastable.");
    }
  }
  return out;
}

// Computes the broadcast shape of every tensor in ``inputs``. ``inputs`` must
// be non-empty and all tensors must share ``expected_dtype``.
std::vector<int64_t> ValidateAndBroadcastShape(const std::vector<Tensor> &inputs,
                                               const char *dtype_name, int32_t expected_dtype) {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMeanName) + " requires at least one input.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == expected_dtype,
                        std::string(kMeanName) + " only supports " + dtype_name + " tensors.");
  }
  std::vector<int64_t> shape = inputs[0].shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    shape = BroadcastShape(shape, inputs[i].shape);
  }
  return shape;
}

template <typename T>
void AccumulateAndScale(const char *dtype_name, int32_t dtype, const std::vector<Tensor> &inputs,
                        Tensor &output) {
  // Single input: copy verbatim. ``Mean`` of a single tensor is the tensor itself.
  if (inputs.size() == 1) {
    std::memcpy(output.data.data(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return;
  }
  // First pair: accumulate into the output buffer.
  detail::BinaryElementwise<T, T>(kMeanName, dtype_name, dtype, inputs[0], inputs[1], output,
                                  [](T a, T b) -> T { return a + b; });
  // Subsequent inputs: accumulate in place by re-running the binary
  // element-wise driver with ``output`` as both an input and the output.
  for (size_t i = 2; i < inputs.size(); ++i) {
    Tensor partial = output;
    detail::BinaryElementwise<T, T>(kMeanName, dtype_name, dtype, partial, inputs[i], output,
                                    [](T a, T b) -> T { return a + b; });
  }
  // Divide the accumulated sum by the input count to obtain the mean.
  const T inv_n = static_cast<T>(1) / static_cast<T>(inputs.size());
  T *out_ptr = reinterpret_cast<T *>(output.data.data());
  const int64_t n_elements = output.element_count();
  for (int64_t i = 0; i < n_elements; ++i) {
    out_ptr[i] *= inv_n;
  }
}

template <typename T>
Tensor MeanAlloc(const char *dtype_name, int32_t dtype, const std::vector<Tensor> &inputs) {
  const std::vector<int64_t> out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  Tensor z("", dtype, out_shape, std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(T)));
  AccumulateAndScale<T>(dtype_name, dtype, inputs, z);
  return z;
}

template <typename T>
void MeanInPlace(const char *dtype_name, int32_t dtype, const std::vector<Tensor> &inputs,
                 Tensor &output) {
  const std::vector<int64_t> out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
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

Tensor Mean::operator()(const std::vector<Tensor> &inputs) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMeanName) + " requires at least one input.");
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return MeanAlloc<float>("FLOAT", DataType::FLOAT, inputs);
  case DataType::DOUBLE:
    return MeanAlloc<double>("DOUBLE", DataType::DOUBLE, inputs);
  default:
    throw std::invalid_argument(std::string(kMeanName) + kSupportedMeanTypesMsg);
  }
}

void Mean::operator()(const std::vector<Tensor> &inputs, Tensor &output) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMeanName) + " requires at least one input.");
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return MeanInPlace<float>("FLOAT", DataType::FLOAT, inputs, output);
  case DataType::DOUBLE:
    return MeanInPlace<double>("DOUBLE", DataType::DOUBLE, inputs, output);
  default:
    throw std::invalid_argument(std::string(kMeanName) + kSupportedMeanTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
