// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kPReluName = "kernel::PRelu";

// Branch on the sign of ``x`` rather than evaluating
// ``max(0, x) + slope * min(0, x)``: the latter form turns ``+inf`` /
// ``-inf`` inputs into ``NaN`` because ``slope * 0`` is fine but the
// arithmetic on the masked branch evaluates ``slope * inf`` first. See
// microsoft/onnxruntime#28732 for the regression that motivated this
// reference kernel.
template <typename T> inline T PReluOp(T x, T slope) {
  return x < static_cast<T>(0) ? static_cast<T>(slope * x) : x;
}

template <typename T>
Tensor PReluAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &slope) {
  return detail::BinaryElementwiseAlloc<T, T>(kPReluName, dtype_name, dtype, x, slope,
                                              [](T a, T b) -> T { return PReluOp<T>(a, b); });
}

template <typename T>
void PReluInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &slope,
                  Tensor &output) {
  detail::BinaryElementwise<T, T>(kPReluName, dtype_name, dtype, x, slope, output,
                                  [](T a, T b) -> T { return PReluOp<T>(a, b); });
}

constexpr const char *kSupportedPReluTypesMsg =
    " only supports FLOAT, DOUBLE, INT32, INT64, UINT32 and UINT64 inputs.";
} // namespace

Tensor PRelu::operator()(const Tensor &x, const Tensor &slope) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PReluAlloc<float>("FLOAT", DataType::FLOAT, x, slope);
  case DataType::DOUBLE:
    return PReluAlloc<double>("DOUBLE", DataType::DOUBLE, x, slope);
  case DataType::INT32:
    return PReluAlloc<int32_t>("INT32", DataType::INT32, x, slope);
  case DataType::INT64:
    return PReluAlloc<int64_t>("INT64", DataType::INT64, x, slope);
  case DataType::UINT32:
    return PReluAlloc<uint32_t>("UINT32", DataType::UINT32, x, slope);
  case DataType::UINT64:
    return PReluAlloc<uint64_t>("UINT64", DataType::UINT64, x, slope);
  default:
    throw std::invalid_argument(std::string(kPReluName) + kSupportedPReluTypesMsg);
  }
}

void PRelu::operator()(const Tensor &x, const Tensor &slope, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PReluInPlace<float>("FLOAT", DataType::FLOAT, x, slope, output);
  case DataType::DOUBLE:
    return PReluInPlace<double>("DOUBLE", DataType::DOUBLE, x, slope, output);
  case DataType::INT32:
    return PReluInPlace<int32_t>("INT32", DataType::INT32, x, slope, output);
  case DataType::INT64:
    return PReluInPlace<int64_t>("INT64", DataType::INT64, x, slope, output);
  case DataType::UINT32:
    return PReluInPlace<uint32_t>("UINT32", DataType::UINT32, x, slope, output);
  case DataType::UINT64:
    return PReluInPlace<uint64_t>("UINT64", DataType::UINT64, x, slope, output);
  default:
    throw std::invalid_argument(std::string(kPReluName) + kSupportedPReluTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
