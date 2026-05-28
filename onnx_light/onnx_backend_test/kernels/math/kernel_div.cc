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
constexpr const char *kDivName = "kernel::Div";

// Div uses C/C++ integer division (truncating toward zero) for all integer
// dtypes. For signed types this is the behaviour validated by the upstream
// ``test_div_int32_trunc`` reference (e.g. ``-3 / 2 == -1``), which differs
// from NumPy's default floor division for negative operands. For the
// unsigned dtypes used by the upstream ``test_div_uint{8,16,32,64}`` cases
// truncation and floor division coincide (both operands are non-negative).
template <typename T>
Tensor DivAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<T, T>(kDivName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return a / b; });
}

template <typename T>
void DivInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output) {
  detail::BinaryElementwise<T, T>(kDivName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return a / b; });
}

constexpr const char *kSupportedDivTypesMsg =
    " only supports FLOAT, INT8, INT16, INT32, UINT8, UINT16, UINT32 and UINT64 inputs.";
} // namespace

Tensor Div::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case TensorProto::DataType::FLOAT:
    return DivAlloc<float>("FLOAT", TensorProto::DataType::FLOAT, x, y);
  case TensorProto::DataType::INT8:
    return DivAlloc<int8_t>("INT8", TensorProto::DataType::INT8, x, y);
  case TensorProto::DataType::INT16:
    return DivAlloc<int16_t>("INT16", TensorProto::DataType::INT16, x, y);
  case TensorProto::DataType::INT32:
    return DivAlloc<int32_t>("INT32", TensorProto::DataType::INT32, x, y);
  case TensorProto::DataType::UINT8:
    return DivAlloc<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y);
  case TensorProto::DataType::UINT16:
    return DivAlloc<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y);
  case TensorProto::DataType::UINT32:
    return DivAlloc<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y);
  case TensorProto::DataType::UINT64:
    return DivAlloc<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kDivName) + kSupportedDivTypesMsg);
  }
}

void Div::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case TensorProto::DataType::FLOAT:
    return DivInPlace<float>("FLOAT", TensorProto::DataType::FLOAT, x, y, output);
  case TensorProto::DataType::INT8:
    return DivInPlace<int8_t>("INT8", TensorProto::DataType::INT8, x, y, output);
  case TensorProto::DataType::INT16:
    return DivInPlace<int16_t>("INT16", TensorProto::DataType::INT16, x, y, output);
  case TensorProto::DataType::INT32:
    return DivInPlace<int32_t>("INT32", TensorProto::DataType::INT32, x, y, output);
  case TensorProto::DataType::UINT8:
    return DivInPlace<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y, output);
  case TensorProto::DataType::UINT16:
    return DivInPlace<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y, output);
  case TensorProto::DataType::UINT32:
    return DivInPlace<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y, output);
  case TensorProto::DataType::UINT64:
    return DivInPlace<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kDivName) + kSupportedDivTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
