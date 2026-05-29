// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kEqualName = "kernel::Equal";
constexpr const char *kBoolName = "BOOL";

// Helper used to label the input dtype branch in error messages and to
// route dispatch through a single templated entry point. The label is the
// upstream ONNX :class:`TensorProto::DataType` enumerator name so the
// resulting "kernel::Equal only supports <DTYPE> inputs." message is
// self-explanatory.
template <typename TIn>
Tensor EqualAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kEqualName, in_dtype_name, in_dtype, kBoolName, TensorProto::DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a == b ? 1 : 0; });
}

template <typename TIn>
void EqualInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                  Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kEqualName, in_dtype_name, in_dtype, kBoolName, TensorProto::DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a == b ? 1 : 0; });
}

constexpr const char *kUnsupportedDtypeMessage =
    " only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";
} // namespace

Tensor Equal::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case TensorProto::DataType::BOOL:
    return EqualAlloc<uint8_t>("BOOL", TensorProto::DataType::BOOL, x, y);
  case TensorProto::DataType::FLOAT:
    return EqualAlloc<float>("FLOAT", TensorProto::DataType::FLOAT, x, y);
  case TensorProto::DataType::DOUBLE:
    return EqualAlloc<double>("DOUBLE", TensorProto::DataType::DOUBLE, x, y);
  case TensorProto::DataType::INT8:
    return EqualAlloc<int8_t>("INT8", TensorProto::DataType::INT8, x, y);
  case TensorProto::DataType::INT16:
    return EqualAlloc<int16_t>("INT16", TensorProto::DataType::INT16, x, y);
  case TensorProto::DataType::INT32:
    return EqualAlloc<int32_t>("INT32", TensorProto::DataType::INT32, x, y);
  case TensorProto::DataType::INT64:
    return EqualAlloc<int64_t>("INT64", TensorProto::DataType::INT64, x, y);
  case TensorProto::DataType::UINT8:
    return EqualAlloc<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y);
  case TensorProto::DataType::UINT16:
    return EqualAlloc<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y);
  case TensorProto::DataType::UINT32:
    return EqualAlloc<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y);
  case TensorProto::DataType::UINT64:
    return EqualAlloc<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kEqualName) + kUnsupportedDtypeMessage);
  }
}

void Equal::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case TensorProto::DataType::BOOL:
    return EqualInPlace<uint8_t>("BOOL", TensorProto::DataType::BOOL, x, y, output);
  case TensorProto::DataType::FLOAT:
    return EqualInPlace<float>("FLOAT", TensorProto::DataType::FLOAT, x, y, output);
  case TensorProto::DataType::DOUBLE:
    return EqualInPlace<double>("DOUBLE", TensorProto::DataType::DOUBLE, x, y, output);
  case TensorProto::DataType::INT8:
    return EqualInPlace<int8_t>("INT8", TensorProto::DataType::INT8, x, y, output);
  case TensorProto::DataType::INT16:
    return EqualInPlace<int16_t>("INT16", TensorProto::DataType::INT16, x, y, output);
  case TensorProto::DataType::INT32:
    return EqualInPlace<int32_t>("INT32", TensorProto::DataType::INT32, x, y, output);
  case TensorProto::DataType::INT64:
    return EqualInPlace<int64_t>("INT64", TensorProto::DataType::INT64, x, y, output);
  case TensorProto::DataType::UINT8:
    return EqualInPlace<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y, output);
  case TensorProto::DataType::UINT16:
    return EqualInPlace<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y, output);
  case TensorProto::DataType::UINT32:
    return EqualInPlace<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y, output);
  case TensorProto::DataType::UINT64:
    return EqualInPlace<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kEqualName) + kUnsupportedDtypeMessage);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
