// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_light_helpers.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kEqualName = "kernel::Equal";
constexpr const char *kBoolName = "BOOL";

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

// String equality path. ``detail::BinaryElementwise*`` operate on POD data
// stored in ``Tensor::data`` and cannot handle STRING tensors, which live in
// ``Tensor::string_data``. We only support equal-shape inputs or scalar
// broadcasting here, which is enough for the upstream ``test_equal_string``
// and ``test_equal_string_broadcast`` cases.
struct StringEqualBroadcast {
  std::vector<int64_t> shape;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

StringEqualBroadcast CheckStringEqualInputs(const Tensor &x, const Tensor &y) {
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  EXT_ENFORCE_INVALID(
      nx == ny || nx == 1 || ny == 1,
      "kernel::Equal STRING inputs only support equal-shape tensors or scalar broadcasting.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == nx,
                      "kernel::Equal input ``x`` string_data size does not match its shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(y.string_data.size()) == ny,
                      "kernel::Equal input ``y`` string_data size does not match its shape.");
  StringEqualBroadcast bi;
  bi.nx = nx;
  bi.ny = ny;
  bi.element_count = nx >= ny ? nx : ny;
  bi.shape = nx >= ny ? x.shape : y.shape;
  return bi;
}

Tensor EqualStringAlloc(const Tensor &x, const Tensor &y) {
  const StringEqualBroadcast bi = CheckStringEqualInputs(x, y);
  Tensor out("", TensorProto::DataType::BOOL, bi.shape,
             std::vector<uint8_t>(static_cast<size_t>(bi.element_count)));
  for (int64_t i = 0; i < bi.element_count; ++i) {
    const std::string &a = x.string_data[bi.nx == 1 ? 0 : static_cast<size_t>(i)];
    const std::string &b = y.string_data[bi.ny == 1 ? 0 : static_cast<size_t>(i)];
    out.data[static_cast<size_t>(i)] = a == b ? 1 : 0;
  }
  return out;
}

void EqualStringInPlace(const Tensor &x, const Tensor &y, Tensor &output) {
  const StringEqualBroadcast bi = CheckStringEqualInputs(x, y);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::BOOL),
                      "kernel::Equal preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(
      output.shape == bi.shape,
      "kernel::Equal preallocated output shape must match the broadcasted input shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) == bi.element_count,
                      "kernel::Equal preallocated output ``data`` has unexpected size.");
  for (int64_t i = 0; i < bi.element_count; ++i) {
    const std::string &a = x.string_data[bi.nx == 1 ? 0 : static_cast<size_t>(i)];
    const std::string &b = y.string_data[bi.ny == 1 ? 0 : static_cast<size_t>(i)];
    output.data[static_cast<size_t>(i)] = a == b ? 1 : 0;
  }
}
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
  case TensorProto::DataType::STRING:
    EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                        "kernel::Equal inputs must share the same dtype.");
    return EqualStringAlloc(x, y);
  default:
    throw std::invalid_argument(std::string(kEqualName) +
                                " only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
                                "INT64, UINT8, UINT16, UINT32, UINT64 and STRING inputs.");
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
  case TensorProto::DataType::STRING:
    EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                        "kernel::Equal inputs must share the same dtype.");
    return EqualStringInPlace(x, y, output);
  default:
    throw std::invalid_argument(std::string(kEqualName) +
                                " only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
                                "INT64, UINT8, UINT16, UINT32, UINT64 and STRING inputs.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
