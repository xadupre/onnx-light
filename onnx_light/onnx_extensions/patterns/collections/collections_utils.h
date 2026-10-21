// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::collections {

/// Returns ``true`` when ``node`` is ``op_type`` from the default ONNX domain.
inline bool IsDefaultOp(const NodeProto &node, const char *op_type) {
  return node.op_type().value() == op_type &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

/// Returns a fresh initializer name derived from ``base`` that is not already
/// used by ``builder`` (without reserving it, so :ref:`MakeInitializer` can).
inline std::string FreeInitializerName(core::builder::GraphBuilder &builder,
                                       const std::string &base) {
  if (!builder.HasName(base)) {
    return base;
  }
  std::string candidate;
  for (int suffix = 0;; ++suffix) {
    candidate = base + "_" + std::to_string(suffix);
    if (!builder.HasName(candidate)) {
      return candidate;
    }
  }
}

/// Returns the ``axis`` integer attribute of ``node`` or ``default_value``.
inline int64_t GetAxis(const NodeProto &node, int64_t default_value = 0) {
  return GetAttributeOr<int64_t>(node, "axis", default_value);
}

/// Reads the leading integer element of the scalar constant ``tensor`` into
/// ``out``. Returns ``false`` when the tensor holds no integer payload.
inline bool ReadScalarInt(const TensorProto &tensor, int64_t &out) {
  std::vector<int64_t> values;
  if (!ReadIntegerValues(tensor, values) || values.empty()) {
    return false;
  }
  out = values.front();
  return true;
}

/// Creates a one-element tensor with an arithmetic value stored in little-endian raw_data.
/// FLOAT16 and BFLOAT16 values are rounded from float; integer values retain their precision.
template <typename T>
inline TensorProto MakeScalarTensor(const std::string &name, TensorProto::DataType type,
                                    const std::vector<int64_t> &dims, T value) {
  for (int64_t dim : dims) {
    if (dim != 1) {
      throw core::builder::BuilderError("MakeScalarTensor requires a one-element shape.");
    }
  }
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (int64_t dim : dims) {
    tensor.add_dims(dim);
  }
  const auto store = [&tensor](auto converted) {
    uint8_t bytes[sizeof(converted)];
    std::memcpy(bytes, &converted, sizeof(converted));
    const uint16_t one = 1;
    if (*reinterpret_cast<const uint8_t *>(&one) != 1) {
      std::reverse(bytes, bytes + sizeof(converted));
    }
    tensor.set_raw_data(bytes, sizeof(converted));
  };
  switch (type) {
  case TensorProto::FLOAT16:
    store(core::runtime::FloatToFloat16Bits(static_cast<float>(value)));
    break;
  case TensorProto::BFLOAT16:
    store(core::runtime::FloatToBfloat16Bits(static_cast<float>(value)));
    break;
  case TensorProto::FLOAT:
    store(static_cast<float>(value));
    break;
  case TensorProto::DOUBLE:
    store(static_cast<double>(value));
    break;
  case TensorProto::INT64:
    store(static_cast<int64_t>(value));
    break;
  case TensorProto::UINT64:
    store(static_cast<uint64_t>(value));
    break;
  case TensorProto::INT32:
    store(static_cast<int32_t>(value));
    break;
  case TensorProto::UINT32:
    store(static_cast<uint32_t>(value));
    break;
  case TensorProto::INT16:
    store(static_cast<int16_t>(value));
    break;
  case TensorProto::UINT16:
    store(static_cast<uint16_t>(value));
    break;
  case TensorProto::INT8:
    store(static_cast<int8_t>(value));
    break;
  case TensorProto::UINT8:
    store(static_cast<uint8_t>(value));
    break;
  case TensorProto::BOOL:
    store(static_cast<uint8_t>(value != 0));
    break;
  default:
    throw core::builder::BuilderError("MakeScalarTensor received an unsupported data type.");
  }
  return tensor;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::collections
