// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_builder.h"
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns::collections
