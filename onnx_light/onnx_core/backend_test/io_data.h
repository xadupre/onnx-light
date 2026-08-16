// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_map.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

using namespace ::onnx_light::core::runtime;

/// Materialized inputs/outputs produced by a lazy case builder.
struct IoData {
  Tensors inputs;
  Tensors outputs;
  /// Map-typed inputs (e.g. for CastMap, DictVectorizer). Each ``Map::name``
  /// must match a non-empty entry in the node's ``input`` list. When present,
  /// ``BuildSingleNodeCase`` declares the corresponding graph input with a
  /// ``map(key_type, value_type)`` TypeProto and stores the Map in the
  /// ``DataSet::maps`` collection so the runtime can retrieve it by name.
  std::vector<Map> maps = {};
};

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
