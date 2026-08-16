// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Map — minimal runtime map used by backend test cases and reference kernel
 * implementations for operators that consume or produce ``map(K, V)`` typed
 * graph edges (e.g. ``CastMap``, ``DictVectorizer``, ``ZipMap``).
 *
 * Companion to :cpp:struct:`Sequence`: where ``Sequence`` carries an ordered
 * list of tensors sharing a common element type, ``Map`` carries a key-value
 * mapping represented as two parallel 1-D tensors (one for keys, one for
 * values). This matches the ONNX ``MapProto`` semantics (a list of key/value
 * pairs) but uses the lightweight ``Tensor`` representation already used by
 * the runtime.
 *
 * The struct owns its underlying ``Tensor`` members: copying or destroying
 * the ``Map`` copies or destroys them too.
 */
struct Map {
  /// Optional name of the map (input/output name in the test model).
  /// May be left empty for intermediate values.
  std::string name;

  /// Key data type (e.g. ``DataType::INT64`` or ``DataType::STRING``).
  int32_t key_type = 0;

  /// Value data type (e.g. ``DataType::FLOAT``, ``DataType::STRING``).
  int32_t value_type = 0;

  /// 1-D tensor of keys.
  Tensor keys;

  /// 1-D tensor of values (same length as ``keys``).
  Tensor values;

  Map() = default;
  Map(std::string n, Tensor k, Tensor v)
      : name(std::move(n)), key_type(k.data_type), value_type(v.data_type), keys(std::move(k)),
        values(std::move(v)) {}

  /// Number of entries in the map.
  int64_t size() const noexcept { return keys.element_count(); }

  /// ``true`` when the map contains no entries.
  bool empty() const noexcept { return keys.element_count() == 0; }
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
