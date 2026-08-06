// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/symbolic/sym_tensor.h"

/**
 * @file sym_map.h
 * @brief Lightweight description of an ONNX ``map(K,V)`` value used by
 *        ``onnx_shapes`` shape-inference passes.
 *
 * The ``onnx_shapes`` stack already exposes :cpp:class:`SymTensor` and
 * :cpp:class:`SymSequence` to describe tensor- and sequence-typed
 * values flowing through a graph. Operators such as ``ZipMap``,
 * ``CastMap`` or ``DictVectorizer`` (from the ``ai.onnx.ml`` domain)
 * consume or produce *map* values, so a matching descriptor is needed.
 *
 * :cpp:class:`SymMap` is the analogue of :cpp:class:`SymTensor` and
 * :cpp:class:`SymSequence` for map values. An ONNX ``map(K,V)`` value
 * associates scalar keys of a fixed key type (``INT64`` or ``STRING``
 * per the ONNX spec) with values of a fixed value type ``V``. This
 * descriptor records the key :cpp:type:`TensorType`, the value
 * :cpp:type:`TensorType`, and the value's :cpp:class:`SymShape` (for
 * value types that are themselves tensors, e.g. ``map(int64, tensor)``
 * is not part of the ONNX spec today but the shape is kept for
 * forward-compatibility with per-value tensor shapes).
 *
 * Like :cpp:class:`SymTensor` and :cpp:class:`SymSequence` it is a
 * small, value-typed, non-owning descriptor: it never allocates the
 * underlying data.
 */

namespace ONNX_LIGHT_NAMESPACE::core::symbolic {

/**
 * Descriptor for an ONNX ``map(K,V)`` value. A map carries a key
 * :cpp:type:`TensorType` (conceptually ``INT64`` or ``STRING`` per the
 * ONNX spec, though this descriptor does not enforce that constraint)
 * and a value :cpp:type:`TensorType` together with an optional
 * :cpp:class:`SymShape` describing the shape of each value.
 *
 * When the key or value dtype is not known (e.g. for the output of an
 * operator whose map type depends on an attribute that has not been
 * resolved yet), the descriptor stores
 * :cpp:enumerator:`TensorType::kUndefined` for the corresponding
 * field. Callers can use :cpp:func:`HasKeyType` and
 * :cpp:func:`HasValueDtype` to distinguish "unknown" from
 * "explicitly undefined".
 */
class SymMap {
public:
  /// Default constructs a map descriptor with unknown key type, unknown
  /// value dtype, and no recorded value shape.
  SymMap() = default;

  /// Constructs a map descriptor from a known key type, a known value
  /// dtype, and an optional value shape (defaults to a scalar, rank-0
  /// shape).
  SymMap(TensorType key_type, TensorType value_dtype, SymShape value_shape = SymShape())
      : key_type_(key_type), value_dtype_(value_dtype), value_shape_(std::move(value_shape)),
        has_key_type_(key_type != TensorType::kUndefined),
        has_value_dtype_(value_dtype != TensorType::kUndefined) {}

  /// Key type of the map. Returns :cpp:enumerator:`TensorType::kUndefined`
  /// when the key type is unknown (see :cpp:func:`HasKeyType`).
  TensorType KeyType() const noexcept { return key_type_; }

  /// Value dtype of the map. Returns
  /// :cpp:enumerator:`TensorType::kUndefined` when the value dtype is
  /// unknown (see :cpp:func:`HasValueDtype`).
  TensorType ValueDtype() const noexcept { return value_dtype_; }

  /// Shape of each value stored in the map. Defaults to a scalar
  /// (rank-0) shape for the common case where values are single
  /// numbers or strings.
  const SymShape &ValueShape() const noexcept { return value_shape_; }
  SymShape &ValueShape() noexcept { return value_shape_; }

  /// ``true`` when a key type has been recorded for this map.
  bool HasKeyType() const noexcept { return has_key_type_; }

  /// ``true`` when a value dtype has been recorded for this map.
  bool HasValueDtype() const noexcept { return has_value_dtype_; }

  /// Replaces the recorded key type. Passing
  /// :cpp:enumerator:`TensorType::kUndefined` clears the key type.
  void SetKeyType(TensorType key_type) noexcept {
    key_type_ = key_type;
    has_key_type_ = (key_type != TensorType::kUndefined);
  }

  /// Replaces the recorded value dtype. Passing
  /// :cpp:enumerator:`TensorType::kUndefined` clears the value dtype.
  void SetValueDtype(TensorType value_dtype) noexcept {
    value_dtype_ = value_dtype;
    has_value_dtype_ = (value_dtype != TensorType::kUndefined);
  }

  /// Replaces the recorded value shape.
  void SetValueShape(SymShape value_shape) { value_shape_ = std::move(value_shape); }

  /// Equality compares the key type, the value dtype, the value shape,
  /// and the "known" flags.
  bool operator==(const SymMap &other) const noexcept {
    return has_key_type_ == other.has_key_type_ && has_value_dtype_ == other.has_value_dtype_ &&
           key_type_ == other.key_type_ && value_dtype_ == other.value_dtype_ &&
           value_shape_ == other.value_shape_;
  }
  bool operator!=(const SymMap &other) const noexcept { return !(*this == other); }

private:
  TensorType key_type_ = TensorType::kUndefined;
  TensorType value_dtype_ = TensorType::kUndefined;
  SymShape value_shape_{};
  bool has_key_type_ = false;
  bool has_value_dtype_ = false;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::symbolic
