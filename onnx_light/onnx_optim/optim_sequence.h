// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "onnx_optim/optim_tensor.h"

/**
 * @file optim_sequence.h
 * @brief Lightweight description of an ONNX tensor sequence value used
 *        by ``onnx_optim`` shape-inference passes.
 *
 * The ``onnx_optim`` stack already exposes :cpp:class:`OptimTensor` to
 * describe tensor-typed values flowing through a graph. Operators such
 * as ``SequenceConstruct``, ``SequenceInsert``, ``SplitToSequence`` or
 * ``SequenceEmpty`` produce *sequence* values rather than tensors, so a
 * matching descriptor is needed.
 *
 * :cpp:class:`OptimSequence` is the analogue of :cpp:class:`OptimTensor`
 * for sequence values. It records the common element dtype and the
 * common element shape across the sequence, plus the sequence length as
 * an :cpp:class:`OptimDim` (which may be either a concrete integer or a
 * symbolic expression). Like :cpp:class:`OptimTensor` it is a small,
 * value-typed, non-owning descriptor: it never allocates the underlying
 * data.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

/**
 * Descriptor for an ONNX tensor-sequence value. A sequence carries a
 * common element :cpp:type:`TensorType` and a common element
 * :cpp:class:`OptimShape`. The sequence length is itself an
 * :cpp:class:`OptimDim` so it can be either a concrete integer (e.g. for
 * a ``SequenceConstruct`` node with ``N`` known inputs) or a symbolic
 * expression (e.g. for the output of ``SplitToSequence`` whose length
 * depends on a runtime ``split`` value).
 *
 * When the element dtype or the element shape is not known (e.g. for
 * the output of ``SequenceEmpty`` whose dtype is set by an attribute or
 * defaults to ``FLOAT``, or for ``SequenceConstruct`` with zero
 * inputs), the descriptor stores :cpp:enumerator:`TensorType::kUndefined`
 * for the dtype and an empty :cpp:class:`OptimShape` for the element
 * shape. Callers can use :cpp:func:`HasElemDtype` and
 * :cpp:func:`HasElemShape` to distinguish "unknown" from "rank-0
 * scalar" element shapes.
 */
class OptimSequence {
public:
  /// Default constructs an empty sequence descriptor with unknown
  /// element dtype, unknown element shape, and a zero length.
  OptimSequence() = default;

  /// Constructs a sequence descriptor from a known element dtype, a
  /// known element shape and a length.
  OptimSequence(TensorType elem_dtype, OptimShape elem_shape, OptimDim length)
      : elem_dtype_(elem_dtype), elem_shape_(std::move(elem_shape)), length_(std::move(length)),
        has_elem_dtype_(elem_dtype != TensorType::kUndefined), has_elem_shape_(true) {}

  /// Element dtype shared by every tensor in the sequence. Returns
  /// :cpp:enumerator:`TensorType::kUndefined` when the dtype is unknown
  /// (see :cpp:func:`HasElemDtype`).
  TensorType ElemDtype() const noexcept { return elem_dtype_; }

  /// Element shape shared by every tensor in the sequence. The returned
  /// shape is meaningless when :cpp:func:`HasElemShape` is ``false``.
  const OptimShape &ElemShape() const noexcept { return elem_shape_; }
  OptimShape &ElemShape() noexcept { return elem_shape_; }

  /// Sequence length, possibly symbolic.
  const OptimDim &Length() const noexcept { return length_; }
  OptimDim &Length() noexcept { return length_; }

  /// ``true`` when an element dtype has been recorded for this sequence.
  bool HasElemDtype() const noexcept { return has_elem_dtype_; }

  /// ``true`` when an element shape has been recorded for this sequence.
  /// An empty stored shape still returns ``true`` and denotes a sequence
  /// of rank-0 (scalar) tensors.
  bool HasElemShape() const noexcept { return has_elem_shape_; }

  /// Replaces the recorded element dtype. Passing
  /// :cpp:enumerator:`TensorType::kUndefined` clears the dtype.
  void SetElemDtype(TensorType dtype) noexcept {
    elem_dtype_ = dtype;
    has_elem_dtype_ = (dtype != TensorType::kUndefined);
  }

  /// Replaces the recorded element shape.
  void SetElemShape(OptimShape shape) {
    elem_shape_ = std::move(shape);
    has_elem_shape_ = true;
  }

  /// Clears the recorded element shape.
  void ClearElemShape() noexcept {
    elem_shape_ = OptimShape{};
    has_elem_shape_ = false;
  }

  /// Replaces the sequence length.
  void SetLength(OptimDim length) noexcept { length_ = std::move(length); }

  /// Equality compares the element dtype, the element shape, the
  /// sequence length, and the "known" flags.
  bool operator==(const OptimSequence &other) const noexcept {
    return has_elem_dtype_ == other.has_elem_dtype_ && has_elem_shape_ == other.has_elem_shape_ &&
           elem_dtype_ == other.elem_dtype_ && elem_shape_ == other.elem_shape_ &&
           length_ == other.length_;
  }
  bool operator!=(const OptimSequence &other) const noexcept { return !(*this == other); }

private:
  TensorType elem_dtype_ = TensorType::kUndefined;
  OptimShape elem_shape_{};
  OptimDim length_{static_cast<int64_t>(0)};
  bool has_elem_dtype_ = false;
  bool has_elem_shape_ = false;
};

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
