// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

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
 * for sequence values. It records the common element dtype and one
 * :cpp:class:`OptimShape` per element of the sequence (the ONNX
 * ``SequenceConstruct`` operator only requires that elements share the
 * same dtype, not the same shape). The sequence length is itself an
 * :cpp:class:`OptimDim` so it can be either a concrete integer (e.g. for
 * a ``SequenceConstruct`` node with ``N`` known inputs) or a symbolic
 * expression (e.g. for the output of ``SplitToSequence`` whose length
 * depends on a runtime ``split`` value). When the per-element shapes
 * are known, the length is implicitly :cpp:func:`ElemShapes().size()`.
 *
 * Like :cpp:class:`OptimTensor` it is a small, value-typed, non-owning
 * descriptor: it never allocates the underlying data.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

/**
 * Descriptor for an ONNX tensor-sequence value. A sequence carries a
 * common element :cpp:type:`TensorType` and one :cpp:class:`OptimShape`
 * per element. The sequence length is itself an :cpp:class:`OptimDim`
 * so it can be either a concrete integer (e.g. for a
 * ``SequenceConstruct`` node with ``N`` known inputs) or a symbolic
 * expression (e.g. for the output of ``SplitToSequence`` whose length
 * depends on a runtime ``split`` value).
 *
 * When the element dtype is not known (e.g. for the output of
 * ``SequenceEmpty`` whose dtype is set by an attribute or defaults to
 * ``FLOAT``, or for ``SequenceConstruct`` with zero inputs), the
 * descriptor stores :cpp:enumerator:`TensorType::kUndefined` for the
 * dtype. When the per-element shapes are unknown (typically for
 * sequences whose length is itself symbolic), the descriptor stores no
 * shapes at all and :cpp:func:`HasElemShapes` returns ``false``.
 * Callers can use :cpp:func:`HasElemDtype` and :cpp:func:`HasElemShapes`
 * to distinguish "unknown" from "empty sequence" (length 0).
 */
class OptimSequence {
public:
  /// Default constructs an empty sequence descriptor with unknown
  /// element dtype, no per-element shapes recorded, and a zero length.
  OptimSequence() = default;

  /// Constructs a sequence descriptor from a known element dtype and a
  /// vector of per-element shapes. The sequence length is set to the
  /// number of supplied shapes.
  OptimSequence(TensorType elem_dtype, std::vector<OptimShape> elem_shapes)
      : elem_dtype_(elem_dtype), elem_shapes_(std::move(elem_shapes)),
        length_(static_cast<int64_t>(elem_shapes_.size())),
        has_elem_dtype_(elem_dtype != TensorType::kUndefined), has_elem_shapes_(true) {}

  /// Constructs a sequence descriptor from a known element dtype and a
  /// (possibly symbolic) length. No per-element shape is recorded.
  OptimSequence(TensorType elem_dtype, OptimDim length)
      : elem_dtype_(elem_dtype), length_(std::move(length)),
        has_elem_dtype_(elem_dtype != TensorType::kUndefined), has_elem_shapes_(false) {}

  /// Element dtype shared by every tensor in the sequence. Returns
  /// :cpp:enumerator:`TensorType::kUndefined` when the dtype is unknown
  /// (see :cpp:func:`HasElemDtype`).
  TensorType ElemDtype() const noexcept { return elem_dtype_; }

  /// Per-element shapes of the sequence. The returned vector is empty
  /// when :cpp:func:`HasElemShapes` is ``false``; otherwise its size is
  /// the (concrete) sequence length and ``ElemShapes()[i]`` is the
  /// shape of the ``i``-th tensor in the sequence.
  const std::vector<OptimShape> &ElemShapes() const noexcept { return elem_shapes_; }
  std::vector<OptimShape> &ElemShapes() noexcept { return elem_shapes_; }

  /// Sequence length, possibly symbolic. When
  /// :cpp:func:`HasElemShapes` is ``true``, the length equals
  /// ``ElemShapes().size()``.
  const OptimDim &Length() const noexcept { return length_; }
  OptimDim &Length() noexcept { return length_; }

  /// ``true`` when an element dtype has been recorded for this sequence.
  bool HasElemDtype() const noexcept { return has_elem_dtype_; }

  /// ``true`` when per-element shapes have been recorded for this
  /// sequence. An empty vector still returns ``true`` and denotes a
  /// sequence of length 0.
  bool HasElemShapes() const noexcept { return has_elem_shapes_; }

  /// Replaces the recorded element dtype. Passing
  /// :cpp:enumerator:`TensorType::kUndefined` clears the dtype.
  void SetElemDtype(TensorType dtype) noexcept {
    elem_dtype_ = dtype;
    has_elem_dtype_ = (dtype != TensorType::kUndefined);
  }

  /// Replaces the recorded per-element shapes. Also synchronises the
  /// sequence length with the number of supplied shapes.
  void SetElemShapes(std::vector<OptimShape> shapes) {
    elem_shapes_ = std::move(shapes);
    has_elem_shapes_ = true;
    length_ = OptimDim(static_cast<int64_t>(elem_shapes_.size()));
  }

  /// Clears the recorded per-element shapes. The sequence length is
  /// left untouched.
  void ClearElemShapes() noexcept {
    elem_shapes_.clear();
    has_elem_shapes_ = false;
  }

  /// Replaces the sequence length. Callers are responsible for keeping
  /// the length consistent with :cpp:func:`ElemShapes` when both are
  /// recorded.
  void SetLength(OptimDim length) noexcept { length_ = std::move(length); }

  /// Equality compares the element dtype, the per-element shapes, the
  /// sequence length, and the "known" flags.
  bool operator==(const OptimSequence &other) const noexcept {
    return has_elem_dtype_ == other.has_elem_dtype_ && has_elem_shapes_ == other.has_elem_shapes_ &&
           elem_dtype_ == other.elem_dtype_ && elem_shapes_ == other.elem_shapes_ &&
           length_ == other.length_;
  }
  bool operator!=(const OptimSequence &other) const noexcept { return !(*this == other); }

private:
  TensorType elem_dtype_ = TensorType::kUndefined;
  std::vector<OptimShape> elem_shapes_{};
  OptimDim length_{static_cast<int64_t>(0)};
  bool has_elem_dtype_ = false;
  bool has_elem_shapes_ = false;
};

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
