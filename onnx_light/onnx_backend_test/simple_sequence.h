// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/**
 * Sequence — minimal runtime tensor sequence used by backend test
 * cases and reference kernel implementations.
 *
 * Companion to :cpp:struct:`Tensor`: where ``Tensor`` carries a single
 * tensor value, ``Sequence`` carries an ordered list of tensors that
 * share a common element type. ``Sequence`` is intentionally separate
 * from ``TensorProto`` / ``SequenceProto`` and from
 * :cpp:class:`onnx_optim::OptimSequence`: it is the *runtime* value of
 * a sequence-typed graph edge (analogous to ``Tensor`` being the
 * runtime value of a tensor-typed edge), without any protobuf
 * dependency.
 *
 * The struct owns its underlying ``Tensor`` elements: copying or
 * destroying the ``Sequence`` copies or destroys its elements too.
 */
struct Sequence {
  /// Optional name of the sequence (input/output name in the test
  /// model). May be left empty for intermediate values.
  std::string name;

  /// Element data type shared by every tensor stored in ``values``,
  /// expressed as a ``DataType`` integer value. Equals
  /// ``DataType::UNDEFINED`` (``0``) when ``values`` is
  /// empty and the element type cannot be inferred.
  int32_t elem_type = 0;

  /// Tensor elements of the sequence, in order. All elements must share
  /// ``elem_type``; element shapes may differ between elements.
  std::vector<Tensor> values;

  Sequence() = default;
  Sequence(std::string n, int32_t et, std::vector<Tensor> v)
      : name(std::move(n)), elem_type(et), values(std::move(v)) {}

  /// Number of tensors in the sequence.
  std::size_t size() const noexcept { return values.size(); }

  /// ``true`` when the sequence contains no tensors.
  bool empty() const noexcept { return values.empty(); }

  /// Random access to the ``i``-th tensor; throws ``std::out_of_range``
  /// if ``i`` is out of bounds.
  const Tensor &at(std::size_t i) const { return values.at(i); }
  Tensor &at(std::size_t i) { return values.at(i); }
};

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
