// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "onnx_optim/optim_tensor.h"

/**
 * @file shapes_context.h
 * @brief Name → :cpp:class:`OptimTensor` map shared by all
 *        ``onnx_optim`` shape-inference functions.
 *
 * ``ShapesContext`` is the in/out parameter consumed and produced by
 * the per-operator ``ComputeShape*`` functions (for example
 * :cpp:func:`ComputeShapeAbs`). It holds the
 * :cpp:class:`OptimTensor` descriptors of every named value (graph
 * input, initializer or intermediate result) currently known to a
 * shape-inference pass. ``ComputeShape*`` functions read the entries
 * corresponding to a node's inputs and insert new entries for the
 * node's outputs.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Lightweight container mapping value names to
 * :cpp:class:`OptimTensor` descriptors, used by the per-operator
 * ``ComputeShape*`` shape-inference functions.
 *
 * The context is a thin wrapper around an ``unordered_map`` and does
 * not own any tensor data: the :cpp:class:`OptimTensor` values stored
 * here are themselves non-owning views.
 */
class ShapesContext {
public:
  ShapesContext() = default;

  /// Inserts or replaces the descriptor for ``name``.
  void Set(const std::string &name, OptimTensor tensor) { tensors_[name] = std::move(tensor); }

  /// Returns ``true`` when an entry exists for ``name``.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Returns the descriptor for ``name``. Throws ``std::out_of_range``
  /// if no such entry exists.
  const OptimTensor &Get(const std::string &name) const { return tensors_.at(name); }
  OptimTensor &Get(const std::string &name) { return tensors_.at(name); }

  /// Number of named entries currently stored.
  std::size_t Size() const noexcept { return tensors_.size(); }

  /// ``true`` when no entries are stored.
  bool Empty() const noexcept { return tensors_.empty(); }

  /// Removes every entry.
  void Clear() noexcept { tensors_.clear(); }

  /// Read-only access to the underlying map (useful for iteration).
  const std::unordered_map<std::string, OptimTensor> &Tensors() const noexcept { return tensors_; }

private:
  std::unordered_map<std::string, OptimTensor> tensors_;
};

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
