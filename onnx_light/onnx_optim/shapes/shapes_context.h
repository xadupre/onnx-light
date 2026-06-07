// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/simple_string.h"

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

/// Sentinel value returned by :cpp:func:`ShapesContext::OpsetVersion`
/// when no opset version has been recorded for the requested domain.
inline constexpr int kUnknownOpsetVersion = -1;

/// Canonical domain string used for the standard ONNX operator set
/// (``ai.onnx``). An empty domain on a ``NodeProto`` is treated as
/// equivalent to this value.
inline constexpr const char *kOnnxDomain = "ai.onnx";

/**
 * Lightweight container shared by the per-operator ``ComputeShape*``
 * shape-inference functions. ``ShapesContext`` carries two pieces of
 * information:
 *
 *   - a ``name → OptimTensor`` map describing every named value
 *     (graph input, initializer or intermediate result) currently
 *     known to the shape-inference pass;
 *   - a ``name → OptimSequence`` map describing every named
 *     sequence-typed value (the output of ``SequenceConstruct``,
 *     ``SequenceEmpty``, ``SplitToSequence``, ...);
 *   - a ``domain → opset_version`` map mirroring the ``opset_import``
 *     entries of the surrounding ``ModelProto``, so that
 *     ``ComputeShape*`` functions can pick the correct schema
 *     revision when shape inference depends on the operator's opset
 *     version.
 *
 * The context is a thin wrapper and does not own any tensor data: the
 * :cpp:class:`OptimTensor` values stored here are themselves
 * non-owning views.
 */
class ShapesContext {
public:
  ShapesContext() = default;

  // ── Tensor descriptors ──────────────────────────────────────────────

  /// Inserts or replaces the descriptor for ``name``. ``tensor`` is
  /// consumed; callers must pass an rvalue (use ``std::move``).
  void Set(const std::string &name, OptimTensor &&tensor) { tensors_[name] = std::move(tensor); }

  /// Overload: ``name`` given as a null-terminated C string.
  void Set(const char *name, OptimTensor &&tensor) {
    tensors_[std::string(name)] = std::move(tensor);
  }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  void Set(const utils::String &name, OptimTensor &&tensor) {
    tensors_[std::string(name.data(), name.size())] = std::move(tensor);
  }

  /// Returns ``true`` when an entry exists for ``name``.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Returns the descriptor for ``name``. Throws ``std::out_of_range``
  /// if no such entry exists.
  const OptimTensor &Get(const std::string &name) const { return tensors_.at(name); }

  /// Number of named entries currently stored.
  std::size_t Size() const noexcept { return tensors_.size(); }

  /// ``true`` when no entries are stored.
  bool Empty() const noexcept { return tensors_.empty(); }

  /// Removes every entry (both tensor descriptors and opset versions).
  void Clear() noexcept {
    tensors_.clear();
    sequences_.clear();
    opsets_.clear();
    local_functions_.clear();
  }

  /// Read-only access to the underlying map (useful for iteration).
  const std::unordered_map<std::string, OptimTensor> &Tensors() const noexcept { return tensors_; }

  // ── Sequence descriptors ────────────────────────────────────────────

  /// Inserts or replaces the descriptor for a sequence-typed value
  /// named ``name``. ``sequence`` is consumed; callers must pass an
  /// rvalue (use ``std::move``).
  void SetSequence(const std::string &name, OptimSequence &&sequence) {
    sequences_[name] = std::move(sequence);
  }

  /// Overload: ``name`` given as a null-terminated C string.
  void SetSequence(const char *name, OptimSequence &&sequence) {
    sequences_[std::string(name)] = std::move(sequence);
  }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  void SetSequence(const utils::String &name, OptimSequence &&sequence) {
    sequences_[std::string(name.data(), name.size())] = std::move(sequence);
  }

  /// Returns ``true`` when a sequence-typed entry exists for ``name``.
  bool HasSequence(const std::string &name) const {
    return sequences_.find(name) != sequences_.end();
  }

  /// Returns the sequence descriptor for ``name``. Throws
  /// ``std::out_of_range`` if no such entry exists.
  const OptimSequence &GetSequence(const std::string &name) const { return sequences_.at(name); }

  /// Number of sequence-typed entries currently stored.
  std::size_t SequencesSize() const noexcept { return sequences_.size(); }

  /// Read-only access to the underlying sequence map (useful for iteration).
  const std::unordered_map<std::string, OptimSequence> &Sequences() const noexcept {
    return sequences_;
  }

  // ── Opset versions ──────────────────────────────────────────────────

  /**
   * Records the opset version of ``domain``. An empty ``domain`` is
   * normalised to :cpp:var:`kOnnxDomain`. Replaces any previous entry
   * for the same domain.
   */
  void SetOpsetVersion(const std::string &domain, int opset_version) {
    opsets_[NormaliseDomain(domain)] = opset_version;
  }

  /// ``true`` when an opset version has been recorded for ``domain``
  /// (after normalising the empty domain to :cpp:var:`kOnnxDomain`).
  bool HasOpsetVersion(const std::string &domain) const {
    return opsets_.find(NormaliseDomain(domain)) != opsets_.end();
  }

  /**
   * Returns the recorded opset version of ``domain``, or
   * :cpp:var:`kUnknownOpsetVersion` if none was recorded. An empty
   * ``domain`` is normalised to :cpp:var:`kOnnxDomain`.
   */
  int OpsetVersion(const std::string &domain) const {
    auto it = opsets_.find(NormaliseDomain(domain));
    return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
  }

  /// Read-only access to the underlying ``domain → opset_version`` map.
  const std::unordered_map<std::string, int> &Opsets() const noexcept { return opsets_; }

  // ── Model-local functions ──────────────────────────────────────────
  //
  // ``onnx_optim`` shape inference dispatches node-level inference via
  // the registered op schemas (see ``DispatchTable``). A node whose
  // ``op_type`` is not registered may still be valid if it calls a
  // model-local :cpp:class:`FunctionProto` declared in
  // ``ModelProto::functions``. Local functions are addressed by a
  // ``"<domain>:<name>"`` key (matching :cpp:func:`GetFunctionIdentifier`
  // in ``onnx_lib``); :cpp:func:`ComputeShapeModel` registers every
  // entry of ``model.functions()`` here so that
  // :cpp:func:`ComputeShapeNode` can detect and expand the call.

  /// Registers a non-owning pointer to a model-local
  /// :cpp:class:`FunctionProto`. The pointer must remain valid for the
  /// lifetime of the shape-inference pass. Replaces any previous entry
  /// registered under the same ``"<domain>:<name>"`` key. ``func`` must
  /// not be ``nullptr``.
  void SetLocalFunction(const FunctionProto *func) {
    EXT_ENFORCE_INVALID(func != nullptr, "SetLocalFunction: func must not be nullptr.");
    const std::string key = func->domain().as_string() + ":" + func->name().as_string();
    local_functions_[key] = func;
  }

  /// ``true`` when a model-local function is registered for ``key``.
  /// ``key`` is expected to be the ``"<domain>:<name>"`` identifier of
  /// the function.
  bool HasLocalFunction(const std::string &key) const {
    return local_functions_.find(key) != local_functions_.end();
  }

  /// Returns the registered ``FunctionProto`` pointer for ``key``, or
  /// ``nullptr`` when none is registered. ``key`` is expected to be the
  /// ``"<domain>:<name>"`` identifier of the function.
  const FunctionProto *GetLocalFunction(const std::string &key) const {
    auto it = local_functions_.find(key);
    return it == local_functions_.end() ? nullptr : it->second;
  }

  /// Read-only access to the underlying ``"<domain>:<name>" →
  /// FunctionProto*`` map.
  const std::unordered_map<std::string, const FunctionProto *> &LocalFunctions() const noexcept {
    return local_functions_;
  }

private:
  static std::string NormaliseDomain(const std::string &domain) {
    return domain.empty() ? std::string(kOnnxDomain) : domain;
  }

  std::unordered_map<std::string, OptimTensor> tensors_;
  std::unordered_map<std::string, OptimSequence> sequences_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_map<std::string, const FunctionProto *> local_functions_;
};

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
