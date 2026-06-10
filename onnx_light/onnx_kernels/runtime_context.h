// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @file runtime_context.h
 * @brief Per-invocation runtime state shared across the nodes of a
 *        graph evaluated through :cpp:func:`RunNode` /
 *        :cpp:func:`RunNodes`.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

/**
 * Name-keyed map of tensors carrying both the graph inputs/initializers
 * and the intermediate values produced by previously executed nodes.
 * Owned by :cpp:class:`RuntimeContext`; the dispatcher reads a node's
 * inputs from this map by name (matching ``node.input(i)``) and inserts
 * every produced output under the name declared by ``node.output(i)``.
 */
using TensorMap = std::unordered_map<std::string, Tensor>;

/**
 * Name-keyed map of model-local :cpp:type:`FunctionProto` definitions
 * known to the runtime. Populated by :cpp:func:`RunModel` from
 * ``ModelProto::functions()`` so the dispatcher in :cpp:func:`RunNode`
 * can transparently invoke :cpp:func:`RunFunction` whenever a node
 * references a model-local function instead of a built-in kernel.
 *
 * Keys are the canonical ``"<domain>:<op_type>:<overload>"`` triple
 * (the default ONNX domain — the empty ``NodeProto::domain()`` — is
 * normalised to ``"ai.onnx"`` and the overload defaults to the empty
 * string). Values are non-owning pointers into the caller-owned
 * ``ModelProto``; the entries are valid only as long as the model
 * outlives the runtime context.
 */
using FunctionMap = std::unordered_map<std::string, const FunctionProto *>;

/**
 * Maximum element_count for which :cpp:class:`TensorEvent` captures the
 * tensor values inline. Tensors with more elements still emit an event,
 * but ``TensorEvent::values`` / ``TensorEvent::string_values`` are left
 * empty so the event log stays bounded for large activations.
 */
inline constexpr int64_t kTensorEventValueLimit = 8;

/**
 * Kind of tensor map mutation recorded in the :cpp:class:`RuntimeContext`
 * event log.
 *
 *  * ``kAdd``     — a new entry was inserted (e.g. via :cpp:func:`RuntimeContext::Set`
 *                   or :cpp:func:`RuntimeContext::Put` on a previously absent name).
 *  * ``kReplace`` — an existing entry was overwritten via
 *                   :cpp:func:`RuntimeContext::Put`.
 *  * ``kRemove``  — an entry was erased via :cpp:func:`RuntimeContext::Remove`.
 */
enum class TensorEventAction : int32_t { kAdd = 0, kReplace = 1, kRemove = 2 };

/**
 * Returns a short lowercase label for ``action`` (``"add"``, ``"replace"``,
 * ``"remove"``). Useful for human-readable rendering of the event log.
 */
const char *TensorEventActionName(TensorEventAction action) noexcept;

/**
 * Single entry of the :cpp:class:`RuntimeContext` event log.
 *
 * Each mutation of the underlying ``TensorMap`` performed through
 * :cpp:func:`RuntimeContext::Set`, :cpp:func:`RuntimeContext::Put` or
 * :cpp:func:`RuntimeContext::Remove` produces one ``TensorEvent`` capturing
 * the action, the name of the tensor, the wall-clock timestamp (nanoseconds
 * since the Unix epoch), and a snapshot of the tensor's type and shape.
 *
 * For tensors whose ``element_count`` is at most
 * :cpp:var:`kTensorEventValueLimit`, the element values are also captured
 * inline (in ``values`` for numeric dtypes, in ``string_values`` for
 * ``DataType::STRING``). For larger tensors both fields are left empty so
 * the log stays bounded.
 *
 * ``kRemove`` events always set ``data_type = DataType::UNDEFINED``,
 * leave ``shape`` empty and do not populate ``values`` / ``string_values``;
 * they only record the name and timestamp of the removal.
 */
struct TensorEvent {
  /// Kind of mutation recorded by this entry.
  TensorEventAction action = TensorEventAction::kAdd;
  /// Wall-clock timestamp of the event, in nanoseconds since the Unix
  /// epoch (``std::chrono::system_clock``).
  int64_t timestamp_ns = 0;
  /// Name under which the tensor is (or was) stored in the
  /// :cpp:class:`RuntimeContext` tensor map.
  std::string name;
  /// Element data type of the tensor at the moment of the event, encoded
  /// as a ``TensorProto::DataType`` integer value. Set to
  /// ``DataType::UNDEFINED`` for ``kRemove`` events.
  int32_t data_type = 0;
  /// Tensor shape at the moment of the event. Empty for ``kRemove`` and
  /// for scalar tensors (``element_count == 1``).
  std::vector<int64_t> shape;
  /// Numeric values of the tensor (coerced to ``double``), captured only
  /// when ``data_type`` is numeric and ``element_count <= kTensorEventValueLimit``.
  /// Boolean values are recorded as ``0.0`` / ``1.0``. Empty otherwise.
  std::vector<double> values;
  /// String values of the tensor, captured only when ``data_type`` is
  /// ``DataType::STRING`` and ``element_count <= kTensorEventValueLimit``.
  std::vector<std::string> string_values;
};

/**
 * Append-only log of tensor map mutations recorded by
 * :cpp:class:`RuntimeContext`.
 */
using TensorEventLog = std::vector<TensorEvent>;

/**
 * Per-invocation runtime state passed to :cpp:func:`RunNode` /
 * :cpp:func:`RunNodes`.
 *
 * Bundles together everything a chain of nodes needs to execute:
 *  * a :cpp:type:`TensorMap` carrying the graph inputs / initializers
 *    and every intermediate value produced by previously executed
 *    nodes (accessed through :cpp:func:`tensors`);
 *  * the construction-time :cpp:class:`kernel::KernelContext` (opset
 *    and any future construction-time inputs) used to instantiate
 *    each per-operator kernel (accessed through :cpp:func:`kernel_ctx`).
 *
 * Grouping them in a single object keeps the dispatcher signatures
 * stable as more per-invocation state (allocators, device descriptors,
 * profiling hooks, …) is added in the future without forcing every
 * trampoline or call site to take an extra argument.
 *
 * Convenience accessors (:cpp:func:`Set`, :cpp:func:`Get`,
 * :cpp:func:`Has`, :cpp:func:`Remove`) wrap the underlying map so
 * callers do not have to reach for ``rt.tensors()[name]`` directly.
 */
class RuntimeContext {
public:
  RuntimeContext() = default;
  explicit RuntimeContext(kernel::KernelContext kernel_ctx) : kernel_ctx_(std::move(kernel_ctx)) {}
  RuntimeContext(kernel::KernelContext kernel_ctx, TensorMap tensors)
      : tensors_(std::move(tensors)), kernel_ctx_(std::move(kernel_ctx)) {}

  /// In/out tensor map shared across every node in a chain.
  TensorMap &tensors() noexcept { return tensors_; }
  const TensorMap &tensors() const noexcept { return tensors_; }

  /// Kernel construction context (opset).
  kernel::KernelContext &kernel_ctx() noexcept { return kernel_ctx_; }
  const kernel::KernelContext &kernel_ctx() const noexcept { return kernel_ctx_; }

  /// Model-local function registry consulted by :cpp:func:`RunNode`
  /// before falling back to the built-in kernel dispatch table.
  FunctionMap &functions() noexcept { return functions_; }
  const FunctionMap &functions() const noexcept { return functions_; }

  /// Returns ``true`` if a tensor named ``name`` is currently held.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Removes the tensor stored under ``name`` if present. Returns
  /// ``true`` if an entry was erased, ``false`` otherwise. When an entry
  /// is erased a :cpp:class:`TensorEvent` with action
  /// :cpp:enumerator:`TensorEventAction::kRemove` is appended to the
  /// event log; nothing is logged when ``name`` is not present.
  bool Remove(const std::string &name);

  /// Inserts the tensor under ``name``. The name must not already
  /// be present in the map; use :cpp:func:`Put` (or ``tensors()``
  /// directly) to overwrite. A :cpp:class:`TensorEvent` with action
  /// :cpp:enumerator:`TensorEventAction::kAdd` is appended to the event
  /// log on successful insertion.
  void Set(const std::string &name, Tensor tensor);

  /// Inserts or overwrites the tensor stored under ``name``. Appends a
  /// :cpp:class:`TensorEvent` describing the new state with action
  /// :cpp:enumerator:`TensorEventAction::kAdd` when ``name`` was absent
  /// and :cpp:enumerator:`TensorEventAction::kReplace` when an existing
  /// entry was overwritten.
  void Put(const std::string &name, Tensor tensor);

  /**
   * Returns the tensor stored under ``name``.
   *
   * @throws std::out_of_range if ``name`` is not in the map.
   */
  const Tensor &Get(const std::string &name) const;
  Tensor &Get(const std::string &name);

  /// Append-only log of every tensor map mutation performed through
  /// :cpp:func:`Set`, :cpp:func:`Put` and :cpp:func:`Remove`. See
  /// :cpp:class:`TensorEvent` for the captured fields.
  const TensorEventLog &events() const noexcept { return events_; }
  TensorEventLog &events() noexcept { return events_; }

  /// Empties the event log without otherwise touching the tensor map.
  void ClearEvents() noexcept { events_.clear(); }

private:
  TensorMap tensors_;
  kernel::KernelContext kernel_ctx_;
  FunctionMap functions_;
  TensorEventLog events_;
};

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
