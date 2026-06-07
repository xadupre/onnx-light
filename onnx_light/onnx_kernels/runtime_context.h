// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"

#include <string>
#include <unordered_map>

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
  /// ``true`` if an entry was erased, ``false`` otherwise.
  bool Remove(const std::string &name) { return tensors_.erase(name) > 0; }

  /// Inserts the tensor under ``name``. The name must not already
  /// be present in the map; use ``tensors()`` directly to overwrite.
  void Set(const std::string &name, Tensor tensor) {
    EXT_ENFORCE(!Has(name), "RuntimeContext::Set: a tensor named '", name, "' already exists.");
    tensors_[name] = std::move(tensor);
  }

  /**
   * Returns the tensor stored under ``name``.
   *
   * @throws std::out_of_range if ``name`` is not in the map.
   */
  const Tensor &Get(const std::string &name) const;
  Tensor &Get(const std::string &name);

private:
  TensorMap tensors_;
  kernel::KernelContext kernel_ctx_;
  FunctionMap functions_;
};

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
