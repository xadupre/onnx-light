// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_sequence.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

/**
 * @file kernel_dispatch_table.h
 * @brief Per-(domain, op_type) dispatch table used by
 *        :cpp:func:`RunNode` / :cpp:class:`RuntimeSession` to resolve each
 *        ``NodeProto`` to a matching kernel factory.
 *
 * The generic dispatch mechanism (this file, :cpp:class:`RuntimeContext`,
 * :cpp:func:`RunNode`, ...) lives in ``onnx_core`` so it has no
 * dependency on any particular set of operator kernels. The concrete
 * kernel implementations for every standard ONNX operator remain in
 * ``onnx_kernels`` and register themselves here via
 * :cpp:func:`RegisterKernelFn` instead of being hard-coded in this
 * table, which keeps the ``onnx_core`` -> ``onnx_kernels`` dependency
 * direction from ever being introduced.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Factory signature registered in :cpp:func:`KernelDispatchTable` for every
 * ``(domain, op_type)``. Called once per node (during kernel resolution /
 * initialization, e.g. by :cpp:func:`RuntimeSession::Run` or by
 * :cpp:func:`RunNode`): validates the node's input/output counts, constructs
 * the concrete kernel object and attaches the node via
 * :cpp:func:`KernelBase::set_node`, returning a :cpp:class:`KernelBase`. Must
 * NOT perform any computation itself — all per-run computation belongs in the
 * returned kernel's :cpp:func:`KernelBase::Run`.
 */
using NodeKernelFn =
    std::function<std::unique_ptr<KernelBase>(const NodeProto &node, RuntimeContext &rt)>;

/**
 * Signature of the ``SequenceMap`` output-packing callback: given the
 * input sequence (whose length sets the iteration count) and the
 * per-iteration body outputs (``body_outputs_per_iter[k][i]`` is body
 * output ``k`` at iteration ``i``), returns the ``M`` assembled output
 * sequences. Registered by ``onnx_kernels`` (``kernel::SequenceMap``)
 * so that :cpp:func:`RunNode`'s ``SequenceMap`` orchestration (which
 * must live in ``onnx_core`` since it recursively drives a
 * :cpp:class:`RuntimeSession`) never has to include an ``onnx_kernels``
 * header directly.
 */
using SequenceMapPackFn =
    std::function<Sequences(RuntimeContext &rt, const Sequence &input_sequence,
                            const std::vector<Tensors> &body_outputs_per_iter)>;

/**
 * Returns the ``(normalised_domain, op_type) -> NodeKernelFn`` factory
 * dispatch table. Empty until kernel libraries (e.g. ``onnx_kernels``)
 * populate it via :cpp:func:`RegisterKernelFn`.
 */
const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable();

/**
 * Registers (or replaces) the kernel factory for the identifier
 * (@p domain, @p op_type, @p device) in the shared
 * :cpp:func:`KernelDispatchTable`.
 *
 * Use an empty string for @p domain to denote the default ONNX domain
 * (normalised to :cpp:var:`kDefaultOnnxDomain`). @p device is part of the
 * identifier so that a distinct kernel can be registered per device;
 * :cpp:enumerator:`symbolic::Device::kCPU` (and
 * :cpp:enumerator:`symbolic::Device::kUndefined`) denote the default host
 * entry. Intended to be called once per operator during static
 * initialization by kernel libraries that must not be linked into
 * ``onnx_core`` (e.g. ``onnx_kernels``); see
 * ``onnx_kernels::RegisterKernelFunctions``.
 *
 * When @p overwrite is ``true`` (the default) an existing entry for the same
 * identifier is replaced. Pass ``false`` to register the factory only if no
 * kernel is registered for that identifier yet; this lets bulk built-in
 * registration (``onnx_kernels::RegisterKernelFunctions``) run *without*
 * clobbering an override that a downstream library (e.g. ``onnx-light-cpu``)
 * installed earlier, so a custom kernel wins regardless of the order in which
 * the two registrations happen. Explicit per-operator overrides keep the
 * default ``overwrite = true`` so they always take precedence over the
 * built-ins.
 *
 * @param domain    The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type   The ONNX operator type name (e.g. ``"Abs"``).
 * @param device    The device the kernel runs on (e.g. :cpp:enumerator:`symbolic::Device::kCPU`).
 * @param fn        The factory implementing kernel construction for this operator.
 * @param overwrite Whether to replace an existing entry (``true``, default) or
 *                  keep it and ignore @p fn when one already exists (``false``).
 * @return ``true`` when @p fn was stored, ``false`` when an existing entry was
 *         kept because @p overwrite was ``false``.
 */
bool RegisterKernelFn(const std::string &domain, const std::string &op_type,
                      symbolic::Device device, NodeKernelFn fn, bool overwrite = true);

/**
 * Returns the process-wide (global) custom-kernel registry consulted by
 * :cpp:func:`RunNode` / :cpp:class:`RuntimeSession` during kernel resolution.
 *
 * Global custom kernels complement the per-:cpp:class:`RuntimeContext`
 * registry (:cpp:func:`RuntimeContext::RegisterCustomKernel`): they let a
 * caller install a kernel once and have every :cpp:class:`RuntimeContext`
 * pick it up, rather than registering it on each context separately.
 * Resolution precedence is: model-local functions, built-in control-flow
 * operators, per-context custom kernels, **global custom kernels**, then the
 * built-in :cpp:func:`KernelDispatchTable`. A per-context registration for the
 * same ``(domain, op_type)`` therefore overrides the global one.
 *
 * Keys are the canonical ``"<domain>:<op_type>"`` pair (the default ONNX
 * domain — the empty ``NodeProto::domain()`` — is normalised to
 * :cpp:var:`kDefaultOnnxDomain`).
 *
 * Like the built-in :cpp:func:`KernelDispatchTable`, this registry is a plain
 * process-wide singleton and is **not** synchronised: register / unregister
 * global kernels before starting concurrent :cpp:func:`RunNode` /
 * :cpp:class:`RuntimeSession::Run` calls (e.g. during start-up), not while
 * other threads are resolving nodes.
 */
const CustomKernelMap &GlobalCustomKernels();

/**
 * Registers (or replaces) a process-wide (global) custom kernel for
 * (@p domain, @p op_type) in :cpp:func:`GlobalCustomKernels`. The empty
 * domain is normalised to :cpp:var:`kDefaultOnnxDomain`. Unlike
 * :cpp:func:`RegisterKernelFn` (which registers a per-device kernel
 * *factory*), @p fn keeps the simple "run the whole node now" contract of
 * :cpp:type:`CustomKernelFn`.
 */
void RegisterGlobalCustomKernel(const std::string &domain, const std::string &op_type,
                                CustomKernelFn fn);

/**
 * Removes the process-wide custom kernel registered for (@p domain,
 * @p op_type). The empty domain is normalised to :cpp:var:`kDefaultOnnxDomain`.
 * Returns ``true`` when an entry was removed, ``false`` otherwise.
 */
bool UnregisterGlobalCustomKernel(const std::string &domain, const std::string &op_type);

/**
 * Removes every process-wide custom kernel registration from
 * :cpp:func:`GlobalCustomKernels`.
 */
void ClearGlobalCustomKernels();

/**
 * Returns the currently registered ``SequenceMap`` output-packing
 * callback, or an empty ``std::function`` if none has been registered
 * yet (see :cpp:func:`RegisterSequenceMapPackFn`).
 */
const SequenceMapPackFn &GetSequenceMapPackFn();

/**
 * Registers the ``SequenceMap`` output-packing callback (see
 * :cpp:type:`SequenceMapPackFn`). Called once by
 * ``onnx_kernels::RegisterKernelFunctions``.
 */
void RegisterSequenceMapPackFn(SequenceMapPackFn fn);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
