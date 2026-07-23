// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_sequence.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_proto/onnx.h"

/**
 * @file kernel_dispatch_table.h
 * @brief Per-(domain, op_type) dispatch table used by
 *        :cpp:func:`RunNode` to forward each ``NodeProto`` to the
 *        matching kernel trampoline.
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

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

/**
 * Signature of every per-operator trampoline registered in
 * :cpp:func:`KernelDispatchTable`. Implementations read their inputs
 * from ``rt.tensors()`` by name, call the matching kernel
 * (constructed with ``rt.kernel_ctx()``), and insert the produced
 * outputs back into ``rt.tensors()`` under the names declared by
 * ``node.output(i)``.
 */
using NodeKernelFn = std::function<void(const NodeProto &node, RuntimeContext &rt)>;

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
    std::function<std::vector<Sequence>(RuntimeContext &rt, const Sequence &input_sequence,
                                        const std::vector<Tensors> &body_outputs_per_iter)>;

/**
 * Returns the ``(normalised_domain, op_type) -> NodeKernelFn`` dispatch
 * table. Empty until kernel libraries (e.g. ``onnx_kernels``) populate
 * it via :cpp:func:`RegisterKernelFn`.
 */
const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable();

/**
 * Registers (or replaces) the kernel trampoline for (@p domain,
 * @p op_type) in the shared :cpp:func:`KernelDispatchTable`.
 *
 * Use an empty string for @p domain to denote the default ONNX domain
 * (normalised to :cpp:var:`kDefaultOnnxDomain`). Intended to be called
 * once per operator during static initialization by kernel libraries
 * that must not be linked into ``onnx_core`` (e.g. ``onnx_kernels``);
 * see ``onnx_kernels::RegisterKernelFunctions``.
 *
 * @param domain  The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type The ONNX operator type name (e.g. ``"Abs"``).
 * @param fn      The trampoline implementing the kernel dispatch for this operator.
 */
void RegisterKernelFn(const std::string &domain, const std::string &op_type, NodeKernelFn fn);

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

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
