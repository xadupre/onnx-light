// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file kernel_dispatch_table.h
 * @brief Registers every ``onnx_kernels`` operator kernel (the built-in
 *        ``ai.onnx``/``ai.onnx.ml``/``ai.rt``/... operator set) with the
 *        generic kernel dispatch table owned by ``onnx_core``
 *        (:cpp:func:`core::runtime::KernelDispatchTable`).
 *
 * The kernel implementations themselves stay in ``onnx_kernels``, one per
 * ``onnx_kernels/kernels/<domain>/kernel_<name>.cc`` file (except the
 * control-flow kernels ``If``/``Loop``/``Scan``, which live in
 * ``onnx_core/runtime/kernels/controlflow`` since the runtime dispatcher needs to
 * invoke them directly while recursively evaluating sub-graphs), and this
 * translation unit is the single place that wires all of them into the
 * shared registry via :cpp:func:`RegisterKernelFunctions`. Keeping the
 * registration here (instead of in ``onnx_core``) preserves the
 * ``onnx_kernels`` -> ``onnx_core`` dependency direction: ``onnx_core``
 * never needs to know about ``onnx_kernels``'s operator implementations.
 */

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {

/**
 * Registers every built-in ``onnx_kernels`` operator kernel with
 * :cpp:func:`core::runtime::RegisterKernelFn`, and registers
 * ``kernel::SequenceMap`` as the ``core::runtime`` ``SequenceMap``
 * output-packing callback (see
 * :cpp:func:`core::runtime::RegisterSequenceMapPackFn`). Idempotent and
 * cheap to call more than once (the actual registration work only happens
 * once, guarded by a function-local static).
 *
 * Unlike ``onnx_lib``'s ``OpSchemaRegistry::map()`` (which can lazily
 * self-register because both the accessor and the registration functions
 * live in the same library), ``core::runtime::KernelDispatchTable()``
 * cannot do this: ``onnx_core`` must not depend on or call into
 * ``onnx_kernels``. ``lib_onnx_kernels`` is also a plain static archive, so
 * a file-scope static object with no externally-referenced symbol is not
 * reliably linked in either. Every entry point that runs a model (Python
 * bindings, C++ unit tests, examples, fuzzers, ...) must therefore call
 * this function explicitly before calling
 * :cpp:func:`core::runtime::RunNode` / running a
 * :cpp:class:`core::runtime::RuntimeSession`.
 */
void RegisterKernelFunctions();

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
