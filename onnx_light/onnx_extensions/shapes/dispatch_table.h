// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file dispatch_table.h
 * @brief Registers every ``onnx_shapes`` ``ComputeShape*`` shape function
 *        (the built-in ``ai.onnx``/``ai.onnx.ml``/... operator set) with the
 *        generic dispatch table owned by ``onnx_core``
 *        (:cpp:func:`core::shapes::DispatchTable`).
 *
 * The shape functions themselves (e.g. ``math::ComputeShapeAbs``) stay in
 * ``onnx_shapes``, one per ``onnx_extensions/shapes/shapes/<domain>/shape_<domain>.cc``
 * file, and this translation unit is the single place that wires all of
 * them into the shared registry via :cpp:func:`RegisterShapeFunctions`.
 * Keeping the registration here (instead of in ``onnx_core``) preserves
 * the ``onnx_shapes`` -> ``onnx_core`` dependency direction: ``onnx_core``
 * never needs to know about ``onnx_shapes``'s operator implementations.
 */

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

/**
 * Registers every built-in ``onnx_shapes`` shape function with
 * :cpp:func:`core::shapes::RegisterComputeShapeFn`. Idempotent and cheap to
 * call more than once (the actual registration work only happens once,
 * guarded by a function-local static).
 *
 * Unlike ``onnx_lib``'s ``OpSchemaRegistry::map()`` (which can lazily
 * self-register because both the accessor and the registration functions
 * live in the same library), ``core::shapes::DispatchTable()`` cannot do
 * this: ``onnx_core`` must not depend on or call into ``onnx_shapes``.
 * ``lib_onnx_shape`` is also a plain static archive, so a file-scope static
 * object with no externally-referenced symbol is not reliably linked in
 * either. Every entry point that uses the shape-inference engine (Python
 * bindings, C++ unit tests, examples, fuzzers, ...) must therefore call
 * this function explicitly before calling
 * :cpp:func:`core::shapes::InferShapesModel` or
 * :cpp:func:`core::shapes::ShapesContext::ComputeShapeNode`.
 */
void RegisterShapeFunctions();

/**
 * Registers every built-in ``onnx_shapes`` peak-memory function with
 * :cpp:func:`core::shapes::RegisterComputePeakMemoryFn`. Mirrors
 * :cpp:func:`RegisterShapeFunctions` for the peak-memory dispatch table:
 * idempotent, guarded by a function-local static, and must be called
 * explicitly before :cpp:func:`core::shapes::ComputePeakMemory` can resolve a
 * built-in operator (operators without a registered function report ``0``).
 */
void RegisterPeakMemoryFunctions();

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
