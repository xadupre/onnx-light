// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file dispatch_table.h
 * @brief Per-(domain, op_type) dispatch table used by
 *        :cpp:func:`core::shapes::ShapesContext::ComputeShapeNode` to forward
 *        each ``NodeProto`` to the matching ``ComputeShape*``
 *        implementation.
 *
 * The generic dispatch mechanism (this file, :cpp:class:`ShapesContext`,
 * :cpp:func:`InferShapesModel`, ...) lives in ``onnx_core`` so it has no
 * dependency on any particular set of operator implementations. The
 * concrete ``ComputeShape*`` functions ("shape functions") for every
 * standard ONNX operator remain in ``onnx_shapes`` and register themselves
 * here via :cpp:func:`RegisterComputeShapeFn` instead of being hard-coded
 * in this table, which keeps the ``onnx_core`` -> ``onnx_shapes``
 * dependency direction from ever being introduced.
 */

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

/**
 * Signature of every per-operator ``ComputeShape*`` trampoline
 * registered in :cpp:func:`DispatchTable`: it reads the node's inputs
 * from ``ctx`` and inserts the resulting output descriptors back
 * into ``ctx``.
 */
using ComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;

/**
 * Returns the ``(normalised_domain, op_type) -> ComputeShape*``
 * dispatch table. Empty until shape-function libraries (e.g.
 * ``onnx_shapes``) populate it via :cpp:func:`RegisterComputeShapeFn`.
 */
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable();

/**
 * Registers (or replaces) the ``ComputeShape*`` function for
 * (@p domain, @p op_type) in the shared :cpp:func:`DispatchTable`.
 *
 * Use an empty string for @p domain to denote the default ONNX domain
 * (normalised to :cpp:var:`kOnnxDomain`). Intended to be called once per
 * operator during static initialization by shape-function libraries
 * that must not be linked into ``onnx_core`` (e.g. ``onnx_shapes``);
 * see ``onnx_shapes::RegisterShapeFunctions``.
 *
 * @param domain  The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type The ONNX operator type name (e.g. ``"Abs"``).
 * @param fn      The shape function implementing the ``ComputeShape*`` rule.
 */
void RegisterComputeShapeFn(const std::string &domain, const std::string &op_type,
                            ComputeShapeFn fn);

/**
 * Signature of every per-operator peak-memory function registered in
 * :cpp:func:`PeakMemoryDispatchTable`. Mirroring :cpp:type:`ComputeShapeFn`,
 * but for memory rather than shapes, it estimates the peak amount of
 * scratch/working memory (in bytes) an operator's computation needs, in
 * addition to its declared inputs and outputs.
 *
 * The function takes the :cpp:enum:`Device` on which the operator executes
 * followed by the :cpp:class:`SymShape` of each of its inputs, and returns
 * the estimated peak memory as an ``int64_t``. The returned value is the
 * extra scratch/working memory the computation allocates; it excludes the
 * memory already accounted for by the operator's inputs and outputs. When no
 * function is registered for an operator the default is to return ``0`` (see
 * :cpp:func:`ComputePeakMemory`).
 */
using ComputePeakMemoryFn = std::function<int64_t(Device, const std::vector<SymShape> &)>;

/**
 * Returns the ``(normalised_domain, op_type) -> ComputePeakMemoryFn``
 * dispatch table. Empty until libraries populate it via
 * :cpp:func:`RegisterComputePeakMemoryFn`; operators without an entry
 * report a peak memory of ``0`` through :cpp:func:`ComputePeakMemory`.
 */
const std::unordered_map<std::string, ComputePeakMemoryFn> &PeakMemoryDispatchTable();

/**
 * Registers (or replaces) the peak-memory function for the identifier
 * (@p domain, @p op_type, @p device) in the shared
 * :cpp:func:`PeakMemoryDispatchTable`.
 *
 * Use an empty string for @p domain to denote the default ONNX domain
 * (normalised to :cpp:var:`kOnnxDomain`). @p device is part of the
 * identifier so that a distinct estimator can be registered per device;
 * :cpp:enumerator:`Device::kCPU` (and :cpp:enumerator:`Device::kUndefined`)
 * denote the default host entry. Mirrors :cpp:func:`RegisterComputeShapeFn`
 * so that libraries that must not be linked into ``onnx_core`` can
 * contribute their per-operator memory estimators.
 *
 * @param domain  The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type The ONNX operator type name (e.g. ``"Abs"``).
 * @param device  The device the estimator applies to (e.g. :cpp:enumerator:`Device::kCPU`).
 * @param fn      The peak-memory function implementing the estimation rule.
 */
void RegisterComputePeakMemoryFn(const std::string &domain, const std::string &op_type,
                                 Device device, ComputePeakMemoryFn fn);

/**
 * Returns the estimated peak memory (in bytes) for (@p domain, @p op_type)
 * executed on @p device with inputs of shape @p input_shapes.
 *
 * Looks the operator up in the shared :cpp:func:`PeakMemoryDispatchTable`
 * and forwards to its registered :cpp:type:`ComputePeakMemoryFn`. When no
 * function is registered for the operator the default is to return ``0``.
 *
 * @param domain       The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type      The ONNX operator type name (e.g. ``"Abs"``).
 * @param device       The device on which the operator executes.
 * @param input_shapes The shapes of the operator's inputs, in order.
 * @returns The estimated peak memory in bytes, or ``0`` when the operator
 *          has no registered peak-memory function.
 */
int64_t ComputePeakMemory(const std::string &domain, const std::string &op_type, Device device,
                          const std::vector<SymShape> &input_shapes);

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
