// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

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
 * standard ONNX operator remain in ``onnx_optim`` and register themselves
 * here via :cpp:func:`RegisterComputeShapeFn` instead of being hard-coded
 * in this table, which keeps the ``onnx_core`` -> ``onnx_optim``
 * dependency direction from ever being introduced.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace shapes {

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
 * ``onnx_optim``) populate it via :cpp:func:`RegisterComputeShapeFn`.
 */
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable();

/**
 * Registers (or replaces) the ``ComputeShape*`` function for
 * (@p domain, @p op_type) in the shared :cpp:func:`DispatchTable`.
 *
 * Use an empty string for @p domain to denote the default ONNX domain
 * (normalised to :cpp:var:`kOnnxDomain`). Intended to be called once per
 * operator during static initialization by shape-function libraries
 * that must not be linked into ``onnx_core`` (e.g. ``onnx_optim``);
 * see ``onnx_optim::RegisterShapeFunctions``.
 *
 * @param domain  The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type The ONNX operator type name (e.g. ``"Abs"``).
 * @param fn      The shape function implementing the ``ComputeShape*`` rule.
 */
void RegisterComputeShapeFn(const std::string &domain, const std::string &op_type,
                            ComputeShapeFn fn);

} // namespace shapes
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
