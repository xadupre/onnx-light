// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
 * @param accepts_structured_inputs Whether the function explicitly handles structured values.
 * @param accepts_type_only_tensor_inputs Whether the function handles tensor types without rank.
 */
void RegisterComputeShapeFn(const std::string &domain, const std::string &op_type,
                            ComputeShapeFn fn, bool accepts_structured_inputs = false,
                            bool accepts_type_only_tensor_inputs = false);

/** Returns whether a registered shape function explicitly accepts structured/encoded inputs. */
bool AcceptsStructuredShapeInputs(const std::string &domain, const std::string &op_type);

/** Returns whether a registered shape function accepts tensor types without a shape descriptor. */
bool AcceptsTypeOnlyTensorShapeInputs(const std::string &domain, const std::string &op_type);

/** Returns whether all node inputs satisfy its shape function's descriptor requirements. */
bool ShapeInputsAvailable(const ShapesContext &shapes, const NodeProto &node);

/** Returns whether model-local, context-local or global shape inference is available. */
bool HasShapeInference(const ShapesContext &shapes, const NodeProto &node);

/**
 * Describes additional peak memory in bytes, excluding declared inputs and outputs.
 * ``worst_case`` is always present, as a nonnegative integer or a nonempty
 * symbolic expression. An unresolved symbol requires a caller-supplied bound
 * before memory admission; it never means zero bytes.
 *
 * ``average_case`` is absent when no justified expectation is available.
 * Deterministic workspace requirements have equal worst and average cases.
 */
struct PeakMemoryEstimate {
  symbolic::SymDim worst_case{"peak_memory_unknown"};
  std::optional<symbolic::SymDim> average_case;

  PeakMemoryEstimate() = default;
  /// Constructs a concrete upper bound without assuming an average.
  PeakMemoryEstimate(int64_t bytes) : worst_case(bytes) {}
  PeakMemoryEstimate(symbolic::SymDim maximum, std::optional<symbolic::SymDim> average)
      : worst_case(std::move(maximum)), average_case(std::move(average)) {}
  static PeakMemoryEstimate Exact(symbolic::SymDim bytes) { return {bytes, bytes}; }
  bool operator==(const PeakMemoryEstimate &) const = default;
};

/// Returns an unresolved operator-wide bound, covering every invocation and device.
PeakMemoryEstimate UnknownPeakMemory(const std::string &domain, const std::string &op_type);

/** Node-aware peak-memory estimator, with access to attributes and optional input names. */
using ComputePeakMemoryNodeFn =
    std::function<PeakMemoryEstimate(Device, const NodeProto &, const std::vector<SymShape> &)>;

/**
 * Returns the ``(normalised_domain, op_type) -> ComputePeakMemoryNodeFn``
 * dispatch table. Empty until libraries populate it via
 * :cpp:func:`RegisterComputePeakMemoryFn`; operators without an entry
 * report zero additional peak memory through :cpp:func:`ComputePeakMemory`.
 */
const std::unordered_map<std::string, ComputePeakMemoryNodeFn> &PeakMemoryDispatchTable();

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
 * Shape-only queries pass a node containing only its domain and operator type.
 *
 * @param domain  The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type The ONNX operator type name (e.g. ``"Abs"``).
 * @param device  The device the estimator applies to (e.g. :cpp:enumerator:`Device::kCPU`).
 * @param fn      The peak-memory function, or empty for zero additional memory.
 *                An empty entry also prevents non-overwriting default registration.
 * @param overwrite Whether to replace an existing entry.
 */
void RegisterComputePeakMemoryFn(const std::string &domain, const std::string &op_type,
                                 Device device, ComputePeakMemoryNodeFn fn, bool overwrite = true);

/// Evaluates and validates an implementation's estimator without consulting the registry.
/// An empty estimator reports zero additional memory.
PeakMemoryEstimate ComputePeakMemory(const ComputePeakMemoryNodeFn &fn, Device device,
                                     const NodeProto &node,
                                     const std::vector<SymShape> &input_shapes);

/**
 * Returns the estimated peak memory (in bytes) for (@p domain, @p op_type)
 * executed on @p device with inputs of shape @p input_shapes.
 *
 * Looks the operator up in the shared :cpp:func:`PeakMemoryDispatchTable`
 * and forwards a node containing only its domain and operator type to its registered
 * :cpp:type:`ComputePeakMemoryNodeFn`. When no function is registered, returns zero.
 *
 * @param domain       The operator domain (``""`` or ``"ai.onnx"`` for standard ONNX).
 * @param op_type      The ONNX operator type name (e.g. ``"Abs"``).
 * @param device       The device on which the operator executes.
 * @param input_shapes The shapes of the operator's inputs, in order.
 * @returns The mandatory worst-case and optional average peak-memory estimates.
 */
PeakMemoryEstimate ComputePeakMemory(const std::string &domain, const std::string &op_type,
                                     Device device, const std::vector<SymShape> &input_shapes);

/**
 * Returns additional peak memory for a node, forwarding it to the registered estimator.
 * Unregistered operators return zero.
 * Input shapes retain node input positions; absent optional inputs have empty shapes.
 */
PeakMemoryEstimate ComputePeakMemory(Device device, const NodeProto &node,
                                     const std::vector<SymShape> &input_shapes);

/**
 * Returns additional peak memory using descriptors from a shapes context.
 * Missing named inputs yield an unknown bound for registered estimators, not
 * scalar shapes. Omitted optional inputs retain their empty positional shapes.
 * Unregistered operators return zero, including with incomplete descriptors.
 */
PeakMemoryEstimate ComputePeakMemory(const ShapesContext &ctx, Device device,
                                     const NodeProto &node);

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
