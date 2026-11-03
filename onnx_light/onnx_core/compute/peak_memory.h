// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::Device;

/**
 * Legacy concrete allocation hint, read by execution plans. Contains only
 * positive decimal byte counts, never expressions. Its absence is not a
 * zero-memory guarantee; complete analysis uses the mandatory maximum key.
 */
constexpr const char *kNodePeakMemoryMetadataKey = "onnx_light.peak_memory";
/// Mandatory worst-case expression, including unresolved bounds.
constexpr const char *kNodePeakMemoryMaxMetadataKey = "onnx_light.peak_memory_max";
/// Optional average-case expression.
constexpr const char *kNodePeakMemoryAverageMetadataKey = "onnx_light.peak_memory_average";

/// Writes both estimates and refreshes the legacy concrete allocation hint.
void WritePeakMemoryEstimate(NodeProto &node, const shapes::PeakMemoryEstimate &estimate);

/**
 * Computes the estimated additional peak memory for every node of ``graph``
 * using the shapes already inferred into ``ctx``, and records the result in
 * ``metadata_props`` under :cpp:var:`kNodePeakMemoryMaxMetadataKey` and, when
 * available, :cpp:var:`kNodePeakMemoryAverageMetadataKey`. Unknown bounds are
 * preserved symbolically when explicitly returned by an estimator; missing
 * estimators default to zero. Positive concrete maxima also refresh
 * :cpp:var:`kNodePeakMemoryMetadataKey` without passing expressions to legacy
 * execution-plan allocation hints.
 *
 * The peak memory accounts only for the additional memory an
 * operator's computation allocates, not the memory already accounted for by
 * its inputs and outputs.
 *
 * @param graph   Graph whose nodes are mutated in place.
 * @param ctx     Shapes context already populated with the inferred
 *                descriptors for ``graph``.
 * @param device  Logical device passed to the peak-memory dispatch
 *                function (defaults to :cpp:enumerator:`Device::kUndefined`).
 */
void WritePeakMemoryToMetadata(GraphProto &graph, const ShapesContext &ctx,
                               Device device = Device::kUndefined);

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
