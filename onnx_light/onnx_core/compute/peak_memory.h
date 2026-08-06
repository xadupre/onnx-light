// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::Device;

/**
 * Metadata key under which :cpp:func:`WritePeakMemoryToMetadata` records,
 * for every node with a non-zero peak-memory estimate, the estimated peak
 * scratch memory in bytes. The associated value is the decimal string
 * representation of the ``int64_t`` byte count.
 */
constexpr const char *kNodePeakMemoryMetadataKey = "onnx_light.peak_memory";

/**
 * Computes the estimated peak scratch memory for every node of ``graph``
 * using the shapes already inferred into ``ctx``, and records the result in
 * ``metadata_props`` under :cpp:var:`kNodePeakMemoryMetadataKey`. Nodes
 * whose estimated peak memory is zero (either the operator has no registered
 * peak-memory function or all relevant input shapes are symbolic) are left
 * untouched.
 *
 * The peak memory accounts only for the extra scratch/working memory an
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
