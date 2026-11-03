// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/peak_memory.h"

#include "onnx_core/shapes/dispatch_table.h"

#include <algorithm>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::compute {

void WritePeakMemoryEstimate(NodeProto &node, const shapes::PeakMemoryEstimate &estimate) {
  auto &metadata = node.ref_metadata_props();
  metadata.erase(std::remove_if(metadata.begin(), metadata.end(),
                                [](const auto &entry) {
                                  return entry.key() == kNodePeakMemoryMetadataKey ||
                                         entry.key() == kNodePeakMemoryMaxMetadataKey ||
                                         entry.key() == kNodePeakMemoryAverageMetadataKey;
                                }),
                 metadata.end());
  node.add_metadata(kNodePeakMemoryMaxMetadataKey, estimate.worst_case.ToString());
  if (estimate.average_case)
    node.add_metadata(kNodePeakMemoryAverageMetadataKey, estimate.average_case->ToString());
  if (estimate.worst_case.IsInt() && estimate.worst_case.AsInt() > 0)
    node.add_metadata(kNodePeakMemoryMetadataKey, estimate.worst_case.ToString());
}

void WritePeakMemoryToMetadata(GraphProto &graph, const ShapesContext &ctx, Device device) {
  for (NodeProto &node : *graph.mutable_node()) {
    WritePeakMemoryEstimate(node, shapes::ComputePeakMemory(ctx, device, node));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
