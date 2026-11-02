// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/peak_memory.h"

#include "onnx_core/shapes/dispatch_table.h"

#include <algorithm>
#include <string>
#include <vector>

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
    std::vector<symbolic::SymShape> input_shapes;
    input_shapes.reserve(node.input().size());
    for (const auto &input_name : node.input()) {
      if (!input_name.empty() && ctx.Has(input_name)) {
        input_shapes.push_back(ctx.Get(input_name).Shape());
      } else {
        input_shapes.emplace_back();
      }
    }
    WritePeakMemoryEstimate(
        node, shapes::ComputePeakMemory(node.domain(), node.op_type(), device, input_shapes));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
