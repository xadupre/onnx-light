// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/peak_memory.h"

#include "onnx_core/shapes/dispatch_table.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::compute {

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
    const int64_t peak =
        shapes::ComputePeakMemory(node.domain(), node.op_type(), device, input_shapes);
    if (peak > 0) {
      node.add_metadata(kNodePeakMemoryMetadataKey, std::to_string(peak));
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
