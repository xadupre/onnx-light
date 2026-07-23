// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace compute {

std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuse(const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite);
  return inplace.Reuse();
}

void WriteInPlaceReuseToMetadata(GraphProto &graph, const ShapesContext &ctx,
                                 const std::unordered_map<std::string, std::string> &value_tags) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, value_tags);
  inplace.WriteToMetadata(graph);
}

} // namespace compute
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
