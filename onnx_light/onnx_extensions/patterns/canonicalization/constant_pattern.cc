// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"

#include <cstddef>
#include <string>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultConstant(const NodeProto &node) {
  return node.op_type().value() == "Constant" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.output_size() == 1;
}

// Re-encodes a numeric constant tensor so its payload lives in ``raw_data``, the
// canonical form the constant folder materialises (see ProtoFromRuntimeTensor in
// graph_builder.cc). The Identity node this pattern emits is folded away, and the
// folder re-materialises the value as a ``raw_data`` initializer. Producing the
// intermediate initializer in the same encoding lets the byte-strict initializer
// deduplication collapse both copies into a single initializer instead of leaving
// an orphaned duplicate. ``STRING`` tensors already share the ``string_data``
// encoding with the folder, so they are copied unchanged.
TensorProto CanonicaliseInitializer(const TensorProto &value, const std::string &name) {
  if (static_cast<TensorProto::DataType>(value.data_type()) == TensorProto::DataType::STRING ||
      value.has_raw_data()) {
    TensorProto proto = value;
    proto.set_name(name);
    return proto;
  }
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(value);
  TensorProto proto;
  proto.set_name(name);
  proto.set_data_type(value.data_type());
  for (std::size_t i = 0; i < value.dims().size(); ++i) {
    proto.add_dims(value.dims()[i]);
  }
  proto.set_raw_data(tensor.bytes(), tensor.size_bytes());
  return proto;
}

} // namespace

std::set<std::string> ConstantToInitializerPattern::FastOpType() const { return {"Constant"}; }

core::builder::MatchResult ConstantToInitializerPattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (!IsDefaultConstant(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Constant");
  }
  if (graph.GetComputedConstant(candidate.output()[0].value()) == nullptr) {
    return NoMatch(candidate, "the Constant value is not materialised as a tensor");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConstantToInitializerPattern::Apply(core::builder::GraphGraph &graph,
                                    const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ConstantToInitializerPattern::Apply expects one Constant node.");
  }
  const NodeProto &node = *nodes[0];
  const TensorProto *value = graph.GetComputedConstant(node.output()[0].value());
  if (value == nullptr) {
    throw BuilderError(
        "ConstantToInitializerPattern::Apply received a non-materialisable Constant.");
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string base = node.output()[0].value() + "_cst2init";
  std::string init_name = base;
  for (int suffix = 0; builder.HasName(init_name); ++suffix) {
    init_name = base + "_" + std::to_string(suffix);
  }
  builder.MakeInitializer(CanonicaliseInitializer(*value, init_name));

  const std::string name = "ConstantToInitializerPattern--" + node.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakeNode("Identity", {init_name}, {node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
