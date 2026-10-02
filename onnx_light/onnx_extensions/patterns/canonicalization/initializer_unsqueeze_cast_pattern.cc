// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/initializer_unsqueeze_cast_pattern.h"

#include <algorithm>
#include <set>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

bool IsOp(const NodeProto &node, const char *op, std::size_t inputs) {
  return node.op_type().value() == op &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain &&
         static_cast<std::size_t>(node.input_size()) == inputs && node.output_size() == 1;
}

bool IsNumericType(int32_t dtype) {
  switch (static_cast<TensorProto::DataType>(dtype)) {
  case TensorProto::DataType::FLOAT:
  case TensorProto::DataType::DOUBLE:
  case TensorProto::DataType::FLOAT16:
  case TensorProto::DataType::BFLOAT16:
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::UINT16:
  case TensorProto::DataType::INT32:
  case TensorProto::DataType::INT64:
  case TensorProto::DataType::BOOL:
    return true;
  default:
    return false;
  }
}

bool FoldedDims(const TensorProto &source, const TensorProto &axes, std::vector<int64_t> &dims) {
  if (axes.data_type() != TensorProto::DataType::INT64 || axes.dims_size() != 1) {
    return false;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(axes, values) || values.empty() ||
      values.size() != static_cast<std::size_t>(axes.dims(0)) ||
      source.dims_size() + values.size() > core::runtime::Shape::kMaxRank) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(source.dims_size() + values.size());
  std::set<int64_t> positions;
  for (int64_t axis : values) {
    if (axis < -rank || axis >= rank || !positions.insert(axis < 0 ? axis + rank : axis).second) {
      return false;
    }
  }
  std::size_t input_index = 0;
  for (int64_t axis = 0; axis < rank; ++axis) {
    dims.push_back(positions.count(axis) ? 1 : source.dims(input_index++));
  }
  return true;
}

} // namespace

std::set<std::string> InitializerUnsqueezeCastPattern::FastOpType() const { return {"Cast"}; }

core::builder::MatchResult
InitializerUnsqueezeCastPattern::Match(core::builder::GraphGraph &graph,
                                       const NodeProto &candidate) const {
  if (!IsOp(candidate, "Cast", 1) || FindAttribute(candidate, "to") == nullptr ||
      FindAttribute(candidate, "to")->type() != AttributeProto::AttributeType::INT) {
    return NoMatch(candidate, "candidate is not a valid default-domain Cast");
  }
  const NodeProto *unsqueeze = graph.NodeBefore(candidate.input()[0].value());
  if (unsqueeze == nullptr || !IsOp(*unsqueeze, "Unsqueeze", 2) ||
      graph.IsUsedMoreThanOnce(unsqueeze->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "Cast must follow a single-use default-domain Unsqueeze");
  }
  const auto &consumers = graph.NextNodes(candidate.output()[0].value());
  if (consumers.size() != 1 || !IsOp(*consumers[0], "Add", 2)) {
    return NoMatch(candidate, "Cast must feed a single default-domain Add");
  }
  const auto &builder = graph.Builder();
  const std::string &source_name = unsqueeze->input()[0].value();
  if (std::any_of(
          builder.Inputs().begin(), builder.Inputs().end(),
          [&](const ValueInfoProto &input) { return input.name().value() == source_name; }) ||
      std::any_of(builder.Inputs().begin(), builder.Inputs().end(),
                  [&](const ValueInfoProto &input) {
                    return input.name().value() == unsqueeze->input()[1].value();
                  }) ||
      std::none_of(builder.Initializers().begin(), builder.Initializers().end(),
                   [&](const TensorProto &value) { return value.name().value() == source_name; })) {
    return NoMatch(candidate, "Unsqueeze input is not a non-overridable initializer");
  }
  const TensorProto *source = graph.GetComputedConstant(source_name);
  const TensorProto *axes = graph.GetComputedConstant(unsqueeze->input()[1].value());
  std::vector<int64_t> dims;
  if (source == nullptr || axes == nullptr ||
      (source->data_location() == TensorProto::DataLocation::EXTERNAL && !source->has_raw_data()) ||
      !IsNumericType(source->data_type()) || !IsNumericType(FindAttribute(candidate, "to")->i()) ||
      !FoldedDims(*source, *axes, dims)) {
    return NoMatch(candidate, "Unsqueeze axes are not valid constant axes");
  }
  const auto &kernels = core::runtime::KernelDispatchTable();
  if (kernels.find("ai.onnx:Cast") == kernels.end()) {
    return NoMatch(candidate, "Cast runtime kernel is unavailable");
  }
  return core::builder::MatchResult{this, {unsqueeze, &candidate, consumers[0]}, consumers[0]};
}

utils::RepeatedProtoField<NodeProto>
InitializerUnsqueezeCastPattern::Apply(core::builder::GraphGraph &graph,
                                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[1] == nullptr || Match(graph, *nodes[1]).nodes != nodes) {
    throw core::builder::BuilderError(
        "InitializerUnsqueezeCastPattern::Apply received an inconsistent match.");
  }
  const NodeProto &unsqueeze = *nodes[0];
  const NodeProto &cast = *nodes[1];
  const NodeProto &add = *nodes[2];
  TensorProto reshaped = *graph.GetComputedConstant(unsqueeze.input()[0].value());
  std::vector<int64_t> dims;
  FoldedDims(reshaped, *graph.GetComputedConstant(unsqueeze.input()[1].value()), dims);
  reshaped.clear_dims();
  for (int64_t dim : dims) {
    reshaped.add_dims(dim);
  }
  core::runtime::RuntimeContext rt(core::runtime::KernelContext(core::runtime::DefaultOpset(
      graph.Builder().OpsetVersion("") > 0 ? graph.Builder().OpsetVersion("") : 0)));
  rt.Put(cast.input()[0].value(), core::runtime::TensorFromProto(reshaped));
  core::runtime::RunNode(cast, rt);
  const auto &result = rt.tensors().at(cast.output()[0].value());
  TensorProto folded;
  folded.set_data_type(result.data_type);
  for (int64_t dim : result.shape) {
    folded.add_dims(dim);
  }
  folded.set_raw_data(result.bytes(), result.size_bytes());
  auto &builder = graph.Builder();
  const std::string base = cast.output()[0].value() + "_folded";
  std::string name = base;
  for (int suffix = 0; builder.HasName(name); ++suffix) {
    name = base + "_" + std::to_string(suffix);
  }
  folded.set_name(name);
  builder.MakeInitializer(folded);
  NodeProto replacement = add;
  for (auto &input : *replacement.mutable_input()) {
    if (input.value() == cast.output()[0].value()) {
      input = name;
    }
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
