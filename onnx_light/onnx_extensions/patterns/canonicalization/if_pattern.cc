// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/if_pattern.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {
namespace {

using core::builder::BuilderError;
using core::builder::GraphBuilder;
using Bindings = std::unordered_map<std::string, std::size_t>;
using Names = std::unordered_map<std::string, std::string>;

constexpr char kGraphReferenceAliasMarker[] = "_ref__onnx_light_graph_reference_";

bool SameTensor(const TensorProto &left, const TensorProto &right) {
  for (const auto *tensor : {&left, &right})
    if (tensor->has_segment() ||
        (!tensor->HasDeferredWeight() &&
         (tensor->data_location() == TensorProto::EXTERNAL || !tensor->external_data().empty())))
      return false;
  if (left.data_type() != right.data_type() || left.dims_size() != right.dims_size() ||
      !std::equal(left.dims().begin(), left.dims().end(), right.dims().begin()))
    return false;
  const auto has_inline_payload = [](const TensorProto &tensor) {
    return !tensor.raw_data().empty() || !tensor.float_data().empty() ||
           !tensor.int32_data().empty() || !tensor.int64_data().empty() ||
           !tensor.uint64_data().empty() || !tensor.double_data().empty() ||
           !tensor.string_data().empty();
  };
  if (left.HasDeferredWeight() && right.HasDeferredWeight() && !has_inline_payload(left) &&
      !has_inline_payload(right) && HaveSameDeferredWeightSource(left, right))
    return true;
  TensorProto a_copy, b_copy;
  const TensorProto *a_proto = &left;
  const TensorProto *b_proto = &right;
  if (left.HasDeferredWeight() || right.HasDeferredWeight()) {
    a_copy = utils::CopyWithSharedTensorStorage(left);
    b_copy = utils::CopyWithSharedTensorStorage(right);
    if (a_copy.HasDeferredWeight())
      a_copy.LoadDeferredWeight();
    if (b_copy.HasDeferredWeight())
      b_copy.LoadDeferredWeight();
    for (auto *tensor : {&a_copy, &b_copy}) {
      tensor->clr_external_data();
      tensor->set_data_location(TensorProto::DEFAULT);
    }
    a_proto = &a_copy;
    b_proto = &b_copy;
  }
  VerifyTensor(*a_proto);
  VerifyTensor(*b_proto);
  if (left.data_type() == TensorProto::STRING)
    return a_proto->string_data_size() == b_proto->string_data_size() &&
           std::equal(a_proto->string_data().begin(), a_proto->string_data().end(),
                      b_proto->string_data().begin());
  const auto a = core::runtime::TensorFromProto(*a_proto);
  const auto b = core::runtime::TensorFromProto(*b_proto);
  return a.size_bytes() == b.size_bytes() &&
         (a.size_bytes() == 0 || std::memcmp(a.bytes(), b.bytes(), a.size_bytes()) == 0);
}

bool SameValue(const std::string &left, const std::string &right, const Bindings &a,
               const Bindings &b) {
  const auto first = a.find(left), second = b.find(right);
  if (first == a.end() || second == b.end())
    return first == a.end() && second == b.end() && left == right;
  return first->second == second->second;
}

bool SameSparseTensor(const SparseTensorProto &left, const SparseTensorProto &right) {
  return left.dims_size() == right.dims_size() &&
         std::equal(left.dims().begin(), left.dims().end(), right.dims().begin()) &&
         SameTensor(left.values(), right.values()) && SameTensor(left.indices(), right.indices());
}

bool SameGraph(const GraphProto &left, const GraphProto &right, Bindings a, Bindings b,
               std::size_t next = 0, std::size_t depth = 0);

bool SameAttribute(const AttributeProto &left, const AttributeProto &right, const Bindings &a,
                   const Bindings &b, std::size_t next, std::size_t depth) {
  if (left.type() != right.type() || !left.ref_attr_name().empty() ||
      !right.ref_attr_name().empty() || left.has_g() != right.has_g() ||
      left.graphs_size() != right.graphs_size() || left.has_t() != right.has_t() ||
      left.tensors_size() != right.tensors_size() ||
      left.has_sparse_tensor() != right.has_sparse_tensor() ||
      left.sparse_tensors_size() != right.sparse_tensors_size())
    return false;
  if (left.has_g() && !SameGraph(left.g(), right.g(), a, b, next, depth + 1))
    return false;
  for (int i = 0; i < left.graphs_size(); ++i)
    if (!SameGraph(left.graphs(i), right.graphs(i), a, b, next, depth + 1))
      return false;
  if (left.has_t() && !SameTensor(left.t(), right.t()))
    return false;
  for (int i = 0; i < left.tensors_size(); ++i)
    if (!SameTensor(left.tensors(i), right.tensors(i)))
      return false;
  if (left.has_sparse_tensor() && !SameSparseTensor(left.sparse_tensor(), right.sparse_tensor()))
    return false;
  for (int i = 0; i < left.sparse_tensors_size(); ++i)
    if (!SameSparseTensor(left.sparse_tensors(i), right.sparse_tensors(i)))
      return false;
  auto first = utils::CopyWithSharedTensorStorage(left);
  auto second = utils::CopyWithSharedTensorStorage(right);
  for (auto *attribute : {&first, &second}) {
    attribute->clear_doc_string();
    attribute->clear_g();
    attribute->clr_graphs();
    attribute->clear_t();
    attribute->clr_tensors();
    attribute->clear_sparse_tensor();
    attribute->clr_sparse_tensors();
  }
  return first.Equals(second);
}

bool SameGraph(const GraphProto &left, const GraphProto &right, Bindings a, Bindings b,
               std::size_t next, std::size_t depth) {
  if (depth >= 64 || left.input_size() != right.input_size() ||
      left.output_size() != right.output_size() || left.node_size() != right.node_size() ||
      left.initializer_size() != right.initializer_size())
    return false;
  for (const auto *scope : {&left, &right})
    if (!scope->sparse_initializer().empty() || !scope->extended_initializer().empty() ||
        !scope->quantization_annotation().empty() || !scope->persistent_bindings().empty())
      return false;
  for (const auto *scope : {&left, &right})
    for (const auto *declarations : {&scope->input(), &scope->output(), &scope->value_info()})
      for (const auto &value : *declarations)
        if (!value.metadata_props().empty() ||
            (value.has_type() && !value.type().has_tensor_type()))
          return false;
  std::unordered_set<std::string> defined_a, defined_b;
  const auto bind = [&](const std::string &x, const std::string &y) {
    if (x.empty() || y.empty())
      return x.empty() && y.empty();
    if (!defined_a.insert(x).second || !defined_b.insert(y).second)
      return false;
    a[x] = next;
    b[y] = next++;
    return true;
  };
  for (int i = 0; i < left.input_size(); ++i) {
    if (!left.input(i).type().Equals(right.input(i).type()) ||
        !bind(left.input(i).name().value(), right.input(i).name().value()))
      return false;
  }
  for (int i = 0; i < left.initializer_size(); ++i) {
    for (const auto *scope : {&left, &right})
      for (const auto &input : scope->input())
        if (input.name() == scope->initializer(i).name())
          return false;
    if (!SameTensor(left.initializer(i), right.initializer(i)) ||
        !bind(left.initializer(i).name().value(), right.initializer(i).name().value()))
      return false;
  }
  for (int i = 0; i < left.node_size(); ++i) {
    const auto &x = left.node(i);
    const auto &y = right.node(i);
    if (x.op_type() != y.op_type() || NormaliseDomain(x.domain()) != NormaliseDomain(y.domain()) ||
        x.overload() != y.overload() || x.input_size() != y.input_size() ||
        x.output_size() != y.output_size() || x.attribute_size() != y.attribute_size() ||
        !x.device_configurations().empty() || !y.device_configurations().empty())
      return false;
    for (int j = 0; j < x.input_size(); ++j)
      if (!SameValue(x.input(j).value(), y.input(j).value(), a, b))
        return false;
    std::unordered_set<std::string> attributes;
    for (const auto &attribute : x.attribute()) {
      if (!attributes.insert(attribute.name().value()).second)
        return false;
      const auto *other = FindAttribute(y, attribute.name().c_str());
      if (other == nullptr || !SameAttribute(attribute, *other, a, b, next, depth))
        return false;
      if ((attribute.has_g() || !attribute.graphs().empty()) &&
          (FindAttribute(x, (attribute.name().value() + "_ref").c_str()) != nullptr ||
           FindAttribute(y, (attribute.name().value() + "_ref").c_str()) != nullptr))
        return false;
    }
    for (int j = 0; j < x.output_size(); ++j)
      if (!bind(x.output(j).value(), y.output(j).value()))
        return false;
  }
  for (int i = 0; i < left.output_size(); ++i)
    if (left.output(i).name().empty() || right.output(i).name().empty() ||
        left.output(i).has_type() != right.output(i).has_type() ||
        !left.output(i).type().Equals(right.output(i).type()) ||
        !SameValue(left.output(i).name().value(), right.output(i).name().value(), a, b))
      return false;
  return true;
}

const GraphBuilder *Branch(const GraphBuilder &builder, const NodeProto &node, const char *name) {
  const auto *attribute = FindAttribute(node, name);
  if (attribute == nullptr || attribute->type() != AttributeProto::STRING ||
      !builder.HasSubgraph(attribute->s().value()))
    return nullptr;
  return &builder.Subgraph(attribute->s().value());
}

std::optional<GraphProto> BranchGraph(const GraphBuilder &builder, const NodeProto &node,
                                      const char *name) {
  const auto *attribute = FindAttribute(node, name);
  if (attribute != nullptr && attribute->type() == AttributeProto::GRAPH && attribute->has_g())
    return utils::CopyWithSharedTensorStorage(attribute->g());
  const auto *branch = Branch(builder, node, (std::string(name) + "_ref").c_str());
  if (branch != nullptr)
    return branch->BuildGraph();
  return std::nullopt;
}

bool CompatibleDevices(const GraphBuilder &builder, core::symbolic::Device target) {
  if (builder.device() != core::symbolic::Device::kUndefined && builder.device() != target)
    return false;
  for (const auto &child : builder.Subgraphs())
    if (!CompatibleDevices(*child, target))
      return false;
  return true;
}

std::string Rename(const std::string &name, const Names &names) {
  const auto found = names.find(name);
  return found == names.end() ? name : found->second;
}

bool HasNestedName(const GraphBuilder &builder, const std::string &name) {
  if (builder.HasName(name))
    return true;
  for (const auto &child : builder.Subgraphs())
    if (HasNestedName(*child, name))
      return true;
  return false;
}

void RenameGraph(GraphProto &scope, GraphBuilder &builder, const std::string &prefix,
                 std::size_t &index, Names names = {}, bool reserve = true,
                 const NodeProto *call = nullptr) {
  const auto bind = [&](const std::string &name) {
    if (name.empty())
      return;
    std::string fresh;
    do {
      fresh = prefix + "_value_" + std::to_string(index++);
    } while (HasNestedName(builder, fresh));
    if (reserve)
      builder.ReserveName(fresh);
    names[name] = std::move(fresh);
  };
  for (const auto &input : scope.input())
    bind(input.name().value());
  for (const auto &tensor : scope.initializer())
    bind(tensor.name().value());
  for (const auto &node : scope.node())
    for (const auto &output : node.output())
      bind(output.value());
  if (call != nullptr) {
    std::unordered_set<std::string> mapped;
    for (int i = 0; i < scope.output_size(); ++i) {
      const auto name = scope.output(i).name().value();
      if (names.contains(name) && mapped.insert(name).second)
        names[name] = call->output(i).value();
    }
  }
  for (auto &input : scope.ref_input())
    input.set_name(Rename(input.name().value(), names));
  for (auto &tensor : scope.ref_initializer())
    tensor.set_name(Rename(tensor.name().value(), names));
  for (auto &info : scope.ref_value_info())
    info.set_name(Rename(info.name().value(), names));
  for (auto &output : scope.ref_output())
    output.set_name(Rename(output.name().value(), names));
  for (auto &node : scope.ref_node()) {
    node.set_name(prefix + "_node_" + std::to_string(index++));
    for (auto &input : node.ref_input())
      input = Rename(input.value(), names);
    for (auto &output : node.ref_output())
      output = Rename(output.value(), names);
    for (auto &attribute : node.ref_attribute()) {
      if (attribute.has_g())
        RenameGraph(attribute.ref_g(), builder, prefix, index, names, false);
      for (auto &nested : attribute.ref_graphs())
        RenameGraph(nested, builder, prefix, index, names, false);
    }
  }
}

void ImportGraphs(NodeProto &node, GraphBuilder &builder) {
  std::unordered_set<std::string> occupied_names;
  for (const auto &attribute : node.attribute())
    occupied_names.insert(attribute.name().value());
  std::size_t alias_index = 0;
  const auto is_declared_attribute = [&](const std::string &name) {
    const auto *schema = builder.FindSchema(node.op_type(), node.domain(), node.overload());
    if (schema != nullptr) {
      for (const auto &declared : schema->attributes())
        if (declared.name == name)
          return true;
    } else {
      const auto *function =
          builder.Shapes().GetLocalFunction(core::shapes::ShapesContext::LocalFunctionKey(
              node.domain(), node.op_type(), node.overload()));
      if (function != nullptr) {
        for (const auto &declared : function->attribute())
          if (declared == name)
            return true;
        for (const auto &declared : function->attribute_proto())
          if (declared.name() == name)
            return true;
      }
    }
    return false;
  };
  const auto reference_name_for = [&](const AttributeProto &attribute) {
    const auto &name = attribute.name().value();
    const std::string direct_name = name + "_ref";
    if (name.find(kGraphReferenceAliasMarker) == std::string::npos &&
        !is_declared_attribute(direct_name) && occupied_names.count(direct_name) == 0) {
      occupied_names.insert(direct_name);
      return direct_name;
    }
    std::string alias;
    do {
      alias = name + kGraphReferenceAliasMarker + std::to_string(alias_index++) + "_ref";
    } while (is_declared_attribute(alias) || occupied_names.count(alias) != 0);
    occupied_names.insert(alias);
    return alias;
  };
  for (auto &attribute : node.ref_attribute()) {
    if (!attribute.has_g() && attribute.graphs().empty())
      continue;
    const auto import = [&](const GraphProto &body) {
      std::string name = "IfIdenticalBranches_graph";
      for (std::size_t i = 0; builder.HasName(name); ++i)
        name = "IfIdenticalBranches_graph_" + std::to_string(i);
      builder.MakeSubgraph(name, body).set_device(builder.device());
      return name;
    };
    AttributeProto reference;
    reference.set_name(reference_name_for(attribute));
    if (attribute.has_g()) {
      reference.set_type(AttributeProto::STRING);
      reference.set_s(import(attribute.g()));
    } else {
      reference.set_type(AttributeProto::STRINGS);
      for (const auto &body : attribute.graphs())
        reference.add_strings(import(body));
    }
    attribute = std::move(reference);
  }
}

bool CanInlineInitializers(const GraphBuilder &builder, const GraphProto &graph) {
  if (graph.initializer().empty())
    return true;
  if (builder.OpsetVersion("") <= 0)
    return false;
  const auto *schema = builder.FindSchema("Constant");
  if (schema == nullptr || schema->outputs().empty())
    return false;
  const auto &constraints = schema->type_constraints();
  const auto constraint =
      std::find_if(constraints.begin(), constraints.end(), [&](const auto &entry) {
        return entry.type_param_str == schema->outputs().front().type;
      });
  if (constraint == constraints.end())
    return false;
  return std::all_of(
      graph.initializer().begin(), graph.initializer().end(), [&](const auto &initializer) {
        const auto type = core::symbolic::DataTypeToTensorType(initializer.data_type());
        return std::find(constraint->allowed_type_strs.begin(), constraint->allowed_type_strs.end(),
                         type) != constraint->allowed_type_strs.end();
      });
}

} // namespace

std::set<std::string> IfIdenticalBranchesPattern::FastOpType() const { return {"If"}; }

core::builder::MatchResult IfIdenticalBranchesPattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (candidate.op_type() != "If" || NormaliseDomain(candidate.domain()) != kDefaultOnnxDomain ||
      !candidate.overload().empty() || candidate.input_size() != 1 || candidate.input(0).empty() ||
      candidate.output_size() == 0 || candidate.attribute_size() != 2 ||
      !candidate.device_configurations().empty() || !candidate.metadata_props().empty())
    return NoMatch(candidate, "candidate is not an unconfigured default-domain If");
  for (const auto &input : core::graph::CollectNodeInputs(candidate))
    if (!graph.Builder().HasName(input))
      return NoMatch(candidate, "If refers to an unknown outer value");
  std::unordered_set<std::string> outputs;
  for (const auto &output : candidate.output())
    if (output.empty() || !outputs.insert(output.value()).second ||
        (graph.Builder().HasName(output.value()) && graph.NodeBefore(output.value()) != &candidate))
      return NoMatch(candidate, "If outputs are not distinct fresh results");
  const auto *first = Branch(graph.Builder(), candidate, "then_branch_ref");
  const auto *second = Branch(graph.Builder(), candidate, "else_branch_ref");
  const auto a = BranchGraph(graph.Builder(), candidate, "then_branch");
  const auto b = BranchGraph(graph.Builder(), candidate, "else_branch");
  if (!a || !b || !a->input().empty() || !b->input().empty() ||
      a->output().size() != candidate.output().size() ||
      (first != nullptr && !CompatibleDevices(*first, graph.Builder().device())) ||
      (second != nullptr && !CompatibleDevices(*second, graph.Builder().device())) ||
      !CanInlineInitializers(graph.Builder(), *a) || !SameGraph(*a, *b, {}, {}))
    return NoMatch(candidate, "If branches are not equivalent");
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
IfIdenticalBranchesPattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr)
    throw BuilderError("IfIdenticalBranchesPattern::Apply expects one If node.");
  const auto verified = Match(graph, *nodes[0]);
  if (verified.pattern != this || verified.nodes != nodes)
    throw BuilderError("IfIdenticalBranchesPattern::Apply received an inconsistent match.");
  auto &builder = graph.Builder();
  auto body = *BranchGraph(builder, *nodes[0], "then_branch");
  const auto prefix = builder.UniqueName("IfIdenticalBranches");
  std::size_t index = 0;
  RenameGraph(body, builder, prefix, index, {}, true, nodes[0]);
  utils::RepeatedProtoField<NodeProto> replacements;
  for (const auto &tensor : body.initializer()) {
    auto constant = MakeNode("Constant", {}, {tensor.name().value()});
    constant.set_doc_string(nodes[0]->doc_string().value());
    auto &value = *constant.add_attribute();
    value.set_name("value");
    value.set_type(AttributeProto::TENSOR);
    *value.mutable_t() = utils::CopyWithSharedTensorStorage(tensor);
    replacements.push_back(std::move(constant));
  }
  for (auto &node : body.ref_node()) {
    if (!node.has_doc_string() && nodes[0]->has_doc_string())
      node.set_doc_string(nodes[0]->doc_string().value());
    ImportGraphs(node, builder);
    replacements.push_back(std::move(node));
  }
  for (int i = 0; i < body.output_size(); ++i) {
    if (body.output(i).name().value() == nodes[0]->output(i).value())
      continue;
    auto identity =
        MakeNode("Identity", {body.output(i).name().value()}, {nodes[0]->output(i).value()});
    identity.set_doc_string(nodes[0]->doc_string().value());
    identity.ref_metadata_props() = nodes[0]->metadata_props();
    replacements.push_back(std::move(identity));
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
