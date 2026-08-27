// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;
using core::symbolic::TensorType;

constexpr const char *kMlDomain = "ai.onnx.ml";

struct LegacyNode {
  int64_t tree_id = 0;
  int64_t node_id = 0;
  int64_t feature_id = 0;
  float value = 0.0f;
  uint8_t mode = 0;
  int64_t true_node_id = 0;
  int64_t false_node_id = 0;
  int64_t missing_tracks_true = 0;
  bool is_leaf = false;
};

struct ParsedEnsemble {
  bool classifier = false;
  int64_t n_targets = 0;
  std::vector<LegacyNode> nodes;
  std::vector<int64_t> tree_ids;
  std::map<std::pair<int64_t, int64_t>, std::size_t> node_indices;
  std::map<int64_t, std::pair<int64_t, int64_t>> roots;
  std::map<std::pair<std::pair<int64_t, int64_t>, int64_t>, float> weights;
  std::vector<float> base_values;
  std::vector<int64_t> labels_int64;
  std::vector<std::string> labels_strings;
  bool has_missing_tracks_true = false;
};

const AttributeProto *UniqueAttribute(const NodeProto &node, const char *name, bool &valid) {
  const AttributeProto *result = nullptr;
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attribute = node.attribute(i);
    if (attribute.name().value() != name) {
      continue;
    }
    if (result != nullptr) {
      valid = false;
      return nullptr;
    }
    result = &attribute;
  }
  return result;
}

bool ReadInts(const NodeProto &node, const char *name, bool required,
              std::vector<int64_t> &values) {
  bool valid = true;
  const AttributeProto *attribute = UniqueAttribute(node, name, valid);
  if (!valid || attribute == nullptr) {
    return valid && !required;
  }
  if (attribute->type() != AttributeProto::AttributeType::INTS) {
    return false;
  }
  values.assign(attribute->ints().begin(), attribute->ints().end());
  return true;
}

bool ReadFloats(const NodeProto &node, const char *name, bool required,
                std::vector<float> &values) {
  bool valid = true;
  const AttributeProto *attribute = UniqueAttribute(node, name, valid);
  if (!valid || attribute == nullptr) {
    return valid && !required;
  }
  if (attribute->type() != AttributeProto::AttributeType::FLOATS) {
    return false;
  }
  values.assign(attribute->floats().begin(), attribute->floats().end());
  return true;
}

bool ReadStrings(const NodeProto &node, const char *name, bool required,
                 std::vector<std::string> &values) {
  bool valid = true;
  const AttributeProto *attribute = UniqueAttribute(node, name, valid);
  if (!valid || attribute == nullptr) {
    return valid && !required;
  }
  if (attribute->type() != AttributeProto::AttributeType::STRINGS) {
    return false;
  }
  values.assign(attribute->strings().begin(), attribute->strings().end());
  return true;
}

bool ReadInt(const NodeProto &node, const char *name, int64_t default_value, int64_t &value) {
  bool valid = true;
  const AttributeProto *attribute = UniqueAttribute(node, name, valid);
  if (!valid) {
    return false;
  }
  if (attribute == nullptr) {
    value = default_value;
    return true;
  }
  if (attribute->type() != AttributeProto::AttributeType::INT) {
    return false;
  }
  value = attribute->i();
  return true;
}

bool ReadString(const NodeProto &node, const char *name, const char *default_value,
                std::string &value) {
  bool valid = true;
  const AttributeProto *attribute = UniqueAttribute(node, name, valid);
  if (!valid) {
    return false;
  }
  if (attribute == nullptr) {
    value = default_value;
    return true;
  }
  if (attribute->type() != AttributeProto::AttributeType::STRING) {
    return false;
  }
  value = attribute->s();
  return true;
}

bool HasAttribute(const NodeProto &node, const char *name) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    if (node.attribute(i).name().value() == name) {
      return true;
    }
  }
  return false;
}

bool ModeCode(const std::string &mode, uint8_t &code, bool &is_leaf) {
  static const std::map<std::string, uint8_t> modes = {
      {"BRANCH_LEQ", 0}, {"BRANCH_LT", 1}, {"BRANCH_GTE", 2},
      {"BRANCH_GT", 3},  {"BRANCH_EQ", 4}, {"BRANCH_NEQ", 5},
  };
  if (mode == "LEAF") {
    is_leaf = true;
    code = 0;
    return true;
  }
  const auto found = modes.find(mode);
  if (found == modes.end()) {
    return false;
  }
  is_leaf = false;
  code = found->second;
  return true;
}

bool HasUsableFeatureForDegenerateTree(core::builder::GraphGraph &graph, const std::string &input) {
  if (!graph.HasShape(input)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(input).Shape();
  if (shape.Rank() != 1 && shape.Rank() != 2) {
    return false;
  }
  const std::size_t feature_axis = shape.Rank() - 1;
  return shape[feature_axis].IsInt() && shape[feature_axis].AsInt() > 0;
}

bool ValidateTreeStructure(const std::string &input, core::builder::GraphGraph &graph,
                           ParsedEnsemble &parsed) {
  const core::symbolic::SymShape &input_shape = graph.GetShape(input).Shape();
  if (input_shape.Rank() != 2) {
    return false;
  }
  const core::symbolic::SymDim &feature_count = input_shape[input_shape.Rank() - 1];
  if (feature_count.IsInt() && feature_count.AsInt() <= 0) {
    return false;
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> incoming;
  for (const auto &[key, index] : parsed.node_indices) {
    (void)index;
    incoming.emplace(key, 0);
  }

  for (const LegacyNode &node : parsed.nodes) {
    if (node.tree_id < 0 || node.node_id < 0 || (!node.is_leaf && node.feature_id < 0) ||
        (!node.is_leaf && feature_count.IsInt() && node.feature_id >= feature_count.AsInt())) {
      return false;
    }
    if (node.missing_tracks_true != 0 && node.missing_tracks_true != 1) {
      return false;
    }
    if (node.is_leaf) {
      continue;
    }
    for (int64_t child_id : {node.true_node_id, node.false_node_id}) {
      const std::pair<int64_t, int64_t> child{node.tree_id, child_id};
      if (!parsed.node_indices.contains(child)) {
        return false;
      }
      if (++incoming[child] != 1) {
        return false;
      }
    }
  }

  for (int64_t tree_id : parsed.tree_ids) {
    std::vector<std::pair<int64_t, int64_t>> roots;
    std::size_t tree_size = 0;
    for (const auto &[key, count] : incoming) {
      if (key.first != tree_id) {
        continue;
      }
      ++tree_size;
      if (count == 0) {
        roots.push_back(key);
      }
    }
    if (roots.size() != 1) {
      return false;
    }
    const LegacyNode &root = parsed.nodes[parsed.node_indices.at(roots[0])];
    if (root.is_leaf && (tree_size != 1 || !HasUsableFeatureForDegenerateTree(graph, input))) {
      return false;
    }

    std::set<std::pair<int64_t, int64_t>> visited;
    std::vector<std::pair<int64_t, int64_t>> pending = {roots[0]};
    while (!pending.empty()) {
      const std::pair<int64_t, int64_t> key = pending.back();
      pending.pop_back();
      if (!visited.insert(key).second) {
        return false;
      }
      const LegacyNode &node = parsed.nodes[parsed.node_indices.at(key)];
      if (!node.is_leaf) {
        pending.emplace_back(tree_id, node.true_node_id);
        pending.emplace_back(tree_id, node.false_node_id);
      }
    }
    if (visited.size() != tree_size) {
      return false;
    }
    parsed.roots.emplace(tree_id, roots[0]);
  }
  return true;
}

bool ParseLegacyEnsemble(core::builder::GraphGraph &graph, const NodeProto &node,
                         ParsedEnsemble &parsed) {
  const std::string op_type = node.op_type().value();
  parsed.classifier = op_type == "TreeEnsembleClassifier";
  const int ml_opset = graph.Builder().OpsetVersion(kMlDomain);
  const int default_opset = graph.Builder().OpsetVersion("");
  if (ml_opset < 5 || (parsed.classifier && default_opset < 1) ||
      NormaliseDomain(node.domain().value()) != kMlDomain ||
      (op_type != "TreeEnsembleRegressor" && !parsed.classifier) || node.input_size() != 1 ||
      node.output_size() != (parsed.classifier ? 2 : 1) || node.input()[0].value().empty() ||
      !graph.HasType(node.input()[0].value()) || !graph.HasShape(node.input()[0].value()) ||
      graph.GetType(node.input()[0].value()) != TensorType::kFloat) {
    return false;
  }
  for (int i = 0; i < node.output_size(); ++i) {
    if (node.output()[i].value().empty()) {
      return false;
    }
  }

  std::string post_transform;
  if (!ReadString(node, "post_transform", "NONE", post_transform) || post_transform != "NONE" ||
      HasAttribute(node, "nodes_values_as_tensor") || HasAttribute(node, "base_values_as_tensor")) {
    return false;
  }
  if (!parsed.classifier) {
    std::string aggregate_function;
    if (!ReadString(node, "aggregate_function", "SUM", aggregate_function) ||
        aggregate_function != "SUM" || !ReadInt(node, "n_targets", 1, parsed.n_targets) ||
        parsed.n_targets < 1 || HasAttribute(node, "target_weights_as_tensor")) {
      return false;
    }
  } else {
    std::string aggregate_function;
    if (!ReadString(node, "aggregate_function", "SUM", aggregate_function) ||
        aggregate_function != "SUM" || HasAttribute(node, "class_weights_as_tensor")) {
      return false;
    }
  }

  std::vector<int64_t> nodes_treeids;
  std::vector<int64_t> nodes_nodeids;
  std::vector<int64_t> nodes_featureids;
  std::vector<float> nodes_values;
  std::vector<std::string> nodes_modes;
  std::vector<int64_t> nodes_truenodeids;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_missing;
  if (!ReadInts(node, "nodes_treeids", true, nodes_treeids) ||
      !ReadInts(node, "nodes_nodeids", true, nodes_nodeids) ||
      !ReadInts(node, "nodes_featureids", true, nodes_featureids) ||
      !ReadFloats(node, "nodes_values", true, nodes_values) ||
      !ReadStrings(node, "nodes_modes", true, nodes_modes) ||
      !ReadInts(node, "nodes_truenodeids", true, nodes_truenodeids) ||
      !ReadInts(node, "nodes_falsenodeids", true, nodes_falsenodeids) ||
      !ReadInts(node, "nodes_missing_value_tracks_true", false, nodes_missing)) {
    return false;
  }
  const std::size_t node_count = nodes_treeids.size();
  if (node_count == 0 || nodes_nodeids.size() != node_count ||
      nodes_featureids.size() != node_count || nodes_values.size() != node_count ||
      nodes_modes.size() != node_count || nodes_truenodeids.size() != node_count ||
      nodes_falsenodeids.size() != node_count ||
      (!nodes_missing.empty() && nodes_missing.size() != node_count)) {
    return false;
  }
  parsed.has_missing_tracks_true = !nodes_missing.empty();
  std::set<int64_t> seen_tree_ids;
  parsed.nodes.reserve(node_count);
  for (std::size_t i = 0; i < node_count; ++i) {
    LegacyNode legacy;
    legacy.tree_id = nodes_treeids[i];
    legacy.node_id = nodes_nodeids[i];
    legacy.feature_id = nodes_featureids[i];
    legacy.value = nodes_values[i];
    legacy.true_node_id = nodes_truenodeids[i];
    legacy.false_node_id = nodes_falsenodeids[i];
    legacy.missing_tracks_true = nodes_missing.empty() ? 0 : nodes_missing[i];
    if (!ModeCode(nodes_modes[i], legacy.mode, legacy.is_leaf)) {
      return false;
    }
    const std::pair<int64_t, int64_t> key{legacy.tree_id, legacy.node_id};
    if (!parsed.node_indices.emplace(key, i).second) {
      return false;
    }
    if (seen_tree_ids.insert(legacy.tree_id).second) {
      parsed.tree_ids.push_back(legacy.tree_id);
    }
    parsed.nodes.push_back(legacy);
  }
  if (!ValidateTreeStructure(node.input()[0].value(), graph, parsed)) {
    return false;
  }

  std::vector<int64_t> weight_treeids;
  std::vector<int64_t> weight_nodeids;
  std::vector<int64_t> target_ids;
  std::vector<float> weights;
  const char *treeids_name = parsed.classifier ? "class_treeids" : "target_treeids";
  const char *nodeids_name = parsed.classifier ? "class_nodeids" : "target_nodeids";
  const char *ids_name = parsed.classifier ? "class_ids" : "target_ids";
  const char *weights_name = parsed.classifier ? "class_weights" : "target_weights";
  if (!ReadInts(node, treeids_name, true, weight_treeids) ||
      !ReadInts(node, nodeids_name, true, weight_nodeids) ||
      !ReadInts(node, ids_name, true, target_ids) ||
      !ReadFloats(node, weights_name, true, weights)) {
    return false;
  }

  if (parsed.classifier) {
    bool int_labels_valid = true;
    bool string_labels_valid = true;
    const AttributeProto *int_labels =
        UniqueAttribute(node, "classlabels_int64s", int_labels_valid);
    const AttributeProto *string_labels =
        UniqueAttribute(node, "classlabels_strings", string_labels_valid);
    if (!int_labels_valid || !string_labels_valid ||
        (int_labels == nullptr) == (string_labels == nullptr)) {
      return false;
    }
    if (int_labels != nullptr) {
      if (int_labels->type() != AttributeProto::AttributeType::INTS || int_labels->ints().empty()) {
        return false;
      }
      parsed.labels_int64.assign(int_labels->ints().begin(), int_labels->ints().end());
      parsed.n_targets = static_cast<int64_t>(parsed.labels_int64.size());
    } else {
      if (string_labels->type() != AttributeProto::AttributeType::STRINGS ||
          string_labels->strings().empty()) {
        return false;
      }
      parsed.labels_strings.assign(string_labels->strings().begin(),
                                   string_labels->strings().end());
      parsed.n_targets = static_cast<int64_t>(parsed.labels_strings.size());
    }
    if (parsed.n_targets < 2) {
      return false;
    }
  }

  const std::size_t weight_count = weight_treeids.size();
  if (weight_nodeids.size() != weight_count || target_ids.size() != weight_count ||
      weights.size() != weight_count) {
    return false;
  }
  std::set<int64_t> classifier_target_ids;
  for (std::size_t i = 0; i < weight_count; ++i) {
    const std::pair<int64_t, int64_t> key{weight_treeids[i], weight_nodeids[i]};
    const auto found = parsed.node_indices.find(key);
    if (found == parsed.node_indices.end() || !parsed.nodes[found->second].is_leaf ||
        target_ids[i] < 0 || target_ids[i] >= parsed.n_targets) {
      return false;
    }
    if (parsed.classifier) {
      classifier_target_ids.insert(target_ids[i]);
    }
    parsed.weights[{key, target_ids[i]}] += weights[i];
  }
  if (parsed.classifier && classifier_target_ids.size() < 2) {
    return false;
  }

  if (!ReadFloats(node, "base_values", false, parsed.base_values) ||
      (!parsed.base_values.empty() &&
       static_cast<int64_t>(parsed.base_values.size()) != parsed.n_targets) ||
      (!parsed.base_values.empty() && default_opset < 7)) {
    return false;
  }
  return true;
}

void AddTensorAttribute(NodeProto &node, const char *name, const TensorProto &tensor) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->mutable_t() = tensor;
}

void AddUint8TensorAttribute(NodeProto &node, const char *name,
                             const std::vector<uint8_t> &values) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::DataType::UINT8);
  tensor.add_dims(static_cast<int64_t>(values.size()));
  for (uint8_t value : values) {
    tensor.add_int32_data(static_cast<int32_t>(value));
  }
  AddTensorAttribute(node, name, tensor);
}

std::string InitializerName(core::builder::GraphBuilder &builder, const std::string &prefix) {
  const std::string unique_prefix = builder.UniqueName(prefix);
  std::string name = unique_prefix + "_initializer";
  for (int suffix = 0; builder.HasName(name); ++suffix) {
    name = unique_prefix + "_initializer_" + std::to_string(suffix);
  }
  return name;
}

NodeProto BuildTreeEnsemble(const NodeProto &source, const ParsedEnsemble &parsed,
                            const std::string &output) {
  std::vector<int64_t> tree_roots;
  std::vector<int64_t> nodes_featureids;
  std::vector<float> nodes_splits;
  std::vector<uint8_t> nodes_modes;
  std::vector<int64_t> nodes_truenodeids;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_trueleafs;
  std::vector<int64_t> nodes_falseleafs;
  std::vector<int64_t> nodes_missing;
  std::vector<int64_t> leaf_targetids;
  std::vector<float> leaf_weights;

  for (int64_t tree_id : parsed.tree_ids) {
    std::vector<std::pair<int64_t, int64_t>> tree_nodes;
    std::vector<std::pair<int64_t, int64_t>> tree_leaves;
    for (const LegacyNode &node : parsed.nodes) {
      if (node.tree_id != tree_id) {
        continue;
      }
      const std::pair<int64_t, int64_t> key{tree_id, node.node_id};
      (node.is_leaf ? tree_leaves : tree_nodes).push_back(key);
    }

    for (int64_t target_id = 0; target_id < parsed.n_targets; ++target_id) {
      std::map<std::pair<int64_t, int64_t>, int64_t> new_node_ids;
      std::map<std::pair<int64_t, int64_t>, int64_t> new_leaf_ids;
      for (const auto &key : tree_nodes) {
        new_node_ids.emplace(key, static_cast<int64_t>(nodes_featureids.size()));
        nodes_featureids.push_back(0);
        nodes_splits.push_back(0.0f);
        nodes_modes.push_back(0);
        nodes_truenodeids.push_back(0);
        nodes_falsenodeids.push_back(0);
        nodes_trueleafs.push_back(0);
        nodes_falseleafs.push_back(0);
        if (parsed.has_missing_tracks_true) {
          nodes_missing.push_back(0);
        }
      }
      for (const auto &key : tree_leaves) {
        new_leaf_ids.emplace(key, static_cast<int64_t>(leaf_targetids.size()));
        leaf_targetids.push_back(target_id);
        const auto weight = parsed.weights.find({key, target_id});
        leaf_weights.push_back(weight == parsed.weights.end() ? 0.0f : weight->second);
      }

      const std::pair<int64_t, int64_t> root = parsed.roots.at(tree_id);
      const LegacyNode &root_node = parsed.nodes[parsed.node_indices.at(root)];
      if (root_node.is_leaf) {
        const int64_t dummy_id = static_cast<int64_t>(nodes_featureids.size());
        tree_roots.push_back(dummy_id);
        nodes_featureids.push_back(0);
        nodes_splits.push_back(0.0f);
        nodes_modes.push_back(0);
        nodes_truenodeids.push_back(new_leaf_ids.at(root));
        nodes_falsenodeids.push_back(new_leaf_ids.at(root));
        nodes_trueleafs.push_back(1);
        nodes_falseleafs.push_back(1);
        if (parsed.has_missing_tracks_true) {
          nodes_missing.push_back(0);
        }
        continue;
      }
      tree_roots.push_back(new_node_ids.at(root));

      for (const auto &key : tree_nodes) {
        const LegacyNode &legacy = parsed.nodes[parsed.node_indices.at(key)];
        const std::size_t new_index = static_cast<std::size_t>(new_node_ids.at(key));
        nodes_featureids[new_index] = legacy.feature_id;
        nodes_splits[new_index] = legacy.value;
        nodes_modes[new_index] = legacy.mode;
        if (parsed.has_missing_tracks_true) {
          nodes_missing[new_index] = legacy.missing_tracks_true;
        }
        const std::pair<int64_t, int64_t> true_key{tree_id, legacy.true_node_id};
        const std::pair<int64_t, int64_t> false_key{tree_id, legacy.false_node_id};
        const LegacyNode &true_node = parsed.nodes[parsed.node_indices.at(true_key)];
        const LegacyNode &false_node = parsed.nodes[parsed.node_indices.at(false_key)];
        nodes_trueleafs[new_index] = true_node.is_leaf ? 1 : 0;
        nodes_falseleafs[new_index] = false_node.is_leaf ? 1 : 0;
        nodes_truenodeids[new_index] =
            true_node.is_leaf ? new_leaf_ids.at(true_key) : new_node_ids.at(true_key);
        nodes_falsenodeids[new_index] =
            false_node.is_leaf ? new_leaf_ids.at(false_key) : new_node_ids.at(false_key);
      }
    }
  }

  NodeProto replacement =
      MakeNode("TreeEnsemble", {source.input()[0].value()}, {output}, kMlDomain,
               source.name().value().empty() ? nullptr : source.name().value().c_str());
  replacement.set_doc_string(source.doc_string().value());
  AddAttribute(replacement, "tree_roots", tree_roots);
  AddAttribute(replacement, "nodes_featureids", nodes_featureids);
  AddTensorAttribute(
      replacement, "nodes_splits",
      MakeInitializer<float>("", {static_cast<int64_t>(nodes_splits.size())}, nodes_splits));
  AddUint8TensorAttribute(replacement, "nodes_modes", nodes_modes);
  AddAttribute(replacement, "nodes_truenodeids", nodes_truenodeids);
  AddAttribute(replacement, "nodes_falsenodeids", nodes_falsenodeids);
  AddAttribute(replacement, "nodes_trueleafs", nodes_trueleafs);
  AddAttribute(replacement, "nodes_falseleafs", nodes_falseleafs);
  if (parsed.has_missing_tracks_true) {
    AddAttribute(replacement, "nodes_missing_value_tracks_true", nodes_missing);
  }
  AddAttribute(replacement, "leaf_targetids", leaf_targetids);
  AddTensorAttribute(
      replacement, "leaf_weights",
      MakeInitializer<float>("", {static_cast<int64_t>(leaf_weights.size())}, leaf_weights));
  AddAttribute(replacement, "n_targets", parsed.n_targets);
  AddAttribute(replacement, "aggregate_function", static_cast<int64_t>(1));
  AddAttribute(replacement, "post_transform", static_cast<int64_t>(0));
  return replacement;
}

} // namespace

std::set<std::string> TreeEnsemblePattern::FastOpType() const {
  return {"TreeEnsembleClassifier", "TreeEnsembleRegressor"};
}

core::builder::MatchResult TreeEnsemblePattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  ParsedEnsemble parsed;
  if (!ParseLegacyEnsemble(graph, candidate, parsed)) {
    return NoMatch(candidate, "the classic tree ensemble is invalid or unsupported");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
TreeEnsemblePattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("TreeEnsemblePattern::Apply expects one classic tree ensemble node.");
  }
  const NodeProto &source = *nodes[0];
  ParsedEnsemble parsed;
  if (!ParseLegacyEnsemble(graph, source, parsed)) {
    throw BuilderError("TreeEnsemblePattern::Apply received an unsafe or inconsistent match.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  utils::RepeatedProtoField<NodeProto> replacements;

  const std::string scores_output = source.output()[parsed.classifier ? 1 : 0].value();
  const std::string tree_output =
      parsed.base_values.empty() ? scores_output : builder.UniqueName(scores_output + "_trees");
  replacements.push_back(BuildTreeEnsemble(source, parsed, tree_output));

  if (!parsed.base_values.empty()) {
    const std::string base_name = InitializerName(builder, source.name().value() + "_base_values");
    builder.MakeInitializer(
        MakeInitializer<float>(base_name.c_str(), {parsed.n_targets}, parsed.base_values));
    replacements.push_back(MakeNode("Add", {tree_output, base_name}, {scores_output}, "",
                                    (source.name().value() + "_base_values").c_str()));
  }

  if (parsed.classifier) {
    const std::string indices = builder.UniqueName(source.output()[0].value() + "_indices");
    NodeProto argmax = MakeNode("ArgMax", {scores_output}, {indices}, "",
                                (source.name().value() + "_argmax").c_str());
    AddAttribute(argmax, "axis", static_cast<int64_t>(1));
    AddAttribute(argmax, "keepdims", static_cast<int64_t>(0));
    replacements.push_back(std::move(argmax));

    if (parsed.labels_strings.empty()) {
      const std::string labels_name = InitializerName(builder, source.name().value() + "_labels");
      builder.MakeInitializer(
          MakeInitializer<int64_t>(labels_name.c_str(), {parsed.n_targets}, parsed.labels_int64));
      NodeProto gather = MakeNode("Gather", {labels_name, indices}, {source.output()[0].value()},
                                  "", (source.name().value() + "_labels").c_str());
      AddAttribute(gather, "axis", static_cast<int64_t>(0));
      replacements.push_back(std::move(gather));
    } else {
      std::vector<int64_t> keys(static_cast<std::size_t>(parsed.n_targets));
      for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = static_cast<int64_t>(i);
      }
      NodeProto encoder = MakeNode("LabelEncoder", {indices}, {source.output()[0].value()},
                                   kMlDomain, (source.name().value() + "_labels").c_str());
      AddAttribute(encoder, "keys_int64s", keys);
      AddAttribute(encoder, "values_strings", parsed.labels_strings);
      replacements.push_back(std::move(encoder));
    }
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
