// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/layer_extraction.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {

using core::builder::GraphBuilder;
using core::builder::GraphGraph;

constexpr std::size_t kMaxRegion = 256;
constexpr std::size_t kMaxInterval = 4096;
constexpr std::size_t kMaxFunctionWork = 4096;
constexpr std::size_t kMaxScopeDepth = 16;
constexpr std::size_t kMaxSignatureBytes = 65536;
constexpr std::size_t kMaxPorts = 128;
constexpr std::size_t kMaxExtensions = 256;
using Region = std::vector<const NodeProto *>;
using Regions = std::vector<Region>;

bool IsLayerName(const std::string &domain, const std::string &name) {
  return domain == "ai.rt" && (name == "Layer" || name.rfind("Layer_", 0) == 0);
}

bool IsLayer(const NodeProto &node) {
  return IsLayerName(node.domain().value(), node.op_type().value());
}

// Collects lexical uses, not just deduplicated values: the cost counts actual edges.
bool GraphCaptures(const GraphProto &graph, std::vector<std::string> &captures, std::size_t depth,
                   std::size_t &work);

bool NodeCaptures(const NodeProto &node, std::vector<std::string> &captures, std::size_t depth,
                  std::size_t &work) {
  if (depth >= kMaxScopeDepth)
    return false;
  std::vector<const AttributeProto *> attributes;
  for (const auto &attribute : node.attribute())
    attributes.push_back(&attribute);
  std::sort(attributes.begin(), attributes.end(),
            [](const auto *a, const auto *b) { return a->name().value() < b->name().value(); });
  for (const AttributeProto *attribute : attributes) {
    if (attribute->has_g() && !GraphCaptures(attribute->g(), captures, depth + 1, work))
      return false;
    for (const GraphProto &nested : attribute->graphs()) {
      if (!GraphCaptures(nested, captures, depth + 1, work))
        return false;
    }
  }
  return true;
}

bool GraphCaptures(const GraphProto &graph, std::vector<std::string> &captures, std::size_t depth,
                   std::size_t &work) {
  if (depth >= kMaxScopeDepth)
    return false;
  std::unordered_set<std::string> local;
  for (const auto &input : graph.input())
    local.insert(input.name().value());
  for (const auto &initializer : graph.initializer())
    local.insert(initializer.name().value());
  for (const auto &initializer : graph.sparse_initializer())
    local.insert(initializer.values().name().value());
  for (const auto &initializer : graph.extended_initializer())
    local.insert(initializer.name().value());
  auto use = [&](const std::string &name) {
    if (!name.empty() && local.count(name) == 0)
      captures.push_back(name);
  };
  for (const NodeProto &node : graph.node()) {
    if (++work > kMaxFunctionWork)
      return false;
    for (const auto &input : node.input())
      use(input.value());
    std::vector<std::string> nested;
    if (!NodeCaptures(node, nested, depth, work))
      return false;
    for (const std::string &name : nested)
      use(name);
    for (const auto &output : node.output()) {
      if (!output.value().empty())
        local.insert(output.value());
    }
  }
  for (const auto &output : graph.output())
    use(output.name().value());
  return true;
}

bool Materialize(const NodeProto &node, const GraphBuilder &builder, NodeProto &out) {
  out = utils::CopyWithSharedTensorStorage(node);
  for (std::size_t i = 0; i < out.attribute().size(); ++i) {
    auto &attribute = *out.mutable_attribute(static_cast<int>(i));
    const std::string name = attribute.name().value();
    if (name.size() < 4 || name.substr(name.size() - 4) != "_ref")
      continue;
    if (attribute.type() == AttributeProto::AttributeType::STRING &&
        builder.HasSubgraph(attribute.s().value())) {
      GraphProto nested = builder.Subgraph(attribute.s().value()).BuildGraph();
      attribute = AttributeProto();
      attribute.set_name(name.substr(0, name.size() - 4));
      attribute.set_type(AttributeProto::AttributeType::GRAPH);
      *attribute.mutable_g() = std::move(nested);
    } else if (attribute.type() == AttributeProto::AttributeType::STRINGS) {
      std::vector<GraphProto> nested;
      for (const auto &reference : attribute.strings()) {
        if (!builder.HasSubgraph(reference.value()))
          return false;
        nested.push_back(builder.Subgraph(reference.value()).BuildGraph());
      }
      attribute = AttributeProto();
      attribute.set_name(name.substr(0, name.size() - 4));
      attribute.set_type(AttributeProto::AttributeType::GRAPHS);
      for (GraphProto &graph : nested)
        *attribute.add_graphs() = std::move(graph);
    }
  }
  return true;
}

bool SignaturePart(std::string &signature, const std::string &part) {
  if (part.size() > kMaxSignatureBytes || signature.size() + part.size() + 24 > kMaxSignatureBytes)
    return false;
  signature += std::to_string(part.size()) + ":" + part;
  return true;
}

using SignatureNames = std::map<std::string, std::string>;

std::string SignatureValue(SignatureNames &names, const std::string &name) {
  if (name.empty())
    return "-";
  const auto [it, inserted] = names.emplace(name, "capture" + std::to_string(names.size()));
  (void)inserted;
  return it->second;
}

bool AttributeSignature(const AttributeProto &attribute, std::string &signature,
                        const SignatureNames &outer, std::size_t depth, std::size_t &work);

bool SafeGraphDeclarations(const GraphProto &graph) {
  if (graph.node().size() > kMaxFunctionWork || graph.input().size() > kMaxPorts ||
      graph.output().size() > kMaxPorts || graph.initializer().size() > kMaxFunctionWork ||
      graph.sparse_initializer().size() > kMaxFunctionWork ||
      graph.extended_initializer().size() > kMaxFunctionWork)
    return false;
  std::unordered_set<std::string> local;
  auto define = [&](const std::string &name) { return !name.empty() && local.insert(name).second; };
  for (const auto &input : graph.input()) {
    if (!define(input.name().value()))
      return false;
  }
  std::unordered_set<std::string> initializers;
  auto define_initializer = [&](const std::string &name) {
    if (name.empty() || !initializers.insert(name).second)
      return false;
    local.insert(name);
    return true;
  };
  for (const auto &initializer : graph.initializer()) {
    if (!define_initializer(initializer.name().value()))
      return false;
  }
  for (const auto &initializer : graph.sparse_initializer()) {
    if (!define_initializer(initializer.values().name().value()))
      return false;
  }
  for (const auto &initializer : graph.extended_initializer()) {
    if (!define_initializer(initializer.name().value()))
      return false;
  }
  for (const NodeProto &node : graph.node()) {
    for (const auto &output : node.output()) {
      if (!output.value().empty() && !define(output.value()))
        return false;
    }
  }
  std::unordered_set<std::string> outputs;
  for (const auto &output : graph.output()) {
    const std::string &name = output.name().value();
    if (name.empty() || !outputs.insert(name).second || local.count(name) == 0)
      return false;
  }
  return true;
}

bool GraphSignature(const GraphProto &graph, std::string &signature, SignatureNames names,
                    std::size_t depth, std::size_t &work) {
  if (depth >= kMaxScopeDepth || !graph.sparse_initializer().empty() ||
      !graph.extended_initializer().empty() || !SafeGraphDeclarations(graph))
    return false;
  if (!SignaturePart(signature, "graph") ||
      !SignaturePart(signature, std::to_string(graph.input_size())))
    return false;
  const std::string scope = "scope" + std::to_string(depth) + ":";
  std::unordered_set<std::string> formal_inputs;
  for (std::size_t i = 0; i < graph.input().size(); ++i) {
    names[graph.input()[i].name().value()] = scope + "input" + std::to_string(i);
    formal_inputs.insert(graph.input()[i].name().value());
    if (!SignaturePart(signature, graph.input()[i].type().SerializeAsString()))
      return false;
  }
  if (!SignaturePart(signature, std::to_string(graph.initializer_size())))
    return false;
  for (std::size_t i = 0; i < graph.initializer().size(); ++i) {
    const auto &initializer = graph.initializer()[i];
    if (formal_inputs.count(initializer.name().value()) == 0)
      names[initializer.name().value()] = scope + "weight" + std::to_string(i);
    if (!SignaturePart(signature, names.at(initializer.name().value())) ||
        !SignaturePart(signature, std::to_string(initializer.data_type())) ||
        !SignaturePart(signature, std::to_string(initializer.dims_size())))
      return false;
    for (const auto dimension : initializer.dims()) {
      if (!SignaturePart(signature, std::to_string(dimension)))
        return false;
    }
  }
  if (!SignaturePart(signature, std::to_string(graph.node_size())))
    return false;
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
    if (++work > kMaxFunctionWork)
      return false;
    const NodeProto &node = graph.node()[i];
    if (IsLayer(node) || !SignaturePart(signature, node.domain().value()) ||
        !SignaturePart(signature, node.op_type().value()) ||
        !SignaturePart(signature, node.overload().value()) ||
        !SignaturePart(signature, std::to_string(node.input_size())) ||
        !SignaturePart(signature, std::to_string(node.output_size())))
      return false;
    for (const auto &input : node.input()) {
      if (!SignaturePart(signature, SignatureValue(names, input.value())))
        return false;
    }
    std::vector<const AttributeProto *> attributes;
    for (const auto &attribute : node.attribute())
      attributes.push_back(&attribute);
    std::sort(attributes.begin(), attributes.end(),
              [](const auto *a, const auto *b) { return a->name().value() < b->name().value(); });
    if (!SignaturePart(signature, std::to_string(attributes.size())))
      return false;
    for (const AttributeProto *attribute : attributes) {
      if (!AttributeSignature(*attribute, signature, names, depth + 1, work))
        return false;
    }
    for (std::size_t port = 0; port < node.output().size(); ++port) {
      const auto &name = node.output()[port].value();
      if (!SignaturePart(signature, name.empty() ? "omitted" : "output"))
        return false;
      if (!name.empty())
        names[name] = scope + "node" + std::to_string(i) + ":" + std::to_string(port);
    }
  }
  if (!SignaturePart(signature, std::to_string(graph.output_size())))
    return false;
  for (const auto &output : graph.output()) {
    if (!SignaturePart(signature, SignatureValue(names, output.name().value())) ||
        !SignaturePart(signature, output.type().SerializeAsString()))
      return false;
  }
  return true;
}

bool HasComparableLayerTensorContent(const TensorProto &tensor) {
  if (!tensor.raw_data().empty() || !tensor.float_data().empty() || !tensor.int32_data().empty() ||
      !tensor.int64_data().empty() || !tensor.double_data().empty() ||
      !tensor.uint64_data().empty() || !tensor.string_data().empty())
    return true;
  if (tensor.HasDeferredWeight())
    return false;
  return (tensor.data_location() == TensorProto::EXTERNAL && !tensor.external_data().empty()) ||
         std::find(tensor.dims().begin(), tensor.dims().end(), 0) != tensor.dims().end();
}

bool MaterializeLayerTensor(TensorProto &tensor) {
  if (!tensor.HasDeferredWeight())
    return HasComparableLayerTensorContent(tensor);
  if (!tensor.has_raw_data()) {
    const uint32_t bits = FixedBitWidth(tensor.data_type());
    if (bits == 0)
      return false;
    const uint64_t limit = kMaxSignatureBytes * 8 / bits;
    uint64_t count =
        std::find(tensor.dims().begin(), tensor.dims().end(), 0) == tensor.dims().end() ? 1 : 0;
    for (const int64_t dim : tensor.dims()) {
      if (dim < 0 || (dim != 0 && count > limit / static_cast<uint64_t>(dim)))
        return false;
      count *= static_cast<uint64_t>(dim);
    }
    if (count > limit)
      return false;
    tensor.LoadDeferredWeight();
  }
  tensor.deferred_weight_source.reset();
  tensor.clr_external_data();
  if (tensor.data_location() == TensorProto::EXTERNAL)
    tensor.set_data_location(TensorProto::DEFAULT);
  return true;
}

bool AttributeSignature(const AttributeProto &source, std::string &signature,
                        const SignatureNames &outer, std::size_t depth, std::size_t &work) {
  auto attribute = utils::CopyWithSharedTensorStorage(source);
  // Tensor attribute values constrain matching exactly, without serializing
  // unbounded parameter payloads. Top-level Constant producers stay external.
  auto bounded_tensor = [](TensorProto &tensor) {
    if (!MaterializeLayerTensor(tensor) || tensor.raw_data().size() > kMaxSignatureBytes ||
        tensor.float_data().size() > kMaxSignatureBytes / sizeof(float) ||
        tensor.double_data().size() > kMaxSignatureBytes / sizeof(double) ||
        tensor.int32_data().size() > kMaxSignatureBytes / sizeof(int32_t) ||
        tensor.int64_data().size() > kMaxSignatureBytes / sizeof(int64_t) ||
        tensor.uint64_data().size() > kMaxSignatureBytes / sizeof(uint64_t))
      return false;
    std::size_t bytes = 0;
    for (const auto &value : tensor.string_data()) {
      if (value.value().size() > kMaxSignatureBytes - bytes)
        return false;
      bytes += value.value().size();
    }
    return true;
  };
  if (attribute.tensors().size() > kMaxPorts || attribute.sparse_tensors().size() > kMaxPorts)
    return false;
  if (attribute.has_t() && !bounded_tensor(attribute.ref_t()))
    return false;
  for (auto &tensor : attribute.ref_tensors()) {
    if (!bounded_tensor(tensor))
      return false;
  }
  if (attribute.has_sparse_tensor() &&
      (!bounded_tensor(attribute.ref_sparse_tensor().ref_values()) ||
       !bounded_tensor(attribute.ref_sparse_tensor().ref_indices())))
    return false;
  for (auto &tensor : attribute.ref_sparse_tensors()) {
    if (!bounded_tensor(tensor.ref_values()) || !bounded_tensor(tensor.ref_indices()))
      return false;
  }
  if (!attribute.has_g() && attribute.graphs().empty())
    return SignaturePart(signature, attribute.SerializeAsString());
  if (!SignaturePart(signature, attribute.name().value()) ||
      !SignaturePart(signature, std::to_string(attribute.type())) ||
      !SignaturePart(signature, attribute.ref_attr_name().value()) ||
      !SignaturePart(signature, attribute.has_g() ? "graph" : "graphs") ||
      !SignaturePart(signature, std::to_string(attribute.graphs().size())))
    return false;
  if (attribute.has_g() && !GraphSignature(attribute.g(), signature, outer, depth + 1, work))
    return false;
  for (const GraphProto &graph : attribute.graphs()) {
    if (!GraphSignature(graph, signature, outer, depth + 1, work))
      return false;
  }
  return true;
}

struct NodeData {
  NodeProto materialized;
  std::string signature;
  std::string match_key;
  // Port-labelled successors. Null marks genuinely ambiguous fanout.
  std::unordered_map<std::string, const NodeProto *> successors;
  std::vector<std::string> dependencies;
  bool usable = false;
  bool anchor = false;
  bool captures_complete = true;
  // Value identifiers of inputs (kNoValue when omitted), outputs and dependencies.
  std::vector<uint32_t> input_ids, output_ids, dependency_ids;
  // Producers of the dependencies, nullptr for graph inputs, initializers and captures.
  std::vector<const NodeProto *> producers;
  // Positions of those producers, kNoPosition when not produced in this graph.
  std::vector<std::size_t> producer_positions;
  std::size_t position = 0;
};

constexpr std::size_t kNoPosition = std::numeric_limits<std::size_t>::max();

constexpr uint32_t kNoValue = std::numeric_limits<uint32_t>::max();

bool KnownType(const TypeProto &type, std::size_t depth = 0);

bool IsSafeNode(const NodeProto &node, const GraphBuilder &builder,
                std::map<std::string, bool> &functions, std::set<std::string> &visiting,
                std::size_t depth, std::size_t &work, bool allow_attribute_references = false);

bool SafeAttributeGraphs(const AttributeProto &attribute, const GraphBuilder &builder,
                         std::map<std::string, bool> &functions, std::set<std::string> &visiting,
                         std::size_t depth, std::size_t &work, bool allow_attribute_references) {
  if (!allow_attribute_references && !attribute.ref_attr_name().empty())
    return false;
  auto safe_graph = [&](const GraphProto &graph) {
    if (!SafeGraphDeclarations(graph))
      return false;
    for (const NodeProto &inner : graph.node()) {
      if (!IsSafeNode(inner, builder, functions, visiting, depth + 1, work,
                      allow_attribute_references))
        return false;
    }
    return true;
  };
  if (attribute.has_g() && !safe_graph(attribute.g()))
    return false;
  for (const GraphProto &graph : attribute.graphs()) {
    if (!safe_graph(graph))
      return false;
  }
  return true;
}

bool IsSafeNode(const NodeProto &node, const GraphBuilder &builder,
                std::map<std::string, bool> &functions, std::set<std::string> &visiting,
                std::size_t depth, std::size_t &work, bool allow_attribute_references) {
  if (IsLayer(node) || depth >= kMaxScopeDepth || (depth != 0 && ++work > kMaxFunctionWork))
    return false;
  const std::string key = core::shapes::ShapesContext::LocalFunctionKey(
      node.domain().value(), node.op_type().value(), node.overload().value());
  const FunctionProto *function = builder.Shapes().GetLocalFunction(key);
  if (function != nullptr) {
    auto cached = functions.find(key);
    if (cached == functions.end()) {
      if (!visiting.insert(key).second)
        return false;
      bool safe = true;
      for (const NodeProto &inner : function->node()) {
        if (!IsSafeNode(inner, builder, functions, visiting, depth + 1, work, true)) {
          safe = false;
          break;
        }
      }
      for (const auto &attribute : function->attribute_proto()) {
        if (!SafeAttributeGraphs(attribute, builder, functions, visiting, depth + 1, work, true))
          safe = false;
      }
      visiting.erase(key);
      cached = functions.emplace(key, safe).first;
    }
    if (!cached->second)
      return false;
  }
  for (const auto &attribute : node.attribute()) {
    if (!SafeAttributeGraphs(attribute, builder, functions, visiting, depth, work,
                             allow_attribute_references))
      return false;
  }
  return true;
}

struct Index;
ValueInfoProto ValueInfo(Index &index, const std::string &name);

// Type, shape and constness of a value, looked up once per search.
struct ValueKey {
  const TypeProto *type = nullptr;
  bool structured = false;
  int32_t dtype = TensorProto::DataType::UNDEFINED;
  const core::symbolic::SymTensor *shape = nullptr;
  bool constant = false;
};

struct NodePairHash {
  std::size_t operator()(const std::pair<const NodeProto *, const NodeProto *> &key) const {
    const auto a = std::hash<const NodeProto *>()(key.first);
    return a ^ (std::hash<const NodeProto *>()(key.second) + 0x9e3779b97f4a7c15ULL + (a << 6) +
                (a >> 2));
  }
};

struct Index {
  GraphGraph &graph;
  const RepeatedLayerOptions options;
  std::unordered_map<const NodeProto *, NodeData> data;
  // The graph does not change during a search, so value and node comparisons are cached.
  std::unordered_map<std::string, ValueKey> values;
  std::unordered_map<std::pair<const NodeProto *, const NodeProto *>, bool, NodePairHash>
      compatible;
  std::unordered_map<std::string, uint32_t> value_ids;
  // Node data by graph position.
  std::vector<NodeData *> entries;
  // Per value identifier: name, consumers, graph-output flag and cached boundary analyses
  // (-1 when not computed yet).
  std::vector<std::string> names;
  std::vector<std::size_t> value_producers;
  std::vector<std::vector<const NodeProto *>> consumers;
  std::vector<bool> graph_outputs;
  std::vector<int8_t> boundary_types;
  // Generation-stamped scratch storage indexed by value identifier.
  std::vector<int64_t> scratch;
  std::vector<uint32_t> scratch_generation;
  uint32_t generation = 0;

  uint32_t NextGeneration() {
    if (++generation == 0) {
      std::fill(scratch_generation.begin(), scratch_generation.end(), 0);
      generation = 1;
    }
    return generation;
  }

  uint32_t ValueId(const std::string &name) {
    if (name.empty())
      return kNoValue;
    auto [it, inserted] = value_ids.try_emplace(name, static_cast<uint32_t>(names.size()));
    if (inserted)
      names.push_back(name);
    return it->second;
  }
  std::map<std::string, Region> anchors;
  std::unordered_map<std::string, const ValueInfoProto *> declarations;
  bool valid = true;
  // Any earlier value may be captured by this last incompletely analyzed node.
  std::size_t last_incomplete_capture = kNoPosition;

  Index(GraphGraph &graph, const RepeatedLayerOptions &options) : graph(graph), options(options) {
    for (const auto &input : graph.Builder().Inputs())
      declarations[input.name().value()] = &input;
    for (const auto &output : graph.Builder().Outputs())
      declarations[output.name().value()] = &output;
    std::map<std::tuple<std::string, std::string, std::string>, bool> anchor_cache;
    std::size_t function_work = 0, signature_work = 0, safety_work = 0;
    std::map<std::string, bool> safe_functions;
    for (const NodeProto &node : graph.Builder().Nodes()) {
      NodeData entry;
      if (IsLayer(node)) {
        entry.materialized = utils::CopyWithSharedTensorStorage(node);
      } else if (!Materialize(node, graph.Builder(), entry.materialized)) {
        valid = false;
        return;
      }
      for (const auto &input : node.input()) {
        if (!input.value().empty())
          entry.dependencies.push_back(input.value());
      }
      std::size_t capture_work = 0;
      entry.captures_complete =
          IsLayer(node) || NodeCaptures(entry.materialized, entry.dependencies, 0, capture_work);
      if (!entry.captures_complete)
        last_incomplete_capture = data.size();
      for (const auto &input : node.input())
        entry.input_ids.push_back(ValueId(input.value()));
      for (const auto &output : node.output())
        entry.output_ids.push_back(ValueId(output.value()));
      for (const std::string &name : entry.dependencies) {
        entry.dependency_ids.push_back(ValueId(name));
        entry.producers.push_back(graph.NodeBefore(name));
      }
      entry.usable = entry.captures_complete && !IsLayer(node) &&
                     entry.dependencies.size() <= kMaxPorts && node.output().size() <= kMaxPorts &&
                     node.attribute().size() <= kMaxPorts;
      if (entry.usable) {
        std::set<std::string> visiting;
        entry.usable = IsSafeNode(entry.materialized, graph.Builder(), safe_functions, visiting, 0,
                                  safety_work);
      }
      if (entry.usable && !core::shapes::HasShapeInference(graph.Builder().Shapes(), node)) {
        std::vector<ValueInfoProto> outputs;
        outputs.reserve(node.output().size());
        for (const auto &name : node.output())
          if (!name.empty())
            outputs.push_back(ValueInfo(*this, name));
        std::unordered_map<std::string, const ValueInfoProto *> annotations;
        for (const auto &output : outputs)
          annotations.emplace(output.name(), &output);
        entry.usable = core::shapes::CanSeedAnnotatedOutputsForUnknownCustomOp(
            graph.Builder().Shapes(), node, annotations);
      }
      if (entry.usable) {
        std::vector<std::string> attributes;
        std::size_t attribute_bytes = 0;
        SignatureNames names;
        for (std::size_t i = 0; i < node.input().size(); ++i) {
          if (!node.input()[i].value().empty())
            names.emplace(node.input()[i].value(), "port" + std::to_string(i));
        }
        for (const auto &attribute : entry.materialized.attribute()) {
          std::string bytes;
          if (!AttributeSignature(attribute, bytes, names, 0, signature_work)) {
            entry.usable = false;
            break;
          }
          attribute_bytes += bytes.size();
          if (attribute_bytes > kMaxSignatureBytes) {
            entry.usable = false;
            break;
          }
          attributes.push_back(std::move(bytes));
        }
        std::sort(attributes.begin(), attributes.end());
        auto append = [&](const std::string &value) {
          entry.signature += std::to_string(value.size()) + ":" + value;
        };
        append(node.domain().value());
        append(node.op_type().value());
        append(node.overload().value());
        append(std::to_string(node.input_size()));
        append(std::to_string(node.output_size()));
        for (const std::string &attribute : attributes)
          append(attribute);
      }
      if (entry.usable) {
        const auto identity =
            std::make_tuple(node.domain().value(), node.op_type().value(), node.overload().value());
        auto [it, inserted] = anchor_cache.emplace(identity, false);
        if (inserted) {
          it->second = IsOperatorGroupAnchor(node, graph.Builder(), function_work);
        }
        entry.anchor = it->second;
        if (entry.anchor)
          anchors[entry.signature].push_back(&node);
      }
      data.emplace(&node, std::move(entry));
    }
    const auto &nodes = graph.Builder().Nodes();
    std::unordered_map<const NodeProto *, std::size_t> positions;
    for (std::size_t position = 0; position < nodes.size(); ++position) {
      positions.emplace(&nodes[position], position);
      entries.push_back(&data.at(&nodes[position]));
    }
    consumers.resize(names.size());
    value_producers.assign(names.size(), kNoPosition);
    graph_outputs.assign(names.size(), false);
    boundary_types.assign(names.size(), -1);
    scratch.assign(names.size(), 0);
    scratch_generation.assign(names.size(), 0);
    for (std::size_t position = 0; position < entries.size(); ++position) {
      NodeData *entry = entries[position];
      entry->position = position;
      for (uint32_t id : entry->dependency_ids)
        consumers[id].push_back(&nodes[position]);
      for (uint32_t id : entry->output_ids) {
        if (id != kNoValue) {
          value_producers[id] = position;
          graph_outputs[id] = graph.IsOutput(names[id]);
        }
      }
      for (const NodeProto *producer : entry->producers) {
        const auto found = producer == nullptr ? positions.end() : positions.find(producer);
        entry->producer_positions.push_back(found == positions.end() ? kNoPosition : found->second);
      }
    }
  }
};

const TypeProto *DeclaredType(Index &index, const std::string &name) {
  const auto &context = index.graph.Builder().Shapes();
  if (context.HasType(name) && KnownType(context.GetType(name)))
    return &context.GetType(name);
  const auto declaration = index.declarations.find(name);
  if (declaration != index.declarations.end() && declaration->second->has_type() &&
      KnownType(declaration->second->type()))
    return &declaration->second->type();
  return nullptr;
}

const ValueKey &GetValueKey(Index &index, const std::string &name) {
  auto [it, inserted] = index.values.try_emplace(name);
  if (!inserted)
    return it->second;
  GraphGraph &graph = index.graph;
  ValueKey &key = it->second;
  key.type = DeclaredType(index, name);
  key.structured = key.type != nullptr && core::shapes::HasStructuredType(*key.type);
  if (key.structured)
    return key;
  if (key.type != nullptr && key.type->has_tensor_type())
    key.dtype = key.type->tensor_type().elem_type();
  else if (graph.HasType(name))
    key.dtype = core::symbolic::TensorTypeToDataType(graph.GetType(name));
  if (graph.HasShape(name)) {
    key.shape = &graph.GetShape(name);
    key.constant = graph.IsConstant(name);
  }
  return key;
}

bool CompatibleValue(Index &index, const std::string &left, const std::string &right) {
  if (left.empty() || right.empty())
    return left.empty() && right.empty();
  const ValueKey &l = GetValueKey(index, left);
  const ValueKey &r = GetValueKey(index, right);
  if (l.structured || r.structured)
    return l.structured && r.structured && core::shapes::SameDeclaredType(*l.type, *r.type);
  if (l.dtype != TensorProto::UNDEFINED && r.dtype != TensorProto::UNDEFINED && l.dtype != r.dtype)
    return false;
  if (l.shape == nullptr || r.shape == nullptr)
    return true;
  const auto &a = l.shape->Shape();
  const auto &b = r.shape->Shape();
  if (a.Rank() != b.Rank())
    return false;
  const bool activation = !l.constant && !r.constant;
  for (std::size_t i = 0; i < a.Rank(); ++i) {
    if (a[i].IsInt() && b[i].IsInt() && a[i].AsInt() != b[i].AsInt() &&
        !(activation && i + 1 < a.Rank()))
      return false;
  }
  return true;
}

bool ComputeCompatibleNode(Index &index, const NodeProto &left, const NodeProto &right) {
  const auto &a = index.data.at(&left);
  const auto &b = index.data.at(&right);
  if (!a.usable || !b.usable || a.signature != b.signature ||
      a.dependencies.size() != b.dependencies.size())
    return false;
  for (int i = 0; i < left.input_size(); ++i) {
    if (!CompatibleValue(index, left.input()[i].value(), right.input()[i].value()))
      return false;
  }
  for (int i = 0; i < left.output_size(); ++i) {
    if (!CompatibleValue(index, left.output()[i].value(), right.output()[i].value()))
      return false;
  }
  for (std::size_t i = 0; i < a.dependencies.size(); ++i) {
    if (!CompatibleValue(index, a.dependencies[i], b.dependencies[i]))
      return false;
  }
  return true;
}

bool CompatibleNode(Index &index, const NodeProto &left, const NodeProto &right) {
  auto [it, inserted] = index.compatible.try_emplace({&left, &right}, false);
  if (inserted)
    it->second = ComputeCompatibleNode(index, left, right);
  return it->second;
}

// Corresponding values have the same internal producer/output port or external
// alias class. This also detects additional edges between already matched nodes.
// Internal values are encoded as (role << 8 | port) >= 0, omitted inputs as -1 and
// external alias classes as -2, -3, ...
std::vector<int64_t> Topology(Index &index, const Region &region) {
  const uint32_t generation = index.NextGeneration();
  auto set = [&](uint32_t id, int64_t code) {
    index.scratch_generation[id] = generation;
    index.scratch[id] = code;
  };
  for (std::size_t role = 0; role < region.size(); ++role) {
    const auto &outputs = index.data.at(region[role]).output_ids;
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      if (outputs[port] != kNoValue && index.scratch_generation[outputs[port]] != generation)
        set(outputs[port], static_cast<int64_t>((role << 8) | port));
    }
  }
  int64_t external = 0;
  std::vector<int64_t> topology;
  auto add = [&](uint32_t id) {
    if (index.scratch_generation[id] != generation)
      set(id, -2 - external++);
    topology.push_back(index.scratch[id]);
  };
  for (const NodeProto *node : region) {
    const auto &entry = index.data.at(node);
    for (uint32_t id : entry.input_ids) {
      if (id == kNoValue)
        topology.push_back(-1);
      else
        add(id);
    }
    for (uint32_t id : entry.dependency_ids)
      add(id);
  }
  return topology;
}

struct Boundary : OperatorGroup {
  std::vector<uint32_t> edges, input_ids, output_ids;
};

Boundary GetBoundary(Index &index, const Region &region) {
  Boundary boundary;
  std::unordered_set<const NodeProto *> inside(region.begin(), region.end());
  std::vector<const NodeData *> ordered;
  ordered.reserve(region.size());
  for (const NodeProto *node : region)
    ordered.push_back(&index.data.at(node));
  std::sort(ordered.begin(), ordered.end(),
            [](const NodeData *a, const NodeData *b) { return a->position < b->position; });
  // Scratch flags: 1 produced inside, 2 boundary input, 4 boundary output.
  const uint32_t generation = index.NextGeneration();
  auto flags = [&](uint32_t id) -> int64_t & {
    if (index.scratch_generation[id] != generation) {
      index.scratch_generation[id] = generation;
      index.scratch[id] = 0;
    }
    return index.scratch[id];
  };
  for (const NodeData *entry : ordered) {
    boundary.node_indices.push_back(entry->position);
    for (uint32_t id : entry->output_ids) {
      if (id != kNoValue)
        flags(id) |= 1;
    }
  }
  for (const NodeData *entry : ordered) {
    for (uint32_t id : entry->dependency_ids) {
      int64_t &flag = flags(id);
      if ((flag & 1) == 0) {
        boundary.edges.push_back(id);
        if ((flag & 2) == 0) {
          flag |= 2;
          boundary.input_ids.push_back(id);
          boundary.inputs.push_back(index.names[id]);
        }
      }
    }
    for (uint32_t id : entry->output_ids) {
      if (id == kNoValue)
        continue;
      bool external = index.graph_outputs[id] || (index.last_incomplete_capture != kNoPosition &&
                                                  entry->position < index.last_incomplete_capture);
      if (external)
        boundary.edges.push_back(id);
      for (const NodeProto *user : index.consumers[id]) {
        if (inside.count(user) == 0) {
          external = true;
          boundary.edges.push_back(id);
        }
      }
      int64_t &flag = flags(id);
      if (external && (flag & 4) == 0) {
        flag |= 4;
        boundary.output_ids.push_back(id);
        boundary.outputs.push_back(index.names[id]);
      }
    }
  }
  return boundary;
}

struct Plan : OperatorGroupProposal {
  Regions regions;
  // Per group, positions of input Shape nodes also copied before the call for outer consumers.
  std::vector<std::vector<std::size_t>> duplicated;
  // Half-open ranges of groups grown together from one seed.
  std::vector<std::pair<std::size_t, std::size_t>> families;
};

bool KnownType(const TypeProto &type, std::size_t depth) {
  if (depth >= kMaxScopeDepth)
    return false;
  if (type.has_tensor_type())
    return type.tensor_type().elem_type() != TensorProto::DataType::UNDEFINED;
  if (type.has_sparse_tensor_type())
    return type.sparse_tensor_type().elem_type() != TensorProto::DataType::UNDEFINED;
  if (type.has_sequence_type())
    return KnownType(type.sequence_type().elem_type(), depth + 1);
  if (type.has_optional_type())
    return KnownType(type.optional_type().elem_type(), depth + 1);
  if (type.has_map_type())
    return type.map_type().key_type() != TensorProto::DataType::UNDEFINED &&
           KnownType(type.map_type().value_type(), depth + 1);
  return type.has_struct_type() || type.has_opaque_type();
}

bool SupportedBoundaryType(const TypeProto &type) {
  if (type.has_tensor_type() || type.has_struct_type())
    return true;
  if (type.has_sequence_type())
    return SupportedBoundaryType(type.sequence_type().elem_type());
  if (type.has_map_type())
    return type.map_type().value_type().has_tensor_type();
  return false;
}

bool KnownBoundaryType(Index &index, const std::string &name) {
  if (const auto *type = DeclaredType(index, name))
    return SupportedBoundaryType(*type);
  GraphGraph &graph = index.graph;
  return graph.HasType(name) && core::symbolic::TensorTypeToDataType(graph.GetType(name)) !=
                                    TensorProto::DataType::UNDEFINED;
}

// A unit is either a contracted region or one untouched interleaving node.
// Sorting the contracted DAG rejects re-entry cycles, even through nested captures.
bool Schedule(Index &index, Plan &plan) {
  plan.first = std::numeric_limits<std::size_t>::max();
  plan.last = 0;
  std::vector<std::size_t> priority;
  for (std::size_t i = 0; i < plan.regions.size(); ++i) {
    const OperatorGroup &boundary = plan.groups[i];
    if (boundary.outputs.empty())
      return false;
    plan.first = std::min(plan.first, boundary.node_indices.front());
    plan.last = std::max(plan.last, boundary.node_indices.back());
    priority.push_back(boundary.node_indices.front());
  }
  if (plan.last - plan.first >= kMaxInterval)
    return false;
  // Units are indexed by position relative to plan.first.
  std::vector<std::size_t> owner(plan.last - plan.first + 1, kNoPosition);
  for (std::size_t i = 0; i < plan.groups.size(); ++i) {
    for (std::size_t position : plan.groups[i].node_indices)
      owner[position - plan.first] = i;
  }
  for (std::size_t position = plan.first; position <= plan.last; ++position) {
    if (owner[position - plan.first] == kNoPosition) {
      owner[position - plan.first] = priority.size();
      priority.push_back(position);
    }
  }
  // Duplicate edges are harmless: each one increments and later decrements the in-degree.
  std::vector<std::vector<std::size_t>> successors(priority.size());
  std::vector<std::size_t> indegree(priority.size(), 0);
  auto connect = [&](std::size_t from, std::size_t to) {
    if (from != to) {
      successors[from].push_back(to);
      ++indegree[to];
    }
  };
  for (std::size_t position = plan.first; position <= plan.last; ++position) {
    const std::size_t to = owner[position - plan.first];
    for (std::size_t producer : index.entries[position]->producer_positions) {
      if (producer != kNoPosition && producer >= plan.first)
        connect(owner[producer - plan.first], to);
    }
  }
  // Unknown captures prevent every unit, including preparations, from crossing their owner.
  if (index.last_incomplete_capture != kNoPosition) {
    std::size_t barrier = kNoPosition;
    std::vector<std::size_t> since_barrier;
    for (std::size_t position = plan.first; position <= plan.last; ++position) {
      const std::size_t unit = owner[position - plan.first];
      if (barrier != kNoPosition)
        connect(barrier, unit);
      if (!index.entries[position]->captures_complete) {
        for (std::size_t before : since_barrier)
          connect(before, unit);
        since_barrier.clear();
        barrier = unit;
      } else {
        since_barrier.push_back(unit);
      }
    }
  }
  using Ready = std::pair<std::size_t, std::size_t>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  for (std::size_t i = 0; i < priority.size(); ++i) {
    if (indegree[i] == 0)
      ready.emplace(priority[i], i);
  }
  plan.schedule.clear();
  while (!ready.empty()) {
    const std::size_t unit = ready.top().second;
    ready.pop();
    if (unit < plan.regions.size())
      plan.schedule.emplace_back(true, unit);
    else
      plan.schedule.emplace_back(false, priority[unit]);
    for (std::size_t next : successors[unit]) {
      if (--indegree[next] == 0)
        ready.emplace(priority[next], next);
    }
  }
  return plan.schedule.size() == priority.size();
}

bool CachedKnownBoundaryType(Index &index, uint32_t id) {
  int8_t &known = index.boundary_types[id];
  if (known < 0)
    known = KnownBoundaryType(index, index.names[id]) ? 1 : 0;
  return known != 0;
}

// A Shape of a region input belongs to the shared function. Outer consumers read a
// duplicate placed before the call instead of a function output.
bool IsInputShape(Index &index, const Region &region, const NodeProto &node) {
  if (NormaliseDomain(node.domain()) != "ai.onnx" || node.op_type() != "Shape" ||
      !node.overload().empty() || !node.device_configurations().empty() ||
      node.input().size() != 1 || node.output().size() != 1 ||
      index.graph.Builder().Shapes().HasLocalFunction(
          core::shapes::ShapesContext::LocalFunctionKey(node.domain().value(), "Shape", "")))
    return false;
  const NodeData &entry = index.data.at(&node);
  return entry.producers.size() == 1 &&
         std::find(region.begin(), region.end(), entry.producers[0]) == region.end();
}

std::vector<std::size_t> DuplicateInputShapes(Index &index, const Region &region,
                                              Boundary &boundary) {
  std::unordered_set<uint32_t> duplicated_ids;
  std::vector<std::size_t> positions;
  for (const NodeProto *node : region) {
    const NodeData &entry = index.data.at(node);
    if (!IsInputShape(index, region, *node))
      continue;
    const uint32_t id = entry.output_ids[0];
    if (id == kNoValue || std::find(boundary.output_ids.begin(), boundary.output_ids.end(), id) ==
                              boundary.output_ids.end())
      continue;
    duplicated_ids.insert(id);
    positions.push_back(entry.position);
  }
  if (positions.empty())
    return positions;
  std::sort(positions.begin(), positions.end());
  std::vector<uint32_t> output_ids;
  std::vector<std::string> outputs;
  for (std::size_t i = 0; i < boundary.output_ids.size(); ++i) {
    if (duplicated_ids.contains(boundary.output_ids[i]))
      continue;
    output_ids.push_back(boundary.output_ids[i]);
    outputs.push_back(std::move(boundary.outputs[i]));
  }
  boundary.output_ids = std::move(output_ids);
  boundary.outputs = std::move(outputs);
  return positions;
}

bool MakePlan(Index &index, const Regions &regions, Plan &plan) {
  if (regions.size() < 2 || regions[0].size() < 2)
    return false;
  plan.regions = regions;
  for (const Region &region : regions) {
    Boundary boundary = GetBoundary(index, region);
    plan.duplicated.push_back(DuplicateInputShapes(index, region, boundary));
    for (uint32_t id : boundary.input_ids)
      if (!CachedKnownBoundaryType(index, id))
        return false;
    for (uint32_t id : boundary.output_ids)
      if (!CachedKnownBoundaryType(index, id))
        return false;
    plan.groups.push_back(std::move(static_cast<OperatorGroup &>(boundary)));
  }
  return true;
}

struct Extension {
  std::size_t role;
  bool predecessor;
  std::size_t port;
  std::size_t other_port;
  const NodeProto *node;
};

std::vector<Extension> Extensions(Index &index, const Region &region) {
  std::unordered_set<const NodeProto *> inside(region.begin(), region.end());
  std::vector<Extension> extensions;
  for (std::size_t role = 0; role < region.size() && extensions.size() < kMaxExtensions; ++role) {
    const NodeProto &node = *region[role];
    const auto &dependencies = index.data.at(&node).dependencies;
    for (std::size_t port = 0; port < dependencies.size() && extensions.size() < kMaxExtensions;
         ++port) {
      const NodeProto *before = index.data.at(&node).producers[port];
      if (before == nullptr || inside.count(before) != 0 || !index.data.at(before).usable)
        continue;
      for (int out = 0; out < before->output_size(); ++out) {
        if (before->output()[out].value() == dependencies[port])
          extensions.push_back({role, true, port, static_cast<std::size_t>(out), before});
      }
    }
    for (int out = 0; out < node.output_size() && extensions.size() < kMaxExtensions; ++out) {
      const uint32_t id = index.data.at(&node).output_ids[out];
      if (id == kNoValue)
        continue;
      const auto &users = index.consumers[id];
      for (const NodeProto *after : users) {
        if (users.size() > kMaxExtensions || extensions.size() >= kMaxExtensions)
          break;
        if (inside.count(after) != 0 || !index.data.at(after).usable)
          continue;
        const auto &inputs = index.data.at(after).dependency_ids;
        for (std::size_t port = 0; port < inputs.size() && extensions.size() < kMaxExtensions;
             ++port) {
          if (inputs[port] == id)
            extensions.push_back({role, false, static_cast<std::size_t>(out), port, after});
        }
      }
    }
  }
  std::sort(extensions.begin(), extensions.end(), [&](const Extension &a, const Extension &b) {
    return std::tie(a.predecessor, a.role, a.port, a.other_port, index.data.at(a.node).signature) <
           std::tie(b.predecessor, b.role, b.port, b.other_port, index.data.at(b.node).signature);
  });
  extensions.erase(std::unique(extensions.begin(), extensions.end(),
                               [](const Extension &a, const Extension &b) {
                                 return a.node == b.node && a.role == b.role &&
                                        a.predecessor == b.predecessor && a.port == b.port &&
                                        a.other_port == b.other_port;
                               }),
                   extensions.end());
  extensions.erase(std::remove_if(extensions.begin(), extensions.end(),
                                  [&](const Extension &extension) {
                                    for (uint32_t id : index.data.at(extension.node).output_ids)
                                      if (id != kNoValue) {
                                        const auto *type = DeclaredType(index, index.names[id]);
                                        if (type != nullptr && !SupportedBoundaryType(*type))
                                          return true;
                                      }
                                    return false;
                                  }),
                   extensions.end());
  if (extensions.size() > kMaxExtensions)
    extensions.resize(kMaxExtensions);
  return extensions;
}

std::vector<uint32_t> NodeValues(const NodeData &entry) {
  std::vector<uint32_t> values = entry.input_ids;
  values.insert(values.end(), entry.output_ids.begin(), entry.output_ids.end());
  values.insert(values.end(), entry.dependency_ids.begin(), entry.dependency_ids.end());
  return values;
}

std::string MatchingKey(Index &index, const NodeProto &node) {
  std::string key = index.data.at(&node).signature;
  for (uint32_t id : NodeValues(index.data.at(&node))) {
    if (id == kNoValue) {
      key += "/omitted";
      continue;
    }
    const ValueKey &value = GetValueKey(index, index.names[id]);
    key += "/" + std::to_string(value.dtype);
    if (value.structured) {
      SignaturePart(key, value.type->SerializeAsString());
    } else if (value.shape != nullptr) {
      const auto &shape = value.shape->Shape();
      key += ":" + std::to_string(shape.Rank());
      for (std::size_t dim = 0; dim < shape.Rank(); ++dim) {
        key += ":";
        if (shape[dim].IsInt() && (value.constant || dim + 1 == shape.Rank()))
          key += std::to_string(shape[dim].AsInt());
        else
          key += "*";
      }
    }
  }
  return key;
}

std::string SuccessorKey(std::size_t output, std::size_t input, const std::string &signature) {
  return std::to_string(output) + ":" + std::to_string(input) + ":" + signature;
}

void IndexSuccessors(Index &index) {
  for (const auto &node : index.graph.Builder().Nodes()) {
    auto &entry = index.data.at(&node);
    if (entry.usable)
      entry.match_key = MatchingKey(index, node);
  }
  for (const auto &node : index.graph.Builder().Nodes()) {
    const auto &entry = index.data.at(&node);
    if (!entry.usable)
      continue;
    for (std::size_t input = 0; input < entry.producers.size(); ++input) {
      const NodeProto *producer = entry.producers[input];
      if (producer == nullptr)
        continue;
      auto &before = index.data.at(producer);
      for (std::size_t output = 0; output < before.output_ids.size(); ++output) {
        if (before.output_ids[output] != entry.dependency_ids[input])
          continue;
        const auto [it, inserted] =
            before.successors.emplace(SuccessorKey(output, input, entry.match_key), &node);
        if (!inserted && it->second != &node)
          it->second = nullptr;
      }
    }
  }
}

struct Occurrence {
  Region nodes;
  std::vector<uint32_t> values;
  std::unordered_map<uint32_t, std::size_t> slots;
};

struct Binding {
  const NodeProto *node = nullptr;
  std::vector<uint32_t> added;
};

// Each value has one slot shared by the family. Updating a new node's ports
// checks internal edges and external aliases without revisiting earlier nodes.
std::vector<std::size_t> BindingSlots(Index &index, const Occurrence &reference,
                                      const NodeProto &node) {
  std::unordered_map<uint32_t, std::size_t> added;
  std::vector<std::size_t> result;
  for (uint32_t value : NodeValues(index.data.at(&node))) {
    if (value == kNoValue) {
      result.push_back(kNoPosition);
      continue;
    }
    const auto existing = reference.slots.find(value);
    if (existing != reference.slots.end()) {
      result.push_back(existing->second);
    } else {
      const auto [it, inserted] = added.emplace(value, reference.values.size() + added.size());
      (void)inserted;
      result.push_back(it->second);
    }
  }
  return result;
}

bool Bind(Index &index, const Occurrence &occurrence, const NodeProto &node,
          const std::vector<std::size_t> &expected, Binding &binding) {
  const auto values = NodeValues(index.data.at(&node));
  if (values.size() != expected.size())
    return false;
  std::unordered_map<uint32_t, std::size_t> added_slots;
  for (std::size_t port = 0; port < values.size(); ++port) {
    const uint32_t value = values[port];
    const std::size_t slot = expected[port];
    if (slot == kNoPosition || value == kNoValue) {
      if (slot != kNoPosition || value != kNoValue)
        return false;
      continue;
    }
    if (slot < occurrence.values.size()) {
      if (occurrence.values[slot] != value)
        return false;
      continue;
    }
    if (occurrence.slots.count(value) != 0)
      return false;
    const auto [it, inserted] = added_slots.emplace(value, slot);
    if (!inserted && it->second != slot)
      return false;
    const std::size_t offset = slot - occurrence.values.size();
    if (offset >= binding.added.size())
      binding.added.resize(offset + 1, kNoValue);
    if (binding.added[offset] != kNoValue && binding.added[offset] != value)
      return false;
    binding.added[offset] = value;
  }
  binding.node = &node;
  return true;
}

void Commit(Occurrence &occurrence, Binding binding) {
  occurrence.nodes.push_back(binding.node);
  for (uint32_t value : binding.added) {
    occurrence.slots.emplace(value, occurrence.values.size());
    occurrence.values.push_back(value);
  }
}

const NodeProto *Neighbor(Index &index, const Occurrence &occurrence, const Extension &extension) {
  const auto &base = index.data.at(occurrence.nodes[extension.role]);
  if (extension.predecessor) {
    const NodeProto *node = base.producers[extension.port];
    if (node == nullptr)
      return nullptr;
    const auto &before = index.data.at(node);
    if (extension.other_port >= before.output_ids.size() ||
        before.output_ids[extension.other_port] != base.dependency_ids[extension.port])
      return nullptr;
    return node;
  }
  const auto found = base.successors.find(
      SuccessorKey(extension.port, extension.other_port, index.data.at(extension.node).match_key));
  return found == base.successors.end() ? nullptr : found->second;
}

void AppendFrontier(Index &index, const std::vector<Occurrence> &occurrences, std::size_t role,
                    std::deque<Extension> &frontier) {
  std::set<std::tuple<bool, std::size_t, std::size_t, std::string>> seen;
  for (const auto &occurrence : occurrences) {
    for (auto extension : Extensions(index, Region{occurrence.nodes[role]})) {
      extension.role = role;
      if (seen.emplace(extension.predecessor, extension.port, extension.other_port,
                       index.data.at(extension.node).match_key)
              .second)
        frontier.push_back(extension);
    }
  }
}

bool HasFrontierEdge(Index &index, const Occurrence &occurrence, const Extension &extension) {
  const auto &base = index.data.at(occurrence.nodes[extension.role]);
  if (extension.predecessor)
    return base.producers[extension.port] != nullptr;
  const auto prefix = SuccessorKey(extension.port, extension.other_port, "");
  return std::any_of(base.successors.begin(), base.successors.end(), [&](const auto &entry) {
    return entry.second != nullptr && entry.first.starts_with(prefix);
  });
}

struct Seed {
  Regions regions;
  std::size_t anchor_count = 0;
};

std::vector<Seed> IndexSeeds(Index &index) {
  std::map<std::string, Region> anchors;
  for (const auto &node : index.graph.Builder().Nodes()) {
    const auto &entry = index.data.at(&node);
    if (entry.usable && entry.anchor)
      anchors[entry.match_key].push_back(&node);
  }
  std::vector<Seed> seeds;
  for (const auto &[key, bucket] : anchors) {
    if (bucket.size() < 2)
      continue;
    std::map<std::string, Regions> motifs;
    for (const NodeProto *anchor : bucket) {
      for (const auto &extension : Extensions(index, Region{anchor})) {
        Region region{anchor, extension.node};
        std::string signature;
        signature += std::to_string(extension.predecessor) + ":" + std::to_string(extension.port) +
                     ":" + std::to_string(extension.other_port);
        for (const NodeProto *node : region)
          SignaturePart(signature, index.data.at(node).match_key);
        for (int64_t code : Topology(index, region))
          signature += "/" + std::to_string(code);
        motifs[signature].push_back(region);
      }
    }
    for (auto &[signature, regions] : motifs) {
      if (regions.size() >= 2)
        seeds.push_back({std::move(regions), bucket.size()});
    }
  }
  std::stable_sort(seeds.begin(), seeds.end(), [&](const Seed &left, const Seed &right) {
    if (left.anchor_count != right.anchor_count)
      return left.anchor_count < right.anchor_count;
    if (left.regions.size() != right.regions.size())
      return left.regions.size() > right.regions.size();
    if (left.regions[0].size() != right.regions[0].size())
      return left.regions[0].size() > right.regions[0].size();
    return index.data.at(left.regions[0][0]).position < index.data.at(right.regions[0][0]).position;
  });
  return seeds;
}

void Grow(Index &index, Regions seeds, std::vector<bool> &claimed, Plan &plan,
          std::size_t &attempts) {
  std::vector<Regions> pending;
  pending.push_back(std::move(seeds));
  while (!pending.empty() && attempts < index.options.max_growth_attempts) {
    Regions regions = std::move(pending.back());
    pending.pop_back();
    std::unordered_set<const NodeProto *> occupied;
    std::vector<Occurrence> occurrences;
    Occurrence reference;
    for (const Region &region : regions) {
      if (std::any_of(region.begin(), region.end(), [&](const NodeProto *node) {
            return claimed[index.data.at(node).position] || occupied.count(node) != 0;
          }))
        continue;
      Occurrence occurrence;
      if (reference.nodes.empty()) {
        for (const NodeProto *node : region) {
          Binding binding;
          const auto slots = BindingSlots(index, occurrence, *node);
          if (!Bind(index, occurrence, *node, slots, binding))
            throw core::builder::BuilderError("RepeatedLayer: inconsistent seed.");
          Commit(occurrence, std::move(binding));
        }
        reference = occurrence;
      } else {
        bool matches = true;
        for (std::size_t role = 0; role < region.size(); ++role) {
          Binding binding;
          const auto expected = BindingSlots(index, reference, *reference.nodes[role]);
          if (!CompatibleNode(index, *reference.nodes[role], *region[role]) ||
              !Bind(index, occurrence, *region[role], expected, binding)) {
            matches = false;
            break;
          }
          Commit(occurrence, std::move(binding));
        }
        if (!matches)
          continue;
      }
      occupied.insert(region.begin(), region.end());
      occurrences.push_back(std::move(occurrence));
    }
    if (occurrences.size() < 2)
      continue;
    std::deque<Extension> frontier;
    for (std::size_t role = 0; role < occurrences[0].nodes.size(); ++role)
      AppendFrontier(index, occurrences, role, frontier);
    while (!frontier.empty() && occurrences[0].nodes.size() < kMaxRegion &&
           attempts < index.options.max_growth_attempts) {
      Extension extension = frontier.front();
      frontier.pop_front();
      std::size_t reference_index = kNoPosition;
      for (std::size_t i = 0; i < occurrences.size(); ++i) {
        const NodeProto *node = Neighbor(index, occurrences[i], extension);
        if (node != nullptr && !occupied.contains(node) && !claimed[index.data.at(node).position] &&
            index.data.at(node).match_key == index.data.at(extension.node).match_key) {
          reference_index = i;
          extension.node = node;
          break;
        }
      }
      if (reference_index == kNoPosition)
        continue;
      ++attempts;
      const auto expected = BindingSlots(index, occurrences[reference_index], *extension.node);
      std::vector<Binding> bindings(occurrences.size());
      std::unordered_map<const NodeProto *, std::size_t> counts;
      for (std::size_t i = 0; i < occurrences.size(); ++i) {
        const NodeProto *node = Neighbor(index, occurrences[i], extension);
        if (node != nullptr && !claimed[index.data.at(node).position] &&
            occupied.count(node) == 0 && CompatibleNode(index, *extension.node, *node) &&
            Bind(index, occurrences[i], *node, expected, bindings[i]))
          ++counts[node];
      }
      const std::size_t matching =
          std::count_if(bindings.begin(), bindings.end(), [&](const Binding &binding) {
            return binding.node != nullptr && counts[binding.node] == 1;
          });
      if (matching < 2)
        continue;
      // A terminal occurrence limits the common period; a distinct repeated
      // neighborhood is a variant and can grow independently.
      if (matching + 1 == occurrences.size()) {
        const auto missing =
            std::find_if(bindings.begin(), bindings.end(), [&](const Binding &binding) {
              return binding.node == nullptr || counts[binding.node] != 1;
            });
        if (!HasFrontierEdge(index, occurrences[missing - bindings.begin()], extension))
          continue;
      }
      Regions remaining;
      std::vector<Occurrence> grown;
      for (std::size_t i = 0; i < occurrences.size(); ++i) {
        if (bindings[i].node == nullptr || counts[bindings[i].node] != 1) {
          remaining.push_back(occurrences[i].nodes);
          continue;
        }
        grown.push_back(std::move(occurrences[i]));
        Commit(grown.back(), std::move(bindings[i]));
      }
      if (remaining.size() >= 2)
        pending.push_back(remaining);
      for (const auto &region : remaining)
        for (const NodeProto *node : region)
          occupied.erase(node);
      occurrences = std::move(grown);
      for (const auto &occurrence : occurrences)
        occupied.insert(occurrence.nodes.back());
      AppendFrontier(index, occurrences, occurrences[0].nodes.size() - 1, frontier);
    }
    if (occurrences.size() < 2)
      continue;
    regions.clear();
    for (auto &occurrence : occurrences)
      regions.push_back(std::move(occurrence.nodes));
    Plan family;
    if (!MakePlan(index, regions, family))
      continue;
    const std::size_t first = plan.groups.size();
    for (std::size_t i = 0; i < regions.size(); ++i) {
      for (const NodeProto *node : regions[i])
        claimed[index.data.at(node).position] = true;
      plan.regions.push_back(std::move(regions[i]));
      plan.groups.push_back(std::move(family.groups[i]));
      plan.duplicated.push_back(std::move(family.duplicated[i]));
    }
    plan.families.emplace_back(first, plan.groups.size());
  }
}

bool FindPlan(Index &index, Plan &plan) {
  if (!index.valid)
    return false;
  IndexSuccessors(index);
  auto seeds = IndexSeeds(index);
  std::vector<bool> claimed(index.entries.size(), false);
  std::size_t attempts = 0;
  for (auto &seed : seeds) {
    if (++attempts >= index.options.max_growth_attempts)
      break;
    Grow(index, std::move(seed.regions), claimed, plan, attempts);
  }
  if (plan.groups.empty())
    return false;
  if (Schedule(index, plan))
    return true;
  // Only terminal cuts are scheduled. Drop entire families blocked by a
  // contraction cycle, never restart growth or replay already tested cuts.
  std::vector<bool> scheduled(plan.groups.size(), false);
  for (const auto &step : plan.schedule)
    if (step.is_group)
      scheduled[step.index] = true;
  Plan safe;
  for (const auto &[first, last] : plan.families) {
    if (!std::all_of(scheduled.begin() + first, scheduled.begin() + last,
                     [](bool emitted) { return emitted; }))
      continue;
    const std::size_t begin = safe.groups.size();
    for (std::size_t i = first; i < last; ++i) {
      safe.groups.push_back(std::move(plan.groups[i]));
      safe.regions.push_back(std::move(plan.regions[i]));
      safe.duplicated.push_back(std::move(plan.duplicated[i]));
    }
    safe.families.emplace_back(begin, safe.groups.size());
  }
  plan = std::move(safe);
  return !plan.groups.empty() && Schedule(index, plan);
}

ValueInfoProto ValueInfo(Index &index, const std::string &name) {
  GraphGraph &graph = index.graph;
  ValueInfoProto value;
  const auto declaration = index.declarations.find(name);
  if (declaration != index.declarations.end())
    value = *declaration->second;
  value.set_name(name);
  const auto &context = graph.Builder().Shapes();
  if (context.HasType(name) && KnownType(context.GetType(name)))
    *value.mutable_type() = context.GetType(name);
  if (graph.HasShape(name) && (!value.has_type() || value.type().has_tensor_type())) {
    core::symbolic::SymTensorToValueInfo(graph.GetShape(name), value);
  } else if (!value.has_type() && graph.HasType(name)) {
    value.mutable_type()->mutable_tensor_type()->set_elem_type(
        core::symbolic::TensorTypeToDataType(graph.GetType(name)));
  }
  const auto &tags = graph.Builder().Compute().ValueTags();
  const auto tag = tags.find(name);
  if (tag != tags.end()) {
    StringStringEntryProto *metadata = nullptr;
    for (auto &entry : *value.mutable_metadata_props()) {
      if (entry.key().value() == core::compute::kValueTagMetadataKey) {
        metadata = &entry;
        break;
      }
    }
    if (metadata == nullptr) {
      metadata = value.add_metadata_props();
      metadata->set_key(core::compute::kValueTagMetadataKey);
    }
    metadata->set_value(tag->second);
  }
  return value;
}

void ClearLayerAnalysis(GraphProto &scope,
                        const std::unordered_set<std::string> &literal_values = {}) {
  const auto clear_range = [&](ValueInfoProto &value) {
    if (literal_values.contains(value.name().value()))
      return;
    auto &metadata = value.ref_metadata_props();
    for (auto it = metadata.begin(); it != metadata.end();) {
      if (it->key() == core::symbolic::kValueInfoMinMetadataKey ||
          it->key() == core::symbolic::kValueInfoMaxMetadataKey)
        it = metadata.erase(it);
      else
        ++it;
    }
  };
  // Ranges describe caller values, not every invocation of a shared body.
  for (auto &input : scope.ref_input())
    clear_range(input);
  for (auto &output : scope.ref_output())
    clear_range(output);
  for (auto &info : scope.ref_value_info())
    clear_range(info);
  const std::unordered_set<std::string> computed_keys = {
      core::compute::kInPlaceReuseMetadataKey,
      core::compute::kReleaseAfterMetadataKey,
      core::compute::kNotUsedAfterMetadataKey,
      core::compute::kReleaseAfterShapeTagMetadataKey,
      core::compute::kNodePeakMemoryMetadataKey,
      core::compute::kNodePeakMemoryMaxMetadataKey,
      core::compute::kNodePeakMemoryAverageMetadataKey,
      core::compute::kConstantMetadataKey,
      core::compute::kLifetimeNamesEncodingMetadataKey};
  for (auto &node : scope.ref_node()) {
    auto &metadata = node.ref_metadata_props();
    for (auto it = metadata.begin(); it != metadata.end();) {
      if (computed_keys.contains(it->key().value()))
        it = metadata.erase(it);
      else
        ++it;
    }
  }
}

std::string MajorityInputPrefix(const std::vector<OperatorGroup> &boundaries, std::size_t port) {
  std::vector<std::string> names;
  for (const OperatorGroup &boundary : boundaries)
    names.push_back(boundary.inputs.at(port));
  std::sort(names.begin(), names.end());
  const std::size_t majority = names.size() / 2 + 1;
  std::string prefix;
  for (std::size_t first = 0; first + majority <= names.size(); ++first) {
    const std::string &left = names[first], &right = names[first + majority - 1];
    std::size_t length = 0;
    while (length < left.size() && length < right.size() && left[length] == right[length])
      ++length;
    // A shared byte prefix may end inside a UTF-8 code point.
    while (length > 0 && length < left.size() &&
           (static_cast<unsigned char>(left[length]) & 0xC0) == 0x80)
      --length;
    if (length > prefix.size())
      prefix = left.substr(0, length);
  }
  return prefix;
}

void RenameLayerInputs(GraphProto &body, GraphGraph &graph,
                       const std::vector<OperatorGroup> &boundaries) {
  const auto visit = [](const auto &self, GraphProto &scope, const auto &action) -> void {
    action(scope);
    for (auto &node : scope.ref_node())
      for (auto &attribute : node.ref_attribute()) {
        if (attribute.has_g())
          self(self, attribute.ref_g(), action);
        for (auto &nested : attribute.ref_graphs())
          self(self, nested, action);
      }
  };
  std::unordered_set<std::string> used;
  visit(visit, body, [&](const GraphProto &scope) {
    for (const auto &input : scope.input())
      used.insert(input.name());
    for (const auto &output : scope.output())
      used.insert(output.name());
    for (const auto &info : scope.value_info())
      used.insert(info.name());
    for (const auto &initializer : scope.initializer())
      used.insert(initializer.name());
    for (const auto &initializer : scope.sparse_initializer())
      used.insert(initializer.values().name());
    for (const auto &initializer : scope.extended_initializer()) {
      if (initializer.has_encoded_value())
        used.insert(initializer.encoded_value().name());
      if (initializer.has_paged_cache())
        used.insert(initializer.paged_cache().name());
    }
    for (const auto &node : scope.node()) {
      used.insert(node.input().begin(), node.input().end());
      used.insert(node.output().begin(), node.output().end());
    }
  });
  // Graph attributes obey lexical SSA even though Layer binds its inputs positionally.
  std::unordered_map<std::string, std::string> names;
  for (std::size_t i = 0; i < body.input().size(); ++i) {
    std::string base = MajorityInputPrefix(boundaries, i);
    if (base.empty())
      base = "input_" + std::to_string(i);
    std::string name = base;
    if (used.contains(name) || graph.Builder().HasName(name))
      name = base + "_" + std::to_string(i);
    const std::string unique_base = name;
    for (std::size_t suffix = 0; used.contains(name) || graph.Builder().HasName(name); ++suffix)
      name = unique_base + "_" + std::to_string(suffix);
    used.insert(name);
    names.emplace(body.input(i).name().value(), std::move(name));
  }
  visit(visit, body, [&](GraphProto &scope) {
    ClearLayerAnalysis(scope);
    const auto rename_value = [&](auto &value) {
      const auto found = names.find(value.name().value());
      if (found != names.end())
        value.set_name(found->second);
    };
    for (auto &input : scope.ref_input())
      rename_value(input);
    for (auto &output : scope.ref_output())
      rename_value(output);
    for (auto &info : scope.ref_value_info())
      rename_value(info);
    for (auto &node : scope.ref_node())
      for (auto &input : node.ref_input()) {
        const auto found = names.find(input.value());
        if (found != names.end())
          input = found->second;
      }
  });
}

bool NormalizeLayerBody(GraphProto &body, const GraphBuilder &builder, SignatureNames names,
                        std::size_t depth, std::size_t &work, bool reject_ranges) {
  if (depth >= kMaxScopeDepth || !body.sparse_initializer().empty() ||
      !body.extended_initializer().empty() || !body.quantization_annotation().empty() ||
      !body.persistent_bindings().empty() || !SafeGraphDeclarations(body))
    return false;
  // Literal ranges are compared exactly, rather than rejected as caller-specific.
  std::unordered_set<std::string> literal_values;
  for (const auto &node : body.node())
    if (NormaliseDomain(node.domain()) == "ai.onnx" && node.op_type() == "Constant")
      literal_values.insert(node.output().begin(), node.output().end());
  if (reject_ranges) {
    const auto has_range = [&](const ValueInfoProto &value) {
      if (literal_values.contains(value.name().value()))
        return false;
      return std::any_of(value.metadata_props().begin(), value.metadata_props().end(),
                         [](const auto &entry) {
                           return entry.key() == core::symbolic::kValueInfoMinMetadataKey ||
                                  entry.key() == core::symbolic::kValueInfoMaxMetadataKey;
                         });
    };
    if (std::any_of(body.input().begin(), body.input().end(), has_range) ||
        std::any_of(body.output().begin(), body.output().end(), has_range) ||
        std::any_of(body.value_info().begin(), body.value_info().end(), has_range))
      return false;
  }
  ClearLayerAnalysis(body, literal_values);
  std::unordered_set<std::string> inferred;
  for (const auto &node : body.node())
    if (core::shapes::HasShapeInference(builder.Shapes(), node))
      inferred.insert(node.output().begin(), node.output().end());
  const auto normalize_value = [&](ValueInfoProto &value) {
    value.clear_doc_string();
    // Recomputed shapes belong to each call, not to the function's identity.
    if (inferred.contains(value.name().value()) && value.type().has_tensor_type())
      value.mutable_type()->mutable_tensor_type()->clear_shape();
  };
  for (auto &input : body.ref_input())
    normalize_value(input);
  for (auto &output : body.ref_output())
    normalize_value(output);
  for (auto &info : body.ref_value_info())
    normalize_value(info);
  body.clear_name();
  const std::string scope = "scope" + std::to_string(depth) + ":";
  std::unordered_set<std::string> formal_inputs;
  for (std::size_t i = 0; i < body.input().size(); ++i)
    formal_inputs.insert(body.input(i).name().value());
  for (std::size_t i = 0; i < body.input().size(); ++i)
    names[body.input(i).name().value()] = scope + "input" + std::to_string(i);
  for (std::size_t i = 0; i < body.initializer().size(); ++i) {
    auto &initializer = *body.mutable_initializer(static_cast<int>(i));
    if (!MaterializeLayerTensor(initializer))
      return false;
    const std::string name = initializer.name().value();
    if (!formal_inputs.contains(name))
      names[name] = scope + "weight" + std::to_string(i);
    initializer.set_name(names.at(name));
  }
  const auto rename = [&](auto &value) {
    const auto found = names.find(value.name().value());
    if (found == names.end())
      return false;
    value.set_name(found->second);
    return true;
  };
  for (auto &input : body.ref_input())
    if (!rename(input))
      return false;
  for (std::size_t i = 0; i < body.node().size(); ++i) {
    if (++work > kMaxFunctionWork)
      return false;
    auto &node = *body.mutable_node(static_cast<int>(i));
    // Device annotations may themselves refer to value names.
    if (!node.device_configurations().empty())
      return false;
    node.clear_name();
    node.clear_doc_string();
    // Recomputed node tags and exporter provenance do not change the computation.
    auto &metadata = node.ref_metadata_props();
    for (auto it = metadata.begin(); it != metadata.end();) {
      if (it->key() == core::compute::kNodeTagMetadataKey ||
          it->key() == core::compute::kValueTagMetadataKey || it->key() == "namespace" ||
          it->key() == "pkg.onnxscript.class_hierarchy" ||
          it->key() == "pkg.onnxscript.name_scopes" || it->key() == "pkg.onnxscript.stack_trace" ||
          it->key() == "pkg.onnxscript.shape_inference_error")
        it = metadata.erase(it);
      else
        ++it;
    }
    for (auto &input : node.ref_input()) {
      if (input.value().empty())
        continue;
      const auto found = names.find(input.value());
      if (found == names.end())
        return false;
      input = found->second;
    }
    if (node.attribute().size() > 1) {
      std::vector<std::shared_ptr<AttributeProto>> attributes;
      attributes.reserve(node.attribute().size());
      for (std::size_t index = 0; index < node.attribute().size(); ++index)
        attributes.push_back(node.attribute().shared_at(index));
      std::sort(attributes.begin(), attributes.end(), [](const auto &left, const auto &right) {
        return left->name().value() < right->name().value();
      });
      for (std::size_t index = 0; index < attributes.size(); ++index)
        node.ref_attribute().get(index) = std::move(attributes[index]);
    }
    for (auto &attribute : node.ref_attribute()) {
      if ((attribute.has_t() && !MaterializeLayerTensor(attribute.ref_t())) ||
          !std::all_of(attribute.ref_tensors().begin(), attribute.ref_tensors().end(),
                       MaterializeLayerTensor))
        return false;
      const auto comparable_sparse = [](SparseTensorProto &tensor) {
        return MaterializeLayerTensor(tensor.ref_values()) &&
               MaterializeLayerTensor(tensor.ref_indices());
      };
      if ((attribute.has_sparse_tensor() && !comparable_sparse(attribute.ref_sparse_tensor())) ||
          !std::all_of(attribute.ref_sparse_tensors().begin(), attribute.ref_sparse_tensors().end(),
                       comparable_sparse))
        return false;
      if (attribute.has_g() &&
          !NormalizeLayerBody(attribute.ref_g(), builder, names, depth + 1, work, reject_ranges))
        return false;
      for (auto &nested : attribute.ref_graphs())
        if (!NormalizeLayerBody(nested, builder, names, depth + 1, work, reject_ranges))
          return false;
    }
    for (std::size_t port = 0; port < node.output().size(); ++port) {
      auto &output = *node.mutable_output(static_cast<int>(port));
      if (!output.value().empty()) {
        const std::string name = scope + "node" + std::to_string(i) + ":" + std::to_string(port);
        names[output.value()] = name;
        output = name;
      }
    }
  }
  for (auto &output : body.ref_output())
    if (!rename(output))
      return false;
  for (auto &info : body.ref_value_info())
    if (!rename(info))
      return false;
  return true;
}

bool CollectLayerDomains(const GraphProto &body, std::unordered_set<std::string> &domains,
                         std::size_t depth, std::size_t &work) {
  if (depth >= kMaxScopeDepth)
    return false;
  for (const auto &node : body.node()) {
    if (++work > kMaxFunctionWork)
      return false;
    domains.insert(node.domain().value());
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g() && !CollectLayerDomains(attribute.g(), domains, depth + 1, work))
        return false;
      for (const auto &nested : attribute.graphs())
        if (!CollectLayerDomains(nested, domains, depth + 1, work))
          return false;
    }
  }
  return true;
}

bool CompatibleLayerOpsets(const GraphBuilder &candidate, const GraphBuilder &source,
                           const GraphProto &body) {
  std::unordered_set<std::string> used_domains;
  std::size_t work = 0;
  if (!CollectLayerDomains(body, used_domains, 0, work))
    return false;
  for (const auto &[domain, version] : source.OpsetVersions()) {
    const auto found = candidate.OpsetVersions().find(domain);
    if (found == candidate.OpsetVersions().end() || found->second != version)
      return false;
  }
  for (const auto &[domain, version] : candidate.OpsetVersions()) {
    if (!source.OpsetVersions().contains(domain) && used_domains.contains(domain))
      return false;
  }
  return true;
}

std::size_t OperatorGroupFingerprint(const GraphProto &body) {
  std::size_t hash = 0;
  const auto add = [&](const std::string &value) {
    hash ^= std::hash<std::string>{}(value) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
  };
  for (const auto *values : {&body.input(), &body.output(), &body.value_info()}) {
    add(std::to_string(values->size()));
    for (const auto &value : *values)
      add(value.SerializeAsString());
  }
  for (const auto &initializer : body.initializer()) {
    add(initializer.name().value());
    add(std::to_string(initializer.ContentHash(true)));
  }
  for (const auto &node : body.node()) {
    add(node.Signature({node.input().begin(), node.input().end()}));
    for (const auto &output : node.output())
      add(output.value());
    for (const auto &metadata : node.metadata_props())
      add(metadata.SerializeAsString());
  }
  return hash;
}

} // namespace

std::string OperatorGroupFunctions::Find(const GraphProto &body) {
  GraphProto normalized = utils::CopyWithSharedTensorStorage(body);
  std::size_t work = 0;
  if (!NormalizeLayerBody(normalized, source_, {}, 0, work, false))
    return {};
  GraphBuilder &builder = source_.ModelBuilder();
  const auto &functions = builder.LocalFunctions();
  while (indexed_functions_ < functions.size()) {
    const auto &local = functions[indexed_functions_++];
    if (!local->IsLocalFunction(local->name(), "ai.rt", "") ||
        !(local->name() == prefix_ || local->name().rfind(prefix_ + "_", 0) == 0) ||
        local->Nodes().size() > kMaxFunctionWork || local->Inputs().size() > kMaxPorts ||
        local->Outputs().size() > kMaxPorts)
      continue;
    const auto *definition = builder.Shapes().GetLocalFunction(
        core::shapes::ShapesContext::LocalFunctionKey("ai.rt", local->name(), ""));
    if (definition == nullptr || !definition->attribute().empty() ||
        !definition->attribute_proto().empty())
      continue;
    GraphProto candidate = local->BuildGraph();
    work = 0;
    ++stats_.candidate_normalizations;
    if (NormalizeLayerBody(candidate, *local, {}, 0, work, true) &&
        CompatibleLayerOpsets(*local, source_, candidate)) {
      const auto fingerprint = OperatorGroupFingerprint(candidate);
      definitions_[fingerprint].emplace_back(local->name(), std::move(candidate));
    }
  }
  const auto found = definitions_.find(OperatorGroupFingerprint(normalized));
  if (found != definitions_.end())
    for (const auto &[name, candidate] : found->second) {
      ++stats_.exact_comparisons;
      if (normalized.Equals(candidate))
        return name;
    }
  return {};
}

namespace {

NodeProto MakeLayer(Index &index, const OperatorGroup &boundary,
                    const std::vector<OperatorGroup> &boundaries, std::size_t occurrence,
                    const std::vector<OperatorGroupInput> &producers, std::size_t &suffix,
                    OperatorGroupFunctions &functions) {
  GraphProto body;
  for (const std::string &name : boundary.inputs)
    *body.add_input() = ValueInfo(index, name);
  for (const std::string &name : boundary.outputs)
    *body.add_output() = ValueInfo(index, name);
  const std::unordered_set<std::string> outputs(boundary.outputs.begin(), boundary.outputs.end());
  for (const auto position : boundary.node_indices) {
    const NodeProto *node = &index.graph.Builder().Nodes()[position];
    *body.add_node() = utils::CopyWithSharedTensorStorage(index.data.at(node).materialized);
    for (const auto &output : node->output()) {
      if (!output.value().empty() && outputs.count(output.value()) == 0) {
        ValueInfoProto info = ValueInfo(index, output.value());
        if (info.has_type() && KnownType(info.type()))
          *body.add_value_info() = std::move(info);
      }
    }
  }
  RenameLayerInputs(body, index.graph, boundaries);
  const auto inputs = InlineOperatorGroupInputs(body, boundary, producers);
  core::builder::GraphBuilder &source = index.graph.Builder();
  source.SetOpsetVersion("ai.rt", 1);
  const std::string name = "RepeatedLayer_n" + std::to_string(occurrence) + "_p" +
                           std::to_string(boundary.node_indices.front());
  const std::string function_name = functions.Add(std::move(body), suffix);
  return MakeNode(function_name.c_str(), inputs, boundary.outputs, "ai.rt", name.c_str());
}

OperatorGroupStatus Eligibility(const GraphGraph &graph) {
  if (IsLayerName("ai.rt", graph.Builder().name()) &&
      graph.Builder().IsLocalFunction(graph.Builder().name(), "ai.rt"))
    return OperatorGroupStatus::kInsideFunction;
  for (const auto &entry : graph.Builder().ModelBuilder().Shapes().LocalFunctions()) {
    const auto &function = entry.second;
    if (IsLayerName(function->domain().value(), function->name().value()) &&
        graph.Builder().IsInsideLocalFunction(function->name().value(), function->domain().value(),
                                              function->overload().value()))
      return OperatorGroupStatus::kInsideFunction;
  }
  const int version = graph.Builder().OpsetVersion("ai.rt");
  if (version > 0 && version != 1)
    return OperatorGroupStatus::kIncompatibleOpset;
  return OperatorGroupStatus::kFound;
}

utils::RepeatedProtoField<NodeProto> MakeReplacements(Index &index, const Plan &proposal) {
  auto &graph = index.graph;
  std::vector<std::vector<OperatorGroup>> families;
  std::vector<std::vector<OperatorGroupInput>> producers;
  std::vector<std::size_t> group_family(proposal.groups.size());
  for (const auto &[first, last] : proposal.families) {
    std::fill(group_family.begin() + first, group_family.begin() + last, families.size());
    families.emplace_back(proposal.groups.begin() + first, proposal.groups.begin() + last);
    producers.push_back(FindCommonOperatorGroupInputs(graph, families.back()));
  }
  std::size_t suffix = 0;
  OperatorGroupFunctions functions(graph.Builder(), "Layer");
  auto replacement = MakeOperatorGroupReplacements(graph, proposal, [&](std::size_t group) {
    const auto family = group_family[group];
    return MakeLayer(index, proposal.groups[group], families[family], group, producers[family],
                     suffix, functions);
  });
  if (std::all_of(proposal.duplicated.begin(), proposal.duplicated.end(),
                  [](const auto &positions) { return positions.empty(); }))
    return replacement;
  utils::RepeatedProtoField<NodeProto> expanded;
  for (std::size_t step = 0; step < proposal.schedule.size(); ++step) {
    const auto &[is_group, group] = proposal.schedule[step];
    if (is_group)
      for (std::size_t position : proposal.duplicated[group])
        expanded.push_back(utils::CopyWithSharedTensorStorage(graph.Builder().Nodes()[position]));
    expanded.push_back(std::move(replacement[step]));
  }
  return expanded;
}

} // namespace

OperatorGroupProposal
ProposeRepeatedLayers(GraphGraph &graph,
                      std::shared_ptr<const builder::PatternApplication> *application) {
  return ProposeRepeatedLayers(graph, RepeatedLayerOptions{}, application);
}

OperatorGroupProposal
ProposeRepeatedLayers(GraphGraph &graph, const RepeatedLayerOptions &options,
                      std::shared_ptr<const builder::PatternApplication> *application) {
  if (options.max_growth_attempts == 0)
    throw std::invalid_argument("RepeatedLayer requires max_growth_attempts > 0.");
  if (application != nullptr)
    application->reset();
  if (graph.Builder().Nodes().empty())
    return {};
  const auto status = Eligibility(graph);
  if (status != OperatorGroupStatus::kFound) {
    OperatorGroupProposal proposal;
    proposal.status = status;
    return proposal;
  }
  auto index = std::make_shared<Index>(graph, options);
  Plan plan;
  const bool found = FindPlan(*index, plan);
  plan.capture_analysis_limited = index->last_incomplete_capture != kNoPosition;
  if (!found) {
    OperatorGroupProposal proposal;
    proposal.capture_analysis_limited = plan.capture_analysis_limited;
    if (proposal.capture_analysis_limited)
      proposal.status = OperatorGroupStatus::kResourceLimit;
    return proposal;
  }
  plan.status = OperatorGroupStatus::kFound;
  auto proposal = static_cast<const OperatorGroupProposal &>(plan);
  if (application != nullptr)
    *application = std::make_shared<builder::PatternApplication>(
        [index = std::move(index), plan = std::move(plan)]() {
          return MakeReplacements(*index, plan);
        });
  return proposal;
}

utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) {
  return MakeRepeatedLayerReplacements(graph, nodes, RepeatedLayerOptions{});
}

utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(GraphGraph &graph, const std::vector<const NodeProto *> &nodes,
                              const RepeatedLayerOptions &options) {
  if (options.max_growth_attempts == 0)
    throw std::invalid_argument("RepeatedLayer requires max_growth_attempts > 0.");
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("RepeatedLayer::Apply expects a nonempty match.");
  graph.Position(*nodes[0]);
  if (Eligibility(graph) != OperatorGroupStatus::kFound)
    throw core::builder::BuilderError("RepeatedLayer::Apply could not reproduce its regions.");
  Index index(graph, options);
  Plan plan;
  if (!FindPlan(index, plan))
    throw core::builder::BuilderError("RepeatedLayer::Apply could not reproduce its regions.");
  plan.status = OperatorGroupStatus::kFound;
  if (plan.MatchedNodes(graph.Builder().Nodes()) != nodes)
    throw core::builder::BuilderError("RepeatedLayer::Apply received an inconsistent match.");
  return MakeReplacements(index, plan);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
