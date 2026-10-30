// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/layer_extraction.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
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

constexpr std::size_t kMaxOccurrences = 32;
constexpr std::size_t kMaxRegion = 256;
constexpr std::size_t kMaxInterval = 4096;
constexpr std::size_t kMaxFunctionWork = 4096;
constexpr std::size_t kMaxScopeDepth = 16;
constexpr std::size_t kMaxSignatureBytes = 65536;
constexpr std::size_t kMaxPorts = 128;
constexpr std::size_t kMaxExtensions = 256;
constexpr std::size_t kMaxAnchorStride = 8;
constexpr std::size_t kMaxGrowthAttempts = 4096;
constexpr std::size_t kMaxSeedGroups = 32;
using Region = std::vector<const NodeProto *>;
using Regions = std::vector<Region>;

bool IsLayerName(const std::string &domain, const std::string &name) {
  return domain == "ai.rt" && (name == "Layer" || name.rfind("Layer_", 0) == 0);
}

bool IsLayer(const NodeProto &node) {
  return IsLayerName(node.domain().value(), node.op_type().value());
}

bool IsPrimitiveAnchor(const NodeProto &node) {
  const std::string &domain = node.domain().value();
  const std::string &op = node.op_type().value();
  return node.overload().value().empty() &&
         (((domain.empty() || domain == "ai.onnx") && (op == "MatMul" || op == "Attention")) ||
          (domain == "com.microsoft" && op == "GroupQueryAttention") ||
          (domain == "ai.rt" && op == "PagedAttention"));
}

// The key includes domain and overload: similarly named user operators are not anchors.
bool IsAnchor(const NodeProto &node, const GraphBuilder &builder, std::size_t depth,
              std::size_t &work, std::set<std::string> &visiting, bool in_function = false) {
  if (IsLayer(node))
    return false;
  const std::string key = core::shapes::ShapesContext::LocalFunctionKey(
      node.domain().value(), node.op_type().value(), node.overload().value());
  const FunctionProto *function = builder.Shapes().GetLocalFunction(key);
  if (function == nullptr && IsPrimitiveAnchor(node))
    return true;
  if (depth >= kMaxScopeDepth || work >= kMaxFunctionWork)
    return false;
  bool found = false;
  if (function != nullptr && visiting.insert(key).second) {
    for (const NodeProto &inner : function->node()) {
      if (++work > kMaxFunctionWork)
        break;
      if (IsAnchor(inner, builder, depth + 1, work, visiting, true)) {
        found = true;
        break;
      }
    }
    visiting.erase(key);
  }
  if (!found && in_function) {
    auto search = [&](const GraphProto &graph) {
      for (const NodeProto &inner : graph.node()) {
        if (++work > kMaxFunctionWork)
          return false;
        if (IsAnchor(inner, builder, depth + 1, work, visiting, true))
          return true;
      }
      return false;
    };
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g() && search(attribute.g()))
        return true;
      for (const GraphProto &nested : attribute.graphs()) {
        if (search(nested))
          return true;
      }
    }
  }
  return found;
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

bool AttributeSignature(const AttributeProto &attribute, std::string &signature,
                        const SignatureNames &outer, std::size_t depth, std::size_t &work) {
  // Tensor attribute values constrain matching exactly, without serializing
  // unbounded parameter payloads. Top-level Constant producers stay external.
  auto bounded_tensor = [](const TensorProto &tensor) {
    if (!HasComparableLayerTensorContent(tensor) || tensor.raw_data().size() > kMaxSignatureBytes ||
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
  if (attribute.has_t() && !bounded_tensor(attribute.t()))
    return false;
  for (const auto &tensor : attribute.tensors()) {
    if (!bounded_tensor(tensor))
      return false;
  }
  if (attribute.has_sparse_tensor() && (!bounded_tensor(attribute.sparse_tensor().values()) ||
                                        !bounded_tensor(attribute.sparse_tensor().indices())))
    return false;
  for (const auto &tensor : attribute.sparse_tensors()) {
    if (!bounded_tensor(tensor.values()) || !bounded_tensor(tensor.indices()))
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
  std::vector<std::string> dependencies;
  bool usable = false;
  bool anchor = false;
};

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

struct Index {
  GraphGraph &graph;
  std::unordered_map<const NodeProto *, NodeData> data;
  std::unordered_map<std::string, std::vector<const NodeProto *>> users;
  std::map<std::string, Region> anchors;
  std::unordered_map<std::string, const ValueInfoProto *> declarations;
  bool valid = true;

  explicit Index(GraphGraph &graph) : graph(graph) {
    for (const auto &input : graph.Builder().Inputs())
      declarations[input.name().value()] = &input;
    for (const auto &output : graph.Builder().Outputs())
      declarations[output.name().value()] = &output;
    std::map<std::tuple<std::string, std::string, std::string>, bool> anchor_cache;
    std::size_t function_work = 0, capture_work = 0, signature_work = 0, safety_work = 0;
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
      if (!IsLayer(node) &&
          !NodeCaptures(entry.materialized, entry.dependencies, 0, capture_work)) {
        valid = false;
        return;
      }
      for (const std::string &name : entry.dependencies)
        users[name].push_back(&node);
      // Constant nodes remain outside the body, just like initializers.
      entry.usable = !IsLayer(node) && node.op_type().value() != "Constant" &&
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
        for (const auto &name : node.output()) {
          if (!name.empty())
            outputs.push_back(ValueInfo(*this, name));
        }
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
          std::set<std::string> visiting;
          it->second = IsAnchor(node, graph.Builder(), 0, function_work, visiting);
        }
        entry.anchor = it->second;
        if (entry.anchor)
          anchors[entry.signature].push_back(&node);
      }
      data.emplace(&node, std::move(entry));
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

bool CompatibleValue(Index &index, const std::string &left, const std::string &right) {
  GraphGraph &graph = index.graph;
  if (left.empty() || right.empty())
    return left.empty() && right.empty();
  const TypeProto *left_type = DeclaredType(index, left);
  const TypeProto *right_type = DeclaredType(index, right);
  const bool left_structured = left_type != nullptr && core::shapes::HasStructuredType(*left_type);
  const bool right_structured =
      right_type != nullptr && core::shapes::HasStructuredType(*right_type);
  if (left_structured || right_structured)
    return left_structured && right_structured &&
           core::shapes::SameDeclaredType(*left_type, *right_type);
  auto dtype = [&](const std::string &name, const TypeProto *type) {
    if (type != nullptr && type->has_tensor_type())
      return type->tensor_type().elem_type();
    return graph.HasType(name) ? core::symbolic::TensorTypeToDataType(graph.GetType(name))
                               : TensorProto::DataType::UNDEFINED;
  };
  if (dtype(left, left_type) != dtype(right, right_type) ||
      graph.HasShape(left) != graph.HasShape(right))
    return false;
  if (!graph.HasShape(left))
    return true;
  const auto &a = graph.GetShape(left).Shape();
  const auto &b = graph.GetShape(right).Shape();
  if (a.Rank() != b.Rank())
    return false;
  const bool activation = !graph.IsConstant(left) && !graph.IsConstant(right);
  for (std::size_t i = 0; i < a.Rank(); ++i) {
    if (a[i].IsInt() && b[i].IsInt() && a[i].AsInt() != b[i].AsInt() &&
        !(activation && i + 1 < a.Rank()))
      return false;
  }
  return true;
}

bool CompatibleNode(Index &index, const NodeProto &left, const NodeProto &right) {
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

// Corresponding values have the same internal producer/output port or external
// alias class. This also detects additional edges between already matched nodes.
std::vector<std::string> Topology(Index &index, const Region &region) {
  std::unordered_map<std::string, std::string> values;
  for (std::size_t role = 0; role < region.size(); ++role) {
    for (int port = 0; port < region[role]->output_size(); ++port) {
      const std::string &name = region[role]->output()[port].value();
      if (!name.empty())
        values.emplace(name, "n" + std::to_string(role) + ":" + std::to_string(port));
    }
  }
  std::size_t external = 0;
  std::vector<std::string> topology;
  for (const NodeProto *node : region) {
    for (const auto &input : node->input()) {
      if (input.value().empty()) {
        topology.push_back("-");
      } else {
        auto [it, inserted] = values.emplace(input.value(), "e" + std::to_string(external));
        if (inserted)
          ++external;
        topology.push_back(it->second);
      }
    }
    const auto &dependencies = index.data.at(node).dependencies;
    for (std::size_t i = 0; i < dependencies.size(); ++i) {
      auto [it, inserted] = values.emplace(dependencies[i], "e" + std::to_string(external));
      if (inserted)
        ++external;
      topology.push_back(it->second);
    }
  }
  return topology;
}

bool Correspond(Index &index, const Regions &regions) {
  std::unordered_set<const NodeProto *> used;
  const auto topology = Topology(index, regions[0]);
  for (const Region &region : regions) {
    for (std::size_t role = 0; role < region.size(); ++role) {
      if (!used.insert(region[role]).second ||
          !CompatibleNode(index, *regions[0][role], *region[role]))
        return false;
    }
    if (Topology(index, region) != topology)
      return false;
  }
  return true;
}

struct Boundary : OperatorGroup {
  std::vector<std::string> edges;
};

Boundary GetBoundary(Index &index, const Region &region) {
  Boundary boundary;
  std::unordered_set<const NodeProto *> inside(region.begin(), region.end());
  std::unordered_set<std::string> produced, inputs, outputs;
  Region ordered = region;
  std::sort(ordered.begin(), ordered.end(), [&](const NodeProto *a, const NodeProto *b) {
    return index.graph.Position(*a) < index.graph.Position(*b);
  });
  for (const NodeProto *node : ordered) {
    const std::size_t position = index.graph.Position(*node);
    boundary.node_indices.push_back(position);
    for (const auto &output : node->output()) {
      if (!output.value().empty())
        produced.insert(output.value());
    }
  }
  for (const NodeProto *node : ordered) {
    for (const std::string &name : index.data.at(node).dependencies) {
      if (produced.count(name) == 0) {
        boundary.edges.push_back(name);
        if (inputs.insert(name).second)
          boundary.inputs.push_back(name);
      }
    }
    for (const auto &output : node->output()) {
      const std::string &name = output.value();
      if (name.empty())
        continue;
      bool external = index.graph.IsOutput(name);
      if (external)
        boundary.edges.push_back(name);
      const auto users = index.users.find(name);
      if (users != index.users.end()) {
        for (const NodeProto *user : users->second) {
          if (inside.count(user) == 0) {
            external = true;
            boundary.edges.push_back(name);
          }
        }
      }
      if (external && outputs.insert(name).second)
        boundary.outputs.push_back(name);
    }
  }
  return boundary;
}

uint64_t SaturatingAdd(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max()
                                                      : a + b;
}

uint64_t SaturatingMultiply(uint64_t a, uint64_t b) {
  return a != 0 && b > std::numeric_limits<uint64_t>::max() / a
             ? std::numeric_limits<uint64_t>::max()
             : a * b;
}

bool ValueBytes(GraphGraph &graph, const std::string &name, uint64_t &bytes) {
  if (!graph.HasType(name) || !graph.HasShape(name))
    return false;
  const auto type = core::symbolic::TensorTypeToDataType(graph.GetType(name));
  uint64_t bits = 0;
  switch (type) {
  case TensorProto::DataType::BOOL:
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::FLOAT8E4M3FN:
  case TensorProto::DataType::FLOAT8E4M3FNUZ:
  case TensorProto::DataType::FLOAT8E5M2:
  case TensorProto::DataType::FLOAT8E5M2FNUZ:
  case TensorProto::DataType::FLOAT8E8M0:
    bits = 8;
    break;
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::UINT16:
  case TensorProto::DataType::FLOAT16:
  case TensorProto::DataType::BFLOAT16:
    bits = 16;
    break;
  case TensorProto::DataType::FLOAT:
  case TensorProto::DataType::INT32:
  case TensorProto::DataType::UINT32:
    bits = 32;
    break;
  case TensorProto::DataType::DOUBLE:
  case TensorProto::DataType::INT64:
  case TensorProto::DataType::UINT64:
  case TensorProto::DataType::COMPLEX64:
    bits = 64;
    break;
  case TensorProto::DataType::COMPLEX128:
    bits = 128;
    break;
  case TensorProto::DataType::UINT4:
  case TensorProto::DataType::INT4:
  case TensorProto::DataType::FLOAT4E2M1:
    bits = 4;
    break;
  case TensorProto::DataType::UINT2:
  case TensorProto::DataType::INT2:
    bits = 2;
    break;
  default:
    return false;
  }
  uint64_t count = 1;
  bool overflow = false;
  const auto &shape = graph.GetShape(name).Shape();
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt() || shape[i].AsInt() < 0)
      return false;
    const auto dimension = static_cast<uint64_t>(shape[i].AsInt());
    if (dimension == 0) {
      bytes = 0;
      return true;
    }
    overflow |= count > std::numeric_limits<uint64_t>::max() / dimension;
    count = SaturatingMultiply(count, dimension);
  }
  if (overflow) {
    bytes = std::numeric_limits<uint64_t>::max();
  } else if (bits >= 8) {
    bytes = SaturatingMultiply(count, bits / 8);
  } else {
    bytes = SaturatingAdd(SaturatingMultiply(count / 8, bits), (count % 8 * bits + 7) / 8);
  }
  return true;
}

struct Plan : OperatorGroupProposal {
  Regions regions;
  std::size_t anchor_roles = 0;
  std::size_t family_occurrences = 0;
  std::tuple<uint64_t, uint64_t, uint64_t, std::size_t, std::vector<std::size_t>> score;
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

bool MovableConstantPreparation(Index &index, const NodeProto &node) {
  if ((!node.domain().empty() && node.domain() != "ai.onnx") || !node.overload().empty() ||
      !node.device_configurations().empty() ||
      index.graph.Builder().Shapes().GetLocalFunction(core::shapes::ShapesContext::LocalFunctionKey(
          node.domain(), node.op_type(), "")) != nullptr)
    return false;
  if (node.op_type() == "Constant")
    return node.input().empty();
  if (node.op_type() != "Cast" && node.op_type() != "CastLike" && node.op_type() != "Identity")
    return false;
  return std::all_of(node.input().begin(), node.input().end(), [&](const auto &input) {
    return !input.empty() && index.graph.IsConstant(input);
  });
}

// A unit is either a contracted region or one untouched interleaving node.
// Sorting the contracted DAG rejects re-entry cycles, even through nested captures.
bool Schedule(Index &index, Plan &plan) {
  plan.first = std::numeric_limits<std::size_t>::max();
  plan.last = 0;
  std::unordered_map<const NodeProto *, std::size_t> owner;
  std::vector<std::size_t> priority;
  for (std::size_t i = 0; i < plan.regions.size(); ++i) {
    const OperatorGroup &boundary = plan.groups[i];
    if (boundary.outputs.empty())
      return false;
    plan.first = std::min(plan.first, boundary.node_indices.front());
    plan.last = std::max(plan.last, boundary.node_indices.back());
    priority.push_back(boundary.node_indices.front());
    for (const NodeProto *node : plan.regions[i])
      owner.emplace(node, i);
  }
  if (plan.last - plan.first >= kMaxInterval)
    return false;
  const auto &nodes = index.graph.Builder().Nodes();
  std::vector<const NodeProto *> originals;
  for (std::size_t position = plan.first; position <= plan.last; ++position) {
    if (owner.count(&nodes[position]) == 0) {
      owner.emplace(&nodes[position], priority.size());
      priority.push_back(position);
      originals.push_back(&nodes[position]);
    }
  }
  std::vector<std::set<std::size_t>> successors(priority.size());
  std::vector<std::size_t> indegree(priority.size(), 0);
  for (std::size_t position = plan.first; position <= plan.last; ++position) {
    const NodeProto *node = &nodes[position];
    const std::size_t to = owner.at(node);
    for (const std::string &name : index.data.at(node).dependencies) {
      const NodeProto *producer = index.graph.NodeBefore(name);
      if (producer == nullptr)
        continue;
      const auto from = owner.find(producer);
      if (from == owner.end()) {
        if (index.graph.Position(*producer) >= plan.first)
          return false;
      } else if (from->second != to && successors[from->second].insert(to).second) {
        ++indegree[to];
      }
    }
  }
  // Shared weight conversions may precede calls without reading or folding weights.
  // All other operations keep their execution order, including retained calls.
  const NodeProto *previous = nullptr;
  for (std::size_t position = plan.first; position <= plan.last; ++position) {
    const NodeProto *node = &nodes[position];
    if (MovableConstantPreparation(index, *node))
      continue;
    if (previous != nullptr) {
      const std::size_t from = owner.at(previous), to = owner.at(node);
      if (from != to && successors[from].insert(to).second)
        ++indegree[to];
    }
    previous = node;
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
    if (unit < plan.regions.size()) {
      plan.schedule.emplace_back(true, unit);
    } else {
      plan.schedule.emplace_back(false,
                                 index.graph.Position(*originals[unit - plan.regions.size()]));
    }
    for (std::size_t next : successors[unit]) {
      if (--indegree[next] == 0)
        ready.emplace(priority[next], next);
    }
  }
  return plan.schedule.size() == priority.size();
}

bool MakePlan(Index &index, const Regions &regions, Plan &plan) {
  if (regions.size() < 2 || regions[0].size() < 2)
    return false;
  plan.regions = regions;
  for (const NodeProto *node : regions[0])
    plan.anchor_roles += index.data.at(node).anchor ? 1 : 0;
  uint64_t crossings = 0, unknown = 0, bytes = 0;
  std::vector<std::size_t> positions;
  for (const Region &region : regions) {
    Boundary boundary = GetBoundary(index, region);
    for (const std::string &name : boundary.inputs) {
      if (!KnownBoundaryType(index, name))
        return false;
    }
    for (const std::string &name : boundary.outputs) {
      if (!KnownBoundaryType(index, name))
        return false;
    }
    for (const std::string &name : boundary.edges) {
      ++crossings;
      uint64_t size = 0;
      if (ValueBytes(index.graph, name, size))
        bytes = SaturatingAdd(bytes, size);
      else
        ++unknown;
    }
    for (const NodeProto *node : region)
      positions.push_back(index.graph.Position(*node));
    plan.groups.push_back(std::move(static_cast<OperatorGroup &>(boundary)));
  }
  std::sort(positions.begin(), positions.end());
  plan.score = {crossings, unknown, bytes, kMaxRegion - regions[0].size(), std::move(positions)};
  return Schedule(index, plan);
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
      const NodeProto *before = index.graph.NodeBefore(dependencies[port]);
      if (before == nullptr || inside.count(before) != 0 || !index.data.at(before).usable)
        continue;
      for (int out = 0; out < before->output_size(); ++out) {
        if (before->output()[out].value() == dependencies[port])
          extensions.push_back({role, true, port, static_cast<std::size_t>(out), before});
      }
    }
    for (int out = 0; out < node.output_size() && extensions.size() < kMaxExtensions; ++out) {
      const std::string &name = node.output()[out].value();
      auto users = index.users.find(name);
      if (users == index.users.end())
        continue;
      for (const NodeProto *after : users->second) {
        if (users->second.size() > kMaxExtensions || extensions.size() >= kMaxExtensions)
          break;
        if (inside.count(after) != 0 || !index.data.at(after).usable)
          continue;
        const auto &inputs = index.data.at(after).dependencies;
        for (std::size_t port = 0; port < inputs.size() && extensions.size() < kMaxExtensions;
             ++port) {
          if (inputs[port] == name)
            extensions.push_back({role, false, static_cast<std::size_t>(out), port, after});
        }
      }
    }
  }
  std::sort(extensions.begin(), extensions.end(), [&](const Extension &a, const Extension &b) {
    return std::make_tuple(index.graph.Position(*a.node), a.role, a.predecessor, a.port,
                           a.other_port) < std::make_tuple(index.graph.Position(*b.node), b.role,
                                                           b.predecessor, b.port, b.other_port);
  });
  extensions.erase(std::unique(extensions.begin(), extensions.end(),
                               [](const Extension &a, const Extension &b) {
                                 return a.node == b.node && a.role == b.role &&
                                        a.predecessor == b.predecessor && a.port == b.port &&
                                        a.other_port == b.other_port;
                               }),
                   extensions.end());
  if (extensions.size() > kMaxExtensions)
    extensions.resize(kMaxExtensions);
  return extensions;
}

bool Extend(Index &index, const Regions &regions, const Extension &extension, Regions &grown) {
  grown = regions;
  grown[0].push_back(extension.node);
  for (std::size_t occurrence = 1; occurrence < regions.size(); ++occurrence) {
    const NodeProto *matched = nullptr;
    const NodeProto *base = regions[occurrence][extension.role];
    if (extension.predecessor) {
      const std::string &name = index.data.at(base).dependencies[extension.port];
      matched = index.graph.NodeBefore(name);
      if (matched == nullptr || extension.other_port >= matched->output().size() ||
          matched->output()[extension.other_port].value() != name ||
          !CompatibleNode(index, *extension.node, *matched))
        return false;
    } else {
      const std::string &name = base->output()[extension.port].value();
      const auto users = index.users.find(name);
      if (users == index.users.end() || users->second.size() > kMaxExtensions)
        return false;
      for (const NodeProto *other : users->second) {
        const auto &dependencies = index.data.at(other).dependencies;
        if (extension.other_port >= dependencies.size() ||
            dependencies[extension.other_port] != name ||
            !CompatibleNode(index, *extension.node, *other))
          continue;
        // Ambiguous fanout is left on the boundary rather than guessed.
        if (matched != nullptr && matched != other)
          return false;
        matched = other;
      }
    }
    if (matched == nullptr)
      return false;
    grown[occurrence].push_back(matched);
  }
  return Correspond(index, grown);
}

bool Grow(Index &index, Regions regions, std::size_t selected, Plan &best, std::size_t &attempts) {
  auto make_plan = [&](Plan &plan) {
    if (selected == regions.size())
      return MakePlan(index, regions, plan);
    return MakePlan(index, Regions(regions.begin(), regions.begin() + selected), plan);
  };
  bool found = make_plan(best);
  while (regions[0].size() < kMaxRegion) {
    bool extended = false;
    for (const Extension &extension : Extensions(index, regions[0])) {
      if (attempts >= kMaxGrowthAttempts)
        return found;
      ++attempts;
      Regions grown;
      if (!Extend(index, regions, extension, grown))
        continue;
      regions = std::move(grown);
      extended = true;
      Plan plan;
      // Adding another anchor establishes a larger seed skeleton. Boundary
      // cuts compete within that skeleton rather than against projections
      // omitting the rest of a jointly grown attention/MLP structure.
      if (make_plan(plan) &&
          (!found || plan.anchor_roles > best.anchor_roles ||
           (plan.anchor_roles == best.anchor_roles && plan.score < best.score))) {
        best = std::move(plan);
        found = true;
      }
      break;
    }
    if (!extended)
      break;
  }
  return found;
}

bool FindPlan(Index &index, const NodeProto &candidate, Plan &best, std::size_t &attempts,
              std::size_t &groups) {
  if (!index.valid)
    return false;
  const auto &entry = index.data.at(&candidate);
  auto bucket = index.anchors.find(entry.signature);
  if (!entry.usable || bucket == index.anchors.end() || bucket->second.size() < 2)
    return false;
  Region anchors;
  for (const NodeProto *node : bucket->second) {
    if (attempts >= kMaxGrowthAttempts)
      return false;
    ++attempts;
    if (CompatibleNode(index, candidate, *node)) {
      if (index.graph.Position(*node) < index.graph.Position(candidate))
        return false;
      anchors.push_back(node);
      if (anchors.size() == kMaxOccurrences * kMaxAnchorStride)
        break;
    }
  }
  bool found = false;
  auto consider = [&](Regions regions) {
    if (regions.size() < 2 || groups >= kMaxSeedGroups || attempts >= kMaxGrowthAttempts)
      return;
    const std::size_t family_occurrences = regions.size();
    std::size_t selected = regions.size();
    if (regions.size() > kMaxOccurrences) {
      const std::size_t count =
          regions.size() % kMaxOccurrences == 1 ? kMaxOccurrences - 1 : kMaxOccurrences;
      Region tail = std::move(regions.back());
      regions.resize(count);
      // The un-emitted tail probe prevents a capped family from appearing to
      // have a longer period merely because an incomplete last repeat was cut.
      regions.push_back(std::move(tail));
      selected = count;
    }
    ++groups;
    Plan plan;
    if (!Grow(index, std::move(regions), selected, plan, attempts))
      return;
    plan.family_occurrences = family_occurrences;
    const auto coverage = [](const Plan &value) {
      return std::make_tuple(value.family_occurrences * value.anchor_roles,
                             value.family_occurrences * value.regions[0].size(),
                             value.family_occurrences, value.regions.size(),
                             value.regions[0].size());
    };
    // Equal coverage favors more occurrences (a smaller period) before larger
    // individual regions. Within each skeleton Grow uses boundary-cost ordering.
    if (!found || coverage(plan) > coverage(best) ||
        (coverage(plan) == coverage(best) && plan.score < best.score)) {
      best = std::move(plan);
      found = true;
    }
  };
  Regions all;
  for (std::size_t i = 0; i < anchors.size(); ++i)
    all.push_back({anchors[i]});
  consider(std::move(all));
  // Refines identical operator anchors by a directed, port-labelled one-hop
  // seed. In particular Q/K/V or two projections in one layer need not have
  // identical neighborhoods, nor occur at a regular distance.
  std::size_t seed_count = 0;
  for (const Extension &extension : Extensions(index, Region{&candidate})) {
    if (seed_count++ == kMaxSeedGroups / 2 || groups >= kMaxSeedGroups ||
        attempts >= kMaxGrowthAttempts)
      break;
    Regions regions{{&candidate, extension.node}};
    std::unordered_set<const NodeProto *> occupied{&candidate, extension.node};
    for (std::size_t i = 1; i < anchors.size(); ++i) {
      if (attempts >= kMaxGrowthAttempts)
        break;
      ++attempts;
      Regions pair;
      if (!Extend(index, Regions{{&candidate}, {anchors[i]}}, extension, pair))
        continue;
      const NodeProto *neighbor = pair[1][1];
      if (occupied.count(anchors[i]) != 0 || occupied.count(neighbor) != 0)
        continue;
      occupied.insert(anchors[i]);
      occupied.insert(neighbor);
      regions.push_back(std::move(pair[1]));
    }
    consider(std::move(regions));
  }
  // A bounded periodic fallback also recognizes indistinguishable projections
  // whose one-hop neighborhoods are identical but whose larger roles differ.
  for (std::size_t stride = 2;
       stride <= kMaxAnchorStride && stride < anchors.size() && attempts < kMaxGrowthAttempts;
       ++stride) {
    Regions regions;
    for (std::size_t i = 0; i < anchors.size(); i += stride)
      regions.push_back({anchors[i]});
    consider(std::move(regions));
  }
  return found;
}

bool FindPlan(Index &index, Plan &best) {
  if (!index.valid)
    return false;
  Region candidates;
  for (const NodeProto &node : index.graph.Builder().Nodes()) {
    const auto &entry = index.data.at(&node);
    if (entry.usable && entry.anchor)
      candidates.push_back(&node);
  }
  // Rare anchors distinguish a complete periodic motif from its common projections.
  std::stable_sort(candidates.begin(), candidates.end(),
                   [&](const NodeProto *left, const NodeProto *right) {
                     return index.anchors.at(index.data.at(left).signature).size() <
                            index.anchors.at(index.data.at(right).signature).size();
                   });
  std::map<std::string, Region> representatives;
  std::size_t groups = 0, attempts = 0;
  for (const NodeProto *candidate : candidates) {
    const NodeProto &node = *candidate;
    const auto &entry = index.data.at(&node);
    if (!entry.usable || index.anchors.count(entry.signature) == 0)
      continue;
    Region &previous = representatives[entry.signature];
    bool represented = false;
    for (const NodeProto *other : previous) {
      if (attempts >= kMaxGrowthAttempts)
        return false;
      ++attempts;
      if (CompatibleNode(index, node, *other)) {
        represented = true;
        break;
      }
    }
    if (represented)
      continue;
    if (groups >= kMaxSeedGroups || attempts >= kMaxGrowthAttempts)
      return false;
    previous.push_back(&node);
    if (FindPlan(index, node, best, attempts, groups))
      return true;
  }
  return false;
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

void ClearLayerAnalysis(GraphProto &scope) {
  const auto clear_range = [](ValueInfoProto &value) {
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
  if (reject_ranges) {
    const auto has_range = [](const ValueInfoProto &value) {
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
  ClearLayerAnalysis(body);
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
    if (!HasComparableLayerTensorContent(initializer))
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
    // Exporter provenance does not change the computation.
    auto &metadata = node.ref_metadata_props();
    for (auto it = metadata.begin(); it != metadata.end();) {
      if (it->key() == "namespace" || it->key() == "pkg.onnxscript.class_hierarchy" ||
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
      if ((attribute.has_t() && !HasComparableLayerTensorContent(attribute.t())) ||
          !std::all_of(attribute.tensors().begin(), attribute.tensors().end(),
                       HasComparableLayerTensorContent))
        return false;
      const auto comparable_sparse = [](const SparseTensorProto &tensor) {
        return HasComparableLayerTensorContent(tensor.values()) &&
               HasComparableLayerTensorContent(tensor.indices());
      };
      if ((attribute.has_sparse_tensor() && !comparable_sparse(attribute.sparse_tensor())) ||
          !std::all_of(attribute.sparse_tensors().begin(), attribute.sparse_tensors().end(),
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

std::string FindLayerFunction(GraphBuilder &source, const GraphProto &body) {
  GraphProto normalized = utils::CopyWithSharedTensorStorage(body);
  std::size_t work = 0;
  if (!NormalizeLayerBody(normalized, source, {}, 0, work, false))
    return {};
  GraphBuilder &builder = source.ModelBuilder();
  for (const auto &local : builder.LocalFunctions()) {
    if (!local->IsLocalFunction(local->name(), "ai.rt", "") ||
        !IsLayerName("ai.rt", local->name()) || local->Nodes().size() != body.node().size() ||
        local->Inputs().size() != body.input().size() ||
        local->Outputs().size() != body.output().size())
      continue;
    const auto *definition = builder.Shapes().GetLocalFunction(
        core::shapes::ShapesContext::LocalFunctionKey("ai.rt", local->name(), ""));
    if (definition == nullptr || !definition->attribute().empty() ||
        !definition->attribute_proto().empty())
      continue;
    GraphProto candidate = local->BuildGraph();
    work = 0;
    if (NormalizeLayerBody(candidate, *local, {}, 0, work, true) &&
        CompatibleLayerOpsets(*local, source, candidate) && normalized.Equals(candidate))
      return local->name();
  }
  return {};
}

NodeProto MakeLayer(Index &index, const OperatorGroup &boundary,
                    const std::vector<OperatorGroup> &boundaries, std::size_t occurrence) {
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
  core::builder::GraphBuilder &source = index.graph.Builder();
  source.SetOpsetVersion("ai.rt", 1);
  core::builder::GraphBuilder &builder = source.ModelBuilder();
  const std::string name = "RepeatedLayer_n" + std::to_string(occurrence) + "_p" +
                           std::to_string(boundary.node_indices.front());
  if (const std::string existing = FindLayerFunction(source, body); !existing.empty())
    return MakeNode(existing.c_str(), boundary.inputs, boundary.outputs, "ai.rt", name.c_str());
  std::string function_name = "Layer";
  for (std::size_t suffix = 1;
       builder.HasLocalFunction(function_name) || builder.HasName(function_name); ++suffix)
    function_name = "Layer_" + std::to_string(suffix);
  FunctionProto definition;
  definition.set_name(function_name);
  definition.set_domain("ai.rt");
  for (const auto &[domain, version] : source.OpsetVersions()) {
    auto *opset = definition.add_opset_import();
    if (!domain.empty())
      opset->set_domain(domain);
    opset->set_version(version);
  }
  for (const ValueInfoProto &input : body.input()) {
    definition.add_input(input.name());
    *definition.add_value_info() = input;
  }
  for (const ValueInfoProto &output : body.output()) {
    definition.add_output(output.name());
    *definition.add_value_info() = output;
  }
  for (const ValueInfoProto &value_info : body.value_info())
    *definition.add_value_info() = value_info;
  definition.ref_node() = utils::CopyWithSharedTensorStorage(body.node());
  builder.MakeLocalFunction(definition);
  return MakeNode(function_name.c_str(), boundary.inputs, boundary.outputs, "ai.rt", name.c_str());
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

} // namespace

OperatorGroupProposal ProposeRepeatedLayers(GraphGraph &graph) {
  if (graph.Builder().Nodes().empty())
    return {};
  const auto status = Eligibility(graph);
  if (status != OperatorGroupStatus::kFound) {
    OperatorGroupProposal proposal;
    proposal.status = status;
    return proposal;
  }
  Index index(graph);
  Plan plan;
  if (!FindPlan(index, plan))
    return {};
  plan.status = OperatorGroupStatus::kFound;
  return std::move(static_cast<OperatorGroupProposal &>(plan));
}

utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) {
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("RepeatedLayer::Apply expects a nonempty match.");
  graph.Position(*nodes[0]);
  if (Eligibility(graph) != OperatorGroupStatus::kFound)
    throw core::builder::BuilderError("RepeatedLayer::Apply could not reproduce its regions.");
  Index index(graph);
  Plan plan;
  if (!FindPlan(index, plan))
    throw core::builder::BuilderError("RepeatedLayer::Apply could not reproduce its regions.");
  plan.status = OperatorGroupStatus::kFound;
  if (plan.MatchedNodes(graph.Builder().Nodes()) != nodes)
    throw core::builder::BuilderError("RepeatedLayer::Apply received an inconsistent match.");
  return MakeOperatorGroupReplacements(graph, plan, [&](std::size_t group) {
    return MakeLayer(index, plan.groups[group], plan.groups, group);
  });
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
