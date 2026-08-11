// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include <cstring>

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

// Shared empty successor list returned by NextNodes for unused values.
const std::vector<const NodeProto *> &EmptyNodeList() {
  static const std::vector<const NodeProto *> empty;
  return empty;
}

// Returns the attribute named ``name`` of ``node``, or nullptr when absent.
const AttributeProto *FindAttribute(const NodeProto &node, const std::string &name) {
  for (const AttributeProto &attribute : node.attribute()) {
    if (attribute.name().value() == name) {
      return &attribute;
    }
  }
  return nullptr;
}

// Reads the first element of ``tensor`` as a double into ``out`` (the tensor is
// assumed to hold at least one element). Returns ``false`` for element types
// whose scalar value cannot be represented as a double.
template <typename T> T ReadRaw(const TensorProto &tensor) {
  T value{};
  std::memcpy(&value, tensor.raw_data().data(), sizeof(T));
  return value;
}

bool ReadScalarAsDouble(const TensorProto &tensor, double &out) {
  const bool raw = tensor.is_raw_data() && !tensor.raw_data().empty();
  switch (tensor.data_type()) {
  case TensorProto::DataType::FLOAT:
    if (raw) {
      out = static_cast<double>(ReadRaw<float>(tensor));
    } else if (!tensor.float_data().empty()) {
      out = static_cast<double>(tensor.float_data()[0]);
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::DOUBLE:
    if (raw) {
      out = ReadRaw<double>(tensor);
    } else if (!tensor.double_data().empty()) {
      out = tensor.double_data()[0];
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::INT64:
    if (raw) {
      out = static_cast<double>(ReadRaw<int64_t>(tensor));
    } else if (!tensor.int64_data().empty()) {
      out = static_cast<double>(tensor.int64_data()[0]);
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::INT32:
    if (raw) {
      out = static_cast<double>(ReadRaw<int32_t>(tensor));
    } else if (!tensor.int32_data().empty()) {
      out = static_cast<double>(tensor.int32_data()[0]);
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::INT16:
    if (raw) {
      out = static_cast<double>(ReadRaw<int16_t>(tensor));
    } else if (!tensor.int32_data().empty()) {
      out = static_cast<double>(static_cast<int16_t>(tensor.int32_data()[0]));
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::INT8:
    if (raw) {
      out = static_cast<double>(ReadRaw<int8_t>(tensor));
    } else if (!tensor.int32_data().empty()) {
      out = static_cast<double>(static_cast<int8_t>(tensor.int32_data()[0]));
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::BOOL:
    if (raw) {
      out = static_cast<double>(ReadRaw<uint8_t>(tensor));
    } else if (!tensor.int32_data().empty()) {
      out = static_cast<double>(static_cast<uint8_t>(tensor.int32_data()[0]));
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::UINT16:
    if (raw) {
      out = static_cast<double>(ReadRaw<uint16_t>(tensor));
    } else if (!tensor.int32_data().empty()) {
      out = static_cast<double>(static_cast<uint16_t>(tensor.int32_data()[0]));
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::UINT32:
    if (raw) {
      out = static_cast<double>(ReadRaw<uint32_t>(tensor));
    } else if (!tensor.uint64_data().empty()) {
      out = static_cast<double>(static_cast<uint32_t>(tensor.uint64_data()[0]));
    } else {
      return false;
    }
    return true;
  case TensorProto::DataType::UINT64:
    if (raw) {
      out = static_cast<double>(ReadRaw<uint64_t>(tensor));
    } else if (!tensor.uint64_data().empty()) {
      out = static_cast<double>(tensor.uint64_data()[0]);
    } else {
      return false;
    }
    return true;
  default:
    return false;
  }
}

} // namespace

GraphGraph::GraphGraph(const GraphBuilder &builder) : builder_(builder) {
  for (const TensorProto &initializer : builder_.Initializers()) {
    initializers_.emplace(initializer.name().value(), &initializer);
  }
  for (const ValueInfoProto &output : builder_.Outputs()) {
    output_names_.insert(output.name().value());
  }

  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &node = nodes[i];
    positions_.emplace(&node, i);
    for (int o = 0; o < node.output().size(); ++o) {
      std::string name(node.output(static_cast<std::size_t>(o)));
      if (!name.empty()) {
        predecessors_[std::move(name)] = &node;
      }
    }
    for (int in = 0; in < node.input().size(); ++in) {
      std::string name(node.input(static_cast<std::size_t>(in)));
      if (name.empty()) {
        continue;
      }
      std::vector<const NodeProto *> &consumers = successors_[name];
      // Avoid recording the same consumer twice when a node reads the same
      // value from more than one input slot. Consumer lists are short, so a
      // linear membership scan is cheap.
      bool present = false;
      for (const NodeProto *consumer : consumers) {
        if (consumer == &node) {
          present = true;
          break;
        }
      }
      if (!present) {
        consumers.push_back(&node);
      }
    }
    // Mark every value a referenced subgraph reads from the enclosing scope so
    // a rewrite never deletes a producer the subgraph still relies on.
    for (const GraphBuilder *subgraph : builder_.ReferencedSubgraphs(node)) {
      std::unordered_set<std::string> captured;
      subgraph->CollectImplicitInputs(captured);
      for (const std::string &name : captured) {
        subgraph_captured_.insert(name);
      }
    }
  }
}

const NodeProto *GraphGraph::NodeBefore(const std::string &name) const {
  auto it = predecessors_.find(name);
  return it == predecessors_.end() ? nullptr : it->second;
}

const std::vector<const NodeProto *> &GraphGraph::NextNodes(const std::string &name) const {
  auto it = successors_.find(name);
  return it == successors_.end() ? EmptyNodeList() : it->second;
}

bool GraphGraph::IsOutput(const std::string &name) const { return output_names_.count(name) != 0; }

bool GraphGraph::IsUsedBySubgraph(const std::string &name) const {
  return subgraph_captured_.count(name) != 0;
}

bool GraphGraph::IsUsed(const std::string &name) const {
  return IsUsedBySubgraph(name) || successors_.count(name) != 0 || IsOutput(name);
}

bool GraphGraph::IsUsedMoreThanOnce(const std::string &name) const {
  if (IsUsedBySubgraph(name) || IsOutput(name)) {
    return true;
  }
  auto it = successors_.find(name);
  return it != successors_.end() && it->second.size() > 1;
}

std::size_t GraphGraph::Position(const NodeProto &node) const {
  auto it = positions_.find(&node);
  if (it == positions_.end()) {
    throw BuilderError("GraphGraph::Position: node is not part of the index.");
  }
  return it->second;
}

bool GraphGraph::HasShape(const std::string &name) const { return builder_.HasShape(name); }

const SymTensor &GraphGraph::GetShape(const std::string &name) const {
  return builder_.GetShape(name);
}

bool GraphGraph::HasType(const std::string &name) const {
  return builder_.HasShape(name) && builder_.GetShape(name).Dtype() != TensorType::kUndefined;
}

TensorType GraphGraph::GetType(const std::string &name) const {
  return builder_.GetShape(name).Dtype();
}

bool GraphGraph::IsConstant(const std::string &name) const {
  return builder_.Compute().IsConstantValue(name);
}

const TensorProto *GraphGraph::GetComputedConstant(const std::string &name) const {
  auto cached = computed_constants_.find(name);
  if (cached != computed_constants_.end()) {
    return &cached->second;
  }
  auto init = initializers_.find(name);
  if (init != initializers_.end()) {
    return init->second;
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value = FindAttribute(*node, "value");
    if (value != nullptr && value->has_t()) {
      return &value->t();
    }
  }
  return nullptr;
}

void GraphGraph::SetComputedConstant(const std::string &name, TensorProto value) {
  computed_constants_[name] = std::move(value);
}

bool GraphGraph::ConstantShape(const std::string &name, std::vector<int64_t> &dims) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    dims.clear();
    for (int i = 0; i < tensor->dims().size(); ++i) {
      dims.push_back(tensor->dims()[static_cast<std::size_t>(i)]);
    }
    return true;
  }
  // A ``Constant`` node using ``value_int`` / ``value_float`` is a scalar.
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    if (FindAttribute(*node, "value_int") != nullptr ||
        FindAttribute(*node, "value_float") != nullptr) {
      dims.clear();
      return true;
    }
  }
  return false;
}

bool GraphGraph::ConstantScalarValue(const std::string &name, double &out) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    return ReadScalarAsDouble(*tensor, out);
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value_int = FindAttribute(*node, "value_int");
    if (value_int != nullptr && value_int->has_i()) {
      out = static_cast<double>(value_int->i());
      return true;
    }
    const AttributeProto *value_float = FindAttribute(*node, "value_float");
    if (value_float != nullptr && value_float->has_f()) {
      out = static_cast<double>(value_float->f());
      return true;
    }
  }
  return false;
}

namespace {

// Returns ``true`` when the concrete shape ``dims`` is a scalar shape according
// to the broadcast rule: without broadcasting only ``()`` and ``(1,)`` qualify;
// with broadcasting every dimension must be ``1``.
bool IsScalarShape(const std::vector<int64_t> &dims, bool broadcast) {
  if (broadcast) {
    for (int64_t d : dims) {
      if (d != 1) {
        return false;
      }
    }
    return true;
  }
  return dims.empty() || (dims.size() == 1 && dims[0] == 1);
}

} // namespace

bool GraphGraph::IsConstantScalar(const std::string &name, bool broadcast) const {
  if (!IsConstant(name)) {
    return false;
  }
  std::vector<int64_t> dims;
  if (!ConstantShape(name, dims)) {
    return false;
  }
  return IsScalarShape(dims, broadcast);
}

bool GraphGraph::IsConstantScalar(const std::string &name, double value, bool broadcast) const {
  if (!IsConstantScalar(name, broadcast)) {
    return false;
  }
  double scalar = 0.0;
  if (!ConstantScalarValue(name, scalar)) {
    return false;
  }
  return scalar == value;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
