// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file function.h
 * @brief Declares helpers for constructing and expanding ONNX FunctionProto bodies.
 *
 * This header provides utility types and helper APIs to build function-body nodes in C++,
 * attach attributes and opsets, and inline graph fragments into function
 * definitions used by operator schemas.
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "onnx/common/status.h"
#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_LIGHT_NAMESPACE {
/**
 * Expands a function-call node into graph nodes according to a function proto.
 * @param node Function-call node to expand.
 * @param func Function definition used for expansion.
 * @param g Graph receiving the expanded nodes.
 * @param node_prefix Prefix prepended to generated value names.
 */
ONNX_API void FunctionExpandHelper(const NodeProto &node, const FunctionProto &func, GraphProto &g,
                                   const std::string &node_prefix = "");

/// Provides utility types and helpers to build function-body nodes.
class FunctionBodyHelper {
public:
  /// Wraps an attribute proto or typed attribute value.
  struct AttributeProtoWrapper {
    AttributeProto proto;

    AttributeProtoWrapper() = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    AttributeProtoWrapper(AttributeProto attr_prot) : proto(std::move(attr_prot)) {}

    template <typename T>
    AttributeProtoWrapper(const std::string &attr_name, const T &value)
        : proto(MakeAttribute(attr_name, value)) {}
  };

  /// Describes one node definition used by BuildNodes.
  struct NodeDef {
    NodeDef(std::vector<std::string> outputs, std::string op_type, std::vector<std::string> inputs,
            std::vector<AttributeProtoWrapper> attributes = {}, std::string domain = "")
        : outputs(std::move(outputs)), op_type(std::move(op_type)), inputs(std::move(inputs)),
          attributes(std::move(attributes)), domain(std::move(domain)) {}

    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<AttributeProtoWrapper> attributes;
    std::string domain;
  };

  /**
   * Builds NodeProto entries from lightweight node definitions.
   *
   * Example:
   * - ``{{"Z"}, "Add", {"X", "Y"}}`` models ``Z = Add(X, Y)``.
   * - ``{{"Y"}, "Concat", {"X1", "X2", "X3"}, {{"axis", 1}}}`` adds attributes (axis=1).
   * - ``{{"Z"}, "Foo", {"X", "Y"}, {}, "customdomain"}`` targets a custom-domain operator.
   *
   * @param node_defs Node definitions describing outputs, op type, inputs, and attributes.
   * @return Materialized nodes for a function body.
   */
  ONNX_API static std::vector<NodeProto> BuildNodes(const std::vector<NodeDef> &node_defs);

  /**
   * Appends nodes built from node_defs into functionProto.
   * @param functionProto Function proto receiving appended nodes.
   * @param node_defs Node definitions to materialize and append.
   */
  ONNX_API static void BuildNodes(FunctionProto &functionProto,
                                  const std::vector<NodeDef> &node_defs);

  /**
   * Builds a complete FunctionProto body and relied-opset list.
   * @param functionProto Function proto to populate.
   * @param schema Schema describing inputs, outputs, and attributes.
   * @param node_defs Node definitions used to create the function body.
   * @param relied_opsets Opsets that the generated function body relies on.
   * @return True if the function body is built successfully.
   */
  ONNX_API static bool BuildFunctionProto(FunctionProto &functionProto, const OpSchema &schema,
                                          const std::vector<NodeDef> &node_defs,
                                          const std::vector<OperatorSetIdProto> &relied_opsets);

  /**
   * Creates a scalar Constant node from a single value.
   * @tparam T Value type accepted by ToTensor.
   * @param name Output value name for the Constant node.
   * @param value Scalar value stored in the Constant tensor.
   * @return Node definition for a Constant node.
   */
  template <typename T> ONNX_API static NodeDef Const(const std::string &name, const T &value) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(value)}}};
  }

  /**
   * Creates a Constant node from a 1D vector literal.
   * @tparam T Element type accepted by ToTensor.
   * @param name Output value name for the Constant node.
   * @param values Literal values stored as a 1D Constant tensor.
   * @return Node definition for a Constant node.
   */
  template <typename T>
  ONNX_API static NodeDef Const(const std::string &name, const std::vector<T> &values) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(values)}}};
  }
};

/// Fluent builder for FunctionProto definitions used by operator schemas.
class FunctionBuilder {
public:
  explicit FunctionBuilder(FunctionProto &funProto_) : funProto(funProto_) {}

  ONNX_API FunctionBuilder &Add(const char *nodes_txt) {
    OnnxParser parser(nodes_txt);
    while (!parser.EndOfInput()) {
      auto status = parser.Parse(*funProto.add_node());
      if (!status.IsOK())
        ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    return *this;
  }

  ONNX_API FunctionBuilder &Add(const char *node_txt, const AttributeProto &attr) {
    OnnxParser parser(node_txt);
    auto *node = funProto.add_node();
    auto status = parser.Parse(*node);
    if (!status.IsOK()) {
      ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    if (!parser.EndOfInput()) {
      ONNX_THROW_EX(
          std::logic_error("Error unexpected extra input in node:" + status.ErrorMessage()));
    }

    *node->add_attribute() = attr;

    return *this;
  }

  template <typename T>
  ONNX_API FunctionBuilder &Add(const char *node_txt, const std::string &attr_name,
                                const T &attr_value) {
    return Add(node_txt, MakeAttribute(attr_name, attr_value));
  }

  template <typename T>
  ONNX_API FunctionBuilder &AddAttributeToNode(const std::string &attr_name, const T &attr_value) {
    auto &nodes = funProto.ref_node();
    int nodes_size = static_cast<int>(nodes.size());
    if (nodes_size != 0) {
      auto &node = nodes[nodes_size - 1];
      node.add_attribute() = MakeAttribute(attr_name, attr_value);
    } else {
      ONNX_THROW_EX(std::logic_error("Error adding attribute to node of a graph with no nodes"));
    }
    return *this;
  }

  template <typename T, typename... Args>
  ONNX_API FunctionBuilder &AddAttributes(const std::string &attr_name, const T &attr_value,
                                          Args... args) {
    AddAttributeToNode(attr_name, attr_value);
    if constexpr (sizeof...(args) > 0) {
      AddAttributes(args...);
    }
    return *this;
  }

  // Adds variable number of attributes to a node
  template <typename... Args>
  ONNX_API FunctionBuilder &Add(const char *node_txt, const Args &...args) {
    Add(node_txt);
    if constexpr (sizeof...(args) % 2 == 0) {
      AddAttributes(args...);
    }
    return *this;
  }

  ONNX_API FunctionBuilder &Const(const std::string &name, const TensorProto &tensor) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  // Creates a scalar constant (a tensor of rank zero).
  template <typename T> ONNX_API FunctionBuilder &Const(const std::string &name, T const_value) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add(constant_op.c_str(), MakeAttribute("value", ToTensor(const_value)));
  }

  // Creates a 1D tensor constant consisting of a single value.
  template <typename T> ONNX_API FunctionBuilder &Const1D(const std::string &name, T const_value) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    auto tensor = ToTensor(const_value);
    tensor.add_dims(1);
    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  // Creates a 1D tensor constant consisting of zero or more values.
  template <typename T>
  ONNX_API FunctionBuilder &Const(const std::string &name, const std::vector<T> &values) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    auto tensor = ToTensor(values);
    tensor.add_dims(values.size()); // Treat as 1D tensor.

    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  ONNX_API FunctionBuilder &AddOpset(const char *domain, int version) {
    auto *opset = funProto.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
    return *this;
  }

  /**
   * @brief Adds an inlined call to a graph as a sequence of nodes in the function.
   *
   * This method effectively inlines the logic from the given graph into the function
   * being constructed. It:
   * - Adds a Constant node for every initializer in the graph
   * - Adds a copy of every node in the graph
   * - Renames formal input parameters to match actual inputs
   * - Renames formal output parameters to match actual outputs
   * - Renames all other intermediate values with a unique prefix
   * - Leaves references to undefined names (outer scope variables) unchanged
   *
   * @param outputs List of output variable names for the inlined call
   * @param graph The graph to inline
   * @param inputs List of input variable names for the inlined call
   * @param prefix Prefix to add to intermediate variable names for uniqueness
   * @return Reference to this FunctionBuilder for method chaining
   */
  ONNX_API FunctionBuilder &AddInlinedCall(std::initializer_list<std::string_view> outputs,
                                           const GraphProto &graph,
                                           std::initializer_list<std::string_view> inputs,
                                           std::string_view prefix);

private:
  FunctionProto &funProto;
};

} // namespace ONNX_LIGHT_NAMESPACE
