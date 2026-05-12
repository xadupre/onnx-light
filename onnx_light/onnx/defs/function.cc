// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/function.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/defs/schema.h"

static std::string InternalTensorNameGenerator(const std::string &node_name,
                                               const std::string &internal_name) {
  std::string new_name = "Func_" + node_name + internal_name;
  return new_name;
}

namespace ONNX_LIGHT_NAMESPACE {
void FunctionExpandHelper(const NodeProto &node, const FunctionProto &func, GraphProto &g,
                          const std::string &node_prefix) {
  // Create a temporary unique node prefix for tensor names
  std::string uniq_prefix = node_prefix;
  if (uniq_prefix.empty()) {
    const void *address = static_cast<const void *>(&node);
    std::stringstream ss;
    ss << address;
    uniq_prefix = ss.str();
  }
  std::string node_name = node.has_name() ? node.ref_name().as_string() : func.ref_name().as_string() + uniq_prefix;
  std::unordered_map<std::string, std::string> io_names_map;
  std::unordered_map<std::string, AttributeProto> attr_map;

  for (int idx = 0; idx < (int)node.ref_input().size(); ++idx) {
    if (idx >= (int)func.ref_input().size()) {
      ONNX_THROW("Input for function node " + node_name + " is out of bounds");
    }
    io_names_map[func.ref_input()[idx].as_string()] = node.ref_input()[idx].as_string();
  }
  for (int idx = 0; idx < (int)node.ref_output().size(); ++idx) {
    if (idx >= (int)func.ref_output().size()) {
      ONNX_THROW("Output for function node " + node_name + " is out of bounds");
    }
    // If the node output is missing, the corresponding function output should
    // be treated as an internal value (not as missing) because it could also be
    // an intermediate value.
    if (node.ref_output()[idx].empty()) {
      continue;
    }
    io_names_map[func.ref_output()[idx].as_string()] = node.ref_output()[idx].as_string();
  }

  for (const auto &attr : node.ref_attribute()) {
    attr_map[attr.ref_name().as_string()] = attr;
  }

  // For undefined attributes of the function node
  // add default values obtained from the function schema.
  // get the domain version for function schema
  int domain_version = -1;
  for (const auto &opset_import : func.ref_opset_import()) {
    if (opset_import.ref_domain().as_string() == node.ref_domain().as_string()) {
      domain_version = static_cast<int>(opset_import.ref_version());
    }
  }
  if (domain_version == -1) {
    ONNX_THROW("No opset import registered for domain '" + node.ref_domain().as_string() +
               "' in function proto");
  }

  const OpSchemaRegistry *schema_registry = OpSchemaRegistry::Instance();
  const auto *const schema =
      schema_registry->GetSchema(node.ref_op_type().as_string(), domain_version,
                                 node.ref_domain().as_string());
  auto default_attrs = schema->attributes();

  for (const auto &[attr_name, attr] : default_attrs) {
    if (!attr_map.count(attr_name)) {
      attr_map[attr_name] = attr.default_value;
    }
  }

  for (const auto &function_node : func.ref_node()) {
    NodeProto &new_node = g.add_node();
    new_node.CopyFrom(function_node);
    new_node.clr_input();
    new_node.clr_output();
    new_node.clr_attribute();
    for (const auto &input : function_node.ref_input()) {
      if (io_names_map.count(input.as_string())) {
        new_node.add_input() = io_names_map[input.as_string()];
      } else {
        new_node.add_input() = InternalTensorNameGenerator(node_name, input.as_string());
      }
    }
    for (const auto &output : function_node.ref_output()) {
      if (io_names_map.count(output.as_string())) {
        new_node.add_output() = io_names_map[output.as_string()];
      } else {
        new_node.add_output() = InternalTensorNameGenerator(node_name, output.as_string());
      }
    }
    for (const auto &attr : function_node.ref_attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_ref_attr_name().as_string())) {
          AttributeProto &new_attr = new_node.add_attribute();
          new_attr.CopyFrom(attr_map[attr.ref_ref_attr_name().as_string()]);
          new_attr.set_name(attr.ref_name().as_string());
        }
      } else {
        AttributeProto &new_attr = new_node.add_attribute();
        new_attr.CopyFrom(attr);
      }
    }
  }
}

std::vector<NodeProto> FunctionBodyHelper::BuildNodes(const std::vector<NodeDef> &node_defs) {
  std::vector<NodeProto> nodes(node_defs.size());

  for (size_t i = 0; i < node_defs.size(); i++) {
    const NodeDef &node = node_defs[i];
    NodeProto &n = nodes[i];

    n.set_op_type(node.op_type);
    n.set_domain(node.domain);
    for (const auto &i : node.inputs) {
      n.add_input() = i;
    }
    for (const auto &o : node.outputs) {
      n.add_output() = o;
    }
    for (const auto &attr : node.attributes) {
      n.add_attribute() = attr.proto;
    }
  }

  return nodes;
}

void FunctionBodyHelper::BuildNodes(FunctionProto &functionProto,
                                    const std::vector<NodeDef> &node_defs) {
  for (const auto &node : node_defs) {
    auto &np = functionProto.add_node();

    np.set_op_type(node.op_type);
    np.set_domain(node.domain);
    for (const auto &inp : node.inputs) {
      np.add_input() = inp;
    }
    for (const auto &o : node.outputs) {
      np.add_output() = o;
    }
    for (const auto &attr : node.attributes) {
      np.add_attribute() = attr.proto;
    }
  }
}

bool FunctionBodyHelper::BuildFunctionProto(FunctionProto &functionProto, const OpSchema &schema,
                                            const std::vector<NodeDef> &node_defs,
                                            const std::vector<OperatorSetIdProto> &relied_opsets) {
  BuildNodes(functionProto, node_defs);

  for (const auto &relied_opset : relied_opsets) {
    functionProto.add_opset_import() = relied_opset;
  }

  functionProto.set_name(schema.Name());
  functionProto.set_domain(schema.domain());
  for (const auto &input : schema.inputs()) {
    functionProto.add_input() = input.GetName();
  }
  for (const auto &output : schema.outputs()) {
    functionProto.add_output() = output.GetName();
  }
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE
