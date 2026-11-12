// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"
#include "onnx_lib/version_converter/errors.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_lib/common/ir_pb_converter.h"

namespace ONNX_LIGHT_NAMESPACE::version_conversion {

ConvertError::~ConvertError() = default;

ModelProto ConvertVersion(const ModelProto &mp_in, int target_version) {
  // Get initial_opsetid from mp_in
  OpSetID initial_struct(0);
  for (const auto &it : mp_in.opset_import()) {
    if (it.domain().empty() || it.domain() == "ai.onnx") {
      initial_struct.setVersion(it.version());
      break;
    }
  }
  OpSetID target_struct = OpSetID(target_version);
  DefaultVersionConverter v;
  return v.convert_version(mp_in, initial_struct, target_struct);
}

void DefaultVersionConverter::convert_graph(const std::shared_ptr<Graph> &g,
                                            const OpSetID &initial_version,
                                            const OpSetID &target_version) const {
  assertNonNull(g);

  // TODO(ONNX): Move to Inter-Domain Converter
  // Get initial model versions
  // std::vector<OpSetID> initial_versions = g->opset_versions_mutable();

  // No conversion necessary if Model has single, equivalent opset version
  // if (initial_versions.size() == 1 && initial_versions[0].version ==
  //    target_version.version && initial_versions[0].domain ==
  //    target_version.domain) {
  //  return mp_in;
  // }

  // Check if versions are valid
  assertInVersionRange(initial_version.version());
  assertInVersionRange(target_version.version());

  // Iterate over all versions to target_version for specified
  int64_t curr_version = initial_version.version();
  int64_t step = 0;
  if (target_version.version() > initial_version.version()) {
    step = 1;
  } else {
    step = -1;
  }
  // Identify index of the default domain ("" or "ai.onnx") in g.opset_versions.
  int domain_index = -1;
  for (int i = 0; i < static_cast<int>(g->opset_versions_mutable().size()); i++) {
    const std::string &dom = g->opset_versions_mutable()[i].domain();
    if (dom.empty() || dom == "ai.onnx") {
      domain_index = i;
      break;
    }
  }
  ONNX_ASSERTM(domain_index >= 0, "Graph has no default-domain (\"\" or \"ai.onnx\") opset entry.");
  while (curr_version != target_version.version()) {
    debug("curr_version: " + ONNX_LIGHT_NAMESPACE::to_string(curr_version) +
          ", next_version: " + ONNX_LIGHT_NAMESPACE::to_string(curr_version + step));
    Node *cur_op = nullptr;
    graph_node_list_iterator it = g->begin();
    // Iterate through and call adapter returned by adapter_lookup for ops from
    // current_version opset. We have to manipulate the iterator explicitly because cur_op
    // might change when applying the adapter (e.g. for deprecated ops)
    while (it != g->end()) {
      cur_op = *it;
      debug(std::string("Finding schema for ") + std::string(cur_op->kind().toString()));
      const std::string op_name = cur_op->kind().toString();
      if (op_name == "ConstantFill") {
        if (DEBUG) {
          std::cerr << "Warning: skipping schema search for experimental op 'ConstantFill' and "
                       "keeping the op as is. "
                       "Please be advised the converted model may not be working properly if "
                       "target runtime does not support this "
                       "experimental op."
                    << '\n';
        }
      } else if (!cur_op->domain().empty() && cur_op->domain() != "ai.onnx") {
        if (DEBUG) {
          std::cerr << "Warning: opset domain '" << cur_op->domain() << "' is not supported."
                    << '\n';
        }
      } else if (op_name != "Undefined" && op_name != "Captured") {
        const auto schema_it = all_schemas.find(op_name);
        ONNX_ASSERTM(
            schema_it != all_schemas.end(),
            "Op '%s' has no registered schema; cannot convert it from version %lld to %lld.",
            op_name.c_str(), static_cast<long long>(curr_version),
            static_cast<long long>(target_version.version()));
        const auto &op_domain_map = schema_it->second;
        OpSetID curr_id(curr_version);
        OpSetID next_id(curr_version + step);
        if (searchOpDomainMap(op_domain_map, curr_version, step)) {
          // Op is specifically defined for this domain and version
          const auto &op_adapter = adapter_lookup(cur_op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.
          // Error thrown by adapter_lookup
          if (DEBUG) {
            std::cerr << "Applying adapter" << '\n';
          }
          // adapt should handle replacing node in graph
          // An adapter may destroy the original node and return its replacement.
          auto metadata_props = cur_op->metadataProps();
          cur_op = op_adapter.adapt(g, cur_op);
          const auto &adapted_metadata = cur_op->metadataProps();
          if (adapted_metadata.empty()) {
            cur_op->setMetadataProps(std::move(metadata_props));
          } else if (!metadata_props.empty() &&
                     !std::equal(
                         metadata_props.begin(), metadata_props.end(), adapted_metadata.begin(),
                         adapted_metadata.end(),
                         [](const auto &left, const auto &right) { return left.Equals(right); })) {
            // Adapter entries override source keys; other source entries are preserved.
            std::unordered_set<std::string> adapted_keys;
            adapted_keys.reserve(adapted_metadata.size());
            for (const auto &prop : adapted_metadata)
              adapted_keys.insert(prop.key().value());
            std::erase_if(metadata_props, [&](const auto &prop) {
              return adapted_keys.contains(prop.key().value());
            });
            metadata_props.insert(metadata_props.end(), adapted_metadata.begin(),
                                  adapted_metadata.end());
            cur_op->setMetadataProps(std::move(metadata_props));
          }
          it = graph_node_list_iterator(cur_op, kNextDirection);
        }
        // Recursively convert any subgraph attributes
        for (const auto &attr : cur_op->attributeNames()) {
          if (cur_op->kindOf(attr) == AttributeKind::g) {
            convert_graph(cur_op->g(attr), curr_id, next_id);
          }
        }
      }
      ++it;
    }
    // Update model version
    curr_version += step;
    g->opset_versions_mutable()[static_cast<size_t>(domain_index)].incrementVersion(step);
  }
}

ModelProto DefaultVersionConverter::convert_version(const ModelProto &mp_in,
                                                    const OpSetID &initial_version,
                                                    const OpSetID &target_version) const {
  const std::string &initial_domain = initial_version.domain();
  const std::string &target_domain = target_version.domain();
  ONNX_ASSERTM((initial_domain.empty() || initial_domain == "ai.onnx") &&
                   (target_domain.empty() || target_domain == "ai.onnx"),
               "Warning: default onnx version converter can only convert "
               " between default domain opset versions ('' or 'ai.onnx')\n")
  ONNX_ASSERTM(initial_domain == target_domain,
               "initial_version and target_version must have the same domains")

  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == initial_version.domain()) {
      ONNX_ASSERTM(initial_version.version() == it->version(),
                   "initial_version does not reflect current state of model")
    }
  }

  std::shared_ptr<Graph> g(ImportModelProto(mp_in));

  convert_graph(g, initial_version, target_version);

  // Export g as ModelProto
  debug("Finished conversion; returning model");
  ModelProto mp_out = PrepareOutput(mp_in);
  ExportModelProto(&mp_out, g);
  for (auto &function : mp_out.functions()) {
    const auto &imports = function.opset_import();
    const auto default_opset = std::find_if(imports.begin(), imports.end(), [](const auto &opset) {
      return opset.domain().empty() || opset.domain() == "ai.onnx";
    });
    if (default_opset == imports.end() || default_opset->version() == target_version.version())
      continue;

    ModelProto function_model;
    function_model.set_ir_version(mp_in.ir_version());
    function_model.opset_import().extend(imports);
    auto &graph = *function_model.mutable_graph();
    graph.set_name(function.name());
    for (const auto &name : function.input())
      graph.add_input()->set_name(name);
    for (const auto &name : function.output())
      graph.add_output()->set_name(name);
    graph.node().extend(function.node());
    graph.value_info().extend(function.value_info());
    std::shared_ptr<Graph> function_graph(ImportModelProto(function_model));
    convert_graph(function_graph, OpSetID(default_opset->version()), target_version);
    ModelProto converted_function;
    ExportModelProto(&converted_function, function_graph);

    auto convert_default_graph = [&](GraphProto *graph_proto) {
      ModelProto default_model;
      default_model.set_ir_version(mp_in.ir_version());
      default_model.opset_import().extend(imports);
      default_model.mutable_graph()->CopyFrom(*graph_proto);
      std::shared_ptr<Graph> default_graph(ImportModelProto(default_model));
      convert_graph(default_graph, OpSetID(default_opset->version()), target_version);
      ModelProto converted_default;
      ExportModelProto(&converted_default, default_graph);
      graph_proto->CopyFrom(converted_default.graph());
    };
    for (auto &attribute : *function.mutable_attribute_proto()) {
      if (attribute.has_g())
        convert_default_graph(attribute.mutable_g());
      for (auto &attribute_graph : *attribute.mutable_graphs())
        convert_default_graph(&attribute_graph);
    }

    std::unordered_map<std::string, ValueInfoProto> previous_value_info;
    for (const auto &value_info : function.value_info())
      previous_value_info.emplace(value_info.name(), value_info);
    function.clear_value_info();
    std::unordered_set<std::string> value_info_names;
    auto append_value_info = [&](const ValueInfoProto &value_info) {
      auto previous = previous_value_info.find(value_info.name());
      const auto &type = value_info.type();
      const bool has_type_info = value_info.has_type() && type.has_type() &&
                                 (!type.has_tensor_type() || type.tensor_type().has_elem_type() ||
                                  type.tensor_type().has_shape());
      if (!has_type_info &&
          (previous == previous_value_info.end() ||
           (!previous->second.has_doc_string() && previous->second.metadata_props().empty())))
        return;
      if (!value_info_names.insert(value_info.name()).second)
        return;
      auto *converted_value_info = function.add_value_info();
      converted_value_info->CopyFrom(value_info);
      if (previous != previous_value_info.end()) {
        if (previous->second.has_doc_string())
          converted_value_info->set_doc_string(previous->second.doc_string());
        converted_value_info->mutable_metadata_props()->extend(previous->second.metadata_props());
      }
    };
    for (const auto &value_info : converted_function.graph().input())
      append_value_info(value_info);
    for (const auto &value_info : converted_function.graph().output())
      append_value_info(value_info);
    for (const auto &value_info : converted_function.graph().value_info())
      append_value_info(value_info);

    function.node().clear();
    // Functions have no initializers; adapters' constants must be represented as nodes.
    for (const auto &initializer : converted_function.graph().initializer()) {
      auto *constant = function.add_node();
      constant->set_op_type("Constant");
      constant->add_output(initializer.name());
      auto *value = constant->add_attribute();
      value->set_name("value");
      value->set_type(AttributeProto::TENSOR);
      value->mutable_t()->CopyFrom(initializer);
    }
    function.node().extend(converted_function.graph().node());
    function.opset_import() = std::move(converted_function.opset_import());
  }
  return mp_out;
}

} // namespace ONNX_LIGHT_NAMESPACE::version_conversion
