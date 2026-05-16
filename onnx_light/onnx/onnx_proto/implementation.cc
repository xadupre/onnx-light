// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapted from onnx/cpp2py_export.cc infer_function_output_types.

#include "implementation.h"

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "stream.h"
#include "stream_class.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace shape_inference {

namespace {

// Inference context for a single node inside a function body.
// Resolves attributes that reference the function's formal parameters
// (identified by a non-empty ref_attr_name field).
class FunctionNodeInferenceCtx : public InferenceContext {
public:
  FunctionNodeInferenceCtx(
      const NodeProto &node, const std::unordered_map<std::string, const TypeProto *> &env,
      const std::unordered_map<std::string, const AttributeProto *> &formal_attrs,
      size_t num_outputs) {
    // Build the resolved attribute map for this node.
    for (size_t i = 0; i < node.ref_attribute().size(); ++i) {
      const AttributeProto &attr = node.ref_attribute()[i];
      const std::string attr_name = attr.ref_name().as_string();
      if (attr.has_ref_attr_name() && !attr.ref_attr_name().empty()) {
        // Attribute references a formal parameter – substitute the real value.
        auto fit = formal_attrs.find(attr.ref_attr_name().as_string());
        if (fit != formal_attrs.end()) {
          resolved_attrs_[attr_name] = fit->second;
        }
      } else {
        resolved_attrs_[attr_name] = &attr;
      }
    }
    // Map node inputs to TypeProto pointers from the environment.
    for (size_t i = 0; i < node.ref_input().size(); ++i) {
      const std::string name = node.ref_input()[i].as_string();
      if (name.empty()) {
        input_types_.push_back(nullptr);
      } else {
        auto it = env.find(name);
        input_types_.push_back(it != env.end() ? it->second : nullptr);
      }
    }
    output_types_.resize(num_outputs);
  }

  const AttributeProto *getAttribute(const std::string &name) const override {
    auto it = resolved_attrs_.find(name);
    return it != resolved_attrs_.end() ? it->second : nullptr;
  }

  size_t getNumInputs() const override { return input_types_.size(); }

  const TypeProto *getInputType(size_t idx) const override {
    return idx < input_types_.size() ? input_types_[idx] : nullptr;
  }

  const TensorProto *getInputData(size_t /*idx*/) const override { return nullptr; }

  size_t getNumOutputs() const override { return output_types_.size(); }

  TypeProto *getOutputType(size_t idx) override {
    return idx < output_types_.size() ? &output_types_[idx] : nullptr;
  }

  GraphInferencer *getGraphAttributeInferencer(const std::string &) override { return nullptr; }

  std::vector<TypeProto> output_types_;

private:
  std::unordered_map<std::string, const AttributeProto *> resolved_attrs_;
  std::vector<const TypeProto *> input_types_;
};

} // namespace

std::vector<TypeProto> InferFunctionOutputTypes(const FunctionProto &function,
                                                const std::vector<TypeProto> &input_types,
                                                const std::vector<AttributeProto> &attributes) {
  // Build formal attribute map: name → const AttributeProto*.
  std::unordered_map<std::string, const AttributeProto *> formal_attrs;
  for (const auto &attr : attributes) {
    formal_attrs[attr.ref_name().as_string()] = &attr;
  }

  // Build opset-version map from the function's opset_import list.
  std::unordered_map<std::string, int> opset_map;
  for (size_t i = 0; i < function.ref_opset_import().size(); ++i) {
    const auto &op = function.ref_opset_import()[i];
    opset_map[op.ref_domain().as_string()] = static_cast<int>(op.version());
  }

  // Populate the type environment from function inputs.
  std::unordered_map<std::string, TypeProto> env;
  for (size_t i = 0; i < function.ref_input().size() && i < input_types.size(); ++i) {
    const std::string name = function.ref_input()[i].as_string();
    if (!name.empty() && input_types[i].has_type()) {
      env[name].CopyFrom(input_types[i]);
    }
  }

  const auto *registry = OpSchemaRegistry::Instance();

  // Iterate the function body and run per-node type/shape inference.
  for (size_t ni = 0; ni < function.ref_node().size(); ++ni) {
    const NodeProto &node = function.ref_node()[ni];
    const std::string op_type = node.ref_op_type().as_string();
    const std::string domain = node.ref_domain().as_string();

    // Determine the opset version for this node's domain.
    int version = 1;
    auto vit = opset_map.find(domain);
    if (vit != opset_map.end()) {
      version = vit->second;
    }

    const OpSchema *schema = registry->GetSchema(op_type, version, domain);
    if (schema == nullptr || !schema->has_type_and_shape_inference_function()) {
      continue;
    }

    // Build a const-pointer view of the current environment.
    std::unordered_map<std::string, const TypeProto *> env_ptrs;
    for (const auto &kv : env) {
      env_ptrs[kv.first] = &kv.second;
    }

    const size_t num_outputs = node.ref_output().size();
    FunctionNodeInferenceCtx ctx(node, env_ptrs, formal_attrs, num_outputs);

    // Re-throw all exceptions from node inference – InferenceError reports a
    // deliberate type/shape mismatch; all other exceptions are propagated so
    // that bugs in inference functions surface immediately.
    schema->GetTypeAndShapeInferenceFunction()(ctx);

    // Propagate inferred output types back into the environment.
    for (size_t oi = 0; oi < node.ref_output().size(); ++oi) {
      const std::string out_name = node.ref_output()[oi].as_string();
      if (!out_name.empty() && oi < ctx.output_types_.size() && ctx.output_types_[oi].has_type()) {
        env[out_name].CopyFrom(ctx.output_types_[oi]);
      }
    }
  }

  // Collect the TypeProto for each function output.
  std::vector<TypeProto> result;
  result.reserve(function.ref_output().size());
  for (size_t i = 0; i < function.ref_output().size(); ++i) {
    const std::string out_name = function.ref_output()[i].as_string();
    TypeProto tp{};
    auto it = env.find(out_name);
    if (it != env.end()) {
      tp.CopyFrom(it->second);
    }
    result.push_back(std::move(tp));
  }
  return result;
}

std::vector<std::string>
InferFunctionOutputTypesFromBytes(const FunctionProto &function,
                                  const std::vector<std::string> &input_type_bytes,
                                  const std::vector<std::string> &attribute_bytes) {
  // Parse input TypeProtos from serialized bytes.
  std::vector<TypeProto> input_types;
  input_types.reserve(input_type_bytes.size());
  for (const std::string &b : input_type_bytes) {
    TypeProto tp{};
    utils::StringStream stream(reinterpret_cast<const uint8_t *>(b.data()),
                               static_cast<int64_t>(b.size()));
    ParseOptions opts;
    tp.ParseFromStream(stream, opts);
    input_types.push_back(std::move(tp));
  }

  // Parse formal AttributeProtos from serialized bytes.
  std::vector<AttributeProto> attributes;
  attributes.reserve(attribute_bytes.size());
  for (const std::string &b : attribute_bytes) {
    AttributeProto attr{};
    utils::StringStream stream(reinterpret_cast<const uint8_t *>(b.data()),
                               static_cast<int64_t>(b.size()));
    ParseOptions opts;
    attr.ParseFromStream(stream, opts);
    attributes.push_back(std::move(attr));
  }

  // Run inference.
  std::vector<TypeProto> output_types = InferFunctionOutputTypes(function, input_types, attributes);

  // Serialize output TypeProtos back to bytes.
  std::vector<std::string> result;
  result.reserve(output_types.size());
  for (TypeProto &tp : output_types) {
    std::string out;
    SerializeOptions sopts;
    tp.SerializeToString(out, sopts);
    result.push_back(std::move(out));
  }
  return result;
}

} // namespace shape_inference
} // namespace ONNX_LIGHT_NAMESPACE
