#include "../onnx_proto/_onnxpy.h"
#include "implementation.h"
#include "onnx.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/node_inference_context.h"
#include "onnx/version_converter/convert.h"
#include "onnx/version_converter/errors.h"
#include <algorithm>
#include <limits>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <unordered_map>
#include <unordered_set>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using ONNX_LIGHT_NAMESPACE::shape_inference::NodeInferenceContextImpl;

void AddOnnxPySubmodules(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `version_converter`
  // -----------------------------------------------------------------------
  auto version_converter = m.def_submodule("version_converter");
  version_converter.doc() = "Version converter submodule";

  nb::exception<version_conversion::ConvertError>(
      version_converter,
      "ConvertError"); // NOLINT(bugprone-unused-raii,bugprone-throw-keyword-missing)

  version_converter.def(
      "convert_version",
      [](const ModelProto &mp_in, int target_version) {
        return version_conversion::ConvertVersion(mp_in, target_version);
      },
      nb::arg("model"), nb::arg("target_version"),
      "Convert a model to the specified target opset version.");

  // -----------------------------------------------------------------------
  // Submodule `parser`
  // -----------------------------------------------------------------------
  auto parser_mod = m.def_submodule("parser");
  parser_mod.doc() = "Parser submodule";

  // Returns (ok: bool, error_message: str, proto_object)
  parser_mod.def("parse_model", [](const std::string &text) -> nb::tuple {
    ModelProto proto{};
    OnnxParser parser(text.c_str());
    auto status = parser.Parse(proto);
    const std::string &error_msg = status.ErrorMessage();
    return nb::make_tuple(status.IsOK(), error_msg, std::move(proto));
  });

  parser_mod.def("parse_graph", [](const std::string &text) -> nb::tuple {
    GraphProto proto{};
    OnnxParser parser(text.c_str());
    auto status = parser.Parse(proto);
    const std::string &error_msg = status.ErrorMessage();
    return nb::make_tuple(status.IsOK(), error_msg, std::move(proto));
  });

  parser_mod.def("parse_function", [](const std::string &text) -> nb::tuple {
    FunctionProto proto{};
    OnnxParser parser(text.c_str());
    auto status = parser.Parse(proto);
    const std::string &error_msg = status.ErrorMessage();
    return nb::make_tuple(status.IsOK(), error_msg, std::move(proto));
  });

  parser_mod.def("parse_node", [](const std::string &text) -> nb::tuple {
    NodeProto proto{};
    OnnxParser parser(text.c_str());
    auto status = parser.Parse(proto);
    const std::string &error_msg = status.ErrorMessage();
    return nb::make_tuple(status.IsOK(), error_msg, std::move(proto));
  });

  // -----------------------------------------------------------------------
  // Submodule `shape_inference`
  // -----------------------------------------------------------------------
  auto shape_inference_mod = m.def_submodule("shape_inference");
  shape_inference_mod.doc() = "Shape inference submodule";

  nb::exception<InferenceError>(
      shape_inference_mod,
      "InferenceError"); // NOLINT(bugprone-unused-raii,bugprone-throw-keyword-missing)

  // Adapted from onnx/cpp2py_export.cc infer_function_output_types binding.
  // Takes a FunctionProto (by reference), a list of serialized input TypeProtos,
  // and a list of serialized formal AttributeProtos. Returns a list of
  // serialized output TypeProtos, one per function output.
  shape_inference_mod.def(
      "infer_function_output_types",
      [](const FunctionProto &function, const std::vector<nb::bytes> &input_types_bytes,
         const std::vector<nb::bytes> &attributes_bytes) -> std::vector<nb::bytes> {
        // Convert nb::bytes to std::string for the C++ implementation.
        std::vector<std::string> input_strs;
        input_strs.reserve(input_types_bytes.size());
        for (const nb::bytes &b : input_types_bytes) {
          input_strs.emplace_back(static_cast<const char *>(b.data()), b.size());
        }
        std::vector<std::string> attr_strs;
        attr_strs.reserve(attributes_bytes.size());
        for (const nb::bytes &b : attributes_bytes) {
          attr_strs.emplace_back(static_cast<const char *>(b.data()), b.size());
        }

        // Delegate to the C++ implementation.
        std::vector<std::string> output_strs =
            ONNX_LIGHT_NAMESPACE::shape_inference::InferFunctionOutputTypesFromBytes(
                function, input_strs, attr_strs);

        // Convert std::string results back to nb::bytes.
        std::vector<nb::bytes> result;
        result.reserve(output_strs.size());
        for (const std::string &s : output_strs) {
          result.emplace_back(s.c_str(), s.size());
        }
        return result;
      },
      nb::arg("function"), nb::arg("input_types"), nb::arg("attributes"),
      "Infers output types of a FunctionProto given serialized input TypeProtos and "
      "AttributeProtos. Adapted from onnx/cpp2py_export.cc infer_function_output_types.");

  // -----------------------------------------------------------------------
  // Submodule `defs`
  // -----------------------------------------------------------------------
  auto defs = m.def_submodule("defs");
  defs.doc() = "Schema submodule";

  nb::exception<SchemaError>(
      defs, "SchemaError"); // NOLINT(bugprone-unused-raii,bugprone-throw-keyword-missing)

  nb::class_<OpSchema> op_schema(defs, "OpSchema", "Schema of an operator.");

  nb::enum_<OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption",
                                             nb::is_arithmetic())
      .value("Single", OpSchema::Single)
      .value("Optional", OpSchema::Optional)
      .value("Variadic", OpSchema::Variadic);

  nb::enum_<OpSchema::DifferentiationCategory>(op_schema, "DifferentiationCategory",
                                               nb::is_arithmetic())
      .value("Unknown", OpSchema::Unknown)
      .value("Differentiable", OpSchema::Differentiable)
      .value("NonDifferentiable", OpSchema::NonDifferentiable);

  nb::enum_<OpSchema::NodeDeterminism>(op_schema, "NodeDeterminism")
      .value("Deterministic", OpSchema::NodeDeterminism::Deterministic)
      .value("NonDeterministic", OpSchema::NodeDeterminism::NonDeterministic)
      .value("Unknown", OpSchema::NodeDeterminism::Unknown);

  nb::enum_<OpSchema::SupportType>(op_schema, "SupportType", nb::is_arithmetic())
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  nb::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def(
          "__init__",
          [](OpSchema::Attribute *self, std::string name, AttributeProto::AttributeType type,
             std::string description, bool required) {
            new (self) OpSchema::Attribute(std::move(name), std::move(description), type, required);
          },
          nb::arg("name"), nb::arg("type"), nb::arg("description") = "", nb::kw_only(),
          nb::arg("required") = true)
      .def(
          "__init__",
          [](OpSchema::Attribute *self, std::string name, const AttributeProto &default_value,
             std::string description) {
            new (self) OpSchema::Attribute(std::move(name), std::move(description),
                                           AttributeProto(default_value));
          },
          nb::arg("name"), nb::arg("default_value"), nb::arg("description") = "")
      .def_ro("name", &OpSchema::Attribute::name)
      .def_ro("description", &OpSchema::Attribute::description)
      .def_ro("type", &OpSchema::Attribute::type)
      .def_prop_ro(
          "_default_value",
          [](const OpSchema::Attribute *attr) -> AttributeProto { return attr->default_value; })
      .def_ro("required", &OpSchema::Attribute::required);

  nb::class_<OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def(nb::init<std::string, std::vector<std::string>, std::string>(),
           nb::arg("type_param_str"), nb::arg("allowed_type_strs"), nb::arg("description") = "")
      .def_ro("type_param_str", &OpSchema::TypeConstraintParam::type_param_str)
      .def_ro("allowed_type_strs", &OpSchema::TypeConstraintParam::allowed_type_strs)
      .def_ro("description", &OpSchema::TypeConstraintParam::description);

  nb::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def(
          "__init__",
          [](OpSchema::FormalParameter *self, std::string name, std::string type_str,
             const std::string &description, OpSchema::FormalParameterOption param_option,
             bool is_homogeneous, int min_arity,
             OpSchema::DifferentiationCategory differentiation_category) {
            new (self) OpSchema::FormalParameter(std::move(name), description, std::move(type_str),
                                                 param_option, is_homogeneous, min_arity,
                                                 differentiation_category);
          },
          nb::arg("name"), nb::arg("type_str"), nb::arg("description") = "", nb::kw_only(),
          nb::arg("param_option") = OpSchema::Single, nb::arg("is_homogeneous") = true,
          nb::arg("min_arity") = 1,
          nb::arg("differentiation_category") = OpSchema::DifferentiationCategory::Unknown)
      .def_prop_ro("name", &OpSchema::FormalParameter::GetName)
      .def_prop_ro("types", &OpSchema::FormalParameter::GetTypes)
      .def_prop_ro("type_str", &OpSchema::FormalParameter::GetTypeStr)
      .def_prop_ro("description", &OpSchema::FormalParameter::GetDescription)
      .def_prop_ro("option", &OpSchema::FormalParameter::GetOption)
      .def_prop_ro("is_homogeneous", &OpSchema::FormalParameter::GetIsHomogeneous)
      .def_prop_ro("min_arity", &OpSchema::FormalParameter::GetMinArity)
      .def_prop_ro("differentiation_category",
                   &OpSchema::FormalParameter::GetDifferentiationCategory);

  op_schema
      .def(
          "__init__",
          [](OpSchema *self, std::string name, std::string domain, int since_version,
             const std::string &doc, std::vector<OpSchema::FormalParameter> inputs,
             std::vector<OpSchema::FormalParameter> outputs,
             std::vector<std::tuple<std::string, std::vector<std::string>, std::string>>
                 type_constraints,
             std::vector<OpSchema::Attribute> attributes,
             OpSchema::NodeDeterminism node_determinism) {
            new (self) OpSchema();
            self->SetName(std::move(name))
                .SetDomain(std::move(domain))
                .SinceVersion(since_version)
                .SetDoc(doc);
            self->SetNodeDeterminism(node_determinism);
            for (size_t i = 0; i < inputs.size(); ++i) {
              self->Input(static_cast<int>(i), std::move(inputs[i]));
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
              self->Output(static_cast<int>(i), std::move(outputs[i]));
            }
            for (auto &tc : type_constraints) {
              std::string type_str;
              std::vector<std::string> constraints;
              std::string description;
              tie(type_str, constraints, description) = std::move(tc);
              self->TypeConstraint(std::move(type_str), std::move(constraints),
                                   std::move(description));
            }
            for (auto &attribute : attributes) {
              self->Attr(std::move(attribute));
            }
            self->Finalize();
          },
          nb::arg("name"), nb::arg("domain"), nb::arg("since_version"), nb::arg("doc") = "",
          nb::kw_only(), nb::arg("inputs") = std::vector<OpSchema::FormalParameter>{},
          nb::arg("outputs") = std::vector<OpSchema::FormalParameter>{},
          nb::arg("type_constraints") =
              std::vector<std::tuple<std::string, std::vector<std::string>, std::string>>{},
          nb::arg("attributes") = std::vector<OpSchema::Attribute>{},
          nb::arg("node_determinism") = OpSchema::NodeDeterminism::Unknown)
      .def_prop_rw("name", &OpSchema::Name,
                   [](OpSchema &self, const std::string &name) { self.SetName(name); })
      .def_prop_rw("domain", &OpSchema::domain,
                   [](OpSchema &self, const std::string &domain) { self.SetDomain(domain); })
      .def_prop_rw(
          "doc",
          [](const OpSchema &self) -> std::string {
            const char *d = self.doc();
            return d ? d : "";
          },
          [](OpSchema &self, const std::string &doc) { self.SetDoc(doc); })
      .def_prop_ro("file", &OpSchema::file)
      .def_prop_ro("line", &OpSchema::line)
      .def_prop_ro("support_level", &OpSchema::support_level)
      .def_prop_ro("since_version", &OpSchema::since_version)
      .def_prop_ro("deprecated", &OpSchema::deprecated)
      .def_prop_ro("function_opset_versions", &OpSchema::function_opset_versions)
      .def_prop_ro("context_dependent_function_opset_versions",
                   &OpSchema::context_dependent_function_opset_versions)
      .def_prop_ro("all_function_opset_versions",
                   [](const OpSchema *op) -> std::vector<int> {
                     auto all = op->function_opset_versions();
                     auto ctx = op->context_dependent_function_opset_versions();
                     all.insert(all.end(), ctx.begin(), ctx.end());
                     std::sort(all.begin(), all.end());
                     all.erase(std::unique(all.begin(), all.end()), all.end());
                     return all;
                   })
      .def_prop_ro("min_input", &OpSchema::min_input)
      .def_prop_ro("max_input", &OpSchema::max_input)
      .def_prop_ro("min_output", &OpSchema::min_output)
      .def_prop_ro("max_output", &OpSchema::max_output)
      .def_prop_ro("attributes", &OpSchema::attributes)
      .def_prop_ro("inputs", &OpSchema::inputs)
      .def_prop_ro("outputs", &OpSchema::outputs)
      .def_prop_ro("has_type_and_shape_inference_function",
                   &OpSchema::has_type_and_shape_inference_function)
      .def_prop_ro("has_data_propagation_function", &OpSchema::has_data_propagation_function)
      .def_prop_ro("type_constraints", &OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) { return v == std::numeric_limits<int>::max(); })
      .def_prop_ro("has_function", &OpSchema::HasFunction)
      .def_prop_ro("_function_body",
                   [](const OpSchema *op) -> nb::object {
                     const FunctionProto *fp = op->GetFunction();
                     if (!fp)
                       return nb::none();
                     FunctionProto copy;
                     copy.CopyFrom(*fp);
                     return nb::cast(std::move(copy));
                   })
      .def("get_function_with_opset_version",
           [](const OpSchema *op, int opset_version) -> nb::object {
             const FunctionProto *fp = op->GetFunction(opset_version);
             if (!fp)
               return nb::none();
             FunctionProto copy;
             copy.CopyFrom(*fp);
             return nb::cast(std::move(copy));
           })
      .def_prop_ro("has_context_dependent_function", &OpSchema::HasContextDependentFunction)
      .def("get_context_dependent_function",
           [](const OpSchema *op, const NodeProto &node,
              const std::vector<TypeProto> &input_types) -> nb::object {
             if (!op->HasContextDependentFunction())
               return nb::none();
             FunctionBodyBuildContextImpl ctx(node, input_types);
             FunctionProto func_proto;
             op->BuildContextDependentFunction(ctx, func_proto);
             return nb::cast(std::move(func_proto));
           })
      .def("get_context_dependent_function_with_opset_version",
           [](const OpSchema *op, int opset_version, const NodeProto &node,
              const std::vector<TypeProto> &input_types) -> nb::object {
             if (!op->HasContextDependentFunctionWithOpsetVersion(opset_version))
               return nb::none();
             FunctionBodyBuildContextImpl ctx(node, input_types);
             FunctionProto func_proto;
             op->BuildContextDependentFunction(ctx, func_proto, opset_version);
             return nb::cast(std::move(func_proto));
           })
      .def(
          "_infer_node_outputs",
          [](const OpSchema *schema, const NodeProto &node,
             const std::unordered_map<std::string, TypeProto> &input_types,
             const std::unordered_map<std::string, TensorProto> &input_data,
             const std::unordered_map<std::string, SparseTensorProto> &input_sparse_data)
              -> std::unordered_map<std::string, TypeProto> {
            // Verify raises an exception if the node has the wrong number of
            // inputs or outputs as declared by the schema.  For skeleton
            // schemas (no .Input()/.Output() declarations) min_input and
            // max_input are both 0, meaning no constraint was specified.
            // Skipping the check in that case lets inference-only schemas
            // work for any arity.
            const bool has_input_constraints = schema->max_input() > 0;
            if (has_input_constraints) {
              schema->Verify(node);
            }
            NodeInferenceContextImpl ctx(node, input_types, input_data, input_sparse_data);
            if (schema->has_type_and_shape_inference_function()) {
              schema->GetTypeAndShapeInferenceFunction()(ctx);
            }
            if (has_input_constraints) {
              schema->CheckInputOutputType(ctx);
            }
            std::unordered_map<std::string, TypeProto> result;
            const auto &outputs = node.ref_output();
            for (size_t i = 0; i < ctx.all_output_types_.size(); ++i) {
              const TypeProto &proto = ctx.all_output_types_[i];
              if (proto.has_type()) {
                result[outputs[i].as_string()] = proto;
              }
            }
            return result;
          },
          nb::arg("node"), nb::arg("input_types"),
          nb::arg("input_data") = std::unordered_map<std::string, TensorProto>{},
          nb::arg("input_sparse_data") = std::unordered_map<std::string, SparseTensorProto>{},
          "Runs type and shape inference for a single node and returns output TypeProto map.");

  defs.def("register_onnx_operator_set_schema", &RegisterAllOnnxOperatorSchemas,
           "Registers all built-in ONNX operator schemas with type-and-shape inference "
           "functions across all opset versions.  Duplicate registrations are silently "
           "ignored so the function is safe to call more than once.")
      .def(
          "has_schema",
          [](const std::string &op_type, const std::string &domain) -> bool {
            return OpSchemaRegistry::Schema(op_type, domain) != nullptr;
          },
          nb::arg("op_type"), nb::arg("domain") = ONNX_DOMAIN)
      .def(
          "has_schema",
          [](const std::string &op_type, int max_inclusive_version,
             const std::string &domain) -> bool {
            return OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain) != nullptr;
          },
          nb::arg("op_type"), nb::arg("max_inclusive_version"), nb::arg("domain") = ONNX_DOMAIN)
      .def("schema_version_map",
           []() -> std::unordered_map<std::string, std::pair<int, int>> {
             return OpSchemaRegistry::DomainToVersionRange::Instance().Map();
           })
      .def(
          "get_schema",
          [](const std::string &op_type, const int max_inclusive_version,
             const std::string &domain) -> OpSchema {
            const auto *const schema =
                OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain);
            if (!schema) {
              fail_schema("No schema registered for '", op_type, "' version '",
                          max_inclusive_version, "' and domain '", domain, "'!");
            }
            return *schema;
          },
          nb::arg("op_type"), nb::arg("max_inclusive_version"), nb::arg("domain") = ONNX_DOMAIN,
          "Returns the schema of *op_type* for a specific version.")
      .def(
          "get_schema",
          [](const std::string &op_type, const std::string &domain) -> OpSchema {
            const auto *const schema = OpSchemaRegistry::Schema(op_type, domain);
            if (!schema) {
              fail_schema("No schema registered for '", op_type, "' and domain '", domain, "'!");
            }
            return *schema;
          },
          nb::arg("op_type"), nb::arg("domain") = ONNX_DOMAIN,
          "Returns the latest schema of *op_type*.")
      .def(
          "get_all_schemas",
          []() -> std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas(); },
          "Returns the schema of all registered operators at their latest version.")
      .def(
          "get_all_schemas_with_history",
          []() -> std::vector<OpSchema> {
            return OpSchemaRegistry::get_all_schemas_with_history();
          },
          "Returns the schema of all registered operators across all versions.")
      .def(
          "set_domain_to_version",
          [](const std::string &domain, int min_version, int max_version,
             int last_release_version) {
            auto &obj = OpSchemaRegistry::DomainToVersionRange::Instance();
            if (obj.Map().count(domain) == 0) {
              obj.AddDomainToVersion(domain, min_version, max_version, last_release_version);
            } else {
              obj.UpdateDomainToVersion(domain, min_version, max_version, last_release_version);
            }
          },
          nb::arg("domain"), nb::arg("min_version"), nb::arg("max_version"),
          nb::arg("last_release_version") = -1)
      .def(
          "register_schema",
          [](OpSchema schema) { RegisterSchema(std::move(schema), 0, true, true); },
          nb::arg("schema"), "Registers a user-provided OpSchema.")
      .def("deregister_schema", &DeregisterSchema, nb::arg("op_type"), nb::arg("version"),
           nb::arg("domain"), "Deregisters the specified OpSchema.");

  // -----------------------------------------------------------------------
  // Submodule `checker`
  // -----------------------------------------------------------------------
  auto checker_mod = m.def_submodule("checker");
  checker_mod.doc() = "Checker submodule";

  // nb::exception registers a new Python exception class and maps C++ throws of that type to it.
  // The RAII object is intentionally discarded after the registration side-effect completes.
  nb::exception<checker::ValidationError>(
      checker_mod,
      "ValidationError"); // NOLINT(bugprone-unused-raii,bugprone-throw-keyword-missing)

  checker_mod.def(
      "check_attribute",
      [](const AttributeProto &attribute) {
        checker::CheckerContext ctx;
        ctx.set_ir_version(IR_VERSION);
        checker::LexicalScopeContext lex;
        checker::check_attribute(attribute, ctx, lex);
      },
      nb::arg("attribute"), "Checks an AttributeProto and raises ValidationError on failure.");

  checker_mod.def(
      "check_sparse_tensor",
      [](const SparseTensorProto &sparse_tensor) {
        checker::CheckerContext ctx;
        ctx.set_ir_version(IR_VERSION);
        checker::check_sparse_tensor(sparse_tensor, ctx);
      },
      nb::arg("sparse_tensor"),
      "Checks a SparseTensorProto and raises ValidationError on failure.");

  checker_mod.def(
      "check_graph",
      [](const GraphProto &graph) {
        checker::CheckerContext ctx;
        ctx.set_ir_version(IR_VERSION);
        ctx.set_opset_imports({{ONNX_DOMAIN, 1}});
        checker::LexicalScopeContext lex;
        checker::check_graph(graph, ctx, lex);
      },
      nb::arg("graph"), "Checks a GraphProto and raises ValidationError on failure.");

  checker_mod.def(
      "check_model", [](const ModelProto &model) { checker::check_model(model); }, nb::arg("model"),
      "Checks model consistency and raises ValidationError on failure.");

  checker_mod.def(
      "check_function_call_cycles",
      [](const ModelProto &model) { inliner::CheckFunctionCallCycles(model); }, nb::arg("model"),
      "Checks for cycles in model-local function call graph. Raises ValidationError if a cycle is "
      "found.");

  // -----------------------------------------------------------------------
  // Submodule `inliner`
  // -----------------------------------------------------------------------
  auto inliner_mod = m.def_submodule("inliner");
  inliner_mod.doc() = "Inliner submodule";

  inliner_mod.def(
      "inline_local_functions",
      [](const ModelProto &model, bool convert_version) {
        inliner::CheckFunctionCallCycles(model);
        ModelProto copy;
        copy.CopyFrom(model);
        inliner::InlineLocalFunctions(copy, convert_version);
        return copy;
      },
      nb::arg("model"), nb::arg("convert_version") = false,
      "Inlines all model-local functions. Returns a new model with functions inlined. Raises "
      "checker.ValidationError if the model contains cyclic function references.");

  inliner_mod.def(
      "inline_selected_local_functions",
      [](const ModelProto &model,
         const std::vector<std::pair<std::string, std::string>> &function_ids, bool invert) {
        inliner::FunctionIdVector ids(function_ids.begin(), function_ids.end());
        auto id_set = inliner::FunctionIdSet::Create(std::move(ids), invert);
        ModelProto copy;
        copy.CopyFrom(model);
        inliner::InlineSelectedLocalFunctions(copy, *id_set);
        return copy;
      },
      nb::arg("model"), nb::arg("function_ids"), nb::arg("invert") = false,
      "Inlines the specified model-local functions. If invert is True, inlines all functions "
      "except those listed. Returns a new model with functions inlined.");

  inliner_mod.def(
      "inline_selected_functions",
      [](const ModelProto &model,
         const std::vector<std::pair<std::string, std::string>> &function_ids, bool invert) {
        inliner::FunctionIdVector ids(function_ids.begin(), function_ids.end());
        auto id_set = inliner::FunctionIdSet::Create(std::move(ids), invert);
        ModelProto copy;
        copy.CopyFrom(model);
        inliner::InlineSelectedFunctions(copy, *id_set, nullptr);
        return copy;
      },
      nb::arg("model"), nb::arg("function_ids"), nb::arg("invert") = false,
      "Inlines the specified functions including schema-defined functions. If invert is True, "
      "inlines all functions except those listed. Returns a new model with functions inlined.");
}
