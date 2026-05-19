#include "../onnx_proto/_onnxpy.h"
#include "cc_onnx_expressions/expressions.h"
#include "onnx.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/parser.h"
#include "onnx_lib/defs/schema.h"
#include "onnx_lib/defs/shape_inference.h"
#include "onnx_lib/inliner/inliner.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_lib/version_converter/errors.h"
#include <algorithm>
#include <limits>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <unordered_map>
#include <unordered_set>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

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

  shape_inference_mod.def(
      "infer_function_output_types",
      [](const FunctionProto &function, const std::vector<TypeProto> &input_types,
         const std::vector<AttributeProto> &attributes) -> nb::list {
        std::vector<TypeProto> output_types =
            shape_inference::InferFunctionOutputTypes(function, input_types, attributes);
        nb::list result;
        for (auto &type_proto : output_types) {
          result.append(nb::cast(type_proto));
        }
        return result;
      },
      nb::arg("function"), nb::arg("input_types"), nb::arg("attributes"),
      "Infers output types of a FunctionProto given serialized input TypeProtos and "
      "AttributeProtos.");

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
      .def_prop_ro("non_deterministic",
                   [](const OpSchema *op) {
                     return op->GetNodeDeterminism() == OpSchema::NodeDeterminism::NonDeterministic;
                   })
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
          [](const OpSchema *schema, NodeProto &node,
             std::unordered_map<std::string, TypeProto *> input_types,
             std::unordered_map<std::string, TensorProto *> input_data,
             std::unordered_map<std::string, SparseTensorProto *> input_sparse_data,
             std::unordered_map<std::string, int> opset_imports,
             int ir_version) -> std::unordered_map<std::string, TypeProto> {
            // ``node`` and the value maps are taken by (non-const) reference /
            // pointer because the underlying protos hold move-only members and
            // cannot be copied; nanobind hands us references to the Python-
            // owned instances which we use directly as backing storage for
            // the inference context.
            // Early fail if node is badly defined - may throw ValidationError.
            schema->Verify(node);
            if (opset_imports.empty()) {
              opset_imports[schema->domain()] = schema->SinceVersion();
            }

            // Adapt the bound maps to the API expected by the inference
            // context, which takes pointers to const protos for the data
            // inputs and pointers to mutable TypeProto for the value-types.
            std::unordered_map<std::string, const TensorProto *> input_data_by_name;
            input_data_by_name.reserve(input_data.size());
            for (const auto &kv : input_data) {
              input_data_by_name[kv.first] = kv.second;
            }
            std::unordered_map<std::string, const SparseTensorProto *> input_sparse_data_by_name;
            input_sparse_data_by_name.reserve(input_sparse_data.size());
            for (const auto &kv : input_sparse_data) {
              input_sparse_data_by_name[kv.first] = kv.second;
            }

            shape_inference::GraphInferenceContext graph_inference_context(
                input_types, opset_imports,
                /*symbol_table_in=*/nullptr,
                /*model_local_functions_in=*/{},
                /*schema_registry_in=*/OpSchemaRegistry::Instance(),
                /*generated_shape_data_by_name_in=*/nullptr,
                /*ir_version_in=*/ir_version);

            // Construct inference context and get results - may throw
            // InferenceError.  We always use the default options here; if it
            // is desirable for infer_node_outputs to provide check_type,
            // strict_mode, data_prop, we can add them to the Python API.
            ShapeInferenceOptions options(/*check_type_val=*/false,
                                          /*strict_mode_val=*/0,
                                          /*data_prop_val=*/false);
            shape_inference::InferenceContextImpl ctx(
                node, input_types, input_data_by_name, input_sparse_data_by_name, options,
                /*generatedShapeData=*/nullptr, &graph_inference_context);

            schema->GetTypeAndShapeInferenceFunction()(ctx);
            // Verify the inference succeeded - may also throw
            // ValidationError.  Note that input types were not validated
            // until now (except that their count was correct).
            schema->CheckInputOutputType(ctx);

            std::unordered_map<std::string, TypeProto> result;
            for (size_t i = 0; i < ctx.allOutputTypes_.size(); ++i) {
              auto &proto = ctx.allOutputTypes_[i];
              if (proto.value_case() != TypeProto::VALUE_NOT_SET) {
                result.emplace(node.output(static_cast<int>(i)).as_string(), std::move(proto));
              }
            }
            return result;
          },
          nb::arg("node"), nb::arg("input_types"),
          nb::arg("input_data") = std::unordered_map<std::string, TensorProto *>{},
          nb::arg("input_sparse_data") = std::unordered_map<std::string, SparseTensorProto *>{},
          nb::arg("opset_imports") = std::unordered_map<std::string, int>{},
          nb::arg("ir_version") = static_cast<int>(IR_VERSION),
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
            // should we register if nothing has been done?
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
            // should we register if nothing has been done?
            if (!schema) {
              fail_schema("No schema registered for '", op_type, "' and domain '", domain, "'");
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
      "check_model",
      [](const ModelProto &model) {
        std::unordered_set<std::string> keys;
        for (const StringStringEntryProto &entry : model.metadata_props()) {
          const std::string key = entry.key().as_string();
          if (!keys.insert(key).second) {
            throw checker::ValidationError("Model contains duplicate keys in metadata_props.");
          }
        }
      },
      nb::arg("model"),
      "Checks model metadata consistency and raises ValidationError on duplicate keys.");

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

  // -----------------------------------------------------------------------
  // Submodule `expressions`
  // Symbolic dimension-expression utilities (simplify, evaluate, rename).
  // -----------------------------------------------------------------------
  {
    namespace expr = ::onnx_light::expressions;

    auto expressions_mod = m.def_submodule("expressions");
    expressions_mod.doc() =
        "Symbolic dimension expression utilities: simplify, evaluate, and rename expressions.";

    // simplify_expression(str | int) -> str | int
    expressions_mod.def(
        "simplify_expression",
        [](nb::object arg) -> nb::object {
          if (nb::isinstance<nb::int_>(arg)) {
            int64_t v = nb::cast<int64_t>(arg);
            auto r = expr::simplify_expression(v);
            return std::holds_alternative<int64_t>(r) ? nb::cast(std::get<int64_t>(r))
                                                      : nb::cast(std::get<std::string>(r));
          }
          std::string s = nb::cast<std::string>(arg);
          auto r = expr::simplify_expression(s);
          return std::holds_alternative<int64_t>(r) ? nb::cast(std::get<int64_t>(r))
                                                    : nb::cast(std::get<std::string>(r));
        },
        nb::arg("expr"),
        "Simplifies a symbolic or integer expression. Returns int when fully numeric, "
        "otherwise a simplified string.");

    // simplify_two_expressions(expr1, expr2) -> dict[str, int]
    expressions_mod.def(
        "simplify_two_expressions",
        [](const std::string &e1, const std::string &e2) {
          return expr::simplify_two_expressions(e1, e2);
        },
        nb::arg("expr1"), nb::arg("expr2"),
        "Returns the non-zero coefficient map of (expr1) - (expr2).");

    // evaluate_expression(expr, context) -> int
    expressions_mod.def(
        "evaluate_expression",
        [](const std::string &e, const std::unordered_map<std::string, int64_t> &ctx) {
          return expr::evaluate_expression(e, ctx);
        },
        nb::arg("expression"), nb::arg("context"),
        "Evaluates an expression given variable assignments. Returns an integer.");

    // parse_expression_tokens(expr) -> set[str]
    expressions_mod.def(
        "parse_expression_tokens",
        [](const std::string &e) { return expr::parse_expression_tokens(e); }, nb::arg("expr"),
        "Returns the set of variable names referenced in the expression.");

    // rename_expression(expr, mapping) -> str
    expressions_mod.def(
        "rename_expression",
        [](const std::string &e, const std::unordered_map<std::string, std::string> &m) {
          return expr::rename_expression(e, m);
        },
        nb::arg("expr"), nb::arg("mapping"),
        "Renames variables in an expression using the provided mapping.");

    // rename_dynamic_expression(expression, replacements) -> str
    expressions_mod.def(
        "rename_dynamic_expression",
        [](const std::string &e, const std::unordered_map<std::string, std::string> &r) {
          return expr::rename_dynamic_expression(e, r);
        },
        nb::arg("expression"), nb::arg("replacements"),
        "Renames variables and applies Max->xor conversion and simplification.");

    // dim_add / dim_sub / dim_mul / dim_div / dim_mod / dim_max / dim_min
    // Each accepts int | str for both arguments and returns int | str.
    auto to_dim = [](nb::object arg) -> expr::DimType {
      if (nb::isinstance<nb::int_>(arg))
        return nb::cast<int64_t>(arg);
      return nb::cast<std::string>(arg);
    };
    auto from_dim = [](const expr::DimType &d) -> nb::object {
      if (std::holds_alternative<int64_t>(d))
        return nb::cast(std::get<int64_t>(d));
      return nb::cast(std::get<std::string>(d));
    };

    expressions_mod.def(
        "dim_add",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_add(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Adds two dimensions.");

    expressions_mod.def(
        "dim_sub",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_sub(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Subtracts dimension b from a.");

    expressions_mod.def(
        "dim_mul",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_mul(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Multiplies two dimensions.");

    expressions_mod.def(
        "dim_multi_mul",
        [to_dim, from_dim](nb::list args) {
          std::vector<expr::DimType> dims;
          dims.reserve(nb::len(args));
          for (auto item : args)
            dims.push_back(to_dim(nb::cast<nb::object>(item)));
          return from_dim(expr::dim_multi_mul(dims));
        },
        nb::arg("args"), "Multiplies a sequence of dimensions.");

    expressions_mod.def(
        "dim_div",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_div(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Floor-divides dimension a by b.");

    expressions_mod.def(
        "dim_mod",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_mod(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Computes a modulo b.");

    expressions_mod.def(
        "dim_max",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_max(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Returns the maximum of two dimensions.");

    expressions_mod.def(
        "dim_min",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_min(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Returns the minimum of two dimensions.");
  }
}
