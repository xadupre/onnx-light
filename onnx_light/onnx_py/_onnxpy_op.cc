// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "_onnxpyprotoop.h"
#include "onnx_core/light_op_schema/light_op_schema.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/type_helper.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <sstream>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {
std::string ToPythonRepr(const std::string &value) {
  return nb::cast<std::string>(nb::repr(nb::cast(value)));
}
} // namespace

void AddOnnxPyOp(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `onnx_op`
  // Read-only light-weight operator schema descriptors and the registration
  // function returning the complete versioned schema history for every
  // supported ONNX domain.
  // -----------------------------------------------------------------------
  auto onnx_op_mod = m.def_submodule("onnx_op");
  onnx_op_mod.doc() = "Light-weight ONNX operator schema descriptors (LightOpSchema) and the "
                      "registration function GetAllOnnxOpSchemasWithHistory exposed from C++.";

  onnx_op_mod.attr("kOnnxDomain") = core::schema::kOnnxDomain;

  nb::enum_<onnx_proto::TensorType>(onnx_op_mod, "TensorType",
                                    "Element or sequence tensor type supported by onnx-light.")
      .value("kBool", onnx_proto::TensorType::kBool)
      .value("kString", onnx_proto::TensorType::kString)
      .value("kUint8", onnx_proto::TensorType::kUint8)
      .value("kUint16", onnx_proto::TensorType::kUint16)
      .value("kUint32", onnx_proto::TensorType::kUint32)
      .value("kUint64", onnx_proto::TensorType::kUint64)
      .value("kInt8", onnx_proto::TensorType::kInt8)
      .value("kInt16", onnx_proto::TensorType::kInt16)
      .value("kInt32", onnx_proto::TensorType::kInt32)
      .value("kInt64", onnx_proto::TensorType::kInt64)
      .value("kFloat16", onnx_proto::TensorType::kFloat16)
      .value("kFloat", onnx_proto::TensorType::kFloat)
      .value("kDouble", onnx_proto::TensorType::kDouble)
      .value("kBfloat16", onnx_proto::TensorType::kBfloat16)
      .value("kFloat8e4m3fn", onnx_proto::TensorType::kFloat8e4m3fn)
      .value("kFloat8e4m3fnuz", onnx_proto::TensorType::kFloat8e4m3fnuz)
      .value("kFloat8e5m2", onnx_proto::TensorType::kFloat8e5m2)
      .value("kFloat8e5m2fnuz", onnx_proto::TensorType::kFloat8e5m2fnuz)
      .value("kFloat8e8m0", onnx_proto::TensorType::kFloat8e8m0)
      .value("kFloat4e2m1", onnx_proto::TensorType::kFloat4e2m1)
      .value("kUint4", onnx_proto::TensorType::kUint4)
      .value("kInt4", onnx_proto::TensorType::kInt4)
      .value("kUint2", onnx_proto::TensorType::kUint2)
      .value("kInt2", onnx_proto::TensorType::kInt2)
      .value("kFloat6e2m3", onnx_proto::TensorType::kFloat6e2m3)
      .value("kFloat6e3m2", onnx_proto::TensorType::kFloat6e3m2)
      .value("kComplex64", onnx_proto::TensorType::kComplex64)
      .value("kComplex128", onnx_proto::TensorType::kComplex128)
      .value("kSeqBool", onnx_proto::TensorType::kSeqBool)
      .value("kSeqString", onnx_proto::TensorType::kSeqString)
      .value("kSeqUint8", onnx_proto::TensorType::kSeqUint8)
      .value("kSeqUint16", onnx_proto::TensorType::kSeqUint16)
      .value("kSeqUint32", onnx_proto::TensorType::kSeqUint32)
      .value("kSeqUint64", onnx_proto::TensorType::kSeqUint64)
      .value("kSeqInt8", onnx_proto::TensorType::kSeqInt8)
      .value("kSeqInt16", onnx_proto::TensorType::kSeqInt16)
      .value("kSeqInt32", onnx_proto::TensorType::kSeqInt32)
      .value("kSeqInt64", onnx_proto::TensorType::kSeqInt64)
      .value("kSeqFloat16", onnx_proto::TensorType::kSeqFloat16)
      .value("kSeqFloat", onnx_proto::TensorType::kSeqFloat)
      .value("kSeqDouble", onnx_proto::TensorType::kSeqDouble)
      .value("kSeqComplex64", onnx_proto::TensorType::kSeqComplex64)
      .value("kSeqComplex128", onnx_proto::TensorType::kSeqComplex128)
      .value("kSeqMapStringFloat", onnx_proto::TensorType::kSeqMapStringFloat)
      .value("kSeqMapInt64Float", onnx_proto::TensorType::kSeqMapInt64Float)
      .value("kMapStringInt64", onnx_proto::TensorType::kMapStringInt64)
      .value("kMapInt64String", onnx_proto::TensorType::kMapInt64String)
      .value("kMapInt64Float", onnx_proto::TensorType::kMapInt64Float)
      .value("kMapInt64Double", onnx_proto::TensorType::kMapInt64Double)
      .value("kMapStringFloat", onnx_proto::TensorType::kMapStringFloat)
      .value("kMapStringDouble", onnx_proto::TensorType::kMapStringDouble)
      .value("kOptSeqBool", onnx_proto::TensorType::kOptSeqBool)
      .value("kOptSeqString", onnx_proto::TensorType::kOptSeqString)
      .value("kOptSeqUint8", onnx_proto::TensorType::kOptSeqUint8)
      .value("kOptSeqUint16", onnx_proto::TensorType::kOptSeqUint16)
      .value("kOptSeqUint32", onnx_proto::TensorType::kOptSeqUint32)
      .value("kOptSeqUint64", onnx_proto::TensorType::kOptSeqUint64)
      .value("kOptSeqInt8", onnx_proto::TensorType::kOptSeqInt8)
      .value("kOptSeqInt16", onnx_proto::TensorType::kOptSeqInt16)
      .value("kOptSeqInt32", onnx_proto::TensorType::kOptSeqInt32)
      .value("kOptSeqInt64", onnx_proto::TensorType::kOptSeqInt64)
      .value("kOptSeqFloat16", onnx_proto::TensorType::kOptSeqFloat16)
      .value("kOptSeqFloat", onnx_proto::TensorType::kOptSeqFloat)
      .value("kOptSeqDouble", onnx_proto::TensorType::kOptSeqDouble)
      .value("kOptSeqComplex64", onnx_proto::TensorType::kOptSeqComplex64)
      .value("kOptSeqComplex128", onnx_proto::TensorType::kOptSeqComplex128)
      .value("kOptBool", onnx_proto::TensorType::kOptBool)
      .value("kOptString", onnx_proto::TensorType::kOptString)
      .value("kOptUint8", onnx_proto::TensorType::kOptUint8)
      .value("kOptUint16", onnx_proto::TensorType::kOptUint16)
      .value("kOptUint32", onnx_proto::TensorType::kOptUint32)
      .value("kOptUint64", onnx_proto::TensorType::kOptUint64)
      .value("kOptInt8", onnx_proto::TensorType::kOptInt8)
      .value("kOptInt16", onnx_proto::TensorType::kOptInt16)
      .value("kOptInt32", onnx_proto::TensorType::kOptInt32)
      .value("kOptInt64", onnx_proto::TensorType::kOptInt64)
      .value("kOptFloat16", onnx_proto::TensorType::kOptFloat16)
      .value("kOptFloat", onnx_proto::TensorType::kOptFloat)
      .value("kOptDouble", onnx_proto::TensorType::kOptDouble)
      .value("kOptComplex64", onnx_proto::TensorType::kOptComplex64)
      .value("kOptComplex128", onnx_proto::TensorType::kOptComplex128)
      .value("kUndefined", onnx_proto::TensorType::kUndefined);

  onnx_op_mod.def(
      "ToTypeString",
      [](onnx_proto::TensorType type) { return std::string(onnx_proto::ToTypeString(type)); },
      nb::arg("type"),
      "Returns the ONNX type-string representation of a TensorType value "
      "(e.g. ``\"tensor(float)\"`` or ``\"seq(tensor(int64))\"``).");

  nb::class_<core::schema::FormalParameter>(
      onnx_op_mod, "FormalParameter",
      "A single formal input or output parameter of an ONNX operator.")
      .def(nb::init<>())
      .def_rw("name", &core::schema::FormalParameter::name)
      .def_rw("description", &core::schema::FormalParameter::description)
      .def_rw("type", &core::schema::FormalParameter::type)
      .def("__repr__", [](const core::schema::FormalParameter &p) {
        return "FormalParameter(name=" + ToPythonRepr(p.name) + ", type=" + ToPythonRepr(p.type) +
               ", description=" + ToPythonRepr(p.description) + ")";
      });

  nb::enum_<core::schema::AttributeType>(onnx_op_mod, "AttributeType",
                                         "ONNX attribute type; mirrors "
                                         "``onnx::AttributeProto::AttributeType``.")
      .value("UNDEFINED", core::schema::AttributeType::UNDEFINED)
      .value("FLOAT", core::schema::AttributeType::FLOAT)
      .value("INT", core::schema::AttributeType::INT)
      .value("STRING", core::schema::AttributeType::STRING)
      .value("TENSOR", core::schema::AttributeType::TENSOR)
      .value("GRAPH", core::schema::AttributeType::GRAPH)
      .value("FLOATS", core::schema::AttributeType::FLOATS)
      .value("INTS", core::schema::AttributeType::INTS)
      .value("STRINGS", core::schema::AttributeType::STRINGS)
      .value("TENSORS", core::schema::AttributeType::TENSORS)
      .value("GRAPHS", core::schema::AttributeType::GRAPHS)
      .value("SPARSE_TENSOR", core::schema::AttributeType::SPARSE_TENSOR)
      .value("SPARSE_TENSORS", core::schema::AttributeType::SPARSE_TENSORS)
      .value("TYPE_PROTO", core::schema::AttributeType::TYPE_PROTO)
      .value("TYPE_PROTOS", core::schema::AttributeType::TYPE_PROTOS);

  nb::class_<core::schema::AttributeParam>(
      onnx_op_mod, "AttributeParam", "A single operator attribute as exposed by LightOpSchema.")
      .def(nb::init<>())
      .def_rw("name", &core::schema::AttributeParam::name)
      .def_rw("description", &core::schema::AttributeParam::description)
      .def_rw("type", &core::schema::AttributeParam::type)
      .def_rw("required", &core::schema::AttributeParam::required)
      .def("__repr__",
           [](const core::schema::AttributeParam &a) {
             const auto quote = [](const std::string &s) {
               return nb::cast<std::string>(nb::repr(nb::cast(s)));
             };
             const std::string name = quote(a.name);
             const std::string description = quote(a.description);
             const std::string default_value = core::schema::AttributeDefaultRepr(a.default_value);
             const std::string default_repr = default_value.empty() ? "None" : quote(default_value);
             std::ostringstream os;
             os << "AttributeParam(name=" << name << ", description=" << description
                << ", type=AttributeType." << core::schema::AttributeType_Name(a.type)
                << ", required=" << (a.required ? "True" : "False")
                << ", default_value=" << default_repr << ")";
             return os.str();
           })
      .def_rw("default_value", &core::schema::AttributeParam::default_value,
              "Typed default value (``None`` if absent, otherwise int, float, "
              "str, or a list thereof).")
      .def_prop_ro(
          "default_value_repr",
          [](const core::schema::AttributeParam &a) {
            return core::schema::AttributeDefaultRepr(a.default_value);
          },
          "Stable textual representation of ``default_value`` (empty when absent).");

  nb::class_<core::schema::TypeConstraintParam>(
      onnx_op_mod, "TypeConstraintParam",
      "Specifies which tensor types are permitted for a named type parameter.")
      .def(nb::init<>())
      .def_rw("type_param_str", &core::schema::TypeConstraintParam::type_param_str)
      .def_rw("allowed_type_strs", &core::schema::TypeConstraintParam::allowed_type_strs)
      .def_rw("description", &core::schema::TypeConstraintParam::description)
      .def("__repr__", [](const core::schema::TypeConstraintParam &tc) {
        std::ostringstream os;
        os << "TypeConstraintParam(type_param_str=" << ToPythonRepr(tc.type_param_str)
           << ", allowed_type_strs=[";
        for (std::size_t i = 0; i < tc.allowed_type_strs.size(); ++i) {
          if (i != 0)
            os << ", ";
          os << nb::cast<std::string>(nb::repr(nb::cast(tc.allowed_type_strs[i])));
        }
        os << "], description=" << ToPythonRepr(tc.description) << ")";
        return os.str();
      });

  auto light_op_schema = nb::class_<core::schema::LightOpSchema>(
      onnx_op_mod, "LightOpSchema",
      "Lightweight read-only description of an ONNX operator schema at one specific "
      "opset version.");

  nb::enum_<core::schema::LightOpSchema::NodeDeterminism>(
      light_op_schema, "NodeDeterminism",
      "Whether evaluating the operator produces deterministic outputs.")
      .value("Unknown", core::schema::LightOpSchema::NodeDeterminism::Unknown)
      .value("NonDeterministic", core::schema::LightOpSchema::NodeDeterminism::NonDeterministic)
      .value("Deterministic", core::schema::LightOpSchema::NodeDeterminism::Deterministic);

  light_op_schema
      .def(nb::init<std::string, std::string, int, std::string,
                    std::vector<core::schema::FormalParameter>,
                    std::vector<core::schema::FormalParameter>,
                    std::vector<core::schema::TypeConstraintParam>, bool>(),
           nb::arg("name"), nb::arg("domain"), nb::arg("since_version"), nb::arg("doc"),
           nb::arg("inputs"), nb::arg("outputs"), nb::arg("type_constraints"),
           nb::arg("has_function_implementation") = false)
      .def(nb::init<std::string, std::string, int, std::string,
                    std::vector<core::schema::FormalParameter>,
                    std::vector<core::schema::FormalParameter>,
                    std::vector<core::schema::TypeConstraintParam>,
                    std::vector<core::schema::AttributeParam>, bool>(),
           nb::arg("name"), nb::arg("domain"), nb::arg("since_version"), nb::arg("doc"),
           nb::arg("inputs"), nb::arg("outputs"), nb::arg("type_constraints"),
           nb::arg("attributes"), nb::arg("has_function_implementation") = false)
      .def_prop_ro("name", &core::schema::LightOpSchema::name)
      .def_prop_ro("domain", &core::schema::LightOpSchema::domain)
      .def_prop_ro("since_version", &core::schema::LightOpSchema::since_version)
      .def_prop_ro("doc", &core::schema::LightOpSchema::doc)
      .def_prop_ro("inputs", &core::schema::LightOpSchema::inputs)
      .def_prop_ro("outputs", &core::schema::LightOpSchema::outputs)
      .def_prop_ro("type_constraints", &core::schema::LightOpSchema::type_constraints)
      .def_prop_ro("attributes", &core::schema::LightOpSchema::attributes)
      .def_prop_ro("has_function_implementation",
                   &core::schema::LightOpSchema::has_function_implementation)
      .def_prop_ro("min_output", &core::schema::LightOpSchema::min_output,
                   "Minimum number of outputs (defaults to ``len(outputs)``).")
      .def_prop_ro("max_output", &core::schema::LightOpSchema::max_output,
                   "Maximum number of outputs (defaults to ``len(outputs)``; can be "
                   "``INT_MAX`` for unbounded variadic outputs).")
      .def_prop_ro("deprecated", &core::schema::LightOpSchema::deprecated,
                   "True if this versioned operator is deprecated.")
      .def_prop_ro("node_determinism", &core::schema::LightOpSchema::node_determinism,
                   "Node determinism of the operator (``NodeDeterminism.Unknown`` when "
                   "unspecified).")
      .def_prop_ro("non_deterministic", &core::schema::LightOpSchema::non_deterministic,
                   "True if the operator is explicitly marked non-deterministic.")
      .def("set_min_output", &core::schema::LightOpSchema::set_min_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the minimum number of outputs and returns ``self``.")
      .def("set_max_output", &core::schema::LightOpSchema::set_max_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the maximum number of outputs and returns ``self``.")
      .def("set_deprecated", &core::schema::LightOpSchema::set_deprecated, nb::arg("value") = true,
           nb::rv_policy::reference_internal,
           "Marks this operator as deprecated and returns ``self``.")
      .def("set_node_determinism", &core::schema::LightOpSchema::set_node_determinism,
           nb::arg("value"), nb::rv_policy::reference_internal,
           "Sets the operator's node determinism and returns ``self``.")
      .def(
          "verify",
          [](const core::schema::LightOpSchema &schema, const NodeProto &node,
             std::optional<std::vector<std::optional<core::schema::SchemaInputValue>>> inputs) {
            schema.Verify(node, inputs.has_value() ? &*inputs : nullptr);
          },
          nb::arg("node"), nb::arg("inputs") = nb::none(),
          "Verifies that ``node`` follows this operator schema's constraints: "
          "``op_type``/``domain`` match, the schema is not deprecated, output arity is "
          "within ``[min_output, max_output]``, and every attribute is recognized, has "
          "the expected type, and every required attribute is present (attribute names "
          "starting with ``__`` are exempt, as they carry internal/optimizer-only data). "
          "``inputs``, when given, is a list aligned with ``node.input``: each entry is "
          "either a ``ValueInfoProto``, an ``SymTensor``, an ``SymSequence``, or ``None`` "
          "to skip that input. When present, an input's resolved type is checked against "
          "the operator's type constraint for that formal parameter (the last formal "
          "parameter is reused for variadic inputs); inputs whose type cannot be resolved "
          "are silently skipped. Raises ``SchemaError`` on the first violation found.")
      .def("__repr__", [](const core::schema::LightOpSchema &s) {
        const auto list_repr = [](const auto &items) {
          std::ostringstream os;
          os << "[";
          for (std::size_t i = 0; i < items.size(); ++i) {
            if (i != 0)
              os << ", ";
            os << nb::cast<std::string>(nb::repr(nb::cast(items[i])));
          }
          os << "]";
          return os.str();
        };
        const auto determinism_name = [](core::schema::LightOpSchema::NodeDeterminism nd) {
          switch (nd) {
          case core::schema::LightOpSchema::NodeDeterminism::NonDeterministic:
            return "NonDeterministic";
          case core::schema::LightOpSchema::NodeDeterminism::Deterministic:
            return "Deterministic";
          default:
            return "Unknown";
          }
        };
        std::ostringstream os;
        os << "LightOpSchema(name=" << ToPythonRepr(s.name())
           << ", domain=" << ToPythonRepr(s.domain()) << ", since_version=" << s.since_version()
           << ", inputs=" << list_repr(s.inputs()) << ", outputs=" << list_repr(s.outputs())
           << ", type_constraints=" << list_repr(s.type_constraints())
           << ", attributes=" << list_repr(s.attributes()) << ", has_function_implementation="
           << (s.has_function_implementation() ? "True" : "False")
           << ", min_output=" << s.min_output() << ", max_output=" << s.max_output()
           << ", deprecated=" << (s.deprecated() ? "True" : "False")
           << ", node_determinism=NodeDeterminism." << determinism_name(s.node_determinism())
           << ")";
        return os.str();
      });

  onnx_op_mod.def("GetAllOnnxOpSchemasWithHistory", &onnx_op::GetAllOnnxOpSchemasWithHistory,
                  nb::arg("op_type") = std::string(), nb::arg("init_doc") = true,
                  "Returns the complete versioned schema history for all supported ONNX "
                  "operator domains as a list of LightOpSchema. When ``op_type`` is non-empty, "
                  "only schemas matching that operator name are returned.");
}
