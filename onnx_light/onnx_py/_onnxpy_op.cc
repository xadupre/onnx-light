// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "_onnxpyprotoop.h"
#include "onnx_core/tensor_type.h"
#include "onnx_op/light_op_schema.h"
#include "onnx_op/operator_sets.h"

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

  onnx_op_mod.attr("kOnnxDomain") = onnx_op::kOnnxDomain;

  nb::enum_<onnx_core::TensorType>(onnx_op_mod, "TensorType",
                                   "Element or sequence tensor type supported by onnx-light.")
      .value("kBool", onnx_core::TensorType::kBool)
      .value("kString", onnx_core::TensorType::kString)
      .value("kUint8", onnx_core::TensorType::kUint8)
      .value("kUint16", onnx_core::TensorType::kUint16)
      .value("kUint32", onnx_core::TensorType::kUint32)
      .value("kUint64", onnx_core::TensorType::kUint64)
      .value("kInt8", onnx_core::TensorType::kInt8)
      .value("kInt16", onnx_core::TensorType::kInt16)
      .value("kInt32", onnx_core::TensorType::kInt32)
      .value("kInt64", onnx_core::TensorType::kInt64)
      .value("kFloat16", onnx_core::TensorType::kFloat16)
      .value("kFloat", onnx_core::TensorType::kFloat)
      .value("kDouble", onnx_core::TensorType::kDouble)
      .value("kBfloat16", onnx_core::TensorType::kBfloat16)
      .value("kFloat8e4m3fn", onnx_core::TensorType::kFloat8e4m3fn)
      .value("kFloat8e4m3fnuz", onnx_core::TensorType::kFloat8e4m3fnuz)
      .value("kFloat8e5m2", onnx_core::TensorType::kFloat8e5m2)
      .value("kFloat8e5m2fnuz", onnx_core::TensorType::kFloat8e5m2fnuz)
      .value("kFloat8e8m0", onnx_core::TensorType::kFloat8e8m0)
      .value("kFloat4e2m1", onnx_core::TensorType::kFloat4e2m1)
      .value("kUint4", onnx_core::TensorType::kUint4)
      .value("kInt4", onnx_core::TensorType::kInt4)
      .value("kUint2", onnx_core::TensorType::kUint2)
      .value("kInt2", onnx_core::TensorType::kInt2)
      .value("kComplex64", onnx_core::TensorType::kComplex64)
      .value("kComplex128", onnx_core::TensorType::kComplex128)
      .value("kSeqBool", onnx_core::TensorType::kSeqBool)
      .value("kSeqString", onnx_core::TensorType::kSeqString)
      .value("kSeqUint8", onnx_core::TensorType::kSeqUint8)
      .value("kSeqUint16", onnx_core::TensorType::kSeqUint16)
      .value("kSeqUint32", onnx_core::TensorType::kSeqUint32)
      .value("kSeqUint64", onnx_core::TensorType::kSeqUint64)
      .value("kSeqInt8", onnx_core::TensorType::kSeqInt8)
      .value("kSeqInt16", onnx_core::TensorType::kSeqInt16)
      .value("kSeqInt32", onnx_core::TensorType::kSeqInt32)
      .value("kSeqInt64", onnx_core::TensorType::kSeqInt64)
      .value("kSeqFloat16", onnx_core::TensorType::kSeqFloat16)
      .value("kSeqFloat", onnx_core::TensorType::kSeqFloat)
      .value("kSeqDouble", onnx_core::TensorType::kSeqDouble)
      .value("kSeqComplex64", onnx_core::TensorType::kSeqComplex64)
      .value("kSeqComplex128", onnx_core::TensorType::kSeqComplex128)
      .value("kSeqMapStringFloat", onnx_core::TensorType::kSeqMapStringFloat)
      .value("kSeqMapInt64Float", onnx_core::TensorType::kSeqMapInt64Float)
      .value("kMapStringInt64", onnx_core::TensorType::kMapStringInt64)
      .value("kMapInt64String", onnx_core::TensorType::kMapInt64String)
      .value("kMapInt64Float", onnx_core::TensorType::kMapInt64Float)
      .value("kMapInt64Double", onnx_core::TensorType::kMapInt64Double)
      .value("kMapStringFloat", onnx_core::TensorType::kMapStringFloat)
      .value("kMapStringDouble", onnx_core::TensorType::kMapStringDouble)
      .value("kOptSeqBool", onnx_core::TensorType::kOptSeqBool)
      .value("kOptSeqString", onnx_core::TensorType::kOptSeqString)
      .value("kOptSeqUint8", onnx_core::TensorType::kOptSeqUint8)
      .value("kOptSeqUint16", onnx_core::TensorType::kOptSeqUint16)
      .value("kOptSeqUint32", onnx_core::TensorType::kOptSeqUint32)
      .value("kOptSeqUint64", onnx_core::TensorType::kOptSeqUint64)
      .value("kOptSeqInt8", onnx_core::TensorType::kOptSeqInt8)
      .value("kOptSeqInt16", onnx_core::TensorType::kOptSeqInt16)
      .value("kOptSeqInt32", onnx_core::TensorType::kOptSeqInt32)
      .value("kOptSeqInt64", onnx_core::TensorType::kOptSeqInt64)
      .value("kOptSeqFloat16", onnx_core::TensorType::kOptSeqFloat16)
      .value("kOptSeqFloat", onnx_core::TensorType::kOptSeqFloat)
      .value("kOptSeqDouble", onnx_core::TensorType::kOptSeqDouble)
      .value("kOptSeqComplex64", onnx_core::TensorType::kOptSeqComplex64)
      .value("kOptSeqComplex128", onnx_core::TensorType::kOptSeqComplex128)
      .value("kOptBool", onnx_core::TensorType::kOptBool)
      .value("kOptString", onnx_core::TensorType::kOptString)
      .value("kOptUint8", onnx_core::TensorType::kOptUint8)
      .value("kOptUint16", onnx_core::TensorType::kOptUint16)
      .value("kOptUint32", onnx_core::TensorType::kOptUint32)
      .value("kOptUint64", onnx_core::TensorType::kOptUint64)
      .value("kOptInt8", onnx_core::TensorType::kOptInt8)
      .value("kOptInt16", onnx_core::TensorType::kOptInt16)
      .value("kOptInt32", onnx_core::TensorType::kOptInt32)
      .value("kOptInt64", onnx_core::TensorType::kOptInt64)
      .value("kOptFloat16", onnx_core::TensorType::kOptFloat16)
      .value("kOptFloat", onnx_core::TensorType::kOptFloat)
      .value("kOptDouble", onnx_core::TensorType::kOptDouble)
      .value("kOptComplex64", onnx_core::TensorType::kOptComplex64)
      .value("kOptComplex128", onnx_core::TensorType::kOptComplex128)
      .value("kUndefined", onnx_core::TensorType::kUndefined);

  onnx_op_mod.def(
      "ToTypeString",
      [](onnx_core::TensorType type) { return std::string(onnx_core::ToTypeString(type)); },
      nb::arg("type"),
      "Returns the ONNX type-string representation of a TensorType value "
      "(e.g. ``\"tensor(float)\"`` or ``\"seq(tensor(int64))\"``).");

  nb::class_<onnx_op::FormalParameter>(
      onnx_op_mod, "FormalParameter",
      "A single formal input or output parameter of an ONNX operator.")
      .def(nb::init<>())
      .def_rw("name", &onnx_op::FormalParameter::name)
      .def_rw("description", &onnx_op::FormalParameter::description)
      .def_rw("type", &onnx_op::FormalParameter::type)
      .def("__repr__", [](const onnx_op::FormalParameter &p) {
        return "FormalParameter(name=" + ToPythonRepr(p.name) + ", type=" + ToPythonRepr(p.type) +
               ", description=" + ToPythonRepr(p.description) + ")";
      });

  nb::enum_<onnx_op::AttributeType>(onnx_op_mod, "AttributeType",
                                    "ONNX attribute type; mirrors "
                                    "``onnx::AttributeProto::AttributeType``.")
      .value("UNDEFINED", onnx_op::AttributeType::UNDEFINED)
      .value("FLOAT", onnx_op::AttributeType::FLOAT)
      .value("INT", onnx_op::AttributeType::INT)
      .value("STRING", onnx_op::AttributeType::STRING)
      .value("TENSOR", onnx_op::AttributeType::TENSOR)
      .value("GRAPH", onnx_op::AttributeType::GRAPH)
      .value("FLOATS", onnx_op::AttributeType::FLOATS)
      .value("INTS", onnx_op::AttributeType::INTS)
      .value("STRINGS", onnx_op::AttributeType::STRINGS)
      .value("TENSORS", onnx_op::AttributeType::TENSORS)
      .value("GRAPHS", onnx_op::AttributeType::GRAPHS)
      .value("SPARSE_TENSOR", onnx_op::AttributeType::SPARSE_TENSOR)
      .value("SPARSE_TENSORS", onnx_op::AttributeType::SPARSE_TENSORS)
      .value("TYPE_PROTO", onnx_op::AttributeType::TYPE_PROTO)
      .value("TYPE_PROTOS", onnx_op::AttributeType::TYPE_PROTOS);

  nb::class_<onnx_op::AttributeParam>(onnx_op_mod, "AttributeParam",
                                      "A single operator attribute as exposed by LightOpSchema.")
      .def(nb::init<>())
      .def_rw("name", &onnx_op::AttributeParam::name)
      .def_rw("description", &onnx_op::AttributeParam::description)
      .def_rw("type", &onnx_op::AttributeParam::type)
      .def_rw("required", &onnx_op::AttributeParam::required)
      .def("__repr__",
           [](const onnx_op::AttributeParam &a) {
             const auto quote = [](const std::string &s) {
               return nb::cast<std::string>(nb::repr(nb::cast(s)));
             };
             const std::string name = quote(a.name);
             const std::string description = quote(a.description);
             const std::string default_value = onnx_op::AttributeDefaultRepr(a.default_value);
             const std::string default_repr = default_value.empty() ? "None" : quote(default_value);
             std::ostringstream os;
             os << "AttributeParam(name=" << name << ", description=" << description
                << ", type=AttributeType." << onnx_op::AttributeType_Name(a.type)
                << ", required=" << (a.required ? "True" : "False")
                << ", default_value=" << default_repr << ")";
             return os.str();
           })
      .def_rw("default_value", &onnx_op::AttributeParam::default_value,
              "Typed default value (``None`` if absent, otherwise int, float, "
              "str, or a list thereof).")
      .def_prop_ro(
          "default_value_repr",
          [](const onnx_op::AttributeParam &a) {
            return onnx_op::AttributeDefaultRepr(a.default_value);
          },
          "Stable textual representation of ``default_value`` (empty when absent).");

  nb::class_<onnx_op::TypeConstraintParam>(
      onnx_op_mod, "TypeConstraintParam",
      "Specifies which tensor types are permitted for a named type parameter.")
      .def(nb::init<>())
      .def_rw("type_param_str", &onnx_op::TypeConstraintParam::type_param_str)
      .def_rw("allowed_type_strs", &onnx_op::TypeConstraintParam::allowed_type_strs)
      .def_rw("description", &onnx_op::TypeConstraintParam::description)
      .def("__repr__", [](const onnx_op::TypeConstraintParam &tc) {
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

  auto light_op_schema = nb::class_<onnx_op::LightOpSchema>(
      onnx_op_mod, "LightOpSchema",
      "Lightweight read-only description of an ONNX operator schema at one specific "
      "opset version.");

  nb::enum_<onnx_op::LightOpSchema::NodeDeterminism>(
      light_op_schema, "NodeDeterminism",
      "Whether evaluating the operator produces deterministic outputs.")
      .value("Unknown", onnx_op::LightOpSchema::NodeDeterminism::Unknown)
      .value("NonDeterministic", onnx_op::LightOpSchema::NodeDeterminism::NonDeterministic)
      .value("Deterministic", onnx_op::LightOpSchema::NodeDeterminism::Deterministic);

  light_op_schema
      .def(nb::init<std::string, std::string, int, std::string,
                    std::vector<onnx_op::FormalParameter>, std::vector<onnx_op::FormalParameter>,
                    std::vector<onnx_op::TypeConstraintParam>, bool>(),
           nb::arg("name"), nb::arg("domain"), nb::arg("since_version"), nb::arg("doc"),
           nb::arg("inputs"), nb::arg("outputs"), nb::arg("type_constraints"),
           nb::arg("has_function_implementation") = false)
      .def(nb::init<std::string, std::string, int, std::string,
                    std::vector<onnx_op::FormalParameter>, std::vector<onnx_op::FormalParameter>,
                    std::vector<onnx_op::TypeConstraintParam>, std::vector<onnx_op::AttributeParam>,
                    bool>(),
           nb::arg("name"), nb::arg("domain"), nb::arg("since_version"), nb::arg("doc"),
           nb::arg("inputs"), nb::arg("outputs"), nb::arg("type_constraints"),
           nb::arg("attributes"), nb::arg("has_function_implementation") = false)
      .def_prop_ro("name", &onnx_op::LightOpSchema::name)
      .def_prop_ro("domain", &onnx_op::LightOpSchema::domain)
      .def_prop_ro("since_version", &onnx_op::LightOpSchema::since_version)
      .def_prop_ro("doc", &onnx_op::LightOpSchema::doc)
      .def_prop_ro("inputs", &onnx_op::LightOpSchema::inputs)
      .def_prop_ro("outputs", &onnx_op::LightOpSchema::outputs)
      .def_prop_ro("type_constraints", &onnx_op::LightOpSchema::type_constraints)
      .def_prop_ro("attributes", &onnx_op::LightOpSchema::attributes)
      .def_prop_ro("has_function_implementation",
                   &onnx_op::LightOpSchema::has_function_implementation)
      .def_prop_ro("min_output", &onnx_op::LightOpSchema::min_output,
                   "Minimum number of outputs (defaults to ``len(outputs)``).")
      .def_prop_ro("max_output", &onnx_op::LightOpSchema::max_output,
                   "Maximum number of outputs (defaults to ``len(outputs)``; can be "
                   "``INT_MAX`` for unbounded variadic outputs).")
      .def_prop_ro("deprecated", &onnx_op::LightOpSchema::deprecated,
                   "True if this versioned operator is deprecated.")
      .def_prop_ro("node_determinism", &onnx_op::LightOpSchema::node_determinism,
                   "Node determinism of the operator (``NodeDeterminism.Unknown`` when "
                   "unspecified).")
      .def_prop_ro("non_deterministic", &onnx_op::LightOpSchema::non_deterministic,
                   "True if the operator is explicitly marked non-deterministic.")
      .def("set_min_output", &onnx_op::LightOpSchema::set_min_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the minimum number of outputs and returns ``self``.")
      .def("set_max_output", &onnx_op::LightOpSchema::set_max_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the maximum number of outputs and returns ``self``.")
      .def("set_deprecated", &onnx_op::LightOpSchema::set_deprecated, nb::arg("value") = true,
           nb::rv_policy::reference_internal,
           "Marks this operator as deprecated and returns ``self``.")
      .def("set_node_determinism", &onnx_op::LightOpSchema::set_node_determinism, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the operator's node determinism and returns ``self``.")
      .def("__repr__", [](const onnx_op::LightOpSchema &s) {
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
        const auto determinism_name = [](onnx_op::LightOpSchema::NodeDeterminism nd) {
          switch (nd) {
          case onnx_op::LightOpSchema::NodeDeterminism::NonDeterministic:
            return "NonDeterministic";
          case onnx_op::LightOpSchema::NodeDeterminism::Deterministic:
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
