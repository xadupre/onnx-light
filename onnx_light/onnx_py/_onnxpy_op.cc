// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"
#include "onnx_op/light_op_schema.h"
#include "onnx_op/operator_sets.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

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

  nb::enum_<onnx_op::TensorType>(onnx_op_mod, "TensorType",
                                 "Element or sequence tensor type supported by onnx-light.")
      .value("kBool", onnx_op::TensorType::kBool)
      .value("kString", onnx_op::TensorType::kString)
      .value("kUint8", onnx_op::TensorType::kUint8)
      .value("kUint16", onnx_op::TensorType::kUint16)
      .value("kUint32", onnx_op::TensorType::kUint32)
      .value("kUint64", onnx_op::TensorType::kUint64)
      .value("kInt8", onnx_op::TensorType::kInt8)
      .value("kInt16", onnx_op::TensorType::kInt16)
      .value("kInt32", onnx_op::TensorType::kInt32)
      .value("kInt64", onnx_op::TensorType::kInt64)
      .value("kFloat16", onnx_op::TensorType::kFloat16)
      .value("kFloat", onnx_op::TensorType::kFloat)
      .value("kDouble", onnx_op::TensorType::kDouble)
      .value("kBfloat16", onnx_op::TensorType::kBfloat16)
      .value("kFloat8e4m3fn", onnx_op::TensorType::kFloat8e4m3fn)
      .value("kFloat8e4m3fnuz", onnx_op::TensorType::kFloat8e4m3fnuz)
      .value("kFloat8e5m2", onnx_op::TensorType::kFloat8e5m2)
      .value("kFloat8e5m2fnuz", onnx_op::TensorType::kFloat8e5m2fnuz)
      .value("kFloat8e8m0", onnx_op::TensorType::kFloat8e8m0)
      .value("kFloat4e2m1", onnx_op::TensorType::kFloat4e2m1)
      .value("kUint4", onnx_op::TensorType::kUint4)
      .value("kInt4", onnx_op::TensorType::kInt4)
      .value("kUint2", onnx_op::TensorType::kUint2)
      .value("kInt2", onnx_op::TensorType::kInt2)
      .value("kComplex64", onnx_op::TensorType::kComplex64)
      .value("kComplex128", onnx_op::TensorType::kComplex128)
      .value("kSeqBool", onnx_op::TensorType::kSeqBool)
      .value("kSeqString", onnx_op::TensorType::kSeqString)
      .value("kSeqUint8", onnx_op::TensorType::kSeqUint8)
      .value("kSeqUint16", onnx_op::TensorType::kSeqUint16)
      .value("kSeqUint32", onnx_op::TensorType::kSeqUint32)
      .value("kSeqUint64", onnx_op::TensorType::kSeqUint64)
      .value("kSeqInt8", onnx_op::TensorType::kSeqInt8)
      .value("kSeqInt16", onnx_op::TensorType::kSeqInt16)
      .value("kSeqInt32", onnx_op::TensorType::kSeqInt32)
      .value("kSeqInt64", onnx_op::TensorType::kSeqInt64)
      .value("kSeqFloat16", onnx_op::TensorType::kSeqFloat16)
      .value("kSeqFloat", onnx_op::TensorType::kSeqFloat)
      .value("kSeqDouble", onnx_op::TensorType::kSeqDouble)
      .value("kSeqComplex64", onnx_op::TensorType::kSeqComplex64)
      .value("kSeqComplex128", onnx_op::TensorType::kSeqComplex128)
      .value("kSeqMapStringFloat", onnx_op::TensorType::kSeqMapStringFloat)
      .value("kSeqMapInt64Float", onnx_op::TensorType::kSeqMapInt64Float)
      .value("kOptSeqBool", onnx_op::TensorType::kOptSeqBool)
      .value("kOptSeqString", onnx_op::TensorType::kOptSeqString)
      .value("kOptSeqUint8", onnx_op::TensorType::kOptSeqUint8)
      .value("kOptSeqUint16", onnx_op::TensorType::kOptSeqUint16)
      .value("kOptSeqUint32", onnx_op::TensorType::kOptSeqUint32)
      .value("kOptSeqUint64", onnx_op::TensorType::kOptSeqUint64)
      .value("kOptSeqInt8", onnx_op::TensorType::kOptSeqInt8)
      .value("kOptSeqInt16", onnx_op::TensorType::kOptSeqInt16)
      .value("kOptSeqInt32", onnx_op::TensorType::kOptSeqInt32)
      .value("kOptSeqInt64", onnx_op::TensorType::kOptSeqInt64)
      .value("kOptSeqFloat16", onnx_op::TensorType::kOptSeqFloat16)
      .value("kOptSeqFloat", onnx_op::TensorType::kOptSeqFloat)
      .value("kOptSeqDouble", onnx_op::TensorType::kOptSeqDouble)
      .value("kOptSeqComplex64", onnx_op::TensorType::kOptSeqComplex64)
      .value("kOptSeqComplex128", onnx_op::TensorType::kOptSeqComplex128)
      .value("kOptBool", onnx_op::TensorType::kOptBool)
      .value("kOptString", onnx_op::TensorType::kOptString)
      .value("kOptUint8", onnx_op::TensorType::kOptUint8)
      .value("kOptUint16", onnx_op::TensorType::kOptUint16)
      .value("kOptUint32", onnx_op::TensorType::kOptUint32)
      .value("kOptUint64", onnx_op::TensorType::kOptUint64)
      .value("kOptInt8", onnx_op::TensorType::kOptInt8)
      .value("kOptInt16", onnx_op::TensorType::kOptInt16)
      .value("kOptInt32", onnx_op::TensorType::kOptInt32)
      .value("kOptInt64", onnx_op::TensorType::kOptInt64)
      .value("kOptFloat16", onnx_op::TensorType::kOptFloat16)
      .value("kOptFloat", onnx_op::TensorType::kOptFloat)
      .value("kOptDouble", onnx_op::TensorType::kOptDouble)
      .value("kOptComplex64", onnx_op::TensorType::kOptComplex64)
      .value("kOptComplex128", onnx_op::TensorType::kOptComplex128)
      .value("kUndefined", onnx_op::TensorType::kUndefined);

  onnx_op_mod.def(
      "ToTypeString",
      [](onnx_op::TensorType type) { return std::string(onnx_op::ToTypeString(type)); },
      nb::arg("type"),
      "Returns the ONNX type-string representation of a TensorType value "
      "(e.g. ``\"tensor(float)\"`` or ``\"seq(tensor(int64))\"``).");

  nb::class_<onnx_op::FormalParameter>(
      onnx_op_mod, "FormalParameter",
      "A single formal input or output parameter of an ONNX operator.")
      .def(nb::init<>())
      .def_rw("name", &onnx_op::FormalParameter::name)
      .def_rw("description", &onnx_op::FormalParameter::description)
      .def_rw("type", &onnx_op::FormalParameter::type);

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
      .def_rw("description", &onnx_op::TypeConstraintParam::description);

  nb::class_<onnx_op::LightOpSchema>(
      onnx_op_mod, "LightOpSchema",
      "Lightweight read-only description of an ONNX operator schema at one specific "
      "opset version.")
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
      .def("set_min_output", &onnx_op::LightOpSchema::set_min_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the minimum number of outputs and returns ``self``.")
      .def("set_max_output", &onnx_op::LightOpSchema::set_max_output, nb::arg("value"),
           nb::rv_policy::reference_internal,
           "Sets the maximum number of outputs and returns ``self``.")
      .def("set_deprecated", &onnx_op::LightOpSchema::set_deprecated, nb::arg("value") = true,
           nb::rv_policy::reference_internal,
           "Marks this operator as deprecated and returns ``self``.");

  onnx_op_mod.def("GetAllOnnxOpSchemasWithHistory", &onnx_op::GetAllOnnxOpSchemasWithHistory,
                  nb::arg("init_doc") = true,
                  "Returns the complete versioned schema history for all supported ONNX "
                  "operator domains as a list of LightOpSchema.");
}
