// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"
#include "onnx_backend_test/simple_tensor.h"
#include "onnx_backend_test/test_case.h"

#include <cstring>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DataSet;
using onnx_backend_test::Tensor;
using onnx_backend_test::TestCase;

namespace {

// Serialize a ModelProto into a Python ``bytes`` object.
nb::bytes SerializeModel(ModelProto &model) {
  std::string buffer;
  model.SerializeToString(buffer);
  return nb::bytes(buffer.data(), buffer.size());
}

} // namespace

void AddOnnxPyBackendTest(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `backend_test`
  // Exposes the C++-implemented backend test node cases.
  // -----------------------------------------------------------------------
  auto bt_mod = m.def_submodule("backend_test");
  bt_mod.doc() = "C++-generated backend test node cases (mirrors the data model used by "
                 "onnx_light.backend.test.case).";

  nb::class_<Tensor>(bt_mod, "Tensor",
                     "A runtime tensor used by C++ backend test cases. Distinct from "
                     "TensorProto: stores raw little-endian row-major bytes.")
      .def(nb::init<>())
      .def_rw("name", &Tensor::name)
      .def_rw("data_type", &Tensor::data_type)
      .def_rw("shape", &Tensor::shape)
      .def("element_count", &Tensor::element_count)
      .def("element_size", &Tensor::element_size)
      .def(
          "raw_data", [](const Tensor &t) { return nb::bytes(t.data.data(), t.data.size()); },
          "Returns the raw element bytes as a Python ``bytes`` object.")
      .def(
          "string_data", [](const Tensor &t) { return t.string_data; },
          "Returns the string element values (only populated when "
          "``data_type == TensorProto::DataType::STRING``).");

  nb::class_<DataSet>(bt_mod, "DataSet",
                      "A single (inputs, expected outputs) data set of a TestCase.")
      .def_rw("inputs", &DataSet::inputs)
      .def_rw("outputs", &DataSet::outputs);

  nb::class_<TestCase>(bt_mod, "TestCase",
                       "A single C++-generated backend test case (mirrors "
                       "onnx_light.backend.test.case.base.TestCase).")
      .def_rw("name", &TestCase::name)
      .def_rw("model_name", &TestCase::model_name)
      .def_rw("kind", &TestCase::kind)
      .def_rw("rtol", &TestCase::rtol)
      .def_rw("atol", &TestCase::atol)
      .def_rw("data_sets", &TestCase::data_sets)
      .def(
          "model_bytes", [](TestCase &tc) { return SerializeModel(tc.model); },
          "Returns the serialized ``ModelProto`` as Python ``bytes``.");

  bt_mod.def(
      "collect_test_cases",
      [](const std::string &op_type) { return onnx_backend_test::CollectTestCases(op_type); },
      nb::arg("op_type") = std::string(),
      "Returns the list of C++-implemented backend test node cases. When ``op_type`` "
      "is non-empty, only cases whose top-level graph contains a node with that "
      "operator type are returned.");
}
