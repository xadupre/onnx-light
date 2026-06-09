// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_tensor.h"

#include <cstdint>
#include <cstring>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <regex>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DataSet;
using onnx_backend_test::TestCase;
using onnx_kernels::Tensor;

void AddOnnxPyBackend(nb::module_ &m);
void AddOnnxPyBackendTest(nb::module_ &m);

NB_MODULE(_onnxpybackend, m) {
  m.doc() = "onnx_light backend bindings: deterministic pseudo-random helpers and "
            "ONNX backend-test case utilities.";

  AddOnnxPyBackendTest(m);
}

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
          "``data_type == TensorProto::DataType::STRING``).")
      .def("__repr__", [](const Tensor &t) {
        std::string r = "Tensor(name='";
        r += t.name;
        r += "', data_type=";
        r += TensorProto::DataType_Name(static_cast<TensorProto::DataType>(t.data_type));
        r += ", shape=[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
          if (i > 0)
            r += ", ";
          r += std::to_string(t.shape[i]);
        }
        r += "])";
        return r;
      });

  nb::class_<DataSet>(bt_mod, "DataSet",
                      "A single (inputs, expected outputs) data set of a TestCase.")
      .def_rw("inputs", &DataSet::inputs)
      .def_rw("outputs", &DataSet::outputs)
      .def("__repr__", [](const DataSet &ds) {
        return "DataSet(inputs=" + std::to_string(ds.inputs.size()) +
               ", outputs=" + std::to_string(ds.outputs.size()) + ")";
      });

  // ``ModelProto`` is exposed by the sibling ``_onnxpyprotoop`` extension.
  // Because ``lib_onnx_proto`` is a shared library linked by both extensions
  // (see CMakeLists.txt), the ``&typeid(ModelProto)`` resolved here is the
  // same as the one used by ``_onnxpyprotoop`` and nanobind's cross-module
  // type registry finds the existing ``nb::class_<ModelProto>`` binding
  // automatically. Callers must therefore have imported ``_onnxpyprotoop``
  // before accessing ``TestCase.model``; the package ``_onnxpy.py`` shim
  // guarantees that ordering.
  nb::class_<TestCase>(bt_mod, "TestCase",
                       "A single C++-generated backend test case (mirrors "
                       "onnx_light.backend.test.case.base.TestCase).")
      .def(nb::init<std::string, std::string, std::string, std::string, double, double>(),
           nb::arg("name"), nb::arg("model_name") = std::string(),
           nb::arg("kind") = std::string("node"), nb::arg("tag") = std::string(),
           nb::arg("rtol") = 1e-3, nb::arg("atol") = 1e-7)
      .def_ro("name", &TestCase::name)
      .def_ro("model_name", &TestCase::model_name)
      .def_ro("kind", &TestCase::kind)
      .def_ro("tag", &TestCase::tag)
      .def_rw("rtol", &TestCase::rtol)
      .def_rw("atol", &TestCase::atol)
      .def_rw("data_sets", &TestCase::data_sets)
      .def_prop_ro(
          "model", [](TestCase &tc) -> ModelProto & { return tc.model; },
          nb::rv_policy::reference_internal,
          "Returns the ``ModelProto`` of this test case, resolved against the "
          "binding registered by ``_onnxpyprotoop``.")
      .def("__repr__", [](const TestCase &tc) {
        return "TestCase(name='" + tc.name + "', kind='" + tc.kind + "')";
      });

  bt_mod.def(
      "collect_test_cases",
      [](const std::string &op_type) {
        return onnx_backend_test::CollectTestCases(op_type_or_cat);
      },
      nb::arg("op_type_or_cat") = std::string(),
      "Returns the list of C++-implemented backend test node cases. When "
      "``op_type_or_cat`` is non-empty, only cases whose top-level graph "
      "contains a node with that operator type are returned. It can also "
      "be 'shape', 'inference', 'nan_inf' to get other backend tests "
      "not testing a specific operator but a specific issues in one "
      "algorithm.");

  bt_mod.def(
      "collect_test_cases_by_name",
      [](const std::string &name_regex) {
        try {
          return onnx_backend_test::CollectTestCasesByName(name_regex);
        } catch (const std::regex_error &e) {
          throw nb::value_error(e.what());
        }
      },
      nb::arg("name_regex") = std::string(),
      "Returns the C++-implemented backend test node cases whose ``name`` matches "
      "the ECMAScript regular expression ``name_regex`` (``std::regex_search`` "
      "semantics: substring match by default; anchor with ``^...$`` to require a "
      "full match). An empty pattern returns every case. Raises ``ValueError`` if "
      "``name_regex`` is not a valid regular expression.");
}
