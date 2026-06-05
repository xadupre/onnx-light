// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/random.h"
#include "onnx_backend_test/simple_tensor.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DataSet;
using onnx_backend_test::Tensor;
using onnx_backend_test::TestCase;

void AddOnnxPyBackend(nb::module_ &m);
void AddOnnxPyBackendTest(nb::module_ &m);

NB_MODULE(_onnxbackend, m) {
  m.doc() = "onnx_light backend bindings: deterministic pseudo-random helpers and "
            "ONNX backend-test case utilities.";

  AddOnnxPyBackend(m);
  AddOnnxPyBackendTest(m);
}

void AddOnnxPyBackend(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `backend`
  // Deterministic pseudo-random helpers backing ``onnx_light.backend``.
  // -----------------------------------------------------------------------
  auto backend_mod = m.def_submodule("backend");
  backend_mod.doc() =
      "Deterministic pseudo-random helpers (SplitMix64) used by onnx_light.backend.";

  backend_mod.def(
      "next_uint64", [](uint64_t state) { return onnx_backend_test::NextUint64(state); },
      nb::arg("state"), "Returns ``(next_state, value)`` for the SplitMix64 generator.");

  backend_mod.def(
      "rand",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_backend_test::Rand(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic uniform values in ``[0, 1)`` as a flat list.");

  backend_mod.def(
      "randint",
      [](int64_t low, int64_t high, const std::vector<int64_t> &shape,
         std::optional<uint64_t> seed) {
        return onnx_backend_test::RandInt(low, high, shape, seed);
      },
      nb::arg("low"), nb::arg("high"), nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic integers in ``[low, high)`` as a flat list.");

  backend_mod.def(
      "randn",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_backend_test::Randn(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` approximately normal-distributed values (Irwin-Hall) "
      "as a flat list.");
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
          "``data_type == TensorProto::DataType::STRING``).");

  nb::class_<DataSet>(bt_mod, "DataSet",
                      "A single (inputs, expected outputs) data set of a TestCase.")
      .def_rw("inputs", &DataSet::inputs)
      .def_rw("outputs", &DataSet::outputs);

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
      [](const std::string &op_type) { return onnx_backend_test::CollectTestCases(op_type); },
      nb::arg("op_type") = std::string(),
      "Returns the list of C++-implemented backend test node cases. When ``op_type`` "
      "is non-empty, only cases whose top-level graph contains a node with that "
      "operator type are returned.");
}
