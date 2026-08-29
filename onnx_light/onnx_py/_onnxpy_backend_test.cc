// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <regex>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DataSet;
using core::backend_test::TestCase;
using core::backend_test::TestMode;
using core::runtime::Map;
using core::runtime::Tensor;

void AddOnnxPyBackend(nb::module_ &m);
void AddOnnxPyBackendTest(nb::module_ &m);

namespace {

/// Maps an ONNX ``TensorProto::DataType`` to the equivalent DLPack dtype.
///
/// Throws ``std::invalid_argument`` for data types that cannot be exchanged
/// through DLPack with a whole-byte element layout, i.e. ``STRING`` and the
/// sub-byte packed types (``INT4``/``UINT4``/``INT2``/``UINT2``/``FLOAT4E2M1``).
nb::dlpack::dtype DLPackDtypeFromOnnx(int32_t data_type) {
  using Code = nb::dlpack::dtype_code;
  auto make = [](Code code, uint8_t bits) -> nb::dlpack::dtype {
    return nb::dlpack::dtype{static_cast<uint8_t>(code), bits, 1};
  };
  switch (static_cast<TensorProto::DataType>(data_type)) {
  case TensorProto::FLOAT:
    return make(Code::Float, 32);
  case TensorProto::DOUBLE:
    return make(Code::Float, 64);
  case TensorProto::FLOAT16:
    return make(Code::Float, 16);
  case TensorProto::BFLOAT16:
    return make(Code::Bfloat, 16);
  case TensorProto::UINT8:
    return make(Code::UInt, 8);
  case TensorProto::INT8:
    return make(Code::Int, 8);
  case TensorProto::UINT16:
    return make(Code::UInt, 16);
  case TensorProto::INT16:
    return make(Code::Int, 16);
  case TensorProto::UINT32:
    return make(Code::UInt, 32);
  case TensorProto::INT32:
    return make(Code::Int, 32);
  case TensorProto::UINT64:
    return make(Code::UInt, 64);
  case TensorProto::INT64:
    return make(Code::Int, 64);
  case TensorProto::BOOL:
    return make(Code::Bool, 8);
  case TensorProto::COMPLEX64:
    return make(Code::Complex, 64);
  case TensorProto::COMPLEX128:
    return make(Code::Complex, 128);
  case TensorProto::FLOAT8E4M3FN:
    return make(Code::Float8_E4M3FN, 8);
  case TensorProto::FLOAT8E4M3FNUZ:
    return make(Code::Float8_E4M3FNUZ, 8);
  case TensorProto::FLOAT8E5M2:
    return make(Code::Float8_E5M2, 8);
  case TensorProto::FLOAT8E5M2FNUZ:
    return make(Code::Float8_E5M2FNUZ, 8);
  default:
    EXT_THROW_INVALID("Tensor.__dlpack__: data type '",
                      TensorProto::DataType_Name(static_cast<TensorProto::DataType>(data_type)),
                      "' cannot be exported through DLPack (STRING and sub-byte packed types are "
                      "not supported).");
  }
}

/// Builds a zero-copy DLPack-capable ``nb::ndarray`` view over ``owner``'s
/// element bytes. ``owner`` is the Python ``Tensor`` wrapper; passing it as the
/// array owner keeps the source tensor alive for as long as the view lives. The
/// view is read-only because the tensor's byte buffer is logically immutable
/// (and may be a borrowed/non-owning span).
nb::object MakeTensorNdarray(nb::handle owner) {
  const Tensor &t = nb::cast<const Tensor &>(owner);
  nb::dlpack::dtype dtype = DLPackDtypeFromOnnx(t.data_type);
  std::vector<size_t> shape;
  shape.reserve(t.shape.size());
  for (size_t i = 0; i < t.shape.size(); ++i) {
    int64_t dim = t.shape[i];
    EXT_ENFORCE_INVALID(dim >= 0, "Tensor.__dlpack__: shape dimension at index ", i,
                        " must be non-negative, got ", dim);
    // Compare in a fixed unsigned width so 32-bit ``size_t`` platforms reject
    // large ``int64_t`` dimensions before any narrowing conversion occurs.
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max(),
                        "Tensor.__dlpack__: shape dimension at index ", i,
                        " does not fit in the platform size type, got ", dim);
    shape.push_back(static_cast<size_t>(dim));
  }
  nb::ndarray<nb::ro> array(t.bytes(), shape.size(), shape.data(), owner,
                            /*strides=*/nullptr, dtype, nb::device::cpu::value, /*device_id=*/0);
  return nb::cast(std::move(array));
}

} // namespace

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
      .def_prop_rw(
          "shape", [](const Tensor &t) -> std::vector<int64_t> { return t.shape; },
          [](Tensor &t, std::vector<int64_t> s) { t.shape = std::move(s); },
          "Tensor shape as a list of ``int64`` dimension values.")
      .def("element_count", &Tensor::element_count)
      .def("element_size", &Tensor::element_size)
      .def(
          "raw_data", [](const Tensor &t) { return nb::bytes(t.bytes(), t.size_bytes()); },
          "Returns the raw element bytes as a Python ``bytes`` object.")
      .def(
          "string_data", [](const Tensor &t) { return t.string_data; },
          "Returns the string element values (only populated when "
          "``data_type == TensorProto::DataType::STRING``).")
      .def(
          "__dlpack__",
          [](nb::handle self, nb::kwargs /*kwargs*/) {
            // Export a zero-copy DLPack capsule so consumers such as NumPy or
            // PyTorch can adopt the tensor's buffer via ``from_dlpack``. The
            // nanobind ndarray view (built with the framework-agnostic export
            // path) yields the capsule directly and keeps the source ``Tensor``
            // alive through its owner handle. The standard keyword arguments
            // (``stream``/``max_version``/``dl_device``/``copy``) are accepted
            // for protocol compatibility; the buffer is always returned as-is
            // on the CPU device.
            return MakeTensorNdarray(self);
          },
          "Returns a DLPack capsule sharing the tensor's element buffer (zero-copy). "
          "Implements the ``__dlpack__`` exchange protocol. Raises ``ValueError`` for "
          "``STRING`` and sub-byte packed data types.")
      .def(
          "__dlpack_device__",
          [](const Tensor &t) {
            // Validate the data type so unsupported tensors fail consistently
            // with ``__dlpack__``; the buffer always lives on the CPU device.
            DLPackDtypeFromOnnx(t.data_type);
            return std::make_pair(static_cast<int>(nb::device::cpu::value), 0);
          },
          "Returns the ``(device_type, device_id)`` pair for the DLPack protocol. "
          "The tensor always resides on the CPU device (``kDLCPU``).")
      .def(
          "has_borrowed_data", [](const Tensor &t) { return t.is_borrowed(); },
          "Returns whether the tensor is a non-owning (borrowed) view over external "
          "memory (for example a zero-copy view into a ``TensorProto``'s ``raw_data``) "
          "rather than owning its bytes inline. Borrowed tensors keep no ownership of "
          "their backing storage, which must outlive the tensor.")
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

  nb::class_<Map>(bt_mod, "Map", "A map-typed value: parallel keys + values tensors.")
      .def_rw("name", &Map::name)
      .def_rw("key_type", &Map::key_type)
      .def_rw("value_type", &Map::value_type)
      .def_rw("keys", &Map::keys)
      .def_rw("values", &Map::values)
      .def("__repr__", [](const Map &m) {
        return "Map(name='" + m.name + "', size=" + std::to_string(m.size()) + ")";
      });

  nb::class_<DataSet>(bt_mod, "DataSet",
                      "A single (inputs, expected outputs) data set of a TestCase.")
      .def_prop_rw(
          "inputs", [](DataSet &ds) -> std::vector<Tensor> & { return ds.inputs; },
          [](DataSet &ds, std::vector<Tensor> v) { ds.inputs = std::move(v); },
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "outputs", [](DataSet &ds) -> std::vector<Tensor> & { return ds.outputs; },
          [](DataSet &ds, std::vector<Tensor> v) { ds.outputs = std::move(v); },
          nb::rv_policy::reference_internal)
      .def_ro("expected_outputs_generated", &DataSet::expected_outputs_generated,
              "Whether expected outputs were generated for this data set.")
      .def_rw("maps", &DataSet::maps)
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
           nb::arg("atol") = 1e-7, nb::arg("rtol") = 1e-3)
      .def_ro("name", &TestCase::name)
      .def_ro("model_name", &TestCase::model_name)
      .def_ro("kind", &TestCase::kind)
      .def_ro("tag", &TestCase::tag)
      .def_rw("rtol", &TestCase::rtol)
      .def_rw("atol", &TestCase::atol)
      .def_prop_ro("has_expected_outputs", &TestCase::has_expected_outputs,
                   "Returns whether this case has generated expected outputs.")
      .def_prop_ro("data_sets", &TestCase::data_set_handles,
                   "Returns the input/output data sets of this test case, materializing "
                   "them first for lazily-built cases.")
      .def_prop_ro("model", &TestCase::model_handle,
                   "Returns the ``ModelProto`` of this test case, resolved against the "
                   "binding registered by ``_onnxpyprotoop``. Built on demand for "
                   "lazily-built (benchmark) cases.")
      .def_prop_ro("materialized", &TestCase::materialized,
                   "Returns whether this case currently retains a materialized payload.")
      .def("unload", &TestCase::unload,
           "Releases this collected case's cached model and data sets. The payload is "
           "rebuilt on the next ``model`` or ``data_sets`` access, while existing Python "
           "references remain valid.")
      .def("__repr__", [](const TestCase &tc) {
        return "TestCase(name='" + tc.name + "', kind='" + tc.kind + "')";
      });

  nb::enum_<TestMode>(bt_mod, "TestMode",
                      "Selects how backend test cases are generated. ``TEST`` (the "
                      "default) yields the standard correctness cases. ``BENCHMARK`` "
                      "yields large-input cases sized so a single kernel evaluation "
                      "runs long enough to be timed (~0.1 s), for categories that "
                      "support it.")
      .value("TEST", TestMode::TEST)
      .value("BENCHMARK", TestMode::BENCHMARK);

  bt_mod.def(
      "collect_test_cases",
      [](const std::string &op_type_or_cat, bool include_big, TestMode mode,
         bool generate_benchmark_expected_outputs) {
        return core::backend_test::CollectTestCases(op_type_or_cat, include_big, mode,
                                                    generate_benchmark_expected_outputs);
      },
      nb::arg("op_type_or_cat") = std::string(), nb::arg("include_big") = false,
      nb::arg("mode") = TestMode::TEST, nb::arg("generate_benchmark_expected_outputs") = false,
      "Returns the list of C++-implemented backend test node cases. When "
      "``op_type_or_cat`` is non-empty, only cases whose top-level graph "
      "contains a node with that operator type are returned. It can also "
      "be 'shape', 'inference', 'nan_inf' to get other backend tests "
      "not testing a specific operator but specific issues in one "
      "algorithm. Test cases whose name contains ``'_big_'`` are excluded "
      "by default; pass ``include_big=True`` to include them. Pass "
      "``mode=TestMode.BENCHMARK`` to emit large benchmark-sized cases "
      "instead of the standard correctness cases where supported.");

  bt_mod.def(
      "collect_test_cases_by_name",
      [](const std::string &name_regex, bool include_big, TestMode mode,
         bool generate_benchmark_expected_outputs) {
        try {
          return core::backend_test::CollectTestCasesByName(name_regex, include_big, mode,
                                                            generate_benchmark_expected_outputs);
        } catch (const std::regex_error &e) {
          throw nb::value_error(e.what());
        }
      },
      nb::arg("name_regex") = std::string(), nb::arg("include_big") = false,
      nb::arg("mode") = TestMode::TEST, nb::arg("generate_benchmark_expected_outputs") = false,
      "Returns the C++-implemented backend test node cases whose ``name`` matches "
      "the ECMAScript regular expression ``name_regex`` (``std::regex_search`` "
      "semantics: substring match by default; anchor with ``^...$`` to require a "
      "full match). An empty pattern returns every case. Test cases whose name "
      "contains ``'_big_'`` are excluded by default; pass ``include_big=True`` "
      "to include them. Pass ``mode=TestMode.BENCHMARK`` to emit large "
      "benchmark-sized cases instead of the standard correctness cases where "
      "supported. Raises ``ValueError`` if ``name_regex`` is not a valid "
      "regular expression.");

  bt_mod.def(
      "get_test_case_by_name",
      [](const std::string &name, bool include_big, TestMode mode,
         bool generate_benchmark_expected_outputs) {
        return core::backend_test::GetTestCaseByName(name, include_big, mode,
                                                     generate_benchmark_expected_outputs);
      },
      nb::arg("name"), nb::arg("include_big") = false, nb::arg("mode") = TestMode::TEST,
      nb::arg("generate_benchmark_expected_outputs") = false,
      "Returns the single C++-implemented backend test case whose ``name`` "
      "matches exactly, as a list of at most one element. An empty list signals "
      "that no case with the requested name was found. More efficient than "
      "``collect_test_cases_by_name`` for exact-name lookups because it avoids "
      "regex compilation and stops at the first match.");
}
