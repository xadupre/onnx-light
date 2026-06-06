// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <cstring>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <regex>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DataSet;
using onnx_kernels::Tensor;
using onnx_kernels::TestCase;

void AddOnnxPyKernels(nb::module_ &m);

NB_MODULE(_onnxbackend, m) {
  m.doc() = "onnx_light kernels bindings.";

  AddOnnxPyKernels(m);
}

void AddOnnxPyKernels(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `backend`
  // Deterministic pseudo-random helpers backing ``onnx_light.backend``.
  // -----------------------------------------------------------------------
  auto backend_mod = m.def_submodule("backend");
  backend_mod.doc() =
      "Deterministic pseudo-random helpers (SplitMix64) used by onnx_light.backend.";

  backend_mod.def(
      "next_uint64", [](uint64_t state) { return onnx_kernels::NextUint64(state); },
      nb::arg("state"), "Returns ``(next_state, value)`` for the SplitMix64 generator.");

  backend_mod.def(
      "rand",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_kernels::Rand(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic uniform values in ``[0, 1)`` as a flat list.");

  backend_mod.def(
      "randint",
      [](int64_t low, int64_t high, const std::vector<int64_t> &shape,
         std::optional<uint64_t> seed) { return onnx_kernels::RandInt(low, high, shape, seed); },
      nb::arg("low"), nb::arg("high"), nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic integers in ``[low, high)`` as a flat list.");

  backend_mod.def(
      "randn",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_kernels::Randn(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` approximately normal-distributed values (Irwin-Hall) "
      "as a flat list.");
}
