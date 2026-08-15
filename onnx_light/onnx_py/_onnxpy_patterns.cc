// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"

#include <memory>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {

std::shared_ptr<core::builder::PatternOptimization> CreatePattern(const std::string &name,
                                                                  std::optional<int> priority) {
  if (name == "Cast") {
    return std::make_shared<onnx_patterns::CastPattern>(priority.value_or(0));
  }
  if (name == "CastCast") {
    return std::make_shared<onnx_patterns::CastCastPattern>(priority.value_or(1));
  }
  if (name == "CastCastBinary") {
    return std::make_shared<onnx_patterns::CastCastBinaryPattern>(priority.value_or(1));
  }
  if (name == "CastOpCast") {
    return std::make_shared<onnx_patterns::CastOpCastPattern>(priority.value_or(1));
  }
  throw std::invalid_argument("Unknown ONNX pattern '" + name + "'.");
}

} // namespace

NB_MODULE(_onnxpypatterns, m) {
  nb::module_::import_("onnx_light.onnx_py._onnxpyprotoop");
  nb::module_::import_("onnx_light.onnx_py._onnxpycore");

  m.doc() = "Concrete ONNX graph-rewriting patterns.";

  nb::class_<onnx_patterns::CastPattern, core::builder::PatternOptimization>(m, "CastPattern")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::CastCastPattern, core::builder::PatternOptimization>(m,
                                                                                 "CastCastPattern")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::CastCastBinaryPattern, core::builder::PatternOptimization>(
      m, "CastCastBinaryPattern")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::CastOpCastPattern, core::builder::PatternOptimization>(
      m, "CastOpCastPattern")
      .def(nb::init<int>(), nb::arg("priority") = 1);

  m.def(
      "registered_pattern_names",
      []() {
        onnx_patterns::RegisterPatterns();
        return core::builder::RegisteredPatternNames();
      },
      "Returns the registered standard ONNX pattern names.");
  m.def("create_pattern", &CreatePattern, nb::arg("name"), nb::arg("priority") = nb::none(),
        "Creates a standard ONNX pattern by its registered name.");
}
