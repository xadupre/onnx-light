// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"
#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_extensions/patterns/canonicalization/not_pattern.h"
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
  if (name == "ClipClip") {
    return std::make_shared<onnx_patterns::ClipClipPattern>(priority.value_or(1));
  }
  if (name == "ConstantToInitializer") {
    return std::make_shared<onnx_patterns::ConstantToInitializerPattern>(priority.value_or(1));
  }
  if (name == "ConvBiasNull") {
    return std::make_shared<onnx_patterns::ConvBiasNullPattern>(priority.value_or(0));
  }
  if (name == "Dropout") {
    return std::make_shared<onnx_patterns::DropoutPattern>(priority.value_or(1));
  }
  if (name == "Identity") {
    return std::make_shared<onnx_patterns::IdentityPattern>(priority.value_or(0));
  }
  if (name == "NotNot") {
    return std::make_shared<onnx_patterns::NotNotPattern>(priority.value_or(1));
  }
  if (name == "PadConv") {
    return std::make_shared<onnx_patterns::PadConvPattern>(priority.value_or(0));
  }
  throw std::invalid_argument("Unknown ONNX pattern '" + name + "'.");
}

} // namespace

NB_MODULE(_onnxpypatterns, m) {
  nb::module_::import_("onnx_light.onnx_py._onnxpyprotoop");
  nb::module_::import_("onnx_light.onnx_py._onnxpycore");
  onnx_patterns::RegisterPatterns();

  m.doc() = "Concrete ONNX graph-rewriting patterns.";

  nb::class_<onnx_patterns::CastPattern, core::builder::PatternOptimization>(
      m, "CastPattern",
      "Replaces a type-preserving ``Cast(to=T)`` with ``Identity``.\n\n"
      "``x:T -> Cast(to=T) -> y:T`` becomes ``x:T -> Identity -> y:T``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::CastCastPattern, core::builder::PatternOptimization>(
      m, "CastCastPattern",
      "Collapses two consecutive compatible Cast nodes.\n\n"
      "``x:A -> Cast(B) -> Cast(C) -> y:C`` becomes one safe ``Cast(C)`` "
      "or ``Identity``.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::CastCastBinaryPattern, core::builder::PatternOptimization>(
      m, "CastCastBinaryPattern",
      "Moves matching floating-point input Cast nodes after a binary operation.\n\n"
      "``Cast(x), Cast(y) -> Binary`` becomes ``Binary(x, y) -> Cast`` when "
      "precision and use guards allow it.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::CastOpCastPattern, core::builder::PatternOptimization>(
      m, "CastOpCastPattern",
      "Moves a unary or binary operation to the result Cast type.\n\n"
      "Compatible input Cast nodes and the trailing result Cast are removed or "
      "relocated while preserving shared outputs.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::ClipClipPattern, core::builder::PatternOptimization>(
      m, "ClipClipPattern",
      "Merges two consecutive Clip nodes with complementary bounds.\n\n"
      "``Clip(x, min) -> Clip(x1, , max)`` becomes one ``Clip(x, min, max)`` "
      "when one Clip defines the minimum and the other the maximum.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::ConstantToInitializerPattern, core::builder::PatternOptimization>(
      m, "ConstantToInitializerPattern",
      "Replaces a Constant node by an initializer and an Identity node.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::ConvBiasNullPattern, core::builder::PatternOptimization>(
      m, "ConvBiasNullPattern", "Removes a null (all-zero) bias input from a Conv node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::DropoutPattern, core::builder::PatternOptimization>(
      m, "DropoutPattern",
      "Replaces an inference Dropout by an Identity node when its mask output "
      "is unused and training mode is disabled.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::IdentityPattern, core::builder::PatternOptimization>(
      m, "IdentityPattern", "Replaces no-op arithmetic and layout operations by an Identity node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::NotNotPattern, core::builder::PatternOptimization>(
      m, "NotNotPattern", "Fuses two consecutive Not nodes into an Identity node.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::PadConvPattern, core::builder::PatternOptimization>(
      m, "PadConvPattern", "Folds a Pad node into the ``pads`` attribute of a following Conv node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);

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
