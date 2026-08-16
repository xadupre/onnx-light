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
#include "onnx_extensions/patterns/collections/concat_pattern.h"
#include "onnx_extensions/patterns/collections/gather_pattern.h"
#include "onnx_extensions/patterns/collections/sequence_pattern.h"
#include "onnx_extensions/patterns/collections/shape_pattern.h"
#include "onnx_extensions/patterns/collections/slice_pattern.h"
#include "onnx_extensions/patterns/collections/split_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"

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
  if (name == "SplitConcat") {
    return std::make_shared<onnx_patterns::SplitConcatPattern>(priority.value_or(0));
  }
  if (name == "GathersSplit") {
    return std::make_shared<onnx_patterns::GathersSplitPattern>(priority.value_or(0));
  }
  if (name == "SlicesSplit") {
    return std::make_shared<onnx_patterns::SlicesSplitPattern>(priority.value_or(0));
  }
  if (name == "ConcatEmpty") {
    return std::make_shared<onnx_patterns::ConcatEmptyPattern>(priority.value_or(0));
  }
  if (name == "ConcatGather") {
    return std::make_shared<onnx_patterns::ConcatGatherPattern>(priority.value_or(0));
  }
  if (name == "ConcatTwiceUnary") {
    return std::make_shared<onnx_patterns::ConcatTwiceUnaryPattern>(priority.value_or(0));
  }
  if (name == "GatherConcat") {
    return std::make_shared<onnx_patterns::GatherConcatPattern>(priority.value_or(0));
  }
  if (name == "GatherGather") {
    return std::make_shared<onnx_patterns::GatherGatherPattern>(priority.value_or(0));
  }
  if (name == "GatherShape") {
    return std::make_shared<onnx_patterns::GatherShapePattern>(priority.value_or(0));
  }
  if (name == "SliceSlice") {
    return std::make_shared<onnx_patterns::SliceSlicePattern>(priority.value_or(0));
  }
  if (name == "SequenceConstructAt") {
    return std::make_shared<onnx_patterns::SequenceConstructAtPattern>(priority.value_or(0));
  }
  if (name == "SplitToSequenceSequenceAt") {
    return std::make_shared<onnx_patterns::SplitToSequenceSequenceAtPattern>(priority.value_or(0));
  }
  if (name == "NotWhere") {
    return std::make_shared<onnx_patterns::NotWherePattern>(priority.value_or(0));
  }
  if (name == "UnsqueezeEqual") {
    return std::make_shared<onnx_patterns::UnsqueezeEqualPattern>(priority.value_or(0));
  }
  if (name == "WhereAdd") {
    return std::make_shared<onnx_patterns::WhereAddPattern>(priority.value_or(0));
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
  nb::class_<onnx_patterns::SplitConcatPattern, core::builder::PatternOptimization>(
      m, "SplitConcatPattern",
      "Replaces a Split immediately followed by a Concat that restores the "
      "original tensor with an Identity node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::GathersSplitPattern, core::builder::PatternOptimization>(
      m, "GathersSplitPattern",
      "Replaces sibling Gather nodes selecting contiguous single indices of a "
      "shared input by a single Split node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SlicesSplitPattern, core::builder::PatternOptimization>(
      m, "SlicesSplitPattern",
      "Replaces sibling Slice nodes cutting a shared input into contiguous "
      "chunks along one axis by a single Split node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ConcatEmptyPattern, core::builder::PatternOptimization>(
      m, "ConcatEmptyPattern",
      "Drops empty inputs from a Concat node, reducing it to an Identity when a "
      "single input remains.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ConcatGatherPattern, core::builder::PatternOptimization>(
      m, "ConcatGatherPattern",
      "Rewrites a Gather reading a single Concat input into a Gather on that "
      "input directly.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ConcatTwiceUnaryPattern, core::builder::PatternOptimization>(
      m, "ConcatTwiceUnaryPattern",
      "Pushes a shape-preserving unary op ahead of a ``Concat(x, x)`` so the "
      "unary op runs once on ``x``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::GatherConcatPattern, core::builder::PatternOptimization>(
      m, "GatherConcatPattern",
      "Merges a Concat of single-index Gather nodes on a shared input into one "
      "Gather node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::GatherGatherPattern, core::builder::PatternOptimization>(
      m, "GatherGatherPattern",
      "Collapses two consecutive scalar Gather nodes into a single Gather node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::GatherShapePattern, core::builder::PatternOptimization>(
      m, "GatherShapePattern",
      "Rewrites a Gather of a scalar index over a Shape node into a narrowed "
      "Shape node.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SliceSlicePattern, core::builder::PatternOptimization>(
      m, "SliceSlicePattern", "Merges two consecutive Slice nodes on distinct axes into one Slice.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SequenceConstructAtPattern, core::builder::PatternOptimization>(
      m, "SequenceConstructAtPattern",
      "Replaces a SequenceAt reading a constant index of a SequenceConstruct by "
      "the corresponding input tensor.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SplitToSequenceSequenceAtPattern, core::builder::PatternOptimization>(
      m, "SplitToSequenceSequenceAtPattern",
      "Replaces a SequenceAt reading a constant index of a SplitToSequence by a "
      "single Split output.")
      .def(nb::init<int>(), nb::arg("priority") = 0);

  nb::class_<onnx_patterns::NotWherePattern, core::builder::PatternOptimization>(
      m, "NotWherePattern", "Rewrites ``Where(Not(c), x, y)`` into ``Where(c, y, x)``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::UnsqueezeEqualPattern, core::builder::PatternOptimization>(
      m, "UnsqueezeEqualPattern",
      "Rewrites ``Equal(Unsqueeze(x), Unsqueeze(y))`` into ``Equal(x, y)`` "
      "when both Unsqueeze nodes use matching constant axes.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::WhereAddPattern, core::builder::PatternOptimization>(
      m, "WhereAddPattern", "Factors a common additive term from Where branches built with Add.")
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
