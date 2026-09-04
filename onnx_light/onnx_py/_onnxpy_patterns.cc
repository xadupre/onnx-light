// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/algebra/common_pattern.h"
#include "onnx_extensions/patterns/algebra/mul_pattern.h"
#include "onnx_extensions/patterns/algebra/range_pattern.h"
#include "onnx_extensions/patterns/algebra/reduce_pattern.h"
#include "onnx_extensions/patterns/algebra/shape_pattern.h"
#include "onnx_extensions/patterns/algebra/sub_pattern.h"
#include "onnx_extensions/patterns/attention/attention_pattern.h"
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
#include "onnx_extensions/patterns/expand/expand_pattern.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"
#include "onnx_extensions/patterns/layout/layout_pattern.h"
#include "onnx_extensions/patterns/matmul/matmul_pattern.h"
#include "onnx_extensions/patterns/normalization/activation_pattern.h"
#include "onnx_extensions/patterns/normalization/normalization_pattern.h"
#include "onnx_extensions/patterns/reshape/reshape_pattern.h"
#include "onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h"
#include "onnx_extensions/patterns/transpose/transpose_pattern.h"
#include "onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h"

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

template <typename T> void BindPattern(nb::module_ &m, const char *name, const char *doc) {
  nb::class_<T, core::builder::PatternOptimization>(m, name, doc)
      .def(nb::init<int>(), nb::arg("priority") = 0);
}

std::shared_ptr<core::builder::PatternOptimization> CreatePattern(const std::string &name,
                                                                  std::optional<int> priority) {
  std::unique_ptr<core::builder::PatternOptimization> pattern =
      onnx_patterns::CreatePattern(name, priority);
  return std::shared_ptr<core::builder::PatternOptimization>(std::move(pattern));
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

  nb::class_<onnx_patterns::ExpandPattern, core::builder::PatternOptimization>(
      m, "ExpandPattern",
      "Replaces ``Expand(x, shape)`` with ``Identity(x)`` when the target shape "
      "equals the input shape.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ExpandBroadcastPattern, core::builder::PatternOptimization>(
      m, "ExpandBroadcastPattern",
      "Drops an ``Expand`` feeding an element-wise binary operator that already "
      "broadcasts the pre-expanded input.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedConcatExpandPattern, core::builder::PatternOptimization>(
      m, "ShapeBasedConcatExpandPattern",
      "Simplifies a dynamic ``Concat`` target when ``Expand`` changes one dimension.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedExpandBroadcastPattern, core::builder::PatternOptimization>(
      m, "ShapeBasedExpandBroadcastPattern",
      "Removes dynamic ``Expand`` nodes before a broadcasting binary operator.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedExpandBroadcastMatMulPattern,
             core::builder::PatternOptimization>(
      m, "ShapeBasedExpandBroadcastMatMulPattern",
      "Removes dynamic ``Expand`` nodes from the batch dimensions of ``MatMul``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedStaticExpandPattern, core::builder::PatternOptimization>(
      m, "ShapeBasedStaticExpandPattern",
      "Replaces a dynamic ``Expand`` target with an equivalent constant target.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedExpandSwapPattern, core::builder::PatternOptimization>(
      m, "ShapeBasedExpandSwapPattern",
      "Moves input ``Expand`` nodes after a broadcasting binary operator.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeBasedExpandCastWhereSwapPattern,
             core::builder::PatternOptimization>(
      m, "ShapeBasedExpandCastWhereSwapPattern",
      "Moves an ``Expand`` after a compatible ``Cast`` and ``Where`` chain.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ExpandSwapPattern, core::builder::PatternOptimization>(
      m, "ExpandSwapPattern",
      "Moves an ``Expand`` past a following unary-like operator so the operator "
      "runs on the smaller tensor.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SwapExpandUnsqueezePattern, core::builder::PatternOptimization>(
      m, "SwapExpandUnsqueezePattern",
      "Swaps ``Expand`` and a following ``Unsqueeze`` so the ``Unsqueeze`` runs "
      "on the smaller, pre-expansion tensor.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ExpandUnsqueezeExpandPattern, core::builder::PatternOptimization>(
      m, "ExpandUnsqueezeExpandPattern",
      "Fuses ``Expand``, ``Unsqueeze`` and ``Expand`` into a single ``Unsqueeze`` "
      "followed by one ``Expand``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  BindPattern<onnx_patterns::ConcatReshapePattern>(
      m, "ConcatReshapePattern", "Simplifies concatenations that construct reshape shapes.");
  BindPattern<onnx_patterns::ReshapePattern>(m, "ReshapePattern",
                                             "Removes or simplifies redundant reshape operations.");
  BindPattern<onnx_patterns::ReduceReshapePattern>(
      m, "ReduceReshapePattern", "Simplifies reshape operations around reductions.");
  BindPattern<onnx_patterns::Reshape2Of3Pattern>(
      m, "Reshape2Of3Pattern", "Simplifies two compatible reshapes among three branches.");
  BindPattern<onnx_patterns::ReshapeReshapeBinaryPattern>(
      m, "ReshapeReshapeBinaryPattern", "Moves compatible reshapes across binary operations.");
  BindPattern<onnx_patterns::ReshapeReshapePattern>(m, "ReshapeReshapePattern",
                                                    "Collapses consecutive compatible reshapes.");
  BindPattern<onnx_patterns::ReshapeSqueezePattern>(m, "ReshapeSqueezePattern",
                                                    "Simplifies a reshape followed by squeeze.");
  BindPattern<onnx_patterns::ShapeBasedEditDistanceReshapePattern>(
      m, "ShapeBasedEditDistanceReshapePattern",
      "Rewrites reshapes according to the distance between known shapes.");
  BindPattern<onnx_patterns::ShapeBasedReshapeIsSqueezePattern>(
      m, "ShapeBasedReshapeIsSqueezePattern", "Replaces eligible reshapes with squeeze.");
  BindPattern<onnx_patterns::ShapedBasedReshapePattern>(
      m, "ShapedBasedReshapePattern",
      "Simplifies reshapes using inferred input and output shapes.");
  BindPattern<onnx_patterns::StaticConcatReshapePattern>(
      m, "StaticConcatReshapePattern", "Folds static concatenated reshape shapes.");
  BindPattern<onnx_patterns::UnsqueezeOrSqueezeReshapePattern>(
      m, "UnsqueezeOrSqueezeReshapePattern",
      "Simplifies reshape operations adjacent to squeeze or unsqueeze.");
  BindPattern<onnx_patterns::UnsqueezeReshapePattern>(
      m, "UnsqueezeReshapePattern", "Simplifies an unsqueeze followed by reshape.");
  BindPattern<onnx_patterns::MulUnsqueezeUnsqueezePattern>(
      m, "MulUnsqueezeUnsqueezePattern", "Simplifies multiplication of unsqueezed inputs.");
  BindPattern<onnx_patterns::SqueezeAddPattern>(
      m, "SqueezeAddPattern", "Moves compatible squeeze operations across addition.");
  BindPattern<onnx_patterns::SqueezeBinaryUnsqueezePattern>(
      m, "SqueezeBinaryUnsqueezePattern",
      "Simplifies squeeze, binary operation, and unsqueeze sequences.");
  BindPattern<onnx_patterns::SwapUnsqueezeTransposePattern>(
      m, "SwapUnsqueezeTransposePattern", "Swaps compatible unsqueeze and transpose operations.");
  BindPattern<onnx_patterns::TransposeEqualReshapePattern>(
      m, "TransposeEqualReshapePattern", "Replaces shape-equivalent transposes with reshapes.");
  BindPattern<onnx_patterns::TransposeReshapeTransposePattern>(
      m, "TransposeReshapeTransposePattern",
      "Simplifies transpose, reshape, and transpose sequences.");
  BindPattern<onnx_patterns::DivMulPattern>(
      m, "DivMulPattern", "Fuses multiplication by a reciprocal into one division.");
  BindPattern<onnx_patterns::MulMulMulScalarPattern>(
      m, "MulMulMulScalarPattern", "Combines scalar factors in multiplication chains.");
  BindPattern<onnx_patterns::SwitchOrderBinaryPattern>(
      m, "SwitchOrderBinaryPattern", "Reorders compatible consecutive binary operations.");
  BindPattern<onnx_patterns::SwapRangeAddScalarPattern>(
      m, "SwapRangeAddScalarPattern", "Moves scalar addition into compatible range operations.");
  BindPattern<onnx_patterns::ReduceArgTopKPattern>(
      m, "ReduceArgTopKPattern", "Simplifies compatible reduction, arg, and TopK operations.");
  BindPattern<onnx_patterns::ReduceSumNormalizePattern>(
      m, "ReduceSumNormalizePattern", "Simplifies reduce-sum normalization subgraphs.");
  BindPattern<onnx_patterns::Sub1MulPattern>(
      m, "Sub1MulPattern", "Simplifies multiplication involving one minus a value.");
  BindPattern<onnx_patterns::SwapUnaryPattern>(m, "SwapUnaryPattern",
                                               "Swaps compatible unary operations.");
  BindPattern<onnx_patterns::SameChildrenPattern>(m, "SameChildrenPattern",
                                                  "Eliminates equivalent child computations.");
  BindPattern<onnx_patterns::SameChildrenFromInputPattern>(
      m, "SameChildrenFromInputPattern", "Eliminates equivalent computations from one input.");
  BindPattern<onnx_patterns::ShapeBasedIdentityPattern>(
      m, "ShapeBasedIdentityPattern", "Eliminates shape-proven identity operations.");
  BindPattern<onnx_patterns::ShapeBasedSameChildrenPattern>(
      m, "ShapeBasedSameChildrenPattern", "Eliminates shape-equivalent child computations.");
  BindPattern<onnx_patterns::ShapeBasedShapeShapeAddPattern>(
      m, "ShapeBasedShapeShapeAddPattern",
      "Exposes the upstream placeholder for additions of two Shape outputs.");
  BindPattern<onnx_patterns::GemmTransposePattern>(m, "GemmTransposePattern",
                                                   "Folds input transposes into a Gemm operation.");
  BindPattern<onnx_patterns::GemmSumFusionPattern>(
      m, "GemmSumFusionPattern", "Fuses a two-input Sum into an unbiased Gemm bias input.");
  nb::class_<onnx_patterns::MatMulAddPattern, core::builder::PatternOptimization>(
      m, "MatMulAddPattern", "Replaces a compatible MatMul and Add with Gemm.")
      .def(nb::init<int, bool>(), nb::arg("priority") = 3, nb::arg("allow_reshape") = false);
  BindPattern<onnx_patterns::MatMulBatchNormalizationFusionPattern>(
      m, "MatMulBatchNormalizationFusionPattern",
      "Folds constant inference BatchNormalization parameters into a rank-two MatMul.");
  BindPattern<onnx_patterns::MatMulReshape2Of3Pattern>(
      m, "MatMulReshape2Of3Pattern", "Simplifies compatible reshapes around MatMul.");
  BindPattern<onnx_patterns::MatMulScaleFusionPattern>(
      m, "MatMulScaleFusionPattern",
      "Absorbs one safe scalar Mul or Div adjacent to a rank-two MatMul.");
  BindPattern<onnx_patterns::MulMulMatMulPattern>(
      m, "MulMulMatMulPattern", "Moves compatible scalar multiplications across MatMul.");
  BindPattern<onnx_patterns::ReshapeMatMulReshapePattern>(
      m, "ReshapeMatMulReshapePattern", "Simplifies reshape, MatMul, and reshape sequences.");
  BindPattern<onnx_patterns::ShapeBasedMatMulToMulPattern>(
      m, "ShapeBasedMatMulToMulPattern", "Replaces shape-proven scalar MatMul with Mul.");
  BindPattern<onnx_patterns::SwitchReshapeActivationPattern>(
      m, "SwitchReshapeActivationPattern", "Moves compatible activations before Reshape.");
  BindPattern<onnx_patterns::TransposeMatMulPattern>(m, "TransposeMatMulPattern",
                                                     "Folds compatible transposes into MatMul.");
  BindPattern<onnx_patterns::TransposeReshapeMatMulPattern>(
      m, "TransposeReshapeMatMulPattern", "Simplifies transpose and reshape inputs to MatMul.");
  BindPattern<onnx_patterns::BatchNormalizationPattern>(
      m, "BatchNormalizationPattern", "Fuses an inference batch-normalization subgraph.");
  BindPattern<onnx_patterns::BatchNormalizationTrainingPattern>(
      m, "BatchNormalizationTrainingPattern", "Fuses a training batch-normalization subgraph.");
  BindPattern<onnx_patterns::CastLayerNormalizationCastPattern>(
      m, "CastLayerNormalizationCastPattern",
      "Removes redundant casts surrounding LayerNormalization.");
  BindPattern<onnx_patterns::LayerNormalizationPattern>(m, "LayerNormalizationPattern",
                                                        "Fuses a layer-normalization subgraph.");
  BindPattern<onnx_patterns::LayerNormalizationScalePattern>(
      m, "LayerNormalizationScalePattern", "Fuses layer normalization with its scale.");
  BindPattern<onnx_patterns::RMSNormalizationPattern>(m, "RMSNormalizationPattern",
                                                      "Fuses an RMS-normalization subgraph.");
  BindPattern<onnx_patterns::RMSNormalizationMulPattern>(
      m, "RMSNormalizationMulPattern", "Fuses RMS normalization with a following scale.");
  BindPattern<onnx_patterns::GeluPattern>(m, "GeluPattern", "Fuses a GELU activation subgraph.");
  BindPattern<onnx_patterns::LeakyReluPattern>(m, "LeakyReluPattern",
                                               "Fuses a LeakyRelu activation subgraph.");
  BindPattern<onnx_patterns::MaxReluPattern>(m, "MaxReluPattern",
                                             "Replaces a compatible maximum with Relu.");
  BindPattern<onnx_patterns::SoftmaxCrossEntropyLossCastPattern>(
      m, "SoftmaxCrossEntropyLossCastPattern",
      "Moves a compatible label cast into SoftmaxCrossEntropyLoss.");
  BindPattern<onnx_patterns::RotaryEmbeddingPattern>(m, "RotaryEmbeddingPattern",
                                                     "Fuses a complete rotary-embedding subgraph.");
  BindPattern<onnx_patterns::RotaryConcatPartPattern>(
      m, "RotaryConcatPartPattern", "Simplifies padded rotary concatenation subgraphs.");
  BindPattern<onnx_patterns::FunctionCausalMaskPattern>(
      m, "FunctionCausalMaskPattern", "Replaces a causal-mask subgraph with a local function.");
  BindPattern<onnx_patterns::FunctionCausalMaskMulAddPattern>(
      m, "FunctionCausalMaskMulAddPattern",
      "Fuses scaling and offset operations into a causal-mask function.");
  BindPattern<onnx_patterns::FunctionCosSinCachePattern>(
      m, "FunctionCosSinCachePattern",
      "Replaces cosine and sine cache construction with a local function.");
  BindPattern<onnx_patterns::FunctionHalfRotaryEmbeddingPattern>(
      m, "FunctionHalfRotaryEmbeddingPattern",
      "Replaces half-rotary embedding construction with a local function.");
  BindPattern<onnx_patterns::FunctionAttentionPattern>(
      m, "FunctionAttentionPattern", "Replaces a scaled dot-product attention subgraph.");
  BindPattern<onnx_patterns::LinearAttentionPattern>(
      m, "LinearAttentionPattern", "Fuses a single-token linear-attention recurrence.");
  BindPattern<onnx_patterns::FunctionAttentionGQAPattern>(
      m, "FunctionAttentionGQAPattern",
      "Replaces grouped-query attention expressed with local functions.");
  BindPattern<onnx_patterns::AttentionGQAPattern>(m, "AttentionGQAPattern",
                                                  "Fuses grouped-query attention cache handling.");
  BindPattern<onnx_patterns::SwapExpandReshapePattern>(
      m, "SwapExpandReshapePattern",
      "Swaps a supported ``Expand`` and constant-shape ``Reshape`` pair.");
  nb::class_<onnx_patterns::TreeEnsemblePattern, core::builder::PatternOptimization>(
      m, "TreeEnsemblePattern",
      "Replaces a classic tree ensemble with the unified ``TreeEnsemble`` operator.")
      .def(nb::init<int>(), nb::arg("priority") = 1);
  nb::class_<onnx_patterns::TransposeTransposePattern, core::builder::PatternOptimization>(
      m, "TransposeTransposePattern",
      "Merges two consecutive ``Transpose`` nodes into a single ``Transpose`` "
      "or an ``Identity`` when the permutations cancel out.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::TransposeGatherPattern, core::builder::PatternOptimization>(
      m, "TransposeGatherPattern",
      "Removes or reorders a ``Transpose`` feeding a ``Gather`` with a scalar "
      "index.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::UnsqueezeUnsqueezePattern, core::builder::PatternOptimization>(
      m, "UnsqueezeUnsqueezePattern",
      "Merges two consecutive ``Unsqueeze`` nodes into a single ``Unsqueeze``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::SqueezeUnsqueezePattern, core::builder::PatternOptimization>(
      m, "SqueezeUnsqueezePattern",
      "Simplifies a ``Squeeze``/``Unsqueeze`` pair into an ``Identity`` or a "
      "single ``Squeeze``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::ShapeTransposePattern, core::builder::PatternOptimization>(
      m, "ShapeTransposePattern",
      "Rewrites ``Shape(Transpose(X, perm))`` into ``Gather(Shape(X), perm)``.")
      .def(nb::init<int>(), nb::arg("priority") = 0);
  nb::class_<onnx_patterns::UnsqueezeShapePattern, core::builder::PatternOptimization>(
      m, "UnsqueezeShapePattern",
      "Rewrites ``Shape(Unsqueeze(X, axes))`` into a ``Concat`` of ranged "
      "``Shape`` slices interleaved with constant ``[1]`` tensors.")
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
