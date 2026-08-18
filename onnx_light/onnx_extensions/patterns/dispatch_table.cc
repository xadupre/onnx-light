// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/dispatch_table.h"

#include <memory>

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/algebra/common_pattern.h"
#include "onnx_extensions/patterns/algebra/mul_pattern.h"
#include "onnx_extensions/patterns/algebra/range_pattern.h"
#include "onnx_extensions/patterns/algebra/reduce_pattern.h"
#include "onnx_extensions/patterns/algebra/shape_pattern.h"
#include "onnx_extensions/patterns/algebra/sub_pattern.h"
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
#include "onnx_extensions/patterns/expand/expand_pattern.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"
#include "onnx_extensions/patterns/layout/layout_pattern.h"
#include "onnx_extensions/patterns/reshape/reshape_pattern.h"
#include "onnx_extensions/patterns/transpose/transpose_pattern.h"
#include "onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

void RegisterPatterns() {
  static const bool registered = []() {
    core::builder::RegisterPattern("Cast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastPattern>();
                                   });
    core::builder::RegisterPattern("CastCast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastCastPattern>();
                                   });
    core::builder::RegisterPattern("CastCastBinary",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastCastBinaryPattern>();
                                   });
    core::builder::RegisterPattern("CastOpCast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastOpCastPattern>();
                                   });
    core::builder::RegisterPattern("ClipClip",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ClipClipPattern>();
                                   });
    core::builder::RegisterPattern("ConstantToInitializer",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ConstantToInitializerPattern>();
                                   });
    core::builder::RegisterPattern("ConvBiasNull",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ConvBiasNullPattern>();
                                   });
    core::builder::RegisterPattern("Dropout",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<DropoutPattern>();
                                   });
    core::builder::RegisterPattern("Identity",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<IdentityPattern>();
                                   });
    core::builder::RegisterPattern("NotNot",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<NotNotPattern>();
                                   });
    core::builder::RegisterPattern("PadConv",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<PadConvPattern>();
                                   });
    core::builder::RegisterPattern("ConcatEmpty",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ConcatEmptyPattern>();
                                   });
    core::builder::RegisterPattern("ConcatGather",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ConcatGatherPattern>();
                                   });
    core::builder::RegisterPattern("ConcatTwiceUnary",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ConcatTwiceUnaryPattern>();
                                   });
    core::builder::RegisterPattern("GatherConcat",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GatherConcatPattern>();
                                   });
    core::builder::RegisterPattern("GatherGather",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GatherGatherPattern>();
                                   });
    core::builder::RegisterPattern("GatherShape",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GatherShapePattern>();
                                   });
    core::builder::RegisterPattern("GathersSplit",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GathersSplitPattern>();
                                   });
    core::builder::RegisterPattern("SliceSlice",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SliceSlicePattern>();
                                   });
    core::builder::RegisterPattern("SlicesSplit",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SlicesSplitPattern>();
                                   });
    core::builder::RegisterPattern("SplitConcat",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SplitConcatPattern>();
                                   });
    core::builder::RegisterPattern("SequenceConstructAt",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SequenceConstructAtPattern>();
                                   });
    core::builder::RegisterPattern("SplitToSequenceSequenceAt",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SplitToSequenceSequenceAtPattern>();
                                   });
    core::builder::RegisterPattern("NotWhere",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<NotWherePattern>();
                                   });
    core::builder::RegisterPattern("UnsqueezeEqual",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<UnsqueezeEqualPattern>();
                                   });
    core::builder::RegisterPattern("WhereAdd",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<WhereAddPattern>();
                                   });
    core::builder::RegisterPattern("Expand",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ExpandPattern>();
                                   });
    core::builder::RegisterPattern("ExpandBroadcast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ExpandBroadcastPattern>();
                                   });
    core::builder::RegisterPattern("ShapeBasedConcatExpand",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ShapeBasedConcatExpandPattern>();
                                   });
    core::builder::RegisterPattern("ShapeBasedExpandBroadcast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ShapeBasedExpandBroadcastPattern>();
                                   });
    core::builder::RegisterPattern(
        "ShapeBasedExpandBroadcastMatMul",
        []() -> std::unique_ptr<core::builder::PatternOptimization> {
          return std::make_unique<ShapeBasedExpandBroadcastMatMulPattern>();
        });
    core::builder::RegisterPattern("ExpandSwap",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ExpandSwapPattern>();
                                   });
    core::builder::RegisterPattern("SwapExpandUnsqueeze",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SwapExpandUnsqueezePattern>();
                                   });
    core::builder::RegisterPattern("ExpandUnsqueezeExpand",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ExpandUnsqueezeExpandPattern>();
                                   });
    core::builder::RegisterPattern("ConcatReshape",
                                   [] { return std::make_unique<ConcatReshapePattern>(); });
    core::builder::RegisterPattern("Reshape", [] { return std::make_unique<ReshapePattern>(); });
    core::builder::RegisterPattern("ReduceReshape",
                                   [] { return std::make_unique<ReduceReshapePattern>(); });
    core::builder::RegisterPattern("Reshape2Of3",
                                   [] { return std::make_unique<Reshape2Of3Pattern>(); });
    core::builder::RegisterPattern("ReshapeReshapeBinary",
                                   [] { return std::make_unique<ReshapeReshapeBinaryPattern>(); });
    core::builder::RegisterPattern("ReshapeReshape",
                                   [] { return std::make_unique<ReshapeReshapePattern>(); });
    core::builder::RegisterPattern("ReshapeSqueeze",
                                   [] { return std::make_unique<ReshapeSqueezePattern>(); });
    core::builder::RegisterPattern("ShapeBasedEditDistanceReshape", [] {
      return std::make_unique<ShapeBasedEditDistanceReshapePattern>();
    });
    core::builder::RegisterPattern("ShapeBasedReshapeIsSqueeze", [] {
      return std::make_unique<ShapeBasedReshapeIsSqueezePattern>();
    });
    core::builder::RegisterPattern("ShapedBasedReshape",
                                   [] { return std::make_unique<ShapedBasedReshapePattern>(); });
    core::builder::RegisterPattern("StaticConcatReshape",
                                   [] { return std::make_unique<StaticConcatReshapePattern>(); });
    core::builder::RegisterPattern("UnsqueezeOrSqueezeReshape", [] {
      return std::make_unique<UnsqueezeOrSqueezeReshapePattern>();
    });
    core::builder::RegisterPattern("UnsqueezeReshape",
                                   [] { return std::make_unique<UnsqueezeReshapePattern>(); });
    core::builder::RegisterPattern("MulUnsqueezeUnsqueeze",
                                   [] { return std::make_unique<MulUnsqueezeUnsqueezePattern>(); });
    core::builder::RegisterPattern("SqueezeAdd",
                                   [] { return std::make_unique<SqueezeAddPattern>(); });
    core::builder::RegisterPattern(
        "SqueezeBinaryUnsqueeze", [] { return std::make_unique<SqueezeBinaryUnsqueezePattern>(); });
    core::builder::RegisterPattern(
        "SwapUnsqueezeTranspose", [] { return std::make_unique<SwapUnsqueezeTransposePattern>(); });
    core::builder::RegisterPattern("TransposeEqualReshape",
                                   [] { return std::make_unique<TransposeEqualReshapePattern>(); });
    core::builder::RegisterPattern("TransposeReshapeTranspose", [] {
      return std::make_unique<TransposeReshapeTransposePattern>();
    });
    core::builder::RegisterPattern("MulMulMulScalar",
                                   [] { return std::make_unique<MulMulMulScalarPattern>(); });
    core::builder::RegisterPattern("SwitchOrderBinary",
                                   [] { return std::make_unique<SwitchOrderBinaryPattern>(); });
    core::builder::RegisterPattern("SwapRangeAddScalar",
                                   [] { return std::make_unique<SwapRangeAddScalarPattern>(); });
    core::builder::RegisterPattern("ReduceArgTopK",
                                   [] { return std::make_unique<ReduceArgTopKPattern>(); });
    core::builder::RegisterPattern("ReduceSumNormalize",
                                   [] { return std::make_unique<ReduceSumNormalizePattern>(); });
    core::builder::RegisterPattern("Sub1Mul", [] { return std::make_unique<Sub1MulPattern>(); });
    core::builder::RegisterPattern("SwapUnary",
                                   [] { return std::make_unique<SwapUnaryPattern>(); });
    core::builder::RegisterPattern("SameChildren",
                                   [] { return std::make_unique<SameChildrenPattern>(); });
    core::builder::RegisterPattern("SameChildrenFromInput",
                                   [] { return std::make_unique<SameChildrenFromInputPattern>(); });
    core::builder::RegisterPattern("ShapeBasedIdentity",
                                   [] { return std::make_unique<ShapeBasedIdentityPattern>(); });
    core::builder::RegisterPattern(
        "ShapeBasedSameChildren", [] { return std::make_unique<ShapeBasedSameChildrenPattern>(); });
    core::builder::RegisterPattern("ShapeBasedShapeShapeAdd", [] {
      return std::make_unique<ShapeBasedShapeShapeAddPattern>();
    });
    core::builder::RegisterPattern("TransposeTranspose",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<TransposeTransposePattern>();
                                   });
    core::builder::RegisterPattern("TransposeGather",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<TransposeGatherPattern>();
                                   });
    core::builder::RegisterPattern("UnsqueezeUnsqueeze",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<UnsqueezeUnsqueezePattern>();
                                   });
    core::builder::RegisterPattern("SqueezeUnsqueeze",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SqueezeUnsqueezePattern>();
                                   });
    core::builder::RegisterPattern("ShapeTranspose",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ShapeTransposePattern>();
                                   });
    core::builder::RegisterPattern("UnsqueezeShape",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<UnsqueezeShapePattern>();
                                   });
    return true;
  }();
  (void)registered;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
