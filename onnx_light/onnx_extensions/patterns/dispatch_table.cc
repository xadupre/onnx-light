// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/dispatch_table.h"

#include <memory>

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
#include "onnx_extensions/patterns/expand/expand_pattern.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"
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
    core::builder::RegisterPattern("ShapeBasedStaticExpand",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ShapeBasedStaticExpandPattern>();
                                   });
    core::builder::RegisterPattern("ShapeBasedExpandSwap",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<ShapeBasedExpandSwapPattern>();
                                   });
    core::builder::RegisterPattern(
        "ShapeBasedExpandCastWhereSwap",
        []() -> std::unique_ptr<core::builder::PatternOptimization> {
          return std::make_unique<ShapeBasedExpandCastWhereSwapPattern>();
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
    core::builder::RegisterPattern("SwapExpandReshape",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<SwapExpandReshapePattern>();
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
