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
#include "onnx_extensions/patterns/collections/split_pattern.h"

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
    core::builder::RegisterPattern("GatherConcat",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GatherConcatPattern>();
                                   });
    core::builder::RegisterPattern("GatherGather",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<GatherGatherPattern>();
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
    return true;
  }();
  (void)registered;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
