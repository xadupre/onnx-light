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
#include "onnx_extensions/patterns/attention/attention_pattern.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"
#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_extensions/patterns/canonicalization/not_pattern.h"
#include "onnx_extensions/patterns/canonicalization/pad_pattern.h"
#include "onnx_extensions/patterns/canonicalization/stft_pattern.h"
#include "onnx_extensions/patterns/collections/concat_pattern.h"
#include "onnx_extensions/patterns/collections/gather_pattern.h"
#include "onnx_extensions/patterns/collections/sequence_pattern.h"
#include "onnx_extensions/patterns/collections/shape_pattern.h"
#include "onnx_extensions/patterns/collections/slice_pattern.h"
#include "onnx_extensions/patterns/collections/split_pattern.h"
#include "onnx_extensions/patterns/expand/expand_pattern.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"
#include "onnx_extensions/patterns/layout/layout_pattern.h"
#include "onnx_extensions/patterns/matmul/matmul_pattern.h"
#include "onnx_extensions/patterns/normalization/activation_pattern.h"
#include "onnx_extensions/patterns/normalization/normalization_pattern.h"
#include "onnx_extensions/patterns/reshape/reshape_pattern.h"
#include "onnx_extensions/patterns/traditionalml/label_encoder_pattern.h"
#include "onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h"
#include "onnx_extensions/patterns/transpose/transpose_pattern.h"
#include "onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

void RegisterPatterns() {
  static const bool registered =
      []() {
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
        core::builder::RegisterPattern("ReluClipFusion",
                                       [] { return std::make_unique<ReluClipFusionPattern>(); });
        core::builder::RegisterPattern("ConstantToInitializer",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConstantToInitializerPattern>();
                                       });
        core::builder::RegisterPattern("ConvBiasNull",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConvBiasNullPattern>();
                                       });
        core::builder::RegisterPattern("ConvAddFusion",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConvAddFusionPattern>();
                                       });
        core::builder::RegisterPattern("ConvMulFusion",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConvMulFusionPattern>();
                                       });
        core::builder::RegisterPattern(
            "ConvBatchNormalizationFusion",
            []() -> std::unique_ptr<core::builder::PatternOptimization> {
              return std::make_unique<ConvBatchNormalizationFusionPattern>();
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
        core::builder::RegisterPattern("PadPadFusion",
                                       [] { return std::make_unique<PadPadFusionPattern>(); });
        core::builder::RegisterPattern("ConcatEmpty",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConcatEmptyPattern>();
                                       });
        core::builder::RegisterPattern("ConcatSliceElimination",
                                       []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                         return std::make_unique<ConcatSliceEliminationPattern>();
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
    core::builder::RegisterPattern("PreShapeNodeElimination",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
      return std::make_unique<PreShapeNodeEliminationPattern>();
      core::builder::RegisterPattern("GatherUpstreamPropagation",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<GatherUpstreamPropagationPattern>();
                                     });
      core::builder::RegisterPattern("GathersSplit",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<GathersSplitPattern>();
                                     });
      core::builder::RegisterPattern("GatherSliceToSplit",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<GatherSliceToSplitPattern>();
                                     });
      core::builder::RegisterPattern("GatherToSlice",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<GatherToSlicePattern>();
                                     });
      core::builder::RegisterPattern("SliceSlice",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<SliceSlicePattern>();
                                     });
      core::builder::RegisterPattern("SliceElimination",
                                     [] { return std::make_unique<SliceEliminationPattern>(); });
      core::builder::RegisterPattern("SlicesSplit",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<SlicesSplitPattern>();
                                     });
      core::builder::RegisterPattern("SplitConcat",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<SplitConcatPattern>();
                                     });
      core::builder::RegisterPattern("SliceConcatToSpaceToDepth",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<SliceConcatToSpaceToDepthPattern>();
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
      core::builder::RegisterPattern("ConcatReshape",
                                     [] { return std::make_unique<ConcatReshapePattern>(); });
      core::builder::RegisterPattern("Reshape", [] { return std::make_unique<ReshapePattern>(); });
      core::builder::RegisterPattern("ReduceReshape",
                                     [] { return std::make_unique<ReduceReshapePattern>(); });
      core::builder::RegisterPattern("Reshape2Of3",
                                     [] { return std::make_unique<Reshape2Of3Pattern>(); });
      core::builder::RegisterPattern(
          "ReshapeReshapeBinary", [] { return std::make_unique<ReshapeReshapeBinaryPattern>(); });
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
      core::builder::RegisterPattern(
          "MulUnsqueezeUnsqueeze", [] { return std::make_unique<MulUnsqueezeUnsqueezePattern>(); });
      core::builder::RegisterPattern("SqueezeAdd",
                                     [] { return std::make_unique<SqueezeAddPattern>(); });
      core::builder::RegisterPattern("SqueezeBinaryUnsqueeze", [] {
        return std::make_unique<SqueezeBinaryUnsqueezePattern>();
      });
      core::builder::RegisterPattern("SwapUnsqueezeTranspose", [] {
        return std::make_unique<SwapUnsqueezeTransposePattern>();
      });
      core::builder::RegisterPattern(
          "TransposeEqualReshape", [] { return std::make_unique<TransposeEqualReshapePattern>(); });
      core::builder::RegisterPattern("TransposeReshapeTranspose", [] {
        return std::make_unique<TransposeReshapeTransposePattern>();
      });
      core::builder::RegisterPattern("DivMul", [] { return std::make_unique<DivMulPattern>(); });
      core::builder::RegisterPattern("STFTFusion",
                                     [] { return std::make_unique<STFTFusionPattern>(); });
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
      core::builder::RegisterPattern(
          "SameChildrenFromInput", [] { return std::make_unique<SameChildrenFromInputPattern>(); });
      core::builder::RegisterPattern("ShapeBasedIdentity",
                                     [] { return std::make_unique<ShapeBasedIdentityPattern>(); });
      core::builder::RegisterPattern("ShapeBasedSameChildren", [] {
        return std::make_unique<ShapeBasedSameChildrenPattern>();
      });
      core::builder::RegisterPattern("ShapeBasedShapeShapeAdd", [] {
        return std::make_unique<ShapeBasedShapeShapeAddPattern>();
      });
      core::builder::RegisterPattern("GemmTranspose",
                                     [] { return std::make_unique<GemmTransposePattern>(); });
      core::builder::RegisterPattern("GemmSumFusion",
                                     [] { return std::make_unique<GemmSumFusionPattern>(); });
      core::builder::RegisterPattern("MatMulAdd",
                                     [] { return std::make_unique<MatMulAddPattern>(); });
      core::builder::RegisterPattern("MatMulBatchNormalizationFusion", [] {
        return std::make_unique<MatMulBatchNormalizationFusionPattern>();
      });
      core::builder::RegisterPattern("MatMulReshape2Of3",
                                     [] { return std::make_unique<MatMulReshape2Of3Pattern>(); });
      core::builder::RegisterPattern("MatMulScaleFusion",
                                     [] { return std::make_unique<MatMulScaleFusionPattern>(); });
      core::builder::RegisterPattern("MulMulMatMul",
                                     [] { return std::make_unique<MulMulMatMulPattern>(); });
      core::builder::RegisterPattern(
          "ReshapeMatMulReshape", [] { return std::make_unique<ReshapeMatMulReshapePattern>(); });
      core::builder::RegisterPattern(
          "ShapeBasedMatMulToMul", [] { return std::make_unique<ShapeBasedMatMulToMulPattern>(); });
      core::builder::RegisterPattern("SwitchReshapeActivation", [] {
        return std::make_unique<SwitchReshapeActivationPattern>();
      });
      core::builder::RegisterPattern("TransposeMatMul",
                                     [] { return std::make_unique<TransposeMatMulPattern>(); });
      core::builder::RegisterPattern("TransposeReshapeMatMul", [] {
        return std::make_unique<TransposeReshapeMatMulPattern>();
      });
      core::builder::RegisterPattern("BatchNormalization",
                                     [] { return std::make_unique<BatchNormalizationPattern>(); });
      core::builder::RegisterPattern("BatchNormalizationTraining", [] {
        return std::make_unique<BatchNormalizationTrainingPattern>();
      });
      core::builder::RegisterPattern("CastLayerNormalizationCast", [] {
        return std::make_unique<CastLayerNormalizationCastPattern>();
      });
      core::builder::RegisterPattern("LayerNormalization",
                                     [] { return std::make_unique<LayerNormalizationPattern>(); });
      core::builder::RegisterPattern("LayerNormalizationScale", [] {
        return std::make_unique<LayerNormalizationScalePattern>();
      });
      core::builder::RegisterPattern("RMSNormalization",
                                     [] { return std::make_unique<RMSNormalizationPattern>(); });
      core::builder::RegisterPattern("RMSNormalizationMul",
                                     [] { return std::make_unique<RMSNormalizationMulPattern>(); });
      core::builder::RegisterPattern("Gelu", [] { return std::make_unique<GeluPattern>(); });
      core::builder::RegisterPattern("LeakyRelu",
                                     [] { return std::make_unique<LeakyReluPattern>(); });
      core::builder::RegisterPattern("MaxRelu", [] { return std::make_unique<MaxReluPattern>(); });
      core::builder::RegisterPattern("SoftmaxCrossEntropyLossCast", [] {
        return std::make_unique<SoftmaxCrossEntropyLossCastPattern>();
      });
      core::builder::RegisterPattern("RotaryEmbedding",
                                     [] { return std::make_unique<RotaryEmbeddingPattern>(); });
      core::builder::RegisterPattern("RotaryConcatPart",
                                     [] { return std::make_unique<RotaryConcatPartPattern>(); });
      core::builder::RegisterPattern("FunctionCausalMask",
                                     [] { return std::make_unique<FunctionCausalMaskPattern>(); });
      core::builder::RegisterPattern("FunctionCausalMaskMulAdd", [] {
        return std::make_unique<FunctionCausalMaskMulAddPattern>();
      });
      core::builder::RegisterPattern("FunctionCosSinCache",
                                     [] { return std::make_unique<FunctionCosSinCachePattern>(); });
      core::builder::RegisterPattern("FunctionHalfRotaryEmbedding", [] {
        return std::make_unique<FunctionHalfRotaryEmbeddingPattern>();
      });
      core::builder::RegisterPattern("FunctionAttention",
                                     [] { return std::make_unique<FunctionAttentionPattern>(); });
      core::builder::RegisterPattern("LinearAttention",
                                     [] { return std::make_unique<LinearAttentionPattern>(); });
      core::builder::RegisterPattern(
          "FunctionAttentionGQA", [] { return std::make_unique<FunctionAttentionGQAPattern>(); });
      core::builder::RegisterPattern("AttentionGQA",
                                     [] { return std::make_unique<AttentionGQAPattern>(); });
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
      core::builder::RegisterPattern("TreeEnsemble",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<TreeEnsemblePattern>();
                                     });
      core::builder::RegisterPattern("LabelEncoderFusion",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<LabelEncoderFusionPattern>();
                                     });
      core::builder::RegisterPattern("UnsqueezeShape",
                                     []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                       return std::make_unique<UnsqueezeShapePattern>();
                                     });
      return true;
  }();
  (void)registered;
      }

  std::unique_ptr<core::builder::PatternOptimization>
  CreatePattern(const std::string &name, std::optional<int> priority) {
    RegisterPatterns();
    return core::builder::CreateRegisteredPattern(name, priority);
  }

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
