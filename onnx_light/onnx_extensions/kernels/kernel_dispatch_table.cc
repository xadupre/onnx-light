// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/kernels/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_extensions/kernels/kernels/optional/include_optional_kernels.h"
#include "onnx_extensions/kernels/kernels/preview/include_preview_kernels.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_extensions/kernels/kernels/rt/include_rt_kernels.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_extensions/kernels/kernels/text/include_text_kernels.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {

// The generic per-node helpers (``RequireInputCount``, ``GetInput``, ...)
// and runtime types (``Tensor``, ``RuntimeContext``, ...) now live in
// ``onnx_core::runtime`` (see ``onnx_core/runtime/node_helpers.h``); pull
// them in unqualified so the dispatch-table factory below can refer to them
// without qualification. Each kernel's ``KernelBase::Run`` implementation
// lives alongside its ``operator()`` in the kernel's own source file.
using namespace ::onnx_light::core::runtime;

namespace {

// Built-in table of every ``onnx_kernels`` operator kernel, keyed by
// ``"<domain>:<op_type>"``. Only used to populate the shared
// ``core::runtime`` dispatch table via :cpp:func:`RegisterKernelFunctions`;
// never queried directly by :cpp:func:`core::runtime::RunNode`.
// Builds the trivial dispatch factory for a concrete kernel ``KernelT``: it
// constructs the kernel from the node's :cpp:class:`KernelContext`, attaches the
// node it runs for and hands ownership to the runtime, which calls
// :cpp:func:`KernelBase::Run` once per node.
template <class KernelT> NodeKernelFn MakeKernel() {
  return [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
    auto kernel = std::make_unique<KernelT>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
}

const std::unordered_map<std::string, NodeKernelFn> &BuiltinKernelFunctions() {
  static const std::unordered_map<std::string, NodeKernelFn> table = {
      {"ai.rt:DelayedInitializer", MakeKernel<onnx_kernels::kernel::DelayedInitializer>()},
      {"ai.onnx:Abs", MakeKernel<onnx_kernels::kernel::Abs>()},
      {"ai.onnx:Acos", MakeKernel<onnx_kernels::kernel::Acos>()},
      {"ai.onnx:Acosh", MakeKernel<onnx_kernels::kernel::Acosh>()},
      {"ai.onnx:Add", MakeKernel<onnx_kernels::kernel::Add>()},
      {"ai.onnx:AffineGrid", MakeKernel<onnx_kernels::kernel::AffineGrid>()},
      {"ai.onnx:And", MakeKernel<onnx_kernels::kernel::And>()},
      {"ai.onnx:ArgMax", MakeKernel<onnx_kernels::kernel::ArgMax>()},
      {"ai.onnx:ArgMin", MakeKernel<onnx_kernels::kernel::ArgMin>()},
      {"ai.onnx:Asin", MakeKernel<onnx_kernels::kernel::Asin>()},
      {"ai.onnx:Asinh", MakeKernel<onnx_kernels::kernel::Asinh>()},
      {"ai.onnx:Atan", MakeKernel<onnx_kernels::kernel::Atan>()},
      {"ai.onnx:Atanh", MakeKernel<onnx_kernels::kernel::Atanh>()},
      {"ai.onnx:Attention", MakeKernel<onnx_kernels::kernel::Attention>()},
      {"ai.onnx:AveragePool", MakeKernel<onnx_kernels::kernel::AveragePool>()},
      {"ai.onnx:BitShift", MakeKernel<onnx_kernels::kernel::BitShift>()},
      {"ai.onnx:BitCast", MakeKernel<onnx_kernels::kernel::BitCast>()},
      {"ai.onnx:BitwiseAnd", MakeKernel<onnx_kernels::kernel::BitwiseAnd>()},
      {"ai.onnx:BitwiseNot", MakeKernel<onnx_kernels::kernel::BitwiseNot>()},
      {"ai.onnx:BitwiseOr", MakeKernel<onnx_kernels::kernel::BitwiseOr>()},
      {"ai.onnx:BitwiseXor", MakeKernel<onnx_kernels::kernel::BitwiseXor>()},
      {"ai.onnx:BatchNormalization", MakeKernel<onnx_kernels::kernel::BatchNormalization>()},
      {"ai.onnx:Bernoulli", MakeKernel<onnx_kernels::kernel::Bernoulli>()},
      {"ai.onnx:BlackmanWindow", MakeKernel<onnx_kernels::kernel::BlackmanWindow>()},
      {"ai.onnx:CausalConvWithState", MakeKernel<onnx_kernels::kernel::CausalConvWithState>()},
      {"ai.onnx:Cast", MakeKernel<onnx_kernels::kernel::Cast>()},
      {"ai.onnx:CastLike", MakeKernel<onnx_kernels::kernel::CastLike>()},
      {"ai.onnx:Ceil", MakeKernel<onnx_kernels::kernel::Ceil>()},
      {"ai.onnx:Celu", MakeKernel<onnx_kernels::kernel::Celu>()},
      {"ai.onnx:CenterCropPad", MakeKernel<onnx_kernels::kernel::CenterCropPad>()},
      {"ai.onnx:Constant", MakeKernel<onnx_kernels::kernel::Constant>()},
      {"ai.onnx:ConstantOfShape", MakeKernel<onnx_kernels::kernel::ConstantOfShape>()},
      {"ai.onnx:Clip", MakeKernel<onnx_kernels::kernel::Clip>()},
      {"ai.onnx:Col2Im", MakeKernel<onnx_kernels::kernel::Col2Im>()},
      {"ai.onnx:Compress", MakeKernel<onnx_kernels::kernel::Compress>()},
      {"ai.onnx:Concat", MakeKernel<onnx_kernels::kernel::Concat>()},
      {"ai.onnx:Conv", MakeKernel<onnx_kernels::kernel::Conv>()},
      {"ai.onnx:ConvInteger", MakeKernel<onnx_kernels::kernel::ConvInteger>()},
      {"ai.onnx:ConvTranspose", MakeKernel<onnx_kernels::kernel::ConvTranspose>()},
      {"ai.onnx:Cos", MakeKernel<onnx_kernels::kernel::Cos>()},
      {"ai.onnx:Cosh", MakeKernel<onnx_kernels::kernel::Cosh>()},
      {"ai.onnx:CumSum", MakeKernel<onnx_kernels::kernel::CumSum>()},
      {"ai.onnx:CumProd", MakeKernel<onnx_kernels::kernel::CumProd>()},
      {"ai.onnx:DeformConv", MakeKernel<onnx_kernels::kernel::DeformConv>()},
      {"ai.onnx:Det", MakeKernel<onnx_kernels::kernel::Det>()},
      {"ai.onnx:DepthToSpace", MakeKernel<onnx_kernels::kernel::DepthToSpace>()},
      {"ai.onnx:SpaceToDepth", MakeKernel<onnx_kernels::kernel::SpaceToDepth>()},
      {"ai.onnx:DequantizeLinear", MakeKernel<onnx_kernels::kernel::DequantizeLinear>()},
      {"ai.onnx:DFT", MakeKernel<onnx_kernels::kernel::DFT>()},
      {"ai.onnx:Div", MakeKernel<onnx_kernels::kernel::Div>()},
      {"ai.onnx:Dropout", MakeKernel<onnx_kernels::kernel::Dropout>()},
      {"ai.onnx:DynamicQuantizeLinear", MakeKernel<onnx_kernels::kernel::DynamicQuantizeLinear>()},
      {"ai.onnx:Einsum", MakeKernel<onnx_kernels::kernel::Einsum>()},
      {"ai.onnx:Elu", MakeKernel<onnx_kernels::kernel::Elu>()},
      {"ai.onnx:Equal", MakeKernel<onnx_kernels::kernel::Equal>()},
      {"ai.onnx:Erf", MakeKernel<onnx_kernels::kernel::Erf>()},
      {"ai.onnx:Exp", MakeKernel<onnx_kernels::kernel::Exp>()},
      {"ai.onnx:Expand", MakeKernel<onnx_kernels::kernel::Expand>()},
      {"ai.onnx:EyeLike", MakeKernel<onnx_kernels::kernel::EyeLike>()},
      {"ai.onnx:Flatten", MakeKernel<onnx_kernels::kernel::Flatten>()},
      {"ai.onnx:Floor", MakeKernel<onnx_kernels::kernel::Floor>()},
      {"ai.onnx:Gather", MakeKernel<onnx_kernels::kernel::Gather>()},
      {"ai.onnx:GatherElements", MakeKernel<onnx_kernels::kernel::GatherElements>()},
      {"ai.onnx:GatherND", MakeKernel<onnx_kernels::kernel::GatherND>()},
      {"ai.onnx:Gelu", MakeKernel<onnx_kernels::kernel::Gelu>()},
      {"ai.onnx:Gemm", MakeKernel<onnx_kernels::kernel::Gemm>()},
      {"ai.onnx:GlobalAveragePool", MakeKernel<onnx_kernels::kernel::GlobalAveragePool>()},
      {"ai.onnx:GlobalLpPool", MakeKernel<onnx_kernels::kernel::GlobalLpPool>()},
      {"ai.onnx:GlobalMaxPool", MakeKernel<onnx_kernels::kernel::GlobalMaxPool>()},
      {"ai.onnx:Greater", MakeKernel<onnx_kernels::kernel::Greater>()},
      {"ai.onnx:GreaterOrEqual", MakeKernel<onnx_kernels::kernel::GreaterOrEqual>()},
      {"ai.onnx:GridSample", MakeKernel<onnx_kernels::kernel::GridSample>()},
      {"ai.onnx:GroupNormalization", MakeKernel<onnx_kernels::kernel::GroupNormalization>()},
      {"ai.onnx:GRU", MakeKernel<onnx_kernels::kernel::GRU>()},
      {"ai.onnx:HardSigmoid", MakeKernel<onnx_kernels::kernel::HardSigmoid>()},
      {"ai.onnx:HardSwish", MakeKernel<onnx_kernels::kernel::HardSwish>()},
      {"ai.onnx:Hardmax", MakeKernel<onnx_kernels::kernel::Hardmax>()},
      {"ai.onnx:HammingWindow", MakeKernel<onnx_kernels::kernel::HammingWindow>()},
      {"ai.onnx:HannWindow", MakeKernel<onnx_kernels::kernel::HannWindow>()},
      {"ai.onnx:Identity", MakeKernel<onnx_kernels::kernel::Identity>()},
      {"ai.onnx:ImageDecoder", MakeKernel<onnx_kernels::kernel::ImageDecoder>()},
      {"ai.onnx:InstanceNormalization", MakeKernel<onnx_kernels::kernel::InstanceNormalization>()},
      {"ai.onnx:IsInf", MakeKernel<onnx_kernels::kernel::IsInf>()},
      {"ai.onnx:IsNaN", MakeKernel<onnx_kernels::kernel::IsNaN>()},
      {"ai.onnx:LayerNormalization", MakeKernel<onnx_kernels::kernel::LayerNormalization>()},
      {"ai.onnx:LeakyRelu", MakeKernel<onnx_kernels::kernel::LeakyRelu>()},
      {"ai.onnx:Less", MakeKernel<onnx_kernels::kernel::Less>()},
      {"ai.onnx:LessOrEqual", MakeKernel<onnx_kernels::kernel::LessOrEqual>()},
      {"ai.onnx:LinearAttention", MakeKernel<onnx_kernels::kernel::LinearAttention>()},
      {"ai.onnx:Log", MakeKernel<onnx_kernels::kernel::Log>()},
      {"ai.onnx:LogSoftmax", MakeKernel<onnx_kernels::kernel::LogSoftmax>()},
      {"ai.onnx:LSTM", MakeKernel<onnx_kernels::kernel::LSTM>()},
      {"ai.onnx:LRN", MakeKernel<onnx_kernels::kernel::LRN>()},
      {"ai.onnx:LpNormalization", MakeKernel<onnx_kernels::kernel::LpNormalization>()},
      {"ai.onnx:LpPool", MakeKernel<onnx_kernels::kernel::LpPool>()},
      {"ai.onnx:MatMul", MakeKernel<onnx_kernels::kernel::MatMul>()},
      {"ai.onnx:MatMulInteger", MakeKernel<onnx_kernels::kernel::MatMulInteger>()},
      {"ai.onnx:Max", MakeKernel<onnx_kernels::kernel::Max>()},
      {"ai.onnx:MaxPool", MakeKernel<onnx_kernels::kernel::MaxPool>()},
      {"ai.onnx:MaxRoiPool", MakeKernel<onnx_kernels::kernel::MaxRoiPool>()},
      {"ai.onnx:MaxUnpool", MakeKernel<onnx_kernels::kernel::MaxUnpool>()},
      {"ai.onnx:Mean", MakeKernel<onnx_kernels::kernel::Mean>()},
      {"ai.onnx:MeanVarianceNormalization",
       MakeKernel<onnx_kernels::kernel::MeanVarianceNormalization>()},
      {"ai.onnx:MelWeightMatrix", MakeKernel<onnx_kernels::kernel::MelWeightMatrix>()},
      {"ai.onnx:Min", MakeKernel<onnx_kernels::kernel::Min>()},
      {"ai.onnx:Mish", MakeKernel<onnx_kernels::kernel::Mish>()},
      {"ai.onnx:Mod", MakeKernel<onnx_kernels::kernel::Mod>()},
      {"ai.onnx:Mul", MakeKernel<onnx_kernels::kernel::Mul>()},
      {"ai.onnx:Multinomial", MakeKernel<onnx_kernels::kernel::Multinomial>()},
      {"ai.onnx:Neg", MakeKernel<onnx_kernels::kernel::Neg>()},
      {"ai.onnx:NegativeLogLikelihoodLoss",
       MakeKernel<onnx_kernels::kernel::NegativeLogLikelihoodLoss>()},
      {"ai.onnx:NonMaxSuppression", MakeKernel<onnx_kernels::kernel::NonMaxSuppression>()},
      {"ai.onnx:NonZero", MakeKernel<onnx_kernels::kernel::NonZero>()},
      {"ai.onnx:Not", MakeKernel<onnx_kernels::kernel::Not>()},
      {"ai.onnx:OneHot", MakeKernel<onnx_kernels::kernel::OneHot>()},
      {"ai.onnx:Or", MakeKernel<onnx_kernels::kernel::Or>()},
      {"ai.onnx:Optional", MakeKernel<onnx_kernels::kernel::Optional>()},
      {"ai.onnx:OptionalGetElement", MakeKernel<onnx_kernels::kernel::OptionalGetElement>()},
      {"ai.onnx:OptionalHasElement", MakeKernel<onnx_kernels::kernel::OptionalHasElement>()},
      {"ai.onnx:Pad", MakeKernel<onnx_kernels::kernel::Pad>()},
      {"ai.onnx:Pow", MakeKernel<onnx_kernels::kernel::Pow>()},
      {"ai.onnx:PRelu", MakeKernel<onnx_kernels::kernel::PRelu>()},
      {"ai.onnx:QLinearConv", MakeKernel<onnx_kernels::kernel::QLinearConv>()},
      {"ai.onnx:QLinearMatMul", MakeKernel<onnx_kernels::kernel::QLinearMatMul>()},
      {"ai.onnx:QuantizeLinear", MakeKernel<onnx_kernels::kernel::QuantizeLinear>()},
      {"ai.onnx:RandomNormal", MakeKernel<onnx_kernels::kernel::RandomNormal>()},
      {"ai.onnx:RandomNormalLike", MakeKernel<onnx_kernels::kernel::RandomNormalLike>()},
      {"ai.onnx:RandomUniform", MakeKernel<onnx_kernels::kernel::RandomUniform>()},
      {"ai.onnx:RandomUniformLike", MakeKernel<onnx_kernels::kernel::RandomUniformLike>()},
      {"ai.onnx:Range", MakeKernel<onnx_kernels::kernel::Range>()},
      {"ai.onnx:RMSNormalization", MakeKernel<onnx_kernels::kernel::RMSNormalization>()},
      {"ai.onnx:Reciprocal", MakeKernel<onnx_kernels::kernel::Reciprocal>()},
      {"ai.onnx:ReduceL1", MakeKernel<onnx_kernels::kernel::ReduceL1>()},
      {"ai.onnx:ReduceL2", MakeKernel<onnx_kernels::kernel::ReduceL2>()},
      {"ai.onnx:ReduceLogSum", MakeKernel<onnx_kernels::kernel::ReduceLogSum>()},
      {"ai.onnx:ReduceLogSumExp", MakeKernel<onnx_kernels::kernel::ReduceLogSumExp>()},
      {"ai.onnx:ReduceMax", MakeKernel<onnx_kernels::kernel::ReduceMax>()},
      {"ai.onnx:ReduceMean", MakeKernel<onnx_kernels::kernel::ReduceMean>()},
      {"ai.onnx:ReduceMin", MakeKernel<onnx_kernels::kernel::ReduceMin>()},
      {"ai.onnx:ReduceProd", MakeKernel<onnx_kernels::kernel::ReduceProd>()},
      {"ai.onnx:ReduceSum", MakeKernel<onnx_kernels::kernel::ReduceSum>()},
      {"ai.onnx:ReduceSumSquare", MakeKernel<onnx_kernels::kernel::ReduceSumSquare>()},
      {"ai.onnx:RegexFullMatch", MakeKernel<onnx_kernels::kernel::RegexFullMatch>()},
      {"ai.onnx:Relu", MakeKernel<onnx_kernels::kernel::Relu>()},
      {"ai.onnx:Reshape", MakeKernel<onnx_kernels::kernel::Reshape>()},
      {"ai.onnx:Resize", MakeKernel<onnx_kernels::kernel::Resize>()},
      {"ai.onnx:ReverseSequence", MakeKernel<onnx_kernels::kernel::ReverseSequence>()},
      {"ai.onnx:RoiAlign", MakeKernel<onnx_kernels::kernel::RoiAlign>()},
      {"ai.onnx:Round", MakeKernel<onnx_kernels::kernel::Round>()},
      {"ai.onnx:RNN", MakeKernel<onnx_kernels::kernel::RNN>()},
      {"ai.onnx:RotaryEmbedding", MakeKernel<onnx_kernels::kernel::RotaryEmbedding>()},
      {"ai.onnx:Scatter", MakeKernel<onnx_kernels::kernel::Scatter>()},
      {"ai.onnx:ScatterElements", MakeKernel<onnx_kernels::kernel::ScatterElements>()},
      {"ai.onnx:ScatterND", MakeKernel<onnx_kernels::kernel::ScatterND>()},
      {"ai.onnx:Selu", MakeKernel<onnx_kernels::kernel::Selu>()},
      {"ai.onnx:ConcatFromSequence", MakeKernel<onnx_kernels::kernel::ConcatFromSequence>()},
      {"ai.onnx:SequenceAt", MakeKernel<onnx_kernels::kernel::SequenceAt>()},
      {"ai.onnx:SequenceConstruct", MakeKernel<onnx_kernels::kernel::SequenceConstruct>()},
      {"ai.onnx:SequenceEmpty", MakeKernel<onnx_kernels::kernel::SequenceEmpty>()},
      {"ai.onnx:SequenceErase", MakeKernel<onnx_kernels::kernel::SequenceErase>()},
      {"ai.onnx:SequenceInsert", MakeKernel<onnx_kernels::kernel::SequenceInsert>()},
      {"ai.onnx:SequenceLength", MakeKernel<onnx_kernels::kernel::SequenceLength>()},
      {"ai.onnx:Split", MakeKernel<onnx_kernels::kernel::Split>()},
      {"ai.onnx:SplitToSequence", MakeKernel<onnx_kernels::kernel::SplitToSequence>()},
      {"ai.onnx:Shape", MakeKernel<onnx_kernels::kernel::Shape>()},
      {"ai.onnx:Shrink", MakeKernel<onnx_kernels::kernel::Shrink>()},
      {"ai.onnx:Sigmoid", MakeKernel<onnx_kernels::kernel::Sigmoid>()},
      {"ai.onnx:Sign", MakeKernel<onnx_kernels::kernel::Sign>()},
      {"ai.onnx:Sin", MakeKernel<onnx_kernels::kernel::Sin>()},
      {"ai.onnx:Sinh", MakeKernel<onnx_kernels::kernel::Sinh>()},
      {"ai.onnx:Size", MakeKernel<onnx_kernels::kernel::Size>()},
      {"ai.onnx:Slice", MakeKernel<onnx_kernels::kernel::Slice>()},
      {"ai.onnx:Softmax", MakeKernel<onnx_kernels::kernel::Softmax>()},
      {"ai.onnx:SoftmaxCrossEntropyLoss",
       MakeKernel<onnx_kernels::kernel::SoftmaxCrossEntropyLoss>()},
      {"ai.onnx:Softplus", MakeKernel<onnx_kernels::kernel::Softplus>()},
      {"ai.onnx:Softsign", MakeKernel<onnx_kernels::kernel::Softsign>()},
      {"ai.onnx:Sqrt", MakeKernel<onnx_kernels::kernel::Sqrt>()},
      {"ai.onnx:Squeeze", MakeKernel<onnx_kernels::kernel::Squeeze>()},
      {"ai.onnx:STFT", MakeKernel<onnx_kernels::kernel::STFT>()},
      {"ai.onnx:StringConcat", MakeKernel<onnx_kernels::kernel::StringConcat>()},
      {"ai.onnx:StringNormalizer", MakeKernel<onnx_kernels::kernel::StringNormalizer>()},
      {"ai.onnx:StringSplit", MakeKernel<onnx_kernels::kernel::StringSplit>()},
      {"ai.onnx:Sub", MakeKernel<onnx_kernels::kernel::Sub>()},
      {"ai.onnx:Sum", MakeKernel<onnx_kernels::kernel::Sum>()},
      {"ai.onnx:Swish", MakeKernel<onnx_kernels::kernel::Swish>()},
      {"ai.onnx:SwiGLU", MakeKernel<onnx_kernels::kernel::SwiGLU>()},
      {"ai.onnx:Tan", MakeKernel<onnx_kernels::kernel::Tan>()},
      {"ai.onnx:Tanh", MakeKernel<onnx_kernels::kernel::Tanh>()},
      {"ai.onnx:TensorScatter", MakeKernel<onnx_kernels::kernel::TensorScatter>()},
      {"ai.onnx:TfIdfVectorizer", MakeKernel<onnx_kernels::kernel::TfIdfVectorizer>()},
      {"ai.onnx:ThresholdedRelu", MakeKernel<onnx_kernels::kernel::ThresholdedRelu>()},
      {"ai.onnx:Tile", MakeKernel<onnx_kernels::kernel::Tile>()},
      {"ai.onnx:TopK", MakeKernel<onnx_kernels::kernel::TopK>()},
      {"ai.onnx:Transpose", MakeKernel<onnx_kernels::kernel::Transpose>()},
      {"ai.onnx:Trilu", MakeKernel<onnx_kernels::kernel::Trilu>()},
      {"ai.onnx:Unique", MakeKernel<onnx_kernels::kernel::Unique>()},
      {"ai.onnx:Unsqueeze", MakeKernel<onnx_kernels::kernel::Unsqueeze>()},
      {"ai.onnx:Upsample", MakeKernel<onnx_kernels::kernel::Upsample>()},
      {"ai.onnx:Where", MakeKernel<onnx_kernels::kernel::Where>()},
      {"ai.onnx:Xor", MakeKernel<onnx_kernels::kernel::Xor>()},
      {"ai.onnx.preview:FlexAttention", MakeKernel<onnx_kernels::kernel::FlexAttention>()},
      {"ai.onnx.preview.training:Adagrad", MakeKernel<onnx_kernels::kernel::Adagrad>()},
      {"ai.onnx.preview.training:Adam", MakeKernel<onnx_kernels::kernel::Adam>()},
      {"ai.onnx.preview.training:Momentum", MakeKernel<onnx_kernels::kernel::Momentum>()},
      {"ai.onnx.ml:CastMap", MakeKernel<onnx_kernels::kernel::CastMap>()},
      {"ai.onnx.ml:DictVectorizer", MakeKernel<onnx_kernels::kernel::DictVectorizer>()},
      {"ai.onnx.ml:SVMRegressor", MakeKernel<onnx_kernels::kernel::SVMRegressor>()},
      {"ai.onnx.ml:SVMClassifier", MakeKernel<onnx_kernels::kernel::SVMClassifier>()},
      {"ai.onnx.ml:LinearRegressor", MakeKernel<onnx_kernels::kernel::LinearRegressor>()},
      {"ai.onnx.ml:TreeEnsembleRegressor",
       MakeKernel<onnx_kernels::kernel::TreeEnsembleRegressor>()},
      {"ai.onnx.ml:LinearClassifier", MakeKernel<onnx_kernels::kernel::LinearClassifier>()},
      {"ai.onnx.ml:TreeEnsembleClassifier",
       MakeKernel<onnx_kernels::kernel::TreeEnsembleClassifier>()},
      {"ai.onnx.ml:Binarizer", MakeKernel<onnx_kernels::kernel::Binarizer>()},
      {"ai.onnx.ml:Normalizer", MakeKernel<onnx_kernels::kernel::Normalizer>()},
      {"ai.onnx.ml:Scaler", MakeKernel<onnx_kernels::kernel::Scaler>()},
      {"ai.onnx.ml:ArrayFeatureExtractor",
       MakeKernel<onnx_kernels::kernel::ArrayFeatureExtractor>()},
      {"ai.onnx.ml:Imputer", MakeKernel<onnx_kernels::kernel::Imputer>()},
      {"ai.onnx.ml:CategoryMapper", MakeKernel<onnx_kernels::kernel::CategoryMapper>()},
      {"ai.onnx.ml:LabelEncoder", MakeKernel<onnx_kernels::kernel::LabelEncoder>()},
      {"ai.onnx.ml:OneHotEncoder", MakeKernel<onnx_kernels::kernel::OneHotEncoder>()},
      {"ai.onnx.ml:FeatureVectorizer", MakeKernel<onnx_kernels::kernel::FeatureVectorizer>()},
      {"ai.onnx.ml:TreeEnsemble", MakeKernel<onnx_kernels::kernel::TreeEnsemble>()},
  };
  return table;
}

} // namespace

void RegisterKernelFunctions() {
  // `lib_onnx_kernels` is a plain static archive: a translation unit with no
  // symbol referenced from elsewhere would simply be dropped by the linker,
  // so registration cannot rely on a file-scope static object running as a
  // side effect of "just being linked in" (unlike, say,
  // `OpSchemaRegistry::map()` in `onnx_lib`, which can call its registration
  // functions lazily because both live in the same library). Callers must
  // invoke this function explicitly before running a model; it is
  // idempotent, so calling it more than once (or from multiple independent
  // entry points) is safe and cheap after the first call.
  static const bool kRegistered = [] {
    onnx_kernels::kernel::Abs::RegisterTuningSchemas();
    onnx_kernels::kernel::Add::RegisterTuningSchemas();
    onnx_kernels::kernel::And::RegisterTuningSchemas();
    onnx_kernels::kernel::Exp::RegisterTuningSchemas();
    onnx_kernels::kernel::Gemm::RegisterTuningSchemas();
    onnx_kernels::kernel::Mul::RegisterTuningSchemas();
    onnx_kernels::kernel::Not::RegisterTuningSchemas();
    for (const auto &entry : BuiltinKernelFunctions()) {
      const std::string &key = entry.first;
      const std::size_t sep = key.find(':');
      // Register the built-in factory only when no kernel is already
      // registered for this identifier, so a downstream override (e.g. the
      // SIMD kernels shipped by onnx-light-cpu) installed *before* this bulk
      // registration is not clobbered. Overrides installed afterwards keep
      // using the default overwriting RegisterKernelFn and still win.
      ::onnx_light::core::runtime::RegisterKernelFn(key.substr(0, sep), key.substr(sep + 1),
                                                    ::onnx_light::core::symbolic::Device::kCPU,
                                                    entry.second, /*overwrite=*/false);
    }
    ::onnx_light::core::runtime::RegisterSequenceMapPackFn(
        [](RuntimeContext &rt, const Sequence &input_sequence,
           const std::vector<Tensors> &body_outputs_per_iter) {
          onnx_kernels::kernel::SequenceMap seq_map_kernel(rt.kernel_ctx());
          return seq_map_kernel(input_sequence, body_outputs_per_iter);
        });
    return true;
  }();
  (void)kRegistered;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
