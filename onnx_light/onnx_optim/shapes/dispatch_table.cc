// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/dispatch_table.h"

#include <string>
#include <unordered_map>

#include "onnx_proto/onnx_helper.h"

#include "onnx_optim/shapes/controlflow/shape_controlflow.h"
#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/nn/shape_nn.h"
#include "onnx_optim/shapes/optional/shape_optional.h"
#include "onnx_optim/shapes/preview/shape_preview.h"
#include "onnx_optim/shapes/quantization/shape_quantization.h"
#include "onnx_optim/shapes/reduction/shape_reduction.h"
#include "onnx_optim/shapes/sequence/shape_sequence.h"
#include "onnx_optim/shapes/tensor/shape_tensor.h"
#include "onnx_optim/shapes/text/shape_text.h"
#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"
#include "onnx_optim/shapes/training/shape_training.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Verifies the node declares at least `expected` inputs.
void RequireInputs(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(node.input_size() >= expected,
                      "ComputeShapeNode: op '" + node.op_type().as_string() +
                          "' expects at least " + std::to_string(expected) + " input(s), got " +
                          std::to_string(node.input_size()) + ".");
}

} // namespace

const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  static const std::unordered_map<std::string, ComputeShapeFn> table = {
      {"ai.onnx:Abs",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAbs(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acos",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcos(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acosh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcosh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Add",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeAdd(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"ai.onnx:AffineGrid",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeAffineGrid(ctx, node);
       }},
      {"ai.onnx:And",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeAnd(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"ai.onnx:ArgMax",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         reduction::ComputeShapeArgReduce(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:ArgMin",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         reduction::ComputeShapeArgReduce(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Asin",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAsin(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Asinh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAsinh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Atan",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAtan(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Atanh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAtanh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Attention",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         const std::string q_name = node.input(0).as_string();
         const std::string k_name = node.input(1).as_string();
         const std::string v_name = node.input(2).as_string();
         const std::string past_k_name = node.input_size() > 4 ? node.input(4).as_string() : "";
         const std::string past_v_name = node.input_size() > 5 ? node.input(5).as_string() : "";
         nn::ComputeShapeAttention(ctx, node, q_name.c_str(), k_name.c_str(), v_name.c_str(),
                                   past_k_name.empty() ? nullptr : past_k_name.c_str(),
                                   past_v_name.empty() ? nullptr : past_v_name.c_str());
       }},
      {"ai.onnx:AveragePool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeAveragePool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:BatchNormalization",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 5);
         const std::string x_name = node.input(0).as_string();
         const std::string mean_name = node.input(3).as_string();
         nn::ComputeShapeBatchNormalization(ctx, node, x_name.c_str(),
                                            mean_name.empty() ? nullptr : mean_name.c_str());
       }},
      {"ai.onnx:Dropout",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        const std::string data_name = node.input(0).as_string();
        const std::string ratio_name =
            node.input_size() >= 2 ? node.input(1).as_string() : std::string();
        const std::string training_mode_name =
            node.input_size() >= 3 ? node.input(2).as_string() : std::string();
        nn::ComputeShapeDropout(ctx, node, data_name.c_str(),
                                ratio_name.empty() ? nullptr : ratio_name.c_str(),
                                training_mode_name.empty() ? nullptr : training_mode_name.c_str());
       }},
      {"ai.onnx:EyeLike",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeEyeLike(ctx, node);
       }},
      {"ai.onnx:Flatten",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeFlatten(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Bernoulli",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeBernoulli(ctx, node);
       }},
      {"ai.onnx:BitCast",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeBitCast(ctx, node);
       }},
      {"ai.onnx:BitShift",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeBitShift(ctx, node, node.input(0).as_string().c_str(),
                                       node.input(1).as_string().c_str());
       }},
      {"ai.onnx:BitwiseAnd",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeBitwiseAnd(ctx, node, node.input(0).as_string().c_str(),
                                         node.input(1).as_string().c_str());
       }},
      {"ai.onnx:BitwiseNot",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         logical::ComputeShapeBitwiseNot(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:BitwiseOr",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeBitwiseOr(ctx, node, node.input(0).as_string().c_str(),
                                        node.input(1).as_string().c_str());
       }},
      {"ai.onnx:BitwiseXor",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeBitwiseXor(ctx, node, node.input(0).as_string().c_str(),
                                         node.input(1).as_string().c_str());
       }},
      {"ai.onnx:BlackmanWindow",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeBlackmanWindow(ctx, node);
       }},
      {"ai.onnx:Cast",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeCast(ctx, node);
       }},
      {"ai.onnx:CastLike",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeCastLike(ctx, node);
       }},
      {"ai.onnx:Compress",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeCompress(ctx, node);
       }},
      {"ai.onnx:Constant",
       [](ShapesContext &ctx, const NodeProto &node) {
         generator::ComputeShapeConstant(ctx, node);
       }},
      {"ai.onnx:ConstantOfShape",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeConstantOfShape(ctx, node);
       }},
      {"ai.onnx:Concat",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeConcat(ctx, node);
       }},
      {"ai.onnx:Cos",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeCos(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Cosh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeCosh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:CumProd",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeCumProd(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:CumSum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeCumSum(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Ceil",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeCeil(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Clip",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeClip(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Floor",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeFloor(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Round",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeRound(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Sin",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSin(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Sinh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSinh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Sqrt",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSqrt(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Sum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSum(ctx, node);
       }},
      {"ai.onnx:Tan",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeTan(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Tanh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeTanh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:DequantizeLinear",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         const std::string x_name = node.input(0).as_string();
         const std::string x_scale_name = node.input(1).as_string();
         quantization::ComputeShapeDequantizeLinear(ctx, node, x_name.c_str(),
                                                    x_scale_name.c_str());
       }},
      {"ai.onnx:Col2Im",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         nn::ComputeShapeCol2Im(ctx, node, node.input(0).as_string().c_str(),
                                node.input(1).as_string().c_str(),
                                node.input(2).as_string().c_str());
       }},
      {"ai.onnx:Conv",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         nn::ComputeShapeConv(ctx, node, node.input(0).as_string().c_str(),
                              node.input(1).as_string().c_str());
       }},
      {"ai.onnx:ConvInteger",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         nn::ComputeShapeConvInteger(ctx, node, node.input(0).as_string().c_str(),
                                     node.input(1).as_string().c_str());
       }},
      {"ai.onnx:ConvTranspose",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         nn::ComputeShapeConvTranspose(ctx, node, node.input(0).as_string().c_str(),
                                       node.input(1).as_string().c_str());
       }},
      {"ai.onnx:DeformConv",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         nn::ComputeShapeDeformConv(ctx, node, node.input(0).as_string().c_str(),
                                    node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Det",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeDet(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Div",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeDiv(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Einsum",
      [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        math::ComputeShapeEinsum(ctx, node);
      }},
      {"ai.onnx:Erf",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
        math::ComputeShapeErf(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Exp",
      [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        math::ComputeShapeExp(ctx, node, node.input(0).as_string().c_str());
      }},
      {"ai.onnx:Equal",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeEqual(ctx, node, node.input(0).as_string().c_str(),
                                    node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Where",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         logical::ComputeShapeWhere(ctx, node, node.input(0).as_string().c_str(),
                                    node.input(1).as_string().c_str(),
                                    node.input(2).as_string().c_str());
       }},
      {"ai.onnx:Xor",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeXor(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Expand",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeExpand(ctx, node);
       }},
      {"ai.onnx:Gemm",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeGemm(ctx, node, node.input(0).as_string().c_str(),
                                node.input(1).as_string().c_str());
       }},
      {"ai.onnx:GlobalAveragePool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeGlobalPool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:GlobalLpPool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeGlobalPool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:GlobalMaxPool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeGlobalPool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Greater",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeGreater(ctx, node, node.input(0).as_string().c_str(),
                                      node.input(1).as_string().c_str());
       }},
      {"ai.onnx:GreaterOrEqual",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeGreaterOrEqual(ctx, node, node.input(0).as_string().c_str(),
                                             node.input(1).as_string().c_str());
       }},
      {"ai.onnx:GridSample",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeGridSample(ctx, node);
       }},
      {"ai.onnx:Gather",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeGather(ctx, node);
       }},
      {"ai.onnx:GatherElements",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeGatherElements(ctx, node);
       }},
      {"ai.onnx:GatherND",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeGatherND(ctx, node);
       }},
      {"ai.onnx:HammingWindow",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeHammingWindow(ctx, node);
       }},
      {"ai.onnx:HannWindow",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         generator::ComputeShapeHannWindow(ctx, node);
       }},
      {"ai.onnx:If",
       [](ShapesContext &ctx, const NodeProto &node) { controlflow::ComputeShapeIf(ctx, node); }},
      {"ai.onnx:Loop",
       [](ShapesContext &ctx, const NodeProto &node) {
         controlflow::ComputeShapeLoop(ctx, node);
       }},
      {"ai.onnx:Log",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeLog(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Less",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeLess(ctx, node, node.input(0).as_string().c_str(),
                                   node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Mul",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeMul(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"ai.onnx:MatMul",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeMatMul(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"ai.onnx:Optional",
       [](ShapesContext &ctx, const NodeProto &node) {
         optional::ComputeShapeOptional(ctx, node);
       }},
      {"ai.onnx:QuantizeLinear",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         const std::string x_name = node.input(0).as_string();
         const std::string zp_name =
             node.input_size() >= 3 ? node.input(2).as_string() : std::string();
         quantization::ComputeShapeQuantizeLinear(ctx, node, x_name.c_str(),
                                                  zp_name.empty() ? nullptr : zp_name.c_str());
       }},
      {"ai.onnx:ReduceSum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceSum(ctx, node, data_name.c_str(),
                                          node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:ReduceSumSquare",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceSumSquare(
             ctx, node, data_name.c_str(),
             node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:ReduceL1",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceL1(ctx, node, data_name.c_str(),
                                         node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:ReduceL2",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceL2(ctx, node, data_name.c_str(),
                                         node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:ReduceMax",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceMax(ctx, node, data_name.c_str(),
                                          node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:ReduceMin",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceMin(ctx, node, data_name.c_str(),
                                          node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:Reshape",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeReshape(ctx, node);
       }},
      {"ai.onnx:DepthToSpace",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeDepthToSpace(ctx, node);
       }},
      {"ai.onnx:Slice",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         tensor::ComputeShapeSlice(ctx, node);
       }},
      {"ai.onnx:Split",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeSplit(ctx, node);
       }},
      {"ai.onnx:Tile",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeTile(ctx, node);
       }},
      {"ai.onnx:Squeeze",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeSqueeze(ctx, node);
       }},
      {"ai.onnx:Transpose",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeTranspose(ctx, node);
       }},
      {"ai.onnx:Trilu",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeTrilu(ctx, node);
       }},
      {"ai.onnx:Unsqueeze",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeUnsqueeze(ctx, node);
       }},
      {"ai.onnx:NonZero",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeNonZero(ctx, node);
       }},
      {"ai.onnx:Shape",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeShape(ctx, node);
       }},
      {"ai.onnx:RoiAlign",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 3);
        nn::ComputeShapeRoiAlign(ctx, node, node.input(0).as_string().c_str(),
                                 node.input(1).as_string().c_str(),
                                  node.input(2).as_string().c_str());
       }},
      {"ai.onnx:RNN",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         const std::string x_name = node.input(0).as_string();
         const std::string r_name = node.input(2).as_string();
         nn::ComputeShapeRNN(ctx, node, x_name.c_str(), r_name.empty() ? nullptr : r_name.c_str());
       }},
      {"ai.onnx:GRU",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         const std::string x_name = node.input(0).as_string();
         const std::string r_name = node.input(2).as_string();
         nn::ComputeShapeRNN(ctx, node, x_name.c_str(), r_name.empty() ? nullptr : r_name.c_str());
       }},
      {"ai.onnx:LSTM",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         const std::string x_name = node.input(0).as_string();
         const std::string r_name = node.input(2).as_string();
         nn::ComputeShapeRNN(ctx, node, x_name.c_str(), r_name.empty() ? nullptr : r_name.c_str());
       }},
      {"ai.onnx:SequenceConstruct",
       [](ShapesContext &ctx, const NodeProto &node) {
         sequence::ComputeShapeSequenceConstruct(ctx, node);
       }},
      {"ai.onnx:SequenceLength",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         sequence::ComputeShapeSequenceLength(ctx, node);
       }},
      {"ai.onnx:SequenceErase",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         sequence::ComputeShapeSequenceErase(ctx, node);
       }},
      {"ai.onnx:SequenceAt",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         sequence::ComputeShapeSequenceAt(ctx, node);
       }},
      {"ai.onnx:SequenceInsert",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         sequence::ComputeShapeSequenceInsert(ctx, node);
       }},
      {"ai.onnx:SequenceMap",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         sequence::ComputeShapeSequenceMap(ctx, node);
       }},
      {"ai.onnx:ConcatFromSequence",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         sequence::ComputeShapeConcatFromSequence(ctx, node);
       }},
      {"ai.onnx:Sigmoid",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSigmoid(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Softmax",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeSoftmax(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:StringConcat",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         text::ComputeShapeStringConcat(ctx, node, node.input(0).as_string().c_str(),
                                        node.input(1).as_string().c_str());
       }},
      {"ai.onnx:StringSplit",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         text::ComputeShapeStringSplit(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:StringNormalizer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         text::ComputeShapeStringNormalizer(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:RegexFullMatch",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         text::ComputeShapeRegexFullMatch(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:Binarizer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeBinarizer(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:CastMap",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeCastMap(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:CategoryMapper",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeCategoryMapper(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:DictVectorizer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeDictVectorizer(ctx, node,
                                                   node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:FeatureVectorizer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         std::vector<std::string> inputs;
         inputs.reserve(static_cast<size_t>(node.input_size()));
         for (int i = 0; i < node.input_size(); ++i) {
           inputs.emplace_back(node.input(i).as_string());
         }
         traditionalml::ComputeShapeFeatureVectorizer(ctx, node, inputs);
       }},
      {"ai.onnx.ml:Imputer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeImputer(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:ArrayFeatureExtractor",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         traditionalml::ComputeShapeArrayFeatureExtractor(
             ctx, node, node.input(0).as_string().c_str(), node.input(1).as_string().c_str());
       }},
      {"ai.onnx.ml:LabelEncoder",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeLabelEncoder(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:LinearClassifier",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeLinearClassifier(ctx, node,
                                                    node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:LinearRegressor",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeLinearRegressor(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:Normalizer",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeNormalizer(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:OneHotEncoder",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeOneHotEncoder(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:Scaler",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeScaler(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:SVMClassifier",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeSVMClassifier(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:SVMRegressor",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeSVMRegressor(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:TreeEnsemble",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeTreeEnsemble(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:TreeEnsembleClassifier",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeTreeEnsembleClassifier(ctx, node,
                                                         node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:TreeEnsembleRegressor",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeTreeEnsembleRegressor(ctx, node,
                                                        node.input(0).as_string().c_str());
       }},
      {"ai.onnx.ml:ZipMap",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 1);
        traditionalml::ComputeShapeZipMap(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.preview:FlexAttention",
       [](ShapesContext &ctx, const NodeProto &node) {
        RequireInputs(node, 3);
        preview::ComputeShapeFlexAttention(ctx, node, node.input(0).as_string().c_str(),
                                           node.input(1).as_string().c_str(),
                                            node.input(2).as_string().c_str());
       }},
      {"ai.onnx.preview.training:Adagrad",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 5);
         training::ComputeShapeAdagrad(ctx, node);
       }},
      {"ai.onnx.preview.training:Adam",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 6);
         training::ComputeShapeAdam(ctx, node);
       }},
      {"ai.onnx.preview.training:Momentum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 5);
         training::ComputeShapeMomentum(ctx, node);
       }},
  };
  return table;
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
