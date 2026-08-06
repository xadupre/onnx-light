// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/math/include_math_grads.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"
#include "onnx_extensions/gradient/gradient/reduction/include_reduction_grads.h"
#include "onnx_extensions/gradient/gradient/tensor/include_tensor_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

const GradRegistry &DefaultGradRegistry() {
  static const GradRegistry kRegistry = [] {
    GradRegistry r;
    RegisterGradientFunction("", "Add", GradAdd, r);
    RegisterGradientFunction("", "Conv", GradConv, r);
    RegisterGradientFunction("", "Div", GradDiv, r);
    RegisterGradientFunction("", "BatchNormalization", GradBatchNormalization, r);
    RegisterGradientFunction("", "Gemm", GradGemm, r);
    RegisterGradientFunction("", "GroupNormalization", GradGroupNormalization, r);
    RegisterGradientFunction("", "Identity", GradIdentity, r);
    RegisterGradientFunction("", "InstanceNormalization", GradInstanceNormalization, r);
    RegisterGradientFunction("", "LayerNormalization", GradLayerNormalization, r);
    RegisterGradientFunction("", "LpNormalization", GradLpNormalization, r);
    RegisterGradientFunction("", "MatMul", GradMatMul, r);
    RegisterGradientFunction("", "MeanVarianceNormalization", GradMeanVarianceNormalization, r);
    RegisterGradientFunction("", "Mul", GradMul, r);
    RegisterGradientFunction("", "Neg", GradNeg, r);
    RegisterGradientFunction("", "ReduceMean", GradReduceMean, r);
    RegisterGradientFunction("", "ReduceSum", GradReduceSum, r);
    RegisterGradientFunction("", "Relu", GradRelu, r);
    RegisterGradientFunction("", "Reshape", GradReshape, r);
    RegisterGradientFunction("", "RMSNormalization", GradRMSNormalization, r);
    RegisterGradientFunction("", "Sigmoid", GradSigmoid, r);
    RegisterGradientFunction("", "Sub", GradSub, r);
    RegisterGradientFunction("", "Tanh", GradTanh, r);
    RegisterGradientFunction("", "Transpose", GradTranspose, r);
    return r;
  }();
  return kRegistry;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
