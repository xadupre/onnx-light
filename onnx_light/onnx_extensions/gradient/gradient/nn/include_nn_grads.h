// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

/**
 * Applies the backward rule for the Conv operator.
 *
 * Y = Conv(X, W, B)  →
 *   dX = ConvTranspose(dY, W, attrs),
 *   dW = Transpose(Conv(Transpose(Pad(X, pads)), Transpose(dY),
 *                       strides=dilations, dilations=strides)),
 *   dB = ReduceSum(dY, axes=[0, spatial…], keepdims=0).
 *
 * @code
 * X --.
 * W --+--> Conv --> Y
 * B --'
 *
 * dY, W --> ConvTranspose --> dX
 * X, dY --> Conv -----------> dW
 * dY ----> ReduceSum -------> dB
 * @endcode
 */
bool GradConv(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Applies the backward rule for the Relu operator.
 *
 * C = relu(A)  →  dA = dC * relu(sign(A)).
 *
 * @code
 * A --> Relu --> C
 *
 * A --> Sign --> Relu --.
 *                       +--> Mul --> dA
 * dC ------------------'
 * @endcode
 */
bool GradRelu(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Applies the backward rule for the Sigmoid operator.
 *
 * C = sigmoid(A)  →  dA = dC * C * (1 - C).
 *
 * @code
 * A --> Sigmoid --> C
 *
 * C --> 1 - C --.
 * C ------------+--> Mul --.
 * dC ---------------------+--> Mul --> dA
 * @endcode
 */
bool GradSigmoid(const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func);

/**
 * Applies the backward rule for the Tanh operator.
 *
 * C = tanh(A)  →  dA = dC * (1 - C^2).
 *
 * @code
 * A --> Tanh --> C
 *
 * C --> Mul --> 1 - C^2 --.
 * dC ---------------------+--> Mul --> dA
 * @endcode
 */
bool GradTanh(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Computes the backward rule for the BatchNormalization operator.
 *
 * @code
 * X, scale, bias, mean, variance --> BatchNormalization --> Y
 *
 * dY --> GradBatchNormalization --> dX, dscale, dbias, dmean, dvariance
 * @endcode
 */
bool GradBatchNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func);

/**
 * Computes the backward rule for the GroupNormalization operator.
 *
 * @code
 * X, scale, bias --> GroupNormalization --> Y
 *
 * dY --> GradGroupNormalization --> dX, dscale, dbias
 * @endcode
 */
bool GradGroupNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func);

/**
 * Computes the backward rule for the InstanceNormalization operator.
 *
 * @code
 * X, scale, bias --> InstanceNormalization --> Y
 *
 * dY --> GradInstanceNormalization --> dX, dscale, dbias
 * @endcode
 */
bool GradInstanceNormalization(const NodeProto &node, const std::string &output_grad,
                               std::unordered_map<std::string, std::string> &grad_accum,
                               int &counter, FunctionProto &func);

/**
 * Computes the backward rule for the LayerNormalization operator.
 *
 * @code
 * X, scale, bias --> LayerNormalization --> Y
 *
 * dY --> GradLayerNormalization --> dX, dscale, dbias
 * @endcode
 */
bool GradLayerNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func);

/**
 * Computes the backward rule for the LpNormalization operator.
 *
 * @code
 * X --> LpNormalization --> Y
 *
 * dY --> GradLpNormalization --> dX
 * @endcode
 */
bool GradLpNormalization(const NodeProto &node, const std::string &output_grad,
                         std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                         FunctionProto &func);

/**
 * Computes the backward rule for the MeanVarianceNormalization operator.
 *
 * @code
 * X --> MeanVarianceNormalization --> Y
 *
 * dY --> GradMeanVarianceNormalization --> dX
 * @endcode
 */
bool GradMeanVarianceNormalization(const NodeProto &node, const std::string &output_grad,
                                   std::unordered_map<std::string, std::string> &grad_accum,
                                   int &counter, FunctionProto &func);

/**
 * Computes the backward rule for the RMSNormalization operator.
 *
 * @code
 * X, scale --> RMSNormalization --> Y
 *
 * dY --> GradRMSNormalization --> dX, dscale
 * @endcode
 */
bool GradRMSNormalization(const NodeProto &node, const std::string &output_grad,
                          std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                          FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
