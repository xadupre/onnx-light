// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

/**
 * Applies the backward rule for the Add operator.
 *
 * C = A + B  →  dA = dC,  dB = dC.
 */
bool GradAdd(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Div operator.
 *
 * C = A / B  →  dA = dC / B,  dB = -dC * A / B^2.
 */
bool GradDiv(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Gemm operator.
 *
 * C = alpha * A @ B + beta * bias  (simplified: transA=0, transB=0)
 * dA = dC @ B^T,  dB = A^T @ dC,  dbias = ReduceSum(dC, axis=0).
 */
bool GradGemm(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Applies the backward rule for the MatMul operator.
 *
 * C = A @ B  →  dA = dC @ B^T,  dB = A^T @ dC.
 */
bool GradMatMul(const NodeProto &node, const std::string &output_grad,
                std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                FunctionProto &func);

/**
 * Applies the backward rule for the Mul operator.
 *
 * C = A * B  →  dA = dC * B,  dB = dC * A.
 */
bool GradMul(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Neg operator.
 *
 * C = -A  →  dA = -dC.
 */
bool GradNeg(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Sub operator.
 *
 * C = A - B  →  dA = dC,  dB = -dC.
 */
bool GradSub(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
