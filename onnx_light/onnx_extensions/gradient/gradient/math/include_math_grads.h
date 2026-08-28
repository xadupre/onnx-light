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
 *
 * @code
 * A --.
 *     +--> [Add] --> C
 * B --'
 *
 * dC --> dA
 * dC --> dB
 * @endcode
 */
bool GradAdd(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Div operator.
 *
 * C = A / B  →  dA = dC / B,  dB = -dC * A / B^2.
 *
 * @code
 * A --.
 *     +--> [Div] --> C
 * B --'
 *
 * dC, B ------> [Div] --> dA
 * dC, A --> [Mul] --> [Div(B^2)] --> [Neg] --> dB
 * @endcode
 */
bool GradDiv(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Gemm operator.
 *
 * C = alpha * A @ B + beta * bias  (simplified: transA=0, transB=0)
 * dA = dC @ B^T,  dB = A^T @ dC,  dbias = ReduceSum(dC, axis=0).
 *
 * @code
 * A ----.
 * B ----+--> [Gemm] --> C
 * bias -'
 *
 * dC, B^T --> [MatMul] ------> dA
 * A^T, dC --> [MatMul] ------> dB
 * dC ------> [ReduceSum] -----> dbias
 * @endcode
 */
bool GradGemm(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Applies the backward rule for the MatMul operator.
 *
 * C = A @ B  →  dA = dC @ B^T,  dB = A^T @ dC.
 *
 * @code
 * A --.
 *     +--> [MatMul] --> C
 * B --'
 *
 * dC, B^T --> [MatMul] --> dA
 * A^T, dC --> [MatMul] --> dB
 * @endcode
 */
bool GradMatMul(const NodeProto &node, const std::string &output_grad,
                std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                FunctionProto &func);

/**
 * Applies the backward rule for the Mul operator.
 *
 * C = A * B  →  dA = dC * B,  dB = dC * A.
 *
 * @code
 * A --.
 *     +--> [Mul] --> C
 * B --'
 *
 * dC, B --> [Mul] --> dA
 * dC, A --> [Mul] --> dB
 * @endcode
 */
bool GradMul(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Neg operator.
 *
 * C = -A  →  dA = -dC.
 *
 * @code
 * A --> [Neg] --> C
 *
 * dC --> [Neg] --> dA
 * @endcode
 */
bool GradNeg(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

/**
 * Applies the backward rule for the Sub operator.
 *
 * C = A - B  →  dA = dC,  dB = -dC.
 *
 * @code
 * A --.
 *     +--> [Sub] --> C
 * B --'
 *
 * dC ---------> dA
 * dC --> [Neg] --> dB
 * @endcode
 */
bool GradSub(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
