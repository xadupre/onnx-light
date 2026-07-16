// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

/**
 * Applies the backward rule for the Gemm operator.
 *
 * C = alpha * A @ B + beta * bias  (simplified: transA=0, transB=0)
 * dA = dC @ B^T,  dB = A^T @ dC,  dbias = ReduceSum(dC, axis=0)
 */
bool GradGemm(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
