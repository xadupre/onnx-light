// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_dispatcher.h"
#include "onnx_gradient/gradient/grad_add.h"
#include "onnx_gradient/gradient/grad_div.h"
#include "onnx_gradient/gradient/grad_gemm.h"
#include "onnx_gradient/gradient/grad_identity.h"
#include "onnx_gradient/gradient/grad_matmul.h"
#include "onnx_gradient/gradient/grad_mul.h"
#include "onnx_gradient/gradient/grad_neg.h"
#include "onnx_gradient/gradient/grad_reducemean.h"
#include "onnx_gradient/gradient/grad_reducesum.h"
#include "onnx_gradient/gradient/grad_relu.h"
#include "onnx_gradient/gradient/grad_reshape.h"
#include "onnx_gradient/gradient/grad_sigmoid.h"
#include "onnx_gradient/gradient/grad_sub.h"
#include "onnx_gradient/gradient/grad_tanh.h"
#include "onnx_gradient/gradient/grad_transpose.h"

#include <sstream>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func) {
  // Find the output gradient: take the first output that has a gradient.
  std::string output_grad;
  for (const auto &out : node.output()) {
    auto it = grad_table.find(out.as_string());
    if (it != grad_table.end()) {
      output_grad = it->second;
      break;
    }
  }
  if (output_grad.empty())
    return; // no gradient flows through this node

  const std::string op_type = node.op_type().as_string();

  if (op_type == "Add") {
    GradAdd(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Div") {
    GradDiv(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Gemm") {
    GradGemm(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Identity") {
    GradIdentity(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "MatMul") {
    GradMatMul(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Mul") {
    GradMul(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Neg") {
    GradNeg(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "ReduceMean") {
    GradReduceMean(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "ReduceSum") {
    GradReduceSum(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Relu") {
    GradRelu(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Reshape") {
    GradReshape(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Sigmoid") {
    GradSigmoid(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Sub") {
    GradSub(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Tanh") {
    GradTanh(node, output_grad, grad_accum, counter, func);
  } else if (op_type == "Transpose") {
    GradTranspose(node, output_grad, grad_accum, counter, func);
  } else {
    std::ostringstream oss;
    oss << "onnx_gradient: unsupported op_type '" << op_type << "'";
    throw std::runtime_error(oss.str());
  }
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
