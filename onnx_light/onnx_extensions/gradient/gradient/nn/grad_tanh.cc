// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradTanh(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 1 || inputs[0].empty())
    return true;
  const std::string &A = inputs[0];
  if (node.output().empty() || node.output()[0].empty())
    return true;
  const std::string C = node.output()[0];

  // dA = dC * (1 - C^2): use dA = dC - dC * C * C to avoid a constant node.
  std::string C2 = NewGradName("C2", counter);
  func.add_node("Mul", {C, C}, {C2});
  std::string ones = NewGradName("ones", counter);
  {
    NodeProto &cst = func.add_node("Constant", {}, {ones});
    AttributeProto *attr = cst.add_attribute();
    attr->set_name("value");
    attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto &t = attr->ref_t();
    t.set_data_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));
    t.ref_float_data().push_back(1.0f);
  }
  std::string ones_like = NewGradName("ones_like", counter);
  func.add_node("CastLike", {ones, C2}, {ones_like});
  std::string one_minus_C2 = NewGradName("one_minus_C2", counter);
  func.add_node("Sub", {ones_like, C2}, {one_minus_C2});
  std::string dA = NewGradName("dA", counter);
  func.add_node("Mul", {output_grad, one_minus_C2}, {dA});
  AccumulateGrad(dA, grad_accum[A], counter, func);
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
