// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"
#include "onnx_proto/onnx_helper.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

namespace {

int64_t GetIntAttributeOrDefault(const NodeProto &node, const char *name, int64_t value) {
  const AttributeProto *attr = ONNX_LIGHT_NAMESPACE::FindAttribute(node, name);
  if (attr == nullptr) {
    return value;
  }
  return attr->i();
}

float GetFloatAttributeOrDefault(const NodeProto &node, const char *name, float value) {
  const AttributeProto *attr = ONNX_LIGHT_NAMESPACE::FindAttribute(node, name);
  if (attr == nullptr) {
    return value;
  }
  return attr->f();
}

std::vector<int64_t> GetIntsAttributeOrDefault(const NodeProto &node, const char *name,
                                               std::vector<int64_t> value) {
  const AttributeProto *attr = ONNX_LIGHT_NAMESPACE::FindAttribute(node, name);
  if (attr == nullptr) {
    return value;
  }
  return std::vector<int64_t>(attr->ints().begin(), attr->ints().end());
}

std::string AddFloatConstant(FunctionProto &func, int &counter, const char *prefix, float value) {
  const std::string name = NewGradName(prefix, counter);
  NodeProto &cst = func.add_node("Constant", {}, {name});
  AttributeProto *attr = cst.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &t = attr->ref_t();
  t.set_data_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));
  t.ref_float_data().push_back(value);
  return name;
}

std::string AddInt64VectorConstant(FunctionProto &func, int &counter, const char *prefix,
                                   const std::vector<int64_t> &values) {
  const std::string name = NewGradName(prefix, counter);
  NodeProto &cst = func.add_node("Constant", {}, {name});
  AttributeProto *attr = cst.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &t = attr->ref_t();
  t.set_data_type(static_cast<int32_t>(TensorProto::DataType::INT64));
  t.ref_dims().push_back(static_cast<int64_t>(values.size()));
  for (int64_t value : values) {
    t.ref_int64_data().push_back(value);
  }
  return name;
}

std::string AddBinaryNode(FunctionProto &func, int &counter, const char *op_type,
                          const std::string &a, const std::string &b, const char *prefix) {
  const std::string name = NewGradName(prefix, counter);
  func.add_node(op_type, {a, b}, {name});
  return name;
}

std::string AddUnaryNode(FunctionProto &func, int &counter, const char *op_type,
                         const std::string &x, const char *prefix) {
  const std::string name = NewGradName(prefix, counter);
  func.add_node(op_type, {x}, {name});
  return name;
}

std::string AddReduceNode(FunctionProto &func, int &counter, const char *op_type,
                          const std::string &x, const std::vector<int64_t> &axes, int64_t keepdims,
                          const char *prefix) {
  const std::string name = NewGradName(prefix, counter);
  NodeProto &node = func.add_node(op_type, {x}, {name});
  AddAttribute(node, "axes", axes);
  AddAttribute(node, "keepdims", keepdims);
  return name;
}

std::string AddReshapeToShape(FunctionProto &func, int &counter, const std::string &x,
                              const std::string &like, const char *prefix) {
  const std::string shape = NewGradName("shape", counter);
  func.add_node("Shape", {like}, {shape});
  const std::string name = NewGradName(prefix, counter);
  func.add_node("Reshape", {x, shape}, {name});
  return name;
}

std::string AddReshapeWithConstShape(FunctionProto &func, int &counter, const std::string &x,
                                     const std::vector<int64_t> &shape, const char *prefix) {
  const std::string shape_name = AddInt64VectorConstant(func, counter, "shape", shape);
  const std::string name = NewGradName(prefix, counter);
  func.add_node("Reshape", {x, shape_name}, {name});
  return name;
}

std::string AddFlatten(FunctionProto &func, int &counter, const std::string &x, int64_t axis,
                       const char *prefix) {
  const std::string name = NewGradName(prefix, counter);
  NodeProto &node = func.add_node("Flatten", {x}, {name});
  AddAttribute(node, "axis", axis);
  return name;
}

std::string AddEpsilonLike(FunctionProto &func, int &counter, const std::string &like,
                           float epsilon) {
  const std::string eps = AddFloatConstant(func, counter, "eps", epsilon);
  return AddBinaryNode(func, counter, "CastLike", eps, like, "eps_like");
}

std::string AddNormalizationDx(FunctionProto &func, int &counter, const std::string &x,
                               const std::string &scaled_grad, const std::vector<int64_t> &axes,
                               float epsilon, const char *prefix = "dx") {
  const std::string mean = AddReduceNode(func, counter, "ReduceMean", x, axes, 1, "mean");
  const std::string centered = AddBinaryNode(func, counter, "Sub", x, mean, "centered");
  const std::string centered_sq =
      AddBinaryNode(func, counter, "Mul", centered, centered, "centered_sq");
  const std::string var = AddReduceNode(func, counter, "ReduceMean", centered_sq, axes, 1, "var");
  const std::string eps_like = AddEpsilonLike(func, counter, var, epsilon);
  const std::string var_eps = AddBinaryNode(func, counter, "Add", var, eps_like, "var_eps");
  const std::string stddev = AddUnaryNode(func, counter, "Sqrt", var_eps, "stddev");
  const std::string inv_std = AddUnaryNode(func, counter, "Reciprocal", stddev, "inv_std");
  const std::string normalized =
      AddBinaryNode(func, counter, "Mul", centered, inv_std, "normalized");
  const std::string mean_g =
      AddReduceNode(func, counter, "ReduceMean", scaled_grad, axes, 1, "mean_g");
  const std::string gx = AddBinaryNode(func, counter, "Mul", scaled_grad, normalized, "gx");
  const std::string mean_gx = AddReduceNode(func, counter, "ReduceMean", gx, axes, 1, "mean_gx");
  const std::string corr = AddBinaryNode(func, counter, "Mul", normalized, mean_gx, "corr");
  const std::string centered_grad =
      AddBinaryNode(func, counter, "Sub", scaled_grad, mean_g, "centered_g");
  const std::string inner = AddBinaryNode(func, counter, "Sub", centered_grad, corr, "inner");
  return AddBinaryNode(func, counter, "Mul", inv_std, inner, prefix);
}

std::string AddNormalizationValue(FunctionProto &func, int &counter, const std::string &x,
                                  const std::vector<int64_t> &axes, float epsilon,
                                  const char *prefix = "normalized") {
  const std::string mean = AddReduceNode(func, counter, "ReduceMean", x, axes, 1, "mean");
  const std::string centered = AddBinaryNode(func, counter, "Sub", x, mean, "centered");
  const std::string centered_sq =
      AddBinaryNode(func, counter, "Mul", centered, centered, "centered_sq");
  const std::string var = AddReduceNode(func, counter, "ReduceMean", centered_sq, axes, 1, "var");
  const std::string eps_like = AddEpsilonLike(func, counter, var, epsilon);
  const std::string var_eps = AddBinaryNode(func, counter, "Add", var, eps_like, "var_eps");
  const std::string stddev = AddUnaryNode(func, counter, "Sqrt", var_eps, "stddev");
  const std::string inv_std = AddUnaryNode(func, counter, "Reciprocal", stddev, "inv_std");
  return AddBinaryNode(func, counter, "Mul", centered, inv_std, prefix);
}

} // namespace

bool GradBatchNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 5 || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::string &scale = inputs[1];
  const std::string &bias = inputs[2];
  const std::string &input_mean = inputs[3];
  const std::string &input_var = inputs[4];
  const float epsilon = GetFloatAttributeOrDefault(node, "epsilon", 1e-5f);
  const bool training_mode = GetIntAttributeOrDefault(node, "training_mode", 0) != 0;

  const std::string x3 = AddReshapeWithConstShape(func, counter, x, {0, 0, -1}, "x3");
  const std::string dy3 = AddReshapeWithConstShape(func, counter, output_grad, {0, 0, -1}, "dy3");
  const std::string scale3 = AddReshapeWithConstShape(func, counter, scale, {1, 0, 1}, "scale3");

  std::string dx3;
  std::string normalized3;

  if (training_mode) {
    const std::string scaled_grad = AddBinaryNode(func, counter, "Mul", dy3, scale3, "scaled_grad");
    dx3 = AddNormalizationDx(func, counter, x3, scaled_grad, {2}, epsilon, "dx3");
    normalized3 = AddNormalizationValue(func, counter, x3, {2}, epsilon, "normalized3");
  } else {
    const std::string mean3 =
        AddReshapeWithConstShape(func, counter, input_mean, {1, 0, 1}, "mean3");
    const std::string var3 = AddReshapeWithConstShape(func, counter, input_var, {1, 0, 1}, "var3");
    const std::string centered = AddBinaryNode(func, counter, "Sub", x3, mean3, "centered");
    const std::string eps_like = AddEpsilonLike(func, counter, var3, epsilon);
    const std::string var_eps = AddBinaryNode(func, counter, "Add", var3, eps_like, "var_eps");
    const std::string stddev = AddUnaryNode(func, counter, "Sqrt", var_eps, "stddev");
    const std::string inv_std = AddUnaryNode(func, counter, "Reciprocal", stddev, "inv_std");
    normalized3 = AddBinaryNode(func, counter, "Mul", centered, inv_std, "normalized3");
    const std::string scale_inv = AddBinaryNode(func, counter, "Mul", scale3, inv_std, "scale_inv");
    dx3 = AddBinaryNode(func, counter, "Mul", dy3, scale_inv, "dx3");

    if (!input_mean.empty()) {
      const std::string neg_scale_inv =
          AddUnaryNode(func, counter, "Neg", scale_inv, "neg_scale_inv");
      const std::string dmean_term =
          AddBinaryNode(func, counter, "Mul", dy3, neg_scale_inv, "dmean_term");
      const std::string dmean =
          AddReduceNode(func, counter, "ReduceSum", dmean_term, {0, 2}, 0, "dmean");
      AccumulateGrad(dmean, grad_accum[input_mean], counter, func);
    }

    if (!input_var.empty()) {
      const std::string dy_scale = AddBinaryNode(func, counter, "Mul", dy3, scale3, "dy_scale");
      const std::string inv_sq = AddBinaryNode(func, counter, "Mul", inv_std, inv_std, "inv_sq");
      const std::string inv_cu = AddBinaryNode(func, counter, "Mul", inv_sq, inv_std, "inv_cu");
      const std::string var_term0 =
          AddBinaryNode(func, counter, "Mul", dy_scale, centered, "var_term0");
      const std::string var_term1 =
          AddBinaryNode(func, counter, "Mul", var_term0, inv_cu, "var_term1");
      const std::string minus_half = AddFloatConstant(func, counter, "minus_half", -0.5f);
      const std::string minus_half_like =
          AddBinaryNode(func, counter, "CastLike", minus_half, var_term1, "minus_half_like");
      const std::string var_term =
          AddBinaryNode(func, counter, "Mul", var_term1, minus_half_like, "var_term");
      const std::string dvar =
          AddReduceNode(func, counter, "ReduceSum", var_term, {0, 2}, 0, "dvar");
      AccumulateGrad(dvar, grad_accum[input_var], counter, func);
    }
  }

  if (!x.empty()) {
    const std::string dx = AddReshapeToShape(func, counter, dx3, x, "dx");
    AccumulateGrad(dx, grad_accum[x], counter, func);
  }
  if (!scale.empty()) {
    const std::string scale_term =
        AddBinaryNode(func, counter, "Mul", dy3, normalized3, "dscale_term");
    const std::string dscale =
        AddReduceNode(func, counter, "ReduceSum", scale_term, {0, 2}, 0, "dscale");
    AccumulateGrad(dscale, grad_accum[scale], counter, func);
  }
  if (!bias.empty()) {
    const std::string dbias = AddReduceNode(func, counter, "ReduceSum", dy3, {0, 2}, 0, "dbias");
    AccumulateGrad(dbias, grad_accum[bias], counter, func);
  }
  return true;
}

bool GradGroupNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 3 || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::string &scale = inputs[1];
  const std::string &bias = inputs[2];
  const int64_t num_groups = GetIntAttributeOrDefault(node, "num_groups", 1);
  const float epsilon = GetFloatAttributeOrDefault(node, "epsilon", 1e-5f);

  const std::string x3 = AddReshapeWithConstShape(func, counter, x, {0, 0, -1}, "x3");
  const std::string dy3 = AddReshapeWithConstShape(func, counter, output_grad, {0, 0, -1}, "dy3");
  const std::string scale3 = AddReshapeWithConstShape(func, counter, scale, {1, 0, 1}, "scale3");
  const std::string scaled_grad = AddBinaryNode(func, counter, "Mul", dy3, scale3, "scaled_grad");
  const std::string xg = AddReshapeWithConstShape(func, counter, x, {0, num_groups, -1}, "xg");
  const std::string gg =
      AddReshapeWithConstShape(func, counter, scaled_grad, {0, num_groups, -1}, "gg");
  const std::string dxg = AddNormalizationDx(func, counter, xg, gg, {2}, epsilon, "dxg");
  const std::string dx = AddReshapeToShape(func, counter, dxg, x, "dx");
  AccumulateGrad(dx, grad_accum[x], counter, func);

  const std::string normalized_g =
      AddNormalizationValue(func, counter, xg, {2}, epsilon, "normalized_g");
  const std::string normalized3 = AddReshapeToShape(func, counter, normalized_g, x3, "normalized3");
  const std::string dscale_term =
      AddBinaryNode(func, counter, "Mul", dy3, normalized3, "dscale_term");
  const std::string dscale =
      AddReduceNode(func, counter, "ReduceSum", dscale_term, {0, 2}, 0, "dscale");
  AccumulateGrad(dscale, grad_accum[scale], counter, func);

  const std::string dbias = AddReduceNode(func, counter, "ReduceSum", dy3, {0, 2}, 0, "dbias");
  AccumulateGrad(dbias, grad_accum[bias], counter, func);
  return true;
}

bool GradInstanceNormalization(const NodeProto &node, const std::string &output_grad,
                               std::unordered_map<std::string, std::string> &grad_accum,
                               int &counter, FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 3 || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::string &scale = inputs[1];
  const std::string &bias = inputs[2];
  const float epsilon = GetFloatAttributeOrDefault(node, "epsilon", 1e-5f);

  const std::string x3 = AddReshapeWithConstShape(func, counter, x, {0, 0, -1}, "x3");
  const std::string dy3 = AddReshapeWithConstShape(func, counter, output_grad, {0, 0, -1}, "dy3");
  const std::string scale3 = AddReshapeWithConstShape(func, counter, scale, {1, 0, 1}, "scale3");
  const std::string scaled_grad = AddBinaryNode(func, counter, "Mul", dy3, scale3, "scaled_grad");
  const std::string dx3 = AddNormalizationDx(func, counter, x3, scaled_grad, {2}, epsilon, "dx3");
  const std::string dx = AddReshapeToShape(func, counter, dx3, x, "dx");
  AccumulateGrad(dx, grad_accum[x], counter, func);

  const std::string normalized3 =
      AddNormalizationValue(func, counter, x3, {2}, epsilon, "normalized3");
  const std::string dscale_term =
      AddBinaryNode(func, counter, "Mul", dy3, normalized3, "dscale_term");
  const std::string dscale =
      AddReduceNode(func, counter, "ReduceSum", dscale_term, {0, 2}, 0, "dscale");
  AccumulateGrad(dscale, grad_accum[scale], counter, func);

  const std::string dbias = AddReduceNode(func, counter, "ReduceSum", dy3, {0, 2}, 0, "dbias");
  AccumulateGrad(dbias, grad_accum[bias], counter, func);
  return true;
}

bool GradLayerNormalization(const NodeProto &node, const std::string &output_grad,
                            std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                            FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 2 || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::string &scale = inputs[1];
  const bool has_bias = inputs.size() >= 3 && !inputs[2].empty();
  const int64_t axis = GetIntAttributeOrDefault(node, "axis", -1);
  const float epsilon = GetFloatAttributeOrDefault(node, "epsilon", 1e-5f);

  const std::string x2 = AddFlatten(func, counter, x, axis, "x2");
  const std::string dy2 = AddFlatten(func, counter, output_grad, axis, "dy2");
  const std::string scale2 = AddFlatten(func, counter, scale, 0, "scale2");
  const std::string scaled_grad = AddBinaryNode(func, counter, "Mul", dy2, scale2, "scaled_grad");
  const std::string dx2 = AddNormalizationDx(func, counter, x2, scaled_grad, {1}, epsilon, "dx2");
  const std::string dx = AddReshapeToShape(func, counter, dx2, x, "dx");
  AccumulateGrad(dx, grad_accum[x], counter, func);

  const std::string normalized2 =
      AddNormalizationValue(func, counter, x2, {1}, epsilon, "normalized2");
  const std::string dscale_term =
      AddBinaryNode(func, counter, "Mul", dy2, normalized2, "dscale_term");
  const std::string dscale_flat =
      AddReduceNode(func, counter, "ReduceSum", dscale_term, {0}, 0, "dscale_flat");
  const std::string dscale = AddReshapeToShape(func, counter, dscale_flat, scale, "dscale");
  AccumulateGrad(dscale, grad_accum[scale], counter, func);

  if (has_bias) {
    const std::string dbias_flat =
        AddReduceNode(func, counter, "ReduceSum", dy2, {0}, 0, "dbias_flat");
    const std::string dbias = AddReshapeToShape(func, counter, dbias_flat, inputs[2], "dbias");
    AccumulateGrad(dbias, grad_accum[inputs[2]], counter, func);
  }
  return true;
}

bool GradLpNormalization(const NodeProto &node, const std::string &output_grad,
                         std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                         FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.empty() || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const int64_t axis = GetIntAttributeOrDefault(node, "axis", -1);
  const int64_t p = GetIntAttributeOrDefault(node, "p", 2);

  std::string dx;
  if (p == 1) {
    const std::string abs_x = AddUnaryNode(func, counter, "Abs", x, "abs_x");
    const std::string norm = AddReduceNode(func, counter, "ReduceSum", abs_x, {axis}, 1, "norm");
    const std::string grad_div = AddBinaryNode(func, counter, "Div", output_grad, norm, "grad_div");
    const std::string dyx = AddBinaryNode(func, counter, "Mul", output_grad, x, "dyx");
    const std::string sum_dyx =
        AddReduceNode(func, counter, "ReduceSum", dyx, {axis}, 1, "sum_dyx");
    const std::string norm_sq = AddBinaryNode(func, counter, "Mul", norm, norm, "norm_sq");
    const std::string coeff = AddBinaryNode(func, counter, "Div", sum_dyx, norm_sq, "coeff");
    const std::string sign_x = AddUnaryNode(func, counter, "Sign", x, "sign_x");
    const std::string corr = AddBinaryNode(func, counter, "Mul", sign_x, coeff, "corr");
    dx = AddBinaryNode(func, counter, "Sub", grad_div, corr, "dx");
  } else {
    const std::string xx = AddBinaryNode(func, counter, "Mul", x, x, "xx");
    const std::string norm_sq = AddReduceNode(func, counter, "ReduceSum", xx, {axis}, 1, "norm_sq");
    const std::string norm = AddUnaryNode(func, counter, "Sqrt", norm_sq, "norm");
    const std::string grad_div = AddBinaryNode(func, counter, "Div", output_grad, norm, "grad_div");
    std::string y;
    if (!node.output().empty() && !node.output()[0].empty()) {
      y = node.output()[0];
    } else {
      y = AddBinaryNode(func, counter, "Div", x, norm, "y");
    }
    const std::string dyy = AddBinaryNode(func, counter, "Mul", output_grad, y, "dyy");
    const std::string dot = AddReduceNode(func, counter, "ReduceSum", dyy, {axis}, 1, "dot");
    const std::string corr = AddBinaryNode(func, counter, "Mul", y, dot, "corr");
    dx = AddBinaryNode(func, counter, "Sub", grad_div, corr, "dx");
  }
  AccumulateGrad(dx, grad_accum[x], counter, func);
  return true;
}

bool GradMeanVarianceNormalization(const NodeProto &node, const std::string &output_grad,
                                   std::unordered_map<std::string, std::string> &grad_accum,
                                   int &counter, FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.empty() || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::vector<int64_t> axes =
      GetIntsAttributeOrDefault(node, "axes", std::vector<int64_t>{0, 2, 3});
  const std::string dx = AddNormalizationDx(func, counter, x, output_grad, axes, 1e-9f, "dx");
  AccumulateGrad(dx, grad_accum[x], counter, func);
  return true;
}

bool GradRMSNormalization(const NodeProto &node, const std::string &output_grad,
                          std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                          FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 2 || inputs[0].empty()) {
    return true;
  }
  const std::string &x = inputs[0];
  const std::string &scale = inputs[1];
  const int64_t axis = GetIntAttributeOrDefault(node, "axis", -1);
  const float epsilon = GetFloatAttributeOrDefault(node, "epsilon", 1e-5f);

  const std::string x2 = AddFlatten(func, counter, x, axis, "x2");
  const std::string dy2 = AddFlatten(func, counter, output_grad, axis, "dy2");
  const std::string scale2 = AddFlatten(func, counter, scale, 0, "scale2");
  const std::string xx = AddBinaryNode(func, counter, "Mul", x2, x2, "xx");
  const std::string mean_sq = AddReduceNode(func, counter, "ReduceMean", xx, {1}, 1, "mean_sq");
  const std::string eps_like = AddEpsilonLike(func, counter, mean_sq, epsilon);
  const std::string mean_sq_eps =
      AddBinaryNode(func, counter, "Add", mean_sq, eps_like, "mean_sq_eps");
  const std::string rms = AddUnaryNode(func, counter, "Sqrt", mean_sq_eps, "rms");
  const std::string inv_rms = AddUnaryNode(func, counter, "Reciprocal", rms, "inv_rms");
  const std::string scaled_grad = AddBinaryNode(func, counter, "Mul", dy2, scale2, "scaled_grad");
  const std::string xg = AddBinaryNode(func, counter, "Mul", x2, scaled_grad, "xg");
  const std::string mean_xg = AddReduceNode(func, counter, "ReduceMean", xg, {1}, 1, "mean_xg");
  const std::string inv_sq = AddBinaryNode(func, counter, "Mul", inv_rms, inv_rms, "inv_sq");
  const std::string corr0 = AddBinaryNode(func, counter, "Mul", x2, mean_xg, "corr0");
  const std::string corr = AddBinaryNode(func, counter, "Mul", corr0, inv_sq, "corr");
  const std::string inner = AddBinaryNode(func, counter, "Sub", scaled_grad, corr, "inner");
  const std::string dx2 = AddBinaryNode(func, counter, "Mul", inv_rms, inner, "dx2");
  const std::string dx = AddReshapeToShape(func, counter, dx2, x, "dx");
  AccumulateGrad(dx, grad_accum[x], counter, func);

  const std::string normalized = AddBinaryNode(func, counter, "Mul", x2, inv_rms, "normalized");
  const std::string dscale_term =
      AddBinaryNode(func, counter, "Mul", dy2, normalized, "dscale_term");
  const std::string dscale_flat =
      AddReduceNode(func, counter, "ReduceSum", dscale_term, {0}, 0, "dscale_flat");
  const std::string dscale = AddReshapeToShape(func, counter, dscale_flat, scale, "dscale");
  AccumulateGrad(dscale, grad_accum[scale], counter, func);
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
