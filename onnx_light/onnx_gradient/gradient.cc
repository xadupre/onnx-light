// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient.h"
#include "onnx_proto/onnx_helper.h"
#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

namespace {

// Builds a mapping from each output name to the index of the node that
// produces it.
std::unordered_map<std::string, size_t>
BuildOutputToNodeIndex(const std::vector<NodeProto> &nodes) {
  std::unordered_map<std::string, size_t> output_map;
  output_map.reserve(nodes.size() * 2);
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (const auto &output : nodes[i].output()) {
      output_map[output.as_string()] = i;
    }
  }
  return output_map;
}

// Returns the indices of nodes that transitively contribute to @p target in
// topological order (producers before consumers).
std::vector<size_t> TopologicalOrder(const std::vector<NodeProto> &nodes, const std::string &target,
                                     const std::unordered_map<std::string, size_t> &output_map) {
  std::unordered_set<size_t> visited;
  std::vector<size_t> order;
  order.reserve(nodes.size());

  std::function<void(const std::string &)> visit = [&](const std::string &name) {
    auto it = output_map.find(name);
    if (it == output_map.end())
      return; // leaf input
    size_t idx = it->second;
    if (!visited.insert(idx).second)
      return; // already processed
    for (const auto &inp : nodes[idx].input()) {
      if (!inp.null() && !inp.empty())
        visit(inp.as_string());
    }
    order.push_back(idx);
  };

  visit(target);
  return order;
}

// Accumulates @p contrib_name into @p acc_name inside @p func; if @p acc_name
// is empty, sets it to @p contrib_name directly.
void AccumulateGrad(const std::string &contrib_name, std::string &acc_name, int &counter,
                    FunctionProto &func) {
  if (acc_name.empty()) {
    acc_name = contrib_name;
  } else {
    std::string new_acc = "grad_acc_" + std::to_string(counter++);
    func.add_node("Add", {acc_name, contrib_name}, {new_acc});
    acc_name = new_acc;
  }
}

// Applies the backward rule for a single forward @p node and accumulates the
// resulting input gradients into @p grad_table.  New nodes are appended to
// @p func.  @p counter is used to generate unique intermediate names.
void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func) {
  // Find the output gradient.  We only handle nodes with a single output (the
  // most common case); multi-output ops can be added later.
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

  auto input_name = [&](size_t i) -> std::string {
    const auto &inputs = node.input();
    if (i >= inputs.size())
      return {};
    return inputs[i].as_string();
  };

  auto new_name = [&](const std::string &prefix) -> std::string {
    return prefix + "_" + std::to_string(counter++);
  };

  if (op_type == "MatMul") {
    // C = A @ B  →  dA = dC @ B^T,  dB = A^T @ dC
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (A.empty() || B.empty())
      return;

    // dA = dC @ B^T
    std::string B_T = new_name("B_T");
    func.add_node("Transpose", {B}, {B_T});
    std::string dA = new_name("dA");
    func.add_node("MatMul", {output_grad, B_T}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);

    // dB = A^T @ dC
    std::string A_T = new_name("A_T");
    func.add_node("Transpose", {A}, {A_T});
    std::string dB = new_name("dB");
    func.add_node("MatMul", {A_T, output_grad}, {dB});
    AccumulateGrad(dB, grad_accum[B], counter, func);

  } else if (op_type == "Gemm") {
    // C = alpha * A @ B + beta * C_init  (attributes: transA, transB, alpha, beta)
    // Simplified to the default (transA=0, transB=0, alpha=1, beta=1) case.
    // dA = dC @ B^T,  dB = A^T @ dC
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (A.empty() || B.empty())
      return;

    std::string B_T = new_name("B_T");
    func.add_node("Transpose", {B}, {B_T});
    std::string dA = new_name("dA");
    func.add_node("MatMul", {output_grad, B_T}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);

    std::string A_T = new_name("A_T");
    func.add_node("Transpose", {A}, {A_T});
    std::string dB = new_name("dB");
    func.add_node("MatMul", {A_T, output_grad}, {dB});
    AccumulateGrad(dB, grad_accum[B], counter, func);

    // Gradient for optional bias C_init: sum of dC over the batch axis.
    std::string C_init = input_name(2);
    if (!C_init.empty()) {
      // Use ReduceSum over axis 0 to collapse the batch dimension.
      std::string dC_init = new_name("dC_init");
      NodeProto &rs = func.add_node("ReduceSum", {output_grad}, {dC_init});
      AddAttribute(rs, "axes", std::vector<int64_t>{0});
      AddAttribute(rs, "keepdims", int64_t{0});
      AccumulateGrad(dC_init, grad_accum[C_init], counter, func);
    }

  } else if (op_type == "Add") {
    // C = A + B  →  dA = dC,  dB = dC
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (!A.empty())
      AccumulateGrad(output_grad, grad_accum[A], counter, func);
    if (!B.empty())
      AccumulateGrad(output_grad, grad_accum[B], counter, func);

  } else if (op_type == "Sub") {
    // C = A - B  →  dA = dC,  dB = -dC
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (!A.empty())
      AccumulateGrad(output_grad, grad_accum[A], counter, func);
    if (!B.empty()) {
      std::string neg = new_name("neg_grad");
      func.add_node("Neg", {output_grad}, {neg});
      AccumulateGrad(neg, grad_accum[B], counter, func);
    }

  } else if (op_type == "Mul") {
    // C = A * B  →  dA = dC * B,  dB = dC * A
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (!A.empty()) {
      std::string dA = new_name("dA");
      func.add_node("Mul", {output_grad, B}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }
    if (!B.empty()) {
      std::string dB = new_name("dB");
      func.add_node("Mul", {output_grad, A}, {dB});
      AccumulateGrad(dB, grad_accum[B], counter, func);
    }

  } else if (op_type == "Div") {
    // C = A / B  →  dA = dC / B,  dB = -dC * A / B^2
    std::string A = input_name(0);
    std::string B = input_name(1);
    if (!A.empty()) {
      std::string dA = new_name("dA");
      func.add_node("Div", {output_grad, B}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }
    if (!B.empty()) {
      std::string B2 = new_name("B2");
      func.add_node("Mul", {B, B}, {B2});
      std::string A_div_B2 = new_name("A_div_B2");
      func.add_node("Div", {A, B2}, {A_div_B2});
      std::string neg_grad = new_name("neg_grad");
      func.add_node("Neg", {output_grad}, {neg_grad});
      std::string dB = new_name("dB");
      func.add_node("Mul", {neg_grad, A_div_B2}, {dB});
      AccumulateGrad(dB, grad_accum[B], counter, func);
    }

  } else if (op_type == "Neg") {
    // C = -A  →  dA = -dC
    std::string A = input_name(0);
    if (!A.empty()) {
      std::string neg = new_name("neg_grad");
      func.add_node("Neg", {output_grad}, {neg});
      AccumulateGrad(neg, grad_accum[A], counter, func);
    }

  } else if (op_type == "Identity") {
    std::string A = input_name(0);
    if (!A.empty())
      AccumulateGrad(output_grad, grad_accum[A], counter, func);

  } else if (op_type == "Relu") {
    // C = relu(A)  →  dA = dC * (A > 0)
    // Approximated as dA = dC * relu(sign(A)).
    std::string A = input_name(0);
    if (!A.empty()) {
      std::string sgn = new_name("relu_sign");
      func.add_node("Sign", {A}, {sgn});
      std::string mask = new_name("relu_mask");
      func.add_node("Relu", {sgn}, {mask});
      std::string dA = new_name("dA");
      func.add_node("Mul", {output_grad, mask}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "Sigmoid") {
    // C = sigmoid(A)  →  dA = dC * C * (1 - C)
    std::string A = input_name(0);
    std::string C = node.output()[0].as_string();
    if (!A.empty() && !C.empty()) {
      std::string one_minus_C = new_name("one_minus_C");
      // 1 - C: use Neg then Add, or simply Sub
      // Alternatively: C * (1 - C) = C - C*C
      std::string C2 = new_name("C2");
      func.add_node("Mul", {C, C}, {C2});
      std::string C_minus_C2 = new_name("C_minus_C2");
      func.add_node("Sub", {C, C2}, {C_minus_C2});
      std::string dA = new_name("dA");
      func.add_node("Mul", {output_grad, C_minus_C2}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "Tanh") {
    // C = tanh(A)  →  dA = dC * (1 - C^2) = dC * (C * (1 - C) + (1 - C) * C)
    // Simpler: 1 - C^2 = (1 - C)(1 + C) or dA = dC - dC * C^2
    // Use: dA = dC - dC * C * C  (avoids needing a constant-1 node)
    std::string A = input_name(0);
    std::string C = node.output()[0].as_string();
    if (!A.empty() && !C.empty()) {
      std::string C2 = new_name("C2");
      func.add_node("Mul", {C, C}, {C2});
      // Create Constant node with value=1.0f via direct attribute API.
      std::string ones = new_name("ones");
      {
        NodeProto &cst = func.add_node("Constant", {}, {ones});
        AttributeProto *attr = cst.add_attribute();
        attr->set_name("value");
        attr->set_type(AttributeProto::AttributeType::TENSOR);
        TensorProto &t = attr->ref_t();
        t.set_data_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));
        t.ref_float_data().push_back(1.0f);
      }
      std::string ones_like = new_name("ones_like");
      func.add_node("CastLike", {ones, C2}, {ones_like});
      std::string one_minus_C2 = new_name("one_minus_C2");
      func.add_node("Sub", {ones_like, C2}, {one_minus_C2});
      std::string dA = new_name("dA");
      func.add_node("Mul", {output_grad, one_minus_C2}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "ReduceSum") {
    // Y = ReduceSum(X, axes, keepdims)  →  dX = Expand(dY, Shape(X))
    // Simplified: broadcast dY back using Reshape to the original shape.
    std::string A = input_name(0);
    if (!A.empty()) {
      std::string shape_A = new_name("shape_A");
      func.add_node("Shape", {A}, {shape_A});
      std::string dA = new_name("dA");
      func.add_node("Expand", {output_grad, shape_A}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "ReduceMean") {
    // Y = ReduceMean(X)  →  dX = dY / N, broadcast to shape(X)
    // We use: dX = Expand(dY, Shape(X)) / Size(X) * Size(Y)
    // Simplified implementation: expand and divide by the reduction factor.
    std::string A = input_name(0);
    if (!A.empty()) {
      std::string shape_A = new_name("shape_A");
      func.add_node("Shape", {A}, {shape_A});
      std::string expanded = new_name("expanded");
      func.add_node("Expand", {output_grad, shape_A}, {expanded});
      // Compute the number of elements reduced (Size(X) / Size(Y))
      std::string size_X = new_name("size_X");
      func.add_node("Size", {A}, {size_X});
      // We represent the output to get its size; fall back to the "dy" input if
      // the output tensor is not available (it is an intermediate).
      const std::string &out_name = node.output()[0].as_string();
      std::string size_Y = new_name("size_Y");
      // Use CastLike to keep dtype consistent with the gradient tensor.
      std::string size_X_cast = new_name("size_X_cast");
      func.add_node("CastLike", {size_X, expanded}, {size_X_cast});
      // Workaround: compute dX = expanded / cast(size_X / size_Y)
      // Without size_Y available at runtime we use: divide by the total count
      // via Size(X) and the scalar shape of the output via the "dy" input.
      // For the common case (full reduction), divide expanded by Size(X).
      std::string dA = new_name("dA");
      // Use Div directly with the cast size.
      std::string size_out = new_name("size_out");
      // Compute a scalar containing the number of output elements.
      // Approximate: use the forward value's shape if available, else 1.
      // For ReduceMean over all axes the output is a scalar, so Size(Y) = 1.
      // We compute: n_elements = Size(X) / 1 = Size(X).
      // Then dX_i = dY / n_elements.
      func.add_node("Div", {expanded, size_X_cast}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "Reshape") {
    // Y = Reshape(X, shape)  →  dX = Reshape(dY, Shape(X))
    std::string A = input_name(0);
    if (!A.empty()) {
      std::string shape_A = new_name("shape_A");
      func.add_node("Shape", {A}, {shape_A});
      std::string dA = new_name("dA");
      func.add_node("Reshape", {output_grad, shape_A}, {dA});
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else if (op_type == "Transpose") {
    // Y = Transpose(X, perm)  →  dX = Transpose(dY, inverse_perm)
    std::string A = input_name(0);
    if (!A.empty()) {
      // Check if a permutation attribute was set.
      const AttributeProto *perm_attr = FindAttribute(node, "perm");
      std::string dA = new_name("dA");
      if (perm_attr != nullptr) {
        const auto &perm = perm_attr->ref_ints();
        // Compute inverse permutation.
        std::vector<int64_t> inv_perm(perm.size());
        for (size_t pi = 0; pi < perm.size(); ++pi) {
          inv_perm[static_cast<size_t>(perm[pi])] = static_cast<int64_t>(pi);
        }
        NodeProto &tp = func.add_node("Transpose", {output_grad}, {dA});
        AddAttribute(tp, "perm", inv_perm);
      } else {
        // No perm: reverse all dimensions (default ONNX behaviour).
        func.add_node("Transpose", {output_grad}, {dA});
      }
      AccumulateGrad(dA, grad_accum[A], counter, func);
    }

  } else {
    std::ostringstream oss;
    oss << "onnx_gradient: unsupported op_type '" << op_type << "'";
    throw std::runtime_error(oss.str());
  }
}

// Core algorithm shared by GradientOfNodes and GradientOfFunction.
FunctionProto ComputeGradient(const std::vector<NodeProto> &nodes,
                              const std::vector<std::string> &xs, const std::string &y,
                              const std::vector<std::string> &zs, const std::string &func_domain,
                              const std::string &func_name) {
  if (xs.empty())
    throw std::invalid_argument("onnx_gradient: xs must not be empty");
  if (y.empty())
    throw std::invalid_argument("onnx_gradient: y must not be empty");

  auto output_map = BuildOutputToNodeIndex(nodes);

  if (output_map.find(y) == output_map.end()) {
    std::ostringstream oss;
    oss << "onnx_gradient: output '" << y << "' is not produced by any node";
    throw std::invalid_argument(oss.str());
  }

  auto topo_order = TopologicalOrder(nodes, y, output_map);

  // grad_table maps variable name → current gradient tensor name (in the
  // backward graph).  grad_accum holds the accumulator for each variable.
  std::unordered_map<std::string, std::string> grad_table;
  std::unordered_map<std::string, std::string> grad_accum;
  int counter = 0;

  // Seed: the gradient of y w.r.t. itself is the "dy" function input.
  grad_table[y] = "dy";

  // Build the FunctionProto.
  FunctionProto func;
  func.set_domain(func_domain);
  func.set_name(func_name);

  // Inputs: xs + zs + "dy"
  for (const auto &x : xs)
    func.add_input(x);
  for (const auto &z : zs)
    func.add_input(z);
  func.add_input("dy");

  // Outputs: one per xs element.
  for (const auto &x : xs)
    func.add_output("grad_" + x);

  // Traverse nodes in reverse topological order.
  for (int i = static_cast<int>(topo_order.size()) - 1; i >= 0; --i) {
    const NodeProto &node = nodes[topo_order[static_cast<size_t>(i)]];
    ApplyBackward(node, grad_table, grad_accum, counter, func);
    // Promote newly accumulated gradients into grad_table so downstream
    // backward passes can use them.
    for (const auto &[var, acc] : grad_accum) {
      grad_table[var] = acc;
    }
  }

  // Emit Identity nodes to rename accumulators to canonical output names.
  for (const auto &x : xs) {
    auto it = grad_table.find(x);
    if (it == grad_table.end()) {
      std::ostringstream oss;
      oss << "onnx_gradient: no gradient was computed for '" << x << "'";
      throw std::runtime_error(oss.str());
    }
    const std::string out_name = "grad_" + x;
    if (it->second != out_name) {
      func.add_node("Identity", {it->second}, {out_name});
    }
  }

  // Required opset for the ops we emit.
  func.add_opset("", 21);

  return func;
}

} // namespace

FunctionProto GradientOfNodes(const std::vector<NodeProto> &nodes,
                              const std::vector<std::string> &inputs,
                              const std::vector<TensorProto> &initializers,
                              const std::vector<std::string> &xs, const std::string &y,
                              const std::vector<std::string> &zs) {
  (void)inputs;
  (void)initializers;
  return ComputeGradient(nodes, xs, y, zs, "", "gradient");
}

FunctionProto GradientOfFunction(const FunctionProto &function, const std::vector<std::string> &xs,
                                 const std::string &y, const std::vector<std::string> &zs) {
  std::vector<NodeProto> nodes;
  nodes.reserve(static_cast<size_t>(function.node_size()));
  for (int i = 0; i < function.node_size(); ++i) {
    nodes.push_back(function.node(i));
  }
  const std::string domain = function.has_domain() ? std::string(function.domain()) : "";
  const std::string name =
      function.has_name() ? std::string(function.name()) + "_grad" : "gradient";
  return ComputeGradient(nodes, xs, y, zs, domain, name);
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
