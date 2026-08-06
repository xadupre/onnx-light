// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

namespace {

/// Creates a 1-D INT64 constant tensor node with the given values.
std::string MakeInt64Constant(const std::vector<int64_t> &values, int &counter,
                              FunctionProto &func) {
  std::string name = NewGradName("const_i64", counter);
  NodeProto &const_node = func.add_node("Constant", {}, {name});
  AttributeProto *attr = const_node.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto &tensor = attr->ref_t();
  tensor.set_data_type(static_cast<int32_t>(TensorProto::DataType::INT64));
  tensor.ref_dims().push_back(static_cast<int64_t>(values.size()));
  for (int64_t v : values)
    tensor.ref_int64_data().push_back(v);
  return name;
}

} // namespace

bool GradConv(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 2)
    return false;

  const std::string &X = (!inputs[0].empty()) ? inputs[0] : utils::String::empty_string();
  const std::string &W = (!inputs[1].empty()) ? inputs[1] : utils::String::empty_string();

  if (X.empty() || W.empty())
    return false;

  // Read Conv attributes (all optional; defaults follow the ONNX spec).
  int64_t group = GetAttributeOr<int64_t>(node, "group", int64_t{1});

  std::vector<int64_t> strides, dilations, pads, kernel_shape;
  GetAttributeInts(node, "strides", strides);
  GetAttributeInts(node, "dilations", dilations);
  GetAttributeInts(node, "pads", pads);
  GetAttributeInts(node, "kernel_shape", kernel_shape);

  // Determine spatial rank from the available attribute vectors.
  size_t spatial_rank = 0;
  if (!pads.empty())
    spatial_rank = pads.size() / 2;
  else if (!strides.empty())
    spatial_rank = strides.size();
  else if (!dilations.empty())
    spatial_rank = dilations.size();
  else if (!kernel_shape.empty())
    spatial_rank = kernel_shape.size();
  else
    spatial_rank = 2; // default to 2-D conv

  // Build the (N, C) ↔ (C, N) transpose permutation: [1, 0, 2, 3, …]
  std::vector<int64_t> perm;
  perm.reserve(2 + spatial_rank);
  perm.push_back(1);
  perm.push_back(0);
  for (size_t i = 0; i < spatial_rank; ++i)
    perm.push_back(static_cast<int64_t>(2 + i));

  // ── Gradient w.r.t. X (input) ─────────────────────────────────────────
  //
  // dX = ConvTranspose(dY, W)  with the same strides/dilations/pads/group.
  //
  // Note: for stride > 1, output_padding may be needed to exactly match the
  // spatial shape of X (ConvTranspose output is ≤ the original X shape by
  // at most stride-1 per dimension).  For stride=1, the result is exact.
  {
    std::string dX = NewGradName("dX", counter);
    NodeProto &ct = func.add_node("ConvTranspose", {output_grad, W}, {dX});
    if (!strides.empty())
      AddAttribute(ct, "strides", strides);
    if (!dilations.empty())
      AddAttribute(ct, "dilations", dilations);
    if (!pads.empty())
      AddAttribute(ct, "pads", pads);
    if (group != 1)
      AddAttribute(ct, "group", group);
    AccumulateGrad(dX, grad_accum[X], counter, func);
  }

  // ── Gradient w.r.t. W (weights) ──────────────────────────────────────
  //
  // Derivation (single-group, no stride/dilation for clarity):
  //   Y[n, co, p] = Σ_{ci,k} W[co, ci, k] · X[n, ci, p + k]
  //   dW[co, ci, k] = Σ_{n,p} dY[n, co, p] · X[n, ci, p + k]
  //
  // This equals Conv(X_T, dY_T, strides=dilations_fwd, dilations=strides_fwd)
  // where X_T = Transpose(X_padded, [1,0,…]) and dY_T = Transpose(dY, [1,0,…]),
  // followed by Transpose([1,0,…]) of the result.
  //
  // The padding from the forward Conv is applied to X before the inner Conv
  // so that border effects cancel correctly.
  {
    // Step 1: Pad X with the original Conv pads (if any are non-zero).
    std::string X_padded;
    bool needs_padding = false;
    if (!pads.empty()) {
      for (int64_t p : pads)
        if (p != 0) {
          needs_padding = true;
          break;
        }
    }

    if (needs_padding) {
      // ONNX Pad (opset 11+) pads input is a 1-D INT64 tensor of length
      // 2 * rank where rank is the number of tensor dimensions.
      // Format: [begin_d0, begin_d1, …, end_d0, end_d1, …]
      // For X of shape [N, C, d1, …, dk]:
      //   N and C dimensions → 0 padding
      //   spatial dims → pads from Conv (pads[i] for begin, pads[rank+i] for end)
      size_t total_rank = 2 + spatial_rank;
      std::vector<int64_t> pad_values(2 * total_rank, 0);
      for (size_t i = 0; i < spatial_rank; ++i) {
        pad_values[2 + i] = pads[i];                             // begin pads (spatial)
        pad_values[total_rank + 2 + i] = pads[spatial_rank + i]; // end pads (spatial)
      }
      std::string pads_cst = MakeInt64Constant(pad_values, counter, func);
      X_padded = NewGradName("X_padded", counter);
      func.add_node("Pad", {X, pads_cst}, {X_padded});
    } else {
      X_padded = X;
    }

    // Step 2: Transpose X_padded and dY to bring (N, C) → (C, N).
    std::string X_T = NewGradName("X_T", counter);
    NodeProto &xt_node = func.add_node("Transpose", {X_padded}, {X_T});
    AddAttribute(xt_node, "perm", perm);

    std::string dY_T = NewGradName("dY_T", counter);
    NodeProto &dyt_node = func.add_node("Transpose", {output_grad}, {dY_T});
    AddAttribute(dyt_node, "perm", perm);

    // Step 3: Inner Conv(X_T, dY_T) with strides ↔ dilations swapped.
    //   X_T shape: [C_in, N, H+pads, W+pads]
    //   dY_T shape: [C_out, N, H_out, W_out]   (acts as the filter)
    //   Output: [C_in, C_out, kH, kW]  = dW before final transpose
    std::string dW_T = NewGradName("dW_T", counter);
    NodeProto &wconv = func.add_node("Conv", {X_T, dY_T}, {dW_T});
    // strides for inner conv = dilations of original forward Conv
    if (!dilations.empty()) {
      AddAttribute(wconv, "strides", dilations);
    } else if (!strides.empty()) {
      // dilations were 1 (default) → swap to 1 as well
      std::vector<int64_t> ones(spatial_rank, 1);
      AddAttribute(wconv, "strides", ones);
    }
    // dilations for inner conv = strides of original forward Conv
    if (!strides.empty()) {
      AddAttribute(wconv, "dilations", strides);
    } else if (!dilations.empty()) {
      std::vector<int64_t> ones(spatial_rank, 1);
      AddAttribute(wconv, "dilations", ones);
    }
    // No padding for the inner Conv: we already embedded the padding in X_padded.

    // Step 4: Transpose [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW]
    std::string dW = NewGradName("dW", counter);
    NodeProto &tw_node = func.add_node("Transpose", {dW_T}, {dW});
    AddAttribute(tw_node, "perm", perm);

    AccumulateGrad(dW, grad_accum[W], counter, func);
  }

  // ── Gradient w.r.t. B (bias, optional) ───────────────────────────────
  //
  // dB = ReduceSum(dY, axes=[0, 2, …, spatial_rank+1], keepdims=0)
  if (inputs.size() >= 3 && !inputs[2].empty()) {
    const std::string &B = inputs[2];
    std::string dB = NewGradName("dB", counter);
    std::vector<int64_t> axes;
    axes.push_back(0); // batch axis
    for (size_t i = 0; i < spatial_rank; ++i)
      axes.push_back(static_cast<int64_t>(2 + i)); // spatial axes
    NodeProto &rs = func.add_node("ReduceSum", {output_grad}, {dB});
    AddAttribute(rs, "axes", axes);
    AddAttribute(rs, "keepdims", int64_t{0});
    AccumulateGrad(dB, grad_accum[B], counter, func);
  }

  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
