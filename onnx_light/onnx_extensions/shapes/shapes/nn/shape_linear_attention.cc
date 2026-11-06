// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <algorithm>
#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

namespace {

SymDim MergeDim(const SymDim &a, const SymDim &b, const char *what) {
  if (a.IsInt() && b.IsInt()) {
    EXT_ENFORCE_INVALID(a.AsInt() == b.AsInt(), "ComputeShapeLinearAttention: ", what,
                        " mismatch: ", a.AsInt(), " vs ", b.AsInt(), ".");
    return a;
  }
  if (a.IsInt()) {
    return a;
  }
  if (b.IsInt()) {
    return b;
  }
  return a;
}

void RequireRank3(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 3, "ComputeShapeLinearAttention: input '", name,
                      "' must have rank 3, got ", shape.Rank(), ".");
}

void RequireRank4(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 4, "ComputeShapeLinearAttention: input '", name,
                      "' must have rank 4, got ", shape.Rank(), ".");
}

} // namespace

void ComputeShapeLinearAttention(ShapesContext &ctx, const NodeProto &node, const char *query,
                                 const char *key, const char *value, const char *past_state) {
  CheckNodeOpAndOutput(node, "LinearAttention", "ComputeShapeLinearAttention");

  // Required attributes ``q_num_heads`` and ``kv_num_heads``.
  int64_t q_num_heads = 0;
  int64_t kv_num_heads = 0;
  bool has_q = false;
  bool has_kv = false;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "q_num_heads") {
      q_num_heads = attr.i();
      has_q = true;
    } else if (attr.name() == "kv_num_heads") {
      kv_num_heads = attr.i();
      has_kv = true;
    }
  }
  EXT_ENFORCE_INVALID(
      !(!has_q || !has_kv),
      "ComputeShapeLinearAttention: attributes 'q_num_heads' and 'kv_num_heads' are required.");
  EXT_ENFORCE_INVALID(
      !(q_num_heads <= 0 || kv_num_heads <= 0),
      "ComputeShapeLinearAttention: 'q_num_heads' and 'kv_num_heads' must be positive.");
  EXT_ENFORCE_INVALID((q_num_heads % kv_num_heads) == 0,
                      "ComputeShapeLinearAttention: q_num_heads (", q_num_heads,
                      ") must be a multiple of kv_num_heads (", kv_num_heads, ").");

  const SymTensor &Q = ctx.Get(query);
  const SymTensor &K = ctx.Get(key);
  const SymTensor &V = ctx.Get(value);

  const SymShape &q_shape = Q.Shape();
  const SymShape &k_shape = K.Shape();
  const SymShape &v_shape = V.Shape();
  RequireRank3(q_shape, query);
  RequireRank3(k_shape, key);
  RequireRank3(v_shape, value);

  // Batch and sequence length: shared between Q, K and V.
  SymDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");
  SymDim seq_len = MergeDim(q_shape[1], k_shape[1], "sequence_length");
  seq_len = MergeDim(seq_len, v_shape[1], "sequence_length");

  // d_k = key.shape[-1] / kv_num_heads; d_v = value.shape[-1] / kv_num_heads.
  // When the value/key last dim is symbolic we fall back to a fresh symbolic
  // placeholder ("?") so the rank is correct even if the exact value is
  // unknown. ``past_state`` (if provided) can refine these below.
  SymDim d_k;
  bool d_k_resolved = false;
  if (k_shape[2].IsInt()) {
    const int64_t hidden_k = k_shape[2].AsInt();
    EXT_ENFORCE_INVALID((hidden_k % kv_num_heads) == 0,
                        "ComputeShapeLinearAttention: key last dim (", hidden_k,
                        ") must be divisible by kv_num_heads (", kv_num_heads, ").");
    if (q_shape[2].IsInt()) {
      const int64_t hidden_q = q_shape[2].AsInt();
      EXT_ENFORCE_INVALID((hidden_q % q_num_heads) == 0,
                          "ComputeShapeLinearAttention: query last dim (", hidden_q,
                          ") must be divisible by q_num_heads (", q_num_heads, ").");
      const int64_t q_d_k = hidden_q / q_num_heads;
      const int64_t k_d_k = hidden_k / kv_num_heads;
      EXT_ENFORCE_INVALID(q_d_k == k_d_k, "ComputeShapeLinearAttention: query head_size (", q_d_k,
                          ") must equal key head_size (", k_d_k, ").");
    }
    d_k = SymDim(hidden_k / kv_num_heads);
    d_k_resolved = true;
  }

  SymDim d_v;
  bool d_v_resolved = false;
  if (v_shape[2].IsInt()) {
    const int64_t hidden_v = v_shape[2].AsInt();
    EXT_ENFORCE_INVALID((hidden_v % kv_num_heads) == 0,
                        "ComputeShapeLinearAttention: value last dim (", hidden_v,
                        ") must be divisible by kv_num_heads (", kv_num_heads, ").");
    d_v = SymDim(hidden_v / kv_num_heads);
    d_v_resolved = true;
  }

  // Cross-check / refine using past_state (B, H_kv, d_k, d_v) if present.
  if (past_state != nullptr && ctx.Has(past_state)) {
    const SymShape &ps_shape = ctx.Get(past_state).Shape();
    RequireRank4(ps_shape, past_state);
    batch = MergeDim(batch, ps_shape[0], "batch");
    EXT_ENFORCE_INVALID(!(ps_shape[1].IsInt() && ps_shape[1].AsInt() != kv_num_heads),
                        "ComputeShapeLinearAttention: past_state dim 1 (", ps_shape[1].AsInt(),
                        ") must equal kv_num_heads (", kv_num_heads, ").");
    if (d_k_resolved) {
      d_k = MergeDim(d_k, ps_shape[2], "d_k");
    } else {
      d_k = ps_shape[2];
      d_k_resolved = ps_shape[2].IsInt();
    }
    if (d_v_resolved) {
      d_v = MergeDim(d_v, ps_shape[3], "d_v");
    } else {
      d_v = ps_shape[3];
      d_v_resolved = ps_shape[3].IsInt();
    }
  }

  // Compute output last dim = q_num_heads * d_v.
  SymDim out_last;
  if (d_v_resolved) {
    out_last = SymDim(core::shapes::CheckedDimensionMultiply(q_num_heads, d_v.AsInt()));
  } else if (q_num_heads == kv_num_heads && !v_shape[2].IsInt()) {
    // Special case: q_num_heads * d_v = q_num_heads * (v_shape[2] / kv_num_heads) = v_shape[2].
    out_last = v_shape[2];
  } else {
    out_last = SymDim(std::string("?"));
  }
  if (!d_k_resolved) {
    d_k = SymDim(std::string("?"));
  }
  if (!d_v_resolved) {
    d_v = SymDim(std::string("?"));
  }

  // Output 0: output = (B, T, q_num_heads * d_v).
  {
    SymShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(seq_len);
    out_shape.PushBack(out_last);
    ctx.Set(node.output(0), SymTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_state = (B, kv_num_heads, d_k, d_v).
  if (node.output_size() > 1 && !node.output(1).empty()) {
    SymShape ps_shape;
    ps_shape.PushBack(batch);
    ps_shape.PushBack(SymDim(kv_num_heads));
    ps_shape.PushBack(d_k);
    ps_shape.PushBack(d_v);
    // Dtype: prefer past_state dtype when available, otherwise inherit from Q.
    TensorType state_dtype = Q.Dtype();
    if (past_state != nullptr && ctx.Has(past_state)) {
      state_dtype = ctx.Get(past_state).Dtype();
    }
    ctx.Set(node.output(1), SymTensor(nullptr, state_dtype, std::move(ps_shape)));
  }
}

core::shapes::PeakMemoryEstimate
ComputePeakMemoryLinearAttention(Device device, const std::vector<SymShape> &input_shapes) {
  return ComputePeakMemoryLinearAttention(device, NodeProto{}, input_shapes);
}

core::shapes::PeakMemoryEstimate
ComputePeakMemoryLinearAttention(Device, const NodeProto &node,
                                 const std::vector<SymShape> &input_shapes) {
  if (input_shapes.size() < 3 || input_shapes[0].Rank() != 3 || input_shapes[1].Rank() != 3 ||
      input_shapes[2].Rank() != 3) {
    return core::shapes::UnknownPeakMemory("", "LinearAttention");
  }

  for (int slot = 3; slot < std::min(node.input_size(), 6); ++slot) {
    if (!node.input(slot).empty() &&
        (static_cast<size_t>(slot) >= input_shapes.size() || input_shapes[slot].Empty()))
      return core::shapes::UnknownPeakMemory("", "LinearAttention");
  }
  const auto present_input = [&](size_t slot) {
    return slot < input_shapes.size() &&
           (slot < 3 || (node.input_size() == 0 ? !input_shapes[slot].Empty()
                                                : slot < static_cast<size_t>(node.input_size()) &&
                                                      !node.input(static_cast<int>(slot)).empty()));
  };
  bool unresolved = false;
  for (size_t slot = 0; slot < input_shapes.size(); ++slot) {
    if (!present_input(slot))
      continue;
    for (const auto &dim : input_shapes[slot].Dims()) {
      EXT_ENFORCE_INVALID(!dim.IsInt() || dim.AsInt() >= 0,
                          "Negative LinearAttention scratch dimension.");
      unresolved |= dim.IsExpr() && (dim.AsExpr().empty() || dim.AsExpr() == "?");
    }
  }
  if (unresolved)
    return core::shapes::UnknownPeakMemory("", "LinearAttention");

  int64_t q_heads = 0, kv_heads = 0;
  bool use_beta = true;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "q_num_heads" || attr.name() == "kv_num_heads") {
      EXT_ENFORCE_INVALID(attr.type() == AttributeProto::INT && attr.i() > 0,
                          "LinearAttention scratch: head counts must be positive integers.");
      (attr.name() == "q_num_heads" ? q_heads : kv_heads) = attr.i();
    } else if (attr.name() == "update_rule") {
      const auto &rule = attr.s();
      EXT_ENFORCE_INVALID(
          attr.type() == AttributeProto::STRING &&
              (rule == "linear" || rule == "gated" || rule == "delta" || rule == "gated_delta"),
          "LinearAttention scratch: invalid update_rule.");
      use_beta = rule == "delta" || rule == "gated_delta";
    }
  }
  if (q_heads != 0 && kv_heads != 0)
    EXT_ENFORCE_INVALID(q_heads % kv_heads == 0,
                        "LinearAttention scratch: incompatible head counts.");
  using namespace core::expressions;
  using core::symbolic::ToDimType;
  const auto elements = [&](size_t slot) -> DimType {
    if (!present_input(slot))
      return int64_t{0};
    std::vector<core::expressions::DimType> factors;
    for (const auto &dim : input_shapes[slot].Dims())
      factors.push_back(ToDimType(dim));
    return dim_multi_mul(factors);
  };
  const DimType batch = ToDimType(input_shapes[0][0]);
  const DimType sequence = ToDimType(input_shapes[0][1]);
  const DimType hidden_k = ToDimType(input_shapes[1][2]);
  const DimType hidden_v = ToDimType(input_shapes[2][2]);
  const bool has_state = present_input(3);
  if (has_state && input_shapes[3].Rank() != 4)
    return core::shapes::UnknownPeakMemory("", "LinearAttention");
  const DimType state =
      has_state
          ? elements(3)
          : dim_multi_mul(
                {batch, dim_exact_div(hidden_k, int64_t{kv_heads > 0 ? kv_heads : 1}), hidden_v});
  const DimType output =
      q_heads > 0 && kv_heads > 0
          ? dim_multi_mul(
                {batch, sequence, int64_t{q_heads}, dim_exact_div(hidden_v, int64_t{kv_heads})})
          : dim_multi_mul({batch, sequence, ToDimType(input_shapes[0][2]), hidden_v});
  DimType input_elements = int64_t{0};
  for (size_t slot = 0; slot < input_shapes.size(); ++slot)
    input_elements = dim_add(input_elements, elements(slot));
  // Covers promotions, FLOAT32 results and temporary half-precision demotions.
  // The final declared output/state buffers are accounted for by shape analysis.
  DimType bytes =
      dim_add(dim_mul(int64_t{4}, input_elements), dim_mul(int64_t{6}, dim_add(output, state)));
  if (node.output_size() != 0 && (node.output_size() < 2 || node.output(1).empty()))
    bytes = dim_add(bytes, dim_mul(int64_t{4}, state));
  if (use_beta && (!input_shapes[0][1].IsInt() || input_shapes[0][1].AsInt() != 0))
    bytes = dim_add(bytes, dim_multi_mul({int64_t{4}, batch, hidden_v}));
  const SymDim maximum = core::symbolic::FromDimType(bytes);
  if (maximum.IsInt() && maximum.AsInt() == 0)
    return core::shapes::PeakMemoryEstimate::Exact(maximum);
  return {maximum, std::nullopt};
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
