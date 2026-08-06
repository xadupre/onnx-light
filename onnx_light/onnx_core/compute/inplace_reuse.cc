// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/inplace_reuse.h"

#include <algorithm>
#include <unordered_set>

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

namespace {

// Squeeze/Unsqueeze data tensor is input 0 by ONNX spec.
constexpr int kSqueezeDataInputIndex = 0;
constexpr int kUnsqueezeDataInputIndex = 0;

// Number of bits used by a single element of ``t``. Returns ``0`` for element
// types whose byte size is not a fixed scalar (strings, sequences, maps,
// optionals and the undefined type), for which no buffer-size comparison is
// attempted.
constexpr int ElementBitWidth(TensorType t) {
  switch (t) {
  case TensorType::kBool:
  case TensorType::kUint8:
  case TensorType::kInt8:
  case TensorType::kFloat8e4m3fn:
  case TensorType::kFloat8e4m3fnuz:
  case TensorType::kFloat8e5m2:
  case TensorType::kFloat8e5m2fnuz:
  case TensorType::kFloat8e8m0:
    return 8;
  case TensorType::kUint16:
  case TensorType::kInt16:
  case TensorType::kFloat16:
  case TensorType::kBfloat16:
    return 16;
  case TensorType::kUint32:
  case TensorType::kInt32:
  case TensorType::kFloat:
    return 32;
  case TensorType::kUint64:
  case TensorType::kInt64:
  case TensorType::kDouble:
  case TensorType::kComplex64:
    return 64;
  case TensorType::kComplex128:
    return 128;
  case TensorType::kFloat4e2m1:
  case TensorType::kUint4:
  case TensorType::kInt4:
    return 4;
  case TensorType::kUint2:
  case TensorType::kInt2:
    return 2;
  default:
    return 0;
  }
}

// Packed byte size of ``t`` when its shape is fully known and its element type
// has a fixed bit width, otherwise ``std::nullopt``. Sub-byte element types are
// packed, matching the ONNX storage convention.
std::optional<int64_t> ConcreteByteSize(const SymTensor &t) {
  const int bits = ElementBitWidth(t.Dtype());
  if (bits == 0) {
    return std::nullopt;
  }
  const SymShape &shape = t.Shape();
  if (!shape.IsFullyKnown()) {
    return std::nullopt;
  }
  const int64_t num_elements = shape.NumElements();
  return (num_elements * bits + 7) / 8;
}

std::optional<expressions::DimType>
ByteSizeExpr(const SymTensor &t, expressions::SimplifiedExpressionCache *cache = nullptr) {
  const int bits = ElementBitWidth(t.Dtype());
  if (bits == 0) {
    return std::nullopt;
  }
  expressions::DimType num_elements = int64_t{1};
  for (std::size_t i = 0; i < t.Shape().Rank(); ++i) {
    num_elements = expressions::dim_mul(num_elements, ToDimType(t.Shape()[i]));
  }
  if (bits % 8 == 0) {
    return expressions::simplify_dim_type(
        expressions::dim_mul(num_elements, expressions::DimType{int64_t{bits / 8}}), cache);
  }
  // Sub-byte element types pack multiple values per byte, so the buffer size is
  // ceil(num_elements * bits / 8). ``dim_div`` is floor division, hence the
  // ``+7`` round-up term before dividing by 8.
  return expressions::simplify_dim_type(
      expressions::dim_div(
          expressions::dim_add(
              expressions::dim_mul(num_elements, expressions::DimType{int64_t{bits}}),
              expressions::DimType{int64_t{7}}),
          expressions::DimType{int64_t{8}}),
      cache);
}

// Returns whether two descriptors are byte-for-byte interchangeable: same
// element type and identical shape (rank and every dimension, concrete or
// symbolic). Equal descriptors imply equal element counts and therefore
// equal buffer sizes, which is the precondition for an in-place reuse.
bool SameStorage(const SymTensor &a, const SymTensor &b) {
  if (a.Dtype() != b.Dtype()) {
    return false;
  }
  const SymShape &sa = a.Shape();
  const SymShape &sb = b.Shape();
  if (sa.Rank() != sb.Rank()) {
    return false;
  }
  for (std::size_t i = 0; i < sa.Rank(); ++i) {
    if (sa[i] != sb[i]) {
      return false;
    }
  }
  return true;
}

// Classifies a candidate reuse of input ``in`` by output ``out`` by comparing
// their buffer sizes. ``kEqual`` is returned when the input and output buffers
// have the same byte size (e.g. a Transpose or same-total-size Reshape),
// making this the preferred, space-optimal reuse. ``kGreater`` is returned
// when the input buffer is strictly larger than the output, so the output
// still fits but leaves part of the buffer unused.  Any other case (input
// smaller) yields no opportunity.
//
// When a concrete byte-size comparison is unavailable for either tensor (e.g.
// either has symbolic dimensions), ``ByteSizeExpr`` is compared symbolically.
// Two expressions that simplify to the same canonical string are guaranteed to
// describe equal byte counts at runtime (e.g. a permutation of the same
// symbolic dimension names), so ``kEqual`` is returned for them.
std::optional<InPlaceReuseKind> ClassifyReuse(
    const ShapesContext &ctx, const std::string &out_name, const SymTensor &out,
    const std::string &in_name, const SymTensor &in,
    std::unordered_map<std::string, std::optional<expressions::DimType>> &byte_size_expr_cache,
    expressions::SimplifiedExpressionCache *simplification_cache = nullptr) {
  if (SameStorage(out, in)) {
    return InPlaceReuseKind::kEqual;
  }
  const std::optional<int64_t> out_bytes = ConcreteByteSize(out);
  const std::optional<int64_t> in_bytes = ConcreteByteSize(in);
  if (out_bytes.has_value() && in_bytes.has_value()) {
    if (*in_bytes == *out_bytes) {
      return InPlaceReuseKind::kEqual;
    }
    if (*in_bytes > *out_bytes) {
      return InPlaceReuseKind::kGreater;
    }
    return std::nullopt;
  }
  // Fall back to symbolic comparison when concrete sizes are unavailable for
  // either tensor.  Two byte-size expressions that simplify to the same
  // canonical form describe equal buffer sizes regardless of the runtime
  // values of symbolic dimension variables.
  const auto &out_expr =
      GetCachedByteSizeExpr(ctx, out_name, byte_size_expr_cache, simplification_cache);
  const auto &in_expr =
      GetCachedByteSizeExpr(ctx, in_name, byte_size_expr_cache, simplification_cache);
  if (out_expr.has_value() && in_expr.has_value() && *out_expr == *in_expr) {
    return InPlaceReuseKind::kEqual;
  }
  return std::nullopt;
}

} // namespace

const std::optional<expressions::DimType> &
GetCachedByteSizeExpr(const ShapesContext &ctx, const std::string &name,
                      std::unordered_map<std::string, std::optional<expressions::DimType>> &cache,
                      expressions::SimplifiedExpressionCache *simplification_cache) {
  auto [it, inserted] = cache.try_emplace(name);
  if (inserted) {
    it->second = ByteSizeExpr(ctx.Get(name), simplification_cache);
  }
  return it->second;
}

std::vector<InPlaceReuse> ComputeSingleNodeReuse(
    const NodeProto &node, int i, const ShapesContext &ctx,
    const std::unordered_set<std::string> &keep,
    const std::unordered_map<std::string, int> &producer,
    const std::unordered_map<std::string, int> &last_use,
    std::unordered_map<std::string, std::optional<expressions::DimType>> &byte_size_expr_cache,
    expressions::SimplifiedExpressionCache &simplified_dim_cache) {
  std::vector<InPlaceReuse> node_result;
  const std::string op_type = node.op_type();
  const bool is_squeeze = op_type == "Squeeze";
  const bool is_unsqueeze = op_type == "Unsqueeze";

  // Count direct-input occurrences so a value read twice is never aliased.
  std::unordered_map<std::string, int> input_occurrences;
  for (int k = 0; k < node.input_size(); ++k) {
    const std::string name = node.input(k);
    if (!name.empty()) {
      ++input_occurrences[name];
    }
  }

  std::unordered_set<int> used_inputs;
  std::unordered_set<int> matched_outputs;
  // Two passes so that same-sized (kEqual) reuse is always preferred over a
  // strictly larger (kGreater) input before either buffer is claimed.
  for (const InPlaceReuseKind kind : {InPlaceReuseKind::kEqual, InPlaceReuseKind::kGreater}) {
    for (int o = 0; o < node.output_size(); ++o) {
      if (matched_outputs.count(o)) {
        continue;
      }
      const std::string out_name = node.output(o);
      if (out_name.empty() || !ctx.Has(out_name)) {
        continue;
      }
      const SymTensor &out_tensor = ctx.Get(out_name);

      for (int k = 0; k < node.input_size(); ++k) {
        if (used_inputs.count(k)) {
          continue;
        }
        const std::string in_name = node.input(k);
        if (in_name.empty() || in_name == out_name) {
          continue;
        }
        if (input_occurrences[in_name] != 1) {
          continue;
        }
        if (keep.count(in_name)) {
          continue;
        }
        auto prod_it = producer.find(in_name);
        if (prod_it == producer.end() || prod_it->second >= i) {
          continue;
        }
        auto use_it = last_use.find(in_name);
        if (use_it == last_use.end() || use_it->second != i) {
          continue;
        }
        if (!ctx.Has(in_name)) {
          continue;
        }
        std::optional<InPlaceReuseKind> match;
        // Squeeze/Unsqueeze are shape-only view transforms on their data
        // input: they keep dtype and element count, so the output can always
        // alias that input when lifetime constraints allow it.
        // The dtype guard keeps this fast-path defensive for malformed graphs
        // or partial type information: aliasing is only safe when input/output
        // element storage matches.
        if (((k == kSqueezeDataInputIndex && is_squeeze) ||
             (k == kUnsqueezeDataInputIndex && is_unsqueeze)) &&
            out_tensor.Dtype() == ctx.Get(in_name).Dtype()) {
          match = InPlaceReuseKind::kEqual;
        } else {
          match = ClassifyReuse(ctx, out_name, out_tensor, in_name, ctx.Get(in_name),
                                byte_size_expr_cache, &simplified_dim_cache);
        }
        if (!match.has_value() || *match != kind) {
          continue;
        }
        InPlaceReuse reuse;
        reuse.output_index = o;
        reuse.input_index = k;
        reuse.kind = kind;
        node_result.push_back(reuse);
        used_inputs.insert(k);
        matched_outputs.insert(o);
        break;
      }
    }
  }
  // Keep each node's opportunities ordered by output index regardless of the
  // pass in which they were discovered.
  std::sort(
      node_result.begin(), node_result.end(),
      [](const InPlaceReuse &a, const InPlaceReuse &b) { return a.output_index < b.output_index; });
  return node_result;
}

std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuseMatches(const GraphProto &graph, const ShapesContext &ctx,
                           const ResultLifetimeInfo &lifetime) {
  const int num_nodes = graph.node().size();
  std::vector<std::vector<InPlaceReuse>> result(static_cast<std::size_t>(num_nodes));
  expressions::SimplifiedExpressionCache simplified_dim_cache;
  std::unordered_map<std::string, std::optional<expressions::DimType>> byte_size_expr_cache;

  for (int i = 0; i < num_nodes; ++i) {
    result[static_cast<std::size_t>(i)] =
        ComputeSingleNodeReuse(graph.node()[i], i, ctx, lifetime.keep, lifetime.producer,
                               lifetime.last_use, byte_size_expr_cache, simplified_dim_cache);
  }
  return result;
}

std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuse(const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite);
  return inplace.Reuse();
}

void WriteInPlaceReuseToMetadata(GraphProto &graph, const ShapesContext &ctx,
                                 const std::unordered_map<std::string, std::string> &value_tags) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, value_tags);
  inplace.WriteToMetadata(graph);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
