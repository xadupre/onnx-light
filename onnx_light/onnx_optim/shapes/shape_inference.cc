// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_proto/onnx_helper.h"

#include "onnx_lib/shape_inference/attribute_binder.h"
#include "onnx_optim/expressions.h"
#include "onnx_optim/shapes/dispatch_table.h"
#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_optim/shapes/preview/shape_preview.h"
#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"
#include "onnx_optim/shapes/training/shape_training.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Checks the node belongs to a supported domain: the default ONNX
// domain (empty string or "ai.onnx") or the traditional ML domain
// ("ai.onnx.ml"). Throws std::invalid_argument otherwise.
// Domain-specific dispatch can be added here when other domains gain
// support.
void CheckOnnxDomain(const NodeProto &node) {
  EXT_ENFORCE_INVALID(node.domain().empty() || node.domain() == kOnnxDomain ||
                          node.domain() == traditionalml::kOnnxMlDomain ||
                          node.domain() == preview::kOnnxPreviewDomain ||
                          node.domain() == training::kOnnxPreviewTrainingDomain,
                      "ComputeShapeNode: unsupported domain '" + node.domain().as_string() +
                          "' for op '" + node.op_type().as_string() + "'.");
}

// Returns the ``"<domain>:<name>"`` identifier used as a key in
// :cpp:func:`ShapesContext::SetLocalFunction` /
// :cpp:func:`ShapesContext::GetLocalFunction`. The empty default ONNX
// domain is kept as-is here because local functions live in non-default
// domains in practice; the domain is matched literally against the
// FunctionProto's own ``domain`` field.
std::string LocalFunctionKey(const std::string &domain, const std::string &name) {
  return domain + ":" + name;
}

// Expands a single local-function call ``node`` into shape inference
// over ``func.node()`` with the function's input/output names rebound
// to the caller's names. Bound through positional binding of
// ``node.input(i)`` to ``func.input(i)`` and ``node.output(i)`` to
// ``func.output(i)``.
//
// The function body is processed in an isolated :cpp:class:`ShapesContext`
// that carries the function's own opset imports (falling back to the
// caller's opsets for any domain not redeclared by the function) and
// the same local-function map, so nested local-function calls are also
// supported.
void ExpandLocalFunctionCall(ShapesContext &ctx, const NodeProto &node, const FunctionProto &func) {
  ShapesContext sub_ctx;
  // Inherit caller opsets first, then let the function's own opset
  // imports override them.
  for (const auto &kv : ctx.Opsets()) {
    sub_ctx.SetOpsetVersion(kv.first, kv.second);
  }
  for (int i = 0; i < func.opset_import().size(); ++i) {
    const OperatorSetIdProto &osi = func.opset_import()[i];
    sub_ctx.SetOpsetVersion(osi.domain().as_string(), static_cast<int>(osi.version()));
  }
  // Forward the local-function map so nested calls are dispatched too.
  for (const auto &kv : ctx.LocalFunctions()) {
    sub_ctx.SetLocalFunction(kv.second);
  }
  // Positional binding: function input names take the descriptors of
  // the caller's input names.
  const int n_inputs = std::min(node.input_size(), func.input_size());
  for (int i = 0; i < n_inputs; ++i) {
    const std::string caller_name = node.input(i).as_string();
    const std::string callee_name = func.input(i).as_string();
    if (caller_name.empty() || callee_name.empty()) {
      continue;
    }
    if (ctx.Has(caller_name)) {
      sub_ctx.Set(callee_name, OptimTensor(ctx.Get(caller_name)));
    } else if (ctx.HasSequence(caller_name)) {
      sub_ctx.SetSequence(callee_name, OptimSequence(ctx.GetSequence(caller_name)));
    }
  }
  // Resolve linked attributes (``ref_attr_name``) in the function body
  // against the call-site node's attributes before running shape
  // inference. Attributes referencing a name not supplied by the call
  // site are removed (matching ``AttributeBinder`` semantics).
  internal::AttributeMap attr_map;
  for (const auto &attr : node.attribute()) {
    attr_map[attr.name().as_string()] = &attr;
  }
  internal::AttributeBinder attribute_binder(attr_map);
  // Recursively run shape inference on the function body, binding
  // attribute references on a per-node copy to avoid mutating ``func``.
  for (const auto &fn_node : func.node()) {
    NodeProto bound_node;
    bound_node.CopyFrom(fn_node);
    attribute_binder.VisitNode(&bound_node);
    sub_ctx.ComputeShapeNode(bound_node);
  }
  // Map function outputs back to caller-visible names.
  const int n_outputs = std::min(node.output_size(), func.output_size());
  for (int i = 0; i < n_outputs; ++i) {
    const std::string callee_name = func.output(i).as_string();
    const std::string caller_name = node.output(i).as_string();
    if (caller_name.empty() || callee_name.empty()) {
      continue;
    }
    if (sub_ctx.Has(callee_name)) {
      ctx.Set(caller_name, OptimTensor(sub_ctx.Get(callee_name)));
    } else if (sub_ctx.HasSequence(callee_name)) {
      ctx.SetSequence(caller_name, OptimSequence(sub_ctx.GetSequence(callee_name)));
    }
  }
}

// Normalises the empty default ONNX domain to ``kOnnxDomain`` so that
// dispatch-table lookups always use a canonical key.
std::string NormaliseDispatchDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  return domain.empty() ? std::string(kOnnxDomain) : domain;
}

using AnchorMap = std::unordered_map<std::string, OptimTensor>;

// Records ``a == b`` (a constraint between two symbolic expressions) and,
// when their algebraic difference reduces to ``c*x - c*y`` (a coefficient
// map with exactly two non-zero entries of equal magnitude and opposite
// signs), also records the implied leaf-level equality ``x == y``. This
// lets output anchors like ``Y: [2*dnz]`` propagate down to intermediate
// tensors whose inferred shape uses a scaled symbol such as
// ``[2*NonZero_nz_nnz]`` — without the derived ``dnz == NonZero_nz_nnz``
// constraint, the canonicalisation pass would only rename the compound
// expression and leave the leaf occurrence of ``NonZero_nz_nnz``
// untouched.
void AddSymbolicConstraintWithLeafDerivation(ShapesContext &ctx, const std::string &a,
                                             const std::string &b) {
  ctx.AddConstraint(a, b);
  if (a == b) {
    return;
  }
  std::map<std::string, int64_t> diff;
  try {
    diff = expressions::simplify_two_expressions(a, b);
  } catch (const std::runtime_error &) {
    return;
  }
  if (diff.size() != 2) {
    return;
  }
  auto it = diff.begin();
  const std::string &name1 = it->first;
  const int64_t coeff1 = it->second;
  ++it;
  const std::string &name2 = it->first;
  const int64_t coeff2 = it->second;
  if (coeff1 + coeff2 != 0) {
    return;
  }
  if (name1.empty() || name2.empty() || name1 == name2) {
    return;
  }
  ctx.AddConstraint(name1, name2);
}

// Returns ``true`` when ``vi`` carries a tensor type with a non-empty
// ``shape`` field. ValueInfo entries that only declare an element type
// (no shape annotation at all) produce a rank-0 ``OptimTensor`` which
// would conflict with every non-scalar inferred shape, so they must be
// skipped when building the anchor set.
bool ValueInfoHasTensorShape(const ValueInfoProto &vi) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) {
    return false;
  }
  return vi.type().tensor_type().has_shape();
}

void AddValueInfoAsAnchor(const ValueInfoProto &vi, AnchorMap &anchors) {
  const std::string name = vi.name().as_string();
  if (name.empty() || !ValueInfoHasTensorShape(vi)) {
    return;
  }
  OptimTensor tensor;
  if (!OptimTensorFromValueInfo(vi, tensor)) {
    return;
  }
  anchors.try_emplace(name, std::move(tensor));
}

// Collects anchors from ``graph.output`` only. Used for the
// "always-anchor outputs" pass that runs unconditionally so that user-
// declared output shape expressions (e.g. ``Y: [2*dnz]``) propagate
// into intermediate tensors via ``PropagateAnchorConstraintsIntoContext``
// regardless of whether ``prefill_with_value_info_output`` was set.
AnchorMap CollectGraphOutputAnchors(const GraphProto &graph) {
  AnchorMap anchors;
  for (int i = 0; i < graph.output_size(); ++i) {
    AddValueInfoAsAnchor(graph.output(i), anchors);
  }
  return anchors;
}

AnchorMap CollectGraphAnchors(const GraphProto &graph) {
  AnchorMap anchors = CollectGraphOutputAnchors(graph);
  // Outputs are considered more authoritative than value_info for the
  // same name (first insert wins).
  for (int i = 0; i < graph.value_info_size(); ++i) {
    AddValueInfoAsAnchor(graph.value_info(i), anchors);
  }
  return anchors;
}

// Merges ``anchor`` into ``inferred`` while privileging anchor information.
//
// The function returns the merged :cpp:class:`OptimTensor` and, as a
// side effect, records any symbolic-dimension equality (``inferred``
// symbol ↔ ``anchor`` symbol) into ``ctx.AddConstraint``. It throws
// ``std::invalid_argument`` when the two descriptors are provably
// incompatible — different known element types, different ranks, or
// different concrete integer dimensions — so that callers learn early
// about contradictions instead of silently overwriting one side.
//
// The merge rules per field are:
//   - dtype/device: if both sides know a value, they must be equal,
//     otherwise the known side wins (kUndefined is "no information").
//   - rank: must match.
//   - per-dim: two equal dims are kept as-is; two different concrete
//     integers are a conflict; one concrete + one symbolic resolves to
//     the concrete value; two different symbolic expressions become
//     the anchor's symbol and an equality constraint is recorded.
//   - value_as_shape: same per-dim rules as above when both sides have
//     an annotation; otherwise the present annotation is kept.
//   - min/max bounds: a known bound is kept; when both are known the
//     tighter bound wins (higher min, lower max). Provably disjoint
//     intervals are a conflict.
// Returns std::nullopt when ``strict`` is false and a conflict is
// detected (incompatible dtype/rank/dim/device/bounds). When ``strict``
// is true, conflicts trigger EXT_ENFORCE_INVALID as before.
std::optional<OptimTensor> MergeWithAnchor(ShapesContext &ctx, const std::string &name,
                                           const OptimTensor &inferred, const OptimTensor &anchor,
                                           bool strict = true) {
  // dtype: both known and different → conflict.
  if (inferred.Dtype() != TensorType::kUndefined && anchor.Dtype() != TensorType::kUndefined &&
      inferred.Dtype() != anchor.Dtype()) {
    if (!strict) {
      return std::nullopt;
    }
    EXT_ENFORCE_INVALID(
        false, "MergeWithAnchor: incompatible element type for '" + name +
                   "': inferred has dtype " + std::to_string(static_cast<int>(inferred.Dtype())) +
                   ", anchor has dtype " + std::to_string(static_cast<int>(anchor.Dtype())) + ".");
  }
  // Rank check.
  if (inferred.Shape().Rank() != anchor.Shape().Rank()) {
    if (!strict) {
      return std::nullopt;
    }
    EXT_ENFORCE_INVALID(false, "MergeWithAnchor: incompatible rank for '" + name +
                                   "': inferred has rank " +
                                   std::to_string(inferred.Shape().Rank()) + ", anchor has rank " +
                                   std::to_string(anchor.Shape().Rank()) + ".");
  }
  // Per-dim merge.
  OptimShape merged_shape;
  for (std::size_t i = 0; i < inferred.Shape().Rank(); ++i) {
    const OptimDim &di = inferred.Shape()[i];
    const OptimDim &da = anchor.Shape()[i];
    if (di == da) {
      merged_shape.PushBack(di);
    } else if (di.IsInt() && da.IsInt()) {
      if (!strict) {
        return std::nullopt;
      }
      EXT_ENFORCE_INVALID(false, "MergeWithAnchor: incompatible dim " + std::to_string(i) +
                                     " for '" + name + "': inferred=" + std::to_string(di.AsInt()) +
                                     ", anchor=" + std::to_string(da.AsInt()) + ".");
    } else if (di.IsInt()) {
      // anchor is symbolic, inferred is concrete: keep the concrete one
      // but still record the anchor symbol equals the concrete value.
      ctx.AddConstraint(da.AsExpr(), std::to_string(di.AsInt()));
      merged_shape.PushBack(di);
    } else if (da.IsInt()) {
      // anchor is concrete, inferred is symbolic: privilege the
      // concrete anchor value but remember the symbol's binding.
      ctx.AddConstraint(di.AsExpr(), std::to_string(da.AsInt()));
      merged_shape.PushBack(da);
    } else {
      // Both symbolic and different: record the equality and privilege
      // the anchor's symbol.
      AddSymbolicConstraintWithLeafDerivation(ctx, di.AsExpr(), da.AsExpr());
      merged_shape.PushBack(da);
    }
  }
  // Start from the inferred tensor (it carries the data pointer, if any)
  // and overwrite the shape with the merged result. Element type and
  // device prefer the known side; when both are known and equal we keep
  // either (anchor wins by convention).
  TensorType merged_dtype = inferred.Dtype();
  if (anchor.Dtype() != TensorType::kUndefined) {
    merged_dtype = anchor.Dtype();
  }
  OptimTensor out(inferred.Data(), merged_dtype, std::move(merged_shape));
  Device merged_device = inferred.GetDevice();
  if (anchor.GetDevice() != Device::kUndefined) {
    if (inferred.GetDevice() != Device::kUndefined && inferred.GetDevice() != anchor.GetDevice()) {
      if (!strict) {
        return std::nullopt;
      }
      EXT_ENFORCE_INVALID(false, "MergeWithAnchor: incompatible device for '" + name + "'.");
    }
    merged_device = anchor.GetDevice();
  }
  if (merged_device != Device::kUndefined) {
    out.SetDevice(merged_device);
  }
  // value_as_shape: merge per-dim with the same rules.
  if (inferred.HasValueAsShape() && anchor.HasValueAsShape()) {
    const OptimShape &a = inferred.ValueAsShape();
    const OptimShape &b = anchor.ValueAsShape();
    if (a.Rank() != b.Rank()) {
      if (!strict) {
        return std::nullopt;
      }
      EXT_ENFORCE_INVALID(false,
                          "MergeWithAnchor: incompatible value_as_shape rank for '" + name + "'.");
    }
    OptimShape merged_vas;
    for (std::size_t i = 0; i < a.Rank(); ++i) {
      const OptimDim &di = a[i];
      const OptimDim &da = b[i];
      if (di == da) {
        merged_vas.PushBack(di);
      } else if (di.IsInt() && da.IsInt()) {
        if (!strict) {
          return std::nullopt;
        }
        EXT_ENFORCE_INVALID(false,
                            "MergeWithAnchor: incompatible value_as_shape dim for '" + name + "'.");
      } else if (di.IsInt()) {
        ctx.AddConstraint(da.AsExpr(), std::to_string(di.AsInt()));
        merged_vas.PushBack(di);
      } else if (da.IsInt()) {
        ctx.AddConstraint(di.AsExpr(), std::to_string(da.AsInt()));
        merged_vas.PushBack(da);
      } else {
        ctx.AddConstraint(di.AsExpr(), da.AsExpr());
        merged_vas.PushBack(da);
      }
    }
    out.SetValueAsShape(std::move(merged_vas));
  } else if (anchor.HasValueAsShape()) {
    out.SetValueAsShape(anchor.ValueAsShape());
  } else if (inferred.HasValueAsShape()) {
    out.SetValueAsShape(inferred.ValueAsShape());
  }
  // min/max bounds.
  std::optional<double> merged_min;
  if (inferred.HasMin() && anchor.HasMin()) {
    merged_min = std::max(inferred.Min(), anchor.Min());
  } else if (inferred.HasMin()) {
    merged_min = inferred.Min();
  } else if (anchor.HasMin()) {
    merged_min = anchor.Min();
  }
  std::optional<double> merged_max;
  if (inferred.HasMax() && anchor.HasMax()) {
    merged_max = std::min(inferred.Max(), anchor.Max());
  } else if (inferred.HasMax()) {
    merged_max = inferred.Max();
  } else if (anchor.HasMax()) {
    merged_max = anchor.Max();
  }
  if (merged_min.has_value() && merged_max.has_value()) {
    if (*merged_min > *merged_max) {
      if (!strict) {
        return std::nullopt;
      }
      EXT_ENFORCE_INVALID(false,
                          "MergeWithAnchor: incompatible min/max bounds for '" + name + "'.");
    }
    out.SetMinMax(*merged_min, *merged_max);
  } else {
    if (merged_min.has_value()) {
      out.SetMin(*merged_min);
    }
    if (merged_max.has_value()) {
      out.SetMax(*merged_max);
    }
  }
  return out;
}

void MergeAnchorsIntoContext(ShapesContext &ctx, const AnchorMap &anchors, bool strict = true) {
  for (const auto &kv : anchors) {
    const std::string &name = kv.first;
    const OptimTensor &anchor = kv.second;
    if (!ctx.Has(name)) {
      ctx.Set(name, OptimTensor(anchor));
      continue;
    }
    if (strict) {
      std::optional<OptimTensor> merged =
          MergeWithAnchor(ctx, name, ctx.Get(name), anchor, /*strict=*/true);
      if (merged.has_value() && *merged != ctx.Get(name)) {
        ctx.Set(name, std::move(*merged));
      }
    } else {
      // Lenient mode: the inferred shape may legitimately disagree
      // with the anchor (e.g. ``Resize`` has historically reported a
      // smaller output shape than the model declares). Skip the
      // anchor on conflict instead of aborting the whole pipeline so
      // that the well-formed anchors still drive constraint
      // propagation downstream.
      std::optional<OptimTensor> merged =
          MergeWithAnchor(ctx, name, ctx.Get(name), anchor, /*strict=*/false);
      if (merged.has_value() && *merged != ctx.Get(name)) {
        ctx.Set(name, std::move(*merged));
      }
    }
  }
}

// Inserts the dim's expression as an anchor symbol, plus the individual
// tokens it references when it is a compound expression. Anchors coming
// from graph outputs / value_info often carry single symbolic names
// (e.g. ``"N"``) but they may also be arithmetic expressions such as
// ``"s0+seq_len"``. In the latter case we want both the expression and
// its leaf tokens (``s0`` and ``seq_len``) registered as "preferred"
// names so that ``rename_dynamic_dimensions`` can reuse them when
// canonicalising symbols inferred elsewhere in the graph.
void AddDimAnchorSymbols(const OptimDim &dim, std::unordered_set<std::string> &symbols) {
  if (!dim.IsExpr()) {
    return;
  }
  const std::string &expr = dim.AsExpr();
  symbols.insert(expr);
  const std::unordered_set<std::string> tokens = expressions::parse_expression_tokens(expr);
  for (const std::string &token : tokens) {
    if (!token.empty()) {
      symbols.insert(token);
    }
  }
}

std::unordered_set<std::string> CollectAnchorSymbols(const AnchorMap &anchors) {
  std::unordered_set<std::string> symbols;
  for (const auto &kv : anchors) {
    const OptimTensor &tensor = kv.second;
    for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
      AddDimAnchorSymbols(tensor.Shape()[i], symbols);
    }
    if (tensor.HasValueAsShape()) {
      for (std::size_t i = 0; i < tensor.ValueAsShape().Rank(); ++i) {
        AddDimAnchorSymbols(tensor.ValueAsShape()[i], symbols);
      }
    }
  }
  return symbols;
}

// Collects the symbolic dim names attached to a value info entry (and
// the leaf tokens of any compound expressions) into ``symbols``.
void AddValueInfoSymbols(const ValueInfoProto &vi, std::unordered_set<std::string> &symbols) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) {
    return;
  }
  const auto &shape = vi.type().tensor_type().shape();
  for (int j = 0; j < shape.dim_size(); ++j) {
    const auto &dim = shape.dim(j);
    if (!dim.has_dim_param()) {
      continue;
    }
    const std::string param = dim.dim_param().as_string();
    if (param.empty()) {
      continue;
    }
    symbols.insert(param);
    const std::unordered_set<std::string> tokens = expressions::parse_expression_tokens(param);
    for (const std::string &token : tokens) {
      if (!token.empty()) {
        symbols.insert(token);
      }
    }
  }
}

// Collects the symbolic dim names attached to graph inputs, graph outputs,
// and existing value_info entries (along with the leaf tokens of any
// compound expressions). These names are user-provided and must not be
// renamed by anchor-driven propagation: if an output anchor declares ``Y``
// as ``[ANCHOR, 4]`` while ``X`` is declared as ``[N, 4]`` and
// ``Y = Relu(X)``, the merge records the equality ``N == ANCHOR`` but the
// renaming pass should keep ``X`` as ``[N, 4]`` (and ``Y`` as
// ``[ANCHOR, 4]``) instead of forcing one symbol to become the other.
void AddGraphDeclaredSymbols(const GraphProto &graph, std::unordered_set<std::string> &symbols) {
  for (int i = 0; i < graph.input_size(); ++i) {
    AddValueInfoSymbols(graph.input(i), symbols);
  }
  for (int i = 0; i < graph.output_size(); ++i) {
    AddValueInfoSymbols(graph.output(i), symbols);
  }
  for (int i = 0; i < graph.value_info_size(); ++i) {
    AddValueInfoSymbols(graph.value_info(i), symbols);
  }
}

OptimShape
RenameShapeWithReplacements(const OptimShape &shape,
                            const std::unordered_map<std::string, std::string> &replacements) {
  OptimShape renamed;
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    const OptimDim &dim = shape[i];
    if (dim.IsInt()) {
      renamed.PushBack(dim.AsInt());
      continue;
    }
    const std::string &expr = dim.AsExpr();
    auto it = replacements.find(expr);
    if (it != replacements.end()) {
      renamed.PushBack(it->second);
      continue;
    }
    renamed.PushBack(expressions::rename_dynamic_expression(expr, replacements));
  }
  return renamed;
}

void PropagateAnchorConstraintsIntoContext(ShapesContext &ctx, const AnchorMap &anchors,
                                           const GraphProto &graph) {
  if (ctx.ConstraintsSize() == 0) {
    return;
  }
  std::unordered_set<std::string> preferred = CollectAnchorSymbols(anchors);
  AddGraphDeclaredSymbols(graph, preferred);
  if (preferred.empty()) {
    return;
  }
  std::map<std::string, std::unordered_set<std::string>> constraints;
  for (const auto &c : ctx.Constraints()) {
    constraints[c.first].insert(c.second);
    constraints[c.second].insert(c.first);
  }
  const std::map<std::string, std::string> rep =
      expressions::rename_dynamic_dimensions(constraints, preferred);
  if (rep.empty()) {
    return;
  }
  std::unordered_map<std::string, std::string> replacements(rep.begin(), rep.end());

  std::vector<std::string> names;
  names.reserve(ctx.Tensors().size());
  for (const auto &kv : ctx.Tensors()) {
    names.push_back(kv.first);
  }

  for (const std::string &name : names) {
    const OptimTensor &tensor = ctx.Get(name);
    OptimTensor updated(tensor);
    bool changed = false;

    OptimShape renamed_shape = RenameShapeWithReplacements(tensor.Shape(), replacements);
    if (renamed_shape != tensor.Shape()) {
      updated.Shape() = std::move(renamed_shape);
      changed = true;
    }

    if (tensor.HasValueAsShape()) {
      OptimShape renamed_value_shape =
          RenameShapeWithReplacements(tensor.ValueAsShape(), replacements);
      if (renamed_value_shape != tensor.ValueAsShape()) {
        updated.SetValueAsShape(std::move(renamed_value_shape));
        changed = true;
      }
    }

    if (changed) {
      ctx.Set(name, std::move(updated));
    }
  }

  // Final verification pass. Assuming outputs (and the leaf tokens of
  // their dim expressions) are registered as anchors, before returning
  // we double-check that every dim expression and every token inside
  // such an expression has been replaced by its equivalent anchor
  // wherever the ``replacements`` map provides one. The first rewrite
  // pass above can leave a shape stale when one of its dims was newly
  // populated by an earlier iteration over ``names`` (for example a
  // ``value_as_shape`` entry copied verbatim from another tensor), so
  // we repeat the rewrite until ``ctx`` reaches a fixed point.
  bool dirty = true;
  int max_iters = 4;
  while (dirty && max_iters-- > 0) {
    dirty = false;
    for (const std::string &name : names) {
      const OptimTensor &tensor = ctx.Get(name);
      OptimTensor updated(tensor);
      bool changed = false;
      OptimShape renamed_shape = RenameShapeWithReplacements(tensor.Shape(), replacements);
      if (renamed_shape != tensor.Shape()) {
        updated.Shape() = std::move(renamed_shape);
        changed = true;
      }
      if (tensor.HasValueAsShape()) {
        OptimShape renamed_value_shape =
            RenameShapeWithReplacements(tensor.ValueAsShape(), replacements);
        if (renamed_value_shape != tensor.ValueAsShape()) {
          updated.SetValueAsShape(std::move(renamed_value_shape));
          changed = true;
        }
      }
      if (changed) {
        ctx.Set(name, std::move(updated));
        dirty = true;
      }
    }
  }
}

} // namespace

void ShapesContext::CheckInputsAvailable(const NodeProto &node) const {
  for (int i = 0; i < node.input_size(); ++i) {
    const std::string name = node.input(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(Has(name) || HasSequence(name),
                        "CheckInputsAvailable: input '" + name + "' of op '" +
                            node.op_type().as_string() + "' is missing from ShapesContext.");
  }
}

void ShapesContext::CheckOutputsNotAvailable(const NodeProto &node) const {
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(!Has(name) && !HasSequence(name),
                        "CheckOutputsNotAvailable: output '" + name + "' of op '" +
                            node.op_type().as_string() + "' is already present in ShapesContext.");
  }
}

void ShapesContext::ComputeShapeNode(const NodeProto &node) {
  // Model-local function calls bypass the domain check (their domain
  // is arbitrary) and the op-type dispatch table; they are expanded
  // by recursively running shape inference on the FunctionProto body.
  const std::string op_type = node.op_type().as_string();
  const std::string local_key = LocalFunctionKey(node.domain().as_string(), op_type);
  if (const FunctionProto *func = GetLocalFunction(local_key); func != nullptr) {
    CheckInputsAvailable(node);
    CheckOutputsNotAvailable(node);
    ExpandLocalFunctionCall(*this, node, *func);
    return;
  }
  if (const CustomComputeShapeFn *custom_shape_fn =
          GetCustomShapeInferenceFunction(node.domain().as_string(), op_type);
      custom_shape_fn != nullptr) {
    CheckInputsAvailable(node);
    CheckOutputsNotAvailable(node);
    (*custom_shape_fn)(*this, node);
    return;
  }
  CheckOnnxDomain(node);
  CheckInputsAvailable(node);
  CheckOutputsNotAvailable(node);
  const std::string key = NormaliseDispatchDomain(node) + ":" + op_type;
  const auto &table = DispatchTable();
  auto it = table.find(key);
  EXT_ENFORCE_INVALID(it != table.end(), "ComputeShapeNode: unsupported op_type '" + op_type +
                                             "' in domain '" + NormaliseDispatchDomain(node) +
                                             "'.");
  it->second(*this, node);
}

void ShapesContext::ComputeShapes(const utils::RepeatedProtoField<NodeProto> &nodes) {
  for (int i = 0; i < nodes.size(); ++i) {
    ComputeShapeNode(nodes[i]);
  }
}

void ShapesContext::ComputeShapeGraph(const GraphProto &graph) {
  // Seed initializers first so that they shadow any duplicate input
  // (an ONNX initializer may appear both in ``graph.initializer()``
  // and ``graph.input()``; the initializer wins).
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const TensorProto &init = graph.initializer()[i];
    const std::string name = init.name().as_string();
    if (name.empty() || Has(name)) {
      continue;
    }
    OptimTensor tensor;
    if (OptimTensorFromTensorProto(init, tensor)) {
      Set(name, std::move(tensor));
    }
  }
  // Then seed graph inputs (skipping those already known via the
  // initializers or via outer-scope entries carried in ``*this``).
  for (int i = 0; i < graph.input().size(); ++i) {
    const ValueInfoProto &vi = graph.input()[i];
    const std::string name = vi.name().as_string();
    if (name.empty() || Has(name) || HasSequence(name)) {
      continue;
    }
    OptimTensor tensor;
    if (OptimTensorFromValueInfo(vi, tensor)) {
      Set(name, std::move(tensor));
    }
  }
  ComputeShapes(graph.node());
}

void ShapesContext::ComputeShapeModel(const ModelProto &model,
                                      bool prefill_with_value_info_output) {
  for (int i = 0; i < model.opset_import().size(); ++i) {
    const OperatorSetIdProto &osi = model.opset_import()[i];
    SetOpsetVersion(osi.domain().as_string(), static_cast<int>(osi.version()));
  }
  // Register every model-local function so node-level dispatch can
  // expand calls to them. The pointers reference entries owned by
  // ``model`` and remain valid for the duration of this call.
  for (int i = 0; i < model.functions().size(); ++i) {
    SetLocalFunction(&model.functions()[i]);
  }
  EXT_ENFORCE_INVALID(model.has_graph(),
                      "ComputeShapeModel: the ModelProto has no graph to run shape inference on.");
  // Graph outputs are always registered as anchors so that the
  // user-authored output dim expressions (for example ``Y: [2*dnz]``)
  // are propagated back into the intermediate tensors via the symbolic
  // constraint solver. Output anchors are merged leniently: a shape
  // mismatch between an output anchor and the freshly inferred shape
  // is treated as a pre-existing inference imperfection and silently
  // skipped instead of aborting the whole pass. When
  // ``prefill_with_value_info_output`` is set, ``value_info`` entries
  // are additionally registered as anchors and the full anchor set is
  // merged strictly, matching the long-standing prefill contract.
  AnchorMap anchors = prefill_with_value_info_output ? CollectGraphAnchors(model.graph())
                                                     : CollectGraphOutputAnchors(model.graph());
  ComputeShapeGraph(model.graph());
  if (!anchors.empty()) {
    MergeAnchorsIntoContext(*this, anchors, /*strict=*/prefill_with_value_info_output);
    PropagateAnchorConstraintsIntoContext(*this, anchors, model.graph());
  }
}

void ShapesContext::ApplyInferredShapesToGraph(GraphProto &graph) const {
  // Names that already have authoritative type/shape information in
  // the proto and must not be overwritten.
  std::unordered_set<std::string> seeded;
  for (int i = 0; i < graph.input().size(); ++i) {
    seeded.insert(graph.input()[i].name().as_string());
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    seeded.insert(graph.initializer()[i].name().as_string());
  }
  // Update graph outputs in place.
  std::unordered_set<std::string> output_names;
  for (int i = 0; i < graph.output_size(); ++i) {
    ValueInfoProto &vi = *graph.mutable_output(i);
    const std::string name = vi.name().as_string();
    output_names.insert(name);
    if (!name.empty() && Has(name)) {
      OptimTensorToValueInfo(Get(name), vi);
    }
  }
  // Track existing value_info entries to avoid creating duplicates;
  // update them in place when the name matches.
  std::unordered_set<std::string> existing_value_info;
  for (int i = 0; i < graph.value_info_size(); ++i) {
    ValueInfoProto &vi = *graph.mutable_value_info(i);
    const std::string name = vi.name().as_string();
    existing_value_info.insert(name);
    if (!name.empty() && Has(name)) {
      OptimTensorToValueInfo(Get(name), vi);
    }
  }
  // Append a new value_info entry for every other inferred tensor.
  // Iteration order over the unordered map is not specified, so the
  // names are gathered and sorted to make the output deterministic.
  std::vector<std::string> new_names;
  new_names.reserve(Tensors().size());
  for (const auto &kv : Tensors()) {
    const std::string &name = kv.first;
    if (name.empty() || seeded.count(name) != 0 || output_names.count(name) != 0 ||
        existing_value_info.count(name) != 0) {
      continue;
    }
    new_names.push_back(name);
  }
  std::sort(new_names.begin(), new_names.end());
  for (const std::string &name : new_names) {
    const OptimTensor &tensor = Get(name);
    if (TensorTypeToDataType(tensor.Dtype()) == TensorProto::DataType::UNDEFINED) {
      continue;
    }
    ValueInfoProto *vi = graph.add_value_info();
    vi->set_name(name);
    OptimTensorToValueInfo(tensor, *vi);
  }
}

void ShapesContext::ApplyInferredShapesToModel(ModelProto &model) const {
  EXT_ENFORCE_INVALID(
      model.has_graph(),
      "ApplyInferredShapesToModel: the ModelProto has no graph to write shape inference into.");
  ApplyInferredShapesToGraph(*model.mutable_graph());
}

void InferShapesModel(ModelProto &model, bool prefill_with_value_info_output) {
  ShapesContext ctx;
  ctx.ComputeShapeModel(model, prefill_with_value_info_output);
  ctx.ApplyInferredShapesToModel(model);
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
