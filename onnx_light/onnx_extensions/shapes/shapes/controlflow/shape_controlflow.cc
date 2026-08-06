// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/controlflow/shape_controlflow.h"

#include <cstddef>
#include <string>
#include <utility>

#include "onnx_proto/onnx_helper.h"

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::controlflow {

namespace {

// Runs shape inference on the body of ``subgraph`` using a copy of
// ``parent_ctx`` so that outer-scope values referenced from inside the
// sub-graph remain visible. The resulting context (containing both the
// inherited entries and the new ones produced by the sub-graph nodes)
// is returned. When event logging is enabled on ``parent_ctx``, events
// produced during the subgraph inference are propagated back to
// ``parent_ctx`` with ``subgraph_node_index`` and ``subgraph_attr_name``
// set to ``branch_name``.
ShapesContext InferSubgraph(ShapesContext &parent_ctx, const std::string &branch_name,
                            const GraphProto &subgraph) {
  ShapesContext local = parent_ctx;
  local.set_current_subgraph(local.current_node_index(), branch_name);
  for (int i = 0; i < static_cast<int>(subgraph.initializer().size()); ++i) {
    const TensorProto &init = subgraph.initializer()[i];
    const std::string name = init.name();
    if (name.empty() || local.Has(name)) {
      continue;
    }
    SymTensor tensor;
    if (SymTensorFromTensorProto(init, tensor)) {
      local.Set(name, std::move(tensor));
    }
  }
  const size_t events_before = local.Events().size();
  local.ComputeShapes(subgraph.node());
  if (parent_ctx.events_enabled()) {
    const auto &local_events = local.Events();
    for (size_t i = events_before; i < local_events.size(); ++i) {
      parent_ctx.Events().push_back(local_events[i]);
    }
  }
  // Retain the child context on the parent so the subgraph internals stay
  // inspectable once the parent inference has completed.
  parent_ctx.RegisterSubgraphContext(parent_ctx.current_node_index(), branch_name, local);
  return local;
}

// Retrieves the SymTensor describing the i-th output of ``subgraph``
// from ``local_ctx``. The output name is taken from the sub-graph's
// ValueInfoProto output list. Throws std::invalid_argument when the
// sub-graph has fewer outputs than ``expected`` or when the named
// value is unknown / is a sequence rather than a tensor.
const SymTensor &GetSubgraphOutput(const ShapesContext &local_ctx, const GraphProto &subgraph,
                                   const char *branch_name, int output_index, int expected) {
  EXT_ENFORCE_INVALID(static_cast<int>(subgraph.output().size()) == expected,
                      "ComputeShapeIf: sub-graph '", branch_name, "' declares ",
                      std::to_string(subgraph.output().size()), " output(s), expected ",
                      std::to_string(expected), ".");
  const std::string name = subgraph.output()[output_index].name();
  EXT_ENFORCE_INVALID(local_ctx.Has(name), "ComputeShapeIf: output '", name, "' of sub-graph '",
                      branch_name, "' is missing from the inferred context.");
  return local_ctx.Get(name);
}

// Merges two output descriptors coming from the ``then_branch`` and
// ``else_branch`` sub-graphs into a single SymTensor describing the
// corresponding output of the ``If`` node. When a merged dimension is
// made fully symbolic (because the two branches disagree on that
// dimension), an upper-bound constraint ``merged_dim <= max(then_dim,
// else_dim)`` is recorded into ``ctx`` so downstream passes know the
// merged dim cannot exceed either branch's value.
SymTensor MergeBranchOutputs(ShapesContext &ctx, const SymTensor &then_t, const SymTensor &else_t,
                             const std::string &if_output_name) {
  const TensorType dtype =
      (then_t.Dtype() == else_t.Dtype()) ? then_t.Dtype() : TensorType::kUndefined;

  const SymShape &then_shape = then_t.Shape();
  const SymShape &else_shape = else_t.Shape();

  if (then_shape == else_shape) {
    return SymTensor(nullptr, dtype, then_shape);
  }
  EXT_ENFORCE_INVALID(then_shape.Rank() == else_shape.Rank(),
                      "ComputeShapeIf: rank mismatch between branches for output '", if_output_name,
                      "': then_branch has rank ", std::to_string(then_shape.Rank()),
                      ", else_branch has rank ", std::to_string(else_shape.Rank()),
                      ". This is not supposed to happen for a well-formed If node.");

  SymShape merged;
  for (std::size_t i = 0; i < then_shape.Rank(); ++i) {
    if (then_shape[i] == else_shape[i]) {
      merged.PushBack(then_shape[i]);
    } else {
      const std::string sym = std::string("If_") + if_output_name + "_d" + std::to_string(i);
      // The merged dim equals one of the two branch dims at runtime, so
      // it is upper-bounded by their maximum.
      const expressions::DimType bound =
          expressions::dim_max(ToDimType(then_shape[i]), ToDimType(else_shape[i]));
      ctx.AddLessEqualConstraint(sym, expressions::dim_to_string(bound));
      merged.PushBack(SymDim(sym));
    }
  }
  return SymTensor(nullptr, dtype, std::move(merged));
}

} // namespace

void ComputeShapeIf(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "If", "ComputeShapeIf");

  EXT_ENFORCE_INVALID(node.input_size() == 1,
                      "ComputeShapeIf: op 'If' expects exactly one input (cond), got ",
                      std::to_string(node.input_size()), ".");

  const GraphProto &then_branch = FindGraphAttribute(node, "then_branch", "ComputeShapeIf");
  const GraphProto &else_branch = FindGraphAttribute(node, "else_branch", "ComputeShapeIf");

  const int n_outputs = node.output_size();
  EXT_ENFORCE_INVALID(static_cast<int>(then_branch.output().size()) == n_outputs,
                      "ComputeShapeIf: 'then_branch' sub-graph declares ",
                      std::to_string(then_branch.output().size()), " output(s), expected ",
                      std::to_string(n_outputs), ".");
  EXT_ENFORCE_INVALID(static_cast<int>(else_branch.output().size()) == n_outputs,
                      "ComputeShapeIf: 'else_branch' sub-graph declares ",
                      std::to_string(else_branch.output().size()), " output(s), expected ",
                      std::to_string(n_outputs), ".");

  const ShapesContext then_ctx = InferSubgraph(ctx, "then_branch", then_branch);
  const ShapesContext else_ctx = InferSubgraph(ctx, "else_branch", else_branch);

  for (int i = 0; i < n_outputs; ++i) {
    const std::string out_name = node.output(i);
    if (out_name.empty()) {
      continue; // Optional output not produced.
    }
    const SymTensor &then_t = GetSubgraphOutput(then_ctx, then_branch, "then_branch", i, n_outputs);
    const SymTensor &else_t = GetSubgraphOutput(else_ctx, else_branch, "else_branch", i, n_outputs);
    ctx.Set(out_name, MergeBranchOutputs(ctx, then_t, else_t, out_name));
  }
}

namespace {

// Builds a fully-symbolic shape of the given ``rank``, where each axis
// is named ``"<prefix>_d<i>"``. Used when a loop-carried or scan output
// shape cannot be reproduced verbatim from the body subgraph.
SymShape SymbolicShape(std::size_t rank, const std::string &prefix) {
  SymShape s;
  for (std::size_t i = 0; i < rank; ++i) {
    s.PushBack(SymDim(prefix + "_d" + std::to_string(i)));
  }
  return s;
}

} // namespace

void ComputeShapeLoop(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Loop", "ComputeShapeLoop");

  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeLoop: op 'Loop' expects at least two inputs (M, cond), got ",
                      std::to_string(node.input_size()), ".");

  const GraphProto &body = FindGraphAttribute(node, "body", "ComputeShapeLoop");

  // N = number of loop-carried dependencies = node.input_size() - 2.
  // K = number of scan outputs = node.output_size() - N.
  // The body declares 2 + N inputs and 1 + N + K outputs.
  const int n_carried = node.input_size() - 2;
  EXT_ENFORCE_INVALID(n_carried >= 0,
                      "ComputeShapeLoop: invalid number of loop-carried dependencies.");
  EXT_ENFORCE_INVALID(node.output_size() >= n_carried, "ComputeShapeLoop: Loop node declares ",
                      std::to_string(node.output_size()),
                      " output(s), expected at least N=", std::to_string(n_carried),
                      " loop-carried outputs.");
  const int k_scan = node.output_size() - n_carried;

  EXT_ENFORCE_INVALID(static_cast<int>(body.input().size()) == n_carried + 2,
                      "ComputeShapeLoop: 'body' sub-graph declares ",
                      std::to_string(body.input().size()),
                      " input(s), expected 2 + N = ", std::to_string(n_carried + 2), ".");
  EXT_ENFORCE_INVALID(
      static_cast<int>(body.output().size()) == n_carried + k_scan + 1,
      "ComputeShapeLoop: 'body' sub-graph declares ", std::to_string(body.output().size()),
      " output(s), expected 1 + N + K = ", std::to_string(n_carried + k_scan + 1), ".");

  // Seed a child context with the body's formal input descriptors so that
  // shape inference can walk the body. The first two body inputs are the
  // iteration number (INT64 scalar) and the incoming termination
  // condition (BOOL scalar); the remaining N are the loop-carried
  // dependency values, inherited from the matching ``v_initial`` outer
  // descriptor.
  ShapesContext local = ctx;
  local.set_current_subgraph(local.current_node_index(), "body");
  for (int i = 0; i < static_cast<int>(body.initializer().size()); ++i) {
    const TensorProto &init = body.initializer()[i];
    const std::string name = init.name();
    if (name.empty() || local.Has(name)) {
      continue;
    }
    SymTensor tensor;
    if (SymTensorFromTensorProto(init, tensor)) {
      local.Set(name, std::move(tensor));
    }
  }
  local.Set(body.input()[0].name(), SymTensor(nullptr, TensorType::kInt64, SymShape{}));
  local.Set(body.input()[1].name(), SymTensor(nullptr, TensorType::kBool, SymShape{}));
  for (int i = 0; i < n_carried; ++i) {
    const std::string v_initial_name = node.input(2 + i);
    EXT_ENFORCE_INVALID(!v_initial_name.empty(), "ComputeShapeLoop: 'v_initial' input #",
                        std::to_string(i), " has an empty name.");
    EXT_ENFORCE_INVALID(local.Has(v_initial_name), "ComputeShapeLoop: 'v_initial' input '",
                        v_initial_name, "' is missing from the inferred context.");
    local.Set(body.input()[2 + i].name(), SymTensor(local.Get(v_initial_name)));
  }

  const size_t events_before = local.Events().size();
  local.ComputeShapes(body.node());
  if (ctx.events_enabled()) {
    const auto &local_events = local.Events();
    for (size_t i = events_before; i < local_events.size(); ++i) {
      ctx.Events().push_back(local_events[i]);
    }
  }

  // Validate that every body output is known in the local context.
  for (int i = 0; i < static_cast<int>(body.output().size()); ++i) {
    const std::string body_out = body.output()[i].name();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeLoop: body output '", body_out,
                        "' is missing from the inferred context.");
  }

  // N loop-carried outputs: dtype is taken from ``v_out`` and validated
  // against ``v_initial``; shape is kept when it matches the initial shape
  // (a common case the body just identity-forwards) and made fully
  // symbolic otherwise to model the fact that the body may grow/shrink
  // the carried tensor's shape across iterations.
  for (int i = 0; i < n_carried; ++i) {
    const std::string node_out = node.output(i);
    if (node_out.empty()) {
      continue;
    }
    const SymTensor &v_initial = ctx.Get(node.input(2 + i));
    const SymTensor &v_out = local.Get(body.output()[1 + i].name());
    EXT_ENFORCE_INVALID(v_out.Dtype() == v_initial.Dtype(), "ComputeShapeLoop: body output #",
                        std::to_string(1 + i),
                        " has a different element type than the matching 'v_initial' input.");
    SymShape out_shape = (v_out.Shape() == v_initial.Shape())
                             ? v_initial.Shape()
                             : SymbolicShape(v_out.Shape().Rank(), "Loop_" + node_out);
    ctx.Set(node_out, SymTensor(nullptr, v_initial.Dtype(), std::move(out_shape)));
  }

  // K scan outputs: dtype is taken from the body's scan output; shape is
  // the body's shape prefixed by the trip count dimension. When M's
  // ValueAsShape carries a single element (the symbolic or concrete trip
  // count), that element is used as the leading axis; otherwise a fresh
  // symbolic dim is introduced.
  SymDim trip_dim;
  {
    const std::string m_name = node.input(0);
    bool resolved = false;
    if (!m_name.empty() && ctx.Has(m_name)) {
      const SymTensor &m_tensor = ctx.Get(m_name);
      if (m_tensor.HasValueAsShape() && m_tensor.ValueAsShape().Rank() == 1) {
        trip_dim = m_tensor.ValueAsShape()[0];
        resolved = true;
      }
    }
    if (!resolved) {
      trip_dim = SymDim("Loop_trip");
    }
  }
  for (int k = 0; k < k_scan; ++k) {
    const std::string node_out = node.output(n_carried + k);
    if (node_out.empty()) {
      continue;
    }
    const SymTensor &scan_out = local.Get(body.output()[1 + n_carried + k].name());
    SymShape stacked;
    stacked.PushBack(trip_dim);
    for (std::size_t d = 0; d < scan_out.Shape().Rank(); ++d) {
      stacked.PushBack(scan_out.Shape()[d]);
    }
    ctx.Set(node_out, SymTensor(nullptr, scan_out.Dtype(), std::move(stacked)));
  }

  // Retain the body's child context so its internals stay inspectable
  // once the parent inference has completed.
  ctx.RegisterSubgraphContext(ctx.current_node_index(), "body", std::move(local));
}

namespace {

// Returns the value of an INT scalar attribute or throws if absent.
int64_t RequireIntAttribute(const NodeProto &node, const char *name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, "ComputeShapeScan: missing required INT attribute '", name,
                      "'.");
  EXT_ENFORCE_INVALID(attr->type() == AttributeProto::AttributeType::INT,
                      "ComputeShapeScan: attribute '", name, "' must be of type INT.");
  return attr->i();
}

// Normalizes an axis in the range [-rank, rank-1] to a non-negative value.
int64_t NormalizeAxis(int64_t axis, std::size_t rank, const char *attr_name) {
  const int64_t r = static_cast<int64_t>(rank);
  if (axis < 0) {
    axis += r;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis <= r, "ComputeShapeScan: '", attr_name,
                      "' out of range for rank ", std::to_string(rank), ".");
  return axis;
}

} // namespace

void ComputeShapeScan(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Scan", "ComputeShapeScan");

  const GraphProto &body = FindGraphAttribute(node, "body", "ComputeShapeScan");
  const int64_t num_scan_inputs64 = RequireIntAttribute(node, "num_scan_inputs");
  EXT_ENFORCE_INVALID(num_scan_inputs64 > 0,
                      "ComputeShapeScan: 'num_scan_inputs' must be strictly positive, got ",
                      std::to_string(num_scan_inputs64), ".");
  const int num_scan_inputs = static_cast<int>(num_scan_inputs64);

  // Scan opset 8 prepends an optional ``sequence_lens`` input (which is not
  // a state variable and has no corresponding body input). Detect it from
  // the model's default-domain opset version and skip that slot when
  // counting / indexing state inputs.
  const int opset =
      ctx.HasOpsetVersion(kOnnxDomain) ? ctx.OpsetVersion(kOnnxDomain) : kUnknownOpsetVersion;
  const bool is_scan8 = opset >= 1 && opset <= 8;
  const int scan8_offset = is_scan8 ? 1 : 0;

  EXT_ENFORCE_INVALID(
      node.input_size() >= num_scan_inputs + scan8_offset,
      "ComputeShapeScan: 'Scan' node declares ", std::to_string(node.input_size()),
      " input(s), expected at least num_scan_inputs = ", std::to_string(num_scan_inputs), ".");
  const int n_state = node.input_size() - num_scan_inputs - scan8_offset;
  EXT_ENFORCE_INVALID(node.output_size() >= n_state, "ComputeShapeScan: Scan node declares ",
                      std::to_string(node.output_size()),
                      " output(s), expected at least N=", std::to_string(n_state),
                      " state outputs.");
  const int k_scan = node.output_size() - n_state;

  EXT_ENFORCE_INVALID(
      static_cast<int>(body.input().size()) == n_state + num_scan_inputs,
      "ComputeShapeScan: 'body' sub-graph declares ", std::to_string(body.input().size()),
      " input(s), expected N + M = ", std::to_string(n_state + num_scan_inputs), ".");
  EXT_ENFORCE_INVALID(static_cast<int>(body.output().size()) == n_state + k_scan,
                      "ComputeShapeScan: 'body' sub-graph declares ",
                      std::to_string(body.output().size()),
                      " output(s), expected N + K = ", std::to_string(n_state + k_scan), ".");

  // Scan opset 8 does not have scan_input_axes / scan_output_axes; the scan
  // axis is always 1 of each batched input (axis 0 after stripping the batch
  // dimension). Opset 9+ supports explicit per-input/output axes.
  std::vector<int64_t> scan_input_axes;
  std::vector<int64_t> scan_output_axes;
  if (!is_scan8) {
    // Optional scan_input_axes attribute (per scan input).
    GetAttributeInts(node, "scan_input_axes", scan_input_axes);
    EXT_ENFORCE_INVALID(scan_input_axes.empty() ||
                            scan_input_axes.size() == static_cast<std::size_t>(num_scan_inputs),
                        "ComputeShapeScan: 'scan_input_axes' must have num_scan_inputs entries.");

    // Optional scan_output_axes attribute (per scan output).
    GetAttributeInts(node, "scan_output_axes", scan_output_axes);
    EXT_ENFORCE_INVALID(scan_output_axes.empty() ||
                            scan_output_axes.size() == static_cast<std::size_t>(k_scan),
                        "ComputeShapeScan: 'scan_output_axes' must have K entries.");
  }

  // Build a child context with the body's formal input descriptors so that
  // shape inference can walk the body. The first N body inputs are the
  // state variables (inherited from the matching node inputs); the
  // remaining M are per-iteration scan-input slices, obtained by dropping
  // the scan axis from the matching scan_input shape.
  //
  // For Scan opset 8, every state and scan input carries a leading batch
  // dimension B. The body inputs are batch-stripped (rank reduced by one):
  //   state input [B, D...] → body input [D...]
  //   scan input  [B, T, D...] → body input [D...]  (B and T stripped)
  ShapesContext local = ctx;
  local.set_current_subgraph(local.current_node_index(), "body");
  for (int i = 0; i < static_cast<int>(body.initializer().size()); ++i) {
    const TensorProto &init = body.initializer()[i];
    const std::string name = init.name();
    if (name.empty() || local.Has(name)) {
      continue;
    }
    SymTensor tensor;
    if (SymTensorFromTensorProto(init, tensor)) {
      local.Set(name, std::move(tensor));
    }
  }

  // Batch dimension for Scan opset 8 (shared by all batched inputs/outputs).
  // Use the first scan input in the node (after the sequence_lens slot) as the
  // anchor for the symbolic name — this avoids accessing node.output(0), which
  // could be out-of-bounds if the node has no outputs.
  const std::string &scan8_anchor =
      is_scan8 ? node.input(scan8_offset < node.input_size() ? scan8_offset : 0)
               : utils::String::empty_string();
  SymDim batch_dim(std::string("Scan8_") + scan8_anchor + "_batch");

  for (int i = 0; i < n_state; ++i) {
    const std::string state_in_name = node.input(scan8_offset + i);
    EXT_ENFORCE_INVALID(local.Has(state_in_name), "ComputeShapeScan: state input '", state_in_name,
                        "' is missing from the inferred context.");
    const SymTensor &state_in = local.Get(state_in_name);
    if (is_scan8) {
      // Strip leading batch dimension; body sees [D...].
      EXT_ENFORCE_INVALID(state_in.Shape().Rank() >= 1,
                          "ComputeShapeScan: Scan opset 8 state input '", state_in_name,
                          "' must have rank >= 1.");
      if (i == 0) {
        batch_dim = state_in.Shape()[0];
      }
      SymShape body_state_shape;
      for (std::size_t d = 1; d < state_in.Shape().Rank(); ++d) {
        body_state_shape.PushBack(state_in.Shape()[d]);
      }
      local.Set(body.input()[i].name(),
                SymTensor(nullptr, state_in.Dtype(), std::move(body_state_shape)));
    } else {
      local.Set(body.input()[i].name(), SymTensor(state_in));
    }
  }

  // The trip count is taken from the first scan input's scan axis.
  SymDim trip_count_dim(std::string("Scan_") + node.output(0) + "_trip_count");
  bool trip_count_known = false;

  for (int m = 0; m < num_scan_inputs; ++m) {
    const std::string scan_in_name = node.input(scan8_offset + n_state + m);
    EXT_ENFORCE_INVALID(local.Has(scan_in_name), "ComputeShapeScan: scan input '", scan_in_name,
                        "' is missing from the inferred context.");
    const SymTensor &scan_in = local.Get(scan_in_name);
    EXT_ENFORCE_INVALID(scan_in.Shape().Rank() >= 1, "ComputeShapeScan: scan input '", scan_in_name,
                        "' must have rank >= 1.");

    SymShape body_in_shape;
    if (is_scan8) {
      // Scan opset 8: input shape is [B, T, D...]. Strip both B (axis 0) and
      // T (axis 1); body input sees [D...]. Trip count = T = axis 1.
      EXT_ENFORCE_INVALID(scan_in.Shape().Rank() >= 2,
                          "ComputeShapeScan: Scan opset 8 scan input '", scan_in_name,
                          "' must have rank >= 2.");
      if (m == 0) {
        trip_count_dim = scan_in.Shape()[1];
        trip_count_known = true;
        // When there are no state inputs (n_state=0), take the batch dimension
        // from the first scan input.
        if (n_state == 0) {
          batch_dim = scan_in.Shape()[0];
        }
      }
      for (std::size_t d = 2; d < scan_in.Shape().Rank(); ++d) {
        body_in_shape.PushBack(scan_in.Shape()[d]);
      }
    } else {
      const int64_t axis_raw = scan_input_axes.empty() ? 0 : scan_input_axes[m];
      const int64_t axis = NormalizeAxis(axis_raw, scan_in.Shape().Rank(), "scan_input_axes");
      if (m == 0) {
        trip_count_dim = scan_in.Shape()[static_cast<std::size_t>(axis)];
        trip_count_known = true;
      }
      // Body input shape for this scan input = scan_in.shape with axis removed.
      for (std::size_t d = 0; d < scan_in.Shape().Rank(); ++d) {
        if (static_cast<int64_t>(d) == axis) {
          continue;
        }
        body_in_shape.PushBack(scan_in.Shape()[d]);
      }
    }
    local.Set(body.input()[n_state + m].name(),
              SymTensor(nullptr, scan_in.Dtype(), std::move(body_in_shape)));
  }

  const size_t events_before_scan = local.Events().size();
  local.ComputeShapes(body.node());
  if (ctx.events_enabled()) {
    const auto &local_events = local.Events();
    for (size_t i = events_before_scan; i < local_events.size(); ++i) {
      ctx.Events().push_back(local_events[i]);
    }
  }

  for (int i = 0; i < static_cast<int>(body.output().size()); ++i) {
    const std::string body_out = body.output()[i].name();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeScan: body output '", body_out,
                        "' is missing from the inferred context.");
  }

  // N state outputs: dtype/shape are taken from the body's v_out (and
  // validated against v_initial when shapes agree).
  // For Scan opset 8, restore the leading batch dimension on state outputs.
  for (int i = 0; i < n_state; ++i) {
    const std::string node_out = node.output(i);
    if (node_out.empty()) {
      continue;
    }
    const SymTensor &state_in = ctx.Get(node.input(scan8_offset + i));
    const SymTensor &v_out = local.Get(body.output()[i].name());
    EXT_ENFORCE_INVALID(v_out.Dtype() == state_in.Dtype(), "ComputeShapeScan: body output #",
                        std::to_string(i),
                        " has a different element type than the matching state input.");
    if (is_scan8) {
      // Output shape = [B, D...] where D... is the body output shape.
      SymShape out_shape;
      out_shape.PushBack(batch_dim);
      for (std::size_t d = 0; d < v_out.Shape().Rank(); ++d) {
        out_shape.PushBack(v_out.Shape()[d]);
      }
      ctx.Set(node_out, SymTensor(nullptr, state_in.Dtype(), std::move(out_shape)));
    } else {
      SymShape out_shape = (v_out.Shape() == state_in.Shape())
                               ? state_in.Shape()
                               : SymbolicShape(v_out.Shape().Rank(), "Scan_" + node_out);
      ctx.Set(node_out, SymTensor(nullptr, state_in.Dtype(), std::move(out_shape)));
    }
  }

  // K scan outputs: dtype is the body's scan-output element dtype; shape
  // is the body's per-iteration scan-output shape with a new axis of
  // length ``trip_count_dim`` inserted at position scan_output_axes[k]
  // (default 0).
  // For Scan opset 8, an additional leading batch dimension B is also
  // prepended so the output shape is [B, T, D...].
  for (int k = 0; k < k_scan; ++k) {
    const std::string node_out = node.output(n_state + k);
    if (node_out.empty()) {
      continue;
    }
    const SymTensor &scan_out_elt = local.Get(body.output()[n_state + k].name());
    const SymDim trip_dim =
        trip_count_known ? trip_count_dim : SymDim(std::string("Scan_") + node_out + "_trip");

    if (is_scan8) {
      // Output shape = [B, T, D...].
      SymShape stacked;
      stacked.PushBack(batch_dim);
      stacked.PushBack(trip_dim);
      for (std::size_t d = 0; d < scan_out_elt.Shape().Rank(); ++d) {
        stacked.PushBack(scan_out_elt.Shape()[d]);
      }
      ctx.Set(node_out, SymTensor(nullptr, scan_out_elt.Dtype(), std::move(stacked)));
    } else {
      const int64_t axis_raw = scan_output_axes.empty() ? 0 : scan_output_axes[k];
      // Output rank = elt rank + 1.
      const int64_t axis =
          NormalizeAxis(axis_raw, scan_out_elt.Shape().Rank() + 1, "scan_output_axes");
      SymShape stacked;
      for (std::size_t d = 0; d <= scan_out_elt.Shape().Rank(); ++d) {
        if (static_cast<int64_t>(d) == axis) {
          stacked.PushBack(trip_dim);
        }
        if (d < scan_out_elt.Shape().Rank()) {
          stacked.PushBack(scan_out_elt.Shape()[d]);
        }
      }
      ctx.Set(node_out, SymTensor(nullptr, scan_out_elt.Dtype(), std::move(stacked)));
    }
  }

  // Retain the body's child context so its internals stay inspectable
  // once the parent inference has completed.
  ctx.RegisterSubgraphContext(ctx.current_node_index(), "body", std::move(local));
}

} // namespace shapes::controlflow
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
