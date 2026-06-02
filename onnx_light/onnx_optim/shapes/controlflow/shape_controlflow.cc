// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/controlflow/shape_controlflow.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_proto/onnx_helper.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_optim/shapes/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace controlflow {

namespace {

// Runs shape inference on the body of ``subgraph`` using a copy of
// ``parent_ctx`` so that outer-scope values referenced from inside the
// sub-graph remain visible. The resulting context (containing both the
// inherited entries and the new ones produced by the sub-graph nodes)
// is returned.
ShapesContext InferSubgraph(const ShapesContext &parent_ctx, const GraphProto &subgraph) {
  ShapesContext local = parent_ctx;
  ComputeShapes(local, subgraph.node());
  return local;
}

// Retrieves the OptimTensor describing the i-th output of ``subgraph``
// from ``local_ctx``. The output name is taken from the sub-graph's
// ValueInfoProto output list. Throws std::invalid_argument when the
// sub-graph has fewer outputs than ``expected`` or when the named
// value is unknown / is a sequence rather than a tensor.
const OptimTensor &GetSubgraphOutput(const ShapesContext &local_ctx, const GraphProto &subgraph,
                                     const char *branch_name, int output_index, int expected) {
  EXT_ENFORCE_INVALID(subgraph.output().size() == expected,
                      std::string("ComputeShapeIf: sub-graph '") + branch_name + "' declares " +
                          std::to_string(subgraph.output().size()) + " output(s), expected " +
                          std::to_string(expected) + ".");
  const std::string name = subgraph.output()[output_index].name().as_string();
  EXT_ENFORCE_INVALID(local_ctx.Has(name), std::string("ComputeShapeIf: output '") + name +
                                               "' of sub-graph '" + branch_name +
                                               "' is missing from the inferred context.");
  return local_ctx.Get(name);
}

// Merges two output descriptors coming from the ``then_branch`` and
// ``else_branch`` sub-graphs into a single OptimTensor describing the
// corresponding output of the ``If`` node.
OptimTensor MergeBranchOutputs(const OptimTensor &then_t, const OptimTensor &else_t,
                               const std::string &if_output_name) {
  const TensorType dtype =
      (then_t.Dtype() == else_t.Dtype()) ? then_t.Dtype() : TensorType::kUndefined;

  const OptimShape &then_shape = then_t.Shape();
  const OptimShape &else_shape = else_t.Shape();

  if (then_shape == else_shape) {
    return OptimTensor(nullptr, dtype, then_shape);
  }
  EXT_ENFORCE_INVALID(then_shape.Rank() == else_shape.Rank(),
                      std::string("ComputeShapeIf: rank mismatch between branches for output '") +
                          if_output_name + "': then_branch has rank " +
                          std::to_string(then_shape.Rank()) + ", else_branch has rank " +
                          std::to_string(else_shape.Rank()) +
                          ". This is not supposed to happen for a well-formed If node.");

  OptimShape merged;
  for (std::size_t i = 0; i < then_shape.Rank(); ++i) {
    if (then_shape[i] == else_shape[i]) {
      merged.PushBack(then_shape[i]);
    } else {
      merged.PushBack(OptimDim(std::string("If_") + if_output_name + "_d" + std::to_string(i)));
    }
  }
  return OptimTensor(nullptr, dtype, std::move(merged));
}

} // namespace

void ComputeShapeIf(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "If", "ComputeShapeIf");

  EXT_ENFORCE_INVALID(node.input_size() == 1,
                      "ComputeShapeIf: op 'If' expects exactly one input (cond), got " +
                          std::to_string(node.input_size()) + ".");

  const GraphProto &then_branch = FindGraphAttribute(node, "then_branch", "ComputeShapeIf");
  const GraphProto &else_branch = FindGraphAttribute(node, "else_branch", "ComputeShapeIf");

  const int n_outputs = node.output_size();
  EXT_ENFORCE_INVALID(then_branch.output().size() == n_outputs,
                      "ComputeShapeIf: 'then_branch' sub-graph declares " +
                          std::to_string(then_branch.output().size()) + " output(s), expected " +
                          std::to_string(n_outputs) + ".");
  EXT_ENFORCE_INVALID(else_branch.output().size() == n_outputs,
                      "ComputeShapeIf: 'else_branch' sub-graph declares " +
                          std::to_string(else_branch.output().size()) + " output(s), expected " +
                          std::to_string(n_outputs) + ".");

  const ShapesContext then_ctx = InferSubgraph(ctx, then_branch);
  const ShapesContext else_ctx = InferSubgraph(ctx, else_branch);

  for (int i = 0; i < n_outputs; ++i) {
    const std::string out_name = node.output(i).as_string();
    if (out_name.empty()) {
      continue; // Optional output not produced.
    }
    const OptimTensor &then_t =
        GetSubgraphOutput(then_ctx, then_branch, "then_branch", i, n_outputs);
    const OptimTensor &else_t =
        GetSubgraphOutput(else_ctx, else_branch, "else_branch", i, n_outputs);
    ctx.Set(out_name, MergeBranchOutputs(then_t, else_t, out_name));
  }
}

namespace {

// Builds a fully-symbolic shape of the given ``rank``, where each axis
// is named ``"<prefix>_d<i>"``. Used when a loop-carried or scan output
// shape cannot be reproduced verbatim from the body subgraph.
OptimShape SymbolicShape(std::size_t rank, const std::string &prefix) {
  OptimShape s;
  for (std::size_t i = 0; i < rank; ++i) {
    s.PushBack(OptimDim(prefix + "_d" + std::to_string(i)));
  }
  return s;
}

} // namespace

void ComputeShapeLoop(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Loop", "ComputeShapeLoop");

  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeLoop: op 'Loop' expects at least two inputs (M, cond), got " +
                          std::to_string(node.input_size()) + ".");

  const GraphProto &body = FindGraphAttribute(node, "body", "ComputeShapeLoop");

  // N = number of loop-carried dependencies = node.input_size() - 2.
  // K = number of scan outputs = node.output_size() - N.
  // The body declares 2 + N inputs and 1 + N + K outputs.
  const int n_carried = node.input_size() - 2;
  EXT_ENFORCE_INVALID(n_carried >= 0,
                      "ComputeShapeLoop: invalid number of loop-carried dependencies.");
  EXT_ENFORCE_INVALID(node.output_size() >= n_carried,
                      "ComputeShapeLoop: Loop node declares " + std::to_string(node.output_size()) +
                          " output(s), expected at least N=" + std::to_string(n_carried) +
                          " loop-carried outputs.");
  const int k_scan = node.output_size() - n_carried;

  EXT_ENFORCE_INVALID(body.input().size() == n_carried + 2,
                      "ComputeShapeLoop: 'body' sub-graph declares " +
                          std::to_string(body.input().size()) +
                          " input(s), expected 2 + N = " + std::to_string(n_carried + 2) + ".");
  EXT_ENFORCE_INVALID(
      body.output().size() == n_carried + k_scan + 1,
      "ComputeShapeLoop: 'body' sub-graph declares " + std::to_string(body.output().size()) +
          " output(s), expected 1 + N + K = " + std::to_string(n_carried + k_scan + 1) + ".");

  // Seed a child context with the body's formal input descriptors so that
  // shape inference can walk the body. The first two body inputs are the
  // iteration number (INT64 scalar) and the incoming termination
  // condition (BOOL scalar); the remaining N are the loop-carried
  // dependency values, inherited from the matching ``v_initial`` outer
  // descriptor.
  ShapesContext local = ctx;
  local.Set(body.input()[0].name().as_string(),
            OptimTensor(nullptr, TensorType::kInt64, OptimShape{}));
  local.Set(body.input()[1].name().as_string(),
            OptimTensor(nullptr, TensorType::kBool, OptimShape{}));
  for (int i = 0; i < n_carried; ++i) {
    const std::string v_initial_name = node.input(2 + i).as_string();
    EXT_ENFORCE_INVALID(!v_initial_name.empty(), "ComputeShapeLoop: 'v_initial' input #" +
                                                     std::to_string(i) + " has an empty name.");
    EXT_ENFORCE_INVALID(local.Has(v_initial_name), "ComputeShapeLoop: 'v_initial' input '" +
                                                       v_initial_name +
                                                       "' is missing from the inferred context.");
    local.Set(body.input()[2 + i].name().as_string(), OptimTensor(local.Get(v_initial_name)));
  }

  ComputeShapes(local, body.node());

  // Validate that every body output is known in the local context.
  for (int i = 0; i < body.output().size(); ++i) {
    const std::string body_out = body.output()[i].name().as_string();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeLoop: body output '" + body_out +
                                                 "' is missing from the inferred context.");
  }

  // N loop-carried outputs: dtype is taken from ``v_out`` and validated
  // against ``v_initial``; shape is kept when it matches the initial shape
  // (a common case the body just identity-forwards) and made fully
  // symbolic otherwise to model the fact that the body may grow/shrink
  // the carried tensor's shape across iterations.
  for (int i = 0; i < n_carried; ++i) {
    const std::string node_out = node.output(i).as_string();
    if (node_out.empty()) {
      continue;
    }
    const OptimTensor &v_initial = ctx.Get(node.input(2 + i).as_string());
    const OptimTensor &v_out = local.Get(body.output()[1 + i].name().as_string());
    EXT_ENFORCE_INVALID(v_out.Dtype() == v_initial.Dtype(),
                        "ComputeShapeLoop: body output #" + std::to_string(1 + i) +
                            " has a different element type than the matching 'v_initial' input.");
    OptimShape out_shape = (v_out.Shape() == v_initial.Shape())
                               ? v_initial.Shape()
                               : SymbolicShape(v_out.Shape().Rank(), "Loop_" + node_out);
    ctx.Set(node_out, OptimTensor(nullptr, v_initial.Dtype(), std::move(out_shape)));
  }

  // K scan outputs: dtype is taken from the body's scan output; shape is
  // the body's shape prefixed by a symbolic leading axis (the trip count).
  for (int k = 0; k < k_scan; ++k) {
    const std::string node_out = node.output(n_carried + k).as_string();
    if (node_out.empty()) {
      continue;
    }
    const OptimTensor &scan_out = local.Get(body.output()[1 + n_carried + k].name().as_string());
    OptimShape stacked;
    stacked.PushBack(OptimDim("Loop_" + node_out + "_d0"));
    for (std::size_t d = 0; d < scan_out.Shape().Rank(); ++d) {
      stacked.PushBack(scan_out.Shape()[d]);
    }
    ctx.Set(node_out, OptimTensor(nullptr, scan_out.Dtype(), std::move(stacked)));
  }
}

namespace {

// Returns the value of an INT scalar attribute or throws if absent.
int64_t RequireIntAttribute(const NodeProto &node, const char *name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr,
                      std::string("ComputeShapeScan: missing required INT attribute '") + name +
                          "'.");
  EXT_ENFORCE_INVALID(attr->type() == AttributeProto::AttributeType::INT,
                      std::string("ComputeShapeScan: attribute '") + name +
                          "' must be of type INT.");
  return attr->i();
}

// Normalizes an axis in the range [-rank, rank-1] to a non-negative value.
int64_t NormalizeAxis(int64_t axis, std::size_t rank, const char *attr_name) {
  const int64_t r = static_cast<int64_t>(rank);
  if (axis < 0) {
    axis += r;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis <= r, std::string("ComputeShapeScan: '") + attr_name +
                                                  "' out of range for rank " +
                                                  std::to_string(rank) + ".");
  return axis;
}

} // namespace

void ComputeShapeScan(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Scan", "ComputeShapeScan");

  const GraphProto &body = FindGraphAttribute(node, "body", "ComputeShapeScan");
  const int64_t num_scan_inputs64 = RequireIntAttribute(node, "num_scan_inputs");
  EXT_ENFORCE_INVALID(num_scan_inputs64 > 0,
                      "ComputeShapeScan: 'num_scan_inputs' must be strictly positive, got " +
                          std::to_string(num_scan_inputs64) + ".");
  const int num_scan_inputs = static_cast<int>(num_scan_inputs64);
  EXT_ENFORCE_INVALID(
      node.input_size() >= num_scan_inputs,
      "ComputeShapeScan: 'Scan' node declares " + std::to_string(node.input_size()) +
          " input(s), expected at least num_scan_inputs = " + std::to_string(num_scan_inputs) +
          ".");
  const int n_state = node.input_size() - num_scan_inputs;
  EXT_ENFORCE_INVALID(node.output_size() >= n_state,
                      "ComputeShapeScan: Scan node declares " + std::to_string(node.output_size()) +
                          " output(s), expected at least N=" + std::to_string(n_state) +
                          " state outputs.");
  const int k_scan = node.output_size() - n_state;

  EXT_ENFORCE_INVALID(
      body.input().size() == n_state + num_scan_inputs,
      "ComputeShapeScan: 'body' sub-graph declares " + std::to_string(body.input().size()) +
          " input(s), expected N + M = " + std::to_string(n_state + num_scan_inputs) + ".");
  EXT_ENFORCE_INVALID(body.output().size() == n_state + k_scan,
                      "ComputeShapeScan: 'body' sub-graph declares " +
                          std::to_string(body.output().size()) +
                          " output(s), expected N + K = " + std::to_string(n_state + k_scan) + ".");

  // Optional scan_input_axes attribute (per scan input).
  std::vector<int64_t> scan_input_axes;
  GetAttributeInts(node, "scan_input_axes", scan_input_axes);
  EXT_ENFORCE_INVALID(scan_input_axes.empty() ||
                          scan_input_axes.size() == static_cast<std::size_t>(num_scan_inputs),
                      "ComputeShapeScan: 'scan_input_axes' must have num_scan_inputs entries.");

  // Optional scan_output_axes attribute (per scan output).
  std::vector<int64_t> scan_output_axes;
  GetAttributeInts(node, "scan_output_axes", scan_output_axes);
  EXT_ENFORCE_INVALID(scan_output_axes.empty() ||
                          scan_output_axes.size() == static_cast<std::size_t>(k_scan),
                      "ComputeShapeScan: 'scan_output_axes' must have K entries.");

  // Build a child context with the body's formal input descriptors so that
  // shape inference can walk the body. The first N body inputs are the
  // state variables (inherited from the matching node inputs); the
  // remaining M are per-iteration scan-input slices, obtained by dropping
  // the scan axis from the matching scan_input shape.
  ShapesContext local = ctx;
  for (int i = 0; i < n_state; ++i) {
    const std::string state_in_name = node.input(i).as_string();
    EXT_ENFORCE_INVALID(local.Has(state_in_name), "ComputeShapeScan: state input '" +
                                                      state_in_name +
                                                      "' is missing from the inferred context.");
    local.Set(body.input()[i].name().as_string(), OptimTensor(local.Get(state_in_name)));
  }

  // The trip count is taken from the first scan input's scan axis.
  OptimDim trip_count_dim(std::string("Scan_") + node.output(0).as_string() + "_trip_count");
  bool trip_count_known = false;

  for (int m = 0; m < num_scan_inputs; ++m) {
    const std::string scan_in_name = node.input(n_state + m).as_string();
    EXT_ENFORCE_INVALID(local.Has(scan_in_name), "ComputeShapeScan: scan input '" + scan_in_name +
                                                     "' is missing from the inferred context.");
    const OptimTensor &scan_in = local.Get(scan_in_name);
    const int64_t axis_raw = scan_input_axes.empty() ? 0 : scan_input_axes[m];
    const int64_t axis = NormalizeAxis(axis_raw, scan_in.Shape().Rank(), "scan_input_axes");
    EXT_ENFORCE_INVALID(scan_in.Shape().Rank() >= 1,
                        "ComputeShapeScan: scan input '" + scan_in_name + "' must have rank >= 1.");
    if (m == 0) {
      trip_count_dim = scan_in.Shape()[static_cast<std::size_t>(axis)];
      trip_count_known = true;
    }
    // Body input shape for this scan input = scan_in.shape with axis removed.
    OptimShape body_in_shape;
    for (std::size_t d = 0; d < scan_in.Shape().Rank(); ++d) {
      if (static_cast<int64_t>(d) == axis) {
        continue;
      }
      body_in_shape.PushBack(scan_in.Shape()[d]);
    }
    local.Set(body.input()[n_state + m].name().as_string(),
              OptimTensor(nullptr, scan_in.Dtype(), std::move(body_in_shape)));
  }

  ComputeShapes(local, body.node());

  for (int i = 0; i < body.output().size(); ++i) {
    const std::string body_out = body.output()[i].name().as_string();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeScan: body output '" + body_out +
                                                 "' is missing from the inferred context.");
  }

  // N state outputs: dtype/shape are taken from the body's v_out (and
  // validated against v_initial when shapes agree).
  for (int i = 0; i < n_state; ++i) {
    const std::string node_out = node.output(i).as_string();
    if (node_out.empty()) {
      continue;
    }
    const OptimTensor &state_in = ctx.Get(node.input(i).as_string());
    const OptimTensor &v_out = local.Get(body.output()[i].name().as_string());
    EXT_ENFORCE_INVALID(v_out.Dtype() == state_in.Dtype(),
                        "ComputeShapeScan: body output #" + std::to_string(i) +
                            " has a different element type than the matching state input.");
    OptimShape out_shape = (v_out.Shape() == state_in.Shape())
                               ? state_in.Shape()
                               : SymbolicShape(v_out.Shape().Rank(), "Scan_" + node_out);
    ctx.Set(node_out, OptimTensor(nullptr, state_in.Dtype(), std::move(out_shape)));
  }

  // K scan outputs: dtype is the body's scan-output element dtype; shape
  // is the body's per-iteration scan-output shape with a new axis of
  // length ``trip_count_dim`` inserted at position scan_output_axes[k]
  // (default 0).
  for (int k = 0; k < k_scan; ++k) {
    const std::string node_out = node.output(n_state + k).as_string();
    if (node_out.empty()) {
      continue;
    }
    const OptimTensor &scan_out_elt = local.Get(body.output()[n_state + k].name().as_string());
    const int64_t axis_raw = scan_output_axes.empty() ? 0 : scan_output_axes[k];
    // Output rank = elt rank + 1.
    const int64_t axis =
        NormalizeAxis(axis_raw, scan_out_elt.Shape().Rank() + 1, "scan_output_axes");
    OptimShape stacked;
    const OptimDim trip_dim =
        trip_count_known ? trip_count_dim : OptimDim(std::string("Scan_") + node_out + "_trip");
    for (std::size_t d = 0; d <= scan_out_elt.Shape().Rank(); ++d) {
      if (static_cast<int64_t>(d) == axis) {
        stacked.PushBack(trip_dim);
      }
      if (d < scan_out_elt.Shape().Rank()) {
        stacked.PushBack(scan_out_elt.Shape()[d]);
      }
    }
    ctx.Set(node_out, OptimTensor(nullptr, scan_out_elt.Dtype(), std::move(stacked)));
  }
}

} // namespace controlflow
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
