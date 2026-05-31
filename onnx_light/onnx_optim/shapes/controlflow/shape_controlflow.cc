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

} // namespace controlflow
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
