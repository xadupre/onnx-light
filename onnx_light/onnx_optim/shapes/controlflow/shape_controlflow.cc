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
  if (subgraph.output().size() != expected) {
    throw std::invalid_argument(std::string("ComputeShapeIf: sub-graph '") + branch_name +
                                "' declares " + std::to_string(subgraph.output().size()) +
                                " output(s), expected " + std::to_string(expected) + ".");
  }
  const std::string name = subgraph.output()[output_index].name().as_string();
  if (!local_ctx.Has(name)) {
    throw std::invalid_argument(std::string("ComputeShapeIf: output '") + name +
                                "' of sub-graph '" + branch_name +
                                "' is missing from the inferred context.");
  }
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
  if (then_shape.Rank() != else_shape.Rank()) {
    throw std::invalid_argument(
        std::string("ComputeShapeIf: rank mismatch between branches for output '") +
        if_output_name + "': then_branch has rank " + std::to_string(then_shape.Rank()) +
        ", else_branch has rank " + std::to_string(else_shape.Rank()) +
        ". This is not supposed to happen for a well-formed If node.");
  }

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

  if (node.input_size() != 1) {
    throw std::invalid_argument("ComputeShapeIf: op 'If' expects exactly one input (cond), got " +
                                std::to_string(node.input_size()) + ".");
  }

  const GraphProto &then_branch = FindGraphAttribute(node, "then_branch", "ComputeShapeIf");
  const GraphProto &else_branch = FindGraphAttribute(node, "else_branch", "ComputeShapeIf");

  const int n_outputs = node.output_size();
  if (then_branch.output().size() != n_outputs) {
    throw std::invalid_argument("ComputeShapeIf: 'then_branch' sub-graph declares " +
                                std::to_string(then_branch.output().size()) +
                                " output(s), expected " + std::to_string(n_outputs) + ".");
  }
  if (else_branch.output().size() != n_outputs) {
    throw std::invalid_argument("ComputeShapeIf: 'else_branch' sub-graph declares " +
                                std::to_string(else_branch.output().size()) +
                                " output(s), expected " + std::to_string(n_outputs) + ".");
  }

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

} // namespace controlflow
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
