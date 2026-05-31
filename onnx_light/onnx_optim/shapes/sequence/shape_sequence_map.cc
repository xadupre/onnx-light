// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "onnx_proto/onnx_helper.h"

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_optim/shapes/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

namespace {

// Returns the common per-element shape of ``seq`` when every recorded
// per-element shape is identical; otherwise returns an empty (unknown)
// shape. When the input sequence has no per-element shapes recorded, an
// empty shape is returned as well.
OptimShape CommonElemShape(const OptimSequence &seq) {
  if (!seq.HasElemShapes() || seq.ElemShapes().empty()) {
    return OptimShape{};
  }
  const std::vector<OptimShape> &shapes = seq.ElemShapes();
  for (std::size_t i = 1; i < shapes.size(); ++i) {
    if (shapes[i] != shapes[0]) {
      return OptimShape{};
    }
  }
  return shapes[0];
}

} // namespace

void ComputeShapeSequenceMap(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceMap", "ComputeShapeSequenceMap");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceMap: SequenceMap requires at least one input "
                      "(input_sequence).");

  const GraphProto &body = FindGraphAttribute(node, "body", "ComputeShapeSequenceMap");

  EXT_ENFORCE_INVALID(body.input().size() == node.input_size(),
                      "ComputeShapeSequenceMap: 'body' sub-graph declares " +
                          std::to_string(body.input().size()) + " input(s), expected " +
                          std::to_string(node.input_size()) +
                          " (one per SequenceMap input: the per-iteration element of "
                          "input_sequence followed by the additional inputs).");
  EXT_ENFORCE_INVALID(body.output().size() == node.output_size(),
                      "ComputeShapeSequenceMap: 'body' sub-graph declares " +
                          std::to_string(body.output().size()) + " output(s), expected " +
                          std::to_string(node.output_size()) +
                          " (one per SequenceMap output sequence).");

  // Seed a child context with the body's formal inputs: the first body
  // input represents one element of the input sequence; the remaining
  // body inputs mirror the outer-scope additional inputs of ``node``
  // (which can be either tensors or sequences).
  ShapesContext local = ctx;

  const std::string input_seq_name = node.input(0).as_string();
  const OptimSequence &input_seq = ctx.GetSequence(input_seq_name);
  const TensorType elem_dtype = input_seq.ElemDtype();
  const OptimShape elem_shape = CommonElemShape(input_seq);
  local.Set(body.input()[0].name().as_string(), OptimTensor(nullptr, elem_dtype, elem_shape));

  for (int i = 1; i < node.input_size(); ++i) {
    const std::string outer_name = node.input(i).as_string();
    const std::string body_in_name = body.input()[i].name().as_string();
    if (ctx.HasSequence(outer_name)) {
      local.SetSequence(body_in_name, OptimSequence(ctx.GetSequence(outer_name)));
    } else {
      EXT_ENFORCE_INVALID(ctx.Has(outer_name), "ComputeShapeSequenceMap: additional input '" +
                                                   outer_name +
                                                   "' is missing from the inferred context.");
      local.Set(body_in_name, OptimTensor(ctx.Get(outer_name)));
    }
  }

  ComputeShapes(local, body.node());

  // Validate that every body output is known in the local context.
  for (int i = 0; i < body.output().size(); ++i) {
    const std::string body_out = body.output()[i].name().as_string();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeSequenceMap: body output '" + body_out +
                                                 "' is missing from the inferred context.");
  }

  // Each output sequence has its element dtype equal to the body output
  // dtype; the length matches the input sequence length (concrete when
  // known, otherwise symbolic). Per-element shapes are not recorded
  // because the body may vary the per-element shape across iterations.
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string node_out = node.output(i).as_string();
    if (node_out.empty()) {
      continue;
    }
    const OptimTensor &body_out = local.Get(body.output()[i].name().as_string());
    OptimDim out_length = input_seq.Length().IsInt() ? OptimDim(input_seq.Length().AsInt())
                                                     : OptimDim("SequenceMap_" + node_out + "_len");
    ctx.SetSequence(node_out, OptimSequence(body_out.Dtype(), std::move(out_length)));
  }
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
