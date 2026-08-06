// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "onnx_proto/onnx_helper.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

namespace {

// Returns the common per-element shape of ``seq`` when every recorded
// per-element shape is identical; otherwise returns an empty (unknown)
// shape. When the input sequence has no per-element shapes recorded, an
// empty shape is returned as well.
SymShape CommonElemShape(const SymSequence &seq) {
  if (!seq.HasElemShapes() || seq.ElemShapes().empty()) {
    return SymShape{};
  }
  const std::vector<SymShape> &shapes = seq.ElemShapes();
  for (std::size_t i = 1; i < shapes.size(); ++i) {
    if (shapes[i] != shapes[0]) {
      return SymShape{};
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

  EXT_ENFORCE_INVALID(static_cast<int>(body.input().size()) == node.input_size(),
                      "ComputeShapeSequenceMap: 'body' sub-graph declares ",
                      std::to_string(body.input().size()), " input(s), expected ",
                      std::to_string(node.input_size()),
                      " (one per SequenceMap input: the per-iteration element of "
                      "input_sequence followed by the additional inputs).");
  EXT_ENFORCE_INVALID(static_cast<int>(body.output().size()) == node.output_size(),
                      "ComputeShapeSequenceMap: 'body' sub-graph declares ",
                      std::to_string(body.output().size()), " output(s), expected ",
                      std::to_string(node.output_size()),
                      " (one per SequenceMap output sequence).");

  // Seed a child context with the body's formal inputs: the first body
  // input represents one element of the input sequence; the remaining
  // body inputs mirror the outer-scope additional inputs of ``node``
  // (which can be either tensors or sequences).
  ShapesContext local = ctx;

  const std::string input_seq_name = node.input(0);
  const SymSequence &input_seq = ctx.GetSequence(input_seq_name);
  const TensorType elem_dtype = input_seq.ElemDtype();
  const SymShape elem_shape = CommonElemShape(input_seq);
  local.Set(body.input()[0].name(), SymTensor(nullptr, elem_dtype, elem_shape));

  for (int i = 1; i < node.input_size(); ++i) {
    const std::string outer_name = node.input(i);
    const std::string body_in_name = body.input()[i].name();
    if (ctx.HasSequence(outer_name)) {
      // Additional inputs that are sequences contribute one element per
      // iteration; bind the body input to a per-element tensor descriptor,
      // not to the whole sequence.
      const SymSequence &add_seq = ctx.GetSequence(outer_name);
      local.Set(body_in_name, SymTensor(nullptr, add_seq.ElemDtype(), CommonElemShape(add_seq)));
    } else {
      EXT_ENFORCE_INVALID(ctx.Has(outer_name), "ComputeShapeSequenceMap: additional input '",
                          outer_name, "' is missing from the inferred context.");
      local.Set(body_in_name, SymTensor(ctx.Get(outer_name)));
    }
  }

  local.ComputeShapes(body.node());

  // Validate that every body output is known in the local context.
  for (int i = 0; i < static_cast<int>(body.output().size()); ++i) {
    const std::string body_out = body.output()[i].name();
    EXT_ENFORCE_INVALID(local.Has(body_out), "ComputeShapeSequenceMap: body output '", body_out,
                        "' is missing from the inferred context.");
  }

  // Each output sequence has its element dtype equal to the body output
  // dtype; the length matches the input sequence length (concrete when
  // known, otherwise symbolic). Per-element shapes are not recorded
  // because the body may vary the per-element shape across iterations.
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string node_out = node.output(i);
    if (node_out.empty()) {
      continue;
    }
    const SymTensor &body_out = local.Get(body.output()[i].name());
    SymDim out_length = input_seq.Length().IsInt() ? SymDim(input_seq.Length().AsInt())
                                                   : SymDim("SequenceMap_" + node_out + "_len");
    ctx.SetSequence(node_out, SymSequence(body_out.Dtype(), std::move(out_length)));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
