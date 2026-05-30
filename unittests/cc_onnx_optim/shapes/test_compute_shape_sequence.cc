// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeSequenceConstructNode(const std::vector<std::string> &inputs,
                                    const std::string &output) {
  NodeProto node;
  node.set_op_type("SequenceConstruct");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(output);
  return node;
}

NodeProto MakeSequenceLengthNode(const std::string &input, const std::string &output) {
  NodeProto node;
  node.set_op_type("SequenceLength");
  node.add_input(input);
  node.add_output(output);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceConstruct, ThreeInputsCommonShape) {
  NodeProto node = MakeSequenceConstructNode({"a", "b", "c"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("c", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 3u);
  EXPECT_EQ(seq.ElemShapes()[0], shape);
  EXPECT_EQ(seq.ElemShapes()[1], shape);
  EXPECT_EQ(seq.ElemShapes()[2], shape);
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 3);
}

TEST(OnnxOptimShapeSequenceConstruct, ElementShapesAreRecordedVerbatim) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_a));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_b));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kInt64);
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 2u);
  EXPECT_EQ(seq.ElemShapes()[0], shape_a);
  EXPECT_EQ(seq.ElemShapes()[1], shape_b);
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 2);
}

TEST(OnnxOptimShapeSequenceConstruct, DifferentRanksAreAllowed) {
  // SequenceConstruct only constrains the element dtype: tensors of
  // different ranks may be combined. The output sequence records each
  // input shape verbatim.
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 2u);
  EXPECT_EQ(seq.ElemShapes()[0], shape_a);
  EXPECT_EQ(seq.ElemShapes()[1], shape_b);
  EXPECT_EQ(seq.Length().AsInt(), 2);
}

TEST(OnnxOptimShapeSequenceConstruct, DtypeMismatchThrows) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(3)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceConstruct, ZeroInputsProducesEmptySequence) {
  NodeProto node = MakeSequenceConstructNode({}, "y");
  onnx_optim::shapes::ShapesContext ctx;

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_FALSE(seq.HasElemDtype());
  ASSERT_TRUE(seq.HasElemShapes());
  EXPECT_TRUE(seq.ElemShapes().empty());
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 0);
}

TEST(OnnxOptimShapeSequenceConstruct, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceConstruct");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceConstruct) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  EXPECT_EQ(ctx.GetSequence("y").Length().AsInt(), 2);
  EXPECT_EQ(ctx.GetSequence("y").ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(ctx.GetSequence("y").HasElemShapes());
  ASSERT_EQ(ctx.GetSequence("y").ElemShapes().size(), 2u);
  EXPECT_EQ(ctx.GetSequence("y").ElemShapes()[0], shape);
  EXPECT_EQ(ctx.GetSequence("y").ElemShapes()[1], shape);
}

TEST(OnnxOptimShapeSequenceLength, ProducesScalarInt64Tensor) {
  NodeProto node = MakeSequenceLengthNode("s", "len");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s",
                  onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat,
                                            std::vector<onnx_optim::OptimShape>{
                                                onnx_optim::OptimShape{onnx_optim::OptimDim(2)},
                                                onnx_optim::OptimShape{onnx_optim::OptimDim(3)}}));

  onnx_optim::shapes::sequence::ComputeShapeSequenceLength(ctx, node);

  ASSERT_TRUE(ctx.Has("len"));
  const onnx_optim::OptimTensor &out = ctx.Get("len");
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(out.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeSequenceLength, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceLength");
  node.add_input("s");
  node.add_output("len");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("N")));
  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceLength(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceLength) {
  NodeProto node = MakeSequenceLengthNode("s", "len");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("N")));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("len"));
  const onnx_optim::OptimTensor &out = ctx.Get("len");
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(out.Shape().Rank(), 0u);
}

// ──────────────────────────────────────────────────────────────────────
// ONNX-derived test cases. The scenarios below mirror the
// ``SequenceConstruct`` usages from the upstream
// ``onnx/backend/test/case/model/sequence.py`` test models — three
// FLOAT tensors of shape [2, 3, 4] (test_sequence_model2..5) and three
// FLOAT tensors of mixed shapes [2, 3, 4] / [1, 3, 4] / [3, 3, 4]
// (test_sequence_model1).
// ──────────────────────────────────────────────────────────────────────

TEST(OnnxOptimShapeSequenceConstruct, OnnxSequenceModel2_ThreeFloatTensorsSameShape) {
  NodeProto node = MakeSequenceConstructNode({"X", "Y", "Z"}, "seq_1");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                               onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("Z", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("seq_1"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("seq_1");
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 3u);
  EXPECT_EQ(seq.ElemShapes()[0], shape);
  EXPECT_EQ(seq.ElemShapes()[1], shape);
  EXPECT_EQ(seq.ElemShapes()[2], shape);
  EXPECT_EQ(seq.Length().AsInt(), 3);
}

TEST(OnnxOptimShapeSequenceConstruct, OnnxSequenceModel1_ThreeFloatTensorsMixedShapes) {
  // SequenceInsert in test_sequence_model1 builds a sequence of three
  // tensors of shapes [2, 3, 4] / [1, 3, 4] / [3, 3, 4]. The equivalent
  // SequenceConstruct must accept those mixed shapes.
  NodeProto node = MakeSequenceConstructNode({"X", "Y", "Z"}, "seq_1");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape sx{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                            onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape sy{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                            onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape sz{onnx_optim::OptimDim(3), onnx_optim::OptimDim(3),
                            onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, sx));
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, sy));
  ctx.Set("Z", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, sz));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("seq_1"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("seq_1");
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 3u);
  EXPECT_EQ(seq.ElemShapes()[0], sx);
  EXPECT_EQ(seq.ElemShapes()[1], sy);
  EXPECT_EQ(seq.ElemShapes()[2], sz);
  EXPECT_EQ(seq.Length().AsInt(), 3);
}

TEST(OnnxOptimOptimSequence, DefaultsAndSetters) {
  onnx_optim::OptimSequence seq;
  EXPECT_FALSE(seq.HasElemDtype());
  EXPECT_FALSE(seq.HasElemShapes());
  EXPECT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 0);

  std::vector<onnx_optim::OptimShape> shapes{
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")},
      onnx_optim::OptimShape{onnx_optim::OptimDim(3)}};
  seq.SetElemDtype(onnx_optim::TensorType::kInt64);
  seq.SetElemShapes(shapes);
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_TRUE(seq.HasElemShapes());
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(seq.ElemShapes().size(), 2u);
  EXPECT_EQ(seq.ElemShapes()[0], shapes[0]);
  EXPECT_EQ(seq.ElemShapes()[1], shapes[1]);
  // SetElemShapes synchronises the length.
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 2);

  seq.SetLength(onnx_optim::OptimDim("L"));
  EXPECT_TRUE(seq.Length().IsExpr());
  EXPECT_EQ(seq.Length().AsExpr(), "L");

  seq.ClearElemShapes();
  EXPECT_FALSE(seq.HasElemShapes());
  seq.SetElemDtype(onnx_optim::TensorType::kUndefined);
  EXPECT_FALSE(seq.HasElemDtype());
}

TEST(OnnxOptimOptimSequence, SymbolicLengthConstructor) {
  onnx_optim::OptimSequence seq(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("L"));
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kFloat);
  EXPECT_FALSE(seq.HasElemShapes());
  EXPECT_TRUE(seq.ElemShapes().empty());
  ASSERT_TRUE(seq.Length().IsExpr());
  EXPECT_EQ(seq.Length().AsExpr(), "L");
}

TEST(OnnxOptimOptimSequence, Equality) {
  std::vector<onnx_optim::OptimShape> shapes{onnx_optim::OptimShape{onnx_optim::OptimDim(3)},
                                             onnx_optim::OptimShape{onnx_optim::OptimDim(3)}};
  onnx_optim::OptimSequence a(onnx_optim::TensorType::kFloat, shapes);
  onnx_optim::OptimSequence b(onnx_optim::TensorType::kFloat, shapes);
  EXPECT_EQ(a, b);
  b.ElemShapes()[1] = onnx_optim::OptimShape{onnx_optim::OptimDim(4)};
  EXPECT_NE(a, b);
}

TEST(OnnxOptimShapesContext, SequencesMapIsIndependentFromTensorsMap) {
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  ctx.Set("t", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.SetSequence("s", onnx_optim::OptimSequence(
                           onnx_optim::TensorType::kFloat,
                           std::vector<onnx_optim::OptimShape>{shape, shape, shape, shape}));
  EXPECT_TRUE(ctx.Has("t"));
  EXPECT_FALSE(ctx.HasSequence("t"));
  EXPECT_TRUE(ctx.HasSequence("s"));
  EXPECT_FALSE(ctx.Has("s"));
  EXPECT_EQ(ctx.SequencesSize(), 1u);
  EXPECT_EQ(ctx.Size(), 1u);
  EXPECT_EQ(ctx.GetSequence("s").Length().AsInt(), 4);

  ctx.Clear();
  EXPECT_EQ(ctx.SequencesSize(), 0u);
  EXPECT_EQ(ctx.Size(), 0u);
}

namespace {

NodeProto MakeConcatFromSequenceNode(const std::string &input, const std::string &output,
                                     int64_t axis, int64_t new_axis = 0,
                                     bool include_new_axis = true) {
  NodeProto node;
  node.set_op_type("ConcatFromSequence");
  node.add_input(input);
  node.add_output(output);
  AttributeProto *axis_attr = node.add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::AttributeType::INT);
  axis_attr->set_i(axis);
  if (include_new_axis) {
    AttributeProto *new_axis_attr = node.add_attribute();
    new_axis_attr->set_name("new_axis");
    new_axis_attr->set_type(AttributeProto::AttributeType::INT);
    new_axis_attr->set_i(new_axis);
  }
  return node;
}

onnx_optim::OptimSequence
MakeFloatSequence(const std::vector<onnx_optim::OptimShape> &elem_shapes) {
  return onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, elem_shapes);
}

} // namespace

TEST(OnnxOptimShapeConcatFromSequence, ConcatAxis0SumsConcreteDims) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
                           onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)},
                           onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)},
                       }));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  const onnx_optim::OptimTensor &out = ctx.Get("y");
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(out.Shape().Rank(), 2u);
  ASSERT_TRUE(out.Shape()[0].IsInt());
  EXPECT_EQ(out.Shape()[0].AsInt(), 7);
  ASSERT_TRUE(out.Shape()[1].IsInt());
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, ConcatNegativeAxisIsResolvedAgainstRank) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/-1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)},
                       }));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 2);
  EXPECT_EQ(out.Shape()[1].AsInt(), 8);
}

TEST(OnnxOptimShapeConcatFromSequence, ConcatSymbolicAxisYieldsSymbolicOutputDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s",
                  MakeFloatSequence({
                      onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3)},
                      onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)},
                  }));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_TRUE(out.Shape()[0].IsExpr());
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, NewAxisInsertsSequenceLengthDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0, /*new_axis=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  const onnx_optim::OptimShape elem{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.SetSequence("s", MakeFloatSequence({elem, elem, elem}));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 3u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 3);
  EXPECT_EQ(out.Shape()[1].AsInt(), 2);
  EXPECT_EQ(out.Shape()[2].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, NewAxisAtTailAppendsLengthDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/2, /*new_axis=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  const onnx_optim::OptimShape elem{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.SetSequence("s", MakeFloatSequence({elem, elem, elem, elem}));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 3u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 2);
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
  EXPECT_EQ(out.Shape()[2].AsInt(), 4);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsMissingAxisAttribute) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0, /*new_axis=*/0,
                                              /*include_new_axis=*/false);
  // Drop the required axis attribute too.
  node.ref_attribute().clear();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({onnx_optim::OptimShape{onnx_optim::OptimDim(2)}}));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsAxisOutOfRange) {
  // rank = 2; new_axis = 0 → axis must lie in [-2, 1]. axis = 2 is out.
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/2);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                 onnx_optim::OptimDim(3)}}));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsInvalidNewAxis) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0, /*new_axis=*/2);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({onnx_optim::OptimShape{onnx_optim::OptimDim(2)}}));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsRankMismatchAcrossElements) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2)},
                       }));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, UnknownElementShapesForwardsDtypeOnly) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  // Sequence with known dtype but unknown per-element shapes (symbolic length).
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("L")));

  onnx_optim::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  const onnx_optim::OptimTensor &out = ctx.Get("y");
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(out.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeConcatFromSequence, DispatchedViaComputeShapeNode) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
                           onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)},
                       }));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 2);
  EXPECT_EQ(out.Shape()[1].AsInt(), 8);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceErase shape-inference tests.
// ──────────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeSequenceEraseNode(const std::string &input_seq, const std::string &output,
                                bool with_position = false) {
  NodeProto node;
  node.set_op_type("SequenceErase");
  node.add_input(input_seq);
  if (with_position) {
    node.add_input("position");
  }
  node.add_output(output);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceErase, KnownLengthProducesLengthMinusOne) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat,
                                     std::vector<onnx_optim::OptimShape>{shape, shape, shape}));

  onnx_optim::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const onnx_optim::OptimSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 2);
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceErase, SingleElementProducesLengthZero) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s", onnx_optim::OptimSequence(
                           onnx_optim::TensorType::kInt64,
                           std::vector<onnx_optim::OptimShape>{onnx_optim::OptimShape{}}));

  onnx_optim::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const onnx_optim::OptimSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), onnx_optim::TensorType::kInt64);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 0);
}

TEST(OnnxOptimShapeSequenceErase, SymbolicLengthProducesSymbolicLength) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("N")));

  onnx_optim::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const onnx_optim::OptimSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), onnx_optim::TensorType::kFloat);
  EXPECT_FALSE(out.Length().IsInt());
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceErase, ElemDtypeIsForwardedFromInput) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kDouble, onnx_optim::OptimDim("N")));

  onnx_optim::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  EXPECT_EQ(ctx.GetSequence("out").ElemDtype(), onnx_optim::TensorType::kDouble);
}

TEST(OnnxOptimShapeSequenceErase, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceErase");
  node.add_input("s");
  node.add_output("out");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, onnx_optim::OptimDim("N")));
  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceErase(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceErase) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetSequence("s",
                  onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat,
                                            std::vector<onnx_optim::OptimShape>{
                                                onnx_optim::OptimShape{onnx_optim::OptimDim(3)},
                                                onnx_optim::OptimShape{onnx_optim::OptimDim(3)},
                                            }));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const onnx_optim::OptimSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), onnx_optim::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 1);
}

} // namespace Test
