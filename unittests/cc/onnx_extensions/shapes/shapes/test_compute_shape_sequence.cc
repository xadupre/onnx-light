// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("c", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("y");
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kFloat);
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(2), core::symbolic::SymDim(5)};
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, shape_a));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, shape_b));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("y");
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kInt64);
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(4)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(2), core::symbolic::SymDim(2)};
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("y");
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 2u);
  EXPECT_EQ(seq.ElemShapes()[0], shape_a);
  EXPECT_EQ(seq.ElemShapes()[1], shape_b);
  EXPECT_EQ(seq.Length().AsInt(), 2);
}

TEST(OnnxOptimShapeSequenceConstruct, DtypeMismatchThrows) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3)};
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, shape));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceConstruct, ZeroInputsProducesEmptySequence) {
  NodeProto node = MakeSequenceConstructNode({}, "y");
  core::shapes::ShapesContext ctx;

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("y");
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
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceConstruct) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  EXPECT_EQ(ctx.GetSequence("y").Length().AsInt(), 2);
  EXPECT_EQ(ctx.GetSequence("y").ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(ctx.GetSequence("y").HasElemShapes());
  ASSERT_EQ(ctx.GetSequence("y").ElemShapes().size(), 2u);
  EXPECT_EQ(ctx.GetSequence("y").ElemShapes()[0], shape);
  EXPECT_EQ(ctx.GetSequence("y").ElemShapes()[1], shape);
}

TEST(OnnxOptimShapeSequenceLength, ProducesScalarInt64Tensor) {
  NodeProto node = MakeSequenceLengthNode("s", "len");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{
                                           core::symbolic::SymShape{core::symbolic::SymDim(2)},
                                           core::symbolic::SymShape{core::symbolic::SymDim(3)}}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceLength(ctx, node);

  ASSERT_TRUE(ctx.Has("len"));
  const core::symbolic::SymTensor &out = ctx.Get("len");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(out.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeSequenceLength, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceLength");
  node.add_input("s");
  node.add_output("len");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceLength(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceLength) {
  NodeProto node = MakeSequenceLengthNode("s", "len");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("len"));
  const core::symbolic::SymTensor &out = ctx.Get("len");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                 core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("Z", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("seq_1"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("seq_1");
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kFloat);
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape sx{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                              core::symbolic::SymDim(4)};
  core::symbolic::SymShape sy{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                              core::symbolic::SymDim(4)};
  core::symbolic::SymShape sz{core::symbolic::SymDim(3), core::symbolic::SymDim(3),
                              core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, sx));
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, sy));
  ctx.Set("Z", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, sz));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("seq_1"));
  const core::symbolic::SymSequence &seq = ctx.GetSequence("seq_1");
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(seq.HasElemShapes());
  ASSERT_EQ(seq.ElemShapes().size(), 3u);
  EXPECT_EQ(seq.ElemShapes()[0], sx);
  EXPECT_EQ(seq.ElemShapes()[1], sy);
  EXPECT_EQ(seq.ElemShapes()[2], sz);
  EXPECT_EQ(seq.Length().AsInt(), 3);
}

TEST(OnnxOptimOptimSequence, DefaultsAndSetters) {
  core::symbolic::SymSequence seq;
  EXPECT_FALSE(seq.HasElemDtype());
  EXPECT_FALSE(seq.HasElemShapes());
  EXPECT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 0);

  std::vector<core::symbolic::SymShape> shapes{
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim("N")},
      core::symbolic::SymShape{core::symbolic::SymDim(3)}};
  seq.SetElemDtype(core::symbolic::TensorType::kInt64);
  seq.SetElemShapes(shapes);
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_TRUE(seq.HasElemShapes());
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(seq.ElemShapes().size(), 2u);
  EXPECT_EQ(seq.ElemShapes()[0], shapes[0]);
  EXPECT_EQ(seq.ElemShapes()[1], shapes[1]);
  // SetElemShapes synchronises the length.
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 2);

  seq.SetLength(core::symbolic::SymDim("L"));
  EXPECT_TRUE(seq.Length().IsExpr());
  EXPECT_EQ(seq.Length().AsExpr(), "L");

  seq.ClearElemShapes();
  EXPECT_FALSE(seq.HasElemShapes());
  seq.SetElemDtype(core::symbolic::TensorType::kUndefined);
  EXPECT_FALSE(seq.HasElemDtype());
}

TEST(OnnxOptimOptimSequence, SymbolicLengthConstructor) {
  core::symbolic::SymSequence seq(core::symbolic::TensorType::kFloat, core::symbolic::SymDim("L"));
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_EQ(seq.ElemDtype(), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(seq.HasElemShapes());
  EXPECT_TRUE(seq.ElemShapes().empty());
  ASSERT_TRUE(seq.Length().IsExpr());
  EXPECT_EQ(seq.Length().AsExpr(), "L");
}

TEST(OnnxOptimOptimSequence, Equality) {
  std::vector<core::symbolic::SymShape> shapes{core::symbolic::SymShape{core::symbolic::SymDim(3)},
                                               core::symbolic::SymShape{core::symbolic::SymDim(3)}};
  core::symbolic::SymSequence a(core::symbolic::TensorType::kFloat, shapes);
  core::symbolic::SymSequence b(core::symbolic::TensorType::kFloat, shapes);
  EXPECT_EQ(a, b);
  b.ElemShapes()[1] = core::symbolic::SymShape{core::symbolic::SymDim(4)};
  EXPECT_NE(a, b);
}

TEST(OnnxOptimShapesContext, SequencesMapIsIndependentFromTensorsMap) {
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  ctx.Set("t", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.SetSequence("s", core::symbolic::SymSequence(
                           core::symbolic::TensorType::kFloat,
                           std::vector<core::symbolic::SymShape>{shape, shape, shape, shape}));
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

core::symbolic::SymSequence
MakeFloatSequence(const std::vector<core::symbolic::SymShape> &elem_shapes) {
  return core::symbolic::SymSequence(core::symbolic::TensorType::kFloat, elem_shapes);
}

} // namespace

TEST(OnnxOptimShapeConcatFromSequence, ConcatAxis0SumsConcreteDims) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", MakeFloatSequence({
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3)},
           }));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  const core::symbolic::SymTensor &out = ctx.Get("y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(out.Shape().Rank(), 2u);
  ASSERT_TRUE(out.Shape()[0].IsInt());
  EXPECT_EQ(out.Shape()[0].AsInt(), 7);
  ASSERT_TRUE(out.Shape()[1].IsInt());
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, ConcatNegativeAxisIsResolvedAgainstRank) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/-1);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", MakeFloatSequence({
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)},
           }));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const core::symbolic::SymTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 2);
  EXPECT_EQ(out.Shape()[1].AsInt(), 8);
}

TEST(OnnxOptimShapeConcatFromSequence, ConcatSymbolicAxisYieldsSymbolicOutputDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", MakeFloatSequence({
               core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)},
           }));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const core::symbolic::SymTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_TRUE(out.Shape()[0].IsExpr());
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, NewAxisInsertsSequenceLengthDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0, /*new_axis=*/1);
  core::shapes::ShapesContext ctx;
  const core::symbolic::SymShape elem{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.SetSequence("s", MakeFloatSequence({elem, elem, elem}));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const core::symbolic::SymTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 3u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 3);
  EXPECT_EQ(out.Shape()[1].AsInt(), 2);
  EXPECT_EQ(out.Shape()[2].AsInt(), 3);
}

TEST(OnnxOptimShapeConcatFromSequence, NewAxisAtTailAppendsLengthDim) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/2, /*new_axis=*/1);
  core::shapes::ShapesContext ctx;
  const core::symbolic::SymShape elem{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.SetSequence("s", MakeFloatSequence({elem, elem, elem, elem}));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  const core::symbolic::SymTensor &out = ctx.Get("y");
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
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({core::symbolic::SymShape{core::symbolic::SymDim(2)}}));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsAxisOutOfRange) {
  // rank = 2; new_axis = 0 → axis must lie in [-2, 1]. axis = 2 is out.
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/2);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                   core::symbolic::SymDim(3)}}));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsInvalidNewAxis) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0, /*new_axis=*/2);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", MakeFloatSequence({core::symbolic::SymShape{core::symbolic::SymDim(2)}}));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, RejectsRankMismatchAcrossElements) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", MakeFloatSequence({
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(2)},
           }));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeConcatFromSequence, UnknownElementShapesForwardsDtypeOnly) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/0);
  core::shapes::ShapesContext ctx;
  // Sequence with known dtype but unknown per-element shapes (symbolic length).
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("L")));

  onnx_shapes::shapes::sequence::ComputeShapeConcatFromSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  const core::symbolic::SymTensor &out = ctx.Get("y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(out.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeConcatFromSequence, DispatchedViaComputeShapeNode) {
  NodeProto node = MakeConcatFromSequenceNode("s", "y", /*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", MakeFloatSequence({
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
               core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)},
           }));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  const core::symbolic::SymTensor &out = ctx.Get("y");
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

NodeProto MakeSequenceInsertNode(const std::string &input_seq, const std::string &input_tensor,
                                 const std::string &output, bool with_position = false) {
  NodeProto node;
  node.set_op_type("SequenceInsert");
  node.add_input(input_seq);
  node.add_input(input_tensor);
  if (with_position) {
    node.add_input("position");
  }
  node.add_output(output);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceErase, KnownLengthProducesLengthMinusOne) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{shape, shape, shape}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 2);
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceErase, SingleElementProducesLengthZero) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(
                           core::symbolic::TensorType::kInt64,
                           std::vector<core::symbolic::SymShape>{core::symbolic::SymShape{}}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kInt64);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 0);
}

TEST(OnnxOptimShapeSequenceErase, SymbolicLengthProducesSymbolicLength) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(out.Length().IsInt());
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceErase, ElemDtypeIsForwardedFromInput) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kDouble,
                                                   core::symbolic::SymDim("N")));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceErase(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  EXPECT_EQ(ctx.GetSequence("out").ElemDtype(), core::symbolic::TensorType::kDouble);
}

TEST(OnnxOptimShapeSequenceInsert, KnownLengthProducesLengthPlusOne) {
  NodeProto node = MakeSequenceInsertNode("s", "x", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(
                           core::symbolic::TensorType::kFloat,
                           std::vector<core::symbolic::SymShape>{core::symbolic::SymShape{}}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceInsert(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 2);
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceInsert, SymbolicLengthProducesSymbolicLength) {
  NodeProto node = MakeSequenceInsertNode("s", "x", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceInsert(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(out.Length().IsInt());
}

TEST(OnnxOptimShapeSequenceInsert, UnknownSequenceDtypeFallsBackToInsertedTensorDtype) {
  NodeProto node = MakeSequenceInsertNode("s", "x", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kUndefined,
                                                   core::symbolic::SymDim(int64_t{0})));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceInsert(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  EXPECT_EQ(ctx.GetSequence("out").ElemDtype(), core::symbolic::TensorType::kInt64);
  ASSERT_TRUE(ctx.GetSequence("out").Length().IsInt());
  EXPECT_EQ(ctx.GetSequence("out").Length().AsInt(), 1);
}

TEST(OnnxOptimShapeSequenceInsert, RejectsDtypeMismatch) {
  NodeProto node = MakeSequenceInsertNode("s", "x", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim(int64_t{1})));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{}));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceInsert(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceErase, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceErase");
  node.add_input("s");
  node.add_output("out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceErase(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceInsert, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceInsert");
  node.add_input("s");
  node.add_input("x");
  node.add_output("out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim(int64_t{1})));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{}));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceInsert(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceErase) {
  NodeProto node = MakeSequenceEraseNode("s", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{
                                           core::symbolic::SymShape{core::symbolic::SymDim(3)},
                                           core::symbolic::SymShape{core::symbolic::SymDim(3)},
                                       }));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 1);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceInsert) {
  NodeProto node = MakeSequenceInsertNode("s", "x", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim(int64_t{2})));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 3);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceAt shape-inference tests.
// ──────────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeSequenceAtNode(const std::string &input_seq, const std::string &input_pos,
                             const std::string &output) {
  NodeProto node;
  node.set_op_type("SequenceAt");
  node.add_input(input_seq);
  node.add_input(input_pos);
  node.add_output(output);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceAt, CommonElemShapeProducesThatShape) {
  NodeProto node = MakeSequenceAtNode("s", "p", "out");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{shape, shape, shape}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceAt(ctx, node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(out.Shape(), shape);
}

TEST(OnnxOptimShapeSequenceAt, MismatchedElemShapesProducesEmptyShape) {
  NodeProto node = MakeSequenceAtNode("s", "p", "out");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(2), core::symbolic::SymDim(4)};
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{shape_a, shape_b}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceAt(ctx, node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_TRUE(out.Shape().Empty());
}

TEST(OnnxOptimShapeSequenceAt, SymbolicLengthForwardsDtypeOnly) {
  NodeProto node = MakeSequenceAtNode("s", "p", "out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kDouble,
                                                   core::symbolic::SymDim("N")));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceAt(ctx, node);

  ASSERT_TRUE(ctx.Has("out"));
  EXPECT_EQ(ctx.Get("out").Dtype(), core::symbolic::TensorType::kDouble);
}

TEST(OnnxOptimShapeSequenceAt, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceAt");
  node.add_input("s");
  node.add_input("p");
  node.add_output("out");
  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceAt(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceAt) {
  NodeProto node = MakeSequenceAtNode("s", "p", "out");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3)};
  ctx.SetSequence("s",
                  core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                              std::vector<core::symbolic::SymShape>{shape, shape}));
  ctx.Set("p", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  EXPECT_EQ(ctx.Get("out").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("out").Shape(), shape);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceMap shape-inference tests.
// ──────────────────────────────────────────────────────────────────────

namespace {

// Builds a SequenceMap body subgraph:
//   inputs : (elem [<dtype>, elem_shape])
//   nodes  : out = Identity(elem)
//   outputs: (out [<dtype>, elem_shape])
GraphProto MakeIdentitySequenceMapBody(int32_t elem_type, const std::vector<int64_t> &elem_shape) {
  GraphProto g;
  g.set_name("seq_map_body");

  ValueInfoProto *in_vi = g.add_input();
  in_vi->set_name("elem");
  TypeProto::Tensor *in_tt = in_vi->ref_type().mutable_tensor_type();
  in_tt->set_elem_type(elem_type);
  TensorShapeProto &in_shape = in_tt->ref_shape();
  for (int64_t d : elem_shape) {
    in_shape.add_dim()->set_dim_value(d);
  }

  NodeProto *n = g.add_node();
  n->set_op_type("Abs");
  n->add_input("elem");
  n->add_output("out");

  ValueInfoProto *out_vi = g.add_output();
  out_vi->set_name("out");
  TypeProto::Tensor *out_tt = out_vi->ref_type().mutable_tensor_type();
  out_tt->set_elem_type(elem_type);
  TensorShapeProto &out_shape = out_tt->ref_shape();
  for (int64_t d : elem_shape) {
    out_shape.add_dim()->set_dim_value(d);
  }

  return g;
}

NodeProto MakeSequenceMapNode(const std::string &input_seq, const std::string &output,
                              GraphProto body) {
  NodeProto node;
  node.set_op_type("SequenceMap");
  node.add_input(input_seq);
  node.add_output(output);

  AttributeProto *body_attr = node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = std::move(body);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceMap, KnownLengthForwardsLengthAndElemDtype) {
  GraphProto body =
      MakeIdentitySequenceMapBody(static_cast<int32_t>(core::symbolic::TensorType::kFloat), {2, 3});
  NodeProto node = MakeSequenceMapNode("s", "out", std::move(body));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.SetSequence(
      "s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                       std::vector<core::symbolic::SymShape>{shape, shape, shape}));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceMap(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 3);
}

TEST(OnnxOptimShapeSequenceMap, SymbolicInputLengthProducesSymbolicOutputLength) {
  GraphProto body =
      MakeIdentitySequenceMapBody(static_cast<int32_t>(core::symbolic::TensorType::kDouble), {});
  NodeProto node = MakeSequenceMapNode("s", "out", std::move(body));

  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kDouble,
                                                   core::symbolic::SymDim("N")));

  onnx_shapes::shapes::sequence::ComputeShapeSequenceMap(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kDouble);
  EXPECT_FALSE(out.Length().IsInt());
  EXPECT_FALSE(out.HasElemShapes());
}

TEST(OnnxOptimShapeSequenceMap, RejectsBodyArityMismatch) {
  // Body declares two inputs but the node only has one.
  GraphProto body =
      MakeIdentitySequenceMapBody(static_cast<int32_t>(core::symbolic::TensorType::kFloat), {});
  ValueInfoProto *extra = body.add_input();
  extra->set_name("extra");
  extra->ref_type().mutable_tensor_type()->set_elem_type(
      static_cast<int>(core::symbolic::TensorType::kFloat));

  NodeProto node = MakeSequenceMapNode("s", "out", std::move(body));

  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceMap(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceMap, RejectsMissingBodyAttribute) {
  NodeProto node;
  node.set_op_type("SequenceMap");
  node.add_input("s");
  node.add_output("out");

  core::shapes::ShapesContext ctx;
  ctx.SetSequence("s", core::symbolic::SymSequence(core::symbolic::TensorType::kFloat,
                                                   core::symbolic::SymDim("N")));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceMap(ctx, node),
               std::invalid_argument);
}

// ──────────────────────────────────────────────────────────────────────
// SequenceEmpty shape-inference tests.
// ──────────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeSequenceEmptyNode(const std::string &output, bool with_dtype = false,
                                int64_t dtype = 0) {
  NodeProto node;
  node.set_op_type("SequenceEmpty");
  node.add_output(output);
  if (with_dtype) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("dtype");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(dtype);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceEmpty, NoDtypeAttributeDefaultsToFloat) {
  NodeProto node = MakeSequenceEmptyNode("out");
  core::shapes::ShapesContext ctx;

  onnx_shapes::shapes::sequence::ComputeShapeSequenceEmpty(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 0);
  ASSERT_TRUE(out.HasElemShapes());
  EXPECT_TRUE(out.ElemShapes().empty());
}

TEST(OnnxOptimShapeSequenceEmpty, DtypeAttributeIsHonoured) {
  NodeProto node = MakeSequenceEmptyNode("out", /*with_dtype=*/true,
                                         /*dtype=*/static_cast<int64_t>(TensorProto::INT64));
  core::shapes::ShapesContext ctx;

  onnx_shapes::shapes::sequence::ComputeShapeSequenceEmpty(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kInt64);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 0);
}

TEST(OnnxOptimShapeSequenceEmpty, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceEmpty");
  node.add_output("out");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSequenceEmpty(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceEmpty) {
  NodeProto node = MakeSequenceEmptyNode("out", /*with_dtype=*/true,
                                         /*dtype=*/static_cast<int64_t>(TensorProto::DOUBLE));
  core::shapes::ShapesContext ctx;

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kDouble);
  ASSERT_TRUE(out.Length().IsInt());
  EXPECT_EQ(out.Length().AsInt(), 0);
}

// ──────────────────────────────────────────────────────────────────────
// SplitToSequence shape-inference tests.
// ──────────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeSplitToSequenceNode(const std::string &input, const std::string &output,
                                  bool with_split, int64_t axis = 0, int64_t keepdims = 1,
                                  bool with_keepdims = false) {
  NodeProto node;
  node.set_op_type("SplitToSequence");
  node.add_input(input);
  if (with_split) {
    node.add_input("split");
  }
  node.add_output(output);
  AttributeProto *axis_attr = node.add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::INT);
  axis_attr->set_i(axis);
  if (with_keepdims) {
    AttributeProto *kd_attr = node.add_attribute();
    kd_attr->set_name("keepdims");
    kd_attr->set_type(AttributeProto::INT);
    kd_attr->set_i(keepdims);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapeSplitToSequence, OmittedSplitProducesUnitChunks) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/false, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.HasElemShapes());
  ASSERT_EQ(out.ElemShapes().size(), 6u);
  for (const auto &shape : out.ElemShapes()) {
    ASSERT_EQ(shape.Rank(), 2u);
    EXPECT_EQ(shape[0].AsInt(), 3);
    EXPECT_EQ(shape[1].AsInt(), 1);
  }
}

TEST(OnnxOptimShapeSplitToSequence, OmittedSplitKeepdimsZeroSqueezesAxis) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/false, /*axis=*/1,
                                           /*keepdims=*/0, /*with_keepdims=*/true);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  ASSERT_TRUE(out.HasElemShapes());
  ASSERT_EQ(out.ElemShapes().size(), 6u);
  for (const auto &shape : out.ElemShapes()) {
    ASSERT_EQ(shape.Rank(), 1u);
    EXPECT_EQ(shape[0].AsInt(), 3);
  }
}

TEST(OnnxOptimShapeSplitToSequence, KnownVectorSplitProducesPerChunkShapes) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/true, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));
  // Encode the value of ``split`` ([1, 2]) via the tensor's value-as-shape.
  core::symbolic::SymTensor split_t(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2)});
  split_t.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2)});
  ctx.Set("split", std::move(split_t));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  ASSERT_TRUE(out.HasElemShapes());
  ASSERT_EQ(out.ElemShapes().size(), 2u);
  EXPECT_EQ(out.ElemShapes()[0][0].AsInt(), 1);
  EXPECT_EQ(out.ElemShapes()[0][1].AsInt(), 6);
  EXPECT_EQ(out.ElemShapes()[1][0].AsInt(), 2);
  EXPECT_EQ(out.ElemShapes()[1][1].AsInt(), 6);
}

TEST(OnnxOptimShapeSplitToSequence, KnownScalarSplitProducesEqualChunks) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/true, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));
  // Scalar split with value 2.
  core::symbolic::SymTensor split_t(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{});
  split_t.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("split", std::move(split_t));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  ASSERT_TRUE(out.HasElemShapes());
  ASSERT_EQ(out.ElemShapes().size(), 3u);
  for (const auto &shape : out.ElemShapes()) {
    EXPECT_EQ(shape[0].AsInt(), 3);
    EXPECT_EQ(shape[1].AsInt(), 2);
  }
}

TEST(OnnxOptimShapeSplitToSequence, RejectsZeroScalarSplit) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/true, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));
  core::symbolic::SymTensor split_t(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{});
  split_t.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(int64_t{0})});
  ctx.Set("split", std::move(split_t));

  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSplitToSequence, UnknownSplitValueLeavesShapesUnresolved) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/true, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));
  // ``split`` has known shape but unknown value.
  ctx.Set("split", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                             core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(out.HasElemShapes());
  EXPECT_FALSE(out.Length().IsInt());
}

TEST(OnnxOptimShapeSplitToSequence, NegativeAxisIsResolved) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/false, /*axis=*/-1);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt32,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  ASSERT_TRUE(out.HasElemShapes());
  ASSERT_EQ(out.ElemShapes().size(), 2u);
  for (const auto &shape : out.ElemShapes()) {
    EXPECT_EQ(shape[0].AsInt(), 4);
    EXPECT_EQ(shape[1].AsInt(), 1);
  }
}

TEST(OnnxOptimShapeSplitToSequence, RejectsOutOfRangeAxis) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/false, /*axis=*/5);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSplitToSequence, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSplitToSequence");
  node.add_input("x");
  node.add_output("out");
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::sequence::ComputeShapeSplitToSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSplitToSequence) {
  NodeProto node = MakeSplitToSequenceNode("x", "out", /*with_split=*/false, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("x", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.HasSequence("out"));
  const core::symbolic::SymSequence &out = ctx.GetSequence("out");
  EXPECT_EQ(out.ElemDtype(), core::symbolic::TensorType::kFloat);
  ASSERT_TRUE(out.HasElemShapes());
  EXPECT_EQ(out.ElemShapes().size(), 3u);
}

} // namespace Test
