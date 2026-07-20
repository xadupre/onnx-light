// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_shapes/shapes/tensor/shape_tensor.h"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeConcatNode(const std::vector<std::string> &inputs,
                         const std::optional<int64_t> &axis = std::nullopt,
                         const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Concat");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(out);
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorConcat, ConcatenatesTwoFullyKnownInputsOnAxis0) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(5), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(7), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorConcat, ConcatenatesThreeInputsOnLastAxis) {
  NodeProto node = MakeConcatNode({"A", "B", "C"}, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(1)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(2)}));
  ctx.Set("C", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(8)}));
}

TEST(OnnxOptimShapesTensorConcat, NegativeAxisIsResolvedFromRank) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/-1);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapesTensorConcat, SingleInputProducesCopyOfShape) {
  NodeProto node = MakeConcatNode({"A"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesTensorConcat, SymbolicAxisDimProducesSymbolicOutput) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(3));
}

TEST(OnnxOptimShapesTensorConcat, RepeatedSymbolicAxisDimProducesScaledExpression) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("A",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim("batch"),
                                                             core::symbolic::SymDim("seq"),
                                                             core::symbolic::SymDim("d_model")}));
  ctx.Set("B",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim("batch"),
                                                             core::symbolic::SymDim("seq"),
                                                             core::symbolic::SymDim("d_model")}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("batch"),
                                                            core::symbolic::SymDim("seq"),
                                                            core::symbolic::SymDim("2*d_model")}));
}

TEST(OnnxOptimShapesTensorConcat, MergesSymbolicNonConcatDimWithConcreteOne) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  // First input has a symbolic non-concat dim that should be replaced
  // by the concrete value coming from the second input.
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(2)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnMismatchedRank) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnMismatchedNonConcatDim) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnDtypeMismatch) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnAxisOutOfRange) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/5);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("A");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeConcat(ctx, node), std::out_of_range);
}

// ── Reshape ──────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeReshapeNode(const std::string &data = "X", const std::string &shape = "S",
                          const std::optional<int64_t> &allowzero = std::nullopt,
                          const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Reshape");
  node.add_input(data);
  node.add_input(shape);
  node.add_output(out);
  if (allowzero.has_value()) {
    AddAttribute<int64_t>(node, "allowzero", *allowzero);
  }
  return node;
}

core::symbolic::SymTensor MakeShapeInput(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape static_shape{core::symbolic::SymDim(static_cast<int64_t>(dims.size()))};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64, std::move(static_shape));
  core::symbolic::SymShape value;
  for (int64_t v : dims) {
    value.PushBack(core::symbolic::SymDim(v));
  }
  t.SetValueAsShape(std::move(value));
  return t;
}

} // namespace

TEST(OnnxOptimShapesTensorReshape, AllPositiveTargetDimsAreCopied) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("S", MakeShapeInput({6, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(6), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, ZeroCopiesFromInputDataShape) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("S", MakeShapeInput({0, 0, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                      core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, AllowZeroHonoursLiteralZero) {
  NodeProto node = MakeReshapeNode("X", "S", /*allowzero=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(0))}));
  ctx.Set("S", MakeShapeInput({0, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(0)),
                                      core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneIsInferredFromInputProduct) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("S", MakeShapeInput({-1, 6}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(6)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneWithZeroResolvesAcrossInputDim) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  // 0 copies input[0]=2, -1 is inferred from product(3*4)=12.
  ctx.Set("S", MakeShapeInput({0, -1}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(12)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneIsSymbolicWhenInputHasSymbolicDim) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("S", MakeShapeInput({-1, 2}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(2));
}

TEST(OnnxOptimShapesTensorReshape, UnknownShapeValueProducesSymbolicRankFromShapeInput) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));
  // Shape input is 1-D of size 3 but its values are unknown.
  ctx.Set("S", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(3))}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnMultipleNegativeOnes) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));
  ctx.Set("S", MakeShapeInput({-1, -1}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnInvalidNegativeValue) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));
  ctx.Set("S", MakeShapeInput({-2, 3}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnZeroOutsideInputRank) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));
  ctx.Set("S", MakeShapeInput({3, 0}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnIncompatibleNegativeOneInference) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  // 6 elements / 4 != integer.
  ctx.Set("S", MakeShapeInput({-1, 4}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Concat");
  node.add_input("X");
  node.add_input("S");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("S", MakeShapeInput({2}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, PropagatesValueAsShapeWhenOutputIs1D) {
  // Reshape of a 1-D shape tensor to ``[-1]`` (or to its own length) must
  // forward the ``ValueAsShape`` annotation so downstream consumers can
  // recover the per-element symbolic/concrete values.
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor data(
      nullptr, core::symbolic::TensorType::kInt64,
      core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(3))});
  data.SetValueAsShape(core::symbolic::SymShape{
      core::symbolic::SymDim("N"), core::symbolic::SymDim(4), core::symbolic::SymDim("M")});
  ctx.Set("X", std::move(data));
  ctx.Set("S", MakeShapeInput({-1}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_TRUE(ctx.Get("Y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4),
                                      core::symbolic::SymDim("M")}));
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorReshape, DoesNotPropagateValueAsShapeWhenOutputRankNot1) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor data(
      nullptr, core::symbolic::TensorType::kInt64,
      core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(4))});
  data.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                               core::symbolic::SymDim(4), core::symbolic::SymDim(5)});
  ctx.Set("X", std::move(data));
  ctx.Set("S", MakeShapeInput({2, 2}));

  onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_FALSE(ctx.Get("Y").HasValueAsShape());
}

TEST(OnnxOptimShapesTensorReshape, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeReshapeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReshape(ctx, node), std::out_of_range);
}

namespace {

NodeProto MakeSliceNode(const std::string &data = "X", const std::string &starts = "Starts",
                        const std::string &ends = "Ends", const std::string &axes = "",
                        const std::string &steps = "", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Slice");
  node.add_input(data);
  node.add_input(starts);
  node.add_input(ends);
  if (!axes.empty()) {
    node.add_input(axes);
  }
  if (!steps.empty()) {
    if (axes.empty()) {
      node.add_input("");
    }
    node.add_input(steps);
  }
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorSlice, ComputesKnownShapeWithAxesAndSteps) {
  NodeProto node = MakeSliceNode("X", "Starts", "Ends", "Axes", "Steps");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
  ctx.Set("Starts", MakeShapeInput({1, 0}));
  ctx.Set("Ends", MakeShapeInput({2, 3}));
  ctx.Set("Axes", MakeShapeInput({0, 1}));
  ctx.Set("Steps", MakeShapeInput({1, 2}));

  onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapesTensorSlice, UsesDefaultAxesWhenOmitted) {
  NodeProto node = MakeSliceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
  ctx.Set("Starts", MakeShapeInput({0, 1}));
  ctx.Set("Ends", MakeShapeInput({-1, 1000}));

  onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorSlice, KeepsAnchorDimsAndBuildsSliceExpression) {
  NodeProto node = MakeSliceNode("X", "Starts", "Ends", "Axes");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("a"),
                                                                  core::symbolic::SymDim("b"),
                                                                  core::symbolic::SymDim("c")}));
  ctx.Set("Starts", MakeShapeInput({0}));
  ctx.Set("Ends", MakeShapeInput({-1}));
  ctx.Set("Axes", MakeShapeInput({2}));

  onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("a"), core::symbolic::SymDim("b"),
                                      core::symbolic::SymDim("c-1")}));
}

TEST(OnnxOptimShapesTensorSlice, FallsBackToSymbolicWhenBoundsUnknown) {
  NodeProto node = MakeSliceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
  ctx.Set("Starts", core::symbolic::SymTensor(
                        nullptr, core::symbolic::TensorType::kInt64,
                        core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(2))}));
  ctx.Set("Ends", MakeShapeInput({2, 3}));

  onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorSlice, RejectsWrongOpType) {
  NodeProto node = MakeSliceNode();
  node.set_op_type("Reshape");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("Starts", MakeShapeInput({0}));
  ctx.Set("Ends", MakeShapeInput({1}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSlice, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeSliceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("Starts", MakeShapeInput({0}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSlice(ctx, node), std::out_of_range);
}

namespace {

NodeProto MakeCastNode(int64_t to, bool with_to_attr = true) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("X");
  node.add_output("Y");
  if (with_to_attr) {
    AddAttribute<int64_t>(node, "to", to);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorCast, PreservesShapeAndUsesToAttributeDtype) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorCast, PreservesSymbolicDimensions) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::INT64));
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorCast, ThrowsWhenToAttributeMissing) {
  NodeProto node = MakeCastNode(0, /*with_to_attr=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCast, ThrowsOnUnsupportedToValue) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::UNDEFINED));
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCast, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::FLOAT));
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node), std::out_of_range);
}

TEST(OnnxOptimShapesTensorCast, RejectsWrongOpType) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::FLOAT));
  node.set_op_type("NotCast");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// CastLike shape inference tests.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastLikeNode(int input_count = 2) {
  NodeProto node;
  node.set_op_type("CastLike");
  if (input_count >= 1) {
    node.add_input("X");
  }
  if (input_count >= 2) {
    node.add_input("TT");
  }
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorCastLike, PreservesShapeAndUsesTargetTypeDtype) {
  NodeProto node = MakeCastLikeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("TT", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble,
                                          core::symbolic::SymShape{core::symbolic::SymDim(1)}));

  onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorCastLike, PreservesSymbolicDimensions) {
  NodeProto node = MakeCastLikeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("TT", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                          core::symbolic::SymShape{core::symbolic::SymDim(1)}));

  onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorCastLike, IgnoresTargetTypeShape) {
  // Output shape must come from input(0), not from input(1).
  NodeProto node = MakeCastLikeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(5)}));
  ctx.Set("TT", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                          core::symbolic::SymShape{core::symbolic::SymDim(99),
                                                                   core::symbolic::SymDim(7)}));

  onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsOnUndefinedTargetTypeDtype) {
  NodeProto node = MakeCastLikeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("TT", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUndefined,
                                          core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsWhenSecondInputMissing) {
  NodeProto node = MakeCastLikeNode(/*input_count=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeCastLikeNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node), std::out_of_range);
}

TEST(OnnxOptimShapesTensorCastLike, RejectsWrongOpType) {
  NodeProto node = MakeCastLikeNode();
  node.set_op_type("NotCastLike");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("TT", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                          core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// AffineGrid shape inference tests.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeAffineGridNode(int64_t align_corners = 0) {
  NodeProto node;
  node.set_op_type("AffineGrid");
  node.add_input("theta");
  node.add_input("size");
  node.add_output("grid");
  if (align_corners != 0) {
    AddAttribute<int64_t>(node, "align_corners", align_corners);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorAffineGrid, FullyStatic2DFromConstantSize) {
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  // size is a 1-D INT64 tensor of length 4 whose constant value is
  // (N=2, C=3, H=5, W=6).
  core::symbolic::SymTensor size_tensor(nullptr, core::symbolic::TensorType::kInt64,
                                        core::symbolic::SymShape{core::symbolic::SymDim(4)});
  size_tensor.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                               core::symbolic::SymDim(5), core::symbolic::SymDim(6)});
  ctx.Set("size", std::move(size_tensor));

  onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  ASSERT_TRUE(ctx.Has("grid"));
  const core::symbolic::SymShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 5);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("grid").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesTensorAffineGrid, FullyStatic3DFromConstantSize) {
  NodeProto node = MakeAffineGridNode(/*align_corners=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3),
                                                                      core::symbolic::SymDim(4)}));
  core::symbolic::SymTensor size_tensor(nullptr, core::symbolic::TensorType::kInt64,
                                        core::symbolic::SymShape{core::symbolic::SymDim(5)});
  size_tensor.SetValueAsShape(core::symbolic::SymShape{
      core::symbolic::SymDim(2), core::symbolic::SymDim(3), core::symbolic::SymDim(4),
      core::symbolic::SymDim(5), core::symbolic::SymDim(6)});
  ctx.Set("size", std::move(size_tensor));

  onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const core::symbolic::SymShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 6);
  EXPECT_EQ(out[4].AsInt(), 3);
}

TEST(OnnxOptimShapesTensorAffineGrid, SymbolicSpatialDimsWhenSizeValueMissing) {
  // theta tells us it's a 2D AffineGrid, but ``size`` has no known value;
  // the output spatial dims should be left symbolic.
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const core::symbolic::SymShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[1].IsExpr()); // H
  EXPECT_TRUE(out[2].IsExpr()); // W
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesTensorAffineGrid, ModeInferredFromSizeLengthWhenThetaSymbolic) {
  // theta has rank 3 but its inner dims are symbolic; the 3D vs 2D mode is
  // resolved from the static size length (here 5 → 3D).
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim(7),
                                                             core::symbolic::SymDim("r"),
                                                             core::symbolic::SymDim("c")}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const core::symbolic::SymShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 7);
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_TRUE(out[3].IsExpr());
  EXPECT_EQ(out[4].AsInt(), 3);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsInvalidThetaInnerDims) {
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(4),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsThetaSizeModeMismatch) {
  // theta is 2D ((N, 2, 3)) but size has length 5 (3D).
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(5)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsBadSizeLength) {
  NodeProto node = MakeAffineGridNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsWrongOpType) {
  NodeProto node = MakeAffineGridNode();
  node.set_op_type("GridSample");
  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  ctx.Set("size", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Expand shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeGridSampleNode() {
  NodeProto node;
  node.set_op_type("GridSample");
  node.add_input("X");
  node.add_input("grid");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorGridSample, Static2DPropagatesNCAndGridSpatial) {
  // X: (N=2, C=3, H=4, W=5), grid: (N=2, H_out=6, W_out=7, 2).
  // Expected Y shape: (2, 3, 6, 7).
  NodeProto node = MakeGridSampleNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
  ctx.Set("grid",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(6),
                                       core::symbolic::SymDim(7), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeGridSample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 7);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesTensorGridSample, Static3DVolumetric) {
  // X: (1, 2, 3, 4, 5), grid: (1, 6, 7, 8, 3) → Y: (1, 2, 6, 7, 8).
  NodeProto node = MakeGridSampleNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kDouble,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                            core::symbolic::SymDim(5)}));
  ctx.Set("grid", core::symbolic::SymTensor(
                      nullptr, core::symbolic::TensorType::kDouble,
                      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(6),
                                               core::symbolic::SymDim(7), core::symbolic::SymDim(8),
                                               core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeGridSample(ctx, node);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 7);
  EXPECT_EQ(out[4].AsInt(), 8);
}

TEST(OnnxOptimShapesTensorGridSample, SymbolicBatchIsMerged) {
  // X has symbolic N, grid has concrete N=4 → output N=4.
  NodeProto node = MakeGridSampleNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(8), core::symbolic::SymDim(8)}));
  ctx.Set("grid",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim("H"),
                                       core::symbolic::SymDim("W"), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeGridSample(ctx, node);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 4);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_TRUE(out[3].IsExpr());
}

TEST(OnnxOptimShapesTensorGridSample, RejectsRankMismatch) {
  NodeProto node = MakeGridSampleNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  ctx.Set("grid", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                            core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                     core::symbolic::SymDim(2),
                                                                     core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeGridSample(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorGridSample, RejectsBadGridTrailingDim) {
  NodeProto node = MakeGridSampleNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  // 2D GridSample: last dim must be 2, not 3.
  ctx.Set("grid",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                       core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeGridSample(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Expand shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeExpandNode(const std::string &input = "X", const std::string &shape = "S",
                         const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Expand");
  node.add_input(input);
  node.add_input(shape);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorExpand, BroadcastsKnownShapeInputToLargerShape) {
  // input: [3, 1] → expand to [2, 3, 6] → output shape [2, 3, 6]
  NodeProto node = MakeExpandNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(1)}));
  ctx.Set("S", MakeShapeInput({2, 3, 6}));

  onnx_shapes::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                      core::symbolic::SymDim(6)}));
}

TEST(OnnxOptimShapesTensorExpand, BroadcastsDimUnchangedCase) {
  // input: [3, 1] → expand to [3, 4] → output shape [3, 4]
  NodeProto node = MakeExpandNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(1)}));
  ctx.Set("S", MakeShapeInput({3, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorExpand, PreservesInputDtype) {
  NodeProto node = MakeExpandNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4)}));
  ctx.Set("S", MakeShapeInput({2, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorExpand, FallsBackToSymbolicWhenShapeUnknown) {
  // shape input has no value annotation; its static shape is [3] → output rank 3, symbolic dims.
  NodeProto node = MakeExpandNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("S", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorExpand, RejectsWrongOpType) {
  NodeProto node = MakeExpandNode();
  node.set_op_type("Reshape");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("S", MakeShapeInput({3}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeExpand(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tile shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTileNode(const std::string &input = "X", const std::string &repeats = "R",
                       const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Tile");
  node.add_input(input);
  node.add_input(repeats);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTile, MultipliesEachDimByRepeats) {
  // input: [2, 3] tiled by [2, 4] -> output [4, 12]
  NodeProto node = MakeTileNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(12)}));
}

TEST(OnnxOptimShapesTensorTile, RepeatsOneLeavesDimsUnchanged) {
  NodeProto node = MakeTileNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(5)}));
  ctx.Set("R", MakeShapeInput({1, 1}));

  onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorTile, SymbolicInputDimYieldsSymbolicOutputDim) {
  NodeProto node = MakeTileNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 4}));

  onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(12));
}

TEST(OnnxOptimShapesTensorTile, FallsBackToSymbolicWhenRepeatsUnknown) {
  // repeats input has no value annotation; its static shape is [3] so the
  // output rank is 3 with every dim symbolic.
  NodeProto node = MakeTileNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("R", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorTile, RejectsRepeatsLengthMismatch) {
  NodeProto node = MakeTileNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 3, 4}));

  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorTile, RejectsWrongOpType) {
  NodeProto node = MakeTileNode();
  node.set_op_type("Expand");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("R", MakeShapeInput({2}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTile(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Upsample shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeUpsampleNodeV7(const std::vector<float> &scales) {
  NodeProto node;
  node.set_op_type("Upsample");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<float>>(node, "scales", scales);
  return node;
}

NodeProto MakeUpsampleNodeV9() {
  NodeProto node;
  node.set_op_type("Upsample");
  node.add_input("X");
  node.add_input("scales");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorUpsample, AppliesV7ScalesAttributeFloorToConcreteDims) {
  // Input [1, 1, 2, 2], scales [1, 1, 2, 3] -> [1, 1, 4, 6].
  NodeProto node = MakeUpsampleNodeV7({1.0f, 1.0f, 2.0f, 3.0f});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                      core::symbolic::SymDim(4), core::symbolic::SymDim(6)}));
}

TEST(OnnxOptimShapesTensorUpsample, AppliesV1WidthHeightScalesOnNCHWInput) {
  // 4-D NCHW input with v1 attributes: width_scale=3, height_scale=2.
  NodeProto node;
  node.set_op_type("Upsample");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<float>(node, "width_scale", 3.0f);
  AddAttribute<float>(node, "height_scale", 2.0f);

  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                      core::symbolic::SymDim(4), core::symbolic::SymDim(6)}));
}

TEST(OnnxOptimShapesTensorUpsample, SymbolicInputDimYieldsSymbolicOutputDim) {
  NodeProto node = MakeUpsampleNodeV7({1.0f, 1.0f, 2.0f, 2.0f});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 4u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(1));
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim(4));
  EXPECT_EQ(ctx.Get("Y").Shape()[3], core::symbolic::SymDim(4));
}

TEST(OnnxOptimShapesTensorUpsample, V9FloatScalesInputLeavesDimsSymbolic) {
  // v9/v10 takes scales as a runtime input (1-D FLOAT). The shape-lattice
  // does not carry float values, so output dims are symbolic but the rank
  // is preserved and the dtype matches the input.
  NodeProto node = MakeUpsampleNodeV9();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  ctx.Set("scales", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt32);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorUpsample, RejectsWrongOpType) {
  NodeProto node = MakeUpsampleNodeV7({1.0f, 1.0f, 2.0f, 2.0f});
  node.set_op_type("Resize");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorUpsample, RejectsScalesLengthMismatch) {
  NodeProto node = MakeUpsampleNodeV7({1.0f, 2.0f, 2.0f});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeUpsample(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Transpose shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTransposeNode(const std::string &input = "X", const std::string &out = "Y",
                            const std::vector<int64_t> &perm = {}) {
  NodeProto node;
  node.set_op_type("Transpose");
  node.add_input(input);
  node.add_output(out);
  if (!perm.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "perm", perm);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTranspose, DefaultsToReversePermutation) {
  NodeProto node = MakeTransposeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeTranspose(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(3),
                                      core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapesTensorTranspose, AppliesExplicitPermutation) {
  NodeProto node = MakeTransposeNode("X", "Y", {1, 0, 2});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeTranspose(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(2),
                                      core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorTranspose, RejectsPermutationWithDuplicateAxis) {
  NodeProto node = MakeTransposeNode("X", "Y", {0, 0});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTranspose(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// DepthToSpace shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeDepthToSpaceNode(const std::string &input = "X", const std::string &out = "Y",
                               int64_t blocksize = 2, bool with_blocksize = true,
                               const std::string &mode = "") {
  NodeProto node;
  node.set_op_type("DepthToSpace");
  node.add_input(input);
  node.add_output(out);
  if (with_blocksize) {
    AddAttribute<int64_t>(node, "blocksize", blocksize);
  }
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorDepthToSpace, ComputesConcreteOutputShape) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/3);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(18),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2),
                                      core::symbolic::SymDim(12), core::symbolic::SymDim(15)}));
}

TEST(OnnxOptimShapesTensorDepthToSpace, PreservesSymbolicBatchDim) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt32,
                   core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(8),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(2));
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim(6));
  EXPECT_EQ(ctx.Get("Y").Shape()[3], core::symbolic::SymDim(8));
}

TEST(OnnxOptimShapesTensorDepthToSpace, RejectsMissingBlocksize) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/0, /*with_blocksize=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorDepthToSpace, RejectsNonPositiveBlocksize) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorDepthToSpace, RejectsNonRank4Input) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(8)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorDepthToSpace, RejectsChannelNotDivisibleByBlocksizeSquared) {
  NodeProto node = MakeDepthToSpaceNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(5),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeDepthToSpace(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// SpaceToDepth shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeSpaceToDepthNode(const std::string &input = "X", const std::string &out = "Y",
                               int64_t blocksize = 2, bool with_blocksize = true) {
  NodeProto node;
  node.set_op_type("SpaceToDepth");
  node.add_input(input);
  node.add_output(out);
  if (with_blocksize) {
    AddAttribute<int64_t>(node, "blocksize", blocksize);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorSpaceToDepth, ComputesConcreteOutputShape) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/3);
  core::shapes::ShapesContext ctx;
  ctx.Set("X",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2),
                                       core::symbolic::SymDim(12), core::symbolic::SymDim(15)}));

  onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(18),
                                      core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorSpaceToDepth, PreservesSymbolicBatchDim) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt32,
                   core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(2),
                                            core::symbolic::SymDim(6), core::symbolic::SymDim(8)}));

  onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(8));
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim(3));
  EXPECT_EQ(ctx.Get("Y").Shape()[3], core::symbolic::SymDim(4));
}

TEST(OnnxOptimShapesTensorSpaceToDepth, RejectsMissingBlocksize) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/0, /*with_blocksize=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSpaceToDepth, RejectsNonPositiveBlocksize) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSpaceToDepth, RejectsNonRank4Input) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(8)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSpaceToDepth, RejectsSpatialDimNotDivisibleByBlocksize) {
  NodeProto node = MakeSpaceToDepthNode("X", "Y", /*blocksize=*/2);
  core::shapes::ShapesContext ctx;
  // H=3 is not divisible by blocksize=2.
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSpaceToDepth(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Squeeze / Unsqueeze shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeSqueezeNode(const std::string &data = "X", const std::string &axes = "A",
                          const std::string &out = "Y", bool with_axes = true) {
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input(data);
  if (with_axes) {
    node.add_input(axes);
  }
  node.add_output(out);
  return node;
}

NodeProto MakeUnsqueezeNode(const std::string &data = "X", const std::string &axes = "A",
                            const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Unsqueeze");
  node.add_input(data);
  node.add_input(axes);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorSqueeze, RemovesExplicitAxes) {
  NodeProto node = MakeSqueezeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(1)}));
  ctx.Set("A", MakeShapeInput({1, 3}));

  onnx_shapes::shapes::tensor::ComputeShapeSqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorSqueeze, WithoutAxesRemovesConcreteUnitDims) {
  NodeProto node = MakeSqueezeNode("X", "A", "Y", /*with_axes=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                            core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::tensor::ComputeShapeSqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorUnsqueeze, InsertsDimsAtGivenAxes) {
  NodeProto node = MakeUnsqueezeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("A", MakeShapeInput({0, 2}));

  onnx_shapes::shapes::tensor::ComputeShapeUnsqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                      core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorUnsqueeze, RejectsDuplicateAxes) {
  NodeProto node = MakeUnsqueezeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("A", MakeShapeInput({1, 1}));

  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeUnsqueeze(ctx, node),
               std::invalid_argument);
}

// Unsqueeze of a scalar INT64 tensor carrying a ValueAsShape annotation (e.g.
// after Gather(Shape(x), scalar_idx)) must forward the annotation unchanged.
// This is required for the pattern
//   Reshape(x, Concat(Unsqueeze(Gather(Shape(y),1),0), Unsqueeze(Gather(Shape(z),1),0)))
// to resolve the Reshape output shape without introducing undefined symbolic
// dimension names.
TEST(OnnxOptimShapesTensorUnsqueeze, PropagatesValueAsShapeFromScalarData) {
  NodeProto node = MakeUnsqueezeNode();
  core::shapes::ShapesContext ctx;

  // Scalar INT64 tensor representing a single symbolic dimension "D".
  core::symbolic::SymTensor scalar_data(nullptr, core::symbolic::TensorType::kInt64,
                                        core::symbolic::SymShape{});
  scalar_data.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("D")});
  ctx.Set("X", std::move(scalar_data));
  // axes = [0]
  ctx.Set("A", MakeShapeInput({0}));

  onnx_shapes::shapes::tensor::ComputeShapeUnsqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  // ValueAsShape must be forwarded so downstream Concat can propagate it.
  ASSERT_TRUE(ctx.Get("Y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y").ValueAsShape(), (core::symbolic::SymShape{core::symbolic::SymDim("D")}));
}

// Same as above but for the Squeeze direction: removing a size-1 axis from a
// 1-D INT64 tensor must not drop the ValueAsShape annotation.
TEST(OnnxOptimShapesTensorSqueeze, PropagatesValueAsShapeWhenRemovingUnitAxis) {
  NodeProto node = MakeSqueezeNode();
  core::shapes::ShapesContext ctx;

  // 1-D INT64 tensor of shape [1] with VAS = ["D"].
  core::symbolic::SymTensor vec_data(nullptr, core::symbolic::TensorType::kInt64,
                                     core::symbolic::SymShape{core::symbolic::SymDim(1)});
  vec_data.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("D")});
  ctx.Set("X", std::move(vec_data));
  // axes = [0]
  ctx.Set("A", MakeShapeInput({0}));

  onnx_shapes::shapes::tensor::ComputeShapeSqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  // Shape after squeezing the single axis is scalar (rank 0).
  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
  // ValueAsShape must be forwarded unchanged.
  ASSERT_TRUE(ctx.Get("Y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y").ValueAsShape(), (core::symbolic::SymShape{core::symbolic::SymDim("D")}));
}

// ---------------------------------------------------------------------------
// NonZero shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeNonZeroNode(const std::string &input = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("NonZero");
  node.add_input(input);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorNonZero, ProducesRank2Int64WithRankAsFirstDim) {
  NodeProto node = MakeNonZeroNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeNonZero(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const auto &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  ASSERT_TRUE(y.Shape()[0].IsInt());
  EXPECT_EQ(y.Shape()[0].AsInt(), 3);
  // The number of non-zero elements is a runtime value: must be symbolic.
  EXPECT_FALSE(y.Shape()[1].IsInt());
}

TEST(OnnxOptimShapesTensorNonZero, ScalarInputProducesShapeZeroByNnz) {
  NodeProto node = MakeNonZeroNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{}));

  onnx_shapes::shapes::tensor::ComputeShapeNonZero(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const auto &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  ASSERT_TRUE(y.Shape()[0].IsInt());
  EXPECT_EQ(y.Shape()[0].AsInt(), 0);
  EXPECT_FALSE(y.Shape()[1].IsInt());
}

TEST(OnnxOptimShapesTensorNonZero, RejectsWrongOpType) {
  NodeProto node = MakeNonZeroNode();
  node.set_op_type("Abs");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeNonZero(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorNonZero, RecordsNnzUpperBoundFromInputShape) {
  NodeProto node = MakeNonZeroNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeNonZero(ctx, node);

  // The number of non-zero elements ``nnz`` is upper-bounded by the
  // product of the input dimensions, i.e. ``4*N``.
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 1u);
  EXPECT_TRUE(ctx.HasLessEqualConstraint("NonZero_Y_nnz", "4*N"));
}

TEST(OnnxOptimShapesTensorNonZero, ScalarInputRecordsNoConstraint) {
  NodeProto node = MakeNonZeroNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{}));

  onnx_shapes::shapes::tensor::ComputeShapeNonZero(ctx, node);

  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 0u);
}

// ---------------------------------------------------------------------------
// Compress shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCompressNode(const std::optional<int64_t> &axis = std::nullopt,
                           const std::string &input = "X", const std::string &cond = "C",
                           const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Compress");
  node.add_input(input);
  node.add_input(cond);
  node.add_output(out);
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorCompress, NoAxisRecordsFlattenedUpperBound) {
  NodeProto node = MakeCompressNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim("N")}));
  ctx.Set("C", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim("M")}));

  onnx_shapes::shapes::tensor::ComputeShapeCompress(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 1u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "Compress_Y_count");
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 1u);
  EXPECT_TRUE(ctx.HasLessEqualConstraint("Compress_Y_count", "3*N"));
}

TEST(OnnxOptimShapesTensorCompress, AxisRecordsInputDimUpperBound) {
  NodeProto node = MakeCompressNode(/*axis=*/1);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim("N")}));
  ctx.Set("C", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim("M")}));

  onnx_shapes::shapes::tensor::ComputeShapeCompress(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("Y").Shape()[0], core::symbolic::SymDim(3));
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1].AsExpr(), "Compress_Y_count");
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 1u);
  EXPECT_TRUE(ctx.HasLessEqualConstraint("Compress_Y_count", "N"));
}

TEST(OnnxOptimShapesTensorCompress, AxisIntegerInputDimRecordsIntegerUpperBound) {
  NodeProto node = MakeCompressNode(/*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(7),
                                                                  core::symbolic::SymDim("N")}));
  ctx.Set("C", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(7)}));

  onnx_shapes::shapes::tensor::ComputeShapeCompress(ctx, node);

  EXPECT_TRUE(ctx.HasLessEqualConstraint("Compress_Y_count", "7"));
}

// ---------------------------------------------------------------------------
// Trilu shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTriluNode(bool with_k = false, const std::string &input = "X",
                        const std::string &k = "K", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Trilu");
  node.add_input(input);
  if (with_k) {
    node.add_input(k);
  }
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTrilu, PreservesShapeAndDtypeForMatrix) {
  NodeProto node = MakeTriluNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeTrilu(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorTrilu, PreservesBatchedShapeWithKInput) {
  NodeProto node = MakeTriluNode(/*with_k=*/true);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim("B"),
                                                                  core::symbolic::SymDim(5),
                                                                  core::symbolic::SymDim(5)}));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{}));

  onnx_shapes::shapes::tensor::ComputeShapeTrilu(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(5));
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim(5));
}

TEST(OnnxOptimShapesTensorTrilu, RejectsInputRankLessThanTwo) {
  NodeProto node = MakeTriluNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTrilu(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorTrilu, RejectsWrongOpType) {
  NodeProto node = MakeTriluNode();
  node.set_op_type("Abs");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTrilu(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// ReverseSequence shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeReverseSequenceNode(const std::string &input = "X", const std::string &seq = "S",
                                  const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("ReverseSequence");
  node.add_input(input);
  node.add_input(seq);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorReverseSequence, PreservesShapeAndDtype) {
  NodeProto node = MakeReverseSequenceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("S", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeReverseSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorReverseSequence, PreservesRank3Shape) {
  NodeProto node = MakeReverseSequenceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim("T"),
                                                                  core::symbolic::SymDim("B"),
                                                                  core::symbolic::SymDim(8)}));
  ctx.Set("S", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim("B")}));

  onnx_shapes::shapes::tensor::ComputeShapeReverseSequence(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim(8));
}

TEST(OnnxOptimShapesTensorReverseSequence, RejectsInputRankLessThanTwo) {
  NodeProto node = MakeReverseSequenceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  ctx.Set("S", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReverseSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReverseSequence, RejectsSequenceLensRankNotOne) {
  NodeProto node = MakeReverseSequenceNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("S", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReverseSequence(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReverseSequence, RejectsWrongOpType) {
  NodeProto node = MakeReverseSequenceNode();
  node.set_op_type("Abs");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  ctx.Set("S", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeReverseSequence(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Split — divides ``input`` along ``axis`` into per-output tensors.
// ---------------------------------------------------------------------------

TEST(OnnxOptimShapesTensorSplit, EqualPartsViaNumOutputsAttribute) {
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  node.add_output("Y2");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 3);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(6), core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  for (const char *name : {"Y0", "Y1", "Y2"}) {
    ASSERT_TRUE(ctx.Has(name));
    EXPECT_EQ(ctx.Get(name).Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(ctx.Get(name).Shape(),
              (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)}));
  }
}

TEST(OnnxOptimShapesTensorSplit, UnevenSplitLastChunkSmaller) {
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  node.add_output("Y2");
  node.add_output("Y3");
  AddAttribute<int64_t>(node, "num_outputs", 4);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(7)}));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y2").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y3").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(1)}));
}

TEST(OnnxOptimShapesTensorSplit, NegativeAxisIsResolvedFromRank) {
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", -1);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(6)}));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
  EXPECT_EQ(ctx.Get("Y1").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesTensorSplit, VariablePartsViaSplitAttribute) {
  // Opset 1/2/11 carry ``split`` as an INTS attribute.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<std::vector<int64_t>>(node, "split", {2, 4});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapesTensorSplit, RejectsAxisOutOfRange) {
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  AddAttribute<int64_t>(node, "axis", 5);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(6)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSplit, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("X");
  node.add_output("Y0");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorSplit, PropagatesValueAsShapeViaSplitAttribute) {
  // ``X = [N, 1, B]`` (value-as-shape) split into ``[2, 1]`` along axis 0 must
  // yield per-output value-as-shape annotations ``[N, 1]`` and ``[B]``.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<std::vector<int64_t>>(node, "split", {2, 1});

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  x.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(1),
                                             core::symbolic::SymDim("B")});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  ASSERT_TRUE(ctx.Has("Y0"));
  ASSERT_TRUE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y0").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(1)}));

  ASSERT_TRUE(ctx.Has("Y1"));
  ASSERT_TRUE(ctx.Get("Y1").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y1").ValueAsShape(), (core::symbolic::SymShape{core::symbolic::SymDim("B")}));
}

TEST(OnnxOptimShapesTensorSplit, PropagatesValueAsShapeViaNumOutputs) {
  // ``X = [N, B, 1, 2]`` (value-as-shape) split into 2 even chunks along
  // axis 0 must yield value-as-shape ``[N, B]`` and ``[1, 2]``.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 2);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(4)});
  x.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                             core::symbolic::SymDim("B"), core::symbolic::SymDim(1),
                                             core::symbolic::SymDim(2)});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  ASSERT_TRUE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y0").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim("B")}));
  ASSERT_TRUE(ctx.Get("Y1").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y1").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapesTensorSplit, PropagatesValueAsShapeViaNumOutputsWithSymbolicAxis) {
  // The declared input shape has a symbolic axis dim, but the
  // ``ValueAsShape`` is known (5 entries). ``num_outputs=2`` must derive
  // sizes ``[(d+1)/2, d/2] = [3, 2]`` from the VAS rank and slice it
  // accordingly.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 2);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim("D")});
  x.SetValueAsShape(core::symbolic::SymShape{
      core::symbolic::SymDim("a"), core::symbolic::SymDim("b"), core::symbolic::SymDim("c"),
      core::symbolic::SymDim(4), core::symbolic::SymDim(5)});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  // Y0 takes (5+1)/2 = 3 entries: a, b, c.
  ASSERT_TRUE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_EQ(ctx.Get("Y0").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("a"), core::symbolic::SymDim("b"),
                                      core::symbolic::SymDim("c")}));
  // Y1 takes 5/2 = 2 entries: 4, 5.
  ASSERT_TRUE(ctx.Get("Y1").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y1").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorSplit, PropagatesValueAsShapeViaNumOutputsThreeWayUneven) {
  // ``num_outputs=3`` on a length-7 ``ValueAsShape`` must yield sizes
  // ``[3, 3, 1]`` (chunk = ceil(7/3) = 3; last absorbs the remainder).
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  node.add_output("Y2");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 3);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(7)});
  x.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                             core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                             core::symbolic::SymDim(5), core::symbolic::SymDim(6),
                                             core::symbolic::SymDim(7)});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  ASSERT_TRUE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y0").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                      core::symbolic::SymDim(3)}));
  ASSERT_TRUE(ctx.Get("Y1").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y1").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5),
                                      core::symbolic::SymDim(6)}));
  ASSERT_TRUE(ctx.Get("Y2").HasValueAsShape());
  EXPECT_EQ(ctx.Get("Y2").ValueAsShape(), (core::symbolic::SymShape{core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapesTensorSplit, DoesNotPropagateValueAsShapeAlongNonZeroAxis) {
  // ``X`` is rank 2 and ``axis != 0``; even when it carries a value-as-shape
  // we do not slice it (the annotation only makes sense along axis 0).
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 1);
  AddAttribute<std::vector<int64_t>>(node, "split", {2, 2});

  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kInt64,
      core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)});
  // A value-as-shape on a rank-2 tensor isn't meaningful but the propagation
  // must still refuse to slice it.
  x.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim("a"), core::symbolic::SymDim("b"),
                               core::symbolic::SymDim("c"), core::symbolic::SymDim("d")});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_FALSE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_FALSE(ctx.Get("Y1").HasValueAsShape());
}

TEST(OnnxOptimShapesTensorSplit, SymbolicAxisDimViaNumOutputsTwo) {
  // ``X`` has a purely symbolic axis dim ``d`` and no ``ValueAsShape``.
  // ``num_outputs=2`` must derive symbolic sizes ``[(d+1)/2, d/2]``.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 2);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim("d")});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("(1+d)//2")}));
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("d//2")}));
  EXPECT_FALSE(ctx.Get("Y0").HasValueAsShape());
  EXPECT_FALSE(ctx.Get("Y1").HasValueAsShape());
}

TEST(OnnxOptimShapesTensorSplit, SymbolicAxisDimViaNumOutputsThree) {
  // ``num_outputs=3`` with symbolic ``d`` yields ``[(d+2)//3, (d+2)//3,
  // d - 2*((d+2)//3)]`` for the per-output axis dims (after canonical
  // alphabetical reordering by ``simplify_expression``).
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  node.add_output("Y2");
  AddAttribute<int64_t>(node, "axis", 0);
  AddAttribute<int64_t>(node, "num_outputs", 3);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim("d")});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("(2+d)//3")}));
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("(2+d)//3")}));
  EXPECT_EQ(ctx.Get("Y2").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("d-2*((2+d)//3)")}));
}

TEST(OnnxOptimShapesTensorSplit, SymbolicAxisDimViaOutputCountNoNumOutputs) {
  // Older opsets: no ``split`` and no ``num_outputs`` — the per-output count
  // comes from the declared number of outputs. Symbolic ``d`` still yields
  // chunking expressions rather than fresh placeholders.
  NodeProto node;
  node.set_op_type("Split");
  node.add_input("X");
  node.add_output("Y0");
  node.add_output("Y1");
  AddAttribute<int64_t>(node, "axis", 0);

  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim("d")});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", std::move(x));

  onnx_shapes::shapes::tensor::ComputeShapeSplit(ctx, node);

  EXPECT_EQ(ctx.Get("Y0").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("(1+d)//2")}));
  EXPECT_EQ(ctx.Get("Y1").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim("d//2")}));
}

// ---------------------------------------------------------------------------
// TensorScatter shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTensorScatterNode(bool with_write_indices = true) {
  NodeProto node;
  node.set_op_type("TensorScatter");
  node.add_input("past_cache");
  node.add_input("update");
  if (with_write_indices) {
    node.add_input("write_indices");
  }
  node.add_output("present_cache");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTensorScatter, PreservesPastCacheShapeAndDtype) {
  NodeProto node = MakeTensorScatterNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("past_cache",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(1),
                                       core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
  ctx.Set("update",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(1),
                                       core::symbolic::SymDim(1), core::symbolic::SymDim(5)}));
  ctx.Set("write_indices",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::tensor::ComputeShapeTensorScatter(ctx, node);

  ASSERT_TRUE(ctx.Has("present_cache"));
  EXPECT_EQ(ctx.Get("present_cache").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("present_cache").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(1),
                                      core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorTensorScatter, AcceptsMissingWriteIndices) {
  NodeProto node = MakeTensorScatterNode(/*with_write_indices=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("past_cache",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                    core::symbolic::SymShape{core::symbolic::SymDim(3),
                                                             core::symbolic::SymDim(4),
                                                             core::symbolic::SymDim(5)}));
  ctx.Set("update", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                              core::symbolic::SymShape{core::symbolic::SymDim(3),
                                                                       core::symbolic::SymDim(2),
                                                                       core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::tensor::ComputeShapeTensorScatter(ctx, node);

  ASSERT_TRUE(ctx.Has("present_cache"));
  EXPECT_EQ(ctx.Get("present_cache").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("present_cache").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                      core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorTensorScatter, RejectsRankMismatch) {
  NodeProto node = MakeTensorScatterNode(/*with_write_indices=*/false);
  core::shapes::ShapesContext ctx;
  ctx.Set("past_cache",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                             core::symbolic::SymDim(4),
                                                             core::symbolic::SymDim(5)}));
  ctx.Set("update", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                       core::symbolic::SymDim(5)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTensorScatter(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorTensorScatter, RejectsBatchAxis) {
  NodeProto node = MakeTensorScatterNode(/*with_write_indices=*/false);
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::INT);
  attr->set_i(0);
  core::shapes::ShapesContext ctx;
  ctx.Set("past_cache",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                             core::symbolic::SymDim(4),
                                                             core::symbolic::SymDim(5)}));
  ctx.Set("update", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                       core::symbolic::SymDim(4),
                                                                       core::symbolic::SymDim(5)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTensorScatter(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorTensorScatter, RejectsWrongOpType) {
  NodeProto node = MakeTensorScatterNode(/*with_write_indices=*/false);
  node.set_op_type("Abs");
  core::shapes::ShapesContext ctx;
  ctx.Set("past_cache",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                             core::symbolic::SymDim(4),
                                                             core::symbolic::SymDim(5)}));
  ctx.Set("update", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                       core::symbolic::SymDim(4),
                                                                       core::symbolic::SymDim(5)}));
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeTensorScatter(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// OneHot shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeOneHotNode(bool set_axis = false, int64_t axis = -1) {
  NodeProto node;
  node.set_op_type("OneHot");
  node.add_input("indices");
  node.add_input("depth");
  node.add_input("values");
  node.add_output("Y");
  if (set_axis) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("axis");
    attr->set_type(AttributeProto::INT);
    attr->set_i(axis);
  }
  return node;
}

void SeedOneHotInputs(core::shapes::ShapesContext &ctx,
                      const core::symbolic::SymShape &indices_shape,
                      core::symbolic::TensorType values_dtype, std::optional<int64_t> depth_value) {
  ctx.Set("indices",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, indices_shape));
  core::symbolic::SymTensor depth(nullptr, core::symbolic::TensorType::kInt64,
                                  core::symbolic::SymShape{});
  if (depth_value.has_value()) {
    depth.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(*depth_value)});
  }
  ctx.Set("depth", std::move(depth));
  ctx.Set("values", core::symbolic::SymTensor(nullptr, values_dtype,
                                              core::symbolic::SymShape{core::symbolic::SymDim(2)}));
}

} // namespace

TEST(OnnxOptimShapesTensorOneHot, AppendsDepthDimWithDefaultAxis) {
  NodeProto node = MakeOneHotNode();
  core::shapes::ShapesContext ctx;
  SeedOneHotInputs(ctx,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
                   core::symbolic::TensorType::kFloat, /*depth_value=*/5);

  onnx_shapes::shapes::tensor::ComputeShapeOneHot(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                      core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapesTensorOneHot, InsertsDepthAtExplicitAxis) {
  NodeProto node = MakeOneHotNode(/*set_axis=*/true, /*axis=*/1);
  core::shapes::ShapesContext ctx;
  SeedOneHotInputs(ctx,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)},
                   core::symbolic::TensorType::kInt32, /*depth_value=*/10);

  onnx_shapes::shapes::tensor::ComputeShapeOneHot(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(10),
                                      core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapesTensorOneHot, UsesSymbolicDimWhenDepthUnknown) {
  NodeProto node = MakeOneHotNode();
  core::shapes::ShapesContext ctx;
  SeedOneHotInputs(ctx, core::symbolic::SymShape{core::symbolic::SymDim(3)},
                   core::symbolic::TensorType::kFloat, /*depth_value=*/std::nullopt);

  onnx_shapes::shapes::tensor::ComputeShapeOneHot(ctx, node);

  const auto &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0], core::symbolic::SymDim(3));
  EXPECT_TRUE(out[1].IsExpr());
}

TEST(OnnxOptimShapesTensorOneHot, RejectsAxisOutOfRange) {
  NodeProto node = MakeOneHotNode(/*set_axis=*/true, /*axis=*/5);
  core::shapes::ShapesContext ctx;
  SeedOneHotInputs(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)},
                   core::symbolic::TensorType::kFloat, /*depth_value=*/3);
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeOneHot(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorOneHot, RejectsWrongOpType) {
  NodeProto node = MakeOneHotNode();
  node.set_op_type("NotOneHot");
  core::shapes::ShapesContext ctx;
  SeedOneHotInputs(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)},
                   core::symbolic::TensorType::kFloat, /*depth_value=*/3);
  EXPECT_THROW(onnx_shapes::shapes::tensor::ComputeShapeOneHot(ctx, node), std::invalid_argument);
}

} // namespace Test
