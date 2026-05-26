// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/controlflow/shape_controlflow.h"

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

// Builds a NodeProto with a single op type, the given inputs and a
// single output named ``output``.
NodeProto MakeBodyNode(const std::string &op_type, const std::vector<std::string> &inputs,
                       const std::string &output) {
  NodeProto node;
  node.set_op_type(op_type);
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(output);
  return node;
}

// Builds a single-output sub-graph ``out_name = op_type(input)`` and
// registers ``out_name`` as a graph output.
GraphProto MakeUnaryBranch(const std::string &op_type, const std::string &input,
                           const std::string &out_name) {
  GraphProto g;
  *g.add_node() = MakeBodyNode(op_type, {input}, out_name);
  ValueInfoProto *out = g.add_output();
  out->set_name(out_name);
  return g;
}

// Builds an If node ``out = If(cond)`` with the given branches.
NodeProto MakeIfNode(const std::string &cond, const std::vector<std::string> &outputs,
                     const GraphProto &then_branch, const GraphProto &else_branch) {
  NodeProto node;
  node.set_op_type("If");
  node.add_input(cond);
  for (const auto &o : outputs) {
    node.add_output(o);
  }
  AttributeProto *t = node.add_attribute();
  t->set_name("then_branch");
  t->set_type(AttributeProto::AttributeType::GRAPH);
  t->set_g(then_branch);
  AttributeProto *e = node.add_attribute();
  e->set_name("else_branch");
  e->set_type(AttributeProto::AttributeType::GRAPH);
  e->set_g(else_branch);
  return node;
}

} // namespace

TEST(OnnxOptimShapeIf, IdenticalBranchesSameDtypeAndShape) {
  // then_branch: y_then = Abs(x);  else_branch: y_else = Abs(x)
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeIf, BranchesAgreeOnConstantTensor) {
  // then_branch: y_then = Constant(value=int64 [3]);
  // else_branch: y_else = Constant(value=int64 [3])
  GraphProto then_b;
  {
    NodeProto *n = then_b.add_node();
    n->set_op_type("Constant");
    n->add_output("y_then");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->mutable_t();
    t->set_data_type(TensorProto::DataType::INT64);
    t->add_dims(3);
    ValueInfoProto *out = then_b.add_output();
    out->set_name("y_then");
  }
  GraphProto else_b;
  {
    NodeProto *n = else_b.add_node();
    n->set_op_type("Constant");
    n->add_output("y_else");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->mutable_t();
    t->set_data_type(TensorProto::DataType::INT64);
    t->add_dims(3);
    ValueInfoProto *out = else_b.add_output();
    out->set_name("y_else");
  }
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  ASSERT_TRUE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 3);
}

TEST(OnnxOptimShapeIf, DtypeMismatchYieldsUndefined) {
  // then_branch returns a float tensor (Abs(x));
  // else_branch returns a bool tensor (And(b, b)).
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b;
  {
    NodeProto *n = else_b.add_node();
    n->set_op_type("And");
    n->add_input("b");
    n->add_input("b");
    n->add_output("y_else");
    ValueInfoProto *out = else_b.add_output();
    out->set_name("y_else");
  }
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));

  onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kUndefined);
  // Shapes agree, so the merged shape is preserved.
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeIf, DifferingDimsBecomeSymbolic) {
  // then_branch: y_then = Abs(a)  with a: [2, 3]
  // else_branch: y_else = Abs(b)  with b: [2, 5]
  GraphProto then_b = MakeUnaryBranch("Abs", "a", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "b", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("a", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("b", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  ASSERT_TRUE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 2);
  ASSERT_TRUE(ctx.Get("y").Shape()[1].IsExpr());
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsExpr(), "If_y_d1");
}

TEST(OnnxOptimShapeIf, RankMismatchProducesEmptyShape) {
  GraphProto then_b = MakeUnaryBranch("Abs", "a", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "b", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  ctx.Set("b", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_TRUE(ctx.Get("y").Shape().Empty());
}

TEST(OnnxOptimShapeIf, MultipleOutputsAreMergedIndependently) {
  // Two outputs (y1, y2) coming from two Abs nodes in each branch.
  GraphProto then_b;
  *then_b.add_node() = MakeBodyNode("Abs", {"a"}, "t1");
  *then_b.add_node() = MakeBodyNode("Abs", {"b"}, "t2");
  then_b.add_output()->set_name("t1");
  then_b.add_output()->set_name("t2");

  GraphProto else_b;
  *else_b.add_node() = MakeBodyNode("Abs", {"a"}, "e1");
  *else_b.add_node() = MakeBodyNode("Abs", {"b"}, "e2");
  else_b.add_output()->set_name("e1");
  else_b.add_output()->set_name("e2");

  NodeProto node = MakeIfNode("cond", {"y1", "y2"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape sa{onnx_optim::OptimDim(2)};
  onnx_optim::OptimShape sb{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, sa));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, sb));

  onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y1"));
  EXPECT_EQ(ctx.Get("y1").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y1").Shape(), sa);
  ASSERT_TRUE(ctx.Has("y2"));
  EXPECT_EQ(ctx.Get("y2").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y2").Shape(), sb);
}

TEST(OnnxOptimShapeIf, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotIf");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeIf, RejectsMissingThenBranchAttribute) {
  NodeProto node;
  node.set_op_type("If");
  node.add_input("cond");
  node.add_output("y");
  // Only else_branch attribute is provided.
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  AttributeProto *e = node.add_attribute();
  e->set_name("else_branch");
  e->set_type(AttributeProto::AttributeType::GRAPH);
  e->set_g(else_b);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeIf, RejectsWrongInputArity) {
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);
  node.add_input("extra"); // Now 2 inputs, which is invalid for If.

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("extra", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeIf, RejectsMismatchedOutputCount) {
  // then_branch produces 1 output, else_branch produces 2; the If node
  // declares 1 output. The mismatch on else_branch must be rejected.
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b;
  *else_b.add_node() = MakeBodyNode("Abs", {"x"}, "e1");
  *else_b.add_node() = MakeBodyNode("Abs", {"x"}, "e2");
  else_b.add_output()->set_name("e1");
  else_b.add_output()->set_name("e2");

  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesIf) {
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

} // namespace Test
