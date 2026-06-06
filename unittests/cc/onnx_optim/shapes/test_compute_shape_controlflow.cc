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

TEST(OnnxOptimShapeIf, RankMismatchThrows) {
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

  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
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

namespace Test {

namespace {

// Build a Loop body: inputs=(iter, cond_in, v_in), outputs=(cond_out, v_out, scan_out).
// cond_out = And(cond_in, cond_in); v_out = Abs(v_in); scan_out = Abs(v_in).
GraphProto BuildLoopBodyIdentityCarry() {
  GraphProto g;
  g.set_name("loop_body");
  for (const char *n : {"iter", "cond_in", "v_in"}) {
    g.add_input()->set_name(n);
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("And");
    n->add_input("cond_in");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Abs");
    n->add_input("v_in");
    n->add_output("v_out");
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Abs");
    n->add_input("v_in");
    n->add_output("scan_out");
  }
  for (const char *n : {"cond_out", "v_out", "scan_out"}) {
    g.add_output()->set_name(n);
  }
  return g;
}

NodeProto MakeLoopNode(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, const GraphProto &body) {
  NodeProto node;
  node.set_op_type("Loop");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  AttributeProto *b = node.add_attribute();
  b->set_name("body");
  b->set_type(AttributeProto::AttributeType::GRAPH);
  b->set_g(body);
  return node;
}

} // namespace

TEST(OnnxOptimShapeLoop, PropagatesCarriedShapeAndScanShape) {
  // Loop with 1 carried-dep and 1 scan output. v_initial shape is [2, 3].
  GraphProto body = BuildLoopBodyIdentityCarry();
  NodeProto node = MakeLoopNode({"M", "cond", "v_init"}, {"v_final", "scan_out"}, body);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("M", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, {}));
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("v_init", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.Has("v_final"));
  EXPECT_EQ(ctx.Get("v_final").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("v_final").Shape(), shape);

  ASSERT_TRUE(ctx.Has("scan_out"));
  EXPECT_EQ(ctx.Get("scan_out").Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("scan_out").Shape().Rank(), 3u);
  ASSERT_TRUE(ctx.Get("scan_out").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("scan_out").Shape()[0].AsExpr(), "Loop_scan_out_d0");
  EXPECT_EQ(ctx.Get("scan_out").Shape()[1].AsInt(), 2);
  EXPECT_EQ(ctx.Get("scan_out").Shape()[2].AsInt(), 3);
}

TEST(OnnxOptimShapeLoop, AcceptsOmittedMAndCond) {
  GraphProto body = BuildLoopBodyIdentityCarry();
  // Both M and cond are omitted using empty input names.
  NodeProto node = MakeLoopNode({"", "", "v_init"}, {"v_final", "scan_out"}, body);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("v_init", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.Has("v_final"));
  EXPECT_EQ(ctx.Get("v_final").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("v_final").Shape(), shape);
  ASSERT_EQ(ctx.Get("scan_out").Shape().Rank(), 2u);
}

TEST(OnnxOptimShapeLoop, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLoop");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeLoop(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeLoop, RejectsTooFewInputs) {
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input("M");
  node.add_output("scan");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeLoop(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeLoop, RejectsMissingBodyAttribute) {
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input("M");
  node.add_input("cond");
  node.add_output("scan");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("M", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, {}));
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeLoop(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesLoop) {
  GraphProto body = BuildLoopBodyIdentityCarry();
  NodeProto node = MakeLoopNode({"M", "cond", "v_init"}, {"v_final", "scan_out"}, body);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(5)};
  ctx.Set("M", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, {}));
  ctx.Set("cond", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("v_init", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("v_final"));
  EXPECT_EQ(ctx.Get("v_final").Shape(), shape);
}

namespace {

// Builds a Scan body that applies Abs to the scan-input element to produce
// the scan-output element. No state variables.
GraphProto BuildScanBodyIdentity() {
  GraphProto g;
  g.set_name("scan_body");
  g.add_input()->set_name("x_elt");
  NodeProto *n = g.add_node();
  n->set_op_type("Abs");
  n->add_input("x_elt");
  n->add_output("y_elt");
  g.add_output()->set_name("y_elt");
  return g;
}

NodeProto MakeScanNode(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, const GraphProto &body,
                       int64_t num_scan_inputs) {
  NodeProto node;
  node.set_op_type("Scan");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  AttributeProto *b = node.add_attribute();
  b->set_name("body");
  b->set_type(AttributeProto::AttributeType::GRAPH);
  b->set_g(body);
  AttributeProto *n = node.add_attribute();
  n->set_name("num_scan_inputs");
  n->set_type(AttributeProto::AttributeType::INT);
  n->set_i(num_scan_inputs);
  return node;
}

} // namespace

TEST(OnnxOptimShapeScan, PrependsTripCountAxisToScanOutput) {
  // Scan with 0 state vars, 1 scan input of shape [T, 3]. Identity body
  // produces an element of shape [3]; the stacked output should be [T, 3].
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/1);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape x_shape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, x_shape));

  onnx_optim::shapes::controlflow::ComputeShapeScan(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  ASSERT_TRUE(ctx.Get("Y").Shape()[0].IsInt());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsInt(), 4);
  ASSERT_TRUE(ctx.Get("Y").Shape()[1].IsInt());
  EXPECT_EQ(ctx.Get("Y").Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapeScan, HonorsScanOutputAxes) {
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/1);
  AttributeProto *axes = node.add_attribute();
  axes->set_name("scan_output_axes");
  axes->set_type(AttributeProto::AttributeType::INTS);
  axes->add_ints(1);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape x_shape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(7)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, x_shape));

  onnx_optim::shapes::controlflow::ComputeShapeScan(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  // Element shape is [7]; stacking T=5 along axis 1 yields [7, 5].
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsInt(), 7);
  EXPECT_EQ(ctx.Get("Y").Shape()[1].AsInt(), 5);
}

TEST(OnnxOptimShapeScan, RejectsMissingNumScanInputs) {
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node;
  node.set_op_type("Scan");
  node.add_input("X");
  node.add_output("Y");
  AttributeProto *b = node.add_attribute();
  b->set_name("body");
  b->set_type(AttributeProto::AttributeType::GRAPH);
  b->set_g(body);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape x_shape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, x_shape));
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeScan(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeScan, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotScan");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::controlflow::ComputeShapeScan(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesScan) {
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/1);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape x_shape{onnx_optim::OptimDim(6), onnx_optim::OptimDim(2)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, x_shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(), x_shape);
}

} // namespace Test
