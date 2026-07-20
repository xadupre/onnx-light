// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_shapes/shapes/controlflow/shape_controlflow.h"

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
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

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kUndefined);
  // Shapes agree, so the merged shape is preserved.
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeIf, DifferingDimsBecomeSymbolic) {
  // then_branch: y_then = Abs(a)  with a: [2, 3]
  // else_branch: y_else = Abs(b)  with b: [2, 5]
  GraphProto then_b = MakeUnaryBranch("Abs", "a", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "b", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("a", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("b", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  ASSERT_TRUE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 2);
  ASSERT_TRUE(ctx.Get("y").Shape()[1].IsExpr());
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsExpr(), "If_y_d1");
  // The merged dim is upper-bounded by max(3, 5) == 5.
  EXPECT_TRUE(ctx.HasLessEqualConstraint("If_y_d1", "5"));
}

TEST(OnnxOptimShapeIf, DifferingSymbolicDimsRecordsMaxUpperBound) {
  // Branches disagree on a symbolic dim: the merged dim is bounded
  // above by ``max(then_dim, else_dim)``.
  GraphProto then_b = MakeUnaryBranch("Abs", "a", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "b", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("N")}));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim("M")}));

  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  EXPECT_TRUE(ctx.Get("y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsExpr(), "If_y_d0");
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 1u);
  const auto &cs = ctx.LessEqualConstraints();
  ASSERT_EQ(cs.size(), 1u);
  const auto &c = *cs.begin();
  EXPECT_EQ(c.first, "If_y_d0");
  // The recorded upper bound expresses ``max(M, N)`` (encoded with ``^``).
  EXPECT_TRUE(c.second.find('^') != std::string::npos);
  EXPECT_TRUE(c.second.find('M') != std::string::npos);
  EXPECT_TRUE(c.second.find('N') != std::string::npos);
}

TEST(OnnxOptimShapeIf, RankMismatchThrows) {
  GraphProto then_b = MakeUnaryBranch("Abs", "a", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "b", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  ctx.Set("b", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape sa{core::symbolic::SymDim(2)};
  core::symbolic::SymShape sb{core::symbolic::SymDim(3), core::symbolic::SymDim(4)};
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, sa));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, sb));

  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  ASSERT_TRUE(ctx.Has("y1"));
  EXPECT_EQ(ctx.Get("y1").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y1").Shape(), sa);
  ASSERT_TRUE(ctx.Has("y2"));
  EXPECT_EQ(ctx.Get("y2").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y2").Shape(), sb);
}

TEST(OnnxOptimShapeIf, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotIf");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
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

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeIf, RejectsWrongInputArity) {
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);
  node.add_input("extra"); // Now 2 inputs, which is invalid for If.

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("extra", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
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

  core::shapes::ShapesContext ctx;
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesIf) {
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeIf, RetainsBranchSubgraphContexts) {
  // ComputeShapeIf should retain the child contexts it builds for both
  // branches so the subgraph internals stay inspectable afterwards.
  GraphProto then_b = MakeUnaryBranch("Abs", "x", "y_then");
  GraphProto else_b = MakeUnaryBranch("Abs", "x", "y_else");
  NodeProto node = MakeIfNode("cond", {"y"}, then_b, else_b);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  // current_node_index() is -1 when ComputeShapeIf is called directly.
  onnx_shapes::shapes::controlflow::ComputeShapeIf(ctx, node);

  EXPECT_EQ(ctx.SubgraphContextsSize(), 2u);
  ASSERT_TRUE(ctx.HasSubgraphContext(-1, "then_branch"));
  ASSERT_TRUE(ctx.HasSubgraphContext(-1, "else_branch"));
  EXPECT_FALSE(ctx.HasSubgraphContext(-1, "body"));

  const core::shapes::ShapesContext &then_ctx = ctx.GetSubgraphContext(-1, "then_branch");
  ASSERT_TRUE(then_ctx.Has("y_then"));
  EXPECT_EQ(then_ctx.Get("y_then").Shape(), shape);

  const core::shapes::ShapesContext &else_ctx = ctx.GetSubgraphContext(-1, "else_branch");
  ASSERT_TRUE(else_ctx.Has("y_else"));
  EXPECT_EQ(else_ctx.Get("y_else").Shape(), shape);
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("M", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("v_init", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.Has("v_final"));
  EXPECT_EQ(ctx.Get("v_final").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("v_final").Shape(), shape);

  ASSERT_TRUE(ctx.Has("scan_out"));
  EXPECT_EQ(ctx.Get("scan_out").Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("scan_out").Shape().Rank(), 3u);
  ASSERT_TRUE(ctx.Get("scan_out").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("scan_out").Shape()[0].AsExpr(), "Loop_trip");
  EXPECT_EQ(ctx.Get("scan_out").Shape()[1].AsInt(), 2);
  EXPECT_EQ(ctx.Get("scan_out").Shape()[2].AsInt(), 3);
}

TEST(OnnxOptimShapeLoop, RetainsBodySubgraphContext) {
  GraphProto body = BuildLoopBodyIdentityCarry();
  NodeProto node = MakeLoopNode({"M", "cond", "v_init"}, {"v_final", "scan_out"}, body);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("M", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("v_init", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.HasSubgraphContext(-1, "body"));
  const core::shapes::ShapesContext &body_ctx = ctx.GetSubgraphContext(-1, "body");
  ASSERT_TRUE(body_ctx.Has("v_out"));
  EXPECT_EQ(body_ctx.Get("v_out").Shape(), shape);
}

TEST(OnnxOptimShapeLoop, AcceptsOmittedMAndCond) {
  GraphProto body = BuildLoopBodyIdentityCarry();
  // Both M and cond are omitted using empty input names.
  NodeProto node = MakeLoopNode({"", "", "v_init"}, {"v_final", "scan_out"}, body);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("v_init", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.Has("v_final"));
  EXPECT_EQ(ctx.Get("v_final").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("v_final").Shape(), shape);
  ASSERT_EQ(ctx.Get("scan_out").Shape().Rank(), 2u);
}

TEST(OnnxOptimShapeLoop, UsesTripCountFromValueAsShape) {
  // When M has a ValueAsShape with one element, that element should be
  // used as the leading dimension of scan outputs.
  GraphProto body = BuildLoopBodyIdentityCarry();
  NodeProto node = MakeLoopNode({"M", "cond", "v_init"}, {"v_final", "scan_out"}, body);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  // M is a [1] INT64 tensor with ValueAsShape = [N] (symbolic trip count).
  core::symbolic::SymTensor m_tensor(nullptr, core::symbolic::TensorType::kInt64,
                                     core::symbolic::SymShape{core::symbolic::SymDim(1)});
  core::symbolic::SymShape m_vas;
  m_vas.PushBack(core::symbolic::SymDim("N"));
  m_tensor.SetValueAsShape(std::move(m_vas));
  ctx.Set("M", std::move(m_tensor));
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("v_init", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node);

  ASSERT_TRUE(ctx.Has("scan_out"));
  ASSERT_EQ(ctx.Get("scan_out").Shape().Rank(), 3u);
  // The leading dim should be "N" from M's ValueAsShape, not a generic symbol.
  ASSERT_TRUE(ctx.Get("scan_out").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("scan_out").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("scan_out").Shape()[1].AsInt(), 2);
  EXPECT_EQ(ctx.Get("scan_out").Shape()[2].AsInt(), 3);
}

TEST(OnnxOptimShapeLoop, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLoop");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeLoop, RejectsTooFewInputs) {
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input("M");
  node.add_output("scan");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeLoop, RejectsMissingBodyAttribute) {
  NodeProto node;
  node.set_op_type("Loop");
  node.add_input("M");
  node.add_input("cond");
  node.add_output("scan");
  core::shapes::ShapesContext ctx;
  ctx.Set("M", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeLoop(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesLoop) {
  GraphProto body = BuildLoopBodyIdentityCarry();
  NodeProto node = MakeLoopNode({"M", "cond", "v_init"}, {"v_final", "scan_out"}, body);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(5)};
  ctx.Set("M", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));
  ctx.Set("cond", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("v_init", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));

  onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(5), core::symbolic::SymDim(7)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));

  onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node);

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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(3), core::symbolic::SymDim(2)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeScan, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotScan");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesScan) {
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/1);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(6), core::symbolic::SymDim(2)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(), x_shape);
}

TEST(OnnxOptimShapeScan, RetainsBodySubgraphContext) {
  GraphProto body = BuildScanBodyIdentity();
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/1);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));

  onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node);

  ASSERT_TRUE(ctx.HasSubgraphContext(-1, "body"));
  const core::shapes::ShapesContext &body_ctx = ctx.GetSubgraphContext(-1, "body");
  // The per-iteration scan-input slice drops the trip-count axis: [3].
  ASSERT_TRUE(body_ctx.Has("y_elt"));
  ASSERT_EQ(body_ctx.Get("y_elt").Shape().Rank(), 1u);
  EXPECT_EQ(body_ctx.Get("y_elt").Shape()[0].AsInt(), 3);
}

// Scan opset 8: node inputs are (sequence_lens="", initial, x) where
// ``initial`` has shape [B, D] and ``x`` has shape [B, T, D].  The body
// receives batch-stripped inputs (state=[D], scan=[D]) and the node outputs
// should carry the batch dimension back: state output [B, D], scan output
// [B, T, D].
TEST(OnnxOptimShapeScan, HandlesOpset8BatchDimension) {
  // Build body: sum_in + next → sum_out, scan_out = Identity(sum_out).
  GraphProto body;
  body.set_name("scan8_body");
  body.add_input()->set_name("sum_in");
  body.add_input()->set_name("next");
  NodeProto *add_node = body.add_node();
  add_node->set_op_type("Add");
  add_node->add_input("sum_in");
  add_node->add_input("next");
  add_node->add_output("sum_out");
  NodeProto *id_node = body.add_node();
  id_node->set_op_type("Identity");
  id_node->add_input("sum_out");
  id_node->add_output("scan_out");
  body.add_output()->set_name("sum_out");
  body.add_output()->set_name("scan_out");

  // Node: ("", initial, x) → (y_state, y_scan), num_scan_inputs=1.
  NodeProto node =
      MakeScanNode({"", "initial", "x"}, {"y_state", "y_scan"}, body, /*num_scan_inputs=*/1);

  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 8);

  // initial: [B=1, D=2], x: [B=1, T=3, D=2].
  core::symbolic::SymShape initial_shape{core::symbolic::SymDim(1), core::symbolic::SymDim(2)};
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                   core::symbolic::SymDim(2)};
  ctx.Set("initial",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, initial_shape));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));

  onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node);

  // State output y_state should be [B=1, D=2].
  ASSERT_TRUE(ctx.Has("y_state"));
  ASSERT_EQ(ctx.Get("y_state").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("y_state").Shape()[0].AsInt(), 1);
  EXPECT_EQ(ctx.Get("y_state").Shape()[1].AsInt(), 2);

  // Scan output y_scan should be [B=1, T=3, D=2].
  ASSERT_TRUE(ctx.Has("y_scan"));
  ASSERT_EQ(ctx.Get("y_scan").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("y_scan").Shape()[0].AsInt(), 1);
  EXPECT_EQ(ctx.Get("y_scan").Shape()[1].AsInt(), 3);
  EXPECT_EQ(ctx.Get("y_scan").Shape()[2].AsInt(), 2);
}

// GHSA-qrhj-v62m-vmpf: num_scan_inputs > num_inputs caused a size_t
// underflow in ScanInferenceFunction.  Both paths (opset 9+ and the
// onnx_shapes::shapes::controlflow::ComputeShapeScan path) must raise
// an error rather than silently computing a huge index.
TEST(OnnxOptimShapeScan, RejectsNumScanInputsExceedingNodeInputCount) {
  // A Scan node with 1 input but num_scan_inputs=9. Previously this
  // caused size_t underflow: n_state = 1 - 9 wrapped to ~SIZE_MAX.
  GraphProto body = BuildScanBodyIdentity();
  // num_scan_inputs=9 but the node only has 1 input.
  NodeProto node = MakeScanNode({"X"}, {"Y"}, body, /*num_scan_inputs=*/9);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape x_shape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeScan, RejectsLoopStateVarsExceedingOutputCount) {
  // A Scan node with 3 inputs and num_scan_inputs=1 (so 2 loop-state vars)
  // but only 1 output.  Previously size_t underflow: num_scan_outputs = 1 - 2
  // wrapped to SIZE_MAX.
  GraphProto body;
  body.set_name("body");
  body.add_input()->set_name("s0_in");
  body.add_input()->set_name("s1_in");
  body.add_input()->set_name("x_in");
  NodeProto *id = body.add_node();
  id->set_op_type("Identity");
  id->add_input("s0_in");
  id->add_output("s0_out");
  body.add_output()->set_name("s0_out");

  // 3 inputs (s0, s1, x), num_scan_inputs=1 → 2 loop-state vars.
  // Only 1 output declared on the node → invalid.
  NodeProto node = MakeScanNode({"s0", "s1", "x"}, {"out"}, body, /*num_scan_inputs=*/1);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  ctx.Set("s0", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("s1", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  EXPECT_THROW(onnx_shapes::shapes::controlflow::ComputeShapeScan(ctx, node),
               std::invalid_argument);
}

} // namespace Test
