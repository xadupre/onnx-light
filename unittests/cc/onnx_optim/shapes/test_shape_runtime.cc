// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// C++ translation of
// https://github.com/xadupre/yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_runtime.py
//
// The Python tests exercise the value-shape runtime of ``BasicShapeBuilder``
// (a Python class that tracks both an ordinary shape and a free-form
// ``value_as_shape`` per tensor, where the latter may be a Python int, a
// string expression or a tuple). The C++ shape inference in onnx-light
// represents the same idea via :cpp:func:`SymTensor::ValueAsShape`, but it
// is more strictly typed: the value-as-shape is always an ``SymShape``
// (a 1-D vector of ``SymDim``), and only a subset of operators currently
// propagate it.
//
// This file translates the Python tests that have a direct counterpart in
// the C++ runtime:
//   * ``Shape`` with and without ``start``/``end`` attributes.
//   * ``Concat`` along axis 0 of 1-D tensors that carry a ``ValueAsShape``.
//   * ``Add`` / ``Sub`` broadcast value-as-shape arithmetic.
//
// Tests that exercise operators which do not yet propagate ``ValueAsShape``
// in the C++ runtime (``Identity``, ``Abs``, ``Gather``, ``Squeeze``,
// ``Unsqueeze``, ``Range``, ``Slice``, ``Mul``, ``Div``, ``Mod``) are not
// translated to avoid asserting on unimplemented behaviour.

#include "onnx_core/shapes/shape_inference.h"

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type(op_type);
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  return node;
}

void AddIntAttribute(NodeProto &node, const std::string &name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

// Helper to build an int64 1-D tensor with a value-as-shape annotation.
// Mirrors the Python ``_int64_cst`` + ``_known_value_shape[name] = ...``
// pattern.
core::symbolic::SymTensor MakeInt64ValueAsShape(core::symbolic::SymShape value) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(static_cast<int64_t>(value.Rank())));
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64, std::move(shape));
  t.SetValueAsShape(std::move(value));
  return t;
}

bool DimEqualsInt(const core::symbolic::SymDim &d, int64_t v) {
  return d.IsInt() && d.AsInt() == v;
}

// Strip whitespace so that assertions on symbolic expressions are robust
// against harmless re-formattings by the expressions library.
std::string Canonicalise(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c != ' ') {
      out.push_back(c);
    }
  }
  return out;
}

bool DimEqualsExpr(const core::symbolic::SymDim &d, const std::string &s) {
  return d.IsExpr() && Canonicalise(d.AsExpr()) == Canonicalise(s);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Shape  (translates ``test_shape_no_attrs_known_shape`` &c.)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OnnxOptimShapeRuntime, ShapeNoAttrsKnownShape) {
  // Python: Shape(X) on a (2,3,4) input gives value_as_shape == (2,3,4)
  // and Y has shape (3,).
  NodeProto node = MakeNode("Shape", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                 core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &out = ctx.Get("Y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(out.Shape().Rank(), 1u);
  EXPECT_TRUE(DimEqualsInt(out.Shape()[0], 3));
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 3u);
  EXPECT_TRUE(DimEqualsInt(v[0], 2));
  EXPECT_TRUE(DimEqualsInt(v[1], 3));
  EXPECT_TRUE(DimEqualsInt(v[2], 4));
}

TEST(OnnxOptimShapeRuntime, ShapeNoAttrsUnknownShapeIsEmpty) {
  // Python: Shape(X) on an unknown-shape input returns False (no value
  // shape stored). In C++ an unregistered input has rank 0 so ``Shape``
  // produces a 1-D INT64 tensor with shape (0,) and an empty ValueAsShape.
  NodeProto node = MakeNode("Shape", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &out = ctx.Get("Y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(out.Shape().Rank(), 1u);
  EXPECT_TRUE(DimEqualsInt(out.Shape()[0], 0));
  ASSERT_TRUE(out.HasValueAsShape());
  EXPECT_EQ(out.ValueAsShape().Rank(), 0u);
}

TEST(OnnxOptimShapeRuntime, ShapeWithStartAttr) {
  // Python: Shape(X, start=2) on (2,3,4,5) -> value_as_shape == (4,5).
  NodeProto node = MakeNode("Shape", {"X"}, {"Y"});
  AddIntAttribute(node, "start", 2);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                 core::symbolic::SymDim(4), core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &out = ctx.Get("Y");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 2u);
  EXPECT_TRUE(DimEqualsInt(v[0], 4));
  EXPECT_TRUE(DimEqualsInt(v[1], 5));
}

TEST(OnnxOptimShapeRuntime, ShapeWithStartEndAttrs) {
  // Python: Shape(X, start=1, end=3) on (2,3,4,5) -> value_as_shape == (3,4).
  NodeProto node = MakeNode("Shape", {"X"}, {"Y"});
  AddIntAttribute(node, "start", 1);
  AddIntAttribute(node, "end", 3);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                 core::symbolic::SymDim(4), core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &out = ctx.Get("Y");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 2u);
  EXPECT_TRUE(DimEqualsInt(v[0], 3));
  EXPECT_TRUE(DimEqualsInt(v[1], 4));
}

// ─────────────────────────────────────────────────────────────────────────────
// Concat  (translates ``test_concat_two_shape_tuples``)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OnnxOptimShapeRuntime, ConcatTwoShapeTuples) {
  // Python: Concat([(2,3), (4,5)], axis=0) -> value_as_shape == (2,3,4,5).
  NodeProto node = MakeNode("Concat", {"a", "b"}, {"out"});
  AddIntAttribute(node, "axis", 0);

  core::shapes::ShapesContext ctx;
  ctx.Set("a", MakeInt64ValueAsShape({core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("b", MakeInt64ValueAsShape({core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 4u);
  EXPECT_TRUE(DimEqualsInt(v[0], 2));
  EXPECT_TRUE(DimEqualsInt(v[1], 3));
  EXPECT_TRUE(DimEqualsInt(v[2], 4));
  EXPECT_TRUE(DimEqualsInt(v[3], 5));
}

// ─────────────────────────────────────────────────────────────────────────────
// Add / Sub  (translate ``test_add_tuples_element_wise``, ``test_sub_two_ints``,
// ``test_add_broadcast_scalar_to_tuple``, ``test_add_str_and_int``)
// ─────────────────────────────────────────────────────────────────────────────

TEST(OnnxOptimShapeRuntime, AddTuplesElementWise) {
  // Python: Add([(2,3), (1,4)]) -> (3, 7)
  NodeProto node = MakeNode("Add", {"a", "b"}, {"out"});

  core::shapes::ShapesContext ctx;
  ctx.Set("a", MakeInt64ValueAsShape({core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("b", MakeInt64ValueAsShape({core::symbolic::SymDim(1), core::symbolic::SymDim(4)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 2u);
  EXPECT_TRUE(DimEqualsInt(v[0], 3));
  EXPECT_TRUE(DimEqualsInt(v[1], 7));
}

TEST(OnnxOptimShapeRuntime, SubTuplesElementWise) {
  // Python: Sub([10, 3]) -> 7; here translated to 1-D Sub([(10,), (3,)]) -> (7,).
  NodeProto node = MakeNode("Sub", {"a", "b"}, {"out"});

  core::shapes::ShapesContext ctx;
  ctx.Set("a", MakeInt64ValueAsShape({core::symbolic::SymDim(10)}));
  ctx.Set("b", MakeInt64ValueAsShape({core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 1u);
  EXPECT_TRUE(DimEqualsInt(v[0], 7));
}

TEST(OnnxOptimShapeRuntime, AddBroadcastScalarToTuple) {
  // Python: Add([1, (2,3,4)]) -> (3,4,5). Here translated to a numpy-broadcast
  // VAS Add of a length-1 vector with a length-3 vector.
  NodeProto node = MakeNode("Add", {"a", "b"}, {"out"});

  core::shapes::ShapesContext ctx;
  ctx.Set("a", MakeInt64ValueAsShape({core::symbolic::SymDim(1)}));
  ctx.Set("b", MakeInt64ValueAsShape({core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                      core::symbolic::SymDim(4)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 3u);
  EXPECT_TRUE(DimEqualsInt(v[0], 3));
  EXPECT_TRUE(DimEqualsInt(v[1], 4));
  EXPECT_TRUE(DimEqualsInt(v[2], 5));
}

TEST(OnnxOptimShapeRuntime, AddStrAndInt) {
  // Python: Add(["batch", 1]) -> "batch+1". In C++ the result of adding a
  // symbolic dim and a concrete integer is a symbolic expression that
  // mentions both operands.
  NodeProto node = MakeNode("Add", {"a", "b"}, {"out"});

  core::shapes::ShapesContext ctx;
  ctx.Set("a", MakeInt64ValueAsShape({core::symbolic::SymDim("batch")}));
  ctx.Set("b", MakeInt64ValueAsShape({core::symbolic::SymDim(static_cast<int64_t>(1))}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("out"));
  const core::symbolic::SymTensor &out = ctx.Get("out");
  ASSERT_TRUE(out.HasValueAsShape());
  const core::symbolic::SymShape &v = out.ValueAsShape();
  ASSERT_EQ(v.Rank(), 1u);
  ASSERT_TRUE(v[0].IsExpr());
  EXPECT_NE(v[0].AsExpr().find("batch"), std::string::npos);
  EXPECT_NE(v[0].AsExpr().find("1"), std::string::npos);
}

} // namespace Test
