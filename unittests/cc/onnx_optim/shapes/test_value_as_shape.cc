// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// C++ translation of
// https://github.com/xadupre/yet-another-onnx-builder/blob/main/unittests/xshape/test_value_as_shape.py
//
// Exercises ``ComputeShapeModel`` on a small graph that mixes ``Shape``,
// ``Concat``, ``Add``, ``Sub`` and ``Expand`` and checks that the
// ``ValueAsShape`` annotation is correctly propagated through those
// operators so that the downstream ``Expand`` recovers a precise shape.

#include "onnx_optim/shapes/shape_inference.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shapes_context.h"
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

// Adds a float tensor value-info to ``graph.input()`` with the given
// shape: negative entries become symbolic dim_params taken from
// ``symbolic_names``, non-negative entries become concrete dim_values.
void AddFloatInput(GraphProto &graph, const std::string &name, const std::vector<int64_t> &shape,
                   const std::vector<std::string> &symbolic_names) {
  ValueInfoProto *vi = graph.add_input();
  vi->set_name(name);
  TypeProto *tp = vi->add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    TensorShapeProto::Dimension *d = sp->add_dim();
    if (shape[i] < 0) {
      d->set_dim_param(i < symbolic_names.size() ? symbolic_names[i] : std::string("?"));
    } else {
      d->set_dim_value(shape[i]);
    }
  }
}

void AddFloatOutput(GraphProto &graph, const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<std::string> &symbolic_names) {
  ValueInfoProto *vi = graph.add_output();
  vi->set_name(name);
  TypeProto *tp = vi->add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    TensorShapeProto::Dimension *d = sp->add_dim();
    if (shape[i] < 0) {
      d->set_dim_param(i < symbolic_names.size() ? symbolic_names[i] : std::string("?"));
    } else {
      d->set_dim_value(shape[i]);
    }
  }
}

// Builds the model from ``test_value_as_shape.py``:
//
//   x: float[N, 1], y1/y2/y3: float[1, B], initializer one: int64[1] = [1]
//   n        = Shape(x, start=0, end=1)
//   b        = Shape(x, start=1, end=2)
//   shape    = Concat([n, b], axis=0)
//   shape1   = Add(shape, one)
//   shape2   = Sub(shape1, one)
//   expanded = Expand(x, shape2)
//   z1/z2/z3 = Add(expanded, y{1,2,3})
//   z12 = Add(z1, z2); z = Add(z12, z3)
ModelProto MakeValueAsShapeModel() {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(20);

  GraphProto *graph = model.add_graph();
  graph->set_name("test");

  AddFloatInput(*graph, "x", {-1, 1}, {"N"});
  AddFloatInput(*graph, "y1", {1, -1}, {"", "B"});
  AddFloatInput(*graph, "y2", {1, -1}, {"", "B"});
  AddFloatInput(*graph, "y3", {1, -1}, {"", "B"});

  AddFloatOutput(*graph, "z", {-1, -1}, {"N", "B"});

  // initializer ``one`` : int64[1] = [1]
  TensorProto *init = graph->add_initializer();
  init->set_name("one");
  init->set_data_type(static_cast<int>(TensorProto::DataType::INT64));
  init->add_dims(std::vector<uint64_t>{1});
  init->add_int64_data(std::vector<int64_t>{1});

  // n = Shape(x, start=0, end=1)
  NodeProto n_node = MakeNode("Shape", {"x"}, {"n"});
  AddIntAttribute(n_node, "start", 0);
  AddIntAttribute(n_node, "end", 1);
  *graph->add_node() = std::move(n_node);

  // b = Shape(x, start=1, end=2)
  NodeProto b_node = MakeNode("Shape", {"x"}, {"b"});
  AddIntAttribute(b_node, "start", 1);
  AddIntAttribute(b_node, "end", 2);
  *graph->add_node() = std::move(b_node);

  // shape = Concat([n, b], axis=0)
  NodeProto concat_node = MakeNode("Concat", {"n", "b"}, {"shape"});
  AddIntAttribute(concat_node, "axis", 0);
  *graph->add_node() = std::move(concat_node);

  *graph->add_node() = MakeNode("Add", {"shape", "one"}, {"shape1"});
  *graph->add_node() = MakeNode("Sub", {"shape1", "one"}, {"shape2"});
  *graph->add_node() = MakeNode("Expand", {"x", "shape2"}, {"expanded"});

  *graph->add_node() = MakeNode("Add", {"expanded", "y1"}, {"z1"});
  *graph->add_node() = MakeNode("Add", {"expanded", "y2"}, {"z2"});
  *graph->add_node() = MakeNode("Add", {"expanded", "y3"}, {"z3"});

  *graph->add_node() = MakeNode("Add", {"z1", "z2"}, {"z12"});
  *graph->add_node() = MakeNode("Add", {"z12", "z3"}, {"z"});

  return model;
}

// Drops ``+0`` / ``0+`` / leading ``0-`` noise that ``dim_add`` /
// ``dim_sub`` may produce when one operand simplifies away. Used only
// to make the ``ValueAsShape`` assertions robust to harmless rewrites.
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

bool DimEqualsInt(const onnx_optim::OptimDim &d, int64_t v) { return d.IsInt() && d.AsInt() == v; }

bool DimEqualsExpr(const onnx_optim::OptimDim &d, const std::string &s) {
  return d.IsExpr() && Canonicalise(d.AsExpr()) == Canonicalise(s);
}

} // namespace

TEST(OnnxOptimShapeInference, ValueAsShapePropagatesThroughShapeConcatAddSubExpand) {
  ModelProto model = MakeValueAsShapeModel();

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // ``shape`` (= Concat([Shape(x, 0, 1), Shape(x, 1, 2)])) carries the
  // value-as-shape ``(N, 1)`` lifted from ``x``'s symbolic shape.
  ASSERT_TRUE(ctx.Has("shape"));
  ASSERT_TRUE(ctx.Get("shape").HasValueAsShape());
  const onnx_optim::OptimShape &v_shape = ctx.Get("shape").ValueAsShape();
  ASSERT_EQ(v_shape.Rank(), 2u);
  EXPECT_TRUE(DimEqualsExpr(v_shape[0], "N"));
  EXPECT_TRUE(DimEqualsInt(v_shape[1], 1));

  // ``shape1 = shape + [1]`` broadcasts to value-as-shape ``(N+1, 2)``.
  ASSERT_TRUE(ctx.Has("shape1"));
  ASSERT_TRUE(ctx.Get("shape1").HasValueAsShape());
  const onnx_optim::OptimShape &v_shape1 = ctx.Get("shape1").ValueAsShape();
  ASSERT_EQ(v_shape1.Rank(), 2u);
  // ``dim_add(N, 1)`` is a symbolic expression containing both ``N``
  // and ``1`` (the exact textual ordering is canonicalised by the
  // expressions library).
  ASSERT_TRUE(v_shape1[0].IsExpr());
  EXPECT_NE(v_shape1[0].AsExpr().find("N"), std::string::npos);
  EXPECT_NE(v_shape1[0].AsExpr().find("1"), std::string::npos);
  EXPECT_TRUE(DimEqualsInt(v_shape1[1], 2));

  // ``shape2 = shape1 - [1]`` simplifies to value-as-shape ``(N, 1)``.
  ASSERT_TRUE(ctx.Has("shape2"));
  ASSERT_TRUE(ctx.Get("shape2").HasValueAsShape());
  const onnx_optim::OptimShape &v_shape2 = ctx.Get("shape2").ValueAsShape();
  ASSERT_EQ(v_shape2.Rank(), 2u);
  EXPECT_TRUE(DimEqualsExpr(v_shape2[0], "N"));
  EXPECT_TRUE(DimEqualsInt(v_shape2[1], 1));

  // ``Expand(x, shape2)`` picks up ``ValueAsShape`` of shape2 and
  // produces the precise output shape ``(N, 1)``.
  ASSERT_TRUE(ctx.Has("expanded"));
  const onnx_optim::OptimShape &expanded = ctx.Get("expanded").Shape();
  ASSERT_EQ(expanded.Rank(), 2u);
  EXPECT_TRUE(DimEqualsExpr(expanded[0], "N"));
  EXPECT_TRUE(DimEqualsInt(expanded[1], 1));

  // ``z1 = expanded + y1`` broadcasts (N, 1) with (1, B) to (N, B).
  ASSERT_TRUE(ctx.Has("z1"));
  const onnx_optim::OptimShape &z1 = ctx.Get("z1").Shape();
  ASSERT_EQ(z1.Rank(), 2u);
  EXPECT_TRUE(DimEqualsExpr(z1[0], "N"));
  EXPECT_TRUE(DimEqualsExpr(z1[1], "B"));
}

} // namespace Test
