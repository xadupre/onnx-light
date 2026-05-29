// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeConstantNode() {
  NodeProto node;
  node.set_op_type("Constant");
  node.add_output("y");
  return node;
}

AttributeProto *AddAttr(NodeProto &node, const std::string &name,
                        AttributeProto::AttributeType type) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(type);
  return attr;
}

} // namespace

TEST(OnnxOptimShapeConstant, ValueTensorFloat) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::FLOAT));
  t->add_dims(2);
  t->add_dims(3);
  // Float content is irrelevant for shape inference but provide raw_data
  // so the proto is well-formed.
  const std::vector<uint8_t> raw(2 * 3 * sizeof(float), 0);
  t->set_raw_data(utils::ByteSpan(raw));

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  // Float, rank 2: no ValueAsShape annotation.
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueTensorSmallInt64SetsValueAsShape) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  t->add_dims(3);
  t->ref_int64_data().push_back(4);
  t->ref_int64_data().push_back(8);
  t->ref_int64_data().push_back(16);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(8),
                                    onnx_optim::OptimDim(16)}));
}

TEST(OnnxOptimShapeConstant, ValueTensorLargeInt64DoesNotSetValueAsShape) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  t->add_dims(8);
  for (int64_t i = 0; i < 8; ++i) {
    t->ref_int64_data().push_back(i);
  }

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  // 8 elements is not "small" (the threshold is strictly less than 8).
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueTensorInt64FromRawData) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  t->add_dims(2);
  const int64_t vals[2] = {5, 7};
  std::vector<uint8_t> raw(sizeof(vals));
  std::memcpy(raw.data(), vals, sizeof(vals));
  t->set_raw_data(utils::ByteSpan(raw));

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(7)}));
}

TEST(OnnxOptimShapeConstant, ValueInt) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_int", AttributeProto::AttributeType::INT);
  attr->set_i(42);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), onnx_optim::OptimShape{});
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(42)}));
}

TEST(OnnxOptimShapeConstant, ValueInts) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_ints", AttributeProto::AttributeType::INTS);
  attr->ref_ints().push_back(1);
  attr->ref_ints().push_back(2);
  attr->ref_ints().push_back(3);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                    onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeConstant, ValueFloat) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_float", AttributeProto::AttributeType::FLOAT);
  attr->set_f(1.42f);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), onnx_optim::OptimShape{});
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueFloats) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_floats", AttributeProto::AttributeType::FLOATS);
  attr->ref_floats().push_back(1.0f);
  attr->ref_floats().push_back(1.1f);
  attr->ref_floats().push_back(1.2f);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueString) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_string", AttributeProto::AttributeType::STRING);
  attr->set_s("hello");

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("y").Shape(), onnx_optim::OptimShape{});
}

TEST(OnnxOptimShapeConstant, ValueStrings) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_strings", AttributeProto::AttributeType::STRINGS);
  attr->ref_strings().push_back(utils::String("o"));
  attr->ref_strings().push_back(utils::String("n"));
  attr->ref_strings().push_back(utils::String("n"));
  attr->ref_strings().push_back(utils::String("x"));

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapeConstant, SparseValue) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr =
      AddAttr(node, "sparse_value", AttributeProto::AttributeType::SPARSE_TENSOR);
  SparseTensorProto *sp = attr->add_sparse_tensor();
  TensorProto &vals = sp->ref_values();
  vals.set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  sp->ref_dims().push_back(100);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(100)}));
}

TEST(OnnxOptimShapeConstant, RejectsNodeWithoutAnyValueAttribute) {
  NodeProto node = MakeConstantNode();
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeConstant, RejectsNodeWithTwoValueAttributes) {
  NodeProto node = MakeConstantNode();
  AddAttr(node, "value_int", AttributeProto::AttributeType::INT)->set_i(1);
  AddAttr(node, "value_float", AttributeProto::AttributeType::FLOAT)->set_f(1.0f);
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeConstant, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotConstant");
  node.add_output("y");
  AddAttr(node, "value_int", AttributeProto::AttributeType::INT)->set_i(1);
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeConstant(ctx, node),
               std::invalid_argument);
}

namespace {

NodeProto MakeConstantOfShapeNode() {
  NodeProto node;
  node.set_op_type("ConstantOfShape");
  node.add_input("x");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeConstantOfShape, UsesShapeInputValueAndDefaultsFloatDtype) {
  NodeProto node = MakeConstantOfShapeNode();

  onnx_optim::shapes::ShapesContext ctx;
  // Input ``x`` is a 1-D INT64 tensor of static shape [3] whose
  // ValueAsShape annotation holds the concrete dims (4, 8, 16).
  onnx_optim::OptimTensor x(nullptr, onnx_optim::TensorType::kInt64,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  x.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(8),
                                           onnx_optim::OptimDim(16)});
  ctx.Set("x", std::move(x));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(8),
                                    onnx_optim::OptimDim(16)}));
}

TEST(OnnxOptimShapeConstantOfShape, ValueAttributeOverridesOutputDtype) {
  NodeProto node = MakeConstantOfShapeNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT32));
  t->add_dims(1);
  t->ref_int32_data().push_back(0);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimTensor x(nullptr, onnx_optim::TensorType::kInt64,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  x.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(10), onnx_optim::OptimDim(6)});
  ctx.Set("x", std::move(x));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(10), onnx_optim::OptimDim(6)}));
}

TEST(OnnxOptimShapeConstantOfShape, EmptyShapeProducesScalar) {
  NodeProto node = MakeConstantOfShapeNode();

  onnx_optim::shapes::ShapesContext ctx;
  // Input is a length-0 1-D INT64 tensor -> output is a scalar.
  onnx_optim::OptimTensor x(nullptr, onnx_optim::TensorType::kInt64,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(int64_t{0})});
  x.SetValueAsShape(onnx_optim::OptimShape{});
  ctx.Set("x", std::move(x));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), onnx_optim::OptimShape{});
}

TEST(OnnxOptimShapeConstantOfShape, FallsBackToSymbolicDimsWithoutValueAsShape) {
  NodeProto node = MakeConstantOfShapeNode();

  onnx_optim::shapes::ShapesContext ctx;
  // No ValueAsShape annotation -> output rank derived from the static
  // single dim ``3`` of ``x``.
  onnx_optim::OptimTensor x(nullptr, onnx_optim::TensorType::kInt64,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  ctx.Set("x", std::move(x));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(ctx.Get("y").Shape()[i].IsInt());
  }
}

TEST(OnnxOptimShapeConstantOfShape, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotConstantOfShape");
  node.add_input("x");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimTensor x(nullptr, onnx_optim::TensorType::kInt64,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  x.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeConstantOfShape(ctx, node),
               std::invalid_argument);
}

} // namespace Test
