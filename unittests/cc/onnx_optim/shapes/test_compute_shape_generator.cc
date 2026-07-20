// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
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

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
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

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(8),
                                      core::symbolic::SymDim(16)}));
}

TEST(OnnxOptimShapeConstant, ValueTensorLargeInt64DoesNotSetValueAsShape) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  t->add_dims(28);
  for (int64_t i = 0; i < 28; ++i) {
    t->ref_int64_data().push_back(i);
  }

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  // 8 elements is not "small" (the threshold is strictly less than 28).
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

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(5), core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapeConstant, ValueInt) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_int", AttributeProto::AttributeType::INT);
  attr->set_i(42);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), core::symbolic::SymShape{});
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(), (core::symbolic::SymShape{core::symbolic::SymDim(42)}));
}

TEST(OnnxOptimShapeConstant, ValueInts) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_ints", AttributeProto::AttributeType::INTS);
  attr->ref_ints().push_back(1);
  attr->ref_ints().push_back(2);
  attr->ref_ints().push_back(3);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ASSERT_TRUE(ctx.Get("y").HasValueAsShape());
  EXPECT_EQ(ctx.Get("y").ValueAsShape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                      core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeConstant, ValueFloat) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_float", AttributeProto::AttributeType::FLOAT);
  attr->set_f(1.42f);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), core::symbolic::SymShape{});
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueFloats) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_floats", AttributeProto::AttributeType::FLOATS);
  attr->ref_floats().push_back(1.0f);
  attr->ref_floats().push_back(1.1f);
  attr->ref_floats().push_back(1.2f);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_FALSE(ctx.Get("y").HasValueAsShape());
}

TEST(OnnxOptimShapeConstant, ValueString) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_string", AttributeProto::AttributeType::STRING);
  attr->set_s("hello");

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("y").Shape(), core::symbolic::SymShape{});
}

TEST(OnnxOptimShapeConstant, ValueStrings) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr = AddAttr(node, "value_strings", AttributeProto::AttributeType::STRINGS);
  attr->ref_strings().push_back(utils::String("o"));
  attr->ref_strings().push_back(utils::String("n"));
  attr->ref_strings().push_back(utils::String("n"));
  attr->ref_strings().push_back(utils::String("x"));

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeConstant, SparseValue) {
  NodeProto node = MakeConstantNode();
  AttributeProto *attr =
      AddAttr(node, "sparse_value", AttributeProto::AttributeType::SPARSE_TENSOR);
  SparseTensorProto *sp = attr->add_sparse_tensor();
  TensorProto &vals = sp->ref_values();
  vals.set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT64));
  sp->ref_dims().push_back(100);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(100)}));
}

TEST(OnnxOptimShapeConstant, RejectsNodeWithoutAnyValueAttribute) {
  NodeProto node = MakeConstantNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeConstant, RejectsNodeWithTwoValueAttributes) {
  NodeProto node = MakeConstantNode();
  AddAttr(node, "value_int", AttributeProto::AttributeType::INT)->set_i(1);
  AddAttr(node, "value_float", AttributeProto::AttributeType::FLOAT)->set_f(1.0f);
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeConstant, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotConstant");
  node.add_output("y");
  AddAttr(node, "value_int", AttributeProto::AttributeType::INT)->set_i(1);
  core::shapes::ShapesContext ctx;
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

  core::shapes::ShapesContext ctx;
  // Input ``x`` is a 1-D INT64 tensor of static shape [3] whose
  // ValueAsShape annotation holds the concrete dims (4, 8, 16).
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  x.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(8),
                                             core::symbolic::SymDim(16)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(8),
                                      core::symbolic::SymDim(16)}));
}

TEST(OnnxOptimShapeConstantOfShape, ValueAttributeOverridesOutputDtype) {
  NodeProto node = MakeConstantOfShapeNode();
  AttributeProto *attr = AddAttr(node, "value", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::INT32));
  t->add_dims(1);
  t->ref_int32_data().push_back(0);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(2)});
  x.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(10), core::symbolic::SymDim(6)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(10), core::symbolic::SymDim(6)}));
}

TEST(OnnxOptimShapeConstantOfShape, EmptyShapeProducesScalar) {
  NodeProto node = MakeConstantOfShapeNode();

  core::shapes::ShapesContext ctx;
  // Input is a length-0 1-D INT64 tensor -> output is a scalar.
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(int64_t{0})});
  x.SetValueAsShape(core::symbolic::SymShape{});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), core::symbolic::SymShape{});
}

TEST(OnnxOptimShapeConstantOfShape, FallsBackToSymbolicDimsWithoutValueAsShape) {
  NodeProto node = MakeConstantOfShapeNode();

  core::shapes::ShapesContext ctx;
  // No ValueAsShape annotation -> output rank derived from the static
  // single dim ``3`` of ``x``.
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  x.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeConstantOfShape(ctx, node),
               std::invalid_argument);
}

namespace {

NodeProto MakeEyeLikeNode() {
  NodeProto node;
  node.set_op_type("EyeLike");
  node.add_input("x");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeEyeLike, DefaultsToInputTypeAndShape) {
  NodeProto node = MakeEyeLikeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kInt32,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeEyeLike, DtypeAttributeOverridesOutputType) {
  NodeProto node = MakeEyeLikeNode();
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::INT64));
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapeEyeLike, RejectsNonMatrixInput) {
  NodeProto node = MakeEyeLikeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                       core::symbolic::SymDim(3),
                                                       core::symbolic::SymDim(4)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

namespace {

NodeProto MakeBlackmanWindowNode() {
  NodeProto node;
  node.set_op_type("BlackmanWindow");
  node.add_input("size");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeBlackmanWindow, KnownSizeProducesConcreteDimFloatByDefault) {
  NodeProto node = MakeBlackmanWindowNode();

  core::shapes::ShapesContext ctx;
  // ``size`` is a scalar INT64 whose ValueAsShape annotation holds the
  // concrete value 10.
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(10)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(10)}));
}

TEST(OnnxOptimShapeBlackmanWindow, OutputDatatypeAttributeOverridesDtype) {
  NodeProto node = MakeBlackmanWindowNode();
  AddAttr(node, "output_datatype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::DOUBLE));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(32)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(32)}));
}

TEST(OnnxOptimShapeBlackmanWindow, UnknownSizeFallsBackToSymbolicDim) {
  NodeProto node = MakeBlackmanWindowNode();

  core::shapes::ShapesContext ctx;
  // No ValueAsShape annotation -> output is a 1-D tensor with a single
  // symbolic dim.
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
}

TEST(OnnxOptimShapeBlackmanWindow, PeriodicAttributeDoesNotAffectShape) {
  NodeProto node = MakeBlackmanWindowNode();
  AddAttr(node, "periodic", AttributeProto::AttributeType::INT)->set_i(0);

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt32,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(8)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(8)}));
}

TEST(OnnxOptimShapeBlackmanWindow, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotBlackmanWindow");
  node.add_input("size");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("size", std::move(s));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeBlackmanWindow(ctx, node),
               std::invalid_argument);
}

namespace {

NodeProto MakeHannWindowNode() {
  NodeProto node;
  node.set_op_type("HannWindow");
  node.add_input("size");
  node.add_output("y");
  return node;
}

NodeProto MakeHammingWindowNode() {
  NodeProto node;
  node.set_op_type("HammingWindow");
  node.add_input("size");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeHannWindow, KnownSizeProducesConcreteDimFloatByDefault) {
  NodeProto node = MakeHannWindowNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(10)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(10)}));
}

TEST(OnnxOptimShapeHannWindow, OutputDatatypeAttributeOverridesDtype) {
  NodeProto node = MakeHannWindowNode();
  AddAttr(node, "output_datatype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::DOUBLE));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(32)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(32)}));
}

TEST(OnnxOptimShapeHannWindow, UnknownSizeFallsBackToSymbolicDim) {
  NodeProto node = MakeHannWindowNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
}

TEST(OnnxOptimShapeHannWindow, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotHannWindow");
  node.add_input("size");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("size", std::move(s));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeHannWindow(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeHammingWindow, KnownSizeProducesConcreteDimFloatByDefault) {
  NodeProto node = MakeHammingWindowNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(10)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(10)}));
}

TEST(OnnxOptimShapeHammingWindow, OutputDatatypeAttributeOverridesDtype) {
  NodeProto node = MakeHammingWindowNode();
  AddAttr(node, "output_datatype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::DOUBLE));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(32)});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(32)}));
}

TEST(OnnxOptimShapeHammingWindow, UnknownSizeFallsBackToSymbolicDim) {
  NodeProto node = MakeHammingWindowNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  ctx.Set("size", std::move(s));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
}

TEST(OnnxOptimShapeHammingWindow, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotHammingWindow");
  node.add_input("size");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor s(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  s.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("size", std::move(s));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeHammingWindow(ctx, node),
               std::invalid_argument);
}

namespace {

NodeProto MakeBernoulliNode() {
  NodeProto node;
  node.set_op_type("Bernoulli");
  node.add_input("x");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeBernoulli, KeepsInputShapeAndDtypeByDefault) {
  NodeProto node = MakeBernoulliNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeBernoulli, DtypeAttributeOverridesOutputDtype) {
  NodeProto node = MakeBernoulliNode();
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::INT64));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kDouble,
                              core::symbolic::SymShape{core::symbolic::SymDim(5)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapeBernoulli, PreservesSymbolicDims) {
  NodeProto node = MakeBernoulliNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat16,
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(7)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat16);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_TRUE(ctx.Get("y").Shape()[1].IsInt());
}

TEST(OnnxOptimShapeBernoulli, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotBernoulli");
  node.add_input("x");
  node.add_output("y");

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeBernoulli(ctx, node),
               std::invalid_argument);
}

namespace {

NodeProto MakeMultinomialNode() {
  NodeProto node;
  node.set_op_type("Multinomial");
  node.add_input("x");
  node.add_output("y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeMultinomial, DefaultProducesInt32BatchBySampleSizeOne) {
  NodeProto node = MakeMultinomialNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(5)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(1)}));
}

TEST(OnnxOptimShapeMultinomial, SampleSizeAndDtypeAttributesAreApplied) {
  NodeProto node = MakeMultinomialNode();
  AddAttr(node, "sample_size", AttributeProto::AttributeType::INT)->set_i(7);
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::INT64));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kDouble,
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)});
  ctx.Set("x", std::move(x));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_TRUE(ctx.Get("y").Shape()[1].IsInt());
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsInt(), 7);
}

TEST(OnnxOptimShapeMultinomial, RejectsUnsupportedDtype) {
  NodeProto node = MakeMultinomialNode();
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::FLOAT));

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeMultinomial(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeMultinomial, RejectsNon2DInput) {
  NodeProto node = MakeMultinomialNode();

  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(4)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

namespace {

NodeProto MakeRandomNode(const std::string &op_type, const std::vector<int64_t> &shape) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_output("y");
  AttributeProto *attr = AddAttr(node, "shape", AttributeProto::AttributeType::INTS);
  for (int64_t v : shape) {
    attr->ref_ints().push_back(v);
  }
  return node;
}

NodeProto MakeRandomLikeNode(const std::string &op_type) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_output("y");
  return node;
}

NodeProto MakeRangeNode() {
  NodeProto node;
  node.set_op_type("Range");
  node.add_input("start");
  node.add_input("limit");
  node.add_input("delta");
  node.add_output("y");
  return node;
}

core::symbolic::SymTensor MakeScalar(core::symbolic::TensorType dtype) {
  return core::symbolic::SymTensor(nullptr, dtype, core::symbolic::SymShape{});
}

} // namespace

TEST(OnnxOptimShapeRandomNormal, UsesShapeAttributeAndDefaultsToFloat) {
  NodeProto node = MakeRandomNode("RandomNormal", {2, 3});
  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeRandomNormal, DtypeAttributeOverridesOutputDtype) {
  NodeProto node = MakeRandomNode("RandomNormal", {4});
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::DOUBLE));
  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeRandomNormal, MissingShapeAttributeThrows) {
  NodeProto node;
  node.set_op_type("RandomNormal");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeRandomNormal(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeRandomUniform, UsesShapeAttributeAndDefaultsToFloat) {
  NodeProto node = MakeRandomNode("RandomUniform", {5});
  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapeRandomNormalLike, CopiesInputShapeAndDtypeByDefault) {
  NodeProto node = MakeRandomLikeNode("RandomNormalLike");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat16,
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(8)});
  ctx.Set("x", std::move(x));
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat16);
  EXPECT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
  EXPECT_TRUE(ctx.Get("y").Shape()[1].IsInt());
}

TEST(OnnxOptimShapeRandomNormalLike, DtypeAttributeOverridesOutputDtype) {
  NodeProto node = MakeRandomLikeNode("RandomNormalLike");
  AddAttr(node, "dtype", AttributeProto::AttributeType::INT)
      ->set_i(static_cast<int64_t>(TensorProto::DOUBLE));
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  ctx.Set("x", std::move(x));
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeRandomUniformLike, CopiesInputShapeAndDtypeByDefault) {
  NodeProto node = MakeRandomLikeNode("RandomUniformLike");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)});
  ctx.Set("x", std::move(x));
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeRandomUniformLike, RejectsBadOpType) {
  NodeProto node = MakeRandomLikeNode("NotRandomUniformLike");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor x(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  ctx.Set("x", std::move(x));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeRandomUniformLike(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeRange, UnknownValuesProducesSymbolicDim) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("start", MakeScalar(core::symbolic::TensorType::kFloat));
  ctx.Set("limit", MakeScalar(core::symbolic::TensorType::kFloat));
  ctx.Set("delta", MakeScalar(core::symbolic::TensorType::kFloat));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  EXPECT_FALSE(ctx.Get("y").Shape()[0].IsInt());
}

TEST(OnnxOptimShapeRange, KnownIntegerValuesProducesConcreteDim) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
  core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
  core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
  start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(3)});
  limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(9)});
  delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(3)});
  ctx.Set("start", std::move(start));
  ctx.Set("limit", std::move(limit));
  ctx.Set("delta", std::move(delta));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeRange, NegativeDeltaProducesConcreteDim) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt32);
  core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt32);
  core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt32);
  start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(10)});
  limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(6)});
  delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(-3)});
  ctx.Set("start", std::move(start));
  ctx.Set("limit", std::move(limit));
  ctx.Set("delta", std::move(delta));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeRange, EmptyOutputWhenStartEqualsLimit) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
  core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
  core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
  start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(5)});
  limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(5)});
  delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(1)});
  ctx.Set("start", std::move(start));
  ctx.Set("limit", std::move(limit));
  ctx.Set("delta", std::move(delta));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(0))}));
}

TEST(OnnxOptimShapeRange, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotRange");
  node.add_input("start");
  node.add_input("limit");
  node.add_input("delta");
  node.add_output("y");

  core::shapes::ShapesContext ctx;
  ctx.Set("start", MakeScalar(core::symbolic::TensorType::kFloat));
  ctx.Set("limit", MakeScalar(core::symbolic::TensorType::kFloat));
  ctx.Set("delta", MakeScalar(core::symbolic::TensorType::kFloat));
  EXPECT_THROW(onnx_optim::shapes::generator::ComputeShapeRange(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Symbolic propagation tests — Range(start, limit, 1) where at least one
// of start / limit is symbolic. The output shape must reuse the existing
// symbolic tokens instead of introducing a fresh ``"Range_dim0"`` token.
// ---------------------------------------------------------------------------

// Range(0, sym, 1) → output dim equals the symbolic dim carried by limit.
TEST(OnnxOptimShapeRange, SymbolicLimitDelta1PropagatesToOutput) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;

  // start = 0 (known integer)
  core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
  start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(0))});

  // limit = symbolic "N" (value propagated via ValueAsShape)
  core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
  limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("N")});

  // delta = 1 (known integer)
  core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
  delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(1))});

  ctx.Set("start", std::move(start));
  ctx.Set("limit", std::move(limit));
  ctx.Set("delta", std::move(delta));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  // Output dim must be symbolic and should encode "N" (possibly simplified
  // from "N-0").
  ASSERT_FALSE(ctx.Get("y").Shape()[0].IsInt());
  const std::string &expr = ctx.Get("y").Shape()[0].AsExpr();
  EXPECT_NE(expr.find('N'), std::string::npos) << "expected 'N' in output dim: " << expr;
  EXPECT_EQ(expr.find("Range_dim0"), std::string::npos)
      << "output must not introduce a new 'Range_dim0' token";
}

// Range(sym_start, sym_limit, 1) → output dim encodes the difference.
TEST(OnnxOptimShapeRange, SymbolicStartAndLimitDelta1PropagatesDifference) {
  NodeProto node = MakeRangeNode();
  core::shapes::ShapesContext ctx;

  // start = symbolic "S"
  core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
  start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("S")});

  // limit = symbolic "L"
  core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
  limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("L")});

  // delta = 1
  core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
  delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(1))});

  ctx.Set("start", std::move(start));
  ctx.Set("limit", std::move(limit));
  ctx.Set("delta", std::move(delta));

  ctx.ComputeShapeNode(node);
  ASSERT_TRUE(ctx.Has("y"));
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 1u);
  ASSERT_FALSE(ctx.Get("y").Shape()[0].IsInt());
  const std::string &expr = ctx.Get("y").Shape()[0].AsExpr();
  EXPECT_EQ(expr.find("Range_dim0"), std::string::npos)
      << "output must not introduce a new 'Range_dim0' token";
}

// Two independent Range nodes must NOT share the same symbolic dim name.
TEST(OnnxOptimShapeRange, TwoRangesDoNotShareSymbol) {
  core::shapes::ShapesContext ctx;

  // Range 1: Range(0, "A", 1)
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start1");
    node.add_input("limit1");
    node.add_input("delta1");
    node.add_output("y1");

    core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
    start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(0))});
    core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
    limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("A")});
    core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
    delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(1))});
    ctx.Set("start1", std::move(start));
    ctx.Set("limit1", std::move(limit));
    ctx.Set("delta1", std::move(delta));
    ctx.ComputeShapeNode(node);
  }

  // Range 2: Range(0, "B", 1)
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start2");
    node.add_input("limit2");
    node.add_input("delta2");
    node.add_output("y2");

    core::symbolic::SymTensor start = MakeScalar(core::symbolic::TensorType::kInt64);
    start.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(0))});
    core::symbolic::SymTensor limit = MakeScalar(core::symbolic::TensorType::kInt64);
    limit.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("B")});
    core::symbolic::SymTensor delta = MakeScalar(core::symbolic::TensorType::kInt64);
    delta.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(INT64_C(1))});
    ctx.Set("start2", std::move(start));
    ctx.Set("limit2", std::move(limit));
    ctx.Set("delta2", std::move(delta));
    ctx.ComputeShapeNode(node);
  }

  ASSERT_TRUE(ctx.Has("y1"));
  ASSERT_TRUE(ctx.Has("y2"));
  ASSERT_EQ(ctx.Get("y1").Shape().Rank(), 1u);
  ASSERT_EQ(ctx.Get("y2").Shape().Rank(), 1u);
  // The two Range outputs must have different symbolic dim names.
  EXPECT_NE(ctx.Get("y1").Shape()[0], ctx.Get("y2").Shape()[0])
      << "two Range nodes with different limits must not share the same output dim";
}

} // namespace Test
