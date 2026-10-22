// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/optional/shape_optional.h"

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeOptionalNode(bool with_input, const std::string &out = "y") {
  NodeProto node;
  node.set_op_type("Optional");
  if (with_input) {
    node.add_input("x");
  }
  node.add_output(out);
  return node;
}

// Adds a ``type`` TypeProto attribute wrapping ``Optional<Tensor<dtype, shape>>``
// (or just ``Tensor<dtype, shape>`` when ``wrap_in_optional`` is false).
AttributeProto *AddTypeAttr(NodeProto &node, TensorProto::DataType dtype,
                            const std::vector<int64_t> &shape, bool wrap_in_optional = true) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Tensor *tt = nullptr;
  if (wrap_in_optional) {
    TypeProto::Optional *opt = tp->add_optional_type();
    TypeProto *elem = opt->add_elem_type();
    tt = elem->add_tensor_type();
  } else {
    tt = tp->add_tensor_type();
  }
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *sp = tt->add_shape();
  for (int64_t d : shape) {
    sp->add_dim()->set_dim_value(d);
  }
  return attr;
}

} // namespace

TEST(OnnxOptimShapeOptional, WithInputCopiesDtypeAndShape) {
  NodeProto node = MakeOptionalNode(/*with_input=*/true);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeOptional, WithInputPreservesSymbolicDims) {
  NodeProto node = MakeOptionalNode(/*with_input=*/true);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(std::string("N")),
                                 core::symbolic::SymDim(4)};
  ctx.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeOptional, NoInputWithOptionalOfTensorTypeAttribute) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  AddTypeAttr(node, TensorProto::DataType::FLOAT, {2, 3}, /*wrap_in_optional=*/true);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeOptional, NoInputWithBareTensorTypeAttribute) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  AddTypeAttr(node, TensorProto::DataType::INT32, {4}, /*wrap_in_optional=*/false);

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeOptional, NoInputNoTypeAttributeThrows) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, NoInputWithSequenceTypeAttribute) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  // type = Optional<Sequence<Tensor<FLOAT>>>.
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Optional *opt = tp->add_optional_type();
  TypeProto *elem = opt->add_elem_type();
  TypeProto::Sequence *seq = elem->add_sequence_type();
  TypeProto *inner = seq->add_elem_type();
  TypeProto::Tensor *tt = inner->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));

  core::shapes::ShapesContext ctx;
  ASSERT_NO_THROW(ctx.ComputeShapeNode(node));
  ASSERT_TRUE(ctx.HasSequence("y"));
  EXPECT_EQ(ctx.GetSequence("y").ElemDtype(), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(ctx.GetSequence("y").Length().IsInt());
}

TEST(OnnxOptimShapeOptional, WithSequenceInputPreservesElementShapes) {
  core::shapes::ShapesContext ctx;
  core::symbolic::SymSequence sequence(
      core::symbolic::TensorType::kFloat,
      std::vector<core::symbolic::SymShape>{{core::symbolic::SymDim(2)},
                                            {core::symbolic::SymDim(3)}});
  ctx.SetSequence("x", core::symbolic::SymSequence(sequence));

  ctx.ComputeShapeNode(MakeOptionalNode(/*with_input=*/true));

  ASSERT_TRUE(ctx.HasSequence("y"));
  EXPECT_EQ(ctx.GetSequence("y"), sequence);
}

TEST(OnnxOptimShapeOptional, NoInputMapTypeAttributeThrows) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  auto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::TYPE_PROTO);
  attr->mutable_tp()->mutable_map_type()->set_key_type(TensorProto::INT64);
  attr->mutable_tp()
      ->mutable_map_type()
      ->mutable_value_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::FLOAT);
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, GetElementPreservesUnknownRank) {
  core::shapes::ShapesContext ctx;
  TypeProto type;
  type.mutable_optional_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::FLOAT);
  ctx.SetType("x", type);
  NodeProto node;
  node.set_op_type("OptionalGetElement");
  node.add_input("x");
  node.add_output("y");

  ASSERT_NO_THROW(ctx.ComputeShapeNode(node));

  EXPECT_FALSE(ctx.Has("y"));
  ASSERT_TRUE(ctx.HasType("y"));
  EXPECT_EQ(ctx.GetType("y").tensor_type().elem_type(), TensorProto::FLOAT);
  EXPECT_FALSE(ctx.GetType("y").tensor_type().has_shape());
}

TEST(OnnxOptimShapeOptional, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotOptional");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::optional::ComputeShapeOptional(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, RejectsTooManyInputs) {
  NodeProto node;
  node.set_op_type("Optional");
  node.add_input("a");
  node.add_input("b");
  node.add_output("y");
  core::shapes::ShapesContext ctx;
  ctx.Set("a", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("b", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::optional::ComputeShapeOptional(ctx, node),
               std::invalid_argument);
}

} // namespace Test
