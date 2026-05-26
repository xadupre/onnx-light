// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/optional/shape_optional.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
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
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeOptional, WithInputPreservesSymbolicDims) {
  NodeProto node = MakeOptionalNode(/*with_input=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(std::string("N")), onnx_optim::OptimDim(4)};
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("y").Shape(), shape);
}

TEST(OnnxOptimShapeOptional, NoInputWithOptionalOfTensorTypeAttribute) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  AddTypeAttr(node, TensorProto::DataType::FLOAT, {2, 3}, /*wrap_in_optional=*/true);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeOptional, NoInputWithBareTensorTypeAttribute) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  AddTypeAttr(node, TensorProto::DataType::INT32, {4}, /*wrap_in_optional=*/false);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapeOptional, NoInputNoTypeAttributeThrows) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, NoInputSequenceTypeAttributeThrows) {
  NodeProto node = MakeOptionalNode(/*with_input=*/false);
  // type = Optional<Sequence<Tensor<FLOAT>>> — sequence elements aren't supported.
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

  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, RejectsBadOpType) {
  NodeProto node;
  node.set_op_type("NotOptional");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::optional::ComputeShapeOptional(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeOptional, RejectsTooManyInputs) {
  NodeProto node;
  node.set_op_type("Optional");
  node.add_input("a");
  node.add_input("b");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::optional::ComputeShapeOptional(ctx, node),
               std::invalid_argument);
}

} // namespace Test
