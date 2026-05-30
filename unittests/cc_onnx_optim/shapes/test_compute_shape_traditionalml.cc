// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeLabelEncoderNode() {
  NodeProto node;
  node.set_op_type("LabelEncoder");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

AttributeProto *AddAttr(NodeProto &node, const std::string &name,
                        AttributeProto::AttributeType type) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(type);
  return attr;
}

void SeedInput(onnx_optim::shapes::ShapesContext &ctx, onnx_optim::TensorType dtype,
               onnx_optim::OptimShape shape) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, dtype, std::move(shape)));
}

} // namespace

TEST(OnnxOptimShapeLabelEncoder, MapsStringKeysToInt64Values) {
  // values_int64s → output dtype is INT64, shape mirrors input shape.
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_strings", AttributeProto::AttributeType::STRINGS);
  (*keys->add_strings()) = "Amy";
  (*keys->add_strings()) = "Sally";
  AttributeProto *values = AddAttr(node, "values_int64s", AttributeProto::AttributeType::INTS);
  values->add_ints(static_cast<int64_t>(5));
  values->add_ints(static_cast<int64_t>(6));

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kString, onnx_optim::OptimShape{onnx_optim::OptimDim(5)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapeLabelEncoder, MapsInt64KeysToStringValues) {
  // values_strings → output dtype is STRING.
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_int64s", AttributeProto::AttributeType::INTS);
  keys->add_ints(static_cast<int64_t>(1));
  keys->add_ints(static_cast<int64_t>(2));
  AttributeProto *values = AddAttr(node, "values_strings", AttributeProto::AttributeType::STRINGS);
  (*values->add_strings()) = "one";
  (*values->add_strings()) = "two";

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeLabelEncoder, MapsFloatKeysToFloatValues) {
  // values_floats → output dtype is FLOAT.
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_floats", AttributeProto::AttributeType::FLOATS);
  keys->add_floats(1.0f);
  AttributeProto *values = AddAttr(node, "values_floats", AttributeProto::AttributeType::FLOATS);
  values->add_floats(2.0f);

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat, onnx_optim::OptimShape{onnx_optim::OptimDim(4)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapeLabelEncoder, UsesValuesTensorDtype) {
  // values_tensor → output dtype matches the tensor's data_type.
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_int64s", AttributeProto::AttributeType::INTS);
  keys->add_ints(static_cast<int64_t>(0));
  AttributeProto *values = AddAttr(node, "values_tensor", AttributeProto::AttributeType::TENSOR);
  TensorProto *t = values->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(TensorProto::DataType::DOUBLE));
  t->add_dims(1);

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(7)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(7)}));
}

TEST(OnnxOptimShapeLabelEncoder, RejectsMissingValuesAttribute) {
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_int64s", AttributeProto::AttributeType::INTS);
  keys->add_ints(static_cast<int64_t>(0));
  // No values_* attribute set.

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeLabelEncoder, RejectsMultipleValuesAttributes) {
  NodeProto node = MakeLabelEncoderNode();
  AddAttr(node, "values_int64s", AttributeProto::AttributeType::INTS)
      ->add_ints(static_cast<int64_t>(1));
  AddAttr(node, "values_floats", AttributeProto::AttributeType::FLOATS)->add_floats(1.0f);

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeLabelEncoder, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLabelEncoder");
  node.add_input("X");
  node.add_output("Y");

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  EXPECT_THROW(onnx_optim::shapes::traditionalml::ComputeShapeLabelEncoder(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeArrayFeatureExtractorNode() {
  NodeProto node;
  node.set_op_type("ArrayFeatureExtractor");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_input("Y");
  node.add_output("Z");
  return node;
}

} // namespace

TEST(OnnxOptimShapeArrayFeatureExtractor, ReplacesLastDimWithFlattenedIndicesCount) {
  NodeProto node = MakeArrayFeatureExtractorNode();

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)});
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeArrayFeatureExtractor, PreservesSingleSymbolicIndicesDim) {
  NodeProto node = MakeArrayFeatureExtractorNode();

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt32,
            onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(8)});
  ctx.Set("Y", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim("K")}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("K")}));
}

TEST(OnnxOptimShapeArrayFeatureExtractor, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotArrayFeatureExtractor");
  node.add_input("X");
  node.add_input("Y");
  node.add_output("Z");

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));

  EXPECT_THROW(
      onnx_optim::shapes::traditionalml::ComputeShapeArrayFeatureExtractor(ctx, node, "X", "Y"),
      std::invalid_argument);
}

namespace {

NodeProto MakeBinarizerNode(float threshold = 0.0f) {
  NodeProto node;
  node.set_op_type("Binarizer");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("threshold");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(threshold);
  return node;
}

} // namespace

TEST(OnnxOptimShapeBinarizer, PreservesInputShapeAndFloatDtype) {
  NodeProto node = MakeBinarizerNode(1.0f);

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat,
            onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                   onnx_optim::OptimDim(5)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                    onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapeBinarizer, PreservesInputShapeAndInt64Dtype) {
  NodeProto node = MakeBinarizerNode(2.5f);

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64,
            onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(2)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(2)}));
}

TEST(OnnxOptimShapeBinarizer, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotBinarizer");
  node.add_input("X");
  node.add_output("Y");

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  EXPECT_THROW(onnx_optim::shapes::traditionalml::ComputeShapeBinarizer(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeZipMapNode() {
  NodeProto node;
  node.set_op_type("ZipMap");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Z");
  return node;
}

} // namespace

TEST(OnnxOptimShapeZipMap, UsesStringKeyOutputTypeForClasslabelsStrings) {
  NodeProto node = MakeZipMapNode();
  AttributeProto *labels =
      AddAttr(node, "classlabels_strings", AttributeProto::AttributeType::STRINGS);
  (*labels->add_strings()) = "c0";
  (*labels->add_strings()) = "c1";

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kSeqMapStringFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
}

TEST(OnnxOptimShapeZipMap, UsesInt64KeyOutputTypeForClasslabelsInt64s) {
  NodeProto node = MakeZipMapNode();
  AttributeProto *labels = AddAttr(node, "classlabels_int64s", AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat, onnx_optim::OptimShape{onnx_optim::OptimDim(3)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kSeqMapInt64Float);
  EXPECT_EQ(ctx.Get("Z").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
}

TEST(OnnxOptimShapeZipMap, RejectsInvalidClasslabelsConfiguration) {
  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kFloat,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  {
    NodeProto node = MakeZipMapNode();
    EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
  }

  {
    NodeProto node = MakeZipMapNode();
    (*AddAttr(node, "classlabels_strings", AttributeProto::AttributeType::STRINGS)->add_strings()) =
        "c0";
    AddAttr(node, "classlabels_int64s", AttributeProto::AttributeType::INTS)
        ->add_ints(static_cast<int64_t>(0));
    EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
  }
}

namespace {

NodeProto MakeOneHotEncoderNode() {
  NodeProto node;
  node.set_op_type("OneHotEncoder");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeOneHotEncoder, AppendsCategoryDimForInt64Categories) {
  NodeProto node = MakeOneHotEncoderNode();
  AttributeProto *cats = AddAttr(node, "cats_int64s", AttributeProto::AttributeType::INTS);
  cats->add_ints(static_cast<int64_t>(0));
  cats->add_ints(static_cast<int64_t>(1));
  cats->add_ints(static_cast<int64_t>(2));
  cats->add_ints(static_cast<int64_t>(3));

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim("N")});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapeOneHotEncoder, AppendsCategoryDimForStringCategories) {
  NodeProto node = MakeOneHotEncoderNode();
  AttributeProto *cats = AddAttr(node, "cats_strings", AttributeProto::AttributeType::STRINGS);
  (*cats->add_strings()) = "a";
  (*cats->add_strings()) = "b";
  (*cats->add_strings()) = "c";

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kString,
            onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)});

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5),
                                    onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeOneHotEncoder, RejectsInvalidCategoryAttributeConfiguration) {
  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  {
    // Neither attribute set.
    NodeProto node = MakeOneHotEncoderNode();
    EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
  }
  {
    // Both attributes set.
    NodeProto node = MakeOneHotEncoderNode();
    AddAttr(node, "cats_int64s", AttributeProto::AttributeType::INTS)
        ->add_ints(static_cast<int64_t>(0));
    (*AddAttr(node, "cats_strings", AttributeProto::AttributeType::STRINGS)->add_strings()) = "a";
    EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
  }
}

TEST(OnnxOptimShapeOneHotEncoder, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotOneHotEncoder");
  node.add_input("X");
  node.add_output("Y");

  onnx_optim::shapes::ShapesContext ctx;
  SeedInput(ctx, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});

  EXPECT_THROW(onnx_optim::shapes::traditionalml::ComputeShapeOneHotEncoder(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
