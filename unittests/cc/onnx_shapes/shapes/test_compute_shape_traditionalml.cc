// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_shapes/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
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

void SeedInput(core::shapes::ShapesContext &ctx, core::symbolic::TensorType dtype,
               core::symbolic::SymShape shape) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, dtype, std::move(shape)));
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kString,
            core::symbolic::SymShape{core::symbolic::SymDim(5)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(5)}));
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeLabelEncoder, MapsFloatKeysToFloatValues) {
  // values_floats → output dtype is FLOAT.
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_floats", AttributeProto::AttributeType::FLOATS);
  keys->add_floats(1.0f);
  AttributeProto *values = AddAttr(node, "values_floats", AttributeProto::AttributeType::FLOATS);
  values->add_floats(2.0f);

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(4)}));
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(7)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapeLabelEncoder, RejectsMissingValuesAttribute) {
  NodeProto node = MakeLabelEncoderNode();
  AttributeProto *keys = AddAttr(node, "keys_int64s", AttributeProto::AttributeType::INTS);
  keys->add_ints(static_cast<int64_t>(0));
  // No values_* attribute set.

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeLabelEncoder, RejectsMultipleValuesAttributes) {
  NodeProto node = MakeLabelEncoderNode();
  AddAttr(node, "values_int64s", AttributeProto::AttributeType::INTS)
      ->add_ints(static_cast<int64_t>(1));
  AddAttr(node, "values_floats", AttributeProto::AttributeType::FLOATS)->add_floats(1.0f);

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeLabelEncoder, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLabelEncoder");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeLabelEncoder(ctx, node, "X"),
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)});
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeArrayFeatureExtractor, PreservesSingleSymbolicIndicesDim) {
  NodeProto node = MakeArrayFeatureExtractorNode();

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt32,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(8)});
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim("K")}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim("K")}));
}

TEST(OnnxOptimShapeArrayFeatureExtractor, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotArrayFeatureExtractor");
  node.add_input("X");
  node.add_input("Y");
  node.add_output("Z");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));

  EXPECT_THROW(
      onnx_shapes::shapes::traditionalml::ComputeShapeArrayFeatureExtractor(ctx, node, "X", "Y"),
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                     core::symbolic::SymDim(5)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                      core::symbolic::SymDim(5)}));
}

TEST(OnnxOptimShapeBinarizer, PreservesInputShapeAndInt64Dtype) {
  NodeProto node = MakeBinarizerNode(2.5f);

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(2)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeBinarizer, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotBinarizer");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeBinarizer(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeScalerNode() {
  NodeProto node;
  node.set_op_type("Scaler");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  AttributeProto *offset = AddAttr(node, "offset", AttributeProto::AttributeType::FLOATS);
  offset->add_floats(0.0f);
  AttributeProto *scale = AddAttr(node, "scale", AttributeProto::AttributeType::FLOATS);
  scale->add_floats(1.0f);
  return node;
}

} // namespace

TEST(OnnxOptimShapeScaler, PreservesShapeAndForcesFloatDtypeForFloatInput) {
  NodeProto node = MakeScalerNode();

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeScaler, PreservesShapeAndForcesFloatDtypeForInt64Input) {
  NodeProto node = MakeScalerNode();

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeScaler, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotScaler");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeScaler(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeNormalizerNode(const std::string &norm) {
  NodeProto node;
  node.set_op_type("Normalizer");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  AttributeProto *attr = AddAttr(node, "norm", AttributeProto::AttributeType::STRING);
  attr->set_s(norm);
  return node;
}

} // namespace

TEST(OnnxOptimShapeNormalizer, PreservesShapeAndForcesFloatDtypeForFloatInput) {
  NodeProto node = MakeNormalizerNode("L2");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeNormalizer, PreservesShapeAndForcesFloatDtypeForInt64Input) {
  NodeProto node = MakeNormalizerNode("L1");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeNormalizer, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotNormalizer");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeNormalizer(ctx, node, "X"),
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kSeqMapStringFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeZipMap, UsesInt64KeyOutputTypeForClasslabelsInt64s) {
  NodeProto node = MakeZipMapNode();
  AttributeProto *labels = AddAttr(node, "classlabels_int64s", AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(3)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kSeqMapInt64Float);
  EXPECT_EQ(ctx.Get("Z").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(1)}));
}

TEST(OnnxOptimShapeZipMap, RejectsInvalidClasslabelsConfiguration) {
  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  {
    NodeProto node = MakeZipMapNode();
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  }

  {
    NodeProto node = MakeZipMapNode();
    (*AddAttr(node, "classlabels_strings", AttributeProto::AttributeType::STRINGS)->add_strings()) =
        "c0";
    AddAttr(node, "classlabels_int64s", AttributeProto::AttributeType::INTS)
        ->add_ints(static_cast<int64_t>(0));
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
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

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim("N")});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)}));
}

TEST(OnnxOptimShapeOneHotEncoder, AppendsCategoryDimForStringCategories) {
  NodeProto node = MakeOneHotEncoderNode();
  AttributeProto *cats = AddAttr(node, "cats_strings", AttributeProto::AttributeType::STRINGS);
  (*cats->add_strings()) = "a";
  (*cats->add_strings()) = "b";
  (*cats->add_strings()) = "c";

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kString,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5),
                                      core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeOneHotEncoder, RejectsInvalidCategoryAttributeConfiguration) {
  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  {
    // Neither attribute set.
    NodeProto node = MakeOneHotEncoderNode();
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  }
  {
    // Both attributes set.
    NodeProto node = MakeOneHotEncoderNode();
    AddAttr(node, "cats_int64s", AttributeProto::AttributeType::INTS)
        ->add_ints(static_cast<int64_t>(0));
    (*AddAttr(node, "cats_strings", AttributeProto::AttributeType::STRINGS)->add_strings()) = "a";
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  }
}

TEST(OnnxOptimShapeOneHotEncoder, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotOneHotEncoder");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeOneHotEncoder(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeSVMClassifierNode() {
  NodeProto node;
  node.set_op_type("SVMClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");
  return node;
}

NodeProto MakeSVMRegressorNode() {
  NodeProto node;
  node.set_op_type("SVMRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeSVMClassifier, InfersInt64LabelsAndBinaryScoreShape) {
  NodeProto node = MakeSVMClassifierNode();
  AttributeProto *labels = AddAttr(node, "classlabels_ints", AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(5)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeSVMClassifier, InfersStringLabelsAndMulticlassScoreShape) {
  NodeProto node = MakeSVMClassifierNode();
  AttributeProto *labels =
      AddAttr(node, "classlabels_strings", AttributeProto::AttributeType::STRINGS);
  (*labels->add_strings()) = "a";
  (*labels->add_strings()) = "b";
  (*labels->add_strings()) = "c";

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kDouble,
            core::symbolic::SymShape{core::symbolic::SymDim(7)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeSVMRegressor, InfersBatchByOneFloatOutput) {
  NodeProto node = MakeSVMRegressorNode();
  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt32,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(1)}));
}

TEST(OnnxOptimShapeSVMClassifier, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSVMClassifier");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeSVMClassifier(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSVMRegressor, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSVMRegressor");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeSVMRegressor(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeLinearClassifierNode() {
  NodeProto node;
  node.set_op_type("LinearClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");
  return node;
}

NodeProto MakeLinearRegressorNode() {
  NodeProto node;
  node.set_op_type("LinearRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeLinearClassifier, InfersInt64LabelsBinaryScoreShape) {
  NodeProto node = MakeLinearClassifierNode();
  AttributeProto *intercepts = AddAttr(node, "intercepts", AttributeProto::AttributeType::FLOATS);
  intercepts->add_floats(0.0f);
  AttributeProto *labels = AddAttr(node, "classlabels_ints", AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(5)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kFloat);
  // Binary classifier with one intercept and two labels expands to two
  // score columns.
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeLinearClassifier, InfersStringLabelsMulticlassScoreShape) {
  NodeProto node = MakeLinearClassifierNode();
  AttributeProto *intercepts = AddAttr(node, "intercepts", AttributeProto::AttributeType::FLOATS);
  intercepts->add_floats(0.0f);
  intercepts->add_floats(0.0f);
  intercepts->add_floats(0.0f);
  AttributeProto *labels =
      AddAttr(node, "classlabels_strings", AttributeProto::AttributeType::STRINGS);
  (*labels->add_strings()) = "a";
  (*labels->add_strings()) = "b";
  (*labels->add_strings()) = "c";

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kDouble,
            core::symbolic::SymShape{core::symbolic::SymDim(7)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeLinearRegressor, InfersBatchByTargetsFloatOutput) {
  NodeProto node = MakeLinearRegressorNode();
  AttributeProto *targets = AddAttr(node, "targets", AttributeProto::AttributeType::INT);
  targets->set_i(static_cast<int64_t>(3));

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt32,
            core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeLinearRegressor, DefaultsTargetsToOne) {
  NodeProto node = MakeLinearRegressorNode();
  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(4)});

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1)}));
}

TEST(OnnxOptimShapeLinearClassifier, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLinearClassifier");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeLinearClassifier(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapeLinearRegressor, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotLinearRegressor");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(1)});

  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeLinearRegressor(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeDictVectorizerNode() {
  NodeProto node;
  node.set_op_type("DictVectorizer");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

NodeProto MakeFeatureVectorizerNode(const std::vector<std::string> &input_names) {
  NodeProto node;
  node.set_op_type("FeatureVectorizer");
  node.set_domain("ai.onnx.ml");
  for (const std::string &n : input_names) {
    node.add_input(n);
  }
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeDictVectorizer, OutputShapeIsVocabularyLengthWithStringVocab) {
  NodeProto node = MakeDictVectorizerNode();
  AttributeProto *vocab =
      AddAttr(node, "string_vocabulary", AttributeProto::AttributeType::STRINGS);
  (*vocab->add_strings()) = "a";
  (*vocab->add_strings()) = "b";
  (*vocab->add_strings()) = "c";

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(1))});
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeDictVectorizer, OutputShapeIsVocabularyLengthWithInt64Vocab) {
  NodeProto node = MakeDictVectorizerNode();
  AttributeProto *vocab = AddAttr(node, "int64_vocabulary", AttributeProto::AttributeType::INTS);
  vocab->add_ints(10);
  vocab->add_ints(20);

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kFloat,
            core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(1))});
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapeDictVectorizer, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotDictVectorizer");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeDictVectorizer(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapeFeatureVectorizer, ConcatenatesFeatureDimsWhenKnown) {
  NodeProto node = MakeFeatureVectorizerNode({"A", "B"});

  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(4)),
                                            core::symbolic::SymDim(static_cast<int64_t>(3))}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(4)),
                                            core::symbolic::SymDim(static_cast<int64_t>(2))}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(4)),
                                      core::symbolic::SymDim(static_cast<int64_t>(5))}));
}

TEST(OnnxOptimShapeFeatureVectorizer, UsesInputDimensionsAttributeWhenProvided) {
  NodeProto node = MakeFeatureVectorizerNode({"A", "B"});
  AttributeProto *dims = AddAttr(node, "inputdimensions", AttributeProto::AttributeType::INTS);
  dims->add_ints(3);
  dims->add_ints(2);

  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(2)),
                                            core::symbolic::SymDim(static_cast<int64_t>(3))}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(2)),
                                            core::symbolic::SymDim(static_cast<int64_t>(2))}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(2)),
                                      core::symbolic::SymDim(static_cast<int64_t>(5))}));
}

TEST(OnnxOptimShapeFeatureVectorizer, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotFeatureVectorizer");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeFeatureVectorizer(
                   ctx, node, std::vector<std::string>{"X"}),
               std::invalid_argument);
}

namespace {

NodeProto MakeCastMapNode() {
  NodeProto node;
  node.set_op_type("CastMap");
  node.set_domain("ai.onnx.ml");
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapeCastMap, SparseProducesShapeFromMaxMap) {
  NodeProto node = MakeCastMapNode();
  AttributeProto *cast_to = AddAttr(node, "cast_to", AttributeProto::AttributeType::STRING);
  cast_to->set_s("TO_INT64");
  AttributeProto *map_form = AddAttr(node, "map_form", AttributeProto::AttributeType::STRING);
  map_form->set_s("SPARSE");
  AttributeProto *max_map = AddAttr(node, "max_map", AttributeProto::AttributeType::INT);
  max_map->set_i(7);

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(1))});
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(), (core::symbolic::SymShape{core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapeCastMap, DenseProducesSymbolic1DShape) {
  NodeProto node = MakeCastMapNode();
  AttributeProto *cast_to = AddAttr(node, "cast_to", AttributeProto::AttributeType::STRING);
  cast_to->set_s("TO_STRING");

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(1))});
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 1u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
}

TEST(OnnxOptimShapeCastMap, DefaultsCastToToFloat) {
  // No cast_to / map_form / max_map attributes — should default to TO_FLOAT/DENSE.
  NodeProto node = MakeCastMapNode();

  core::shapes::ShapesContext ctx;
  SeedInput(ctx, core::symbolic::TensorType::kInt64,
            core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(1))});
  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapeCastMap, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotCastMap");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeCastMap(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapeCastMap, DirectCallRejectsUnknownCastTo) {
  NodeProto node = MakeCastMapNode();
  AttributeProto *cast_to = AddAttr(node, "cast_to", AttributeProto::AttributeType::STRING);
  cast_to->set_s("TO_BANANA");

  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::traditionalml::ComputeShapeCastMap(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
