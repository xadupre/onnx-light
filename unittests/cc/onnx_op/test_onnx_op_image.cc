// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_image.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedImageSchemaCount = 1;

static const core::schema::LightOpSchema *
FindByVersion(const std::vector<core::schema::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpImageRegistrationTest, ReturnsImageDecoderSchema) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::image::GetAllOnnxOpImageSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> image_decoder_schemas =
      onnx_op::image::GetAllOnnxOpImageSchemasWithHistory("ImageDecoder");

  EXPECT_EQ(schemas.size(), kExpectedImageSchemaCount);

  const core::schema::LightOpSchema *const image_decoder_v20 =
      FindByVersion(image_decoder_schemas, 20);
  ASSERT_NE(nullptr, image_decoder_v20);

  EXPECT_EQ(image_decoder_v20->domain(), "ai.onnx");
  EXPECT_EQ(image_decoder_v20->inputs().size(), 1u);
  EXPECT_EQ(image_decoder_v20->outputs().size(), 1u);
  EXPECT_EQ(image_decoder_v20->type_constraints().size(), 2u);

  EXPECT_EQ(image_decoder_v20->inputs()[0].name, "encoded_stream");
  EXPECT_EQ(image_decoder_v20->inputs()[0].description, "Encoded stream");
  EXPECT_EQ(image_decoder_v20->inputs()[0].type, "T1");

  EXPECT_EQ(image_decoder_v20->outputs()[0].name, "image");
  EXPECT_EQ(image_decoder_v20->outputs()[0].description, "Decoded image");
  EXPECT_EQ(image_decoder_v20->outputs()[0].type, "T2");

  EXPECT_EQ(image_decoder_v20->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(image_decoder_v20->type_constraints()[0].allowed_type_strs,
            std::vector<core::schema::TensorType>{core::schema::TensorType::kUint8});
  EXPECT_EQ(image_decoder_v20->type_constraints()[0].description,
            "Constrain input types to 8-bit unsigned integer tensor.");

  EXPECT_EQ(image_decoder_v20->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(image_decoder_v20->type_constraints()[1].allowed_type_strs,
            std::vector<core::schema::TensorType>{core::schema::TensorType::kUint8});
  EXPECT_EQ(image_decoder_v20->type_constraints()[1].description,
            "Constrain output types to 8-bit unsigned integer tensor.");

  EXPECT_NE(image_decoder_v20->doc().find("Loads and decodes and image from a file."),
            std::string::npos);
  EXPECT_NE(image_decoder_v20->doc().find("JPEG"), std::string::npos);
}

} // namespace Test
