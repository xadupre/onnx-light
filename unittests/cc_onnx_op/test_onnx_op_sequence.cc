// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_sequence.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpSequenceRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> sequence_empty_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceEmpty");
  const std::vector<onnx_op::LightOpSchema> sequence_length_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceLength");
  const std::vector<onnx_op::LightOpSchema> sequence_construct_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceConstruct");
  const std::vector<onnx_op::LightOpSchema> sequence_insert_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceInsert");
  const std::vector<onnx_op::LightOpSchema> sequence_at_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceAt");
  const std::vector<onnx_op::LightOpSchema> sequence_erase_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceErase");
  const std::vector<onnx_op::LightOpSchema> sequence_map_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SequenceMap");
  const std::vector<onnx_op::LightOpSchema> split_to_sequence_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("SplitToSequence");
  const std::vector<onnx_op::LightOpSchema> concat_from_sequence_schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory("ConcatFromSequence");

  EXPECT_EQ(schemas.size(), 9u);

  const onnx_op::LightOpSchema *const sequence_empty_v11 =
      FindByVersion(sequence_empty_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_length_v11 =
      FindByVersion(sequence_length_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_construct_v11 =
      FindByVersion(sequence_construct_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_insert_v11 =
      FindByVersion(sequence_insert_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_at_v11 = FindByVersion(sequence_at_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_erase_v11 =
      FindByVersion(sequence_erase_schemas, 11);
  const onnx_op::LightOpSchema *const sequence_map_v17 = FindByVersion(sequence_map_schemas, 17);
  const onnx_op::LightOpSchema *const split_to_sequence_v11 =
      FindByVersion(split_to_sequence_schemas, 11);
  const onnx_op::LightOpSchema *const concat_from_sequence_v11 =
      FindByVersion(concat_from_sequence_schemas, 11);

  ASSERT_NE(nullptr, sequence_empty_v11);
  ASSERT_NE(nullptr, sequence_length_v11);
  ASSERT_NE(nullptr, sequence_construct_v11);
  ASSERT_NE(nullptr, sequence_insert_v11);
  ASSERT_NE(nullptr, sequence_at_v11);
  ASSERT_NE(nullptr, sequence_erase_v11);
  ASSERT_NE(nullptr, sequence_map_v17);
  ASSERT_NE(nullptr, split_to_sequence_v11);
  ASSERT_NE(nullptr, concat_from_sequence_v11);

  EXPECT_EQ(sequence_empty_v11->domain(), "ai.onnx");
  EXPECT_EQ(sequence_empty_v11->inputs().size(), 0u);
  EXPECT_EQ(sequence_empty_v11->outputs().size(), 1u);
  EXPECT_EQ(sequence_empty_v11->outputs()[0].name, "output");
  EXPECT_EQ(sequence_empty_v11->outputs()[0].description, "Empty sequence.");
  EXPECT_EQ(sequence_empty_v11->type_constraints().size(), 1u);
  EXPECT_EQ(sequence_empty_v11->type_constraints()[0].type_param_str, "S");
  EXPECT_EQ(sequence_empty_v11->type_constraints()[0].allowed_type_strs.size(), 15u);
  EXPECT_STREQ(
      onnx_op::ToTypeString(sequence_empty_v11->type_constraints()[0].allowed_type_strs[0]),
      "seq(tensor(uint8))");
  EXPECT_STREQ(
      onnx_op::ToTypeString(sequence_empty_v11->type_constraints()[0].allowed_type_strs.back()),
      "seq(tensor(complex128))");

  EXPECT_EQ(sequence_length_v11->domain(), "ai.onnx");
  EXPECT_EQ(sequence_length_v11->inputs().size(), 1u);
  EXPECT_EQ(sequence_length_v11->inputs()[0].name, "input_sequence");
  EXPECT_EQ(sequence_length_v11->outputs().size(), 1u);
  EXPECT_EQ(sequence_length_v11->outputs()[0].name, "length");
  EXPECT_EQ(sequence_length_v11->type_constraints().size(), 2u);
  EXPECT_EQ(sequence_length_v11->type_constraints()[0].allowed_type_strs,
            sequence_empty_v11->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sequence_length_v11->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(sequence_length_v11->type_constraints()[1].allowed_type_strs[0],
            onnx_op::TensorType::kInt64);
  EXPECT_EQ(sequence_length_v11->doc(),
            R"DOC(
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
)DOC");

  EXPECT_EQ(sequence_construct_v11->inputs().size(), 1u);
  EXPECT_EQ(sequence_construct_v11->inputs()[0].name, "inputs");
  EXPECT_EQ(sequence_construct_v11->outputs().size(), 1u);
  EXPECT_EQ(sequence_construct_v11->outputs()[0].name, "output_sequence");
  EXPECT_EQ(sequence_construct_v11->type_constraints().size(), 2u);

  EXPECT_EQ(sequence_insert_v11->inputs().size(), 3u);
  EXPECT_EQ(sequence_insert_v11->inputs()[1].name, "tensor");
  EXPECT_EQ(sequence_insert_v11->inputs()[2].name, "position");
  EXPECT_EQ(sequence_insert_v11->outputs().size(), 1u);
  EXPECT_EQ(sequence_insert_v11->outputs()[0].name, "output_sequence");
  EXPECT_EQ(sequence_insert_v11->type_constraints().size(), 3u);

  EXPECT_EQ(sequence_at_v11->inputs().size(), 2u);
  EXPECT_EQ(sequence_at_v11->outputs()[0].name, "tensor");

  EXPECT_EQ(sequence_erase_v11->inputs().size(), 2u);
  EXPECT_EQ(sequence_erase_v11->outputs()[0].name, "output_sequence");
  EXPECT_EQ(sequence_erase_v11->type_constraints().size(), 2u);

  EXPECT_EQ(sequence_map_v17->inputs().size(), 2u);
  EXPECT_EQ(sequence_map_v17->inputs()[0].name, "input_sequence");
  EXPECT_EQ(sequence_map_v17->inputs()[1].name, "additional_inputs");
  EXPECT_EQ(sequence_map_v17->outputs()[0].name, "out_sequence");
  EXPECT_TRUE(sequence_map_v17->has_function_implementation());
  // V constraint covers both tensor and sequence types.
  EXPECT_EQ(sequence_map_v17->type_constraints()[1].type_param_str, "V");
  EXPECT_EQ(sequence_map_v17->type_constraints()[1].allowed_type_strs.size(), 30u);

  EXPECT_EQ(split_to_sequence_v11->inputs().size(), 2u);
  EXPECT_EQ(split_to_sequence_v11->inputs()[0].name, "input");
  EXPECT_EQ(split_to_sequence_v11->inputs()[1].name, "split");
  EXPECT_EQ(split_to_sequence_v11->outputs()[0].name, "output_sequence");

  EXPECT_EQ(concat_from_sequence_v11->inputs().size(), 1u);
  EXPECT_EQ(concat_from_sequence_v11->inputs()[0].name, "input_sequence");
  EXPECT_EQ(concat_from_sequence_v11->outputs()[0].name, "concat_result");
}

} // namespace Test
