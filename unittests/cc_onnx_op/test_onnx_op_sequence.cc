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

const onnx_op::LightOpSchema *FindSequenceSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                                                 const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpSequenceRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::sequence::GetAllOnnxOpSequenceSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 9u);

  const onnx_op::LightOpSchema *const sequence_empty_v11 =
      FindSequenceSchema(schemas, "SequenceEmpty", 11);
  const onnx_op::LightOpSchema *const sequence_length_v11 =
      FindSequenceSchema(schemas, "SequenceLength", 11);
  const onnx_op::LightOpSchema *const sequence_construct_v11 =
      FindSequenceSchema(schemas, "SequenceConstruct", 11);
  const onnx_op::LightOpSchema *const sequence_insert_v11 =
      FindSequenceSchema(schemas, "SequenceInsert", 11);
  const onnx_op::LightOpSchema *const sequence_at_v11 =
      FindSequenceSchema(schemas, "SequenceAt", 11);
  const onnx_op::LightOpSchema *const sequence_erase_v11 =
      FindSequenceSchema(schemas, "SequenceErase", 11);
  const onnx_op::LightOpSchema *const sequence_map_v17 =
      FindSequenceSchema(schemas, "SequenceMap", 17);
  const onnx_op::LightOpSchema *const split_to_sequence_v11 =
      FindSequenceSchema(schemas, "SplitToSequence", 11);
  const onnx_op::LightOpSchema *const concat_from_sequence_v11 =
      FindSequenceSchema(schemas, "ConcatFromSequence", 11);

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
