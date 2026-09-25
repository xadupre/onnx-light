// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <gtest/gtest.h>

#include "onnx_proto/onnx.h"

using namespace ONNX_LIGHT_NAMESPACE;

TEST(TypeComparison, PresenceShapesAndSymbolicDimensions) {
  TypeProto left, right;
  std::string difference = "stale";
  EXPECT_TRUE(left.Equals(right, &difference));
  EXPECT_TRUE(difference.empty());
  left.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "tensor_type: presence differs");
  right = left;
  left.mutable_tensor_type()->mutable_shape();
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "tensor_type.shape: presence differs");
  right = left;
  EXPECT_TRUE(left.Equals(right, &difference));
  left.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("N");
  right = left;
  right.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_param("M");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "tensor_type.shape.dim[0].dim_param: values differ");
  right = left;
  right.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_denotation("");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "tensor_type.shape.dim[0].denotation: presence differs");
  right = left;
  right.mutable_tensor_type()->mutable_shape()->add_dim();
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "tensor_type.shape.dim: size differs (1 != 2)");
}

TEST(TypeComparison, ContainersSparseAndOpaqueTypes) {
  TypeProto left;
  auto *map = left.mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
  map->set_key_type(TensorProto::INT64);
  auto *sparse = map->mutable_value_type()
                     ->mutable_optional_type()
                     ->mutable_elem_type()
                     ->mutable_sparse_tensor_type();
  sparse->set_elem_type(TensorProto::FLOAT);
  sparse->mutable_shape()->add_dim()->set_dim_value(4);
  auto right = left;
  EXPECT_TRUE(left.Equals(right));
  right.mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_map_type()
      ->mutable_value_type()
      ->mutable_optional_type()
      ->mutable_elem_type()
      ->mutable_sparse_tensor_type()
      ->set_elem_type(TensorProto::DOUBLE);
  std::string difference;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "sequence_type.elem_type.map_type.value_type.optional_type.elem_type."
                        "sparse_tensor_type.elem_type: values differ");
  right = left;
  right.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(
      TensorProto::STRING);
  EXPECT_FALSE(left.Equals(right));
  left = TypeProto();
  left.mutable_opaque_type()->set_domain("example");
  left.mutable_opaque_type()->set_name("opaque");
  right = left;
  EXPECT_TRUE(left.Equals(right));
  right.mutable_opaque_type()->set_name("other");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "opaque_type.name: values differ");
}

TEST(TypeComparison, StructuredDeclarationsConstantsAndMetadata) {
  StructTypeProto left;
  left.set_type_id(uint64_t{7});
  left.set_name("record");
  auto *field = left.mutable_structure()->add_field();
  field->set_name("codes");
  auto *packing = field->mutable_type()->mutable_struct_type()->mutable_bit_packing();
  packing->set_dimension(4);
  packing->add_component()->set_bit_width(2);
  packing->mutable_component(0)->set_name("code");
  auto *constant = left.mutable_structure()->add_field()->mutable_constant();
  constant->set_data_type(TensorProto::INT64);
  constant->add_int64_data(16);
  auto *metadata = left.add_metadata_props();
  metadata->set_key("version");
  metadata->set_value("1");
  auto right = left;
  EXPECT_TRUE(left.Equals(right));
  right.mutable_structure()->mutable_field(1)->mutable_constant()->ref_int64_data()[0] = 32;
  std::string difference;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "structure.field[1].constant.int64_data[0]: values differ");
  right = left;
  right.mutable_structure()
      ->mutable_field(0)
      ->mutable_type()
      ->mutable_struct_type()
      ->mutable_bit_packing()
      ->mutable_component(0)
      ->set_bit_width(3);
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(
      difference,
      "structure.field[0].type.struct_type.bit_packing.component[0].bit_width: values differ");
  right = left;
  right.mutable_metadata_props(0)->set_value("2");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "metadata_props[0].value: values differ");
  right = left;
  right.set_type_id(uint64_t{8});
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "type_id: values differ");
  left.Clear();
  left.mutable_array()->set_dimension(2);
  left.mutable_array()->mutable_element_type()->mutable_struct_type()->set_type_ref(uint64_t{7});
  right = left;
  EXPECT_TRUE(left.Equals(right));
  right.mutable_array()->mutable_element_type()->mutable_struct_type()->set_type_ref(uint64_t{8});
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "array.element_type.struct_type.type_ref: values differ");
}

TEST(TypeComparison, CodecBodiesAndFloatingPointBits) {
  StructTypeProto left;
  auto *node = left.mutable_decoder()->add_node();
  node->set_op_type("Constant");
  auto *attribute = node->add_attribute();
  attribute->set_type(AttributeProto::FLOATS);
  attribute->add_floats(std::bit_cast<float>(uint32_t{0x7fc00001}));
  auto right = left;
  EXPECT_TRUE(left.Equals(right));
  right.mutable_decoder()->mutable_node(0)->mutable_attribute(0)->ref_floats()[0] =
      std::bit_cast<float>(uint32_t{0x7fc00002});
  std::string difference;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "decoder.node[0].attribute[0].floats[0]: values differ");
  attribute->ref_floats()[0] = 0.0f;
  right = left;
  right.mutable_decoder()->mutable_node(0)->mutable_attribute(0)->ref_floats()[0] = -0.0f;
  EXPECT_FALSE(left.Equals(right));
  right = left;
  right.mutable_encoder()->add_node()->set_op_type("Identity");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "encoder: presence differs");
}

TEST(TypeComparison, ExternalConstantsDoNotRequireSerialization) {
  StructTypeProto left;
  auto *constant = left.mutable_structure()->add_field()->mutable_constant();
  constant->set_data_type(TensorProto::FLOAT);
  constant->set_data_location(TensorProto::EXTERNAL);
  constant->add_dims(1);
  constant->add_external_data()->set_key("location");
  constant->mutable_external_data(0)->set_value("unloaded.data");
  const auto right = left;
  EXPECT_TRUE(left.Equals(right));
}

TEST(TypeComparison, EncodedPayloadAndAffineLayout) {
  EncodedValueProto left;
  left.set_raw_data(std::string("\0\1", 2));
  left.mutable_affine()->set_storage_type(TensorProto::INT8);
  left.mutable_affine()->mutable_scale()->set_data_type(TensorProto::FLOAT);
  left.mutable_affine()->mutable_scale()->add_float_data(1.0f);
  auto right = left;
  EXPECT_TRUE(left.Equals(right));
  right.set_raw_data(std::string("\0\2", 2));
  std::string difference;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "raw_data: values differ");
  right = left;
  right.mutable_affine()->set_axis(int64_t{0});
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "affine.axis: presence differs");
  right = left;
  right.set_parameter_ref("common");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "parameter_ref: presence differs");
  left.set_parameter_ref("other");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "parameter_ref: values differ");
  left.set_parameter_ref("common");
  EXPECT_TRUE(left.Equals(right, &difference));
  EXPECT_TRUE(difference.empty());
}

TEST(TypeComparison, ModelMembersAndNestedDiagnostics) {
  ModelProto left;
  left.set_ir_version(10);
  left.mutable_graph()->set_name("g");
  left.add_configuration()->set_num_devices(2);
  left.add_struct_types()->set_type_id(uint64_t{1});
  auto right = left;
  std::string difference;
  EXPECT_TRUE(left.Equals(right, &difference));
  right.mutable_configuration(0)->add_device("cpu");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "configuration[0].device: presence differs");
  right = left;
  right.mutable_graph()->set_name("other");
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "graph.name: values differ");
  right = left;
  right.mutable_struct_types(0)->set_type_id(uint64_t{2});
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "struct_types[0].type_id: values differ");
  right = left;
  EXPECT_TRUE(left.Equals(right, &difference));
  EXPECT_TRUE(difference.empty());
}

TEST(TypeComparison, SequenceMapAndOptionalMembers) {
  OptionalProto left;
  auto *map = left.mutable_sequence_value()->add_map_values();
  map->set_key_type(TensorProto::INT64);
  map->add_keys(0);
  auto *tensor = map->mutable_values()->add_tensor_values();
  tensor->set_data_type(TensorProto::FLOAT);
  tensor->add_float_data(1);
  auto right = left;
  std::string difference;
  EXPECT_TRUE(left.Equals(right, &difference));
  right.mutable_sequence_value()
      ->mutable_map_values(0)
      ->mutable_values()
      ->mutable_tensor_values(0)
      ->ref_float_data()[0] = 2;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference,
            "sequence_value.map_values[0].values.tensor_values[0].float_data[0]: values differ");
  EXPECT_FALSE(left.Equals(right));
  right = left;
  right.mutable_sequence_value()->mutable_map_values(0)->ref_keys()[0] = 1;
  EXPECT_FALSE(left.Equals(right, &difference));
  EXPECT_EQ(difference, "sequence_value.map_values[0].keys[0]: values differ");
}
