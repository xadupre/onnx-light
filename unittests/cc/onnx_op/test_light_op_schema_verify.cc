// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/light_op_schema/light_op_schema.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::schema::AttributeParam;
using core::schema::AttributeType;
using core::schema::FormalParameter;
using core::schema::LightOpSchema;
using core::schema::SchemaError;
using core::schema::SchemaInputValue;
using core::schema::TypeConstraintParam;

namespace Test {

namespace {

// A minimal "Scale" schema: one input "X" of type "T" (float/double), one required
// attribute "alpha" (FLOAT), one output "Y" of type "T".
LightOpSchema MakeScaleSchema() {
  return LightOpSchema(
      "Scale", core::schema::kOnnxDomain, 18, "Scales a tensor by alpha.",
      {{"X", "Input tensor", "T"}}, {{"Y", "Output tensor", "T"}},
      {{"T", {core::schema::TensorType::kFloat, core::schema::TensorType::kDouble}, "float types"}},
      {{"alpha", "scale factor", AttributeType::FLOAT, /*required=*/true, std::monostate{}}});
}

NodeProto MakeValidScaleNode() {
  NodeProto node;
  node.set_op_type("Scale");
  node.add_input("x");
  node.add_output("y");
  AttributeProto attr;
  attr.set_name("alpha");
  attr.set_type(AttributeProto::FLOAT);
  attr.set_f(0.5f);
  node.set_attribute(attr);
  return node;
}

} // namespace

TEST(LightOpSchemaVerify, ValidNodePasses) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();
  EXPECT_NO_THROW(schema.Verify(node));
}

TEST(LightOpSchemaVerify, DeprecatedSchemaRejectsNode) {
  LightOpSchema schema = MakeScaleSchema();
  schema.set_deprecated(true);
  const NodeProto node = MakeValidScaleNode();
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, MismatchedOpTypeRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node = MakeValidScaleNode();
  node.set_op_type("NotScale");
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, MismatchedDomainRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node = MakeValidScaleNode();
  node.set_domain("com.example");
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, TooManyOutputsRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node = MakeValidScaleNode();
  node.add_output("y2");
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, MissingRequiredAttributeRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node;
  node.set_op_type("Scale");
  node.add_input("x");
  node.add_output("y");
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, UnrecognizedAttributeRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node = MakeValidScaleNode();
  AttributeProto extra;
  extra.set_name("beta");
  extra.set_type(AttributeProto::FLOAT);
  extra.set_f(1.0f);
  node.add_attribute(extra);
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, InternalSymbolAttributeAllowed) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node = MakeValidScaleNode();
  AttributeProto extra;
  extra.set_name("__debug_hint");
  extra.set_type(AttributeProto::STRING);
  extra.set_s("noop");
  node.add_attribute(extra);
  EXPECT_NO_THROW(schema.Verify(node));
}

TEST(LightOpSchemaVerify, MismatchedAttributeTypeRejected) {
  const LightOpSchema schema = MakeScaleSchema();
  NodeProto node;
  node.set_op_type("Scale");
  node.add_input("x");
  node.add_output("y");
  AttributeProto attr;
  attr.set_name("alpha");
  attr.set_type(AttributeProto::INT);
  attr.set_i(2);
  node.set_attribute(attr);
  EXPECT_THROW(schema.Verify(node), SchemaError);
}

TEST(LightOpSchemaVerify, InputTypeConstraintSatisfiedWithValueInfo) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  ValueInfoProto vi;
  vi.set_name("x");
  vi.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  vi.mutable_type()->mutable_tensor_type()->mutable_shape();

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(SchemaInputValue(vi));
  EXPECT_NO_THROW(schema.Verify(node, &inputs));
}

TEST(LightOpSchemaVerify, InputTypeConstraintViolatedWithValueInfo) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  ValueInfoProto vi;
  vi.set_name("x");
  vi.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  vi.mutable_type()->mutable_tensor_type()->mutable_shape();

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(SchemaInputValue(vi));
  EXPECT_THROW(schema.Verify(node, &inputs), SchemaError);
}

TEST(LightOpSchemaVerify, InputTypeConstraintSatisfiedWithSymTensor) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  core::symbolic::SymTensor tensor(nullptr, core::symbolic::TensorType::kDouble,
                                   core::symbolic::SymShape({core::symbolic::SymDim(2)}));

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(SchemaInputValue(tensor));
  EXPECT_NO_THROW(schema.Verify(node, &inputs));
}

TEST(LightOpSchemaVerify, InputTypeConstraintViolatedWithSymTensor) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  core::symbolic::SymTensor tensor(nullptr, core::symbolic::TensorType::kBool,
                                   core::symbolic::SymShape({core::symbolic::SymDim(2)}));

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(SchemaInputValue(tensor));
  EXPECT_THROW(schema.Verify(node, &inputs), SchemaError);
}

TEST(LightOpSchemaVerify, UnknownInputTypeIsSkipped) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  // A ValueInfoProto with no type set is unresolvable; the check must be skipped rather than
  // rejecting the node.
  ValueInfoProto vi;
  vi.set_name("x");

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(SchemaInputValue(vi));
  EXPECT_NO_THROW(schema.Verify(node, &inputs));
}

TEST(LightOpSchemaVerify, MissingInputEntryIsSkipped) {
  const LightOpSchema schema = MakeScaleSchema();
  const NodeProto node = MakeValidScaleNode();

  std::vector<std::optional<SchemaInputValue>> inputs;
  inputs.emplace_back(std::nullopt);
  EXPECT_NO_THROW(schema.Verify(node, &inputs));
}

TEST(LightOpSchemaVerify, SequenceElementTypeCheckedAgainstSeqConstraint) {
  // A schema whose single input accepts seq(tensor(float)) or seq(tensor(double)).
  const LightOpSchema schema(
      "ScaleSeq", core::schema::kOnnxDomain, 18, "doc", {{"X", "Input sequence", "T"}},
      {{"Y", "Output sequence", "T"}},
      {{"T",
        {core::schema::TensorType::kSeqFloat, core::schema::TensorType::kSeqDouble},
        "sequence of float types"}});

  NodeProto node;
  node.set_op_type("ScaleSeq");
  node.add_input("x");
  node.add_output("y");

  core::symbolic::SymSequence good_seq(core::symbolic::TensorType::kFloat, /*length=*/2);
  std::vector<std::optional<SchemaInputValue>> good_inputs;
  good_inputs.emplace_back(SchemaInputValue(good_seq));
  EXPECT_NO_THROW(schema.Verify(node, &good_inputs));

  core::symbolic::SymSequence bad_seq(core::symbolic::TensorType::kBool, /*length=*/2);
  std::vector<std::optional<SchemaInputValue>> bad_inputs;
  bad_inputs.emplace_back(SchemaInputValue(bad_seq));
  EXPECT_THROW(schema.Verify(node, &bad_inputs), SchemaError);
}

} // namespace Test
