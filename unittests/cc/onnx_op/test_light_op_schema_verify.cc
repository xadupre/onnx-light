// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/light_op_schema/light_op_schema.h"
#include "onnx_lib/defs/schema.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::schema::AttributeParam;
using core::schema::AttributeType;
using core::schema::FormalParameter;
using core::schema::LightOpSchema;
using core::schema::SchemaInputValue;
using core::schema::TypeConstraintParam;

namespace Test {

using core::schema::SchemaError;

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

TEST(LightOpSchemaVerify, CosineSumWindowAttributeMetadata) {
  for (const auto *op_type : {"HannWindow", "HammingWindow", "BlackmanWindow"}) {
    SCOPED_TRACE(op_type);
    for (bool init_doc : {false, true}) {
      SCOPED_TRACE(init_doc);
      const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, init_doc);
      ASSERT_EQ(schemas.size(), 1u);
      const auto &schema = schemas.front();
      EXPECT_EQ(schema.domain(), core::schema::kOnnxDomain);
      EXPECT_EQ(schema.since_version(), 17);
      ASSERT_EQ(schema.attributes().size(), 2u);
      for (const auto *name : {"periodic", "output_datatype"}) {
        SCOPED_TRACE(name);
        const auto &attributes = schema.attributes();
        const auto found = std::find_if(
            attributes.begin(), attributes.end(),
            [name](const AttributeParam &attribute) { return attribute.name == name; });
        ASSERT_NE(found, attributes.end());
        EXPECT_EQ(found->type, AttributeType::INT);
        EXPECT_FALSE(found->required);
        const auto *default_value = std::get_if<int64_t>(&found->default_value);
        ASSERT_NE(default_value, nullptr);
        EXPECT_EQ(*default_value,
                  found->name == "periodic" ? int64_t{1} : int64_t{TensorProto::FLOAT});
        if (init_doc) {
          EXPECT_FALSE(found->description.empty());
        }
      }
    }
  }
}

TEST(LightOpSchemaVerify, CastAcceptsComplexFromOpset6) {
  using core::schema::TensorType;
  const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory("Cast", true);
  ASSERT_EQ(schemas.size(), 10u);
  for (const auto &schema : schemas) {
    SCOPED_TRACE(schema.since_version());
    ASSERT_EQ(schema.type_constraints().size(), 2u);
    for (const auto &constraint : schema.type_constraints()) {
      for (const auto type : {TensorType::kComplex64, TensorType::kComplex128}) {
        const auto found = std::find(constraint.allowed_type_strs.begin(),
                                     constraint.allowed_type_strs.end(), type);
        if (schema.since_version() == 1) {
          EXPECT_EQ(found, constraint.allowed_type_strs.end());
        } else {
          EXPECT_NE(found, constraint.allowed_type_strs.end());
        }
      }
    }
    for (const auto dtype : {TensorProto::COMPLEX64, TensorProto::COMPLEX128}) {
      SCOPED_TRACE(dtype);
      NodeProto node;
      node.set_op_type("Cast");
      node.add_input("x");
      node.add_output("y");
      auto *attr = node.add_attribute();
      attr->set_name("to");
      if (schema.since_version() == 1) {
        attr->set_type(AttributeProto::STRING);
        attr->set_s(dtype == TensorProto::COMPLEX64 ? "COMPLEX64" : "COMPLEX128");
      } else {
        attr->set_type(AttributeProto::INT);
        attr->set_i(dtype);
      }
      ValueInfoProto input;
      input.set_name("x");
      input.mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
      std::vector<std::optional<SchemaInputValue>> inputs{SchemaInputValue(input)};
      if (schema.since_version() == 1) {
        EXPECT_THROW(schema.Verify(node, &inputs), SchemaError);
      } else {
        EXPECT_NO_THROW(schema.Verify(node, &inputs));
      }
    }
  }
}

TEST(LightOpSchemaVerify, CastLikeStillExcludesComplex) {
  using core::schema::TensorType;
  for (const auto &schema : onnx_op::GetAllOnnxOpSchemasWithHistory("CastLike", false)) {
    SCOPED_TRACE(schema.since_version());
    for (const auto &constraint : schema.type_constraints()) {
      for (const auto type : {TensorType::kComplex64, TensorType::kComplex128}) {
        EXPECT_EQ(std::find(constraint.allowed_type_strs.begin(),
                            constraint.allowed_type_strs.end(), type),
                  constraint.allowed_type_strs.end());
      }
    }
  }
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

TEST(LightOpSchemaVerify, OptionalOutputBoundsMatchNativeSchemas) {
  for (bool init_doc : {false, true}) {
    for (const auto &schema : onnx_op::GetAllOnnxOpSchemasWithHistory("", init_doc)) {
      if (schema.domain() != core::schema::kOnnxDomain) {
        continue;
      }
      const auto *native = OpSchemaRegistry::Schema(schema.name(), schema.since_version());
      ASSERT_NE(native, nullptr) << schema.name();
      bool has_optional_output = false;
      for (const auto &output : native->outputs()) {
        has_optional_output |= output.GetOption() == OpSchema::Optional;
      }
      if (!has_optional_output) {
        continue;
      }
      SCOPED_TRACE(schema.name() + ":" + std::to_string(schema.since_version()));
      EXPECT_EQ(schema.min_output(), native->min_output());
      EXPECT_EQ(schema.max_output(), native->max_output());
    }
  }
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
