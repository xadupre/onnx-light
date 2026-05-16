// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/op_reg_test.cc
// and adapted to work with onnx-light.
//
// Key differences from vanilla ONNX:
//   - Schemas are NOT auto-registered at startup in onnx-light.
//   - The operator-definition .cc files (math/defs.cc, etc.) are not compiled
//     into lib_onnx_cpp, so GetOpSchema<> template specialisations are
//     unavailable.
//   - Instead, a faithful reconstruction of the Gemm-13 schema is built and
//     registered manually using the OpSchema / RegisterSchema API.  The
//     attribute types, input/output arities, and type constraints match the
//     real Gemm-13 definition exactly so that the original assertions hold.

#include "../defs/schema.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// Registers a faithful reconstruction of the Gemm opset-13 schema.
// Inputs, outputs, attributes, and type constraints mirror the real definition
// in onnx/defs/math/defs.cc so that the original op_reg_test.cc assertions
// hold without modification.
void RegisterGemm13() {
  OpSchema schema;
  schema.SetName("Gemm");
  schema.SetDomain(ONNX_DOMAIN);
  schema.SinceVersion(13);
  schema.SetDoc("General Matrix multiplication (Gemm).");
  schema.Input(0, "A", "Input tensor A.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
  schema.Input(1, "B", "Input tensor B.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
  schema.Input(2, "C", "Optional input tensor C.", "T", OpSchema::Optional, true, 1,
               OpSchema::Differentiable);
  schema.Output(0, "Y", "Output tensor of shape (M, N).", "T", OpSchema::Single, true, 1,
                OpSchema::Differentiable);
  schema.TypeConstraint(
      "T",
      {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(uint32)", "tensor(uint64)",
       "tensor(int32)", "tensor(int64)", "tensor(bfloat16)"},
      "Constrain input and output types to float/int tensors.");
  schema.Attr(std::string("transA"), std::string("Whether A should be transposed"),
              AttributeProto::INT, static_cast<int64_t>(0));
  schema.Attr(std::string("transB"), std::string("Whether B should be transposed"),
              AttributeProto::INT, static_cast<int64_t>(0));
  schema.Attr(std::string("alpha"),
              std::string("Scalar multiplier for the product of input tensors A * B."),
              AttributeProto::FLOAT, 1.0f);
  schema.Attr(std::string("beta"), std::string("Scalar multiplier for input tensor C."),
              AttributeProto::FLOAT, 1.0f);
  schema.Finalize();
  // opset_version_to_load == 0 means "register regardless of sinceVersion"
  RegisterSchema(std::move(schema), 0, /*fail_duplicate_schema=*/false);
}

void DeregisterGemm() { DeregisterSchema("Gemm", 13, ONNX_DOMAIN); }

} // namespace

// ---------------------------------------------------------------------------
// Test fixture – registers the required schema once for the suite.
// ---------------------------------------------------------------------------
class OpRegistrationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { RegisterGemm13(); }
  static void TearDownTestSuite() { DeregisterGemm(); }
};

// ---------------------------------------------------------------------------
// Translated tests
// ---------------------------------------------------------------------------

// Original: GemmOp
// Verifies that the Gemm schema has the expected number of inputs (3),
// that the input and output type sets agree, that there are 4 attributes,
// and that alpha and beta are FLOAT attributes.
TEST_F(OpRegistrationTest, GemmOp) {
  const auto* const opSchema = OpSchemaRegistry::Schema("Gemm");
  EXPECT_TRUE(nullptr != opSchema);

  size_t input_size = opSchema->inputs().size();
  EXPECT_EQ(input_size, 3u);

  EXPECT_EQ(opSchema->inputs()[0].GetTypes(), opSchema->outputs()[0].GetTypes());

  size_t attr_size = opSchema->attributes().size();
  EXPECT_EQ(attr_size, 4u);

  EXPECT_NE(opSchema->attributes().count("alpha"), 0u);
  EXPECT_EQ(opSchema->attributes().at("alpha").type, AttributeProto::FLOAT);
  EXPECT_NE(opSchema->attributes().count("beta"), 0u);
  EXPECT_EQ(opSchema->attributes().at("beta").type, AttributeProto::FLOAT);
}

} // namespace Test
