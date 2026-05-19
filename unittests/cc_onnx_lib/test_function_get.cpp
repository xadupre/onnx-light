// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/function_get_test.cc
// and adapted to work with onnx-light.
//
// Key differences from vanilla ONNX:
//   - Schemas are NOT auto-registered at startup in onnx-light.
//   - The operator-definition .cc files (nn/defs.cc, etc.) are not compiled
//     into lib_onnx_lib, so RegisterOnnxOperatorSetSchema() is unavailable.
//   - Instead, minimal MeanVarianceNormalization schemas are built and
//     registered manually using the OpSchema / RegisterSchema API.
//   - OpSchema::BuildFunction() (added to onnx-light) is called from
//     Finalize() to populate the FunctionProto's name/domain/inputs/outputs
//     fields, mirroring the vanilla ONNX behaviour.

#include "../common/constants.h"
#include "../defs/schema.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// Registers a minimal MeanVarianceNormalization schema at the given opset
// version with an empty static function body.  The function body name is
// populated automatically when OpSchema::Finalize() calls BuildFunction().
void RegisterMVN(int sinceVersion) {
  OpSchema schema;
  schema.SetName("MeanVarianceNormalization");
  schema.SetDomain(ONNX_DOMAIN);
  schema.SinceVersion(sinceVersion);
  schema.SetDoc("Minimal MeanVarianceNormalization for function-get tests.");
  schema.Input(0, "X", "Input tensor", "T");
  schema.Output(0, "Y", "Output tensor", "T");
  schema.TypeConstraint("T",
                        {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                        "Constrain to numeric tensors.");
  // Register a static function body (empty node list).  BuildFunction() is
  // called inside Finalize(), which sets the FunctionProto name/domain etc.
  schema.FunctionBody(std::vector<NodeProto>{}, sinceVersion);
  RegisterSchema(std::move(schema), 0, /*fail_duplicate_schema=*/false);
}

// Deregisters all MeanVarianceNormalization schemas registered by this suite.
void DeregisterMVN() {
  for (int ver : {9, 13}) {
    DeregisterSchema("MeanVarianceNormalization", ver, ONNX_DOMAIN);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture – registers the required schemas once for the suite.
// ---------------------------------------------------------------------------
class FunctionAPITest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    RegisterMVN(9);
    RegisterMVN(13);
  }
  static void TearDownTestSuite() { DeregisterMVN(); }
};

// ---------------------------------------------------------------------------
// Translated tests
// ---------------------------------------------------------------------------

// Original: GetFunctionOpWithVersion
// Verifies that MeanVarianceNormalization at opset 9 has a function body and
// that the body carries the correct operator name.
TEST_F(FunctionAPITest, GetFunctionOpWithVersion) {
  const auto *const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, ONNX_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_TRUE(schema->HasFunction());
  const auto *const func = schema->GetFunction();
  ASSERT_TRUE(func);
  EXPECT_EQ(func->ref_name(), "MeanVarianceNormalization");
}

// Original: GetMeanVarianceNormalizationFunctionWithVersion
// Verifies lookup at several opset versions.  Versions 17 and 18 have no
// dedicated schema; the lookup falls back to the version-13 schema
// (the latest sinceVersion <= 17/18).
TEST_F(FunctionAPITest, GetMeanVarianceNormalizationFunctionWithVersion) {
  {
    const auto *const schema =
        OpSchemaRegistry::Schema("MeanVarianceNormalization", 13, ONNX_DOMAIN);
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto *const func = schema->GetFunction();
    ASSERT_TRUE(func);
    EXPECT_EQ(func->ref_name(), "MeanVarianceNormalization");
  }
  {
    // Version 17 has no separate definition; lookup returns the schema for
    // version 13 (the latest sinceVersion <= 17).
    const auto *const schema =
        OpSchemaRegistry::Schema("MeanVarianceNormalization", 17, ONNX_DOMAIN);
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto *const func = schema->GetFunction();
    ASSERT_TRUE(func);
    EXPECT_EQ(func->ref_name(), "MeanVarianceNormalization");
  }
  {
    // Version 18 has no separate definition; lookup returns the schema for
    // version 13 (the latest sinceVersion <= 18).
    const auto *const schema =
        OpSchemaRegistry::Schema("MeanVarianceNormalization", 18, ONNX_DOMAIN);
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto *const func = schema->GetFunction();
    ASSERT_TRUE(func);
    EXPECT_EQ(func->ref_name(), "MeanVarianceNormalization");
  }
}

} // namespace Test
