// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/schema_registration_test.cc and adapted to
// work with onnx-light. onnx-light does not link the full GetOpSchema<>
// specialization library into lib_onnx_cpp, so these tests exercise the same
// registration semantics with a representative subset of hand-built schemas.

#include "../defs/operator_sets.h"
#include "../defs/schema.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

void RegisterTestOp(const char *name, int sinceVersion, int opset_version_to_load = 0,
                    bool fail_duplicate_schema = true, const char *domain = ONNX_DOMAIN) {
  auto schema = OpSchema();
  schema.SetName(name);
  schema.SetDomain(domain);
  schema.SinceVersion(sinceVersion);
  schema.SetDoc("test");
  schema.Finalize();
  RegisterSchema(std::move(schema), opset_version_to_load, fail_duplicate_schema);
}

void RegisterOnnxOperatorSetSchema() {
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(0);
}

void RegisterOnnxOperatorSetSchema(int target_version, bool fail_duplicate_schema = true) {
  if (target_version == 0) {
    RegisterTestOp("Acos", 7, 0, fail_duplicate_schema);
    RegisterTestOp("Add", 1, 0, fail_duplicate_schema);
    RegisterTestOp("Add", 6, 0, fail_duplicate_schema);
    RegisterTestOp("Add", 7, 0, fail_duplicate_schema);
    RegisterTestOp("Add", 13, 0, fail_duplicate_schema);
    RegisterTestOp("Add", 14, 0, fail_duplicate_schema);
    RegisterTestOp("Trilu", 14, 0, fail_duplicate_schema);
  } else {
    RegisterTestOp("Trilu", 14, target_version, fail_duplicate_schema);
    RegisterTestOp("Add", 14, target_version, fail_duplicate_schema);
    RegisterTestOp("Add", 13, target_version, fail_duplicate_schema);
    RegisterTestOp("Acos", 7, target_version, fail_duplicate_schema);
    RegisterTestOp("Add", 7, target_version, fail_duplicate_schema);
    RegisterTestOp("Add", 6, target_version, fail_duplicate_schema);
    RegisterTestOp("Add", 1, target_version, fail_duplicate_schema);
  }
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(target_version);
}

void DeregisterOnnxOperatorSetSchema() {
  OpSchemaRegistry::Instance()->OpSchemaDeregisterAll(ONNX_DOMAIN);
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(-1);
}

} // namespace

// Fixture that snapshots the real ai.onnx schema registry before each test and
// restores it afterwards. These tests deliberately deregister every ai.onnx
// schema and repopulate the registry with a small set of hand-built stubs; if
// that state leaked into later tests running in the same process (e.g. a
// monolithic ``./test_onnx_light`` run rather than the per-test ctest
// processes), those tests would observe a registry missing the real operator
// schemas and fail. Restoring the snapshot keeps each test hermetic.
class SchemaRegistrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure the one-time lazy static registration of the real ONNX schemas has
    // completed, then snapshot the ai.onnx-domain schemas.
    RegisterAllOnnxOperatorSchemas();
    saved_schemas_.clear();
    for (const auto &schema : OpSchemaRegistry::get_all_schemas_with_history()) {
      if (schema.domain() == ONNX_DOMAIN) {
        saved_schemas_.push_back(schema);
      }
    }
    saved_loaded_version_ = OpSchemaRegistry::Instance()->GetLoadedSchemaVersion();
  }

  void TearDown() override {
    OpSchemaRegistry::Instance()->OpSchemaDeregisterAll(ONNX_DOMAIN);
    for (const auto &schema : saved_schemas_) {
      RegisterSchema(schema, 0, false);
    }
    OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(saved_loaded_version_);
  }

  std::vector<OpSchema> saved_schemas_;
  int saved_loaded_version_ = 0;
};

TEST_F(SchemaRegistrationTest, DisabledOnnxStaticRegistrationAPICall) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  EXPECT_TRUE(IsOnnxStaticRegistrationDisabled());
#else
  EXPECT_FALSE(IsOnnxStaticRegistrationDisabled());
#endif
}

TEST_F(SchemaRegistrationTest, RegisterAllByDefaultAndManipulateSchema) {
  DeregisterOnnxOperatorSetSchema();

  RegisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));

  RegisterOnnxOperatorSetSchema();
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));

  DeregisterOnnxOperatorSetSchema();
}

TEST_F(SchemaRegistrationTest, RegisterAndDeregisterAllOpsetSchemaVersion) {
  // Force the one-time lazy static registration to complete before the initial
  // deregister; otherwise the first Schema() lookup below would trigger it and
  // repopulate the registry mid-test (fails when this test runs in isolation).
  RegisterAllOnnxOperatorSchemas();
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));

  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  auto schema = OpSchemaRegistry::Schema("Acos", 7);
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 7);

  schema = OpSchemaRegistry::Schema("Add", 14);
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 14);

  schema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 14);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));
}

TEST_F(SchemaRegistrationTest, RegisterSpecifiedOpsetSchemaVersion) {
  // Force the one-time lazy static registration to complete before the initial
  // deregister; otherwise the first Schema() lookup below would trigger it and
  // repopulate the registry mid-test (fails when this test runs in isolation).
  RegisterAllOnnxOperatorSchemas();
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  RegisterOnnxOperatorSetSchema(13);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  opSchema = OpSchemaRegistry::Schema("Add", 12);
  EXPECT_EQ(nullptr, opSchema);

  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  DeregisterOnnxOperatorSetSchema();
}

TEST_F(SchemaRegistrationTest, RegisterMultipleOpsetSchemaVersions_UpgradeVersion) {
  // Force the one-time lazy static registration to complete before the initial
  // deregister; otherwise the first Schema() lookup below would trigger it and
  // repopulate the registry mid-test (fails when this test runs in isolation).
  RegisterAllOnnxOperatorSchemas();
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  RegisterOnnxOperatorSetSchema(14, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);

  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  DeregisterOnnxOperatorSetSchema();
}

TEST_F(SchemaRegistrationTest, RegisterMultipleOpsetSchemaVersions_DowngradeVersion) {
  // Force the one-time lazy static registration to complete before the initial
  // deregister; otherwise the first Schema() lookup below would trigger it and
  // repopulate the registry mid-test (fails when this test runs in isolation).
  RegisterAllOnnxOperatorSchemas();
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  RegisterOnnxOperatorSetSchema(14);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);

  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  DeregisterOnnxOperatorSetSchema();
}

TEST_F(SchemaRegistrationTest, RegisterSpecificThenAllVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  RegisterOnnxOperatorSetSchema(0, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterOnnxOperatorSetSchema();
}

TEST_F(SchemaRegistrationTest, RegisterAllThenSpecificVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterOnnxOperatorSetSchema();
}

} // namespace Test
