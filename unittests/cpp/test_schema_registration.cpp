// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from onnx/test/cpp/schema_registration_test.cc
// and adapted to work with onnx-light.
//
// Key differences from vanilla ONNX:
//   - Schemas are NOT auto-registered at startup in onnx-light.
//   - OpSchemaRegistry::GetLoadedSchemaVersion() starts at -1 (not 0).
//   - IsOnnxStaticRegistrationDisabled() returns false (macro not defined).
//   - onnx-light does not ship the full GetOpSchema<> specialisation library,
//     so RegisterOnnxOperatorSetSchema() is not called in these tests; instead
//     the tests exercise the same register/deregister API with manually built
//     OpSchema objects — the same API exercised by the original tests.

#include "../defs/schema.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

// ---------------------------------------------------------------------------
// Helper: build a minimal OpSchema and register it.
// ---------------------------------------------------------------------------
namespace {

// Register a trivial single-version operator.
// opset_version_to_load mirrors the vanilla ONNX RegisterOnnxOperatorSetSchema(N) semantics:
// when > 0, only registers schemas whose sinceVersion == opset_version_to_load.
// when == 0, all schemas are registered regardless of their sinceVersion.
void RegisterTestOp(const char *name, int sinceVersion,
                    int opset_version_to_load = 0,
                    const char *domain = ONNX_DOMAIN) {
  auto schema = OpSchema();
  schema.SetName(name);
  schema.SetDomain(domain);
  schema.SinceVersion(sinceVersion);
  schema.SetDoc("test");
  schema.Finalize();
  RegisterSchema(schema, opset_version_to_load, /*fail_duplicate_schema=*/false);
}

// Deregister all test operators registered above.
void DeregisterTestOps() {
  OpSchemaRegistry::Instance()->OpSchemaDeregisterAll(ONNX_DOMAIN);
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(-1);
}

} // namespace

// ---------------------------------------------------------------------------
// Translated tests
// ---------------------------------------------------------------------------

// Original: DisabledOnnxStaticRegistrationAPICall
// Verifies the static-registration query API.
// In onnx-light the macro __ONNX_DISABLE_STATIC_REGISTRATION is NOT defined,
// so IsOnnxStaticRegistrationDisabled() returns false.
TEST(onnx_schema_registration, DisabledOnnxStaticRegistrationAPICall) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  EXPECT_TRUE(IsOnnxStaticRegistrationDisabled());
#else
  EXPECT_FALSE(IsOnnxStaticRegistrationDisabled());
#endif
}

// Original: RegisterAllByDefaultAndManipulateSchema
// In vanilla ONNX all schemas are registered at startup (loaded_schema_version
// == 0) and the test deregisters / re-registers them.
// In onnx-light no schemas are loaded at startup (loaded_schema_version == -1);
// the test explicitly registers a few representative operators and verifies the
// same register / deregister cycle.
TEST(onnx_schema_registration, RegisterAllByDefaultAndManipulateSchema) {
  // onnx-light: registry is empty at start.
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find any op before registration.
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));

  // Register a representative set of ops across several versions.
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);

  // Should find schema for all registered versions.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Deregister everything.
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find any op after deregistration.
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));

  // Re-register.
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));

  // Clean up.
  DeregisterTestOps();
}

// Original: RegisterAndDeregisterAllOpsetSchemaVersion
// Tests the full register / deregister cycle and per-version lookup.
TEST(onnx_schema_registration, RegisterAndDeregisterAllOpsetSchemaVersion) {
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Register a representative set of ops.
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);

  // Verify lookup by exact version.
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

  // Deregister.
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));
}

// Original: RegisterSpecifiedOpsetSchemaVersion
// Mirrors vanilla ONNX RegisterOnnxOperatorSetSchema(13): only schemas with
// sinceVersion == 13 are registered (the same filter opset_version_to_load applies).
TEST(onnx_schema_registration, RegisterSpecifiedOpsetSchemaVersion) {
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Use opset_version_to_load=13: only ops whose sinceVersion==13 are registered.
  // Add has versions 1,6,7,13. With opset_version_to_load=13, only Add-13 is stored.
  // Acos has version 7. With opset_version_to_load=13, sinceVersion(7) != 13, so not stored.
  RegisterTestOp("Add", 1,  13);
  RegisterTestOp("Add", 6,  13);
  RegisterTestOp("Add", 7,  13);
  RegisterTestOp("Add", 13, 13);
  RegisterTestOp("Acos", 7, 13);
  // Intentionally do NOT register Trilu-14 (introduced in opset 14).
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  // Add-13 has sinceVersion=13. Nothing found for maxInclusiveVersion=12 (13 > 12).
  opSchema = OpSchemaRegistry::Schema("Add", 12);
  EXPECT_EQ(nullptr, opSchema);

  // Trilu was not registered.
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  // Acos-7 was filtered out by opset_version_to_load=13 (sinceVersion 7 != 13),
  // so nothing is found.
  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  EXPECT_EQ(nullptr, opSchema);

  DeregisterTestOps();
}

// Original: RegisterMultipleOpsetSchemaVersions_UpgradeVersion
// Registers opset 11 ops then opset 14 ops (upgrade direction).
TEST(onnx_schema_registration, RegisterMultipleOpsetSchemaVersions_UpgradeVersion) {
  DeregisterTestOps();

  // Register opset-11 subset.
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);

  // Register opset-14 additions.
  RegisterTestOp("Add", 13);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);

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
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  DeregisterTestOps();
}

// Original: RegisterMultipleOpsetSchemaVersions_DowngradeVersion
// Registers opset 14 ops then opset 11 ops (downgrade direction).
TEST(onnx_schema_registration, RegisterMultipleOpsetSchemaVersions_DowngradeVersion) {
  DeregisterTestOps();

  // Register opset-14 subset first.
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);

  // Then register older versions.
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);

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
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  DeregisterTestOps();
}

// Original: RegisterSpecificThenAllVersion
// Registers a subset then a superset; verifies union is available.
TEST(onnx_schema_registration, RegisterSpecificThenAllVersion) {
  DeregisterTestOps();

  // Register opset-11 subset.
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);

  // Register all versions (superset).
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 13);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterTestOps();
}

// Original: RegisterAllThenSpecificVersion
// Registers all versions then a specific subset; verifies union is available.
TEST(onnx_schema_registration, RegisterAllThenSpecificVersion) {
  DeregisterTestOps();

  // Register all versions first.
  RegisterTestOp("Acos", 7);
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);
  RegisterTestOp("Add", 14);
  RegisterTestOp("Trilu", 14);

  // Register opset-11 subset again (should not fail).
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  DeregisterTestOps();
}
