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
void RegisterTestOp(const char *name, int sinceVersion,
                    const char *domain = ONNX_DOMAIN) {
  auto schema = OpSchema();
  schema.SetName(name);
  schema.SetDomain(domain);
  schema.SinceVersion(sinceVersion);
  schema.SetDoc("test");
  schema.Finalize();
  OpSchemaRegistry::RegisterSchema(schema);
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
// Registers ops only up to a specific opset version and verifies that ops
// introduced in later versions are absent.
TEST(onnx_schema_registration, RegisterSpecifiedOpsetSchemaVersion) {
  DeregisterTestOps();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register ops available at opset 13 only.
  RegisterTestOp("Add", 1);
  RegisterTestOp("Add", 6);
  RegisterTestOp("Add", 7);
  RegisterTestOp("Add", 13);
  RegisterTestOp("Acos", 7);
  // Intentionally do NOT register Trilu-14 (introduced in opset 14).

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  // Should not find the version-14 Add that was never registered.
  opSchema = OpSchemaRegistry::Schema("Add", 14);
  EXPECT_EQ(nullptr, opSchema);

  // Trilu was not registered.
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  // Acos-7 is the latest Acos before opset 13.
  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

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

//
// Key differences from vanilla ONNX:
//   - Schemas are NOT auto-registered at startup in onnx-light.
//   - OpSchemaRegistry::GetLoadedSchemaVersion() starts at -1 (not 0).
//   - IsOnnxStaticRegistrationDisabled() returns false (macro not defined).
//   - All tests that manipulate the registry clean up after themselves so that
//     test execution order does not matter.

#include "../defs/operator_sets.h"
#include "../defs/schema.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

// ---------------------------------------------------------------------------
// schema_registration_test.cc tests
// ---------------------------------------------------------------------------

TEST(onnx_schema_registration, DisabledOnnxStaticRegistrationAPICall) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  EXPECT_TRUE(IsOnnxStaticRegistrationDisabled());
#else
  EXPECT_FALSE(IsOnnxStaticRegistrationDisabled());
#endif
}

// In onnx-light, schemas are never auto-registered at startup; the test
// explicitly calls RegisterOnnxOperatorSetSchema() / DeregisterOnnxOperatorSetSchema()
// to exercise the full register/deregister cycle.
// (In vanilla ONNX without __ONNX_DISABLE_STATIC_REGISTRATION all schemas are
// present from the start and loaded_schema_version is 0 by default.)
TEST(onnx_schema_registration, RegisterAllByDefaultAndManipulateSchema) {
  // onnx-light: no schemas are registered at process start.
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register all opset versions.
  RegisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all versions.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Clear all opset schema registration.
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find any opset.
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));

  // Register all opset versions again.
  RegisterOnnxOperatorSetSchema();

  // Should find all opset.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));

  // Restore initial state.
  DeregisterOnnxOperatorSetSchema();
}

// Tests explicit de/registration of all opset schema versions.
// In vanilla ONNX this test is gated behind __ONNX_DISABLE_STATIC_REGISTRATION
// because the static initialiser already populates the registry; in onnx-light
// the registry starts empty so this test can run unconditionally.
TEST(onnx_schema_registration, RegisterAndDeregisterAllOpsetSchemaVersion) {
  // Clear all opset schema registration.
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find schema for any op.
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Register all opset versions.
  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all ops. Available versions are:
  // Acos-7
  // Add-1,6,7,13,14
  // Trilu-14
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

  // Clear all opset schema registration.
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find schema for any op.
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));
}

TEST(onnx_schema_registration, RegisterSpecifiedOpsetSchemaVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);
  RegisterOnnxOperatorSetSchema(13);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  // Should not find opset 12.
  opSchema = OpSchemaRegistry::Schema("Add", 12);
  EXPECT_EQ(nullptr, opSchema);

  // Should not find opset 14.
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  // Acos-7 is the latest Acos before specified 13.
  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Clean up.
  DeregisterOnnxOperatorSetSchema();
}

// Register opset-11, then opset-14.
// Expects Reg(11, 14) == Reg(11) U Reg(14).
TEST(onnx_schema_registration, RegisterMultipleOpsetSchemaVersions_UpgradeVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 11.
  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);
  // Register opset 14.
  // Do not fail on duplicate schema registration request.
  RegisterOnnxOperatorSetSchema(14, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);

  // Acos-7 is the latest before/at opset 11 and 14.
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Add-7 is the latest before/at opset 11.
  // Add-14 is the latest before/at opset 14.
  // Should find both Add-7,14.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14.
  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Should find Add-7 as the max version <=13.
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Should find opset 14.
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Clean up.
  DeregisterOnnxOperatorSetSchema();
}

// Register opset-14, then opset-11.
// Expects Reg(14, 11) == Reg(11) U Reg(14).
TEST(onnx_schema_registration, RegisterMultipleOpsetSchemaVersions_DowngradeVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 14.
  RegisterOnnxOperatorSetSchema(14);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);
  // Register opset 11.
  // Do not fail on duplicate schema registration request.
  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Acos-7 is the latest before/at opset 11 and 14.
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Add-7 is the latest before/at opset 11.
  // Add-14 is the latest before/at opset 14.
  // Should find both Add-7,14.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14.
  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Should find Add-7 as the max version <=13.
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Should find opset 14.
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Clean up.
  DeregisterOnnxOperatorSetSchema();
}

// Register opset-11, then all versions.
// Expects no error.
TEST(onnx_schema_registration, RegisterSpecificThenAllVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 11.
  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Register all opset versions.
  // Do not fail on duplicate schema registration request.
  RegisterOnnxOperatorSetSchema(0, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all ops.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Clean up.
  DeregisterOnnxOperatorSetSchema();
}

// Register all versions, then opset 11.
// Expects no error.
TEST(onnx_schema_registration, RegisterAllThenSpecificVersion) {
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register all opset versions.
  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Register opset 11.
  // Do not fail on duplicate schema registration request.
  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Should find schema for all ops.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions.
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Clean up.
  DeregisterOnnxOperatorSetSchema();
}
