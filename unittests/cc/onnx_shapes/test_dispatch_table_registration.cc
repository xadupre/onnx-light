// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx.h"

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_shapes/dispatch_table.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// `core::shapes::DispatchTable()` starts out empty: `onnx_core` must not
// depend on `onnx_shapes`, so it cannot self-populate with the built-in
// `onnx_shapes` shape functions. Any test in this binary that exercises
// `core::shapes::InferShapesModel` / `ShapesContext::ComputeShapeNode`
// (rather than calling a `ComputeShape*` function directly) therefore needs
// `onnx_shapes::RegisterShapeFunctions()` to have run first. Registering it
// once here, via a global test environment, covers every test in
// `test_onnx_light` without having to call it from each test file
// individually.
class RegisterShapeFunctionsEnvironment : public ::testing::Environment {
public:
  void SetUp() override { ::onnx_light::onnx_shapes::RegisterShapeFunctions(); }
};

const ::testing::Environment *const kRegisterShapeFunctionsEnvironment =
    ::testing::AddGlobalTestEnvironment(new RegisterShapeFunctionsEnvironment());

} // namespace

TEST(OnnxOptimDispatchTable, RegisterShapeFunctionsPopulatesCoreDispatchTable) {
  ::onnx_light::onnx_shapes::RegisterShapeFunctions();
  const auto &table = ::onnx_light::core::shapes::DispatchTable();
  EXPECT_GT(table.size(), 200u);
  EXPECT_NE(table.find("ai.onnx:Abs"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Binarizer"), table.end());
}

} // namespace Test
