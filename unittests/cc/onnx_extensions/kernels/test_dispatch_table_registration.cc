// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// `core::runtime::KernelDispatchTable()` starts out empty: `onnx_core` must
// not depend on `onnx_kernels`, so it cannot self-populate with the
// built-in `onnx_kernels` operator kernels. Any test in this binary that
// exercises `core::runtime::RunNode` / `RuntimeSession` (rather than
// instantiating a kernel class directly) therefore needs
// `onnx_kernels::RegisterKernelFunctions()` to have run first. Registering
// it once here, via a global test environment, covers every test in
// `test_onnx_light` without having to call it from each test file
// individually.
class RegisterKernelFunctionsEnvironment : public ::testing::Environment {
public:
  void SetUp() override { ::onnx_light::onnx_kernels::RegisterKernelFunctions(); }
};

const ::testing::Environment *const kRegisterKernelFunctionsEnvironment =
    ::testing::AddGlobalTestEnvironment(new RegisterKernelFunctionsEnvironment());

} // namespace

TEST(OnnxKernelsDispatchTable, RegisterKernelFunctionsPopulatesCoreDispatchTable) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  const auto &table = ::onnx_light::core::runtime::KernelDispatchTable();
  EXPECT_GT(table.size(), 200u);
  EXPECT_NE(table.find("ai.onnx:Add"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Binarizer"), table.end());
}

} // namespace Test
