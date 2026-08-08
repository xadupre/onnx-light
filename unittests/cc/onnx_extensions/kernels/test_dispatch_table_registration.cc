// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

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

// The device is part of a kernel's identifier: a factory registered for a GPU
// device is keyed separately from the CPU/default entry so both can coexist.
TEST(OnnxKernelsDispatchTable, DeviceIsPartOfIdentifier) {
  const core::symbolic::Device gpu = core::symbolic::MakeGPUDevice(0);
  core::runtime::RegisterKernelFn(
      "test.onnxlight.device_kernel", "DeviceOp", gpu,
      [](const NodeProto &, core::runtime::RuntimeContext &)
          -> std::unique_ptr<core::runtime::KernelBase> { return nullptr; });

  const auto &table = core::runtime::KernelDispatchTable();
  EXPECT_EQ(table.find("test.onnxlight.device_kernel:DeviceOp"), table.end());
  EXPECT_NE(table.find("test.onnxlight.device_kernel:DeviceOp:" +
                       std::to_string(static_cast<int32_t>(gpu))),
            table.end());
}

// `RegisterKernelFn` with `overwrite=false` must keep an existing entry and
// report that it did not store the new factory, whereas the default
// (`overwrite=true`) replaces it. This is the primitive that makes a custom
// kernel override survive the built-in bulk registration
// (`RegisterKernelFunctions`) regardless of the order in which the two run:
// the bulk registration uses `overwrite=false`, so it never clobbers a
// previously registered override.
TEST(OnnxKernelsDispatchTable, RegisterKernelFnOverwriteFlagControlsReplacement) {
  const std::string domain = "test.onnxlight.overwrite_flag";
  const std::string key = domain + ":OverwriteOp";
  static int which = 0;
  which = 0;

  core::runtime::RuntimeContext ctx(core::runtime::KernelContext(core::runtime::DefaultOpset(18)));
  const NodeProto node;

  const bool stored_first = core::runtime::RegisterKernelFn(
      domain, "OverwriteOp", core::symbolic::Device::kCPU,
      [&which](const NodeProto &,
               core::runtime::RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        which = 1;
        return nullptr;
      });
  EXPECT_TRUE(stored_first);

  const auto &table = core::runtime::KernelDispatchTable();
  auto it = table.find(key);
  ASSERT_NE(it, table.end());

  // overwrite=false: the existing entry is kept and the call reports false.
  const bool stored_if_absent = core::runtime::RegisterKernelFn(
      domain, "OverwriteOp", core::symbolic::Device::kCPU,
      [&which](const NodeProto &,
               core::runtime::RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        which = 2;
        return nullptr;
      },
      /*overwrite=*/false);
  EXPECT_FALSE(stored_if_absent);
  which = 0;
  table.find(key)->second(node, ctx);
  EXPECT_EQ(which, 1) << "overwrite=false must keep the first factory.";

  // overwrite=true (default): the entry is replaced and the call reports true.
  const bool stored_overwrite = core::runtime::RegisterKernelFn(
      domain, "OverwriteOp", core::symbolic::Device::kCPU,
      [&which](const NodeProto &,
               core::runtime::RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        which = 3;
        return nullptr;
      });
  EXPECT_TRUE(stored_overwrite);
  which = 0;
  table.find(key)->second(node, ctx);
  EXPECT_EQ(which, 3) << "overwrite=true must replace the factory.";
}

} // namespace Test
