// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/tuning/portable_parallel_tuning.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

#include <array>
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

TEST(OnnxKernelsDispatchTable, ConstantFoldingRunsRegisteredIdentityFactory) {
  struct RestoreIdentityFactory {
    core::runtime::NodeKernelFn original =
        core::runtime::KernelDispatchTable().at("ai.onnx:Identity");
    ~RestoreIdentityFactory() {
      core::runtime::RegisterKernelFn("", "Identity", core::symbolic::Device::kCPU,
                                      std::move(original));
    }
  } restore;

  for (const std::string mode : {"transform", "copy", "forward"}) {
    SCOPED_TRACE(mode);
    int factory_calls = 0;
    int runs = 0;
    auto factory = core::runtime::MakeCustomKernelFactory(
        [&, mode](const NodeProto &node, core::runtime::RuntimeContext &rt) {
          ++runs;
          const auto &input = rt.Get(node.input(0));
          EXPECT_NE(input.borrowed_owner().use_count(), 0);
          if (mode == "forward") {
            rt.Put(node.output(0), input.BorrowView());
          } else {
            const float offset = mode == "transform" ? 10.0f : 0.0f;
            rt.Put(node.output(0), core::runtime::Tensor::FromFloat(
                                       node.output(0), input.shape,
                                       {input.AsFloat()[0] + offset, input.AsFloat()[1] + offset}));
          }
        });
    core::runtime::RegisterKernelFn("", "Identity", core::symbolic::Device::kCPU,
                                    [&, factory](const NodeProto &node,
                                                 core::runtime::RuntimeContext &rt,
                                                 const core::runtime::KernelContext &ctx) {
                                      ++factory_calls;
                                      EXPECT_EQ(rt.device(), core::symbolic::Device::kCPU);
                                      return factory(node, rt, ctx);
                                    });

    core::builder::GraphBuilder builder(
        "identity_factory",
        [](const std::string &op) { return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false); },
        {{"", 22}});
    builder.set_device(core::symbolic::Device::kCPU);
    const std::array<float, 2> values{1.0f, 2.0f};
    TensorProto weight;
    weight.set_name("weight");
    weight.set_data_type(TensorProto::DataType::FLOAT);
    weight.add_dims(2);
    weight.set_raw_data(values.data(), sizeof(values));
    builder.MakeInitializer(std::move(weight));
    const auto *original_bytes = builder.Initializers()[0].ref_raw_data().data();
    builder.MakeNode("Identity", {"weight"}, {"Y"});
    builder.MakeOutput("Y");

    ASSERT_EQ(builder.ConstantFold(), 1u);
    EXPECT_EQ(factory_calls, 1);
    EXPECT_EQ(runs, 1);
    ASSERT_EQ(builder.Initializers().size(), 2u);
    const auto &folded = builder.Initializers()[1];
    const auto actual = core::runtime::TensorFromProto(folded);
    EXPECT_FLOAT_EQ(actual.AsFloat()[0], mode == "transform" ? 11.0f : 1.0f);
    EXPECT_FLOAT_EQ(actual.AsFloat()[1], mode == "transform" ? 12.0f : 2.0f);
    EXPECT_EQ(folded.ref_raw_data().data() == original_bytes, mode == "forward");
  }
}

TEST(OnnxKernelsDispatchTable, RegistersAbsCalibrationFunctions) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  const core::runtime::KernelTuningKey key = onnx_kernels::tuning::MakePortableTuningKey(
      "Abs", static_cast<int32_t>(TensorProto::DataType::FLOAT), 2);
  core::runtime::KernelCalibrationFunction calibrate =
      core::runtime::GetKernelTuningRegistry().FindCalibrationFunction(key);
  ASSERT_TRUE(calibrate);

  core::runtime::CpuExecutionDescriptor execution{
      core::platform::GetCpuDescriptor(),
      static_cast<uint32_t>(core::runtime::ParallelForThreadCount())};
  core::runtime::CalibrationOptions options;
  options.maximum_duration_ms = 1;
  options.maximum_memory_bytes = 1 << 20;
  core::runtime::CalibrationReporter reporter;
  const core::runtime::KernelTuningParameters parameters =
      calibrate(key, execution, options, reporter);

  EXPECT_EQ(parameters.key, key);
  EXPECT_GT(parameters.Get<int64_t>(onnx_kernels::tuning::kParallelMinimumElements), 0);
  EXPECT_FALSE(reporter.diagnostics().empty());
}

TEST(OnnxKernelsDispatchTable, RegistersNotCalibrationFunction) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  const core::runtime::KernelTuningKey key = onnx_kernels::tuning::MakePortableTuningKey(
      "Not", static_cast<int32_t>(TensorProto::DataType::BOOL));
  core::runtime::KernelCalibrationFunction calibrate =
      core::runtime::GetKernelTuningRegistry().FindCalibrationFunction(key);
  ASSERT_TRUE(calibrate);

  core::runtime::CpuExecutionDescriptor execution{
      core::platform::GetCpuDescriptor(),
      static_cast<uint32_t>(core::runtime::ParallelForThreadCount())};
  core::runtime::CalibrationOptions options;
  options.maximum_duration_ms = 1;
  options.maximum_memory_bytes = 1 << 20;
  core::runtime::CalibrationReporter reporter;
  const core::runtime::KernelTuningParameters parameters =
      calibrate(key, execution, options, reporter);

  EXPECT_EQ(parameters.key, key);
  EXPECT_GT(parameters.Get<int64_t>(onnx_kernels::tuning::kParallelMinimumElements), 0);
  EXPECT_FALSE(reporter.diagnostics().empty());
}

TEST(OnnxKernelsDispatchTable, RegistersBroadcastingAddCalibrationFunctions) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  const core::runtime::KernelTuningKey key = onnx_kernels::tuning::MakePortableTuningKey(
      "Add", static_cast<int32_t>(TensorProto::DataType::UINT32));
  core::runtime::KernelCalibrationFunction calibrate =
      core::runtime::GetKernelTuningRegistry().FindCalibrationFunction(key);
  ASSERT_TRUE(calibrate);

  core::runtime::CpuExecutionDescriptor execution{
      core::platform::GetCpuDescriptor(),
      static_cast<uint32_t>(core::runtime::ParallelForThreadCount())};
  core::runtime::CalibrationOptions options;
  options.maximum_duration_ms = 100;
  options.maximum_memory_bytes = 1 << 20;
  core::runtime::CalibrationReporter reporter;
  const core::runtime::KernelTuningParameters parameters =
      calibrate(key, execution, options, reporter);

  EXPECT_EQ(parameters.key, key);
  EXPECT_GT(parameters.Get<int64_t>(onnx_kernels::tuning::kParallelMinimumElements), 0);
  EXPECT_FALSE(reporter.diagnostics().empty());
  if (core::runtime::ParallelForThreadCount() > 1) {
    EXPECT_GE(reporter.benchmark_cases(), 3u);
    EXPECT_GT(reporter.peak_memory_bytes(), 0u);
    EXPECT_GT(reporter.measured_duration_ns(), 0u);
  }
}

TEST(OnnxKernelsDispatchTable, RegistersEveryBinaryTuningSchema) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  struct ExpectedSchema {
    const char *kernel;
    int32_t element_type;
  };
  constexpr std::array expected{
      ExpectedSchema{"Add", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Sub", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Mul", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Div", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Mod", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Pow", TensorProto::DataType::FLOAT},
      ExpectedSchema{"PRelu", TensorProto::DataType::FLOAT},
      ExpectedSchema{"SwiGLU", TensorProto::DataType::FLOAT},
      ExpectedSchema{"And", TensorProto::DataType::BOOL},
      ExpectedSchema{"Or", TensorProto::DataType::BOOL},
      ExpectedSchema{"Xor", TensorProto::DataType::BOOL},
      ExpectedSchema{"Equal", TensorProto::DataType::STRING},
      ExpectedSchema{"Greater", TensorProto::DataType::FLOAT},
      ExpectedSchema{"GreaterOrEqual", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Less", TensorProto::DataType::FLOAT},
      ExpectedSchema{"LessOrEqual", TensorProto::DataType::FLOAT},
      ExpectedSchema{"MatMul", TensorProto::DataType::FLOAT},
      ExpectedSchema{"BitwiseAnd", TensorProto::DataType::INT32},
      ExpectedSchema{"BitwiseOr", TensorProto::DataType::INT32},
      ExpectedSchema{"BitwiseXor", TensorProto::DataType::INT32},
      ExpectedSchema{"BitShift", TensorProto::DataType::UINT32},
      ExpectedSchema{"StringConcat", TensorProto::DataType::STRING},
  };

  for (const ExpectedSchema &expected_schema : expected) {
    const core::runtime::KernelTuningKey key = onnx_kernels::tuning::MakePortableTuningKey(
        expected_schema.kernel, expected_schema.element_type);
    const std::shared_ptr<const core::runtime::KernelTuningSchema> schema =
        core::runtime::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr) << expected_schema.kernel;
    EXPECT_EQ(
        schema->portable_defaults().Get<int64_t>(onnx_kernels::tuning::kParallelMinimumElements),
        core::runtime::kParallelForGrainSize)
        << expected_schema.kernel;
  }
}

TEST(OnnxKernelsDispatchTable, RegistersFirstWaveParallelTuningSchemas) {
  ::onnx_light::onnx_kernels::RegisterKernelFunctions();
  struct ExpectedSchema {
    const char *kernel;
    int32_t element_type;
  };
  constexpr std::array expected{
      ExpectedSchema{"Attention", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Attention", TensorProto::DataType::FLOAT16},
      ExpectedSchema{"Attention", TensorProto::DataType::BFLOAT16},
      ExpectedSchema{"Conv", TensorProto::DataType::FLOAT},
      ExpectedSchema{"Conv", TensorProto::DataType::FLOAT16},
      ExpectedSchema{"Conv", TensorProto::DataType::BFLOAT16},
      ExpectedSchema{"ConvInteger", TensorProto::DataType::INT8},
      ExpectedSchema{"ConvInteger", TensorProto::DataType::UINT8},
      ExpectedSchema{"ConvTranspose", TensorProto::DataType::FLOAT},
      ExpectedSchema{"FlexAttention", TensorProto::DataType::FLOAT},
      ExpectedSchema{"FlexAttention", TensorProto::DataType::DOUBLE},
      ExpectedSchema{"FlexAttention", TensorProto::DataType::FLOAT16},
      ExpectedSchema{"FlexAttention", TensorProto::DataType::BFLOAT16},
      ExpectedSchema{"LinearAttention", TensorProto::DataType::FLOAT},
      ExpectedSchema{"LinearAttention", TensorProto::DataType::FLOAT16},
      ExpectedSchema{"LinearAttention", TensorProto::DataType::BFLOAT16},
      ExpectedSchema{"QLinearConv", TensorProto::DataType::INT8},
      ExpectedSchema{"QLinearConv", TensorProto::DataType::UINT8},
  };

  for (const ExpectedSchema &expected_schema : expected) {
    const core::runtime::KernelTuningKey key = onnx_kernels::tuning::MakePortableTuningKey(
        expected_schema.kernel, expected_schema.element_type);
    const std::shared_ptr<const core::runtime::KernelTuningSchema> schema =
        core::runtime::GetKernelTuningRegistry().FindSchema(key);
    ASSERT_NE(schema, nullptr) << expected_schema.kernel;
    EXPECT_EQ(
        schema->portable_defaults().Get<int64_t>(onnx_kernels::tuning::kParallelMinimumElements),
        core::runtime::kParallelForGrainSize)
        << expected_schema.kernel;
  }
}

// The device is part of a kernel's identifier: a factory registered for a GPU
// device is keyed separately from the CPU/default entry so both can coexist.
TEST(OnnxKernelsDispatchTable, DeviceIsPartOfIdentifier) {
  const core::symbolic::Device gpu = core::symbolic::MakeGPUDevice(0);
  core::runtime::RegisterKernelFn(
      "test.onnxlight.device_kernel", "DeviceOp", gpu,
      [](const NodeProto &, core::runtime::RuntimeContext &, const core::runtime::KernelContext &)
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
  // Static: the factories below are stored in the process-global dispatch table
  // and must not capture a reference to a local that dies with the test.
  static int which = 0;
  which = 0;

  core::runtime::RuntimeContext ctx(core::runtime::KernelContext(core::runtime::DefaultOpset(18)));
  const NodeProto node;

  const bool stored_first = core::runtime::RegisterKernelFn(
      domain, "OverwriteOp", core::symbolic::Device::kCPU,
      [](const NodeProto &, core::runtime::RuntimeContext &,
         const core::runtime::KernelContext &) -> std::unique_ptr<core::runtime::KernelBase> {
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
      [](const NodeProto &, core::runtime::RuntimeContext &,
         const core::runtime::KernelContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        which = 2;
        return nullptr;
      },
      /*overwrite=*/false);
  EXPECT_FALSE(stored_if_absent);
  which = 0;
  table.find(key)->second(node, ctx, ctx.kernel_ctx());
  EXPECT_EQ(which, 1) << "overwrite=false must keep the first factory.";

  // overwrite=true (default): the entry is replaced and the call reports true.
  const bool stored_overwrite = core::runtime::RegisterKernelFn(
      domain, "OverwriteOp", core::symbolic::Device::kCPU,
      [](const NodeProto &, core::runtime::RuntimeContext &,
         const core::runtime::KernelContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        which = 3;
        return nullptr;
      });
  EXPECT_TRUE(stored_overwrite);
  which = 0;
  table.find(key)->second(node, ctx, ctx.kernel_ctx());
  EXPECT_EQ(which, 3) << "overwrite=true must replace the factory.";
}

} // namespace Test
