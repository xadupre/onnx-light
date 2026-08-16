// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::BOOL)};

KernelTuningParameters CalibrateNot(const KernelTuningKey &key,
                                    const CpuExecutionDescriptor &execution,
                                    const CalibrationOptions &options,
                                    CalibrationReporter &reporter) {
  const int64_t portable_minimum = core::runtime::kParallelForGrainSize;
  const KernelContext context{DefaultOpset(1)};
  Not reference{context};
  Not candidate{context};
  KernelCalibrationBenchmark benchmark;
  benchmark.portable_parameters = {
      key, {{std::string(tuning::kParallelMinimumElements), portable_minimum}}};
  benchmark.parameter_name = std::string(tuning::kParallelMinimumElements);
  benchmark.cases = MakeElementwiseCalibrationCases(key.element_type, 1, int64_t{1} << 14,
                                                    int64_t{1} << 23, false);
  benchmark.reference.configure = [&](int64_t value) {
    reference.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.reference.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    reference(inputs[0], output);
  };
  benchmark.candidate.configure = [&](int64_t value) {
    candidate.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.candidate.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    candidate(inputs[0], output);
  };
  return CalibrateKernelBenchmark(key, execution, options, reporter, benchmark);
}

} // namespace

Not::Not(const KernelContext &ctx) : KernelBase(ctx), tuning_(kParallelForGrainSize) {}

void Not::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Not", kSupportedElementTypes, kParallelForGrainSize,
                                        kTuningAbi);
  const KernelTuningKey key =
      tuning::MakePortableTuningKey("Not", static_cast<int32_t>(DataType::BOOL), kTuningAbi);
  core::runtime::RegisterKernelCalibrationFunction(key, CalibrateNot);
}

KernelTuningKey Not::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Not", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Not::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Not", parameters, tuning_, kTuningAbi);
}

Tensor Not::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count());
  Tensor y = MakeOutputTensor(DataType::BOOL, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Not::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::BOOL, "kernel::Not only supports BOOL tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::Not preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Not preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Not preallocated output buffer has unexpected size in bytes.");
  const uint8_t *px = x.bytes();
  uint8_t *py = output.mutable_bytes();
  ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[static_cast<size_t>(i)] = static_cast<uint8_t>(px[i] == 0 ? 1 : 0);
    }
  });
}

void Not::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
