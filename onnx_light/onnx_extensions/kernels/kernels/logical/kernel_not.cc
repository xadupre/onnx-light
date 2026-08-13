// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/random.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
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
  Not serial_kernel{context};
  Not parallel_kernel{context};
  serial_kernel.Configure(
      {key,
       {{std::string(tuning::kParallelMinimumElements), std::numeric_limits<int64_t>::max()}}});

  const int64_t minimum_elements = tuning::CalibrateUnaryParallelMinimumElements(
      "Not", execution, options, reporter, portable_minimum, ElementSize(key.element_type),
      [&](int64_t elements, int64_t parallel_minimum, int repetitions) {
        const Tensor input = RandnTensor(key.element_type, {elements}, /*seed=*/5);
        Tensor serial = MakeOutputTensor(DataType::BOOL, {elements}, input.size_bytes(), nullptr);
        Tensor parallel = MakeOutputTensor(DataType::BOOL, {elements}, input.size_bytes(), nullptr);
        parallel_kernel.Configure(
            {key, {{std::string(tuning::kParallelMinimumElements), parallel_minimum}}});
        return tuning::MeasureParallelCalibrationRuns(
            "Not", repetitions, [&]() { serial_kernel(input, serial); },
            [&]() { parallel_kernel(input, parallel); },
            [&]() {
              return std::memcmp(serial.bytes(), parallel.bytes(), serial.size_bytes()) == 0;
            });
      });
  return {key, {{std::string(tuning::kParallelMinimumElements), minimum_elements}}};
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
