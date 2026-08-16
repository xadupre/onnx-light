// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kAddName = "kernel::Add";
constexpr uint32_t kTuningAbi = 1;
constexpr std::array<int32_t, 12> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
    static_cast<int32_t>(DataType::UINT8),   static_cast<int32_t>(DataType::UINT16),
    static_cast<int32_t>(DataType::UINT32),  static_cast<int32_t>(DataType::UINT64),
};

template <typename T>
Tensor AddAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                int64_t parallel_minimum_elements, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kAddName, dtype_name, dtype, x, y, [](T a, T b) -> T { return a + b; }, allocator,
      parallel_minimum_elements);
}

template <typename T>
void AddInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output, int64_t parallel_minimum_elements) {
  detail::BinaryElementwise<T, T>(
      kAddName, dtype_name, dtype, x, y, output, [](T a, T b) -> T { return a + b; },
      parallel_minimum_elements);
}

constexpr const char *kSupportedAddTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";

KernelTuningParameters CalibrateAdd(const KernelTuningKey &key,
                                    const CpuExecutionDescriptor &execution,
                                    const CalibrationOptions &options,
                                    CalibrationReporter &reporter) {
  const int64_t portable_minimum = core::runtime::kParallelForGrainSize;
  const KernelContext context{DefaultOpset(14)};
  Add reference{context};
  Add candidate{context};
  KernelCalibrationBenchmark benchmark;
  benchmark.portable_parameters = {
      key, {{std::string(tuning::kParallelMinimumElements), portable_minimum}}};
  benchmark.parameter_name = std::string(tuning::kParallelMinimumElements);
  benchmark.cases = MakeElementwiseCalibrationCases(key.element_type, 2, int64_t{1} << 14,
                                                    int64_t{1} << 23, true);
  benchmark.reference.configure = [&](int64_t value) {
    reference.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.reference.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    reference(inputs[0], inputs[1], output);
  };
  benchmark.candidate.configure = [&](int64_t value) {
    candidate.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.candidate.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    candidate(inputs[0], inputs[1], output);
  };
  return CalibrateKernelBenchmark(key, execution, options, reporter, benchmark);
}
} // namespace

Add::Add(const KernelContext &ctx) : KernelBase(ctx), tuning_(kParallelForGrainSize) {}

void Add::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Add", kSupportedElementTypes, kParallelForGrainSize,
                                        kTuningAbi);
  for (int32_t element_type : kSupportedElementTypes) {
    const KernelTuningKey key = tuning::MakePortableTuningKey("Add", element_type, kTuningAbi);
    core::runtime::RegisterKernelCalibrationFunction(key, CalibrateAdd);
  }
}

KernelTuningKey Add::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Add", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Add::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Add", parameters, tuning_, kTuningAbi);
}

Tensor Add::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return AddAlloc<float>("FLOAT", DataType::FLOAT, x, y, tuning_.parallel_minimum_elements,
                           rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return AddAlloc<double>("DOUBLE", DataType::DOUBLE, x, y, tuning_.parallel_minimum_elements,
                            rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return AddAlloc<int8_t>("INT8", DataType::INT8, x, y, tuning_.parallel_minimum_elements,
                            rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return AddAlloc<int16_t>("INT16", DataType::INT16, x, y, tuning_.parallel_minimum_elements,
                             rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return AddAlloc<int32_t>("INT32", DataType::INT32, x, y, tuning_.parallel_minimum_elements,
                             rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return AddAlloc<int64_t>("INT64", DataType::INT64, x, y, tuning_.parallel_minimum_elements,
                             rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return AddAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, tuning_.parallel_minimum_elements,
                             rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return AddAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, tuning_.parallel_minimum_elements,
                              rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return AddAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, tuning_.parallel_minimum_elements,
                              rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return AddAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, tuning_.parallel_minimum_elements,
                              rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kAddName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat, FloatToFloat16Bits,
        [](float a, float b) { return a + b; }, rt ? rt->allocator() : nullptr,
        tuning_.parallel_minimum_elements);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kAddName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [](float a, float b) { return a + b; }, rt ? rt->allocator() : nullptr,
        tuning_.parallel_minimum_elements);
  default:
    EXT_THROW_INVALID(kAddName, ": unsupported data type ", x.data_type, kSupportedAddTypesMsg);
  }
}

void Add::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return AddInPlace<float>("FLOAT", DataType::FLOAT, x, y, output,
                             tuning_.parallel_minimum_elements);
  case DataType::DOUBLE:
    return AddInPlace<double>("DOUBLE", DataType::DOUBLE, x, y, output,
                              tuning_.parallel_minimum_elements);
  case DataType::INT8:
    return AddInPlace<int8_t>("INT8", DataType::INT8, x, y, output,
                              tuning_.parallel_minimum_elements);
  case DataType::INT16:
    return AddInPlace<int16_t>("INT16", DataType::INT16, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::INT32:
    return AddInPlace<int32_t>("INT32", DataType::INT32, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::INT64:
    return AddInPlace<int64_t>("INT64", DataType::INT64, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::UINT8:
    return AddInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::UINT16:
    return AddInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::UINT32:
    return AddInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::UINT64:
    return AddInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(
        kAddName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        FloatToFloat16Bits, [](float a, float b) { return a + b; },
        tuning_.parallel_minimum_elements);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(
        kAddName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        FloatToBfloat16Bits, [](float a, float b) { return a + b; },
        tuning_.parallel_minimum_elements);
  default:
    EXT_THROW_INVALID(kAddName, ": unsupported data type ", x.data_type, kSupportedAddTypesMsg);
  }
}

void Add::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
