// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_light_helpers.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kEqualName = "kernel::Equal";
constexpr const char *kBoolName = "BOOL";
constexpr std::array<int32_t, 14> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::BOOL),     static_cast<int32_t>(DataType::FLOAT),
    static_cast<int32_t>(DataType::DOUBLE),   static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16), static_cast<int32_t>(DataType::INT8),
    static_cast<int32_t>(DataType::INT16),    static_cast<int32_t>(DataType::INT32),
    static_cast<int32_t>(DataType::INT64),    static_cast<int32_t>(DataType::UINT8),
    static_cast<int32_t>(DataType::UINT16),   static_cast<int32_t>(DataType::UINT32),
    static_cast<int32_t>(DataType::UINT64),   static_cast<int32_t>(DataType::STRING),
};

template <typename TIn>
Tensor EqualAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                  int64_t grain, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a == b ? 1 : 0; }, allocator, grain);
}

template <typename TIn>
void EqualInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                  Tensor &output, int64_t grain) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a == b ? 1 : 0; }, grain);
}

// String equality path. ``detail::BinaryElementwise*`` operate on POD data
// stored in ``Tensor::data`` and cannot handle STRING tensors, which live in
// ``Tensor::string_data``. We only support equal-shape inputs or scalar
// broadcasting here, which is enough for the upstream ``test_equal_string``
// and ``test_equal_string_broadcast`` cases.
struct StringEqualBroadcast {
  onnx_kernels::Shape shape;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

StringEqualBroadcast CheckStringEqualInputs(const Tensor &x, const Tensor &y) {
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  EXT_ENFORCE_INVALID(
      nx == ny || nx == 1 || ny == 1,
      "kernel::Equal STRING inputs only support equal-shape tensors or scalar broadcasting.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == nx,
                      "kernel::Equal input ``x`` string_data size does not match its shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(y.string_data.size()) == ny,
                      "kernel::Equal input ``y`` string_data size does not match its shape.");
  StringEqualBroadcast bi;
  bi.nx = nx;
  bi.ny = ny;
  bi.element_count = nx >= ny ? nx : ny;
  bi.shape = nx >= ny ? x.shape : y.shape;
  return bi;
}

Tensor EqualStringAlloc(const Tensor &x, const Tensor &y, int64_t grain,
                        RawBufferAllocator *allocator = nullptr) {
  const StringEqualBroadcast bi = CheckStringEqualInputs(x, y);
  const size_t out_n_bytes = static_cast<size_t>(bi.element_count);
  Tensor out = MakeOutputTensor(DataType::BOOL, bi.shape, out_n_bytes, allocator);
  ParallelFor(bi.element_count, grain, [&x, &y, &out, &bi](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const std::string &a = x.string_data[bi.nx == 1 ? 0 : static_cast<size_t>(i)];
      const std::string &b = y.string_data[bi.ny == 1 ? 0 : static_cast<size_t>(i)];
      out.mutable_bytes()[static_cast<size_t>(i)] = a == b ? 1 : 0;
    }
  });
  return out;
}

void EqualStringInPlace(const Tensor &x, const Tensor &y, Tensor &output, int64_t grain) {
  const StringEqualBroadcast bi = CheckStringEqualInputs(x, y);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Equal preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(
      output.shape == bi.shape,
      "kernel::Equal preallocated output shape must match the broadcasted input shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.size_bytes()) == bi.element_count,
                      "kernel::Equal preallocated output ``data`` has unexpected size.");
  ParallelFor(bi.element_count, grain, [&x, &y, &output, &bi](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const std::string &a = x.string_data[bi.nx == 1 ? 0 : static_cast<size_t>(i)];
      const std::string &b = y.string_data[bi.ny == 1 ? 0 : static_cast<size_t>(i)];
      output.mutable_bytes()[static_cast<size_t>(i)] = a == b ? 1 : 0;
    }
  });
}
} // namespace

Equal::Equal(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Equal", kSupportedElementTypes, kParallelForGrainSize) {}

void Equal::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Equal", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Equal::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::BOOL:
    return EqualAlloc<uint8_t>("BOOL", DataType::BOOL, x, y, grain, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT:
    return EqualAlloc<float>("FLOAT", DataType::FLOAT, x, y, grain, rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return EqualAlloc<double>("DOUBLE", DataType::DOUBLE, x, y, grain,
                              rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kEqualName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a == b ? 1 : 0; }, rt ? rt->allocator() : nullptr,
        grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a == b ? 1 : 0; }, rt ? rt->allocator() : nullptr,
        grain);
  case DataType::INT8:
    return EqualAlloc<int8_t>("INT8", DataType::INT8, x, y, grain, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return EqualAlloc<int16_t>("INT16", DataType::INT16, x, y, grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return EqualAlloc<int32_t>("INT32", DataType::INT32, x, y, grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return EqualAlloc<int64_t>("INT64", DataType::INT64, x, y, grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return EqualAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return EqualAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, grain,
                                rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return EqualAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, grain,
                                rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return EqualAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, grain,
                                rt ? rt->allocator() : nullptr);
  case DataType::STRING:
    EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::Equal inputs must share the same dtype.");
    return EqualStringAlloc(x, y, grain, rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kEqualName, ": unsupported data type ", x.data_type,
                      ", only supports BOOL, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32, UINT64 and STRING inputs.");
  }
}

void Equal::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::BOOL:
    return EqualInPlace<uint8_t>("BOOL", DataType::BOOL, x, y, output, grain);
  case DataType::FLOAT:
    return EqualInPlace<float>("FLOAT", DataType::FLOAT, x, y, output, grain);
  case DataType::DOUBLE:
    return EqualInPlace<double>("DOUBLE", DataType::DOUBLE, x, y, output, grain);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kEqualName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a == b ? 1 : 0; }, grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a == b ? 1 : 0; }, grain);
  case DataType::INT8:
    return EqualInPlace<int8_t>("INT8", DataType::INT8, x, y, output, grain);
  case DataType::INT16:
    return EqualInPlace<int16_t>("INT16", DataType::INT16, x, y, output, grain);
  case DataType::INT32:
    return EqualInPlace<int32_t>("INT32", DataType::INT32, x, y, output, grain);
  case DataType::INT64:
    return EqualInPlace<int64_t>("INT64", DataType::INT64, x, y, output, grain);
  case DataType::UINT8:
    return EqualInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output, grain);
  case DataType::UINT16:
    return EqualInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output, grain);
  case DataType::UINT32:
    return EqualInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output, grain);
  case DataType::UINT64:
    return EqualInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output, grain);
  case DataType::STRING:
    EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::Equal inputs must share the same dtype.");
    return EqualStringInPlace(x, y, output, grain);
  default:
    EXT_THROW_INVALID(kEqualName, ": unsupported data type ", x.data_type,
                      ", only supports BOOL, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32, UINT64 and STRING inputs.");
  }
}

void Equal::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
