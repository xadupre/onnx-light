// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_helpers.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Dispatch labels — the upstream ONNX :class:`DataType`
// enumerator names — so error messages such as
// "kernel::BitwiseAnd only supports INT8 ... inputs." stay
// self-explanatory.
constexpr const char *kBitwiseAndName = "kernel::BitwiseAnd";
constexpr const char *kBitwiseOrName = "kernel::BitwiseOr";
constexpr const char *kBitwiseXorName = "kernel::BitwiseXor";
constexpr const char *kBitwiseNotName = "kernel::BitwiseNot";

[[noreturn]] void ThrowUnsupportedBitwise(const char *op_name, int32_t data_type) {
  EXT_THROW_INVALID(op_name, ": unsupported data type ", data_type,
                    ", only supports INT8, INT16, INT32, INT64, UINT8, UINT16, "
                    "UINT32 and UINT64 inputs.");
}

// Allocating binary bitwise dispatcher: routes ``x.data_type`` to a
// typed ``BinaryElementwiseAlloc<T, T>`` call.
template <typename Op>
Tensor BitwiseBinAllocDispatch(const char *op_name, const Tensor &x, const Tensor &y, Op op,
                               RawBufferAllocator *allocator = nullptr) {
#define ONNX_LIGHT_BITWISE_DISPATCH_CASE(ENUM, NAME, CTYPE)                                        \
  case DataType::ENUM:                                                                             \
    return detail::BinaryElementwiseAlloc<CTYPE, CTYPE>(                                           \
        op_name, NAME, DataType::ENUM, x, y,                                                       \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); }, allocator)
  switch (x.data_type) {
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT8, "INT8", int8_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT16, "INT16", int16_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT32, "INT32", int32_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT64, "INT64", int64_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT64, "UINT64", uint64_t);
  default:
    ThrowUnsupportedBitwise(op_name, x.data_type);
  }
#undef ONNX_LIGHT_BITWISE_DISPATCH_CASE
}

// In-place binary bitwise dispatcher: routes ``x.data_type`` to a
// typed ``BinaryElementwise<T, T>`` call.
template <typename Op>
void BitwiseBinInPlaceDispatch(const char *op_name, const Tensor &x, const Tensor &y,
                               Tensor &output, Op op) {
#define ONNX_LIGHT_BITWISE_DISPATCH_CASE(ENUM, NAME, CTYPE)                                        \
  case DataType::ENUM:                                                                             \
    detail::BinaryElementwise<CTYPE, CTYPE>(                                                       \
        op_name, NAME, DataType::ENUM, x, y, output,                                               \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); });                \
    return
  switch (x.data_type) {
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT8, "INT8", int8_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT16, "INT16", int16_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT32, "INT32", int32_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(INT64, "INT64", int64_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_BITWISE_DISPATCH_CASE(UINT64, "UINT64", uint64_t);
  default:
    ThrowUnsupportedBitwise(op_name, x.data_type);
  }
#undef ONNX_LIGHT_BITWISE_DISPATCH_CASE
}

// Promotion-aware bitwise operations expressed as ``auto`` lambdas so the
// template type parameter is deduced from the call site. Cast to ``CTYPE``
// happens inside the dispatcher to silence ``-Wconversion``.
constexpr auto kAndFn = [](auto a, auto b) { return a & b; };
constexpr auto kOrFn = [](auto a, auto b) { return a | b; };
constexpr auto kXorFn = [](auto a, auto b) { return a ^ b; };

// Validates the preallocated output tensor of ``BitwiseNot`` and runs the
// element-wise loop. The element-count check covers both shape mismatches
// (``output.shape != x.shape``) and an unexpected output buffer size.
template <typename T>
void BitwiseNotImpl(const char *dtype_name, int32_t dtype, const Tensor &x, Tensor &output) {
  EXT_ENFORCE_INVALID(x.data_type == dtype, kBitwiseNotName, " expected ``", dtype_name,
                      "`` input.");
  EXT_ENFORCE_INVALID(output.data_type == dtype, kBitwiseNotName,
                      " preallocated output must have dtype ``", dtype_name, "``.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kBitwiseNotName,
                      " preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(T);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes, kBitwiseNotName,
                      " preallocated output buffer has unexpected size in bytes.");
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    py[static_cast<size_t>(i)] = static_cast<T>(~px[i]);
  }
}

template <typename T>
Tensor BitwiseNotAlloc(const char *dtype_name, int32_t dtype, const Tensor &x,
                       RawBufferAllocator *allocator = nullptr) {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * sizeof(T);
  Tensor y = MakeOutputTensor(dtype, x.shape, y_n_bytes, allocator);
  BitwiseNotImpl<T>(dtype_name, dtype, x, y);
  return y;
}

} // namespace

// ---------------------------------------------------------------------------
// BitwiseAnd
// ---------------------------------------------------------------------------
Tensor BitwiseAnd::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return BitwiseBinAllocDispatch(kBitwiseAndName, x, y, kAndFn, rt ? rt->allocator() : nullptr);
}

void BitwiseAnd::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  BitwiseBinInPlaceDispatch(kBitwiseAndName, x, y, output, kAndFn);
}

// ---------------------------------------------------------------------------
// BitwiseOr
// ---------------------------------------------------------------------------
Tensor BitwiseOr::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return BitwiseBinAllocDispatch(kBitwiseOrName, x, y, kOrFn, rt ? rt->allocator() : nullptr);
}

void BitwiseOr::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  BitwiseBinInPlaceDispatch(kBitwiseOrName, x, y, output, kOrFn);
}

// ---------------------------------------------------------------------------
// BitwiseXor
// ---------------------------------------------------------------------------
Tensor BitwiseXor::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return BitwiseBinAllocDispatch(kBitwiseXorName, x, y, kXorFn, rt ? rt->allocator() : nullptr);
}

void BitwiseXor::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  BitwiseBinInPlaceDispatch(kBitwiseXorName, x, y, output, kXorFn);
}

// ---------------------------------------------------------------------------
// BitwiseNot (unary)
// ---------------------------------------------------------------------------
Tensor BitwiseNot::operator()(const Tensor &x, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::INT8:
    return BitwiseNotAlloc<int8_t>("INT8", DataType::INT8, x, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return BitwiseNotAlloc<int16_t>("INT16", DataType::INT16, x, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return BitwiseNotAlloc<int32_t>("INT32", DataType::INT32, x, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return BitwiseNotAlloc<int64_t>("INT64", DataType::INT64, x, rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return BitwiseNotAlloc<uint8_t>("UINT8", DataType::UINT8, x, rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return BitwiseNotAlloc<uint16_t>("UINT16", DataType::UINT16, x, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return BitwiseNotAlloc<uint32_t>("UINT32", DataType::UINT32, x, rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return BitwiseNotAlloc<uint64_t>("UINT64", DataType::UINT64, x, rt ? rt->allocator() : nullptr);
  default:
    ThrowUnsupportedBitwise(kBitwiseNotName, x.data_type);
  }
}

void BitwiseNot::operator()(const Tensor &x, Tensor &output) const {
  switch (x.data_type) {
  case DataType::INT8:
    return BitwiseNotImpl<int8_t>("INT8", DataType::INT8, x, output);
  case DataType::INT16:
    return BitwiseNotImpl<int16_t>("INT16", DataType::INT16, x, output);
  case DataType::INT32:
    return BitwiseNotImpl<int32_t>("INT32", DataType::INT32, x, output);
  case DataType::INT64:
    return BitwiseNotImpl<int64_t>("INT64", DataType::INT64, x, output);
  case DataType::UINT8:
    return BitwiseNotImpl<uint8_t>("UINT8", DataType::UINT8, x, output);
  case DataType::UINT16:
    return BitwiseNotImpl<uint16_t>("UINT16", DataType::UINT16, x, output);
  case DataType::UINT32:
    return BitwiseNotImpl<uint32_t>("UINT32", DataType::UINT32, x, output);
  case DataType::UINT64:
    return BitwiseNotImpl<uint64_t>("UINT64", DataType::UINT64, x, output);
  default:
    ThrowUnsupportedBitwise(kBitwiseNotName, x.data_type);
  }
}

void BitwiseAnd::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

void BitwiseNot::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

void BitwiseOr::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

void BitwiseXor::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
