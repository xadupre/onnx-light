// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kBitShiftName = "kernel::BitShift";
constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::UINT8), static_cast<int32_t>(DataType::UINT16),
    static_cast<int32_t>(DataType::UINT32), static_cast<int32_t>(DataType::UINT64)};

[[noreturn]] void ThrowUnsupportedBitShift() {
  EXT_THROW_INVALID(kBitShiftName, " only supports UINT8, UINT16, UINT32 and UINT64 inputs.");
}

template <typename Op>
Tensor BitShiftAllocDispatch(const Tensor &x, const Tensor &y, Op op, int64_t grain,
                             RawBufferAllocator *allocator = nullptr) {
#define ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(ENUM, NAME, CTYPE)                                       \
  case DataType::ENUM:                                                                             \
    return detail::BinaryElementwiseAlloc<CTYPE, CTYPE>(                                           \
        kBitShiftName, NAME, DataType::ENUM, x, y,                                                 \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); }, allocator,      \
        grain)
  switch (x.data_type) {
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT64, "UINT64", uint64_t);
  default:
    ThrowUnsupportedBitShift();
  }
#undef ONNX_LIGHT_BITSHIFT_DISPATCH_CASE
}

template <typename Op>
void BitShiftInPlaceDispatch(const Tensor &x, const Tensor &y, Tensor &output, Op op,
                             int64_t grain) {
#define ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(ENUM, NAME, CTYPE)                                       \
  case DataType::ENUM:                                                                             \
    detail::BinaryElementwise<CTYPE, CTYPE>(                                                       \
        kBitShiftName, NAME, DataType::ENUM, x, y, output,                                         \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); }, grain);         \
    return
  switch (x.data_type) {
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(UINT64, "UINT64", uint64_t);
  default:
    ThrowUnsupportedBitShift();
  }
#undef ONNX_LIGHT_BITSHIFT_DISPATCH_CASE
}

// Shift functors: cast ``b`` (the shift amount) to ``int`` so the operator
// expression compiles for every unsigned integral type ``a``. Behavior
// when ``b`` is greater than or equal to the bit-width of ``a`` is
// undefined in C++, which matches NumPy's "implementation-defined" wording
// for ``np.left_shift``/``np.right_shift`` in that situation. The
// reference cases registered below restrict shift amounts to safe values.
constexpr auto kLeftShiftFn = [](auto a, auto b) { return a << static_cast<int>(b); };
constexpr auto kRightShiftFn = [](auto a, auto b) { return a >> static_cast<int>(b); };

} // namespace

BitShift::BitShift(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "BitShift", kSupportedElementTypes, kParallelForGrainSize) {}

void BitShift::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("BitShift", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor BitShift::operator()(const Tensor &x, const Tensor &y, Direction direction,
                            RuntimeContext *rt) const {
  if (rt != nullptr) {
    const Shape out_shape = detail::BroadcastShape("kernel::BitShift", x.shape, y.shape);
    const int64_t out_count = out_shape.product();
    Tensor output = rt->MakeOutputTensor(0, x.data_type, out_shape,
                                         static_cast<size_t>(out_count) * ElementSize(x.data_type));
    (*this)(x, y, direction, output);
    return output;
  }
  RawBufferAllocator *allocator = nullptr;
  if (direction == Direction::kLeft) {
    return BitShiftAllocDispatch(x, y, kLeftShiftFn, tuning().parallel_minimum_elements, allocator);
  }
  return BitShiftAllocDispatch(x, y, kRightShiftFn, tuning().parallel_minimum_elements, allocator);
}

void BitShift::operator()(const Tensor &x, const Tensor &y, Direction direction,
                          Tensor &output) const {
  if (direction == Direction::kLeft) {
    BitShiftInPlaceDispatch(x, y, output, kLeftShiftFn, tuning().parallel_minimum_elements);
    return;
  }
  BitShiftInPlaceDispatch(x, y, output, kRightShiftFn, tuning().parallel_minimum_elements);
}

void BitShift::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  const std::string direction = GetRequiredAttributeString(node, "direction");
  onnx_kernels::kernel::BitShift::Direction dir;
  if (direction == "LEFT") {
    dir = onnx_kernels::kernel::BitShift::Direction::kLeft;
  } else if (direction == "RIGHT") {
    dir = onnx_kernels::kernel::BitShift::Direction::kRight;
  } else {
    EXT_THROW_INVALID("RunNode: BitShift 'direction' must be 'LEFT' or 'RIGHT', got '", direction,
                      "'.");
  }
  onnx_kernels::kernel::BitShift k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, y, dir, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
