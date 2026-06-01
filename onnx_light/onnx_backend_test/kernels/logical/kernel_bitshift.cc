// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kBitShiftName = "kernel::BitShift";

[[noreturn]] void ThrowUnsupportedBitShift() {
  throw std::invalid_argument(std::string(kBitShiftName) +
                              " only supports UINT8, UINT16, UINT32 and UINT64 inputs.");
}

template <typename Op>
Tensor BitShiftAllocDispatch(const Tensor &x, const Tensor &y, Op op) {
#define ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(ENUM, NAME, CTYPE)                                       \
  case DataType::ENUM:                                                                             \
    return detail::BinaryElementwiseAlloc<CTYPE, CTYPE>(                                           \
        kBitShiftName, NAME, DataType::ENUM, x, y,                                                 \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); })
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
void BitShiftInPlaceDispatch(const Tensor &x, const Tensor &y, Tensor &output, Op op) {
#define ONNX_LIGHT_BITSHIFT_DISPATCH_CASE(ENUM, NAME, CTYPE)                                       \
  case DataType::ENUM:                                                                             \
    detail::BinaryElementwise<CTYPE, CTYPE>(                                                       \
        kBitShiftName, NAME, DataType::ENUM, x, y, output,                                         \
        [&op](CTYPE a, CTYPE b) -> CTYPE { return static_cast<CTYPE>(op(a, b)); });                \
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

Tensor BitShift::operator()(const Tensor &x, const Tensor &y, Direction direction) const {
  if (direction == Direction::kLeft) {
    return BitShiftAllocDispatch(x, y, kLeftShiftFn);
  }
  return BitShiftAllocDispatch(x, y, kRightShiftFn);
}

void BitShift::operator()(const Tensor &x, const Tensor &y, Direction direction,
                          Tensor &output) const {
  if (direction == Direction::kLeft) {
    BitShiftInPlaceDispatch(x, y, output, kLeftShiftFn);
    return;
  }
  BitShiftInPlaceDispatch(x, y, output, kRightShiftFn);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
