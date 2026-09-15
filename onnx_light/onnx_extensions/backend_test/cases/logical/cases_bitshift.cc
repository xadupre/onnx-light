// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Build a ``BitShift`` node with the required ``direction`` attribute.
NodeProto MakeBitShiftNode(const char *direction) {
  NodeProto node = MakeNode("BitShift", {"x", "y"}, {"z"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("direction");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(direction);
  return node;
}

template <typename T>
void RegisterSignedBitShiftCase(std::vector<TestCase> &registry, const char *name,
                                const char *direction, std::vector<T> x_values,
                                std::vector<T> y_values, std::vector<T> z_values) {
  NodeProto node = MakeBitShiftNode(direction);
  Expect(registry, std::move(node), name, {DefaultOpset(28)},
         [x_values = std::move(x_values), y_values = std::move(y_values),
          z_values = std::move(z_values)]() mutable -> IoData {
           Tensor x = Tensor::From<T>("", {static_cast<int64_t>(x_values.size())}, x_values);
           Tensor y = Tensor::From<T>("", {static_cast<int64_t>(y_values.size())}, y_values);
           Tensor z = Tensor::From<T>("", {static_cast<int64_t>(z_values.size())}, z_values);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

} // namespace

// ---------------------------------------------------------------------------
// BitShift — z = x << y / x >> y, element-wise with broadcasting
// (since opset 11). Mirrors the upstream
// ``onnx.backend.test.case.node.bitshift.BitShift`` class.
// ---------------------------------------------------------------------------
void RegisterBitShiftCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeBitShiftNode("RIGHT");
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_bitshift_right_u8_benchmark", {opset},
           {count, count}, {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(11);

             const KernelContext k_ctx{opset};
             const onnx_kernels::kernel::BitShift k{k_ctx};

             Tensor x =
                 Tensor::FromUint8("", {count}, RandUint<uint8_t>(256, {count}, /*seed=*/9501));
             Tensor y =
                 Tensor::FromUint8("", {count}, RandUint<uint8_t>(8, {count}, /*seed=*/9502));
             Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Fixed-vector smoke variant matching the docstring example
  // (X=[1, 4], S=[1, 1], direction="RIGHT" => Z=[0, 2]).
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Expect(registry, std::move(node), "test_cc_bitshift_right_u8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint8("", {2}, {1, 4});
      Tensor y = Tensor::FromUint8("", {2}, {1, 1});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Expect(registry, std::move(node), "test_cc_bitshift_left_u8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint8("", {2}, {1, 2});
      Tensor y = Tensor::FromUint8("", {2}, {1, 2});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kLeft);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX node cases (BitShift.export_left*/export_right*).
  // Mirrors ``onnx.backend.test.case.node.bitshift.BitShift`` exports.
  // RIGHT direction over UINT8.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Expect(registry, std::move(node), "test_bitshift_right_uint8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint8("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint8("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // RIGHT direction over UINT16.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Expect(registry, std::move(node), "test_bitshift_right_uint16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint16("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint16("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // RIGHT direction over UINT32.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Expect(registry, std::move(node), "test_bitshift_right_uint32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint32("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint32("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // RIGHT direction over UINT64.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Expect(registry, std::move(node), "test_bitshift_right_uint64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint64("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint64("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kRight);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // LEFT direction over UINT8.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Expect(registry, std::move(node), "test_bitshift_left_uint8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint8("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint8("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kLeft);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // LEFT direction over UINT16.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Expect(registry, std::move(node), "test_bitshift_left_uint16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint16("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint16("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kLeft);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // LEFT direction over UINT32.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Expect(registry, std::move(node), "test_bitshift_left_uint32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint32("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint32("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kLeft);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  // LEFT direction over UINT64.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Expect(registry, std::move(node), "test_bitshift_left_uint64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(11);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitShift k{k_ctx};

      Tensor x = Tensor::FromUint64("", {3}, {16, 4, 1});
      Tensor y = Tensor::FromUint64("", {3}, {1, 2, 3});
      Tensor z = k(x, y, onnx_kernels::kernel::BitShift::Direction::kLeft);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_right_int8", "RIGHT", {16, 4, 1},
                                     {1, 2, 3}, {8, 1, 0});
  RegisterSignedBitShiftCase<int16_t>(registry, "test_bitshift_right_int16", "RIGHT", {16, 4, 1},
                                      {1, 2, 3}, {8, 1, 0});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_right_int32", "RIGHT", {16, 4, 1},
                                      {1, 2, 3}, {8, 1, 0});
  RegisterSignedBitShiftCase<int64_t>(registry, "test_bitshift_right_int64", "RIGHT", {16, 4, 1},
                                      {1, 2, 3}, {8, 1, 0});
  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_left_int8", "LEFT", {16, 4, 1},
                                     {1, 2, 3}, {32, 16, 8});
  RegisterSignedBitShiftCase<int16_t>(registry, "test_bitshift_left_int16", "LEFT", {16, 4, 1},
                                      {1, 2, 3}, {32, 16, 8});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_left_int32", "LEFT", {16, 4, 1},
                                      {1, 2, 3}, {32, 16, 8});
  RegisterSignedBitShiftCase<int64_t>(registry, "test_bitshift_left_int64", "LEFT", {16, 4, 1},
                                      {1, 2, 3}, {32, 16, 8});

  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_right_int8_negative_input", "RIGHT",
                                     {-8, -1, -128}, {1, 1, 1}, {-4, -1, -64});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_right_int32_negative_input", "RIGHT",
                                      {-8, -1, std::numeric_limits<int32_t>::min()}, {1, 1, 1},
                                      {-4, -1, std::numeric_limits<int32_t>::min() / 2});
  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_left_int8_overflow", "LEFT",
                                     {64, 1, -64}, {1, 7, 1}, {-128, -128, -128});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_left_int32_overflow", "LEFT",
                                      {1073741824, 1, -1073741824}, {1, 31, 1},
                                      {std::numeric_limits<int32_t>::min(),
                                       std::numeric_limits<int32_t>::min(),
                                       std::numeric_limits<int32_t>::min()});

  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_right_int8_shift_ge_width", "RIGHT",
                                     {-8, 4, -1}, {8, 9, 127}, {-1, 0, -1});
  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_left_int8_shift_ge_width", "LEFT",
                                     {-8, 4, -1}, {8, 9, 127}, {0, 0, 0});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_right_int32_shift_ge_width", "RIGHT",
                                      {-8, 4, -1}, {32, 33, 100}, {-1, 0, -1});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_left_int32_shift_ge_width", "LEFT",
                                      {-8, 4, -1}, {32, 33, 100}, {0, 0, 0});

  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_right_int8_negative_shift", "RIGHT",
                                     {-8, 4, -1}, {-1, -8, -16}, {-1, 0, -1});
  RegisterSignedBitShiftCase<int8_t>(registry, "test_bitshift_left_int8_negative_shift", "LEFT",
                                     {-8, 4, -1}, {-1, -8, -16}, {0, 0, 0});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_right_int32_negative_shift", "RIGHT",
                                      {-8, 4, -1}, {-1, -32, -64}, {-1, 0, -1});
  RegisterSignedBitShiftCase<int32_t>(registry, "test_bitshift_left_int32_negative_shift", "LEFT",
                                      {-8, 4, -1}, {-1, -32, -64}, {0, 0, 0});
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
