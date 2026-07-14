// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

} // namespace

// ---------------------------------------------------------------------------
// BitShift — z = x << y / x >> y, element-wise with broadcasting
// (since opset 11). Mirrors the upstream
// ``onnx.backend.test.case.node.bitshift.BitShift`` class.
// ---------------------------------------------------------------------------
void RegisterBitShiftCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::BitShift k{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeBitShiftNode("RIGHT");
    const int64_t count = kBenchmarkElementwiseSize;
    RegisterLazyBenchmarkCase(
        registry, std::move(node), "test_cc_bitshift_right_u8_benchmark", {opset}, {count, count},
        {count}, [k, count]() -> IoData {
          Tensor x = Tensor::FromUint8("", {count}, RandUint<uint8_t>(256, {count}, /*seed=*/9501));
          Tensor y = Tensor::FromUint8("", {count}, RandUint<uint8_t>(8, {count}, /*seed=*/9502));
          Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
          return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
        });
    return;
  }

  // Fixed-vector smoke variant matching the docstring example
  // (X=[1, 4], S=[1, 1], direction="RIGHT" => Z=[0, 2]).
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Tensor x = Tensor::FromUint8("", {2}, {1, 4});
    Tensor y = Tensor::FromUint8("", {2}, {1, 1});
    Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
    Expect(node, {x, y}, {z}, "test_cc_bitshift_right_u8", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Tensor x = Tensor::FromUint8("", {2}, {1, 2});
    Tensor y = Tensor::FromUint8("", {2}, {1, 2});
    Tensor z = k(x, y, kernel::BitShift::Direction::kLeft);
    Expect(node, {x, y}, {z}, "test_cc_bitshift_left_u8", {opset}, "backend-test", registry);
  }

  // Upstream ONNX node cases (BitShift.export_left*/export_right*).
  // Mirrors ``onnx.backend.test.case.node.bitshift.BitShift`` exports.
  // RIGHT direction over UINT8.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Tensor x = Tensor::FromUint8("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint8("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
    Expect(node, {x, y}, {z}, "test_bitshift_right_uint8", {opset}, "backend-test", registry);
  }
  // RIGHT direction over UINT16.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Tensor x = Tensor::FromUint16("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint16("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
    Expect(node, {x, y}, {z}, "test_bitshift_right_uint16", {opset}, "backend-test", registry);
  }
  // RIGHT direction over UINT32.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Tensor x = Tensor::FromUint32("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint32("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
    Expect(node, {x, y}, {z}, "test_bitshift_right_uint32", {opset}, "backend-test", registry);
  }
  // RIGHT direction over UINT64.
  {
    NodeProto node = MakeBitShiftNode("RIGHT");
    Tensor x = Tensor::FromUint64("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint64("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kRight);
    Expect(node, {x, y}, {z}, "test_bitshift_right_uint64", {opset}, "backend-test", registry);
  }
  // LEFT direction over UINT8.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Tensor x = Tensor::FromUint8("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint8("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kLeft);
    Expect(node, {x, y}, {z}, "test_bitshift_left_uint8", {opset}, "backend-test", registry);
  }
  // LEFT direction over UINT16.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Tensor x = Tensor::FromUint16("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint16("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kLeft);
    Expect(node, {x, y}, {z}, "test_bitshift_left_uint16", {opset}, "backend-test", registry);
  }
  // LEFT direction over UINT32.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Tensor x = Tensor::FromUint32("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint32("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kLeft);
    Expect(node, {x, y}, {z}, "test_bitshift_left_uint32", {opset}, "backend-test", registry);
  }
  // LEFT direction over UINT64.
  {
    NodeProto node = MakeBitShiftNode("LEFT");
    Tensor x = Tensor::FromUint64("", {3}, {16, 4, 1});
    Tensor y = Tensor::FromUint64("", {3}, {1, 2, 3});
    Tensor z = k(x, y, kernel::BitShift::Direction::kLeft);
    Expect(node, {x, y}, {z}, "test_bitshift_left_uint64", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
