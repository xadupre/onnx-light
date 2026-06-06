// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// ---------------------------------------------------------------------------
// Bitwise binary helpers. Each ``RegisterBitwiseBin*Case`` builds one
// upstream-ONNX ``test_bitwise_<op>_<dtype>_<variant>`` case using
// deterministically seeded integer inputs and the matching kernel as the
// expected-output oracle (``np.bitwise_<op>`` equivalent).
// ---------------------------------------------------------------------------

template <typename TKernel>
void RegisterBitwiseBinSignedCase(const std::string &name, const char *op,
                                  const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                  const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                  const TKernel &k, const OpsetId &opset,
                                  std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Tensor x = Tensor::FromInt32("", x_shape, RandnInt<int32_t>(x_shape, x_seed));
  Tensor y = Tensor::FromInt32("", y_shape, RandnInt<int32_t>(y_shape, y_seed));
  Tensor z = k(x, y);
  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

template <typename TKernel>
void RegisterBitwiseBinSignedCase16(const std::string &name, const char *op,
                                    const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                    const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                    const TKernel &k, const OpsetId &opset,
                                    std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Tensor x = Tensor::FromInt16("", x_shape, RandnInt<int16_t>(x_shape, x_seed));
  Tensor y = Tensor::FromInt16("", y_shape, RandnInt<int16_t>(y_shape, y_seed));
  Tensor z = k(x, y);
  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

template <typename TKernel>
void RegisterBitwiseBinUint64Case(const std::string &name, const char *op,
                                  const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                  const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                  const TKernel &k, const OpsetId &opset,
                                  std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Tensor x = Tensor::FromUint64("", x_shape, RandUint<uint64_t>(1 << 16, x_shape, x_seed));
  Tensor y = Tensor::FromUint64("", y_shape, RandUint<uint64_t>(1 << 16, y_shape, y_seed));
  Tensor z = k(x, y);
  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

template <typename TKernel>
void RegisterBitwiseBinUint8Case(const std::string &name, const char *op,
                                 const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                 const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                 const TKernel &k, const OpsetId &opset,
                                 std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Tensor x = Tensor::FromUint8("", x_shape, RandUint<uint8_t>(256, x_shape, x_seed));
  Tensor y = Tensor::FromUint8("", y_shape, RandUint<uint8_t>(256, y_shape, y_seed));
  Tensor z = k(x, y);
  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// BitwiseAnd — z = x & y, element-wise with broadcasting (since opset 18).
// Mirrors the upstream ``onnx.backend.test.case.node.bitwiseand.BitwiseAnd``
// class.
// ---------------------------------------------------------------------------
void RegisterBitwiseAndCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::BitwiseAnd k{ctx};

  // Fixed-vector smoke variant (mirrors the And ``test_cc_and`` style).
  {
    NodeProto node = MakeNode("BitwiseAnd", {"x", "y"}, {"z"});
    Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, -1, 0});
    Tensor y = Tensor::FromInt32("", {4}, {0b1010, 0b0110, 0xFF, -1});
    Tensor z = k(x, y);
    Expect(node, {x, y}, {z}, "test_cc_bitwise_and", {opset}, "backend-test", registry);
  }

  // Upstream ONNX node cases:
  //   - BitwiseAnd.export(): test_bitwise_and_i32_2d, test_bitwise_and_i16_3d
  //   - BitwiseAnd.export_bitwiseand_broadcast():
  //         test_bitwise_and_ui64_bcast_3v1d, test_bitwise_and_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase("test_bitwise_and_i32_2d", "BitwiseAnd", {3, 4}, 1001, {3, 4}, 1002,
                               k, opset, registry);
  RegisterBitwiseBinSignedCase16("test_bitwise_and_i16_3d", "BitwiseAnd", {3, 4, 5}, 1003,
                                 {3, 4, 5}, 1004, k, opset, registry);
  RegisterBitwiseBinUint64Case("test_bitwise_and_ui64_bcast_3v1d", "BitwiseAnd", {3, 4, 5}, 1005,
                               {5}, 1006, k, opset, registry);
  RegisterBitwiseBinUint8Case("test_bitwise_and_ui8_bcast_4v3d", "BitwiseAnd", {3, 4, 5, 6}, 1007,
                              {4, 5, 6}, 1008, k, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseOr — z = x | y, element-wise with broadcasting (since opset 18).
// ---------------------------------------------------------------------------
void RegisterBitwiseOrCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::BitwiseOr k{ctx};

  {
    NodeProto node = MakeNode("BitwiseOr", {"x", "y"}, {"z"});
    Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, 0, 0});
    Tensor y = Tensor::FromInt32("", {4}, {0b0011, 0b0110, 0xFF, 0});
    Tensor z = k(x, y);
    Expect(node, {x, y}, {z}, "test_cc_bitwise_or", {opset}, "backend-test", registry);
  }

  // Upstream ONNX node cases:
  //   - BitwiseOr.export(): test_bitwise_or_i32_2d, test_bitwise_or_i16_4d
  //   - BitwiseOr.export_bitwiseor_broadcast():
  //         test_bitwise_or_ui64_bcast_3v1d, test_bitwise_or_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase("test_bitwise_or_i32_2d", "BitwiseOr", {3, 4}, 1101, {3, 4}, 1102, k,
                               opset, registry);
  RegisterBitwiseBinSignedCase16("test_bitwise_or_i16_4d", "BitwiseOr", {3, 4, 5, 6}, 1103,
                                 {3, 4, 5, 6}, 1104, k, opset, registry);
  RegisterBitwiseBinUint64Case("test_bitwise_or_ui64_bcast_3v1d", "BitwiseOr", {3, 4, 5}, 1105, {5},
                               1106, k, opset, registry);
  RegisterBitwiseBinUint8Case("test_bitwise_or_ui8_bcast_4v3d", "BitwiseOr", {3, 4, 5, 6}, 1107,
                              {4, 5, 6}, 1108, k, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseXor — z = x ^ y, element-wise with broadcasting (since opset 18).
// ---------------------------------------------------------------------------
void RegisterBitwiseXorCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::BitwiseXor k{ctx};

  {
    NodeProto node = MakeNode("BitwiseXor", {"x", "y"}, {"z"});
    Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, -1, 0});
    Tensor y = Tensor::FromInt32("", {4}, {0b1010, 0b0110, 0xFF, 0});
    Tensor z = k(x, y);
    Expect(node, {x, y}, {z}, "test_cc_bitwise_xor", {opset}, "backend-test", registry);
  }

  // Upstream ONNX node cases:
  //   - BitwiseXor.export(): test_bitwise_xor_i32_2d, test_bitwise_xor_i16_3d
  //   - BitwiseXor.export_bitwisexor_broadcast():
  //         test_bitwise_xor_ui64_bcast_3v1d, test_bitwise_xor_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase("test_bitwise_xor_i32_2d", "BitwiseXor", {3, 4}, 1201, {3, 4}, 1202,
                               k, opset, registry);
  RegisterBitwiseBinSignedCase16("test_bitwise_xor_i16_3d", "BitwiseXor", {3, 4, 5}, 1203,
                                 {3, 4, 5}, 1204, k, opset, registry);
  RegisterBitwiseBinUint64Case("test_bitwise_xor_ui64_bcast_3v1d", "BitwiseXor", {3, 4, 5}, 1205,
                               {5}, 1206, k, opset, registry);
  RegisterBitwiseBinUint8Case("test_bitwise_xor_ui8_bcast_4v3d", "BitwiseXor", {3, 4, 5, 6}, 1207,
                              {4, 5, 6}, 1208, k, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseNot — y = ~x, element-wise (since opset 18).
// Mirrors the upstream ``onnx.backend.test.case.node.bitwisenot.BitwiseNot``
// class.
// ---------------------------------------------------------------------------
void RegisterBitwiseNotCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::BitwiseNot k{ctx};

  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Tensor x = Tensor::FromInt32("", {4}, {0, -1, 1, 0x12345});
    Tensor y = k(x);
    Expect(node, {x}, {y}, "test_cc_bitwise_not", {opset}, "backend-test", registry);
  }

  // Upstream ONNX node cases (BitwiseNot.export):
  //   test_bitwise_not_2d (int32), test_bitwise_not_3d (uint16),
  //   test_bitwise_not_4d (uint8).
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Tensor x = Tensor::FromInt32("", {3, 4}, RandnInt<int32_t>({3, 4}, 1301));
    Tensor y = k(x);
    Expect(node, {x}, {y}, "test_bitwise_not_2d", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Tensor x = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(1 << 15, {3, 4, 5}, 1302));
    Tensor y = k(x);
    Expect(node, {x}, {y}, "test_bitwise_not_3d", {opset}, "backend-test", registry);
  }
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Tensor x = Tensor::FromUint8("", {3, 4, 5, 6}, RandUint<uint8_t>(256, {3, 4, 5, 6}, 1303));
    Tensor y = k(x);
    Expect(node, {x}, {y}, "test_bitwise_not_4d", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
