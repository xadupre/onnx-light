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
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
                                  const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = RandnTensor(DataType::INT32, x_shape, x_seed);
           Tensor y = RandnTensor(DataType::INT32, y_shape, y_seed);
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinSignedCase16(const std::string &name, const char *op,
                                    const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                    const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                    const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = RandnTensor(DataType::INT16, x_shape, x_seed);
           Tensor y = RandnTensor(DataType::INT16, y_shape, y_seed);
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinUint64Case(const std::string &name, const char *op,
                                  const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                  const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                  const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = Tensor::FromUint64("", x_shape, RandUint<uint64_t>(1 << 16, x_shape, x_seed));
           Tensor y = Tensor::FromUint64("", y_shape, RandUint<uint64_t>(1 << 16, y_shape, y_seed));
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinUint8Case(const std::string &name, const char *op,
                                 const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                 const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                 const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = Tensor::FromUint8("", x_shape, RandUint<uint8_t>(256, x_shape, x_seed));
           Tensor y = Tensor::FromUint8("", y_shape, RandUint<uint8_t>(256, y_shape, y_seed));
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinInt8Case(const std::string &name, const char *op,
                                const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = RandnTensor(DataType::INT8, x_shape, x_seed);
           Tensor y = RandnTensor(DataType::INT8, y_shape, y_seed);
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinInt64Case(const std::string &name, const char *op,
                                 const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                 const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                 const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = RandnTensor(DataType::INT64, x_shape, x_seed);
           Tensor y = RandnTensor(DataType::INT64, y_shape, y_seed);
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

template <typename TKernel>
void RegisterBitwiseBinUint32Case(const std::string &name, const char *op,
                                  const std::vector<int64_t> &x_shape, uint64_t x_seed,
                                  const std::vector<int64_t> &y_shape, uint64_t y_seed,
                                  const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node = MakeNode(op, {"x", "y"}, {"z"});
  Expect(registry, std::move(node), name, {opset},
         [opset, x_shape, x_seed, y_shape, y_seed]() -> IoData {
           const KernelContext ctx{opset};
           const TKernel k{ctx};

           Tensor x = Tensor::FromUint32("", x_shape, RandUint<uint32_t>(1 << 16, x_shape, x_seed));
           Tensor y = Tensor::FromUint32("", y_shape, RandUint<uint32_t>(1 << 16, y_shape, y_seed));
           Tensor z = k(x, y);
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

} // namespace

// ---------------------------------------------------------------------------
// BitwiseAnd — z = x & y, element-wise with broadcasting (since opset 18).
// Mirrors the upstream ``onnx.backend.test.case.node.bitwiseand.BitwiseAnd``
// class.
// ---------------------------------------------------------------------------
void RegisterBitwiseAndCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("BitwiseAnd", {"x", "y"}, {"z"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_bitwise_and_benchmark", {opset}, {count, count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext k_ctx{opset};
             const onnx_kernels::kernel::BitwiseAnd k{k_ctx};

             Tensor x = RandnTensor(DataType::INT32, {count}, /*seed=*/9601);
             Tensor y = RandnTensor(DataType::INT32, {count}, /*seed=*/9602);
             Tensor z = k(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Fixed-vector smoke variant (mirrors the And ``test_cc_and`` style).
  {
    NodeProto node = MakeNode("BitwiseAnd", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_bitwise_and", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseAnd k{k_ctx};

      Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, -1, 0});
      Tensor y = Tensor::FromInt32("", {4}, {0b1010, 0b0110, 0xFF, -1});
      Tensor z = k(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX node cases:
  //   - BitwiseAnd.export(): test_bitwise_and_i32_2d, test_bitwise_and_i16_3d
  //   - BitwiseAnd.export_bitwiseand_broadcast():
  //         test_bitwise_and_ui64_bcast_3v1d, test_bitwise_and_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase<onnx_kernels::kernel::BitwiseAnd>(
      "test_bitwise_and_i32_2d", "BitwiseAnd", {3, 4}, 1001, {3, 4}, 1002, opset, registry);
  RegisterBitwiseBinSignedCase16<onnx_kernels::kernel::BitwiseAnd>(
      "test_bitwise_and_i16_3d", "BitwiseAnd", {3, 4, 5}, 1003, {3, 4, 5}, 1004, opset, registry);
  RegisterBitwiseBinUint64Case<onnx_kernels::kernel::BitwiseAnd>("test_bitwise_and_ui64_bcast_3v1d",
                                                                 "BitwiseAnd", {3, 4, 5}, 1005, {5},
                                                                 1006, opset, registry);
  RegisterBitwiseBinUint8Case<onnx_kernels::kernel::BitwiseAnd>("test_bitwise_and_ui8_bcast_4v3d",
                                                                "BitwiseAnd", {3, 4, 5, 6}, 1007,
                                                                {4, 5, 6}, 1008, opset, registry);
  RegisterBitwiseBinInt8Case<onnx_kernels::kernel::BitwiseAnd>(
      "test_cc_bitwise_and_i8_2d", "BitwiseAnd", {3, 4}, 1009, {3, 4}, 1010, opset, registry);
  RegisterBitwiseBinInt64Case<onnx_kernels::kernel::BitwiseAnd>(
      "test_cc_bitwise_and_i64_2d", "BitwiseAnd", {3, 4}, 1011, {3, 4}, 1012, opset, registry);
  RegisterBitwiseBinUint32Case<onnx_kernels::kernel::BitwiseAnd>(
      "test_cc_bitwise_and_ui32_2d", "BitwiseAnd", {3, 4}, 1013, {3, 4}, 1014, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseOr — z = x | y, element-wise with broadcasting (since opset 18).
// ---------------------------------------------------------------------------
void RegisterBitwiseOrCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("BitwiseOr", {"x", "y"}, {"z"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_bitwise_or_benchmark", {opset}, {count, count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext k_ctx{opset};
             const onnx_kernels::kernel::BitwiseOr k{k_ctx};

             Tensor x = RandnTensor(DataType::INT32, {count}, /*seed=*/9601);
             Tensor y = RandnTensor(DataType::INT32, {count}, /*seed=*/9602);
             Tensor z = k(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  {
    NodeProto node = MakeNode("BitwiseOr", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_bitwise_or", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseOr k{k_ctx};

      Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, 0, 0});
      Tensor y = Tensor::FromInt32("", {4}, {0b0011, 0b0110, 0xFF, 0});
      Tensor z = k(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX node cases:
  //   - BitwiseOr.export(): test_bitwise_or_i32_2d, test_bitwise_or_i16_4d
  //   - BitwiseOr.export_bitwiseor_broadcast():
  //         test_bitwise_or_ui64_bcast_3v1d, test_bitwise_or_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase<onnx_kernels::kernel::BitwiseOr>(
      "test_bitwise_or_i32_2d", "BitwiseOr", {3, 4}, 1101, {3, 4}, 1102, opset, registry);
  RegisterBitwiseBinSignedCase16<onnx_kernels::kernel::BitwiseOr>(
      "test_bitwise_or_i16_4d", "BitwiseOr", {3, 4, 5, 6}, 1103, {3, 4, 5, 6}, 1104, opset,
      registry);
  RegisterBitwiseBinUint64Case<onnx_kernels::kernel::BitwiseOr>(
      "test_bitwise_or_ui64_bcast_3v1d", "BitwiseOr", {3, 4, 5}, 1105, {5}, 1106, opset, registry);
  RegisterBitwiseBinUint8Case<onnx_kernels::kernel::BitwiseOr>("test_bitwise_or_ui8_bcast_4v3d",
                                                               "BitwiseOr", {3, 4, 5, 6}, 1107,
                                                               {4, 5, 6}, 1108, opset, registry);
  RegisterBitwiseBinInt8Case<onnx_kernels::kernel::BitwiseOr>(
      "test_cc_bitwise_or_i8_2d", "BitwiseOr", {3, 4}, 1109, {3, 4}, 1110, opset, registry);
  RegisterBitwiseBinInt64Case<onnx_kernels::kernel::BitwiseOr>(
      "test_cc_bitwise_or_i64_2d", "BitwiseOr", {3, 4}, 1111, {3, 4}, 1112, opset, registry);
  RegisterBitwiseBinUint32Case<onnx_kernels::kernel::BitwiseOr>(
      "test_cc_bitwise_or_ui32_2d", "BitwiseOr", {3, 4}, 1113, {3, 4}, 1114, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseXor — z = x ^ y, element-wise with broadcasting (since opset 18).
// ---------------------------------------------------------------------------
void RegisterBitwiseXorCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("BitwiseXor", {"x", "y"}, {"z"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_bitwise_xor_benchmark", {opset}, {count, count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext k_ctx{opset};
             const onnx_kernels::kernel::BitwiseXor k{k_ctx};

             Tensor x = RandnTensor(DataType::INT32, {count}, /*seed=*/9601);
             Tensor y = RandnTensor(DataType::INT32, {count}, /*seed=*/9602);
             Tensor z = k(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  {
    NodeProto node = MakeNode("BitwiseXor", {"x", "y"}, {"z"});
    Expect(registry, std::move(node), "test_cc_bitwise_xor", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseXor k{k_ctx};

      Tensor x = Tensor::FromInt32("", {4}, {0b1100, 0b1010, -1, 0});
      Tensor y = Tensor::FromInt32("", {4}, {0b1010, 0b0110, 0xFF, 0});
      Tensor z = k(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX node cases:
  //   - BitwiseXor.export(): test_bitwise_xor_i32_2d, test_bitwise_xor_i16_3d
  //   - BitwiseXor.export_bitwisexor_broadcast():
  //         test_bitwise_xor_ui64_bcast_3v1d, test_bitwise_xor_ui8_bcast_4v3d
  RegisterBitwiseBinSignedCase<onnx_kernels::kernel::BitwiseXor>(
      "test_bitwise_xor_i32_2d", "BitwiseXor", {3, 4}, 1201, {3, 4}, 1202, opset, registry);
  RegisterBitwiseBinSignedCase16<onnx_kernels::kernel::BitwiseXor>(
      "test_bitwise_xor_i16_3d", "BitwiseXor", {3, 4, 5}, 1203, {3, 4, 5}, 1204, opset, registry);
  RegisterBitwiseBinUint64Case<onnx_kernels::kernel::BitwiseXor>("test_bitwise_xor_ui64_bcast_3v1d",
                                                                 "BitwiseXor", {3, 4, 5}, 1205, {5},
                                                                 1206, opset, registry);
  RegisterBitwiseBinUint8Case<onnx_kernels::kernel::BitwiseXor>("test_bitwise_xor_ui8_bcast_4v3d",
                                                                "BitwiseXor", {3, 4, 5, 6}, 1207,
                                                                {4, 5, 6}, 1208, opset, registry);
  RegisterBitwiseBinInt8Case<onnx_kernels::kernel::BitwiseXor>(
      "test_cc_bitwise_xor_i8_2d", "BitwiseXor", {3, 4}, 1209, {3, 4}, 1210, opset, registry);
  RegisterBitwiseBinInt64Case<onnx_kernels::kernel::BitwiseXor>(
      "test_cc_bitwise_xor_i64_2d", "BitwiseXor", {3, 4}, 1211, {3, 4}, 1212, opset, registry);
  RegisterBitwiseBinUint32Case<onnx_kernels::kernel::BitwiseXor>(
      "test_cc_bitwise_xor_ui32_2d", "BitwiseXor", {3, 4}, 1213, {3, 4}, 1214, opset, registry);
}

// ---------------------------------------------------------------------------
// BitwiseNot — y = ~x, element-wise (since opset 18).
// Mirrors the upstream ``onnx.backend.test.case.node.bitwisenot.BitwiseNot``
// class.
// ---------------------------------------------------------------------------
void RegisterBitwiseNotCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_bitwise_not_benchmark", {opset}, {count}, {count},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext k_ctx{opset};
             const onnx_kernels::kernel::BitwiseNot k{k_ctx};

             Tensor x = RandnTensor(DataType::INT32, {count}, /*seed=*/9603);
             Tensor y = k(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_cc_bitwise_not", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseNot k{k_ctx};

      Tensor x = Tensor::FromInt32("", {4}, {0, -1, 1, 0x12345});
      Tensor y = k(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX node cases (BitwiseNot.export):
  //   test_bitwise_not_2d (int32), test_bitwise_not_3d (uint16),
  //   test_bitwise_not_4d (uint8).
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_bitwise_not_2d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseNot k{k_ctx};

      Tensor x = RandnTensor(DataType::INT32, {3, 4}, 1301);
      Tensor y = k(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_bitwise_not_3d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseNot k{k_ctx};

      Tensor x = Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(1 << 15, {3, 4, 5}, 1302));
      Tensor y = k(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeNode("BitwiseNot", {"x"}, {"y"});
    Expect(registry, std::move(node), "test_bitwise_not_4d", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext k_ctx{opset};
      const onnx_kernels::kernel::BitwiseNot k{k_ctx};

      Tensor x = Tensor::FromUint8("", {3, 4, 5, 6}, RandUint<uint8_t>(256, {3, 4, 5, 6}, 1303));
      Tensor y = k(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
