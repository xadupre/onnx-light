// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return RandnTensor(DataType::FLOAT, shape, seed);
}

} // namespace

// ---------------------------------------------------------------------------
// Sub — z = x - y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterSubCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(14);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Sub sub_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat("Sub", sub_kernel, "test_cc_sub_benchmark", opset, registry);
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_sub", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
      Tensor z = sub_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] - y (scalar).
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_sub_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {}, {0.5f});
      Tensor z = sub_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Sub`` operator (mirror the
  // ``onnx.backend.test.case.node.sub.Sub`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Sub` are covered: FLOAT, INT8, INT16,
  // INT32, INT64, UINT8, UINT16, UINT32 and UINT64. Inputs are generated
  // deterministically through the seeded ``Randn``/``RandnInt``/``RandUint``
  // helpers to match the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(..., dtype=...)`` patterns; expected outputs are
  // computed by ``kernel::Sub``.

  // From Sub.export(): the scripted ``test_sub_example`` case.
  {
    NodeProto node;
    node.set_op_type("Sub");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_sub_example", {opset}, [=]() -> IoData {
      // test_sub_example: expected z = [-2, 0, 2].
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
      Tensor z = sub_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  NodeProto node;
  node.set_op_type("Sub");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``sub_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::function<IoData()>>> cases = {
      // From Sub.export():
      {"test_sub",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/1);
         auto inputs_1 = RandnFloat({3, 4, 5}, /*seed=*/2);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_int8",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/11);
         auto inputs_1 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/12);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_int16",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/13);
         auto inputs_1 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/14);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_int32",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT32, {3, 4, 5}, /*seed=*/113);
         auto inputs_1 = RandnTensor(DataType::INT32, {3, 4, 5}, /*seed=*/114);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_int64",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT64, {3, 4, 5}, /*seed=*/115);
         auto inputs_1 = RandnTensor(DataType::INT64, {3, 4, 5}, /*seed=*/116);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_uint8",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/15));
         auto inputs_1 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(12, {3, 4, 5}, /*seed=*/16));
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_uint16",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/17));
         auto inputs_1 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(12, {3, 4, 5}, /*seed=*/18));
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_uint32",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/19));
         auto inputs_1 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(12, {3, 4, 5}, /*seed=*/20));
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_sub_uint64",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/21));
         auto inputs_1 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(12, {3, 4, 5}, /*seed=*/22));
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      // From Sub.export_sub_broadcast():
      {"test_sub_bcast",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/3);
         auto inputs_1 = RandnFloat({5}, /*seed=*/4);
         Tensor z = sub_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
  };

  for (const auto &[name, make_io] : cases) {
    Expect(registry, node, name, {opset}, make_io);
  }

  // FLOAT16
  {
    NodeProto n16;
    n16.set_op_type("Sub");
    n16.add_input("x");
    n16.add_input("y");
    n16.add_output("z");
    Expect(registry, std::move(n16), "test_cc_sub_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor y = MakeFloat16Tensor("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
      Tensor z = sub_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16
  {
    NodeProto nbf;
    nbf.set_op_type("Sub");
    nbf.add_input("x");
    nbf.add_input("y");
    nbf.add_output("z");
    Expect(registry, std::move(nbf), "test_cc_sub_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vx = {1.0f, 2.0f, 3.0f, 4.0f};
      std::vector<float> vy = {0.5f, 1.5f, 2.5f, 3.5f};
      std::vector<uint8_t> rx(vx.size() * 2), ry(vy.size() * 2);
      auto *dx = reinterpret_cast<uint16_t *>(rx.data());
      auto *dy = reinterpret_cast<uint16_t *>(ry.data());
      for (size_t i = 0; i < vx.size(); ++i) {
        dx[i] = FloatToBfloat16Bits(vx[i]);
        dy[i] = FloatToBfloat16Bits(vy[i]);
      }
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {4}, std::move(rx));
      Tensor y("", static_cast<int32_t>(DataType::BFLOAT16), {4}, std::move(ry));
      Tensor z = sub_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // DOUBLE
  {
    NodeProto nd;
    nd.set_op_type("Sub");
    nd.add_input("x");
    nd.add_input("y");
    nd.add_output("z");
    Expect(registry, std::move(nd), "test_cc_sub_double", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      Tensor y = Tensor::FromDouble("", {2, 3}, {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
      Tensor z = sub_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
