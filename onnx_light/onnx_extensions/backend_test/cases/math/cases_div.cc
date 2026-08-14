// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/random.h"
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

Tensor RandFloatUnitOffset(const std::vector<int64_t> &shape, uint64_t seed) {
  // Mirrors the upstream ``np.random.rand(...) + 1.0`` pattern: values are
  // pseudo-random doubles in ``[1, 2)``, guaranteed to avoid division by zero.
  const std::vector<double> values = Rand(shape, seed);
  std::vector<float> floats(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    floats[i] = static_cast<float>(values[i] + 1.0);
  }
  return Tensor::FromFloat("", shape, floats);
}

// Builds a ``RandnInt<TInt>``-distributed divisor and guarantees it is
// non-zero, matching the spirit of the upstream ``+ 1`` shift used to avoid
// division by zero. Any zero value is replaced by ``1``.
template <typename TInt>
std::vector<TInt> RandnIntNonZero(const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<TInt> values = RandnInt<TInt>(shape, seed);
  for (auto &v : values) {
    if (v == 0) {
      v = static_cast<TInt>(1);
    }
  }
  return values;
}

// Builds a ``RandUint<TUInt>``-distributed divisor in ``[1, high]`` by
// drawing in ``[0, high)`` and adding ``1``, mirroring the upstream
// ``np.random.randint(high, ...) + 1`` pattern used by the ``Div`` uint
// cases. Always non-zero by construction.
template <typename TUInt>
std::vector<TUInt> RandUintNonZero(int64_t high, const std::vector<int64_t> &shape, uint64_t seed) {
  std::vector<TUInt> values = RandUint<TUInt>(high, shape, seed);
  for (auto &v : values) {
    v = static_cast<TUInt>(v + 1);
  }
  return values;
}

} // namespace

// ---------------------------------------------------------------------------
// Div — z = x / y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterDivCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(14);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Div div_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat("Div", div_kernel, "test_cc_div_benchmark", opset, registry);
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_div", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
      Tensor y = Tensor::FromFloat("", {2, 3}, {2.0f, 4.0f, 5.0f, 8.0f, 10.0f, 12.0f});
      Tensor z = div_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_div_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {}, {2.0f});
      Tensor z = div_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Div`` operator (mirror the
  // ``onnx.backend.test.case.node.div.Div`` Python class). All numeric input
  // dtypes accepted by :ref:`kernel::Div` are covered: FLOAT, INT8, INT16,
  // INT32 (the ``test_div_int32_trunc`` fixed-vector case exercising
  // truncating signed division), INT64, UINT8, UINT16, UINT32 and UINT64.
  // The divisor for the integer variants is shifted by ``+1`` to mirror the
  // upstream ``np.random.randint(24, ...) + 1`` pattern, which guarantees a
  // non-zero divisor.

  NodeProto node;
  node.set_op_type("Div");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  const std::vector<std::pair<std::string, std::function<IoData()>>> cases = {
      // From Div.export():
      {"test_div_example",
       [=]() -> IoData {
         auto inputs_0 = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
         auto inputs_1 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/33);
         auto inputs_1 = RandFloatUnitOffset({3, 4, 5}, /*seed=*/34);
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_int8",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/61);
         auto inputs_1 =
             Tensor::FromInt8("", {3, 4, 5}, RandnIntNonZero<int8_t>({3, 4, 5}, /*seed=*/62));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_int16",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/63);
         auto inputs_1 =
             Tensor::FromInt16("", {3, 4, 5}, RandnIntNonZero<int16_t>({3, 4, 5}, /*seed=*/64));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      // ``test_div_int32_trunc`` uses a hand-rolled fixed-vector pair that
      // exercises truncation toward zero (e.g. ``-3 / 2 == -1``).
      {"test_div_int32_trunc",
       [=]() -> IoData {
         auto inputs_0 = Tensor::FromInt32("", {4}, {-3, 3, -3, 3});
         auto inputs_1 = Tensor::FromInt32("", {4}, {2, 2, -2, -2});
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_int64",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT64, {3, 4, 5}, /*seed=*/163);
         auto inputs_1 =
             Tensor::FromInt64("", {3, 4, 5}, RandnIntNonZero<int64_t>({3, 4, 5}, /*seed=*/164));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_uint8",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/65));
         auto inputs_1 =
             Tensor::FromUint8("", {3, 4, 5}, RandUintNonZero<uint8_t>(24, {3, 4, 5}, /*seed=*/66));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_uint16",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/67));
         auto inputs_1 = Tensor::FromUint16("", {3, 4, 5},
                                            RandUintNonZero<uint16_t>(24, {3, 4, 5}, /*seed=*/68));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_uint32",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/69));
         auto inputs_1 = Tensor::FromUint32("", {3, 4, 5},
                                            RandUintNonZero<uint32_t>(24, {3, 4, 5}, /*seed=*/70));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_div_uint64",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/71));
         auto inputs_1 = Tensor::FromUint64("", {3, 4, 5},
                                            RandUintNonZero<uint64_t>(24, {3, 4, 5}, /*seed=*/72));
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      // From Div.export_div_broadcast():
      {"test_div_bcast",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/35);
         auto inputs_1 = RandFloatUnitOffset({5}, /*seed=*/36);
         Tensor z = div_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
  };

  for (const auto &[name, make_io] : cases) {
    Expect(registry, node, name, {opset}, make_io);
  }

  // FLOAT16
  {
    NodeProto n16;
    n16.set_op_type("Div");
    n16.add_input("x");
    n16.add_input("y");
    n16.add_output("z");
    Expect(registry, std::move(n16), "test_cc_div_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor y = MakeFloat16Tensor("", {2, 3}, {2.0f, 4.0f, 5.0f, 10.0f, 2.0f, 3.0f});
      Tensor z = div_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16
  {
    NodeProto nbf;
    nbf.set_op_type("Div");
    nbf.add_input("x");
    nbf.add_input("y");
    nbf.add_output("z");
    Expect(registry, std::move(nbf), "test_cc_div_bfloat16", {opset}, [=]() -> IoData {
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
      Tensor z = div_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // DOUBLE
  {
    NodeProto nd;
    nd.set_op_type("Div");
    nd.add_input("x");
    nd.add_input("y");
    nd.add_output("z");
    Expect(registry, std::move(nd), "test_cc_div_double", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
      Tensor y = Tensor::FromDouble("", {2, 3}, {2.0, 4.0, 5.0, 8.0, 10.0, 12.0});
      Tensor z = div_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
