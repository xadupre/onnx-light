// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

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
// Greater — z = x > y, element-wise with broadcasting (since opset 7).
// Inputs are numeric tensors of the same dtype, the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterGreaterCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Greater greater_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_greater_benchmark", {opset}, {count, count}, {count},
           [greater_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/9201);
             Tensor y = RandnTensor(DataType::FLOAT, shape, /*seed=*/9202);
             Tensor z = greater_kernel(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_greater", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
      Tensor z = greater_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] > y (scalar).
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_greater_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {}, {2.5f});
      Tensor z = greater_kernel(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Greater`` operator (mirror the
  // ``onnx.backend.test.case.node.greater.Greater`` Python class). All numeric
  // input dtypes accepted by :ref:`kernel::Greater` are covered: FLOAT,
  // INT8, INT16, UINT8, UINT16, UINT32 and UINT64. Inputs are generated
  // deterministically through the seeded ``Randn``/``RandInt`` helpers to
  // match the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(24, size=..., dtype=...)`` patterns; expected outputs
  // are computed by ``kernel::Greater``.

  NodeProto node;
  node.set_op_type("Greater");
  node.add_input("x");
  node.add_input("y");
  node.add_output("greater");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``greater_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::function<IoData()>>> cases = {
      // From Greater.export():
      {"test_greater",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/21);
         auto inputs_1 = RandnFloat({3, 4, 5}, /*seed=*/22);
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_int8",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/31);
         auto inputs_1 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/32);
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_int16",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/33);
         auto inputs_1 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/34);
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_uint8",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/35));
         auto inputs_1 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/36));
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_uint16",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/37));
         auto inputs_1 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/38));
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_uint32",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/39));
         auto inputs_1 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/40));
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_greater_uint64",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/41));
         auto inputs_1 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/42));
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      // From Greater.export_greater_broadcast():
      {"test_greater_bcast",
       [=]() -> IoData {
         auto inputs_0 = RandnFloat({3, 4, 5}, /*seed=*/23);
         auto inputs_1 = RandnFloat({5}, /*seed=*/24);
         Tensor z = greater_kernel(inputs_0, inputs_1);
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
  };

  for (const auto &[name, make_io] : cases) {
    Expect(registry, node, name, {opset}, make_io);
  }

  // FLOAT16
  {
    NodeProto n16;
    n16.set_op_type("Greater");
    n16.add_input("x");
    n16.add_input("y");
    n16.add_output("z");
    Expect(registry, std::move(n16), "test_cc_greater_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {1.0f, 4.0f, 3.0f, 6.0f, 5.0f, 2.0f});
      Tensor y = MakeFloat16Tensor("", {2, 3}, {2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 1.0f});
      Tensor z = greater_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16
  {
    NodeProto nbf;
    nbf.set_op_type("Greater");
    nbf.add_input("x");
    nbf.add_input("y");
    nbf.add_output("z");
    Expect(registry, std::move(nbf), "test_cc_greater_bfloat16", {opset}, [=]() -> IoData {
      Tensor x = MakeBfloat16Tensor("", {2, 3}, {1.0f, 4.0f, 3.0f, 6.0f, 5.0f, 2.0f});
      Tensor y = MakeBfloat16Tensor("", {2, 3}, {2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 1.0f});
      Tensor z = greater_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
