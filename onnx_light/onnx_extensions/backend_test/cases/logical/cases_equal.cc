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

// ---------------------------------------------------------------------------
// Equal — z = (x == y), element-wise with broadcasting (since opset 7;
// STRING inputs since opset 19). Inputs are tensors of the same dtype,
// the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterEqualCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(19);
  const auto equal_kernel = MakeReferenceKernel<onnx_kernels::kernel::Equal>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Equal");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_cc_equal_benchmark", {opset}, {count, count}, {count},
           [equal_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/9201);
             Tensor y = RandnTensor(DataType::FLOAT, shape, /*seed=*/9202);
             Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Equal");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_equal", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {2, 2}, {1.0f, 5.0f, 3.0f, 0.0f});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] == y (scalar).
  {
    NodeProto node;
    node.set_op_type("Equal");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_equal_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {}, {2.0f});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Equal`` operator (mirror the
  // ``onnx.backend.test.case.node.equal.Equal`` Python class). All numeric
  // input dtypes accepted by :ref:`kernel::Equal` are covered: INT32, INT8,
  // INT16, UINT8, UINT16, UINT32 and UINT64. Inputs are generated
  // deterministically through the seeded ``RandnInt``/``RandUint`` helpers
  // to match the upstream ``np.random.randn(...).astype(dtype)`` and
  // ``np.random.randint(24, size=..., dtype=...)`` patterns; expected
  // outputs are computed by ``kernel::Equal``.

  NodeProto node;
  node.set_op_type("Equal");
  node.add_input("x");
  node.add_input("y");
  node.add_output("equal");

  // Each upstream case is a (test_name, [x, y]) pair; the expected output is
  // computed by ``equal_kernel`` to keep the registry self-consistent.
  const std::vector<std::pair<std::string, std::function<IoData()>>> cases = {
      // From Equal.export():
      {"test_equal",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT32, {3, 4, 5}, /*seed=*/51);
         auto inputs_1 = RandnTensor(DataType::INT32, {3, 4, 5}, /*seed=*/52);
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_int8",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/53);
         auto inputs_1 = RandnTensor(DataType::INT8, {3, 4, 5}, /*seed=*/54);
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_int16",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/55);
         auto inputs_1 = RandnTensor(DataType::INT16, {3, 4, 5}, /*seed=*/56);
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_uint8",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/57));
         auto inputs_1 =
             Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/58));
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_uint16",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/59));
         auto inputs_1 =
             Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/60));
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_uint32",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/61));
         auto inputs_1 =
             Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/62));
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      {"test_equal_uint64",
       [=]() -> IoData {
         auto inputs_0 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/63));
         auto inputs_1 =
             Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/64));
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
      // From Equal.export_equal_broadcast():
      {"test_equal_bcast",
       [=]() -> IoData {
         auto inputs_0 = RandnTensor(DataType::INT32, {3, 4, 5}, /*seed=*/65);
         auto inputs_1 = RandnTensor(DataType::INT32, {5}, /*seed=*/66);
         Tensor z =
             equal_kernel.Invoke([&](const auto &kernel) { return kernel(inputs_0, inputs_1); });
         return IoData{{std::move(inputs_0), std::move(inputs_1)}, {std::move(z)}};
       }},
  };

  for (const auto &[name, make_io] : cases) {
    Expect(registry, node, name, {opset}, make_io);
  }

  // STRING cases — mirror ``Equal.export_equal_string`` and
  // ``Equal.export_equal_string_broadcast`` (since opset 19).
  {
    NodeProto string_node;
    string_node.set_op_type("Equal");
    string_node.add_input("x");
    string_node.add_input("y");
    string_node.add_output("equal");

    {
      Expect(registry, string_node, "test_equal_string", {opset}, [=]() -> IoData {
        Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
        Tensor y = Tensor::FromStrings("", {2}, {"string1", "string3"});
        Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
        return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
      });
    }
    {
      Expect(registry, string_node, "test_equal_string_broadcast", {opset}, [=]() -> IoData {
        Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
        Tensor y = Tensor::FromStrings("", {1}, {"string1"});
        Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
        return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
      });
    }
  }

  // FLOAT16
  {
    NodeProto n16;
    n16.set_op_type("Equal");
    n16.add_input("x");
    n16.add_input("y");
    n16.add_output("z");
    Expect(registry, std::move(n16), "test_cc_equal_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {1.0f, 4.0f, 3.0f, 6.0f, 5.0f, 2.0f});
      Tensor y = MakeFloat16Tensor("", {2, 3}, {2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 1.0f});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16
  {
    NodeProto nbf;
    nbf.set_op_type("Equal");
    nbf.add_input("x");
    nbf.add_input("y");
    nbf.add_output("z");
    Expect(registry, std::move(nbf), "test_cc_equal_bfloat16", {opset}, [=]() -> IoData {
      Tensor x = MakeBfloat16Tensor("", {2, 3}, {1.0f, 4.0f, 3.0f, 6.0f, 5.0f, 2.0f});
      Tensor y = MakeBfloat16Tensor("", {2, 3}, {2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 1.0f});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // DOUBLE
  {
    NodeProto nd;
    nd.set_op_type("Equal");
    nd.add_input("x");
    nd.add_input("y");
    nd.add_output("z");
    Expect(registry, std::move(nd), "test_cc_equal_double", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
      Tensor y = Tensor::FromDouble("", {3}, {1.0, 3.0, 3.0});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // INT64
  {
    NodeProto ni;
    ni.set_op_type("Equal");
    ni.add_input("x");
    ni.add_input("y");
    ni.add_output("z");
    Expect(registry, std::move(ni), "test_cc_equal_int64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
      Tensor y = Tensor::FromInt64("", {3}, {1, 3, 3});
      Tensor z = equal_kernel.Invoke([&](const auto &kernel) { return kernel(x, y); });
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
