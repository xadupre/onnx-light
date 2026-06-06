// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Equal — z = (x == y), element-wise with broadcasting (since opset 7;
// STRING inputs since opset 19). Inputs are tensors of the same dtype,
// the output is BOOL.
// ---------------------------------------------------------------------------
void RegisterEqualCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(19);
  const kernel::KernelContext ctx{opset};
  const kernel::Equal equal_kernel{ctx};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Equal");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {1.0f, 5.0f, 3.0f, 0.0f});
    Tensor z = equal_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_equal", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] == y (scalar).
  {
    NodeProto node;
    node.set_op_type("Equal");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = equal_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_equal_bcast", {opset}, "backend-test", registry);
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
  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Equal.export():
      {"test_equal",
       {Tensor::FromInt32("", {3, 4, 5}, RandnInt<int32_t>({3, 4, 5}, /*seed=*/51)),
        Tensor::FromInt32("", {3, 4, 5}, RandnInt<int32_t>({3, 4, 5}, /*seed=*/52))}},
      {"test_equal_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/53)),
        Tensor::FromInt8("", {3, 4, 5}, RandnInt<int8_t>({3, 4, 5}, /*seed=*/54))}},
      {"test_equal_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/55)),
        Tensor::FromInt16("", {3, 4, 5}, RandnInt<int16_t>({3, 4, 5}, /*seed=*/56))}},
      {"test_equal_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/57)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/58))}},
      {"test_equal_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/59)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/60))}},
      {"test_equal_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/61)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/62))}},
      {"test_equal_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/63)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/64))}},
      // From Equal.export_equal_broadcast():
      {"test_equal_bcast",
       {Tensor::FromInt32("", {3, 4, 5}, RandnInt<int32_t>({3, 4, 5}, /*seed=*/65)),
        Tensor::FromInt32("", {5}, RandnInt<int32_t>({5}, /*seed=*/66))}},
  };

  for (const auto &[name, inputs] : cases) {
    Tensor z = equal_kernel(inputs[0], inputs[1]);
    Expect(node, inputs, {z}, name, {opset}, "backend-test", registry);
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
      Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
      Tensor y = Tensor::FromStrings("", {2}, {"string1", "string3"});
      Tensor z = equal_kernel(x, y);
      Expect(string_node, {x, y}, {z}, "test_equal_string", {opset}, "backend-test", registry);
    }
    {
      Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
      Tensor y = Tensor::FromStrings("", {1}, {"string1"});
      Tensor z = equal_kernel(x, y);
      Expect(string_node, {x, y}, {z}, "test_equal_string_broadcast", {opset}, "backend-test",
             registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
