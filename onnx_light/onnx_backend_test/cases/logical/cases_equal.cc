// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

template <typename T>
std::vector<T> RandnIntScaled10(const std::vector<int64_t> &shape, uint64_t seed) {
  // Mirrors ``(np.random.randn(*shape) * 10).astype(dtype)`` from upstream
  // ``onnx.backend.test.case.node.equal.Equal.export()``: draw approximately
  // standard-normal values via ``Randn<double>`` (Irwin-Hall on SplitMix64),
  // scale by 10 and truncate-cast to the target integer type. This matches
  // the data-shaping behavior used by the upstream reference cases.
  const auto draws = Randn<double>(shape, seed);
  std::vector<T> out;
  out.reserve(draws.size());
  for (double v : draws) {
    out.push_back(static_cast<T>(v * 10.0));
  }
  return out;
}

} // namespace

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

    Tensor x = Tensor::FromInt32("", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromInt32("", {2, 2}, {1, 0, 3, 0});
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

    Tensor x = Tensor::FromInt32("", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromInt32("", {}, {3});
    Tensor z = equal_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_equal_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Equal`` operator (mirror the
  // ``onnx.backend.test.case.node.equal.Equal`` Python class). All numeric
  // input dtypes covered by ``Equal.export()`` are exercised: INT32 (the
  // default), INT8, INT16, UINT8, UINT16, UINT32 and UINT64. Inputs are
  // generated deterministically through the seeded ``Randn``/``RandUint``
  // helpers to match the upstream ``(np.random.randn(...) *
  // 10).astype(dtype)`` and ``np.random.randint(24, size=..., dtype=...)``
  // patterns; expected outputs are computed by ``kernel::Equal``.

  NodeProto node;
  node.set_op_type("Equal");
  node.add_input("x");
  node.add_input("y");
  node.add_output("equal");

  const std::vector<std::pair<std::string, std::vector<Tensor>>> cases = {
      // From Equal.export():
      {"test_equal",
       {Tensor::FromInt32("", {3, 4, 5}, RandnIntScaled10<int32_t>({3, 4, 5}, /*seed=*/71)),
        Tensor::FromInt32("", {3, 4, 5}, RandnIntScaled10<int32_t>({3, 4, 5}, /*seed=*/72))}},
      {"test_equal_int8",
       {Tensor::FromInt8("", {3, 4, 5}, RandnIntScaled10<int8_t>({3, 4, 5}, /*seed=*/73)),
        Tensor::FromInt8("", {3, 4, 5}, RandnIntScaled10<int8_t>({3, 4, 5}, /*seed=*/74))}},
      {"test_equal_int16",
       {Tensor::FromInt16("", {3, 4, 5}, RandnIntScaled10<int16_t>({3, 4, 5}, /*seed=*/75)),
        Tensor::FromInt16("", {3, 4, 5}, RandnIntScaled10<int16_t>({3, 4, 5}, /*seed=*/76))}},
      {"test_equal_uint8",
       {Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/77)),
        Tensor::FromUint8("", {3, 4, 5}, RandUint<uint8_t>(24, {3, 4, 5}, /*seed=*/78))}},
      {"test_equal_uint16",
       {Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/79)),
        Tensor::FromUint16("", {3, 4, 5}, RandUint<uint16_t>(24, {3, 4, 5}, /*seed=*/80))}},
      {"test_equal_uint32",
       {Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/81)),
        Tensor::FromUint32("", {3, 4, 5}, RandUint<uint32_t>(24, {3, 4, 5}, /*seed=*/82))}},
      {"test_equal_uint64",
       {Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/83)),
        Tensor::FromUint64("", {3, 4, 5}, RandUint<uint64_t>(24, {3, 4, 5}, /*seed=*/84))}},
      // From Equal.export_equal_broadcast():
      {"test_equal_bcast",
       {Tensor::FromInt32("", {3, 4, 5}, RandnIntScaled10<int32_t>({3, 4, 5}, /*seed=*/85)),
        Tensor::FromInt32("", {5}, RandnIntScaled10<int32_t>({5}, /*seed=*/86))}},
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
