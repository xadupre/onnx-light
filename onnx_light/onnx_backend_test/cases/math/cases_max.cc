// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Max -- element-wise variadic max with NumPy-style broadcasting (since
// opset 8; opset 12 widens to all numeric types; opset 13 widens further to
// include bfloat16). Mirrors ``onnx.backend.test.case.node.max.Max``.
// ---------------------------------------------------------------------------
void RegisterMaxCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Max max_kernel{ctx};

  // ``Max.export()`` -- three-input example mirroring the upstream Python
  // reference test ``test_max_example``.
  {
    NodeProto node;
    node.set_op_type("Max");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
    Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f});
    Tensor x2 = Tensor::FromFloat("", {3}, {2.0f, 5.0f, 3.0f});
    Tensor z = max_kernel({x0, x1, x2});

    Expect(node, {x0, x1, x2}, {z}, "test_max_example", {opset}, "backend-test", registry);
  }

  // ``Max.export()`` -- single-input variant ``test_max_one_input``.
  {
    NodeProto node;
    node.set_op_type("Max");
    node.add_input("data_0");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
    Tensor z = max_kernel({x0});

    Expect(node, {x0}, {z}, "test_max_one_input", {opset}, "backend-test", registry);
  }

  // ``Max.export()`` -- two-input variant ``test_max_two_inputs``.
  {
    NodeProto node;
    node.set_op_type("Max");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
    Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f});
    Tensor z = max_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_max_two_inputs", {opset}, "backend-test", registry);
  }

  // ``Max.export_max_all_numeric_types()`` -- one two-input case per numeric
  // dtype supported by ``kernel::Max``. ``float16`` and ``bfloat16`` are not
  // exercised because the underlying C++ kernel does not implement them; the
  // upstream ``test_max_float16`` case stays in the known-missing snapshot.
  struct DtypeCase {
    const char *name;
    Tensor x0;
    Tensor x1;
  };
  const std::vector<DtypeCase> dtype_cases = {
      {"test_max_float32", Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f}),
       Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f})},
      {"test_max_float64", Tensor::FromDouble("", {3}, {3.0, 2.0, 1.0}),
       Tensor::FromDouble("", {3}, {1.0, 4.0, 4.0})},
      {"test_max_int8", Tensor::FromInt8("", {3}, {int8_t{3}, int8_t{2}, int8_t{1}}),
       Tensor::FromInt8("", {3}, {int8_t{1}, int8_t{4}, int8_t{4}})},
      {"test_max_int16", Tensor::FromInt16("", {3}, {int16_t{3}, int16_t{2}, int16_t{1}}),
       Tensor::FromInt16("", {3}, {int16_t{1}, int16_t{4}, int16_t{4}})},
      {"test_max_int32", Tensor::FromInt32("", {3}, {3, 2, 1}),
       Tensor::FromInt32("", {3}, {1, 4, 4})},
      {"test_max_int64", Tensor::FromInt64("", {3}, {int64_t{3}, int64_t{2}, int64_t{1}}),
       Tensor::FromInt64("", {3}, {int64_t{1}, int64_t{4}, int64_t{4}})},
      {"test_max_uint8", Tensor::FromUint8("", {3}, {uint8_t{3}, uint8_t{2}, uint8_t{1}}),
       Tensor::FromUint8("", {3}, {uint8_t{1}, uint8_t{4}, uint8_t{4}})},
      {"test_max_uint16", Tensor::FromUint16("", {3}, {uint16_t{3}, uint16_t{2}, uint16_t{1}}),
       Tensor::FromUint16("", {3}, {uint16_t{1}, uint16_t{4}, uint16_t{4}})},
      {"test_max_uint32", Tensor::FromUint32("", {3}, {uint32_t{3}, uint32_t{2}, uint32_t{1}}),
       Tensor::FromUint32("", {3}, {uint32_t{1}, uint32_t{4}, uint32_t{4}})},
      {"test_max_uint64", Tensor::FromUint64("", {3}, {uint64_t{3}, uint64_t{2}, uint64_t{1}}),
       Tensor::FromUint64("", {3}, {uint64_t{1}, uint64_t{4}, uint64_t{4}})},
  };

  NodeProto two_input_node;
  two_input_node.set_op_type("Max");
  two_input_node.add_input("data_0");
  two_input_node.add_input("data_1");
  two_input_node.add_output("result");

  for (const auto &c : dtype_cases) {
    Tensor z = max_kernel({c.x0, c.x1});
    Expect(two_input_node, {c.x0, c.x1}, {z}, c.name, {opset}, "backend-test", registry);
  }

  // Broadcasting variant: rank-2 vs scalar.
  {
    NodeProto node;
    node.set_op_type("Max");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");

    Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor x1 = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = max_kernel({x0, x1});

    Expect(node, {x0, x1}, {z}, "test_cc_max_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
