// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterPowCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);

  NodeProto node;
  node.set_op_type("Pow");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  // From Pow.export().
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
    Expect(node, {x, y}, {z}, "test_pow_example", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 3.0f, 2.0f, 3.0f});
    Tensor z = Tensor::FromFloat("", {2, 2}, {1.0f, 8.0f, 9.0f, 64.0f});
    Expect(node, {x, y}, {z}, "test_pow", {opset}, "backend-test", registry);
  }

  // From Pow.export_pow_broadcast().
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 9.0f});
    Expect(node, {x, y}, {z}, "test_pow_bcast_scalar", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor z = Tensor::FromFloat("", {2, 3}, {1.0f, 4.0f, 27.0f, 4.0f, 25.0f, 216.0f});
    Expect(node, {x, y}, {z}, "test_pow_bcast_array", {opset}, "backend-test", registry);
  }

  // From Pow.export_types().
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromInt64("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
    Expect(node, {x, y}, {z}, "test_pow_types_float32_int64", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
    Tensor z = Tensor::FromInt64("", {3}, {1, 32, 729});
    Expect(node, {x, y}, {z}, "test_pow_types_int64_float32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromInt32("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
    Expect(node, {x, y}, {z}, "test_pow_types_float32_int32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
    Tensor z = Tensor::FromInt32("", {3}, {1, 32, 729});
    Expect(node, {x, y}, {z}, "test_pow_types_int32_float32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromUint64("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
    Expect(node, {x, y}, {z}, "test_pow_types_float32_uint64", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromUint32("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
    Expect(node, {x, y}, {z}, "test_pow_types_float32_uint32", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromInt64("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromInt64("", {3}, {1, 32, 729});
    Expect(node, {x, y}, {z}, "test_pow_types_int64_int64", {opset}, "backend-test", registry);
  }
  {
    Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromInt32("", {3}, {4, 5, 6});
    Tensor z = Tensor::FromInt32("", {3}, {1, 32, 729});
    Expect(node, {x, y}, {z}, "test_pow_types_int32_int32", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
