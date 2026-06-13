// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterReluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Relu relu_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, -1.0f));
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_example", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu", {opset}, "backend-test", registry);
  }

  // FLOAT16 test cases.
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    Tensor x = kernel::MakeFloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_float16", {opset}, "backend-test", registry);
  }

  // BFLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");

    // Build a BFLOAT16 tensor manually.
    std::vector<float> vals = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f};
    std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
    auto *dst = reinterpret_cast<uint16_t *>(raw.data());
    for (size_t i = 0; i < vals.size(); ++i) {
      dst[i] = kernel::FloatToBfloat16Bits(vals[i]);
    }
    Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_bfloat16", {opset}, "backend-test", registry);
  }

  // DOUBLE
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromDouble("", {2, 3}, {-2.0, -0.5, 0.0, 0.5, 1.5, 3.0});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_double", {opset}, "backend-test", registry);
  }

  // INT8
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt8("", {2, 3}, {-5, -1, 0, 1, 3, 127});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_int8", {opset}, "backend-test", registry);
  }

  // INT16
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt16("", {2, 3}, {-500, -1, 0, 1, 300, 1000});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_int16", {opset}, "backend-test", registry);
  }

  // INT32
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt32("", {2, 3}, {-100000, -1, 0, 1, 42, 100000});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_int32", {opset}, "backend-test", registry);
  }

  // INT64
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromInt64("", {2, 3}, {-1000000000000LL, -1, 0, 1, 42, 1000000000000LL});
    Tensor y = relu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_relu_int64", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
