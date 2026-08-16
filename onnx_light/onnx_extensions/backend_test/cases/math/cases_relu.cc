// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterReluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(14);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Relu relu_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Relu", relu_kernel, "test_cc_relu_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_relu_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3, 4, 5}, std::vector<float>(60, -1.0f));
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_relu", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // FLOAT16 test cases.
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_relu_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_relu_bfloat16", {opset}, [=]() -> IoData {
      // Build a BFLOAT16 tensor manually.
      std::vector<float> vals = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i) {
        dst[i] = FloatToBfloat16Bits(vals[i]);
      }
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // DOUBLE
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_relu_double", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {-2.0, -0.5, 0.0, 0.5, 1.5, 3.0});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT8
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_relu_int8", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt8("", {2, 3}, {-5, -1, 0, 1, 3, 127});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT16
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_relu_int16", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt16("", {2, 3}, {-500, -1, 0, 1, 300, 1000});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT32
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_relu_int32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {2, 3}, {-100000, -1, 0, 1, 42, 100000});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT64
  {
    NodeProto node;
    node.set_op_type("Relu");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_relu_int64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt64("", {2, 3}, {-1000000000000LL, -1, 0, 1, 42, 1000000000000LL});
      Tensor y = relu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
