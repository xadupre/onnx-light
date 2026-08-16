// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterGeluCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Gelu gelu_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Gelu", gelu_kernel, "test_cc_gelu_benchmark", opset, registry);
    return;
  }

  // Default approximate ("none"), 1-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_gelu_default_1", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = gelu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Default approximate ("none"), 2-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    Expect(registry, std::move(node), "test_cc_gelu_default_2", {opset}, [=]() -> IoData {
      approximate->set_s("none");

      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = gelu_kernel(x, "none");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // approximate="tanh", 1-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    Expect(registry, std::move(node), "test_cc_gelu_tanh_1", {opset}, [=]() -> IoData {
      approximate->set_s("tanh");

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = gelu_kernel(x, "tanh");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // approximate="tanh", 2-D input.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    Expect(registry, std::move(node), "test_cc_gelu_tanh_2", {opset}, [=]() -> IoData {
      approximate->set_s("tanh");

      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = gelu_kernel(x, "tanh");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // FLOAT16, default approximate.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_gelu_default_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = gelu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // FLOAT16, approximate="tanh".
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *approximate = node.add_attribute();
    approximate->set_name("approximate");
    approximate->set_type(AttributeProto::STRING);
    Expect(registry, std::move(node), "test_cc_gelu_tanh_float16", {opset}, [=]() -> IoData {
      approximate->set_s("tanh");

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f});
      Tensor y = gelu_kernel(x, "tanh");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16, default approximate.
  {
    NodeProto node;
    node.set_op_type("Gelu");
    node.add_input("X");
    node.add_output("Y");
    Expect(registry, std::move(node), "test_cc_gelu_default_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {-1.0f, 0.0f, 1.0f, 0.5f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i) {
        dst[i] = FloatToBfloat16Bits(vals[i]);
      }
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {4}, std::move(raw));
      Tensor y = gelu_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
