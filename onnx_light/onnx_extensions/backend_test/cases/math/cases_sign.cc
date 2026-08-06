// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Sign — y = sign(x) (since opset 9, widened at opset 13).
// Registers a small deterministic ``test_cc_sign`` case and the upstream
// ONNX backend test case ``test_sign``.
// ---------------------------------------------------------------------------
void RegisterSignCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Sign sign_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Sign", sign_kernel, "test_cc_sign_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-2.5f, -1.0f, 0.0f, 0.5f, 1.0f, 3.25f});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // From Sign.export(): ``test_sign`` uses x = np.array(range(-5, 6)).
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_sign", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {11}, {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {-3.0f, -0.5f, 0.0f, 0.5f, 2.0f, -1.0f});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {-3.0f, -0.5f, 0.0f, 0.5f, 2.0f, -1.0f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // UINT8
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_uint8", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromUint8("", {2, 3}, {0, 1, 2, 0, 5, 255});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // UINT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_uint16", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromUint16("", {2, 3}, {0, 1, 1000, 0, 5, 65535});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // UINT32
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_uint32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromUint32("", {2, 3}, {0, 1, 1000000, 0, 5, 4294967295u});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // UINT64
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_uint64", {opset}, [=]() -> IoData {
      Tensor x =
          Tensor::FromUint64("", {2, 3}, {0, 1, 1000000000000ULL, 0, 5, 18446744073709551615ULL});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT8
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_int8", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt8("", {2, 3}, {-1, 0, 2, -127, 3, -5});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT16
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_int16", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt16("", {2, 3}, {-1, 0, 2, -1000, 3, -5});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT32
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_int32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {2, 3}, {-1, 0, 2, -100000, 3, -5});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT64
  {
    NodeProto node;
    node.set_op_type("Sign");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_sign_int64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt64("", {2, 3}, {-1, 0, 2, -1000000000000LL, 3, -5});
      Tensor y = sign_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
