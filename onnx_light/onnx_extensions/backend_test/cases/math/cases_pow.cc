// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterPowCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(14);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Pow pow_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkBinaryFloat("Pow", pow_kernel, "test_cc_pow_benchmark", opset, registry);
    return;
  }

  NodeProto node;
  node.set_op_type("Pow");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  // From Pow.export().
  {
    Expect(registry, node, "test_pow_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 3.0f, 2.0f, 3.0f});
      Tensor z = Tensor::FromFloat("", {2, 2}, {1.0f, 8.0f, 9.0f, 64.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Pow.export_pow_broadcast().
  {
    Expect(registry, node, "test_pow_bcast_scalar", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromFloat("", {}, {2.0f});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 9.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_bcast_array", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor y = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor z = Tensor::FromFloat("", {2, 3}, {1.0f, 4.0f, 27.0f, 4.0f, 25.0f, 216.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Pow.export_types().
  {
    Expect(registry, node, "test_pow_types_float32_int64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromInt64("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_int64_float32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
      Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
      Tensor z = Tensor::FromInt64("", {3}, {1, 32, 729});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_float32_int32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromInt32("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_int32_float32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
      Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
      Tensor z = Tensor::FromInt32("", {3}, {1, 32, 729});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_float32_uint64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromUint64("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_float32_uint32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromUint32("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromFloat("", {3}, {1.0f, 32.0f, 729.0f});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_int64_int64", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
      Tensor y = Tensor::FromInt64("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromInt64("", {3}, {1, 32, 729});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    Expect(registry, node, "test_pow_types_int32_int32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
      Tensor y = Tensor::FromInt32("", {3}, {4, 5, 6});
      Tensor z = Tensor::FromInt32("", {3}, {1, 32, 729});
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // FLOAT16 base with FLOAT exponent.
  {
    Expect(registry, node, "test_cc_pow_types_float16_float32", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});
      Tensor z = pow_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16 base with FLOAT exponent.
  {
    Expect(registry, node, "test_cc_pow_types_bfloat16_float32", {opset}, [=]() -> IoData {
      Tensor x = MakeBfloat16Tensor("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor y = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});
      Tensor z = pow_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
