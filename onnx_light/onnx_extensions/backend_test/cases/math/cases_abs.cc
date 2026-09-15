// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterAbsCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat<onnx_kernels::kernel::Abs>(
        "Abs", "test_cc_abs_benchmark", opset, registry,
        /*with_float16=*/true, /*with_bfloat16=*/true);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
      Tensor y = abs_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_abs", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/5);
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_float16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = MakeFloat16Tensor("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_bfloat16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      std::vector<float> vals = {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT8
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_int8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromInt8("", {2, 3}, {-1, 0, 2, -127, 3, -5});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT16
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_int16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromInt16("", {2, 3}, {-1, 0, 2, -1000, 3, -5});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT32
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_int32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromInt32("", {2, 3}, {-1, 0, 2, -100000, 3, -5});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // INT64
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_int64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromInt64("", {2, 3}, {-1, 0, 2, -1000000000000LL, 3, -5});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // DOUBLE
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_abs_double", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext abs_kernel_ctx{opset};
      const onnx_kernels::kernel::Abs abs_kernel{abs_kernel_ctx};

      Tensor x = Tensor::FromDouble("", {2, 3}, {-1.0, 0.0, 1.5, -2.25, 3.5, -4.75});
      Tensor y = abs_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
