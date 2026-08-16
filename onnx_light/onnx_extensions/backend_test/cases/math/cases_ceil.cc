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

// ---------------------------------------------------------------------------
// Ceil — y = ceil(x) (since opset 6, widened to bfloat16 in opset 13).
// Registers both a deterministic ``test_cc_ceil`` case and the upstream
// ONNX backend test cases (``test_ceil_example`` and ``test_ceil``) mirrored
// from ``onnx.backend.test.case.node.ceil.Ceil``.
// ---------------------------------------------------------------------------
void RegisterCeilCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Ceil ceil_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Ceil", ceil_kernel, "test_cc_ceil_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Ceil");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_ceil", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.2f, 2.0f});
      Tensor y = ceil_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test cases for the ``Ceil`` operator (mirror the
  // ``onnx.backend.test.case.node.ceil.Ceil`` Python class).
  //
  // From Ceil.export(): ``test_ceil_example`` uses x = [-1.5, 1.2].
  {
    NodeProto node;
    node.set_op_type("Ceil");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_ceil_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2}, {-1.5f, 1.2f});
      Tensor y = ceil_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // From Ceil.export(): ``test_ceil`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Ceil");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_ceil", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/1);
      Tensor y = ceil_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Ceil");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_ceil_float16", {opset}, [=]() -> IoData {
      Tensor x = MakeFloat16Tensor("", {2, 3}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.7f});
      Tensor y = ceil_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Ceil");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_ceil_bfloat16", {opset}, [=]() -> IoData {
      std::vector<float> vals = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.7f};
      std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
      auto *dst = reinterpret_cast<uint16_t *>(raw.data());
      for (size_t i = 0; i < vals.size(); ++i)
        dst[i] = FloatToBfloat16Bits(vals[i]);
      Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
      Tensor y = ceil_kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
