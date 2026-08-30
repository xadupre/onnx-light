// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeCumProdNode(bool exclusive, bool reverse) {
  NodeProto node;
  node.set_op_type("CumProd");
  node.add_input("x");
  node.add_input("axis");
  node.add_output("y");
  if (exclusive) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("exclusive");
    attr->set_type(AttributeProto::INT);
    attr->set_i(1);
  }
  if (reverse) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("reverse");
    attr->set_type(AttributeProto::INT);
    attr->set_i(1);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// CumProd — cumulative product of the input tensor along a selected axis.
// Mirrors the docstring example in the upstream ``CumProd`` schema (opset 26).
// ---------------------------------------------------------------------------
void RegisterCumProdCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(26);
  const auto cumprod_kernel = MakeReferenceKernel<onnx_kernels::kernel::CumProd>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize};
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_cumprod_benchmark", {opset}, {count, 1}, {count},
           [cumprod_kernel, shape, count]() -> IoData {
             Tensor x = Tensor::FromDouble("", shape, std::vector<double>(count, 1.0));
             Tensor axis = Tensor::FromInt32("", {}, {0});
             Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
             return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
           });
    return;
  }

  // 1-D inclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_1d", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 1-D exclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_1d_exclusive", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke(
          [&](const auto &kernel) { return kernel(x, axis, /*exclusive=*/true); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 1-D reverse cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/true);
    Expect(registry, std::move(node), "test_cumprod_1d_reverse", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) {
        return kernel(x, axis, /*exclusive=*/false, /*reverse=*/true);
      });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 1-D reverse + exclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/true);
    Expect(registry, std::move(node), "test_cumprod_1d_reverse_exclusive", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
             Tensor axis = Tensor::FromInt32("", {}, {0});
             Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, axis, /*exclusive=*/true, /*reverse=*/true);
             });
             return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
           });
  }

  // 2-D cumulative product along axis 0.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_2d_axis_0", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 2-D cumulative product along axis 1.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_2d_axis_1", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      Tensor axis = Tensor::FromInt32("", {}, {1});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 2-D cumulative product along negative axis (-1) — axis input is INT64.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_2d_negative_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      Tensor axis = Tensor::FromInt64("", {}, {static_cast<int64_t>(-1)});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 2-D cumulative product on INT32 input along axis 0.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_2d_int32", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {2, 3}, {1, 2, 3, 4, 5, 6});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke([&](const auto &kernel) { return kernel(x, axis); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }

  // 1-D exclusive cumulative product on INT32 input (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/false);
    Expect(registry, std::move(node), "test_cumprod_1d_int32_exclusive", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromInt32("", {5}, {1, 2, 3, 4, 5});
      Tensor axis = Tensor::FromInt32("", {}, {0});
      Tensor y = cumprod_kernel.Invoke(
          [&](const auto &kernel) { return kernel(x, axis, /*exclusive=*/true); });
      return IoData{{std::move(x), std::move(axis)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
