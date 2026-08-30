// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeHardmaxNode(int64_t axis, bool include_axis = true) {
  NodeProto node;
  node.set_op_type("Hardmax");
  node.add_input("input");
  node.add_output("output");
  if (include_axis) {
    AttributeProto *axis_attr = node.add_attribute();
    axis_attr->set_name("axis");
    axis_attr->set_type(AttributeProto::INT);
    axis_attr->set_i(axis);
  }
  return node;
}

} // namespace

void RegisterHardmaxCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto hardmax_kernel = MakeReferenceKernel<onnx_kernels::kernel::Hardmax>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeHardmaxNode(/*axis=*/1);
    const std::vector<int64_t> shape = {2048, 2048};
    const int64_t count = 2048 * 2048;
    Expect(registry, std::move(node), "test_cc_hardmax_benchmark", {opset}, {count}, {count},
           [hardmax_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, 429);
             Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x, 1); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Two-dimensional input with explicit axis attribute.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/1);
    Expect(registry, std::move(node), "test_cc_hardmax_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x, 1); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Default axis (-1 in opset 13).
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/0, /*include_axis=*/false);
    Expect(registry, std::move(node), "test_cc_hardmax_default_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x, -1); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Negative axis: -2 on a rank-3 input picks the middle dimension.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/-2);
    Expect(registry, std::move(node), "test_cc_hardmax_negative_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x, -2); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // ---------------------------------------------------------------------------
  // Cases mirroring upstream ``test_hardmax_axis_*`` and ``test_hardmax_one_hot``
  // node tests (``onnx.backend.test.case.node.hardmax``). The substring-based
  // check in unittests/onnxl_vs_onnx/test_backend_test_names_onnx_vs_onnxlight
  // matches the upstream names through these ``test_cc_*`` cases.
  // ---------------------------------------------------------------------------

  // Deterministic rank-3 input used by the three ``hardmax_axis_*`` cases.
  Tensor x_axis = Tensor::FromFloat(
      "", {3, 4, 5},
      {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
       24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
       36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
       48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f});

  // test_cc_hardmax_axis_0 — explicit axis=0 on a rank-3 input.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/0);
    Expect(registry, std::move(node), "test_cc_hardmax_axis_0", {opset}, [=]() -> IoData {
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x_axis, 0); });
      return IoData{{std::move(x_axis)}, {std::move(y)}};
    });
  }

  // test_cc_hardmax_axis_1 — explicit axis=1 on a rank-3 input.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/1);
    Expect(registry, std::move(node), "test_cc_hardmax_axis_1", {opset}, [=]() -> IoData {
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x_axis, 1); });
      return IoData{{std::move(x_axis)}, {std::move(y)}};
    });
  }

  // test_cc_hardmax_axis_2 — explicit axis=2 on a rank-3 input.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/2);
    Expect(registry, std::move(node), "test_cc_hardmax_axis_2", {opset}, [=]() -> IoData {
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x_axis, 2); });
      return IoData{{std::move(x_axis)}, {std::move(y)}};
    });
  }

  // test_cc_hardmax_one_hot — input with repeated maxima; the first occurrence
  // along the reduction axis is selected, mirroring upstream
  // ``test_hardmax_one_hot``.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/0, /*include_axis=*/false);
    Expect(registry, std::move(node), "test_cc_hardmax_one_hot", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 4}, {3.0f, 3.0f, 3.0f, 1.0f});
      Tensor y = hardmax_kernel.Invoke([&](const auto &kernel) { return kernel(x, -1); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
