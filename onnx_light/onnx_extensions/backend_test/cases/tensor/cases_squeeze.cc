// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

Tensor MakeAxesTensor(const std::vector<int64_t> &axes) {
  std::vector<uint8_t> data(axes.size() * sizeof(int64_t));
  if (!axes.empty()) {
    std::memcpy(data.data(), axes.data(), data.size());
  }
  return Tensor("", DataType::INT64, {static_cast<int64_t>(axes.size())}, std::move(data));
}

NodeProto MakeSqueezeNode() {
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input("data");
  node.add_input("axes");
  node.add_output("squeezed");
  return node;
}

NodeProto MakeSqueezeNodeNoAxes() {
  // ``axes`` input slot is omitted entirely (``input_size() == 1``), exercising
  // the optional-input path of the kernel trampoline.
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input("data");
  node.add_output("squeezed");
  return node;
}

NodeProto MakeSqueezeNodeEmptyAxes() {
  // ``axes`` input slot is present but named with the empty string, which is
  // the ONNX convention for an unconnected optional input.
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input("data");
  node.add_input("");
  node.add_output("squeezed");
  return node;
}

} // namespace

void RegisterSqueezeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto squeeze_kernel = MakeReferenceKernel<onnx_kernels::kernel::Squeeze>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeSqueezeNode();
    Expect(registry, std::move(node), "test_cc_squeeze_axes_benchmark", {opset}, {4194304, 2},
           {4194304}, [squeeze_kernel]() -> IoData {
             Tensor data = RandnTensor(DataType::FLOAT, {2048, 1, 2048, 1}, 2001);
             Tensor axes = MakeAxesTensor({1, 3});
             Tensor squeezed =
                 squeeze_kernel.Invoke([&](const auto &kernel) { return kernel(data, {1, 3}); });
             return IoData{{std::move(data), std::move(axes)}, {std::move(squeezed)}};
           });
    return;
  }

  // test_cc_squeeze_axes
  {
    Expect(registry, MakeSqueezeNode(), "test_cc_squeeze_axes", {opset}, [=]() -> IoData {
      const Tensor data = Tensor::FromFloat("", {2, 1, 3, 1}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
      const Tensor axes = MakeAxesTensor({1, 3});
      const Tensor squeezed =
          squeeze_kernel.Invoke([&](const auto &kernel) { return kernel(data, {1, 3}); });
      return IoData{{std::move(data), std::move(axes)}, {std::move(squeezed)}};
    });
  }

  // test_cc_squeeze_all_singleton
  {
    Expect(registry, MakeSqueezeNode(), "test_cc_squeeze_all_singleton", {opset}, [=]() -> IoData {
      const Tensor data = Tensor::FromFloat("", {1, 2, 1, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
      const Tensor axes = MakeAxesTensor({});
      const Tensor squeezed =
          squeeze_kernel.Invoke([&](const auto &kernel) { return kernel(data, {}); });
      return IoData{{std::move(data), std::move(axes)}, {std::move(squeezed)}};
    });
  }

  // test_cc_squeeze_no_axes_input: the optional ``axes`` input slot is omitted
  // entirely; the kernel should squeeze every dimension equal to 1.
  {
    Expect(registry, MakeSqueezeNodeNoAxes(), "test_cc_squeeze_no_axes_input", {opset},
           [=]() -> IoData {
             const Tensor data =
                 Tensor::FromFloat("", {1, 2, 1, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
             const Tensor squeezed =
                 squeeze_kernel.Invoke([&](const auto &kernel) { return kernel(data, {}); });
             return IoData{{std::move(data)}, {std::move(squeezed)}};
           });
  }

  // test_cc_squeeze_empty_axes_name: the optional ``axes`` input slot is
  // present but unconnected (empty name).
  {
    Expect(registry, MakeSqueezeNodeEmptyAxes(), "test_cc_squeeze_empty_axes_name", {opset},
           [=]() -> IoData {
             const Tensor data =
                 Tensor::FromFloat("", {1, 2, 1, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
             const Tensor squeezed =
                 squeeze_kernel.Invoke([&](const auto &kernel) { return kernel(data, {}); });
             return IoData{{std::move(data)}, {std::move(squeezed)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
