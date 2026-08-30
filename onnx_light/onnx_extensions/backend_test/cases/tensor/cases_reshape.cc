// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeReshapeNode(std::optional<int64_t> allowzero = std::nullopt) {
  NodeProto node;
  node.set_op_type("Reshape");
  node.add_input("data");
  node.add_input("shape");
  node.add_output("reshaped");
  if (allowzero.has_value()) {
    AddAttribute<int64_t>(node, "allowzero", *allowzero);
  }
  return node;
}

Tensor MakeShapeTensor(const std::vector<int64_t> &dims) {
  const std::vector<int64_t> shape_shape = {static_cast<int64_t>(dims.size())};
  std::vector<uint8_t> data(dims.size() * sizeof(int64_t));
  if (!dims.empty()) {
    std::memcpy(data.data(), dims.data(), data.size());
  }
  return Tensor("", static_cast<int32_t>(DataType::INT64), shape_shape, std::move(data));
}

} // namespace

void RegisterReshapeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset13 = DefaultOpset(13);
  const auto reshape_kernel13 = MakeReferenceKernel<onnx_kernels::kernel::Reshape>(opset13);
  const OpsetId opset14 = DefaultOpset(14);
  const auto reshape_kernel14 = MakeReferenceKernel<onnx_kernels::kernel::Reshape>(opset14);
  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeReshapeNode();
    Expect(registry, std::move(node), "test_cc_reshape_reordered_benchmark", {opset13},
           {kBenchmarkElementwiseSize, 2}, {kBenchmarkElementwiseSize},
           [reshape_kernel13]() -> IoData {
             Tensor data = RandnTensor(DataType::FLOAT, {2048, 2048}, 2001);
             Tensor shape = MakeShapeTensor({4096, 1024});
             Tensor output =
                 reshape_kernel13.Invoke([&](const auto &kernel) { return kernel(data, shape); });
             return IoData{{std::move(data), std::move(shape)}, {std::move(output)}};
           });
    return;
  }
  {
    Expect(registry, MakeReshapeNode(), "test_cc_reshape_reordered", {opset13},
           [reshape_kernel13]() -> IoData {
             const Tensor data = Tensor::FromFloat("", {2, 3},
                                                   {
                                                       1.0f,
                                                       2.0f,
                                                       3.0f,
                                                       4.0f,
                                                       5.0f,
                                                       6.0f,
                                                   });
             const Tensor shape = MakeShapeTensor({3, 2});
             const Tensor output =
                 reshape_kernel13.Invoke([&](const auto &kernel) { return kernel(data, shape); });
             return IoData{{std::move(data), std::move(shape)}, {std::move(output)}};
           });
  }

  {
    Expect(registry, MakeReshapeNode(/*allowzero=*/1), "test_cc_reshape_allowzero_literal_zero",
           {opset14}, [reshape_kernel14]() -> IoData {
             const Tensor data = Tensor::FromFloat("", {0, 2}, {});
             const Tensor shape = MakeShapeTensor({0, 2});
             const Tensor output = reshape_kernel14.Invoke(
                 [&](const auto &kernel) { return kernel(data, shape, /*allowzero=*/1); });
             return IoData{{std::move(data), std::move(shape)}, {std::move(output)}};
           });
  }

  // Mirror the ONNX backend node tests for ``Reshape`` (without function
  // expansion) so onnx-light covers the same set of named cases.
  {
    const std::vector<std::pair<std::string, std::vector<int64_t>>> cases = {
        {"reordered_all_dims", {4, 2, 3}},
        {"reordered_last_dims", {2, 4, 3}},
        {"reduced_dims", {2, 12}},
        {"extended_dims", {2, 3, 2, 2}},
        {"one_dim", {24}},
        {"negative_dim", {2, -1, 2}},
        {"negative_extended_dims", {-1, 2, 3, 4}},
        {"zero_dim", {2, 0, 4, 1}},
        {"zero_and_negative_dim", {2, 0, 1, -1}},
    };
    for (const auto &c : cases) {
      Expect(registry, MakeReshapeNode(), "test_cc_reshape_" + c.first, {opset14},
             [reshape_kernel14, shape_vec = c.second]() -> IoData {
               std::vector<float> values(24);
               for (size_t i = 0; i < values.size(); ++i) {
                 values[i] = static_cast<float>(i);
               }
               Tensor data = Tensor::FromFloat("", {2, 3, 4}, values);
               Tensor shape = MakeShapeTensor(shape_vec);
               Tensor output =
                   reshape_kernel14.Invoke([&](const auto &kernel) { return kernel(data, shape); });
               return IoData{{std::move(data), std::move(shape)}, {std::move(output)}};
             });
    }
  }

  {
    Expect(registry, MakeReshapeNode(/*allowzero=*/1), "test_cc_reshape_allowzero_reordered",
           {opset14}, [reshape_kernel14]() -> IoData {
             const Tensor data = Tensor::FromFloat("", {0, 3, 4}, {});
             const Tensor shape = MakeShapeTensor({3, 4, 0});
             const Tensor output = reshape_kernel14.Invoke(
                 [&](const auto &kernel) { return kernel(data, shape, /*allowzero=*/1); });
             return IoData{{std::move(data), std::move(shape)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
