// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

void RegisterReshapeCases(std::vector<TestCase> &registry) {
  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::Reshape reshape_kernel{ctx};
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
    const Tensor output = reshape_kernel(data, shape);
    Expect(MakeReshapeNode(), {data, shape}, {output}, "test_cc_reshape_reordered", {opset},
           "backend-test", registry);
  }

  {
    const OpsetId opset = DefaultOpset(14);
    const kernel::KernelContext ctx{opset};
    const kernel::Reshape reshape_kernel{ctx};
    const Tensor data = Tensor::FromFloat("", {0, 2}, {});
    const Tensor shape = MakeShapeTensor({0, 2});
    const Tensor output = reshape_kernel(data, shape, /*allowzero=*/1);
    Expect(MakeReshapeNode(/*allowzero=*/1), {data, shape}, {output},
           "test_cc_reshape_allowzero_literal_zero", {opset}, "backend-test", registry);
  }

  // Mirror the ONNX backend node tests for ``Reshape`` (without function
  // expansion) so onnx-light covers the same set of named cases.
  {
    const OpsetId opset = DefaultOpset(14);
    const kernel::KernelContext ctx{opset};
    const kernel::Reshape reshape_kernel{ctx};
    std::vector<float> values(24);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(i);
    }
    const Tensor data = Tensor::FromFloat("", {2, 3, 4}, values);

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
      const Tensor shape = MakeShapeTensor(c.second);
      const Tensor output = reshape_kernel(data, shape);
      Expect(MakeReshapeNode(), {data, shape}, {output}, "test_cc_reshape_" + c.first, {opset},
             "backend-test", registry);
    }
  }

  {
    const OpsetId opset = DefaultOpset(14);
    const kernel::KernelContext ctx{opset};
    const kernel::Reshape reshape_kernel{ctx};
    const Tensor data = Tensor::FromFloat("", {0, 3, 4}, {});
    const Tensor shape = MakeShapeTensor({3, 4, 0});
    const Tensor output = reshape_kernel(data, shape, /*allowzero=*/1);
    Expect(MakeReshapeNode(/*allowzero=*/1), {data, shape}, {output},
           "test_cc_reshape_allowzero_reordered", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test

} // namespace ONNX_LIGHT_NAMESPACE
