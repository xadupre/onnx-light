// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include "onnx_proto/onnx_helper.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterDropoutCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Dropout dropout_kernel{ctx};

  // Inference-mode Dropout (default training_mode=false): output copies input.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("data");
    node.add_output("output");

    Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor mask("", static_cast<int32_t>(DataType::BOOL), data.shape, std::vector<uint8_t>(6, 1));
    Tensor output = dropout_kernel(data, /*ratio=*/0.5f, /*training_mode=*/false, mask,
                                   kernel::Dropout::kNoSeed);
    Expect(node, {data}, {output}, "test_cc_dropout_default_inference", {opset}, "backend-test",
           registry);
  }

  // Training-mode Dropout with ratio/training_mode inputs and mask output.
  // Use ratio=0 so expected outputs are deterministic across runtimes.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("data");
    node.add_input("ratio");
    node.add_input("training_mode");
    node.add_output("output");
    node.add_output("mask");
    AddAttribute<int64_t>(node, "seed", 123);

    Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor ratio = Tensor::FromFloat("", {}, {0.0f});
    Tensor training_mode = Tensor::FromBool("", {}, {1});
    auto produced = dropout_kernel(data, /*ratio=*/0.0f, /*training_mode=*/true, /*seed=*/123);

    Expect(node, {data, ratio, training_mode}, {produced.first, produced.second},
           "test_cc_dropout_training_mask", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
