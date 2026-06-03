// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSoftmaxCrossEntropyLossCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::SoftmaxCrossEntropyLoss sce_kernel{ctx};

  // 3 samples x 5 classes — simple "mean" reduction (default).
  {
    NodeProto node;
    node.set_op_type("SoftmaxCrossEntropyLoss");
    node.add_input("scores");
    node.add_input("labels");
    node.add_output("output");

    Tensor scores = Tensor::FromFloat("", {3, 5},
                                      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f,
                                       -0.2f, 0.3f, 0.7f, 0.9f, 0.8f});
    Tensor labels = Tensor::FromInt64("", {3}, {2, 0, 4});
    auto [loss, log_prob] = sce_kernel(scores, labels, /*weights=*/nullptr, /*reduction=*/"mean",
                                       /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {scores, labels}, {std::move(loss)}, "test_cc_softmax_cross_entropy_loss_mean",
           {opset}, "backend-test", registry);
  }

  // Reduction = "none", labels of shape (N,) — checks per-sample output shape.
  {
    NodeProto node;
    node.set_op_type("SoftmaxCrossEntropyLoss");
    node.add_input("scores");
    node.add_input("labels");
    node.add_output("output");
    AddAttribute(node, "reduction", std::string("none"));

    Tensor scores = Tensor::FromFloat("", {3, 5},
                                      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f,
                                       -0.2f, 0.3f, 0.7f, 0.9f, 0.8f});
    Tensor labels = Tensor::FromInt64("", {3}, {2, 0, 4});
    auto [loss, log_prob] = sce_kernel(scores, labels, /*weights=*/nullptr, /*reduction=*/"none",
                                       /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {scores, labels}, {std::move(loss)}, "test_cc_softmax_cross_entropy_loss_none",
           {opset}, "backend-test", registry);
  }

  // Reduction = "sum" with a weights tensor; also exercises the optional
  // third input.
  {
    NodeProto node;
    node.set_op_type("SoftmaxCrossEntropyLoss");
    node.add_input("scores");
    node.add_input("labels");
    node.add_input("weights");
    node.add_output("output");
    AddAttribute(node, "reduction", std::string("sum"));

    Tensor scores = Tensor::FromFloat("", {3, 5},
                                      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f,
                                       -0.2f, 0.3f, 0.7f, 0.9f, 0.8f});
    Tensor labels = Tensor::FromInt64("", {3}, {2, 0, 4});
    Tensor weights = Tensor::FromFloat("", {5}, {0.2f, 0.3f, 0.6f, 0.1f, 0.5f});
    auto [loss, log_prob] = sce_kernel(scores, labels, &weights, /*reduction=*/"sum",
                                       /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {scores, labels, weights}, {std::move(loss)},
           "test_cc_softmax_cross_entropy_loss_weighted_sum", {opset}, "backend-test", registry);
  }

  // Two outputs: loss + log_prob.
  {
    NodeProto node;
    node.set_op_type("SoftmaxCrossEntropyLoss");
    node.add_input("scores");
    node.add_input("labels");
    node.add_output("output");
    node.add_output("log_prob");

    Tensor scores = Tensor::FromFloat("", {3, 5},
                                      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f,
                                       -0.2f, 0.3f, 0.7f, 0.9f, 0.8f});
    Tensor labels = Tensor::FromInt64("", {3}, {2, 0, 4});
    auto [loss, log_prob] = sce_kernel(scores, labels, /*weights=*/nullptr, /*reduction=*/"mean",
                                       /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {scores, labels}, {std::move(loss), std::move(log_prob)},
           "test_cc_softmax_cross_entropy_loss_log_prob", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
