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

void RegisterNegativeLogLikelihoodLossCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::NegativeLogLikelihoodLoss nll_kernel{ctx};

  // 3 samples x 5 classes — default "mean" reduction. The ``input`` tensor is
  // assumed to already contain log-probabilities.
  {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    node.add_output("loss");

    Tensor input = Tensor::FromFloat("", {3, 5},
                                     {-1.6f, -1.5f, -1.4f, -1.3f, -1.2f, -1.0f, -1.5f, -1.8f, -1.9f,
                                      -1.95f, -2.2f, -1.7f, -1.3f, -1.1f, -1.2f});
    Tensor target = Tensor::FromInt64("", {3}, {2, 0, 4});
    Tensor loss = nll_kernel(input, target, /*weight=*/nullptr, /*reduction=*/"mean",
                             /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {input, target}, {std::move(loss)}, "test_cc_negative_log_likelihood_loss_mean",
           {opset}, "backend-test", registry);
  }

  // Reduction = "none" produces a per-sample loss tensor.
  {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    node.add_output("loss");
    AddAttribute(node, "reduction", std::string("none"));

    Tensor input = Tensor::FromFloat("", {3, 5},
                                     {-1.6f, -1.5f, -1.4f, -1.3f, -1.2f, -1.0f, -1.5f, -1.8f, -1.9f,
                                      -1.95f, -2.2f, -1.7f, -1.3f, -1.1f, -1.2f});
    Tensor target = Tensor::FromInt64("", {3}, {2, 0, 4});
    Tensor loss = nll_kernel(input, target, /*weight=*/nullptr, /*reduction=*/"none",
                             /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {input, target}, {std::move(loss)}, "test_cc_negative_log_likelihood_loss_none",
           {opset}, "backend-test", registry);
  }

  // Reduction = "sum" with a per-class weight tensor.
  {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    node.add_input("weight");
    node.add_output("loss");
    AddAttribute(node, "reduction", std::string("sum"));

    Tensor input = Tensor::FromFloat("", {3, 5},
                                     {-1.6f, -1.5f, -1.4f, -1.3f, -1.2f, -1.0f, -1.5f, -1.8f, -1.9f,
                                      -1.95f, -2.2f, -1.7f, -1.3f, -1.1f, -1.2f});
    Tensor target = Tensor::FromInt64("", {3}, {2, 0, 4});
    Tensor weight = Tensor::FromFloat("", {5}, {0.2f, 0.3f, 0.6f, 0.1f, 0.5f});
    Tensor loss = nll_kernel(input, target, &weight, /*reduction=*/"sum",
                             /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {input, target, weight}, {std::move(loss)},
           "test_cc_negative_log_likelihood_loss_weighted_sum", {opset}, "backend-test", registry);
  }

  // ``ignore_index`` masks one of the samples; reduction = "mean".
  {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    node.add_output("loss");
    AddAttribute(node, "ignore_index", static_cast<int64_t>(-1));

    Tensor input = Tensor::FromFloat("", {3, 5},
                                     {-1.6f, -1.5f, -1.4f, -1.3f, -1.2f, -1.0f, -1.5f, -1.8f, -1.9f,
                                      -1.95f, -2.2f, -1.7f, -1.3f, -1.1f, -1.2f});
    Tensor target = Tensor::FromInt64("", {3}, {2, -1, 4});
    Tensor loss = nll_kernel(input, target, /*weight=*/nullptr, /*reduction=*/"mean",
                             /*has_ignore_index=*/true, /*ignore_index=*/-1);
    Expect(node, {input, target}, {std::move(loss)},
           "test_cc_negative_log_likelihood_loss_ignore_index", {opset}, "backend-test", registry);
  }

  // K-dimensional input: shape (N, C, D1) with a per-sample 2-D target.
  {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    node.add_output("loss");
    AddAttribute(node, "reduction", std::string("none"));

    Tensor input = Tensor::FromFloat(
        "", {2, 3, 2},
        {-1.0f, -2.0f, -2.0f, -2.0f, -3.0f, -2.0f, 0.0f, -1.0f, -2.0f, -2.0f, -1.0f, -2.0f});
    Tensor target = Tensor::FromInt64("", {2, 2}, {2, 1, 0, 2});
    Tensor loss = nll_kernel(input, target, /*weight=*/nullptr, /*reduction=*/"none",
                             /*has_ignore_index=*/false, /*ignore_index=*/0);
    Expect(node, {input, target}, {std::move(loss)}, "test_cc_negative_log_likelihood_loss_kd_none",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
