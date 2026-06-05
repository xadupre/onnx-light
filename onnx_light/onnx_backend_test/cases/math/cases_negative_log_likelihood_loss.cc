// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/random.h"
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

  // -------------------------------------------------------------------------
  // Upstream ONNX backend test cases for the ``NegativeLogLikelihoodLoss``
  // operator (mirror the ``onnx.backend.test.case.node.negativeloglikelihoodloss
  // .NegativeLogLikelihoodLoss`` Python class). Inputs are generated
  // deterministically through the seeded ``Rand``/``RandInt`` helpers in lieu
  // of the upstream ``np.random.rand(...)`` / ``np.random.randint(...)``
  // patterns (the value seeds differ from NumPy, but the registry remains
  // self-consistent because expected outputs are computed by
  // ``kernel::NegativeLogLikelihoodLoss``). Each case below mirrors one
  // ``NegativeLogLikelihoodLoss.export_*`` method.
  // -------------------------------------------------------------------------
  struct NllCase {
    std::string name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> target_shape;
    std::string reduction;
    bool with_weight;
    bool has_ignore_index;
    int64_t ignore_index;
    // Index into the target flat buffer that must be overwritten with
    // ``ignore_index`` to exercise the masking path; -1 means "do not patch".
    int64_t patched_target_flat_index;
    uint64_t input_seed;
    uint64_t target_seed;
    uint64_t weight_seed;
  };

  const int64_t N = 3;
  const int64_t C = 5;
  const std::vector<NllCase> nll_cases = {
      // From NegativeLogLikelihoodLoss.export_input_shape_is_NC():
      {"test_nllloss_NC", {N, C}, {N}, "none", false, false, 0, -1, 100, 101, 0},
      // From export_input_shape_is_NCd1d2():
      {"test_nllloss_NCd1d2", {N, C, 6, 6}, {N, 6, 6}, "none", false, false, 0, -1, 102, 103, 0},
      // From export_input_shape_is_NCd1d2_reduction_mean():
      {"test_nllloss_NCd1d2_reduction_mean",
       {N, C, 6, 6},
       {N, 6, 6},
       "mean",
       false,
       false,
       0,
       -1,
       104,
       105,
       0},
      // From export_input_shape_is_NCd1d2_reduction_sum():
      {"test_nllloss_NCd1d2_reduction_sum",
       {N, C, 6, 6},
       {N, 6, 6},
       "sum",
       false,
       false,
       0,
       -1,
       106,
       107,
       0},
      // From export_input_shape_is_NCd1d2_with_weight():
      {"test_nllloss_NCd1d2_with_weight",
       {N, C, 6, 6},
       {N, 6, 6},
       "none",
       true,
       false,
       0,
       -1,
       108,
       109,
       110},
      // From export_input_shape_is_NCd1d2_with_weight_reduction_mean():
      {"test_nllloss_NCd1d2_with_weight_reduction_mean",
       {N, C, 6, 6},
       {N, 6, 6},
       "mean",
       true,
       false,
       0,
       -1,
       111,
       112,
       113},
      // From export_input_shape_is_NCd1d2_with_weight_reduction_sum():
      {"test_nllloss_NCd1d2_with_weight_reduction_sum",
       {N, C, 6, 6},
       {N, 6, 6},
       "sum",
       true,
       false,
       0,
       -1,
       114,
       115,
       116},
      // From export_input_shape_is_NCd1d2_with_weight_reduction_sum_ii():
      // target[0][0][0] is patched to ``ignore_index`` (== 0); flat index 0.
      {"test_nllloss_NCd1d2_with_weight_reduction_sum_ii",
       {N, C, 6, 6},
       {N, 6, 6},
       "sum",
       true,
       true,
       0,
       0,
       117,
       118,
       119},
      // From export_input_shape_is_NCd1d2_no_weight_reduction_mean_ii():
      // target[0][0][0] is patched to ``ignore_index`` (== 1); flat index 0.
      {"test_nllloss_NCd1d2_no_weight_reduction_mean_ii",
       {N, C, 6, 6},
       {N, 6, 6},
       "mean",
       false,
       true,
       1,
       0,
       120,
       121,
       0},
      // From export_input_shape_is_NCd1():
      {"test_nllloss_NCd1", {N, C, 2}, {N, 2}, "mean", false, false, 0, -1, 122, 123, 0},
      // From export_input_shape_is_NCd1_weight():
      {"test_nllloss_NCd1_weight", {N, C, 2}, {N, 2}, "mean", true, false, 0, -1, 124, 125, 126},
      // From export_input_shape_is_NCd1_ii():
      // target[0][0] is patched to ``ignore_index`` (== 1); flat index 0.
      {"test_nllloss_NCd1_ii", {N, C, 2}, {N, 2}, "mean", false, true, 1, 0, 127, 128, 0},
      // From export_input_shape_is_NCd1_weight_ii():
      // target[0][0] is patched to ``ignore_index`` (== 1); flat index 0.
      {"test_nllloss_NCd1_weight_ii", {N, C, 2}, {N, 2}, "mean", true, true, 1, 0, 129, 130, 131},
      // From export_input_shape_is_NCd1d2d3d4d5_mean_weight():
      {"test_nllloss_NCd1d2d3d4d5_mean_weight",
       {N, C, 6, 6, 5, 3, 4},
       {N, 6, 6, 5, 3, 4},
       "mean",
       true,
       false,
       0,
       -1,
       132,
       133,
       134},
      // From export_input_shape_is_NCd1d2d3d4d5_none_no_weight():
      {"test_nllloss_NCd1d2d3d4d5_none_no_weight",
       {N, C, 6, 6, 5, 3, 4},
       {N, 6, 6, 5, 3, 4},
       "none",
       false,
       false,
       0,
       -1,
       135,
       136,
       0},
      // From export_input_shape_is_NCd1_mean_weight_negative_ii():
      // target[0][0] is patched to ``ignore_index`` (== -1); flat index 0.
      {"test_nllloss_NCd1_mean_weight_negative_ii",
       {N, C, 6},
       {N, 6},
       "mean",
       true,
       true,
       -1,
       0,
       137,
       138,
       139},
      // From export_input_shape_is_NCd1d2d3_none_no_weight_negative_ii():
      // target[0][0][0][0] is patched to ``ignore_index`` (== -5); flat index 0.
      {"test_nllloss_NCd1d2d3_none_no_weight_negative_ii",
       {N, C, 6, 6, 5},
       {N, 6, 6, 5},
       "none",
       false,
       true,
       -5,
       0,
       140,
       141,
       0},
      // From export_input_shape_is_NCd1d2d3_sum_weight_high_ii():
      // Note: the upstream test uses input shape (N, C) and patches target[0]
      // to ``ignore_index`` (== 10, > C-1); flat index 0.
      {"test_nllloss_NCd1d2d3_sum_weight_high_ii",
       {N, C},
       {N},
       "sum",
       true,
       true,
       10,
       0,
       142,
       143,
       144},
  };

  for (const auto &c : nll_cases) {
    NodeProto node;
    node.set_op_type("NegativeLogLikelihoodLoss");
    node.add_input("input");
    node.add_input("target");
    if (c.with_weight) {
      node.add_input("weight");
    }
    node.add_output("loss");
    if (c.reduction != "mean") {
      AddAttribute(node, "reduction", c.reduction);
    }
    if (c.has_ignore_index) {
      AddAttribute(node, "ignore_index", c.ignore_index);
    }

    Tensor input = Tensor::FromFloat("", c.input_shape, Rand<float>(c.input_shape, c.input_seed));
    std::vector<int64_t> target_data = RandInt(0, C, c.target_shape, c.target_seed);
    if (c.has_ignore_index && c.patched_target_flat_index >= 0) {
      target_data[static_cast<size_t>(c.patched_target_flat_index)] = c.ignore_index;
    }
    Tensor target = Tensor::FromInt64("", c.target_shape, target_data);

    std::vector<Tensor> inputs;
    inputs.reserve(c.with_weight ? 3 : 2);
    inputs.push_back(input);
    inputs.push_back(target);
    Tensor weight;
    if (c.with_weight) {
      weight = Tensor::FromFloat("", {C}, Rand<float>({C}, c.weight_seed));
      inputs.push_back(weight);
    }

    const Tensor *weight_ptr = c.with_weight ? &inputs[2] : nullptr;
    Tensor loss =
        nll_kernel(input, target, weight_ptr, c.reduction, c.has_ignore_index, c.ignore_index);

    Expect(node, inputs, {std::move(loss)}, c.name, {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
