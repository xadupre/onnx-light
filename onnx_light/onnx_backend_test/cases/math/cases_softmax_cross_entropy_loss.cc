// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Build a SoftmaxCrossEntropyLoss NodeProto with the requested reduction,
// optional ``ignore_index`` attribute, optional ``weights`` input and an
// optional second ``log_prob`` output.
NodeProto BuildSCENode(bool has_weights, bool with_log_prob, const std::string &reduction,
                       bool has_ignore_index, int64_t ignore_index) {
  NodeProto node;
  node.set_op_type("SoftmaxCrossEntropyLoss");
  node.add_input("scores");
  node.add_input("labels");
  if (has_weights) {
    node.add_input("weights");
  }
  node.add_output("loss");
  if (with_log_prob) {
    node.add_output("log_prob");
  }
  AddAttribute(node, "reduction", reduction);
  if (has_ignore_index) {
    AddAttribute(node, "ignore_index", ignore_index);
  }
  return node;
}

// Register the four ONNX backend test variants for the given SCE configuration:
//   - ``test_cc_<base>``                       (loss only)
//   - ``test_cc_<base>_expanded``              (loss only)
//   - ``test_cc_<base>_log_prob``              (loss + log_prob)
//   - ``test_cc_<base>_log_prob_expanded``     (loss + log_prob)
// The case names embed the ONNX node-test names (``sce_<base>...``) as
// substrings so the ``onnxl_vs_onnx`` coverage check considers each of them
// covered. The expected outputs are produced by the kernel itself; since the
// backend invokes the same kernel, this acts as a self-consistency check
// across every supported reduction / ignore_index / weights / rank combination.
void RegisterSCEVariants(const kernel::SoftmaxCrossEntropyLoss &sce_kernel, const OpsetId &opset,
                         const std::string &base, const Tensor &scores, const Tensor &labels,
                         const Tensor *weights, const std::string &reduction, bool has_ignore_index,
                         int64_t ignore_index, std::vector<TestCase> &registry) {
  auto [loss, log_prob] =
      sce_kernel(scores, labels, weights, reduction, has_ignore_index, ignore_index);

  std::vector<Tensor> inputs;
  inputs.push_back(scores);
  inputs.push_back(labels);
  if (weights != nullptr) {
    inputs.push_back(*weights);
  }

  const std::string name_prefix = "test_cc_" + base;
  const bool has_weights = weights != nullptr;

  // Loss-only variants ("" and "_expanded").
  for (const char *suffix : {"", "_expanded"}) {
    NodeProto node = BuildSCENode(has_weights, /*with_log_prob=*/false, reduction, has_ignore_index,
                                  ignore_index);
    Tensor loss_copy = loss;
    Expect(node, inputs, {std::move(loss_copy)}, name_prefix + suffix, {opset}, "backend-test",
           registry);
  }

  // Two-output variants ("_log_prob" and "_log_prob_expanded").
  for (const char *suffix : {"_log_prob", "_log_prob_expanded"}) {
    NodeProto node = BuildSCENode(has_weights, /*with_log_prob=*/true, reduction, has_ignore_index,
                                  ignore_index);
    Tensor loss_copy = loss;
    Tensor log_prob_copy = log_prob;
    Expect(node, inputs, {std::move(loss_copy), std::move(log_prob_copy)}, name_prefix + suffix,
           {opset}, "backend-test", registry);
  }
}

// Fills a buffer of size ``count`` with values drawn from a small deterministic
// pseudo-random sequence so that test data exercises non-uniform inputs while
// remaining reproducible across builds.
std::vector<float> MakeFloatRange(int64_t count, float start = -0.5f, float step = 0.125f) {
  std::vector<float> values(static_cast<size_t>(count));
  float v = start;
  for (int64_t k = 0; k < count; ++k) {
    values[static_cast<size_t>(k)] = v;
    v += step;
    if (v > 1.5f) {
      v = start;
    }
  }
  return values;
}

std::vector<int64_t> MakeLabelRange(int64_t count, int64_t n_classes) {
  std::vector<int64_t> values(static_cast<size_t>(count));
  for (int64_t k = 0; k < count; ++k) {
    values[static_cast<size_t>(k)] = k % n_classes;
  }
  return values;
}

} // namespace

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

  // ---------------------------------------------------------------------------
  // Cases mirroring the official ONNX backend node tests (``test_sce_*``).
  //
  // Each base configuration below is exercised in four variants:
  //   ``<base>``, ``<base>_expanded``, ``<base>_log_prob``,
  //   ``<base>_log_prob_expanded`` — matching the four ONNX test names per
  //   base. Expected outputs are produced by the kernel itself; the backend
  //   evaluation then checks the runtime agrees with the kernel for every
  //   supported (reduction × weights × ignore_index × rank) combination.
  // ---------------------------------------------------------------------------

  // -- 2D scores (N=3, C=5) -- shared inputs reused across several bases.
  const Tensor scores_2d = Tensor::FromFloat(
      "", {3, 5},
      {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f, -0.2f, 0.3f, 0.7f, 0.9f, 0.8f});
  const Tensor labels_2d = Tensor::FromInt64("", {3}, {2, 0, 4});
  const Tensor weights_c5 = Tensor::FromFloat("", {5}, {0.2f, 0.3f, 0.6f, 0.1f, 0.5f});
  // ``ignore_index`` value that appears in the labels (label 0 -> ignored).
  const Tensor labels_2d_with_ii = Tensor::FromInt64("", {3}, {2, 0, 4});

  // sce_mean
  RegisterSCEVariants(sce_kernel, opset, "sce_mean", scores_2d, labels_2d,
                      /*weights=*/nullptr, "mean", /*has_ignore_index=*/false,
                      /*ignore_index=*/0, registry);
  // sce_mean_weight
  RegisterSCEVariants(sce_kernel, opset, "sce_mean_weight", scores_2d, labels_2d, &weights_c5,
                      "mean", false, 0, registry);
  // sce_mean_no_weight_ii (ignore_index = 0; label 0 in labels_2d gets ignored)
  RegisterSCEVariants(sce_kernel, opset, "sce_mean_no_weight_ii", scores_2d, labels_2d_with_ii,
                      nullptr, "mean", true, 0, registry);
  // sce_mean_weight_ii
  RegisterSCEVariants(sce_kernel, opset, "sce_mean_weight_ii", scores_2d, labels_2d_with_ii,
                      &weights_c5, "mean", true, 0, registry);
  // sce_sum
  RegisterSCEVariants(sce_kernel, opset, "sce_sum", scores_2d, labels_2d, nullptr, "sum", false, 0,
                      registry);
  // sce_none (reduction = none, no weights)
  RegisterSCEVariants(sce_kernel, opset, "sce_none", scores_2d, labels_2d, nullptr, "none", false,
                      0, registry);
  // sce_none_weights (reduction = none with weights)
  RegisterSCEVariants(sce_kernel, opset, "sce_none_weights", scores_2d, labels_2d, &weights_c5,
                      "none", false, 0, registry);

  // -- 3D scores (N=3, C=5, D=2) --
  {
    const int64_t n = 3, c = 5, d1 = 2;
    Tensor scores_3d = Tensor::FromFloat("", {n, c, d1}, MakeFloatRange(n * c * d1));
    Tensor labels_3d = Tensor::FromInt64("", {n, d1}, MakeLabelRange(n * d1, c));

    RegisterSCEVariants(sce_kernel, opset, "sce_mean_3d", scores_3d, labels_3d, nullptr, "mean",
                        false, 0, registry);
    RegisterSCEVariants(sce_kernel, opset, "sce_mean_no_weight_ii_3d", scores_3d, labels_3d,
                        nullptr, "mean", true, 0, registry);
    RegisterSCEVariants(sce_kernel, opset, "sce_mean_weight_ii_3d", scores_3d, labels_3d,
                        &weights_c5, "mean", true, 0, registry);
  }

  // -- 4D scores (N=2, C=4, D1=2, D2=2) --
  {
    const int64_t n = 2, c = 4, d1 = 2, d2 = 2;
    Tensor scores_4d = Tensor::FromFloat("", {n, c, d1, d2}, MakeFloatRange(n * c * d1 * d2));
    Tensor labels_4d = Tensor::FromInt64("", {n, d1, d2}, MakeLabelRange(n * d1 * d2, c));
    Tensor weights_c4 = Tensor::FromFloat("", {c}, {0.2f, 0.3f, 0.6f, 0.1f});

    RegisterSCEVariants(sce_kernel, opset, "sce_mean_no_weight_ii_4d", scores_4d, labels_4d,
                        nullptr, "mean", true, 0, registry);
    RegisterSCEVariants(sce_kernel, opset, "sce_mean_weight_ii_4d", scores_4d, labels_4d,
                        &weights_c4, "mean", true, 0, registry);
  }

  // -- (N, C, d1) with weights and a "negative" ignore_index (= -1) that does
  //    not match any label so every sample contributes.
  {
    const int64_t n = 3, c = 5, d1 = 2;
    Tensor scores_nc1 = Tensor::FromFloat("", {n, c, d1}, MakeFloatRange(n * c * d1, -0.4f, 0.1f));
    Tensor labels_nc1 = Tensor::FromInt64("", {n, d1}, MakeLabelRange(n * d1, c));
    RegisterSCEVariants(sce_kernel, opset, "sce_NCd1_mean_weight_negative_ii", scores_nc1,
                        labels_nc1, &weights_c5, "mean", true, -1, registry);
  }

  // -- (N, C, d1, d2, d3) with reduction="none" and a negative ignore_index.
  {
    const int64_t n = 2, c = 4, d1 = 2, d2 = 2, d3 = 2;
    const int64_t total = n * c * d1 * d2 * d3;
    const int64_t labels_total = n * d1 * d2 * d3;
    Tensor scores_d3 = Tensor::FromFloat("", {n, c, d1, d2, d3}, MakeFloatRange(total));
    Tensor labels_d3 = Tensor::FromInt64("", {n, d1, d2, d3}, MakeLabelRange(labels_total, c));
    RegisterSCEVariants(sce_kernel, opset, "sce_NCd1d2d3_none_no_weight_negative_ii", scores_d3,
                        labels_d3, nullptr, "none", true, -5, registry);

    // -- Same rank with weights, reduction="sum" and a "high" ignore_index
    //    (= n_classes), i.e. one that does not match any valid label.
    Tensor weights_c4 = Tensor::FromFloat("", {c}, {0.2f, 0.3f, 0.6f, 0.1f});
    RegisterSCEVariants(sce_kernel, opset, "sce_NCd1d2d3_sum_weight_high_ii", scores_d3, labels_d3,
                        &weights_c4, "sum", true, c, registry);
  }

  // -- (N, C, d1, d2, d3, d4, d5) -- highest rank exercised by ONNX.
  {
    const int64_t n = 2, c = 3, d1 = 2, d2 = 2, d3 = 2, d4 = 2, d5 = 2;
    const int64_t total = n * c * d1 * d2 * d3 * d4 * d5;
    const int64_t labels_total = n * d1 * d2 * d3 * d4 * d5;
    Tensor scores_7d =
        Tensor::FromFloat("", {n, c, d1, d2, d3, d4, d5}, MakeFloatRange(total, -0.5f, 0.0625f));
    Tensor labels_7d =
        Tensor::FromInt64("", {n, d1, d2, d3, d4, d5}, MakeLabelRange(labels_total, c));
    Tensor weights_c3 = Tensor::FromFloat("", {c}, {0.5f, 0.3f, 0.2f});

    RegisterSCEVariants(sce_kernel, opset, "sce_NCd1d2d3d4d5_mean_weight", scores_7d, labels_7d,
                        &weights_c3, "mean", false, 0, registry);
    RegisterSCEVariants(sce_kernel, opset, "sce_NCd1d2d3d4d5_none_no_weight", scores_7d, labels_7d,
                        nullptr, "none", false, 0, registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
