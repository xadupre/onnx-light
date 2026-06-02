// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/kernels/training/include_training_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// See ``cases_adam.cc`` for the rationale on duplicating the domain
// constant rather than depending on ``lib_onnx_op``.
constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

OpsetId TrainingOpset(int64_t version) { return OpsetId(kOnnxPreviewTrainingDomain, version); }

} // namespace

// ---------------------------------------------------------------------------
// Momentum — one iteration of the SGD-with-Momentum stochastic gradient
// optimization algorithm (since opset 1 in the
// ``ai.onnx.preview.training`` domain).
//
// Mirrors the upstream Python cases produced by
// ``onnx.backend.test.case.node.momentum.Momentum``:
//
//   * ``test_momentum``           — Single optimized rank-1 tensor,
//                                   ``mode == "standard"``.
//   * ``test_nesterov_momentum``  — Single optimized rank-1 tensor,
//                                   ``mode == "nesterov"``.
//   * ``test_momentum_multiple``  — Two optimized rank-1 tensors of
//                                   lengths 1 and 2, ``mode == "standard"``.
//
// All cases set the four attributes ``norm_coefficient``, ``alpha``,
// ``beta`` (FLOAT) and ``mode`` (STRING) and use the first-iteration
// path (``T == 0``).
// ---------------------------------------------------------------------------
void RegisterMomentumCases(std::vector<TestCase> &registry) {
  const OpsetId opset = TrainingOpset(1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Momentum momentum{ctx};

  // From Momentum.export_momentum():
  {
    NodeProto node;
    node.set_op_type("Momentum");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V"});
    AddOutputs(node, {"X_new", "V_new"});

    const float norm_coefficient = 0.001f;
    const float alpha = 0.95f;
    const float beta = 0.1f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddAttribute(node, "mode", std::string("standard"));

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
    Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
    Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});

    std::vector<Tensor> outs = momentum(R, T, {X}, {G}, {V}, alpha, beta, norm_coefficient,
                                        kernel::Momentum::Mode::kStandard);
    Expect(node, {R, T, X, G, V}, {outs[0], outs[1]}, "test_momentum", {default_opset, opset},
           "backend-test", registry);
  }

  // From Momentum.export_nesterov_momentum():
  {
    NodeProto node;
    node.set_op_type("Momentum");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V"});
    AddOutputs(node, {"X_new", "V_new"});

    const float norm_coefficient = 0.01f;
    const float alpha = 0.95f;
    const float beta = 1.0f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddAttribute(node, "mode", std::string("nesterov"));

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
    Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
    Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});

    std::vector<Tensor> outs = momentum(R, T, {X}, {G}, {V}, alpha, beta, norm_coefficient,
                                        kernel::Momentum::Mode::kNesterov);
    Expect(node, {R, T, X, G, V}, {outs[0], outs[1]}, "test_nesterov_momentum",
           {default_opset, opset}, "backend-test", registry);
  }

  // From Momentum.export_momentum_multiple():
  {
    NodeProto node;
    node.set_op_type("Momentum");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "V1_new", "V2_new"});

    const float norm_coefficient = 0.001f;
    const float alpha = 0.95f;
    const float beta = 0.85f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddAttribute(node, "mode", std::string("standard"));

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
    Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
    Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
    Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
    Tensor V1 = Tensor::FromFloat("", {1}, {2.0f});
    Tensor V2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});

    std::vector<Tensor> outs = momentum(R, T, {X1, X2}, {G1, G2}, {V1, V2}, alpha, beta,
                                        norm_coefficient, kernel::Momentum::Mode::kStandard);
    Expect(node, {R, T, X1, X2, G1, G2, V1, V2}, {outs[0], outs[1], outs[2], outs[3]},
           "test_momentum_multiple", {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
