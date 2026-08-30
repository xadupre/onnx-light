// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/training/include_training_cases.h"
#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Opset id for the ``ai.onnx.preview.training`` domain at the only released
// version of Adam (v1). Mirrors
// ``onnx_op::training::kOnnxPreviewTrainingDomain`` but is duplicated here so
// this library does not need to depend on ``lib_onnx_op``.
constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

OpsetId TrainingOpset(int64_t version) { return OpsetId(kOnnxPreviewTrainingDomain, version); }

} // namespace

// ---------------------------------------------------------------------------
// Adam — one iteration of the Adam stochastic gradient optimization
// algorithm (since opset 1 in the ``ai.onnx.preview.training`` domain).
//
// Two cases are registered, both exercising the full ``norm_coefficient``,
// ``alpha``, ``beta``, ``epsilon`` and ``norm_coefficient_post`` pseudo-code
// from the operator's schema:
//
//   * ``test_cc_adam_single`` — Single optimized rank-1 tensor with the
//     un-corrected learning rate path (``T == 0``).
//   * ``test_cc_adam_multiple`` — Two optimized tensors of different ranks
//     with the bias-corrected learning rate path (``T > 0``).
//
// Inputs are small, fully deterministic tensors so this library does not
// depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAdamCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = TrainingOpset(1);
  const OpsetId default_opset = DefaultOpset(13);
  const auto adam = MakeReferenceKernel<onnx_kernels::kernel::Adam>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V", "H"});
    AddOutputs(node, {"X_new", "V_new", "H_new"});

    const float alpha = 0.95f;
    const float beta = 0.9f;
    const float epsilon = 1e-2f;
    const float norm_coefficient = 0.001f;
    const float norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddFloatAttribute(node, "epsilon", epsilon);
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "norm_coefficient_post", norm_coefficient_post);

    Expect(registry, std::move(node), "test_cc_adam_single_benchmark", {default_opset, opset},
           {1, 1, kBenchmarkElementwiseSize, kBenchmarkElementwiseSize, kBenchmarkElementwiseSize,
            kBenchmarkElementwiseSize},
           {kBenchmarkElementwiseSize, kBenchmarkElementwiseSize, kBenchmarkElementwiseSize},
           [adam, alpha, beta, epsilon, norm_coefficient, norm_coefficient_post]() -> IoData {
             Tensor R = Tensor::FromFloat("", {}, {0.1f});
             Tensor T = Tensor::FromInt64("", {}, {0});
             Tensor X = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654321ULL);
             Tensor G = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654322ULL);
             Tensor V = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654323ULL);
             Tensor H = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654324ULL);
             std::vector<Tensor> outs = adam.Invoke([&](const auto &kernel) {
               return kernel(R, T, {X}, {G}, {V}, {H}, alpha, beta, epsilon, norm_coefficient,
                             norm_coefficient_post);
             });
             return IoData{{std::move(R), std::move(T), std::move(X), std::move(G), std::move(V),
                            std::move(H)},
                           {std::move(outs[0]), std::move(outs[1]), std::move(outs[2])}};
           });
    return;
  }

  // ----- Case 1: single optimized tensor, T == 0.
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V", "H"});
    AddOutputs(node, {"X_new", "V_new", "H_new"});

    const float alpha = 0.95f;
    const float beta = 0.9f;
    const float epsilon = 1e-2f;
    const float norm_coefficient = 0.001f;
    const float norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddFloatAttribute(node, "epsilon", epsilon);
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "norm_coefficient_post", norm_coefficient_post);
    Expect(registry, std::move(node), "test_cc_adam_single", {default_opset, opset},
           [=]() -> IoData {
             Tensor R = Tensor::FromFloat("", {}, {0.1f});
             Tensor T = Tensor::FromInt64("", {}, {0});
             Tensor X = Tensor::FromFloat("", {3}, {1.0f, 2.0f, -1.0f});
             Tensor G = Tensor::FromFloat("", {3}, {0.5f, -0.5f, 0.25f});
             Tensor V = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
             Tensor H = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});

             std::vector<Tensor> outs = adam.Invoke([&](const auto &kernel) {
               return kernel(R, T, {X}, {G}, {V}, {H}, alpha, beta, epsilon, norm_coefficient,
                             norm_coefficient_post);
             });
             return IoData{{std::move(R), std::move(T), std::move(X), std::move(G), std::move(V),
                            std::move(H)},
                           {std::move(outs[0]), std::move(outs[1]), std::move(outs[2])}};
           });
  }

  // ----- Case 2: two optimized tensors of different ranks, T > 0 (uses the
  // bias-corrected learning rate path).
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "V1_new", "V2_new", "H1_new", "H2_new"});

    const float alpha = 0.9f;
    const float beta = 0.999f;
    const float epsilon = 1e-6f;
    const float norm_coefficient = 0.0f;
    const float norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddFloatAttribute(node, "epsilon", epsilon);
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "norm_coefficient_post", norm_coefficient_post);
    Expect(registry, std::move(node), "test_cc_adam_multiple", {default_opset, opset},
           [=]() -> IoData {
             Tensor R = Tensor::FromFloat("", {}, {0.05f});
             Tensor T = Tensor::FromInt64("", {}, {5});
             Tensor X1 = Tensor::FromFloat("", {2}, {0.5f, -0.5f});
             Tensor X2 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor G1 = Tensor::FromFloat("", {2}, {0.1f, -0.2f});
             Tensor G2 = Tensor::FromFloat("", {2, 2}, {-0.5f, 0.25f, 0.75f, -1.0f});
             Tensor V1 = Tensor::FromFloat("", {2}, {0.01f, 0.02f});
             Tensor V2 = Tensor::FromFloat("", {2, 2}, {0.05f, 0.05f, -0.05f, 0.0f});
             Tensor H1 = Tensor::FromFloat("", {2}, {0.001f, 0.002f});
             Tensor H2 = Tensor::FromFloat("", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});

             std::vector<Tensor> outs = adam.Invoke([&](const auto &kernel) {
               return kernel(R, T, {X1, X2}, {G1, G2}, {V1, V2}, {H1, H2}, alpha, beta, epsilon,
                             norm_coefficient, norm_coefficient_post);
             });
             return IoData{{std::move(R), std::move(T), std::move(X1), std::move(X2), std::move(G1),
                            std::move(G2), std::move(V1), std::move(V2), std::move(H1),
                            std::move(H2)},
                           {std::move(outs[0]), std::move(outs[1]), std::move(outs[2]),
                            std::move(outs[3]), std::move(outs[4]), std::move(outs[5])}};
           });
  }

  // ----- Upstream ONNX cases (mirror onnx.backend.test.case.node.adam.Adam).
  // These use the exact inputs and attributes from the upstream Python test
  // case, with expected outputs recomputed by ``kernel::Adam``. Only the four
  // attributes set by the upstream node (``norm_coefficient``, ``alpha``,
  // ``beta``, ``epsilon``) are written; ``norm_coefficient_post`` keeps its
  // schema default of 0.

  // From Adam.export_adam():
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V", "H"});
    AddOutputs(node, {"X_new", "V_new", "H_new"});

    const float alpha = 0.95f;
    const float beta = 0.1f;
    const float epsilon = 1e-7f;
    const float norm_coefficient = 0.001f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddFloatAttribute(node, "epsilon", epsilon);
    Expect(registry, std::move(node), "test_adam", {default_opset, opset}, [=]() -> IoData {
      Tensor R = Tensor::FromFloat("", {}, {0.1f});
      Tensor T = Tensor::FromInt64("", {}, {0});
      Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
      Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
      Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});
      Tensor H = Tensor::FromFloat("", {2}, {0.1f, 0.1f});

      std::vector<Tensor> outs = adam.Invoke([&](const auto &kernel) {
        return kernel(R, T, {X}, {G}, {V}, {H}, alpha, beta, epsilon, norm_coefficient);
      });
      return IoData{
          {std::move(R), std::move(T), std::move(X), std::move(G), std::move(V), std::move(H)},
          {std::move(outs[0]), std::move(outs[1]), std::move(outs[2])}};
    });
  }

  // From Adam.export_adam_multiple():
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "V1_new", "V2_new", "H1_new", "H2_new"});

    const float alpha = 0.95f;
    const float beta = 0.85f;
    const float epsilon = 1e-2f;
    const float norm_coefficient = 0.001f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "alpha", alpha);
    AddFloatAttribute(node, "beta", beta);
    AddFloatAttribute(node, "epsilon", epsilon);
    Expect(
        registry, std::move(node), "test_adam_multiple", {default_opset, opset}, [=]() -> IoData {
          Tensor R = Tensor::FromFloat("", {}, {0.1f});
          Tensor T = Tensor::FromInt64("", {}, {0});
          Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
          Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
          Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
          Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
          Tensor V1 = Tensor::FromFloat("", {1}, {2.0f});
          Tensor V2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});
          Tensor H1 = Tensor::FromFloat("", {1}, {0.5f});
          Tensor H2 = Tensor::FromFloat("", {2}, {1.0f, 10.0f});

          std::vector<Tensor> outs = adam.Invoke([&](const auto &kernel) {
            return kernel(R, T, {X1, X2}, {G1, G2}, {V1, V2}, {H1, H2}, alpha, beta, epsilon,
                          norm_coefficient);
          });
          return IoData{{std::move(R), std::move(T), std::move(X1), std::move(X2), std::move(G1),
                         std::move(G2), std::move(V1), std::move(V2), std::move(H1), std::move(H2)},
                        {std::move(outs[0]), std::move(outs[1]), std::move(outs[2]),
                         std::move(outs[3]), std::move(outs[4]), std::move(outs[5])}};
        });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
