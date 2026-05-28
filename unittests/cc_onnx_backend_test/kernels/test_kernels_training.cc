// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/training/include_training_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Adam;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

namespace {

KernelContext TrainingKernelContext() {
  return KernelContext(OpsetId("ai.onnx.preview.training", 1));
}

// Reference re-implementation of the Adam pseudo-code from the ONNX schema,
// kept intentionally separate from the kernel under test so the test
// double-checks the kernel against an independent formula.
void AdamReference(float R, int64_t T_val, const std::vector<float> &X, const std::vector<float> &G,
                   const std::vector<float> &V, const std::vector<float> &H, float alpha,
                   float beta, float epsilon, float norm_coefficient, float norm_coefficient_post,
                   std::vector<float> &X_out, std::vector<float> &V_out,
                   std::vector<float> &H_out) {
  X_out.resize(X.size());
  V_out.resize(X.size());
  H_out.resize(X.size());

  float R_adjusted = R;
  if (T_val > 0) {
    const double ap = std::pow(static_cast<double>(alpha), static_cast<double>(T_val));
    const double bp = std::pow(static_cast<double>(beta), static_cast<double>(T_val));
    R_adjusted = static_cast<float>(static_cast<double>(R) * std::sqrt(1.0 - bp) / (1.0 - ap));
  }
  for (size_t k = 0; k < X.size(); ++k) {
    const double x = X[k];
    const double g = G[k];
    const double v = V[k];
    const double h = H[k];
    const double g_reg = static_cast<double>(norm_coefficient) * x + g;
    const double v_new =
        static_cast<double>(alpha) * v + (1.0 - static_cast<double>(alpha)) * g_reg;
    const double h_new =
        static_cast<double>(beta) * h + (1.0 - static_cast<double>(beta)) * g_reg * g_reg;
    const double h_sqrt = std::sqrt(h_new) + static_cast<double>(epsilon);
    const double x_new = x - static_cast<double>(R_adjusted) * v_new / h_sqrt;
    const double x_final = (1.0 - static_cast<double>(norm_coefficient_post)) * x_new;
    X_out[k] = static_cast<float>(x_final);
    V_out[k] = static_cast<float>(v_new);
    H_out[k] = static_cast<float>(h_new);
  }
}

} // namespace

TEST(BackendKernelClass, AdamSingleVariableMatchesReferenceUncorrected) {
  const KernelContext ctx = TrainingKernelContext();
  const Adam adam{ctx};
  const float alpha = 0.95f;
  const float beta = 0.9f;
  const float epsilon = 1e-2f;
  const float norm_coefficient = 0.001f;
  const float norm_coefficient_post = 0.0f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {3}, {1.0f, 2.0f, -1.0f});
  const Tensor G = Tensor::FromFloat("", {3}, {0.5f, -0.5f, 0.25f});
  const Tensor V = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  const Tensor H = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});

  const std::vector<Tensor> outs =
      adam(R, T, {X}, {G}, {V}, {H}, alpha, beta, epsilon, norm_coefficient, norm_coefficient_post);
  ASSERT_EQ(outs.size(), 3u);
  ASSERT_EQ(outs[0].shape, X.shape);
  ASSERT_EQ(outs[1].shape, V.shape);
  ASSERT_EQ(outs[2].shape, H.shape);

  std::vector<float> X_ref, V_ref, H_ref;
  AdamReference(0.1f, 0, {1.0f, 2.0f, -1.0f}, {0.5f, -0.5f, 0.25f}, {0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f}, alpha, beta, epsilon, norm_coefficient, norm_coefficient_post,
                X_ref, V_ref, H_ref);
  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[1].AsFloat()[k], V_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[2].AsFloat()[k], H_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(BackendKernelClass, AdamMultiVariableMatchesReferenceBiasCorrected) {
  const KernelContext ctx = TrainingKernelContext();
  const Adam adam{ctx};
  // Defaults from the ONNX schema.
  const float alpha = 0.9f;
  const float beta = 0.999f;
  const float epsilon = 1e-6f;
  const float norm_coefficient = 0.0f;
  const float norm_coefficient_post = 0.0f;

  const Tensor R = Tensor::FromFloat("", {}, {0.05f});
  const Tensor T = Tensor::FromInt64("", {}, {5});
  const Tensor X1 = Tensor::FromFloat("", {2}, {0.5f, -0.5f});
  const Tensor X2 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor G1 = Tensor::FromFloat("", {2}, {0.1f, -0.2f});
  const Tensor G2 = Tensor::FromFloat("", {2, 2}, {-0.5f, 0.25f, 0.75f, -1.0f});
  const Tensor V1 = Tensor::FromFloat("", {2}, {0.01f, 0.02f});
  const Tensor V2 = Tensor::FromFloat("", {2, 2}, {0.05f, 0.05f, -0.05f, 0.0f});
  const Tensor H1 = Tensor::FromFloat("", {2}, {0.001f, 0.002f});
  const Tensor H2 = Tensor::FromFloat("", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});

  const std::vector<Tensor> outs = adam(R, T, {X1, X2}, {G1, G2}, {V1, V2}, {H1, H2}, alpha, beta,
                                        epsilon, norm_coefficient, norm_coefficient_post);
  ASSERT_EQ(outs.size(), 6u);
  ASSERT_EQ(outs[0].shape, X1.shape);
  ASSERT_EQ(outs[1].shape, X2.shape);
  ASSERT_EQ(outs[2].shape, V1.shape);
  ASSERT_EQ(outs[3].shape, V2.shape);
  ASSERT_EQ(outs[4].shape, H1.shape);
  ASSERT_EQ(outs[5].shape, H2.shape);

  std::vector<float> X1_ref, V1_ref, H1_ref;
  AdamReference(0.05f, 5, {0.5f, -0.5f}, {0.1f, -0.2f}, {0.01f, 0.02f}, {0.001f, 0.002f}, alpha,
                beta, epsilon, norm_coefficient, norm_coefficient_post, X1_ref, V1_ref, H1_ref);
  std::vector<float> X2_ref, V2_ref, H2_ref;
  AdamReference(0.05f, 5, {1.0f, 2.0f, 3.0f, 4.0f}, {-0.5f, 0.25f, 0.75f, -1.0f},
                {0.05f, 0.05f, -0.05f, 0.0f}, {0.01f, 0.02f, 0.03f, 0.04f}, alpha, beta, epsilon,
                norm_coefficient, norm_coefficient_post, X2_ref, V2_ref, H2_ref);

  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X1_ref[static_cast<size_t>(k)], 1e-5);
    EXPECT_NEAR(outs[2].AsFloat()[k], V1_ref[static_cast<size_t>(k)], 1e-5);
    EXPECT_NEAR(outs[4].AsFloat()[k], H1_ref[static_cast<size_t>(k)], 1e-5);
  }
  for (int64_t k = 0; k < outs[1].element_count(); ++k) {
    EXPECT_NEAR(outs[1].AsFloat()[k], X2_ref[static_cast<size_t>(k)], 1e-5);
    EXPECT_NEAR(outs[3].AsFloat()[k], V2_ref[static_cast<size_t>(k)], 1e-5);
    EXPECT_NEAR(outs[5].AsFloat()[k], H2_ref[static_cast<size_t>(k)], 1e-5);
  }
}

TEST(BackendKernelClass, AdamPostNormCoefficientScalesXNew) {
  const KernelContext ctx = TrainingKernelContext();
  const Adam adam{ctx};
  const float alpha = 0.9f;
  const float beta = 0.999f;
  const float epsilon = 1e-6f;
  const float norm_coefficient = 0.0f;
  const float norm_coefficient_post = 0.25f;

  const Tensor R = Tensor::FromFloat("", {}, {0.0f}); // R == 0 so X_new == X.
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {4.0f, -8.0f});
  const Tensor G = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor V = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor H = Tensor::FromFloat("", {2}, {0.0f, 0.0f});

  const std::vector<Tensor> outs =
      adam(R, T, {X}, {G}, {V}, {H}, alpha, beta, epsilon, norm_coefficient, norm_coefficient_post);
  ASSERT_EQ(outs.size(), 3u);
  // X_final = (1 - 0.25) * X = 0.75 * X.
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[1], -6.0f);
}

TEST(BackendKernelClass, AdamRejectsInvalidInputs) {
  const KernelContext ctx = TrainingKernelContext();
  const Adam adam{ctx};

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor G = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor V = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor H = Tensor::FromFloat("", {2}, {0.0f, 0.0f});

  // Empty optimized-tensor list is rejected.
  EXPECT_THROW(adam(R, T, {}, {}, {}, {}), std::invalid_argument);

  // Mismatched list lengths are rejected.
  EXPECT_THROW(adam(R, T, {X}, {G, G}, {V}, {H}), std::invalid_argument);

  // Non-FLOAT 'R' is rejected.
  Tensor bad_R("", TensorProto::DataType::INT64, {}, std::vector<uint8_t>(sizeof(int64_t)));
  EXPECT_THROW(adam(bad_R, T, {X}, {G}, {V}, {H}), std::invalid_argument);

  // Non-INT64 'T' is rejected.
  const Tensor bad_T = Tensor::FromFloat("", {}, {0.0f});
  EXPECT_THROW(adam(R, bad_T, {X}, {G}, {V}, {H}), std::invalid_argument);

  // Mismatched shapes within a single optimized tensor are rejected.
  const Tensor G_bad = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  EXPECT_THROW(adam(R, T, {X}, {G_bad}, {V}, {H}), std::invalid_argument);
}

} // namespace Test
