// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::OpsetId;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Adagrad;
using onnx_kernels::kernel::Adam;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Momentum;

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

TEST(KernelClass, AdamSingleVariableMatchesReferenceUncorrected) {
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

TEST(KernelClass, AdamMultiVariableMatchesReferenceBiasCorrected) {
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

TEST(KernelClass, AdamPostNormCoefficientScalesXNew) {
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

TEST(KernelClass, AdamRejectsInvalidInputs) {
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
  Tensor bad_R("", onnx_kernels::DataType::INT64, {}, std::vector<uint8_t>(sizeof(int64_t)));
  EXPECT_THROW(adam(bad_R, T, {X}, {G}, {V}, {H}), std::invalid_argument);

  // Non-INT64 'T' is rejected.
  const Tensor bad_T = Tensor::FromFloat("", {}, {0.0f});
  EXPECT_THROW(adam(R, bad_T, {X}, {G}, {V}, {H}), std::invalid_argument);

  // Mismatched shapes within a single optimized tensor are rejected.
  const Tensor G_bad = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  EXPECT_THROW(adam(R, T, {X}, {G_bad}, {V}, {H}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Adagrad kernel tests.
// ---------------------------------------------------------------------------

namespace {

// Reference re-implementation of the Adagrad pseudo-code from the ONNX
// schema, kept intentionally separate from the kernel under test.
void AdagradReference(float R, int64_t T_val, const std::vector<float> &X,
                      const std::vector<float> &G, const std::vector<float> &H, float epsilon,
                      float decay_factor, float norm_coefficient, std::vector<float> &X_out,
                      std::vector<float> &H_out) {
  X_out.resize(X.size());
  H_out.resize(X.size());
  const float R_adjusted =
      static_cast<float>(static_cast<double>(R) /
                         (1.0 + static_cast<double>(T_val) * static_cast<double>(decay_factor)));
  for (size_t k = 0; k < X.size(); ++k) {
    const double x = X[k];
    const double g = G[k];
    const double h = H[k];
    const double g_reg = static_cast<double>(norm_coefficient) * x + g;
    const double h_new = h + g_reg * g_reg;
    const double h_sqrt = std::sqrt(h_new) + static_cast<double>(epsilon);
    const double x_new = x - static_cast<double>(R_adjusted) * g_reg / h_sqrt;
    X_out[k] = static_cast<float>(x_new);
    H_out[k] = static_cast<float>(h_new);
  }
}

} // namespace

TEST(KernelClass, AdagradSingleVariableMatchesReference) {
  const KernelContext ctx = TrainingKernelContext();
  const Adagrad adagrad{ctx};
  const float norm_coefficient = 0.001f;
  const float epsilon = 1e-5f;
  const float decay_factor = 0.1f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {1}, {1.0f});
  const Tensor G = Tensor::FromFloat("", {1}, {-1.0f});
  const Tensor H = Tensor::FromFloat("", {1}, {2.0f});

  const std::vector<Tensor> outs =
      adagrad(R, T, {X}, {G}, {H}, epsilon, decay_factor, norm_coefficient);
  ASSERT_EQ(outs.size(), 2u);
  ASSERT_EQ(outs[0].shape, X.shape);
  ASSERT_EQ(outs[1].shape, H.shape);

  std::vector<float> X_ref, H_ref;
  AdagradReference(0.1f, 0, {1.0f}, {-1.0f}, {2.0f}, epsilon, decay_factor, norm_coefficient, X_ref,
                   H_ref);
  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[1].AsFloat()[k], H_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(KernelClass, AdagradMultiVariableMatchesReference) {
  const KernelContext ctx = TrainingKernelContext();
  const Adagrad adagrad{ctx};
  const float norm_coefficient = 0.001f;
  const float epsilon = 1e-5f;
  const float decay_factor = 0.1f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
  const Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
  const Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
  const Tensor H1 = Tensor::FromFloat("", {1}, {2.0f});
  const Tensor H2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});

  const std::vector<Tensor> outs =
      adagrad(R, T, {X1, X2}, {G1, G2}, {H1, H2}, epsilon, decay_factor, norm_coefficient);
  ASSERT_EQ(outs.size(), 4u);
  ASSERT_EQ(outs[0].shape, X1.shape);
  ASSERT_EQ(outs[1].shape, X2.shape);
  ASSERT_EQ(outs[2].shape, H1.shape);
  ASSERT_EQ(outs[3].shape, H2.shape);

  std::vector<float> X1_ref, H1_ref;
  AdagradReference(0.1f, 0, {1.0f}, {-1.0f}, {2.0f}, epsilon, decay_factor, norm_coefficient,
                   X1_ref, H1_ref);
  std::vector<float> X2_ref, H2_ref;
  AdagradReference(0.1f, 0, {1.0f, 2.0f}, {-1.0f, -3.0f}, {4.0f, 1.0f}, epsilon, decay_factor,
                   norm_coefficient, X2_ref, H2_ref);

  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X1_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[2].AsFloat()[k], H1_ref[static_cast<size_t>(k)], 1e-6);
  }
  for (int64_t k = 0; k < outs[1].element_count(); ++k) {
    EXPECT_NEAR(outs[1].AsFloat()[k], X2_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[3].AsFloat()[k], H2_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(KernelClass, AdagradAppliesDecayFactorAfterFirstIteration) {
  // With T > 0 the learning rate is divided by ``1 + T * decay_factor``.
  const KernelContext ctx = TrainingKernelContext();
  const Adagrad adagrad{ctx};
  const float norm_coefficient = 0.0f;
  const float epsilon = 0.0f;
  const float decay_factor = 0.5f;

  const Tensor R = Tensor::FromFloat("", {}, {1.0f});
  const Tensor T = Tensor::FromInt64("", {}, {3});
  const Tensor X = Tensor::FromFloat("", {1}, {0.0f});
  const Tensor G = Tensor::FromFloat("", {1}, {1.0f});
  const Tensor H = Tensor::FromFloat("", {1}, {0.0f});

  const std::vector<Tensor> outs =
      adagrad(R, T, {X}, {G}, {H}, epsilon, decay_factor, norm_coefficient);
  // H_new = G^2 = 1, R_adjusted = 1 / (1 + 3 * 0.5) = 0.4,
  // X_new = 0 - 0.4 * 1 / sqrt(1) = -0.4.
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], -0.4f);
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[0], 1.0f);
}

TEST(KernelClass, AdagradRejectsInvalidInputs) {
  const KernelContext ctx = TrainingKernelContext();
  const Adagrad adagrad{ctx};

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor G = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor H = Tensor::FromFloat("", {2}, {0.0f, 0.0f});

  // Empty optimized-tensor list is rejected.
  EXPECT_THROW(adagrad(R, T, {}, {}, {}), std::invalid_argument);

  // Mismatched list lengths are rejected.
  EXPECT_THROW(adagrad(R, T, {X}, {G, G}, {H}), std::invalid_argument);

  // Non-FLOAT 'R' is rejected.
  Tensor bad_R("", onnx_kernels::DataType::INT64, {}, std::vector<uint8_t>(sizeof(int64_t)));
  EXPECT_THROW(adagrad(bad_R, T, {X}, {G}, {H}), std::invalid_argument);

  // Non-INT64 'T' is rejected.
  const Tensor bad_T = Tensor::FromFloat("", {}, {0.0f});
  EXPECT_THROW(adagrad(R, bad_T, {X}, {G}, {H}), std::invalid_argument);

  // Mismatched shapes within a single optimized tensor are rejected.
  const Tensor G_bad = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  EXPECT_THROW(adagrad(R, T, {X}, {G_bad}, {H}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Momentum kernel tests.
// ---------------------------------------------------------------------------

namespace {

// Reference re-implementation of the Momentum pseudo-code from the ONNX
// schema, kept intentionally separate from the kernel under test.
void MomentumReference(float R, int64_t T_val, const std::vector<float> &X,
                       const std::vector<float> &G, const std::vector<float> &V, float alpha,
                       float beta, float norm_coefficient, Momentum::Mode mode,
                       std::vector<float> &X_out, std::vector<float> &V_out) {
  X_out.resize(X.size());
  V_out.resize(X.size());
  const double beta_adjusted = (T_val > 0) ? static_cast<double>(beta) : 1.0;
  for (size_t k = 0; k < X.size(); ++k) {
    const double x = X[k];
    const double g = G[k];
    const double v = V[k];
    const double g_reg = static_cast<double>(norm_coefficient) * x + g;
    const double v_new = static_cast<double>(alpha) * v + beta_adjusted * g_reg;
    const double x_new =
        (mode == Momentum::Mode::kNesterov)
            ? x - static_cast<double>(R) * (g_reg + static_cast<double>(alpha) * v_new)
            : x - static_cast<double>(R) * v_new;
    X_out[k] = static_cast<float>(x_new);
    V_out[k] = static_cast<float>(v_new);
  }
}

} // namespace

TEST(KernelClass, MomentumStandardMatchesReference) {
  const KernelContext ctx = TrainingKernelContext();
  const Momentum momentum{ctx};
  const float norm_coefficient = 0.001f;
  const float alpha = 0.95f;
  const float beta = 0.1f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
  const Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
  const Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});

  const std::vector<Tensor> outs =
      momentum(R, T, {X}, {G}, {V}, alpha, beta, norm_coefficient, Momentum::Mode::kStandard);
  ASSERT_EQ(outs.size(), 2u);
  ASSERT_EQ(outs[0].shape, X.shape);
  ASSERT_EQ(outs[1].shape, V.shape);

  std::vector<float> X_ref, V_ref;
  MomentumReference(0.1f, 0, {1.2f, 2.8f}, {-0.94f, -2.5f}, {1.7f, 3.6f}, alpha, beta,
                    norm_coefficient, Momentum::Mode::kStandard, X_ref, V_ref);
  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[1].AsFloat()[k], V_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(KernelClass, MomentumNesterovMatchesReference) {
  const KernelContext ctx = TrainingKernelContext();
  const Momentum momentum{ctx};
  const float norm_coefficient = 0.01f;
  const float alpha = 0.95f;
  const float beta = 1.0f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
  const Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
  const Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});

  const std::vector<Tensor> outs =
      momentum(R, T, {X}, {G}, {V}, alpha, beta, norm_coefficient, Momentum::Mode::kNesterov);
  ASSERT_EQ(outs.size(), 2u);

  std::vector<float> X_ref, V_ref;
  MomentumReference(0.1f, 0, {1.2f, 2.8f}, {-0.94f, -2.5f}, {1.7f, 3.6f}, alpha, beta,
                    norm_coefficient, Momentum::Mode::kNesterov, X_ref, V_ref);
  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[1].AsFloat()[k], V_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(KernelClass, MomentumMultiVariableMatchesReference) {
  const KernelContext ctx = TrainingKernelContext();
  const Momentum momentum{ctx};
  const float norm_coefficient = 0.001f;
  const float alpha = 0.95f;
  const float beta = 0.85f;

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
  const Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
  const Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
  const Tensor V1 = Tensor::FromFloat("", {1}, {2.0f});
  const Tensor V2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});

  const std::vector<Tensor> outs = momentum(R, T, {X1, X2}, {G1, G2}, {V1, V2}, alpha, beta,
                                            norm_coefficient, Momentum::Mode::kStandard);
  ASSERT_EQ(outs.size(), 4u);

  std::vector<float> X1_ref, V1_ref;
  MomentumReference(0.1f, 0, {1.0f}, {-1.0f}, {2.0f}, alpha, beta, norm_coefficient,
                    Momentum::Mode::kStandard, X1_ref, V1_ref);
  std::vector<float> X2_ref, V2_ref;
  MomentumReference(0.1f, 0, {1.0f, 2.0f}, {-1.0f, -3.0f}, {4.0f, 1.0f}, alpha, beta,
                    norm_coefficient, Momentum::Mode::kStandard, X2_ref, V2_ref);
  for (int64_t k = 0; k < outs[0].element_count(); ++k) {
    EXPECT_NEAR(outs[0].AsFloat()[k], X1_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[2].AsFloat()[k], V1_ref[static_cast<size_t>(k)], 1e-6);
  }
  for (int64_t k = 0; k < outs[1].element_count(); ++k) {
    EXPECT_NEAR(outs[1].AsFloat()[k], X2_ref[static_cast<size_t>(k)], 1e-6);
    EXPECT_NEAR(outs[3].AsFloat()[k], V2_ref[static_cast<size_t>(k)], 1e-6);
  }
}

TEST(KernelClass, MomentumUsesBetaAfterFirstIteration) {
  // With T > 0 the kernel must use ``beta`` as the regularized-gradient
  // coefficient instead of the first-iteration default of 1.
  const KernelContext ctx = TrainingKernelContext();
  const Momentum momentum{ctx};
  const float norm_coefficient = 0.0f;
  const float alpha = 0.0f; // disable momentum contribution.
  const float beta = 0.25f;

  const Tensor R = Tensor::FromFloat("", {}, {1.0f});
  const Tensor T_first = Tensor::FromInt64("", {}, {0});
  const Tensor T_later = Tensor::FromInt64("", {}, {1});
  const Tensor X = Tensor::FromFloat("", {1}, {0.0f});
  const Tensor G = Tensor::FromFloat("", {1}, {1.0f});
  const Tensor V = Tensor::FromFloat("", {1}, {0.0f});

  const std::vector<Tensor> first =
      momentum(R, T_first, {X}, {G}, {V}, alpha, beta, norm_coefficient, Momentum::Mode::kStandard);
  const std::vector<Tensor> later =
      momentum(R, T_later, {X}, {G}, {V}, alpha, beta, norm_coefficient, Momentum::Mode::kStandard);
  // T == 0 ⇒ beta_adjusted == 1, V_new == G == 1, X_new == 0 - 1 * 1 == -1.
  EXPECT_FLOAT_EQ(first[0].AsFloat()[0], -1.0f);
  EXPECT_FLOAT_EQ(first[1].AsFloat()[0], 1.0f);
  // T  > 0 ⇒ beta_adjusted == beta == 0.25, V_new == 0.25,
  // X_new == 0 - 1 * 0.25 == -0.25.
  EXPECT_FLOAT_EQ(later[0].AsFloat()[0], -0.25f);
  EXPECT_FLOAT_EQ(later[1].AsFloat()[0], 0.25f);
}

TEST(KernelClass, MomentumRejectsInvalidInputs) {
  const KernelContext ctx = TrainingKernelContext();
  const Momentum momentum{ctx};

  const Tensor R = Tensor::FromFloat("", {}, {0.1f});
  const Tensor T = Tensor::FromInt64("", {}, {0});
  const Tensor X = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const Tensor G = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  const Tensor V = Tensor::FromFloat("", {2}, {0.0f, 0.0f});

  EXPECT_THROW(momentum(R, T, {}, {}, {}, 0.9f, 0.1f, 0.0f), std::invalid_argument);
  EXPECT_THROW(momentum(R, T, {X}, {G, G}, {V}, 0.9f, 0.1f, 0.0f), std::invalid_argument);

  Tensor bad_R("", onnx_kernels::DataType::INT64, {}, std::vector<uint8_t>(sizeof(int64_t)));
  EXPECT_THROW(momentum(bad_R, T, {X}, {G}, {V}, 0.9f, 0.1f, 0.0f), std::invalid_argument);

  const Tensor bad_T = Tensor::FromFloat("", {}, {0.0f});
  EXPECT_THROW(momentum(R, bad_T, {X}, {G}, {V}, 0.9f, 0.1f, 0.0f), std::invalid_argument);

  const Tensor G_bad = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
  EXPECT_THROW(momentum(R, T, {X}, {G_bad}, {V}, 0.9f, 0.1f, 0.0f), std::invalid_argument);
}

} // namespace Test
