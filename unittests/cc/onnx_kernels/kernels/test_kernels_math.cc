// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Abs;
using onnx_kernels::kernel::Acos;
using onnx_kernels::kernel::Acosh;
using onnx_kernels::kernel::Add;
using onnx_kernels::kernel::Asin;
using onnx_kernels::kernel::Asinh;
using onnx_kernels::kernel::Atan;
using onnx_kernels::kernel::Atanh;
using onnx_kernels::kernel::BlackmanWindow;
using onnx_kernels::kernel::Ceil;
using onnx_kernels::kernel::Clip;
using onnx_kernels::kernel::Cos;
using onnx_kernels::kernel::Cosh;
using onnx_kernels::kernel::Det;
using onnx_kernels::kernel::Div;
using onnx_kernels::kernel::Einsum;
using onnx_kernels::kernel::Erf;
using onnx_kernels::kernel::Exp;
using onnx_kernels::kernel::Floor;
using onnx_kernels::kernel::HammingWindow;
using onnx_kernels::kernel::HannWindow;
using onnx_kernels::kernel::Hardmax;
using onnx_kernels::kernel::HardSigmoid;
using onnx_kernels::kernel::HardSwish;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Log;
using onnx_kernels::kernel::LogSoftmax;
using onnx_kernels::kernel::MatMul;
using onnx_kernels::kernel::MatMulInteger;
using onnx_kernels::kernel::Mish;
using onnx_kernels::kernel::Mod;
using onnx_kernels::kernel::Mul;
using onnx_kernels::kernel::Neg;
using onnx_kernels::kernel::Pow;
using onnx_kernels::kernel::PRelu;
using onnx_kernels::kernel::Reciprocal;
using onnx_kernels::kernel::Round;
using onnx_kernels::kernel::Shrink;
using onnx_kernels::kernel::Sigmoid;
using onnx_kernels::kernel::Sin;
using onnx_kernels::kernel::Sinh;
using onnx_kernels::kernel::Softmax;
using onnx_kernels::kernel::Softplus;
using onnx_kernels::kernel::Softsign;
using onnx_kernels::kernel::Sqrt;
using onnx_kernels::kernel::Sub;
using onnx_kernels::kernel::Tan;
using onnx_kernels::kernel::Tanh;
using onnx_kernels::kernel::TopK;

namespace Test {

TEST(KernelClass, AbsClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y = abs_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(KernelClass, NegClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Neg neg_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y = neg_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -0.0f);
  EXPECT_FLOAT_EQ(py[2], -2.5f);
}

TEST(KernelClass, NegInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Neg neg_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2}, {-4.0f, 2.0f});
  Tensor y("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  neg_kernel(x, y);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 4.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
}

TEST(KernelClass, NegRejectsNonFloatTensors) {
  const KernelContext ctx{DefaultOpset(13)};
  Neg neg_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2}, {-1, 2});
  EXPECT_THROW((void)neg_kernel(x), std::invalid_argument);
}

TEST(KernelClass, AcosClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Acos acos_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = acos_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 3.14159265f, 1e-5f);
  EXPECT_NEAR(py[1], 1.57079633f, 1e-5f);
  EXPECT_NEAR(py[2], 0.0f, 1e-6f);
}

TEST(KernelClass, AcoshClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Acosh acosh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 10.0f});
  Tensor y = acosh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.0f, 1e-6f);
  EXPECT_NEAR(py[1], 1.31695790f, 1e-5f);
  EXPECT_NEAR(py[2], 2.99322285f, 1e-5f);
}

TEST(KernelClass, AsinClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Asin asin_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = asin_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -1.57079633f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 1.57079633f, 1e-5f);
}

TEST(KernelClass, AsinhClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Asinh asinh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = asinh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.88137358f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.88137358f, 1e-5f);
}

TEST(KernelClass, AtanClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Atan atan_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = atan_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.78539816f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.78539816f, 1e-5f);
}

TEST(KernelClass, AtanhClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Atanh atanh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.0f, 0.5f});
  Tensor y = atanh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.54930614f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.54930614f, 1e-5f);
}

TEST(KernelClass, CosClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Cos cos_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = cos_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.54030231f, 1e-5f);
  EXPECT_NEAR(py[1], 1.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.54030231f, 1e-5f);
}

TEST(KernelClass, CoshClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Cosh cosh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = cosh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 1.54308063f, 1e-5f);
  EXPECT_NEAR(py[1], 1.0f, 1e-6f);
  EXPECT_NEAR(py[2], 1.54308063f, 1e-5f);
}

TEST(KernelClass, ExpClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Exp exp_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = exp_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.36787944f, 1e-6f);
  EXPECT_NEAR(py[1], 1.0f, 1e-6f);
  EXPECT_NEAR(py[2], 2.71828183f, 1e-6f);
}

TEST(KernelClass, ErfClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Erf erf_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = erf_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.84270079f, 1e-6f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.84270079f, 1e-6f);
}

TEST(KernelClass, LogClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Log log_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {0.5f, 1.0f, 2.0f});
  Tensor y = log_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.69314718f, 1e-6f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.69314718f, 1e-6f);
}

TEST(KernelClass, SqrtClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Sqrt sqrt_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {0.0f, 1.0f, 4.0f, 9.0f});
  Tensor y = sqrt_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.0f, 1e-6f);
  EXPECT_NEAR(py[1], 1.0f, 1e-6f);
  EXPECT_NEAR(py[2], 2.0f, 1e-6f);
  EXPECT_NEAR(py[3], 3.0f, 1e-6f);
}

TEST(KernelClass, ReciprocalClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Reciprocal reciprocal_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {-4.0f, -0.5f, 1.0f, 2.0f});
  Tensor y = reciprocal_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.25f, 1e-6f);
  EXPECT_NEAR(py[1], -2.0f, 1e-6f);
  EXPECT_NEAR(py[2], 1.0f, 1e-6f);
  EXPECT_NEAR(py[3], 0.5f, 1e-6f);
}

TEST(KernelClass, SigmoidClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Sigmoid sigmoid_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-2.0f, 0.0f, 2.0f});
  Tensor y = sigmoid_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.11920292f, 1e-6f);
  EXPECT_NEAR(py[1], 0.5f, 1e-6f);
  EXPECT_NEAR(py[2], 0.88079708f, 1e-6f);
}

TEST(KernelClass, SoftplusClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Softplus softplus_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {-20.0f, -1.0f, 0.0f, 2.0f});
  Tensor y = softplus_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  // Reference values from y = ln(1 + exp(x)); numerically stable around large magnitudes.
  EXPECT_NEAR(py[0], 2.0611537e-9f, 1e-6f);
  EXPECT_NEAR(py[1], 0.31326169f, 1e-6f);
  EXPECT_NEAR(py[2], 0.69314718f, 1e-6f);
  EXPECT_NEAR(py[3], 2.12692809f, 1e-6f);
}

TEST(KernelClass, MishClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Mish mish_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {-4.0f, -1.0f, 0.0f, 2.0f});
  Tensor y = mish_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  // Reference values from y = x * tanh(softplus(x)) computed in double precision.
  EXPECT_NEAR(py[0], -0.07259174f, 1e-6f);
  EXPECT_NEAR(py[1], -0.30340147f, 1e-6f);
  EXPECT_NEAR(py[2], 0.0f, 1e-6f);
  EXPECT_NEAR(py[3], 1.94395934f, 1e-6f);
}

TEST(KernelClass, SoftsignClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Softsign softsign_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {-3.0f, -1.0f, 0.0f, 4.0f});
  Tensor y = softsign_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.75f, 1e-6f);
  EXPECT_NEAR(py[1], -0.5f, 1e-6f);
  EXPECT_NEAR(py[2], 0.0f, 1e-6f);
  EXPECT_NEAR(py[3], 0.8f, 1e-6f);
}

TEST(KernelClass, HardSigmoidClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  HardSigmoid hard_sigmoid_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {5}, {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
  // alpha = 0.5, beta = 0.6 -> y = max(0, min(1, 0.5*x + 0.6))
  Tensor y = hard_sigmoid_kernel(x, 0.5f, 0.6f);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.0f, 1e-6f);
  EXPECT_NEAR(py[1], 0.1f, 1e-6f);
  EXPECT_NEAR(py[2], 0.6f, 1e-6f);
  EXPECT_NEAR(py[3], 1.0f, 1e-6f);
  EXPECT_NEAR(py[4], 1.0f, 1e-6f);
}

TEST(KernelClass, HardSwishClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  HardSwish hard_swish_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {-4.0f, 0.0f, 1.0f, 4.0f});
  Tensor y = hard_swish_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  // y = x * max(0, min(1, x/6 + 0.5))
  EXPECT_NEAR(py[0], 0.0f, 1e-6f);                        // x=-4 -> hs=max(0,-1/6) = 0
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);                        // x=0 -> hs=0.5, x*hs=0
  EXPECT_NEAR(py[2], 1.0f * (1.0f / 6.0f + 0.5f), 1e-6f); // x=1 -> hs = 2/3
  EXPECT_NEAR(py[3], 4.0f, 1e-6f);                        // x=4 -> hs = min(1, 7/6) = 1
}

TEST(KernelClass, HardmaxClassWritesOneHotAlongAxis) {
  const KernelContext ctx{DefaultOpset(13)};
  Hardmax hardmax_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
  // axis = 1 -> max along columns of each row.
  Tensor y = hardmax_kernel(x, 1);
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 1.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 0.0f);
  EXPECT_FLOAT_EQ(py[5], 0.0f);
}

TEST(KernelClass, HardmaxClassPicksFirstMaxOnTies) {
  const KernelContext ctx{DefaultOpset(13)};
  Hardmax hardmax_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {1, 4}, {1.0f, 3.0f, 3.0f, 2.0f});
  Tensor y = hardmax_kernel(x, -1);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 1.0f); // first occurrence of max wins
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(KernelClass, DetClassComputesScalarFor2DInput) {
  const KernelContext ctx{DefaultOpset(11)};
  Det det_kernel{ctx};

  // [[0, 1], [2, 3]] -> 0 * 3 - 1 * 2 = -2.
  Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  Tensor y = det_kernel(x);
  ASSERT_TRUE(y.shape.empty());
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_NEAR(y.AsFloat()[0], -2.0f, 1e-6f);
}

TEST(KernelClass, DetClassComputesBatchOf2x2Determinants) {
  const KernelContext ctx{DefaultOpset(11)};
  Det det_kernel{ctx};

  // Matches the ONNX ``test_det_nd`` reference: dets = [-2, -3, -8].
  Tensor x = Tensor::FromFloat(
      "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f});
  Tensor y = det_kernel(x);
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -2.0f, 1e-6f);
  EXPECT_NEAR(py[1], -3.0f, 1e-6f);
  EXPECT_NEAR(py[2], -8.0f, 1e-6f);
}

TEST(KernelClass, DetClassRejectsNonSquareInput) {
  const KernelContext ctx{DefaultOpset(11)};
  Det det_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_THROW(det_kernel(x), std::exception);
}

TEST(KernelClass, SoftmaxClassMatchesReferenceAxis1) {
  const KernelContext ctx{DefaultOpset(13)};
  Softmax softmax_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  Tensor y = softmax_kernel(x, 1);
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.09003057f, 1e-6f);
  EXPECT_NEAR(py[1], 0.24472848f, 1e-6f);
  EXPECT_NEAR(py[2], 0.66524094f, 1e-6f);
  EXPECT_NEAR(py[3], 0.09003057f, 1e-6f);
  EXPECT_NEAR(py[4], 0.24472848f, 1e-6f);
  EXPECT_NEAR(py[5], 0.66524094f, 1e-6f);
}

TEST(KernelClass, LogSoftmaxClassMatchesReferenceAxis1) {
  const KernelContext ctx{DefaultOpset(13)};
  LogSoftmax logsoftmax_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  Tensor y = logsoftmax_kernel(x, 1);
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  // log of the Softmax reference values above.
  EXPECT_NEAR(py[0], std::log(0.09003057f), 1e-5f);
  EXPECT_NEAR(py[1], std::log(0.24472848f), 1e-5f);
  EXPECT_NEAR(py[2], std::log(0.66524094f), 1e-5f);
  EXPECT_NEAR(py[3], std::log(0.09003057f), 1e-5f);
  EXPECT_NEAR(py[4], std::log(0.24472848f), 1e-5f);
  EXPECT_NEAR(py[5], std::log(0.66524094f), 1e-5f);
}

TEST(KernelClass, LogSoftmaxClassIsNumericallyStableForLargeInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  LogSoftmax logsoftmax_kernel{ctx};

  // Without the max-subtraction trick, exp(1002) would overflow to +inf.
  Tensor x = Tensor::FromFloat("", {1, 3}, {1000.0f, 1001.0f, 1002.0f});
  Tensor y = logsoftmax_kernel(x, -1);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_TRUE(std::isfinite(py[0]));
  EXPECT_TRUE(std::isfinite(py[1]));
  EXPECT_TRUE(std::isfinite(py[2]));
  // log-sum-exp([1000,1001,1002]) - max == log(1+e+e^2) and y == x - max - log-sum.
  const float lse = std::log(std::exp(-2.0f) + std::exp(-1.0f) + 1.0f);
  EXPECT_NEAR(py[0], -2.0f - lse, 1e-4f);
  EXPECT_NEAR(py[1], -1.0f - lse, 1e-4f);
  EXPECT_NEAR(py[2], 0.0f - lse, 1e-4f);
}

TEST(KernelClass, SinClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Sin sin_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = sin_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.84147098f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.84147098f, 1e-5f);
}

TEST(KernelClass, SinhClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Sinh sinh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = sinh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -1.17520119f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 1.17520119f, 1e-5f);
}

TEST(KernelClass, TanClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(22)};
  Tan tan_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = tan_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -1.55740772f, 1e-5f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 1.55740772f, 1e-5f);
}

TEST(KernelClass, TanhClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Tanh tanh_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = tanh_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -0.76159416f, 1e-6f);
  EXPECT_NEAR(py[1], 0.0f, 1e-6f);
  EXPECT_NEAR(py[2], 0.76159416f, 1e-6f);
}

TEST(KernelClass, AddClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z = add_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.5f);
  EXPECT_FLOAT_EQ(pz[1], 2.5f);
  EXPECT_FLOAT_EQ(pz[2], 3.5f);
  EXPECT_FLOAT_EQ(pz[3], 4.5f);
}

TEST(KernelClass, BlackmanWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  BlackmanWindow blackman_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = blackman_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Blackman window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(KernelClass, AbsInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y("out", onnx_kernels::DataType::FLOAT, {3},
           std::vector<uint8_t>(3 * sizeof(float), 0xFF));
  abs_kernel(x, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(KernelClass, AddInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  add_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.5f);
  EXPECT_FLOAT_EQ(pz[1], 2.5f);
  EXPECT_FLOAT_EQ(pz[2], 3.5f);
  EXPECT_FLOAT_EQ(pz[3], 4.5f);
}

TEST(KernelClass, BlackmanWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  BlackmanWindow blackman_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_kernels::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  blackman_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(KernelClass, HannWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  HannWindow hann_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = hann_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Hann window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(KernelClass, HannWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  HannWindow hann_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_kernels::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  hann_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(KernelClass, HammingWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  HammingWindow hamming_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = hamming_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Hamming window is a0 + a1 = (25 - 21) / 46 = 4/46.
  EXPECT_NEAR(y.AsFloat()[0], static_cast<float>(4.0 / 46.0), 1e-6f);
}

TEST(KernelClass, HammingWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  HammingWindow hamming_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_kernels::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  hamming_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], static_cast<float>(4.0 / 46.0), 1e-6f);
}

TEST(KernelClass, InPlaceRejectsMismatchedShapeOrType) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});

  // Wrong dtype.
  Tensor bad_dtype("", onnx_kernels::DataType::INT32, {3},
                   std::vector<uint8_t>(3 * sizeof(int32_t)));
  EXPECT_THROW(abs_kernel(x, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", onnx_kernels::DataType::FLOAT, {3}, std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_bytes), std::invalid_argument);
}

TEST(KernelClass, AbsInPlaceAliasingInputAndOutput) {
  // Demonstrates that Abs::CanRunInPlace() is honored by the implementation:
  // pass the same Tensor object as both input and output and verify the
  // result is written correctly in-place.
  ASSERT_TRUE(Abs::CanRunInPlace());
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  abs_kernel(x, x);
  const float *px = x.AsFloat();
  EXPECT_FLOAT_EQ(px[0], 1.0f);
  EXPECT_FLOAT_EQ(px[1], 0.0f);
  EXPECT_FLOAT_EQ(px[2], 2.5f);
}

TEST(KernelClass, SubClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], -2.0f);
  EXPECT_FLOAT_EQ(pz[1], 0.0f);
  EXPECT_FLOAT_EQ(pz[2], 2.0f);
}

TEST(KernelClass, SubClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 0.5f);
  EXPECT_FLOAT_EQ(pz[1], 1.5f);
  EXPECT_FLOAT_EQ(pz[2], 2.5f);
  EXPECT_FLOAT_EQ(pz[3], 3.5f);
}

TEST(KernelClass, SubInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  sub_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 0.5f);
  EXPECT_FLOAT_EQ(pz[1], 1.5f);
  EXPECT_FLOAT_EQ(pz[2], 2.5f);
  EXPECT_FLOAT_EQ(pz[3], 3.5f);
}

TEST(KernelClass, SubClassMatchesReferenceInt8) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {4}, {10, 0, -3, 7});
  Tensor y = Tensor::FromInt8("", {4}, {3, 0, 2, -1});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT8));
  const int8_t *pz = z.AsInt8();
  EXPECT_EQ(pz[0], 7);
  EXPECT_EQ(pz[1], 0);
  EXPECT_EQ(pz[2], -5);
  EXPECT_EQ(pz[3], 8);
}

TEST(KernelClass, SubClassMatchesReferenceUint32) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromUint32("", {4}, {10u, 5u, 3u, 100u});
  Tensor y = Tensor::FromUint32("", {4}, {3u, 5u, 1u, 50u});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT32));
  const uint32_t *pz = reinterpret_cast<const uint32_t *>(z.data.data());
  EXPECT_EQ(pz[0], 7u);
  EXPECT_EQ(pz[1], 0u);
  EXPECT_EQ(pz[2], 2u);
  EXPECT_EQ(pz[3], 50u);
}

TEST(KernelClass, SubRejectsUnsupportedDtype) {
  // BOOL inputs are not in the supported dtype set (FLOAT/INT8/INT16/UINT8/
  // UINT16/UINT32/UINT64) so the kernel must reject them.
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x("", onnx_kernels::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_kernels::DataType::BOOL, {2}, {0, 1});
  EXPECT_THROW((void)sub_kernel(x, y), std::invalid_argument);
}

TEST(KernelClass, MulClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  Tensor z = mul_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 4.0f);
  EXPECT_FLOAT_EQ(pz[1], 10.0f);
  EXPECT_FLOAT_EQ(pz[2], 18.0f);
}

TEST(KernelClass, MulClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z = mul_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 2.0f);
  EXPECT_FLOAT_EQ(pz[1], 4.0f);
  EXPECT_FLOAT_EQ(pz[2], 6.0f);
  EXPECT_FLOAT_EQ(pz[3], 8.0f);
}

TEST(KernelClass, MulInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {3.0f});
  Tensor z("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  mul_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 3.0f);
  EXPECT_FLOAT_EQ(pz[1], 6.0f);
  EXPECT_FLOAT_EQ(pz[2], 9.0f);
  EXPECT_FLOAT_EQ(pz[3], 12.0f);
}

TEST(KernelClass, DivClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor z = div_kernel(x, y);
  ASSERT_EQ(z.element_count(), 2);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 3.0f);
  EXPECT_FLOAT_EQ(pz[1], 2.0f);
}

TEST(KernelClass, DivClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z = div_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 2.0f);
  EXPECT_FLOAT_EQ(pz[2], 3.0f);
  EXPECT_FLOAT_EQ(pz[3], 4.0f);
}

TEST(KernelClass, DivInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  div_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 2.0f);
  EXPECT_FLOAT_EQ(pz[2], 3.0f);
  EXPECT_FLOAT_EQ(pz[3], 4.0f);
}

TEST(KernelClass, MulClassSupportsIntegerTypes) {
  // ``kernel::Mul`` must handle every integer dtype exercised by the
  // upstream ``onnx.backend.test.case.node.mul.Mul`` cases.
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  {
    Tensor x = Tensor::FromInt8("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromInt8("", {3}, {4, 5, 6});
    Tensor z = mul_kernel(x, y);
    const int8_t *pz = z.AsInt8();
    EXPECT_EQ(pz[0], 4);
    EXPECT_EQ(pz[1], 10);
    EXPECT_EQ(pz[2], 18);
  }
  {
    Tensor x = Tensor::FromUint32("", {2}, {7u, 11u});
    Tensor y = Tensor::FromUint32("", {}, {3u});
    Tensor z = mul_kernel(x, y);
    const uint32_t *pz = z.AsUint32();
    EXPECT_EQ(pz[0], 21u);
    EXPECT_EQ(pz[1], 33u);
  }
}

TEST(KernelClass, AddClassSupportsIntegerTypes) {
  // ``kernel::Add`` must handle every integer dtype exercised by the
  // upstream ``onnx.backend.test.case.node.add.Add`` cases.
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  {
    Tensor x = Tensor::FromInt8("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromInt8("", {3}, {4, 5, 6});
    Tensor z = add_kernel(x, y);
    const int8_t *pz = z.AsInt8();
    EXPECT_EQ(pz[0], 5);
    EXPECT_EQ(pz[1], 7);
    EXPECT_EQ(pz[2], 9);
  }
  {
    Tensor x = Tensor::FromUint32("", {2}, {7u, 11u});
    Tensor y = Tensor::FromUint32("", {}, {3u});
    Tensor z = add_kernel(x, y);
    const uint32_t *pz = z.AsUint32();
    EXPECT_EQ(pz[0], 10u);
    EXPECT_EQ(pz[1], 14u);
  }
}

TEST(KernelClass, DivClassSupportsIntegerTypesWithTruncation) {
  // ``kernel::Div`` must implement truncating integer division for all
  // signed/unsigned integer dtypes registered by the upstream cases.
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  {
    Tensor x = Tensor::FromInt32("", {4}, {-3, 3, -3, 3});
    Tensor y = Tensor::FromInt32("", {4}, {2, 2, -2, -2});
    Tensor z = div_kernel(x, y);
    const int32_t *pz = z.AsInt32();
    EXPECT_EQ(pz[0], -1);
    EXPECT_EQ(pz[1], 1);
    EXPECT_EQ(pz[2], 1);
    EXPECT_EQ(pz[3], -1);
  }
  {
    Tensor x = Tensor::FromUint16("", {3}, {10, 9, 7});
    Tensor y = Tensor::FromUint16("", {3}, {3, 2, 4});
    Tensor z = div_kernel(x, y);
    const uint16_t *pz = z.AsUint16();
    EXPECT_EQ(pz[0], 3);
    EXPECT_EQ(pz[1], 4);
    EXPECT_EQ(pz[2], 1);
  }
}

TEST(KernelClass, ModClassMatchesPythonAndCSemantics) {
  // ``kernel::Mod`` must match ``numpy.mod`` (sign follows divisor) when
  // ``fmod=0`` and ``numpy.fmod`` / C ``fmod`` (sign follows dividend) when
  // ``fmod=1``. Cross-checked against the upstream
  // ``test_mod_mixed_sign_int*`` / ``test_mod_int64_fmod`` /
  // ``test_mod_mixed_sign_float*`` reference outputs.
  const KernelContext ctx{DefaultOpset(13)};
  Mod mod_kernel{ctx};

  // Default fmod=0 on signed integers (Python-style mod).
  {
    Tensor x = Tensor::FromInt32("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt32("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y);
    const int32_t *pz = z.AsInt32();
    EXPECT_EQ(pz[0], 0);
    EXPECT_EQ(pz[1], -2);
    EXPECT_EQ(pz[2], 5);
    EXPECT_EQ(pz[3], 0);
    EXPECT_EQ(pz[4], 2);
    EXPECT_EQ(pz[5], 3);
  }

  // fmod=1 on signed integers (C-style truncated mod).
  {
    Tensor x = Tensor::FromInt64("", {6}, {-4, 7, 5, 4, -7, 8});
    Tensor y = Tensor::FromInt64("", {6}, {2, -3, 8, -2, 3, 5});
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    const int64_t *pz = z.AsInt64();
    EXPECT_EQ(pz[0], 0);
    EXPECT_EQ(pz[1], 1);
    EXPECT_EQ(pz[2], 5);
    EXPECT_EQ(pz[3], 0);
    EXPECT_EQ(pz[4], -1);
    EXPECT_EQ(pz[5], 3);
  }

  // Unsigned integers (fmod=0).
  {
    Tensor x = Tensor::FromUint16("", {3}, {4, 7, 5});
    Tensor y = Tensor::FromUint16("", {3}, {2, 3, 8});
    Tensor z = mod_kernel(x, y);
    const uint16_t *pz = z.AsUint16();
    EXPECT_EQ(pz[0], 0);
    EXPECT_EQ(pz[1], 1);
    EXPECT_EQ(pz[2], 5);
  }

  // Floating-point inputs require fmod=1.
  {
    Tensor x = Tensor::FromFloat("", {3}, {-4.3f, 7.2f, 5.0f});
    Tensor y = Tensor::FromFloat("", {3}, {2.1f, -3.4f, 8.0f});
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    const float *pz = z.AsFloat();
    EXPECT_NEAR(pz[0], std::fmod(-4.3f, 2.1f), 1e-6f);
    EXPECT_NEAR(pz[1], std::fmod(7.2f, -3.4f), 1e-6f);
    EXPECT_FLOAT_EQ(pz[2], 5.0f);
  }

  // FLOAT16 inputs require fmod=1; the output bit pattern must match
  // ``numpy.fmod`` on the IEEE-754 binary16 inputs (upstream
  // ``test_mod_mixed_sign_float16`` reference).
  {
    Tensor x("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT16), {6}, {});
    Tensor y("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT16), {6}, {});
    // Bit patterns of float16(-4.3, 7.2, 5.0, 4.3, -7.2, 8.0) and
    // float16(2.1, -3.4, 8.0, -2.1, 3.4, 5.0).
    const std::vector<uint16_t> xb = {0xc44dU, 0x4733U, 0x4500U, 0x444dU, 0xc733U, 0x4800U};
    const std::vector<uint16_t> yb = {0x4033U, 0xc2cdU, 0x4800U, 0xc033U, 0x42cdU, 0x4500U};
    x.data.assign(reinterpret_cast<const uint8_t *>(xb.data()),
                  reinterpret_cast<const uint8_t *>(xb.data() + xb.size()));
    y.data.assign(reinterpret_cast<const uint8_t *>(yb.data()),
                  reinterpret_cast<const uint8_t *>(yb.data() + yb.size()));
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    ASSERT_EQ(z.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT16));
    const uint16_t *pz = reinterpret_cast<const uint16_t *>(z.data.data());
    EXPECT_EQ(pz[0], 0xae80U); // -0.1015625
    EXPECT_EQ(pz[1], 0x3660U); //  0.3984375
    EXPECT_EQ(pz[2], 0x4500U); //  5.0
    EXPECT_EQ(pz[3], 0x2e80U); //  0.1015625
    EXPECT_EQ(pz[4], 0xb660U); // -0.3984375
    EXPECT_EQ(pz[5], 0x4200U); //  3.0
  }

  // FLOAT16 with fmod=0 must throw (matches FLOAT/DOUBLE behaviour).
  {
    Tensor x("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT16), {1},
             std::vector<uint8_t>(sizeof(uint16_t), 0));
    Tensor y("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT16), {1},
             std::vector<uint8_t>(sizeof(uint16_t), 0));
    EXPECT_THROW(mod_kernel(x, y), std::invalid_argument);
  }

  // Floating-point with fmod=0 must throw.
  {
    Tensor x = Tensor::FromFloat("", {1}, {1.0f});
    Tensor y = Tensor::FromFloat("", {1}, {2.0f});
    EXPECT_THROW(mod_kernel(x, y), std::invalid_argument);
  }

  // Broadcasting (int32, scalar-ish divisor).
  {
    Tensor x = Tensor::FromInt32("", {2, 3}, {0, 1, 2, 3, 4, 5});
    Tensor y = Tensor::FromInt32("", {1}, {7});
    Tensor z = mod_kernel(x, y);
    ASSERT_EQ(z.shape, (std::vector<int64_t>{2, 3}));
    const int32_t *pz = z.AsInt32();
    EXPECT_EQ(pz[0], 0);
    EXPECT_EQ(pz[5], 5);
  }
}

TEST(KernelClass, MatMulClassMatchesReference2D) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  Tensor a = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::FromFloat("", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor y = matmul_kernel(a, b);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 58.0f);
  EXPECT_FLOAT_EQ(py[1], 64.0f);
  EXPECT_FLOAT_EQ(py[2], 139.0f);
  EXPECT_FLOAT_EQ(py[3], 154.0f);
}

TEST(KernelClass, MatMulClassSupportsVectorMatrix) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  Tensor a = Tensor::FromInt32("", {3}, {2, 3, 4});
  Tensor b = Tensor::FromInt32("", {3, 2}, {1, 5, 2, 6, 3, 7});
  Tensor y = matmul_kernel(a, b);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 20);
  EXPECT_EQ(py[1], 56);
}

TEST(KernelClass, MatMulClassBroadcastsBatchDimensions) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  Tensor a = Tensor::FromFloat("", {2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b = Tensor::FromFloat("", {1, 2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = matmul_kernel(a, b);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 19.0f);
  EXPECT_FLOAT_EQ(py[1], 22.0f);
  EXPECT_FLOAT_EQ(py[2], 43.0f);
  EXPECT_FLOAT_EQ(py[3], 50.0f);
}

TEST(KernelClass, MatMulInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  Tensor a = Tensor::FromUint32("", {2, 3}, {1u, 2u, 3u, 4u, 5u, 6u});
  Tensor b = Tensor::FromUint32("", {3, 2}, {1u, 2u, 3u, 4u, 5u, 6u});
  Tensor y("", onnx_kernels::DataType::UINT32, {2, 2}, std::vector<uint8_t>(4 * sizeof(uint32_t)));
  matmul_kernel(a, b, y);
  const uint32_t *py = y.AsUint32();
  EXPECT_EQ(py[0], 22u);
  EXPECT_EQ(py[1], 28u);
  EXPECT_EQ(py[2], 49u);
  EXPECT_EQ(py[3], 64u);
}

TEST(KernelClass, MatMulIntegerUint8MatchesONNXReference) {
  // Mirrors the ONNX reference ``test_matmulinteger`` example with per-tensor
  // UINT8 zero points: Y = matmul(A - a_zp, B - b_zp).
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  Tensor b = Tensor::FromUint8("", {3, 2}, {1, 4, 2, 5, 3, 6});
  Tensor a_zp("", onnx_kernels::DataType::UINT8, {1}, std::vector<uint8_t>{12});
  Tensor b_zp("", onnx_kernels::DataType::UINT8, {1}, std::vector<uint8_t>{0});
  Tensor y = mmi(a, b, a_zp, b_zp);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 2}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], -38);
  EXPECT_EQ(py[1], -83);
  EXPECT_EQ(py[2], -44);
  EXPECT_EQ(py[3], -98);
  EXPECT_EQ(py[4], -50);
  EXPECT_EQ(py[5], -113);
  EXPECT_EQ(py[6], -56);
  EXPECT_EQ(py[7], -128);
}

TEST(KernelClass, MatMulIntegerWithDefaultZeroPoints) {
  // Default-constructed (empty) zero-point tensors must be treated as 0.
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromUint8("", {3, 2}, {7, 8, 9, 10, 11, 12});
  Tensor a_zp;
  Tensor b_zp;
  Tensor y = mmi(a, b, a_zp, b_zp);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 58);
  EXPECT_EQ(py[1], 64);
  EXPECT_EQ(py[2], 139);
  EXPECT_EQ(py[3], 154);
}

TEST(KernelClass, MatMulIntegerInt8WithScalarZeroPoints) {
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromInt8("", {2, 3}, {1, -2, 3, -4, 5, -6});
  Tensor b = Tensor::FromInt8("", {3, 2}, {1, 2, -3, 4, 5, -6});
  Tensor a_zp("", onnx_kernels::DataType::INT8, {},
              std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(1))});
  Tensor b_zp("", onnx_kernels::DataType::INT8, {},
              std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(-1))});
  Tensor y = mmi(a, b, a_zp, b_zp);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  // (A - 1) * (B + 1): manually computed reference.
  const int32_t *py = y.AsInt32();
  // A' = {{0,-3,2},{-5,4,-7}}, B' = {{2,3},{-2,5},{6,-5}}
  // Row 0: 0*2 + (-3)*(-2) + 2*6 = 18 ; 0*3 + (-3)*5 + 2*(-5) = -25
  // Row 1: -5*2 + 4*(-2) + (-7)*6 = -60 ; -5*3 + 4*5 + (-7)*(-5) = 40
  EXPECT_EQ(py[0], 18);
  EXPECT_EQ(py[1], -25);
  EXPECT_EQ(py[2], -60);
  EXPECT_EQ(py[3], 40);
}

TEST(KernelClass, MatMulIntegerInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromUint8("", {3, 2}, {7, 8, 9, 10, 11, 12});
  Tensor a_zp;
  Tensor b_zp;
  Tensor y("", onnx_kernels::DataType::INT32, {2, 2}, std::vector<uint8_t>(4 * sizeof(int32_t)));
  mmi(a, b, a_zp, b_zp, y);
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 58);
  EXPECT_EQ(py[1], 64);
  EXPECT_EQ(py[2], 139);
  EXPECT_EQ(py[3], 154);
}

TEST(KernelClass, MatMulIntegerRejectsNonByteInput) {
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromInt32("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromUint8("", {3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor a_zp;
  Tensor b_zp;
  EXPECT_THROW(mmi(a, b, a_zp, b_zp), std::invalid_argument);
}

TEST(KernelClass, FloorClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Floor floor_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {5}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f});
  Tensor y = floor_kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -2.0f);
  EXPECT_FLOAT_EQ(py[1], -1.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
  EXPECT_FLOAT_EQ(py[4], 1.0f);
}

TEST(KernelClass, CeilClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Ceil ceil_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {5}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f});
  Tensor y = ceil_kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 2.0f);
}

TEST(KernelClass, RoundClassRoundsHalvesToEven) {
  const KernelContext ctx{DefaultOpset(22)};
  Round round_kernel{ctx};

  // Halves must round to the nearest even integer (banker's rounding).
  Tensor x = Tensor::FromFloat("", {6}, {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f});
  Tensor y = round_kernel(x);
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 2.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
  EXPECT_FLOAT_EQ(py[4], -2.0f);
  EXPECT_FLOAT_EQ(py[5], -2.0f);
}

TEST(KernelClass, RoundClassNonHalvesRoundToNearest) {
  const KernelContext ctx{DefaultOpset(22)};
  Round round_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {4}, {0.4f, 0.6f, -0.4f, -0.6f});
  Tensor y = round_kernel(x);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 0.0f);
  EXPECT_FLOAT_EQ(py[1], 1.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], -1.0f);
}

TEST(KernelClass, EinsumTransposeMatchesNumpy) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = einsum_kernel({x}, "ij->ji");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 4.0f);
  EXPECT_FLOAT_EQ(py[2], 2.0f);
  EXPECT_FLOAT_EQ(py[3], 5.0f);
  EXPECT_FLOAT_EQ(py[4], 3.0f);
  EXPECT_FLOAT_EQ(py[5], 6.0f);
}

TEST(KernelClass, EinsumTraceMatchesSumOfDiagonal) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor y = einsum_kernel({x}, "ii->");
  ASSERT_TRUE(y.shape.empty());
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.0f + 5.0f + 9.0f);
}

TEST(KernelClass, EinsumMatMulMatchesMatrixProduct) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor a = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromFloat("", {3, 2}, {7, 8, 9, 10, 11, 12});
  Tensor y = einsum_kernel({a, b}, "ij,jk->ik");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1 * 7 + 2 * 9 + 3 * 11);
  EXPECT_FLOAT_EQ(py[1], 1 * 8 + 2 * 10 + 3 * 12);
  EXPECT_FLOAT_EQ(py[2], 4 * 7 + 5 * 9 + 6 * 11);
  EXPECT_FLOAT_EQ(py[3], 4 * 8 + 5 * 10 + 6 * 12);
}

TEST(KernelClass, EinsumOuterProductMatchesProduct) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::FromFloat("", {2}, {4.0f, 5.0f});
  Tensor y = einsum_kernel({a, b}, "i,j->ij");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 4.0f);
  EXPECT_FLOAT_EQ(py[1], 5.0f);
  EXPECT_FLOAT_EQ(py[2], 8.0f);
  EXPECT_FLOAT_EQ(py[3], 10.0f);
  EXPECT_FLOAT_EQ(py[4], 12.0f);
  EXPECT_FLOAT_EQ(py[5], 15.0f);
}

TEST(KernelClass, EinsumImplicitOutputIsAlphabetical) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  // Implicit mode: "ji" — output labels are the singletons sorted by ASCII,
  // so the output is the transpose of the input.
  Tensor x = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor y = einsum_kernel({x}, "ji");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 4.0f);
  EXPECT_FLOAT_EQ(py[2], 2.0f);
  EXPECT_FLOAT_EQ(py[3], 5.0f);
}

TEST(KernelClass, EinsumEllipsisBatchMatMul) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor a = Tensor::FromFloat("", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor b = Tensor::FromFloat("", {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1});
  Tensor y = einsum_kernel({a, b}, "...ij,...jk->...ik");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2, 2}));
  const float *py = y.AsFloat();
  // First batch: same as 2D matmul of a[0] and b[0]
  EXPECT_FLOAT_EQ(py[0], 1.0f * 1 + 2 * 0 + 3 * 1);
  EXPECT_FLOAT_EQ(py[1], 1.0f * 0 + 2 * 1 + 3 * 1);
  EXPECT_FLOAT_EQ(py[2], 4.0f * 1 + 5 * 0 + 6 * 1);
  EXPECT_FLOAT_EQ(py[3], 4.0f * 0 + 5 * 1 + 6 * 1);
  // Second batch: matmul of a[1] and b[1]
  EXPECT_FLOAT_EQ(py[4], 7.0f * 2 + 8 * 0 + 9 * 1);
  EXPECT_FLOAT_EQ(py[5], 7.0f * 0 + 8 * 2 + 9 * 1);
  EXPECT_FLOAT_EQ(py[6], 10.0f * 2 + 11 * 0 + 12 * 1);
  EXPECT_FLOAT_EQ(py[7], 10.0f * 0 + 11 * 2 + 12 * 1);
}

TEST(KernelClass, EinsumRejectsEmptyEquation) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(einsum_kernel({x}, ""), std::invalid_argument);
}

TEST(KernelClass, EinsumRejectsRankMismatch) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  // "i" is rank-1 but the input is rank-2.
  EXPECT_THROW(einsum_kernel({x}, "i->i"), std::invalid_argument);
}

TEST(KernelClass, ClipClassClampsToMinAndMax) {
  const KernelContext ctx{DefaultOpset(13)};
  Clip clip_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
  Tensor lo = Tensor::FromFloat("", {}, {-1.0f});
  Tensor hi = Tensor::FromFloat("", {}, {1.0f});
  Tensor y = clip_kernel(x, &lo, &hi);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], -0.5f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.5f);
  EXPECT_FLOAT_EQ(py[4], 1.0f);
}

TEST(KernelClass, ClipClassDefaultsBoundsToDtypeLimits) {
  const KernelContext ctx{DefaultOpset(13)};
  Clip clip_kernel{ctx};

  // Without bounds, ``Clip`` is the identity.
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = clip_kernel(x);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 1.0f);
}

TEST(KernelClass, ClipClassMinGreaterThanMaxCollapsesToMax) {
  const KernelContext ctx{DefaultOpset(13)};
  Clip clip_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {-2.0f, 0.0f, 6.0f});
  Tensor lo = Tensor::FromFloat("", {}, {2.0f});
  Tensor hi = Tensor::FromFloat("", {}, {1.0f});
  Tensor y = clip_kernel(x, &lo, &hi);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 1.0f);
  EXPECT_FLOAT_EQ(py[2], 1.0f);
}

TEST(KernelClass, ClipClassSupportsInt8) {
  const KernelContext ctx{DefaultOpset(12)};
  Clip clip_kernel{ctx};

  Tensor x = Tensor::FromInt8("", {5}, {-50, -1, 0, 1, 50});
  Tensor lo = Tensor::FromInt8("", {}, {-10});
  Tensor hi = Tensor::FromInt8("", {}, {10});
  Tensor y = clip_kernel(x, &lo, &hi);
  const int8_t *py = y.AsInt8();
  EXPECT_EQ(py[0], -10);
  EXPECT_EQ(py[1], -1);
  EXPECT_EQ(py[2], 0);
  EXPECT_EQ(py[3], 1);
  EXPECT_EQ(py[4], 10);
}

TEST(KernelClass, ClipClassRejectsNonScalarBound) {
  const KernelContext ctx{DefaultOpset(13)};
  Clip clip_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3}, {0.0f, 1.0f, 2.0f});
  Tensor bad_lo = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  EXPECT_THROW(clip_kernel(x, &bad_lo, /*max=*/nullptr), std::invalid_argument);
}

TEST(KernelClass, TopKLargestSortedMatchesReference) {
  const KernelContext ctx{DefaultOpset(11)};
  TopK topk_kernel{ctx};

  Tensor x = Tensor::FromFloat(
      "", {3, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 11.0f, 10.0f, 9.0f, 8.0f});
  auto [values, indices] = topk_kernel(x, /*k=*/3, /*axis=*/-1, /*largest=*/true, /*sorted=*/true);
  ASSERT_EQ(values.shape, (std::vector<int64_t>{3, 3}));
  ASSERT_EQ(indices.shape, (std::vector<int64_t>{3, 3}));
  const float *pv = values.AsFloat();
  const int64_t *pi = indices.AsInt64();
  EXPECT_FLOAT_EQ(pv[0], 3.0f);
  EXPECT_FLOAT_EQ(pv[1], 2.0f);
  EXPECT_FLOAT_EQ(pv[2], 1.0f);
  EXPECT_EQ(pi[0], 3);
  EXPECT_EQ(pi[1], 2);
  EXPECT_EQ(pi[2], 1);
  EXPECT_FLOAT_EQ(pv[6], 11.0f);
  EXPECT_FLOAT_EQ(pv[7], 10.0f);
  EXPECT_FLOAT_EQ(pv[8], 9.0f);
  EXPECT_EQ(pi[6], 0);
  EXPECT_EQ(pi[7], 1);
  EXPECT_EQ(pi[8], 2);
}

TEST(KernelClass, TopKSmallestPicksMinima) {
  const KernelContext ctx{DefaultOpset(11)};
  TopK topk_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {1, 5}, {5.0f, 1.0f, 4.0f, 2.0f, 3.0f});
  auto [values, indices] = topk_kernel(x, /*k=*/2, /*axis=*/-1, /*largest=*/false, /*sorted=*/true);
  ASSERT_EQ(values.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_FLOAT_EQ(values.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(values.AsFloat()[1], 2.0f);
  EXPECT_EQ(indices.AsInt64()[0], 1);
  EXPECT_EQ(indices.AsInt64()[1], 3);
}

TEST(KernelClass, TopKTieBreaksOnLowerIndex) {
  const KernelContext ctx{DefaultOpset(11)};
  TopK topk_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {1, 4}, {1.0f, 1.0f, 1.0f, 1.0f});
  auto [values, indices] = topk_kernel(x, /*k=*/2, /*axis=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_EQ(indices.AsInt64()[0], 0);
  EXPECT_EQ(indices.AsInt64()[1], 1);
  EXPECT_FLOAT_EQ(values.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(values.AsFloat()[1], 1.0f);
}

// ---------------------------------------------------------------------------
// PRelu kernel tests
// ---------------------------------------------------------------------------

TEST(KernelClass, PReluClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(16)};
  PRelu prelu_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {-2.0f, -1.0f, 1.0f, 2.0f});
  Tensor slope = Tensor::FromFloat("", {4}, {0.5f, 0.25f, 0.5f, 0.25f});
  Tensor y = prelu_kernel(x, slope);
  ASSERT_EQ(y.element_count(), 4);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], -0.25f);
  EXPECT_FLOAT_EQ(py[2], 1.0f);
  EXPECT_FLOAT_EQ(py[3], 2.0f);
}

TEST(KernelClass, PReluClassBroadcastsSlope) {
  const KernelContext ctx{DefaultOpset(16)};
  PRelu prelu_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f});
  Tensor slope = Tensor::FromFloat("", {3}, {0.1f, 0.2f, 0.3f});
  Tensor y = prelu_kernel(x, slope);
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.1f);
  EXPECT_FLOAT_EQ(py[1], -0.4f);
  EXPECT_FLOAT_EQ(py[2], -0.9f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 2.0f);
  EXPECT_FLOAT_EQ(py[5], 3.0f);
}

// Regression for microsoft/onnxruntime#28732: PRelu must preserve ``+inf``
// and ``-inf`` inputs rather than collapsing them to ``NaN``.
TEST(KernelClass, PReluPreservesInfiniteInputs) {
  const KernelContext ctx{DefaultOpset(16)};
  PRelu prelu_kernel{ctx};
  const float pinf = std::numeric_limits<float>::infinity();
  const float ninf = -std::numeric_limits<float>::infinity();
  Tensor x = Tensor::FromFloat("", {4}, {pinf, ninf, 5e30f, -2.5f});
  Tensor slope = Tensor::FromFloat("", {4}, {0.25f, 0.5f, 0.25f, 0.25f});
  Tensor y = prelu_kernel(x, slope);
  const float *py = y.AsFloat();
  EXPECT_EQ(py[0], pinf);
  EXPECT_EQ(py[1], ninf);
  EXPECT_FLOAT_EQ(py[2], 5e30f);
  EXPECT_FLOAT_EQ(py[3], -0.625f);
  EXPECT_FALSE(std::isnan(py[0]));
  EXPECT_FALSE(std::isnan(py[1]));
}

TEST(KernelClass, PReluInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(16)};
  PRelu prelu_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {-1.0f, -2.0f, 3.0f, -4.0f});
  Tensor slope = Tensor::FromFloat("", {}, {0.5f});
  Tensor y("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  prelu_kernel(x, slope, y);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.5f);
  EXPECT_FLOAT_EQ(py[1], -1.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
  EXPECT_FLOAT_EQ(py[3], -2.0f);
}

TEST(KernelClass, PReluRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(16)};
  PRelu prelu_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {2}, {-1, 2});
  Tensor slope = Tensor::FromInt8("", {2}, {1, 1});
  EXPECT_THROW(prelu_kernel(x, slope), std::invalid_argument);
}

TEST(KernelClass, PowClassMatchesReferenceFloat) {
  // Matches the upstream ``test_pow_example`` reference case.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  Tensor z = pow_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  ASSERT_EQ(z.data_type, onnx_kernels::DataType::FLOAT);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 32.0f);
  EXPECT_FLOAT_EQ(pz[2], 729.0f);
}

TEST(KernelClass, PowClassBroadcastsScalarExponent) {
  // Matches the upstream ``test_pow_bcast_scalar`` reference case.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z = pow_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 4.0f);
  EXPECT_FLOAT_EQ(pz[2], 9.0f);
}

TEST(KernelClass, PowClassBroadcastsArrayExponent) {
  // Matches the upstream ``test_pow_bcast_array`` reference case.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor z = pow_kernel(x, y);
  ASSERT_EQ(z.shape, (std::vector<int64_t>{2, 3}));
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 4.0f);
  EXPECT_FLOAT_EQ(pz[2], 27.0f);
  EXPECT_FLOAT_EQ(pz[3], 4.0f);
  EXPECT_FLOAT_EQ(pz[4], 25.0f);
  EXPECT_FLOAT_EQ(pz[5], 216.0f);
}

TEST(KernelClass, PowClassSupportsMixedBaseExponentDtypes) {
  // ``Pow`` is the only element-wise binary kernel whose ``T`` (base) and
  // ``T1`` (exponent) type constraints differ — exercise the cross-dtype
  // pairs covered by the upstream ``test_pow_types_*`` reference cases.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  {
    // float ^ int64 -> float
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromInt64("", {3}, {4, 5, 6});
    Tensor z = pow_kernel(x, y);
    ASSERT_EQ(z.data_type, onnx_kernels::DataType::FLOAT);
    const float *pz = z.AsFloat();
    EXPECT_FLOAT_EQ(pz[0], 1.0f);
    EXPECT_FLOAT_EQ(pz[1], 32.0f);
    EXPECT_FLOAT_EQ(pz[2], 729.0f);
  }
  {
    // int64 ^ float -> int64 (truncated)
    Tensor x = Tensor::FromInt64("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
    Tensor z = pow_kernel(x, y);
    ASSERT_EQ(z.data_type, onnx_kernels::DataType::INT64);
    const int64_t *pz = z.AsInt64();
    EXPECT_EQ(pz[0], 1);
    EXPECT_EQ(pz[1], 32);
    EXPECT_EQ(pz[2], 729);
  }
  {
    // int32 ^ int32 -> int32
    Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
    Tensor y = Tensor::FromInt32("", {3}, {4, 5, 6});
    Tensor z = pow_kernel(x, y);
    ASSERT_EQ(z.data_type, onnx_kernels::DataType::INT32);
    const int32_t *pz = z.AsInt32();
    EXPECT_EQ(pz[0], 1);
    EXPECT_EQ(pz[1], 32);
    EXPECT_EQ(pz[2], 729);
  }
  {
    // float ^ uint64 -> float
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromUint64("", {3}, {4, 5, 6});
    Tensor z = pow_kernel(x, y);
    ASSERT_EQ(z.data_type, onnx_kernels::DataType::FLOAT);
    const float *pz = z.AsFloat();
    EXPECT_FLOAT_EQ(pz[0], 1.0f);
    EXPECT_FLOAT_EQ(pz[1], 32.0f);
    EXPECT_FLOAT_EQ(pz[2], 729.0f);
  }
}

TEST(KernelClass, PowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 3.0f, 2.0f, 3.0f});
  Tensor z("", onnx_kernels::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  pow_kernel(x, y, z);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 8.0f);
  EXPECT_FLOAT_EQ(pz[2], 9.0f);
  EXPECT_FLOAT_EQ(pz[3], 64.0f);
}

TEST(KernelClass, PowRejectsUnsupportedBaseDtype) {
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  // UINT8 is not in the ``T`` (base) constraint of ONNX Pow.
  Tensor x = Tensor::FromUint8("", {2}, {1u, 2u});
  Tensor y = Tensor::FromFloat("", {2}, {2.0f, 3.0f});
  EXPECT_THROW(pow_kernel(x, y), std::invalid_argument);
}

TEST(KernelClass, PowRejectsIncompatibleShapes) {
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_THROW(pow_kernel(x, y), std::invalid_argument);
}

TEST(KernelClass, ShrinkClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(9)};
  Shrink shrink_kernel{ctx};

  // Soft shrink: bias=1.5, lambd=1.5.
  Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
  Tensor y = shrink_kernel(x, /*bias=*/1.5f, /*lambd=*/1.5f);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.5f); // -2 < -1.5 -> -2 + 1.5
  EXPECT_FLOAT_EQ(py[1], 0.0f);  // -1 in [-1.5, 1.5] -> 0
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f); // 1 in [-1.5, 1.5] -> 0
  EXPECT_FLOAT_EQ(py[4], 0.5f); // 2 > 1.5 -> 2 - 1.5
}

TEST(KernelClass, ShrinkClassHardShrinkDefaultBias) {
  const KernelContext ctx{DefaultOpset(9)};
  Shrink shrink_kernel{ctx};

  // Hard shrink: bias=0.0, lambd=1.5.
  Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
  Tensor y = shrink_kernel(x, /*bias=*/0.0f, /*lambd=*/1.5f);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -2.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
  EXPECT_FLOAT_EQ(py[4], 2.0f);
}

TEST(KernelClass, ShrinkClassDoubleDtype) {
  const KernelContext ctx{DefaultOpset(9)};
  Shrink shrink_kernel{ctx};
  std::vector<double> values{-2.0, -1.0, 0.0, 1.0, 2.0};
  std::vector<uint8_t> bytes(values.size() * sizeof(double));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  Tensor x("", onnx_kernels::DataType::DOUBLE, {static_cast<int64_t>(values.size())}, bytes);
  Tensor y = shrink_kernel(x, /*bias=*/1.5f, /*lambd=*/1.5f);
  const double *py = reinterpret_cast<const double *>(y.data.data());
  EXPECT_DOUBLE_EQ(py[0], -0.5);
  EXPECT_DOUBLE_EQ(py[1], 0.0);
  EXPECT_DOUBLE_EQ(py[4], 0.5);
}

TEST(KernelClass, ShrinkClassRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(9)};
  Shrink shrink_kernel{ctx};
  Tensor x = Tensor::FromUint8("", {2}, {1u, 2u});
  EXPECT_THROW(shrink_kernel(x), std::invalid_argument);
}

} // namespace Test
