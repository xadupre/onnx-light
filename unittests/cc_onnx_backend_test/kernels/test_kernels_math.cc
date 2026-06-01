// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Abs;
using onnx_backend_test::kernel::Acos;
using onnx_backend_test::kernel::Acosh;
using onnx_backend_test::kernel::Add;
using onnx_backend_test::kernel::Asin;
using onnx_backend_test::kernel::Asinh;
using onnx_backend_test::kernel::Atan;
using onnx_backend_test::kernel::Atanh;
using onnx_backend_test::kernel::BlackmanWindow;
using onnx_backend_test::kernel::Ceil;
using onnx_backend_test::kernel::Cos;
using onnx_backend_test::kernel::Cosh;
using onnx_backend_test::kernel::Div;
using onnx_backend_test::kernel::Einsum;
using onnx_backend_test::kernel::Exp;
using onnx_backend_test::kernel::Floor;
using onnx_backend_test::kernel::HammingWindow;
using onnx_backend_test::kernel::HannWindow;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Log;
using onnx_backend_test::kernel::MatMul;
using onnx_backend_test::kernel::Mul;
using onnx_backend_test::kernel::Round;
using onnx_backend_test::kernel::Sigmoid;
using onnx_backend_test::kernel::Sin;
using onnx_backend_test::kernel::Sinh;
using onnx_backend_test::kernel::Softmax;
using onnx_backend_test::kernel::Sub;
using onnx_backend_test::kernel::Tan;
using onnx_backend_test::kernel::Tanh;

namespace Test {

TEST(BackendKernelClass, AbsClassMatchesReference) {
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

TEST(BackendKernelClass, AcosClassMatchesReference) {
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

TEST(BackendKernelClass, AcoshClassMatchesReference) {
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

TEST(BackendKernelClass, AsinClassMatchesReference) {
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

TEST(BackendKernelClass, AsinhClassMatchesReference) {
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

TEST(BackendKernelClass, AtanClassMatchesReference) {
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

TEST(BackendKernelClass, AtanhClassMatchesReference) {
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

TEST(BackendKernelClass, CosClassMatchesReference) {
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

TEST(BackendKernelClass, CoshClassMatchesReference) {
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

TEST(BackendKernelClass, ExpClassMatchesReference) {
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

TEST(BackendKernelClass, LogClassMatchesReference) {
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

TEST(BackendKernelClass, SigmoidClassMatchesReference) {
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

TEST(BackendKernelClass, SoftmaxClassMatchesReferenceAxis1) {
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

TEST(BackendKernelClass, SinClassMatchesReference) {
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

TEST(BackendKernelClass, SinhClassMatchesReference) {
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

TEST(BackendKernelClass, TanClassMatchesReference) {
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

TEST(BackendKernelClass, TanhClassMatchesReference) {
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

TEST(BackendKernelClass, AddClassBroadcastsScalar) {
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

TEST(BackendKernelClass, BlackmanWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  BlackmanWindow blackman_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = blackman_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Blackman window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, AbsInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y("out", onnx_backend_test::DataType::FLOAT, {3},
           std::vector<uint8_t>(3 * sizeof(float), 0xFF));
  abs_kernel(x, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(BackendKernelClass, AddInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z("", onnx_backend_test::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  add_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.5f);
  EXPECT_FLOAT_EQ(pz[1], 2.5f);
  EXPECT_FLOAT_EQ(pz[2], 3.5f);
  EXPECT_FLOAT_EQ(pz[3], 4.5f);
}

TEST(BackendKernelClass, BlackmanWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  BlackmanWindow blackman_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_backend_test::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  blackman_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, HannWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  HannWindow hann_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = hann_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Hann window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, HannWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  HannWindow hann_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_backend_test::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  hann_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, HammingWindowPeriodicLength) {
  const KernelContext ctx{DefaultOpset(17)};
  HammingWindow hamming_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = hamming_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Hamming window is a0 + a1 = (25 - 21) / 46 = 4/46.
  EXPECT_NEAR(y.AsFloat()[0], static_cast<float>(4.0 / 46.0), 1e-6f);
}

TEST(BackendKernelClass, HammingWindowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(17)};
  HammingWindow hamming_kernel{ctx};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", onnx_backend_test::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  hamming_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], static_cast<float>(4.0 / 46.0), 1e-6f);
}

TEST(BackendKernelClass, InPlaceRejectsMismatchedShapeOrType) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});

  // Wrong dtype.
  Tensor bad_dtype("", onnx_backend_test::DataType::INT32, {3},
                   std::vector<uint8_t>(3 * sizeof(int32_t)));
  EXPECT_THROW(abs_kernel(x, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", onnx_backend_test::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", onnx_backend_test::DataType::FLOAT, {3},
                   std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_bytes), std::invalid_argument);
}

TEST(BackendKernelClass, AbsInPlaceAliasingInputAndOutput) {
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

TEST(BackendKernelClass, SubClassMatchesReference) {
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

TEST(BackendKernelClass, SubClassBroadcastsScalar) {
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

TEST(BackendKernelClass, SubInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z("", onnx_backend_test::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  sub_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 0.5f);
  EXPECT_FLOAT_EQ(pz[1], 1.5f);
  EXPECT_FLOAT_EQ(pz[2], 2.5f);
  EXPECT_FLOAT_EQ(pz[3], 3.5f);
}

TEST(BackendKernelClass, SubClassMatchesReferenceInt8) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {4}, {10, 0, -3, 7});
  Tensor y = Tensor::FromInt8("", {4}, {3, 0, 2, -1});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT8));
  const int8_t *pz = z.AsInt8();
  EXPECT_EQ(pz[0], 7);
  EXPECT_EQ(pz[1], 0);
  EXPECT_EQ(pz[2], -5);
  EXPECT_EQ(pz[3], 8);
}

TEST(BackendKernelClass, SubClassMatchesReferenceUint32) {
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromUint32("", {4}, {10u, 5u, 3u, 100u});
  Tensor y = Tensor::FromUint32("", {4}, {3u, 5u, 1u, 50u});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::UINT32));
  const uint32_t *pz = reinterpret_cast<const uint32_t *>(z.data.data());
  EXPECT_EQ(pz[0], 7u);
  EXPECT_EQ(pz[1], 0u);
  EXPECT_EQ(pz[2], 2u);
  EXPECT_EQ(pz[3], 50u);
}

TEST(BackendKernelClass, SubRejectsUnsupportedDtype) {
  // BOOL inputs are not in the supported dtype set (FLOAT/INT8/INT16/UINT8/
  // UINT16/UINT32/UINT64) so the kernel must reject them.
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {0, 1});
  EXPECT_THROW((void)sub_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, MulClassMatchesReference) {
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

TEST(BackendKernelClass, MulClassBroadcastsScalar) {
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

TEST(BackendKernelClass, MulInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {3.0f});
  Tensor z("", onnx_backend_test::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  mul_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 3.0f);
  EXPECT_FLOAT_EQ(pz[1], 6.0f);
  EXPECT_FLOAT_EQ(pz[2], 9.0f);
  EXPECT_FLOAT_EQ(pz[3], 12.0f);
}

TEST(BackendKernelClass, DivClassMatchesReference) {
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

TEST(BackendKernelClass, DivClassBroadcastsScalar) {
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

TEST(BackendKernelClass, DivInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z("", onnx_backend_test::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  div_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.0f);
  EXPECT_FLOAT_EQ(pz[1], 2.0f);
  EXPECT_FLOAT_EQ(pz[2], 3.0f);
  EXPECT_FLOAT_EQ(pz[3], 4.0f);
}

TEST(BackendKernelClass, MulClassSupportsIntegerTypes) {
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

TEST(BackendKernelClass, AddClassSupportsIntegerTypes) {
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

TEST(BackendKernelClass, DivClassSupportsIntegerTypesWithTruncation) {
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

TEST(BackendKernelClass, MatMulClassMatchesReference2D) {
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

TEST(BackendKernelClass, MatMulClassSupportsVectorMatrix) {
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

TEST(BackendKernelClass, MatMulClassBroadcastsBatchDimensions) {
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

TEST(BackendKernelClass, MatMulInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  Tensor a = Tensor::FromUint32("", {2, 3}, {1u, 2u, 3u, 4u, 5u, 6u});
  Tensor b = Tensor::FromUint32("", {3, 2}, {1u, 2u, 3u, 4u, 5u, 6u});
  Tensor y("", onnx_backend_test::DataType::UINT32, {2, 2},
           std::vector<uint8_t>(4 * sizeof(uint32_t)));
  matmul_kernel(a, b, y);
  const uint32_t *py = y.AsUint32();
  EXPECT_EQ(py[0], 22u);
  EXPECT_EQ(py[1], 28u);
  EXPECT_EQ(py[2], 49u);
  EXPECT_EQ(py[3], 64u);
}

TEST(BackendKernelClass, FloorClassMatchesReference) {
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

TEST(BackendKernelClass, CeilClassMatchesReference) {
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

TEST(BackendKernelClass, RoundClassRoundsHalvesToEven) {
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

TEST(BackendKernelClass, RoundClassNonHalvesRoundToNearest) {
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

TEST(BackendKernelClass, EinsumTransposeMatchesNumpy) {
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

TEST(BackendKernelClass, EinsumTraceMatchesSumOfDiagonal) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor y = einsum_kernel({x}, "ii->");
  ASSERT_TRUE(y.shape.empty());
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.0f + 5.0f + 9.0f);
}

TEST(BackendKernelClass, EinsumMatMulMatchesMatrixProduct) {
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

TEST(BackendKernelClass, EinsumOuterProductMatchesProduct) {
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

TEST(BackendKernelClass, EinsumImplicitOutputIsAlphabetical) {
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

TEST(BackendKernelClass, EinsumEllipsisBatchMatMul) {
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

TEST(BackendKernelClass, EinsumRejectsEmptyEquation) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(einsum_kernel({x}, ""), std::invalid_argument);
}

TEST(BackendKernelClass, EinsumRejectsRankMismatch) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  // "i" is rank-1 but the input is rank-2.
  EXPECT_THROW(einsum_kernel({x}, "i->i"), std::invalid_argument);
}

} // namespace Test
