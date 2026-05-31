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
using onnx_backend_test::kernel::Cos;
using onnx_backend_test::kernel::Cosh;
using onnx_backend_test::kernel::Div;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Mul;
using onnx_backend_test::kernel::Sin;
using onnx_backend_test::kernel::Sinh;
using onnx_backend_test::kernel::Sub;

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
  Tensor y("out", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float), 0xFF));
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
  Tensor z("", TensorProto::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  Tensor y("", TensorProto::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  blackman_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, InPlaceRejectsMismatchedShapeOrType) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});

  // Wrong dtype.
  Tensor bad_dtype("", TensorProto::DataType::INT32, {3},
                   std::vector<uint8_t>(3 * sizeof(int32_t)));
  EXPECT_THROW(abs_kernel(x, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(1 * sizeof(float)));
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
  Tensor z("", TensorProto::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::INT8));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::UINT32));
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
  Tensor x("", TensorProto::DataType::BOOL, {2}, {1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {0, 1});
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
  Tensor z("", TensorProto::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  Tensor z("", TensorProto::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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

} // namespace Test
