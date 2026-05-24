// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Abs;
using onnx_backend_test::kernel::Add;
using onnx_backend_test::kernel::BlackmanWindow;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

TEST(BackendKernelClass, KernelContextStoresOpset) {
  KernelContext ctx(DefaultOpset(13));
  EXPECT_EQ(ctx.opset.domain, std::string());
  EXPECT_EQ(ctx.opset.version, 13);
}

TEST(BackendKernelClass, AbsClassMatchesReference) {
  Abs abs_kernel{KernelContext(DefaultOpset(13))};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y = abs_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(BackendKernelClass, AddClassBroadcastsScalar) {
  Add add_kernel{KernelContext(DefaultOpset(14))};
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
  BlackmanWindow blackman_kernel{KernelContext(DefaultOpset(17))};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = blackman_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Blackman window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

} // namespace Test
