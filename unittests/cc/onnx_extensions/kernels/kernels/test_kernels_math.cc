// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/prepared_execution.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::DataType;
using core::runtime::RuntimeContext;
using core::runtime::Shape;
using core::runtime::Tensor;
using onnx_kernels::SimpleRawBufferAllocator;
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
using onnx_kernels::kernel::CumProd;
using onnx_kernels::kernel::CumSum;
using onnx_kernels::kernel::Det;
using onnx_kernels::kernel::Div;
using onnx_kernels::kernel::Einsum;
using onnx_kernels::kernel::Erf;
using onnx_kernels::kernel::Exp;
using onnx_kernels::kernel::Floor;
using onnx_kernels::kernel::Gemm;
using onnx_kernels::kernel::HammingWindow;
using onnx_kernels::kernel::HannWindow;
using onnx_kernels::kernel::Hardmax;
using onnx_kernels::kernel::HardSigmoid;
using onnx_kernels::kernel::HardSwish;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::LeakyRelu;
using onnx_kernels::kernel::Log;
using onnx_kernels::kernel::LogSoftmax;
using onnx_kernels::kernel::MatMul;
using onnx_kernels::kernel::MatMulInteger;
using onnx_kernels::kernel::Max;
using onnx_kernels::kernel::Mean;
using onnx_kernels::kernel::MelWeightMatrix;
using onnx_kernels::kernel::Min;
using onnx_kernels::kernel::Mish;
using onnx_kernels::kernel::Mod;
using onnx_kernels::kernel::Mul;
using onnx_kernels::kernel::Neg;
using onnx_kernels::kernel::Pow;
using onnx_kernels::kernel::PRelu;
using onnx_kernels::kernel::PreparedGemmB;
using onnx_kernels::kernel::Reciprocal;
using onnx_kernels::kernel::Round;
using onnx_kernels::kernel::Shrink;
using onnx_kernels::kernel::Sigmoid;
using onnx_kernels::kernel::Sign;
using onnx_kernels::kernel::Sin;
using onnx_kernels::kernel::Sinh;
using onnx_kernels::kernel::Softmax;
using onnx_kernels::kernel::Softplus;
using onnx_kernels::kernel::Softsign;
using onnx_kernels::kernel::Sqrt;
using onnx_kernels::kernel::Sub;
using onnx_kernels::kernel::Sum;
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

TEST(KernelClass, AbsKeepsSignedMinimumRepresentable) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};

  const Tensor int8 = abs_kernel(Tensor::FromInt8("", {1}, {std::numeric_limits<int8_t>::min()}));
  const Tensor int16 =
      abs_kernel(Tensor::FromInt16("", {1}, {std::numeric_limits<int16_t>::min()}));
  const Tensor int32 =
      abs_kernel(Tensor::FromInt32("", {1}, {std::numeric_limits<int32_t>::min()}));
  const Tensor int64 =
      abs_kernel(Tensor::FromInt64("", {1}, {std::numeric_limits<int64_t>::min()}));

  EXPECT_EQ(int8.AsInt8()[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(int16.AsInt16()[0], std::numeric_limits<int16_t>::min());
  EXPECT_EQ(int32.AsInt32()[0], std::numeric_limits<int32_t>::min());
  EXPECT_EQ(int64.AsInt64()[0], std::numeric_limits<int64_t>::min());
}

TEST(KernelClass, AbsClearsHalfPrecisionSignBitExactly) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};

  for (const auto &[data_type, input_bits, expected_bits] :
       std::array{std::tuple{DataType::FLOAT16, std::array<uint16_t, 3>{0xbc00, 0x8000, 0xfe01},
                             std::array<uint16_t, 3>{0x3c00, 0x0000, 0x7e01}},
                  std::tuple{DataType::BFLOAT16, std::array<uint16_t, 3>{0xbf80, 0x8000, 0xffc1},
                             std::array<uint16_t, 3>{0x3f80, 0x0000, 0x7fc1}}}) {
    Tensor input("", data_type, {3}, std::vector<uint8_t>(sizeof(input_bits)));
    std::memcpy(input.mutable_bytes(), input_bits.data(), sizeof(input_bits));

    const Tensor output = abs_kernel(input);

    EXPECT_EQ(std::memcmp(output.bytes(), expected_bits.data(), sizeof(expected_bits)), 0);
  }
}

TEST(KernelClass, AbsClassParallelPathMatchesReference) {
  // Exercises the multi-threaded ParallelFor path used by the Abs kernel: the
  // element count is large enough to exceed the grain size so several worker
  // threads process disjoint blocks. The result must stay bit-exact and cover
  // the whole range regardless of the number of threads.
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};

  const int64_t n = 32 * core::runtime::kParallelForGrainSize + 7;
  std::vector<float> values(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    values[static_cast<size_t>(i)] = (i % 2 == 0) ? -static_cast<float>(i) : static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {n}, values);
  Tensor y = abs_kernel(x);
  ASSERT_EQ(y.element_count(), n);
  const float *py = y.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(py[static_cast<size_t>(i)], static_cast<float>(i));
  }
}

TEST(KernelClass, AbsUsesTypedParallelTuningPerElementType) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  const auto float_key = abs_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  const auto double_key = abs_kernel.TuningKey(static_cast<int32_t>(DataType::DOUBLE));

  EXPECT_EQ(float_key.library, "onnx_light");
  EXPECT_EQ(float_key.kernel, "Abs");
  EXPECT_EQ(float_key.implementation, "portable");
  EXPECT_NE(float_key, double_key);
  EXPECT_EQ(abs_kernel.TuningKey(static_cast<int32_t>(DataType::STRING)).device,
            core::symbolic::Device::kUndefined);

  const auto float_schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  const auto double_schema = core::runtime::GetKernelTuningRegistry().FindSchema(double_key);
  ASSERT_NE(float_schema, nullptr);
  ASSERT_NE(double_schema, nullptr);
  EXPECT_EQ(float_schema->portable_defaults().Get<int64_t>("parallel.minimum_elements"),
            core::runtime::kParallelForGrainSize);

  core::runtime::KernelTuningParameters tuned{float_key,
                                              {{"parallel.minimum_elements", int64_t{17}}}};
  abs_kernel.Configure(tuned);
  EXPECT_EQ(abs_kernel.tuning().parallel_minimum_elements, 17);

  tuned.values["parallel.minimum_elements"] = int64_t{0};
  EXPECT_THROW(abs_kernel.Configure(tuned), std::invalid_argument);
  tuned.key.library = "other_library";
  tuned.values["parallel.minimum_elements"] = int64_t{19};
  EXPECT_THROW(abs_kernel.Configure(tuned), std::invalid_argument);
}

TEST(ParallelFor, ReusesPersistentPoolAcrossManyCalls) {
  // The pool is created once and reused: driving many parallel regions in a row
  // must keep every element covered exactly once on each call.
  const int64_t n = 4 * core::runtime::kParallelForGrainSize + 3;
  std::vector<int64_t> out(static_cast<size_t>(n), 0);
  for (int rep = 0; rep < 50; ++rep) {
    std::fill(out.begin(), out.end(), 0);
    core::runtime::ParallelFor(n, [&out](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        out[static_cast<size_t>(i)] += 1;
      }
    });
    for (int64_t i = 0; i < n; ++i) {
      ASSERT_EQ(out[static_cast<size_t>(i)], 1) << "rep=" << rep << " i=" << i;
    }
  }
}

TEST(ParallelFor, CustomGrainKeepsSmallRangeInline) {
  const int64_t n = 2 * core::runtime::kParallelForGrainSize;
  int calls = 0;
  core::runtime::ParallelFor(n, n + 1, [&](int64_t begin, int64_t end) {
    ++calls;
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, n);
  });
  EXPECT_EQ(calls, 1);
}

TEST(ParallelFor, NestedCallRunsInlineWithoutDeadlock) {
  // A ParallelFor launched from within a running block must fall back to the
  // serial path instead of deadlocking on the shared pool, and still cover the
  // whole inner range.
  const int64_t outer = 4 * core::runtime::kParallelForGrainSize + 1;
  const int64_t inner = 2 * core::runtime::kParallelForGrainSize + 1;
  std::atomic<int64_t> inner_sum{0};
  core::runtime::ParallelFor(outer, [&](int64_t obegin, int64_t oend) {
    for (int64_t o = obegin; o < oend; ++o) {
      if (o == obegin) {
        core::runtime::ParallelFor(inner, [&inner_sum](int64_t ibegin, int64_t iend) {
          inner_sum.fetch_add(iend - ibegin, std::memory_order_relaxed);
        });
      }
    }
  });
  // Each outer block ran the inner loop once over the full [0, inner) range.
  EXPECT_EQ(inner_sum.load() % inner, 0);
  EXPECT_GT(inner_sum.load(), 0);
}

TEST(ThreadPool, ZeroWorkersRunsBlocksInline) {
  core::runtime::ThreadPool pool(0);
  EXPECT_EQ(pool.worker_count(), 0);
  std::vector<int> hits(5, 0);
  pool.Run(static_cast<int64_t>(hits.size()),
           [&hits](int64_t b) { hits[static_cast<size_t>(b)] += 1; });
  for (int h : hits) {
    EXPECT_EQ(h, 1);
  }
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
  Tensor y("", core::runtime::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  neg_kernel(x, y);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 4.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
}

TEST(KernelClass, NegRejectsUnsupportedDtype) {
  // UINT8 is not in Neg's supported set (FLOAT/DOUBLE/FLOAT16/BFLOAT16/
  // INT8/INT16/INT32/INT64), so the kernel must reject it.
  const KernelContext ctx{DefaultOpset(13)};
  Neg neg_kernel{ctx};
  Tensor x = Tensor::FromUint8("", {2}, {1, 2});
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

TEST(KernelClass, ExpUsesTypedParallelTuningPerElementType) {
  const KernelContext ctx{DefaultOpset(13)};
  Exp exp_kernel{ctx};
  const auto float_key = exp_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  const auto float16_key = exp_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT16));

  EXPECT_EQ(float_key.library, "onnx_light");
  EXPECT_EQ(float_key.kernel, "Exp");
  EXPECT_EQ(float_key.implementation, "portable");
  EXPECT_NE(float_key, float16_key);
  EXPECT_EQ(exp_kernel.TuningKey(static_cast<int32_t>(DataType::INT32)).device,
            core::symbolic::Device::kUndefined);

  const auto float_schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  const auto float16_schema = core::runtime::GetKernelTuningRegistry().FindSchema(float16_key);
  ASSERT_NE(float_schema, nullptr);
  ASSERT_NE(float16_schema, nullptr);
  EXPECT_EQ(float_schema->portable_defaults().Get<int64_t>("parallel.minimum_elements"),
            core::runtime::kParallelForGrainSize);

  core::runtime::KernelTuningParameters tuned{float_key,
                                              {{"parallel.minimum_elements", int64_t{23}}}};
  exp_kernel.Configure(tuned);
  EXPECT_EQ(exp_kernel.tuning().parallel_minimum_elements, 23);

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
  Tensor y = exp_kernel(x);
  EXPECT_NEAR(y.AsFloat()[0], 0.36787944f, 1e-6f);
  EXPECT_NEAR(y.AsFloat()[1], 1.0f, 1e-6f);
  EXPECT_NEAR(y.AsFloat()[2], 2.71828183f, 1e-6f);

  tuned.key = float16_key;
  tuned.values["parallel.minimum_elements"] = int64_t{29};
  exp_kernel.Configure(tuned);
  Tensor x16 = onnx_kernels::DemoteFromFloat32(x, static_cast<int32_t>(DataType::FLOAT16));
  Tensor y16 = onnx_kernels::PromoteToFloat32(exp_kernel(x16));
  EXPECT_EQ(exp_kernel.tuning().parallel_minimum_elements, 29);
  EXPECT_NEAR(y16.AsFloat()[0], 0.36787944f, 1e-3f);
  EXPECT_NEAR(y16.AsFloat()[1], 1.0f, 1e-3f);
  EXPECT_NEAR(y16.AsFloat()[2], 2.71828183f, 2e-3f);
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

TEST(KernelClass, SignClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Sign sign_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {5}, {-2.5f, -0.5f, 0.0f, 0.5f, 2.5f});
  Tensor y = sign_kernel(x);
  ASSERT_EQ(y.element_count(), 5);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], -1.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 1.0f);
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

TEST(KernelClass, SoftmaxClassSupportsFloat16) {
  const KernelContext ctx{DefaultOpset(13)};
  Softmax softmax_kernel{ctx};

  // FLOAT16 inputs are computed in float32 and demoted back to FLOAT16. The
  // result must match the FLOAT softmax reference within half-precision rounding.
  Tensor x32 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  Tensor x16 =
      onnx_kernels::DemoteFromFloat32(x32, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  Tensor y16 = softmax_kernel(x16, 1);
  ASSERT_EQ(y16.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  ASSERT_EQ(y16.element_count(), 6);
  Tensor y = onnx_kernels::PromoteToFloat32(y16);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], 0.09003057f, 1e-3f);
  EXPECT_NEAR(py[1], 0.24472848f, 1e-3f);
  EXPECT_NEAR(py[2], 0.66524094f, 1e-3f);
  EXPECT_NEAR(py[3], 0.09003057f, 1e-3f);
  EXPECT_NEAR(py[4], 0.24472848f, 1e-3f);
  EXPECT_NEAR(py[5], 0.66524094f, 1e-3f);
}

TEST(KernelClass, Float16PromoteUsesAllocatorWhenRuntimeContextHasOne) {
  Tensor x32 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  Tensor x16 =
      onnx_kernels::DemoteFromFloat32(x32, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = onnx_kernels::PromoteToFloat32(x16, &rt);

  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  ASSERT_EQ(y.shape, x32.shape);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  const float *py = y.AsFloat();
  for (int64_t i = 0; i < x32.element_count(); ++i) {
    EXPECT_FLOAT_EQ(py[i], x32.AsFloat()[i]);
  }
}

TEST(KernelClass, Float16DemoteUsesAllocatorWhenRuntimeContextHasOne) {
  Tensor x32 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = onnx_kernels::DemoteFromFloat32(
      x32, static_cast<int32_t>(core::runtime::DataType::FLOAT16), &rt);

  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  ASSERT_EQ(y.shape, x32.shape);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  const Tensor roundtrip = onnx_kernels::PromoteToFloat32(y);
  const float *py = roundtrip.AsFloat();
  for (int64_t i = 0; i < x32.element_count(); ++i) {
    EXPECT_FLOAT_EQ(py[i], x32.AsFloat()[i]);
  }
}

TEST(KernelClass, MinMaxMeanSumUseAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(13)};
  Min min_kernel{ctx};
  Max max_kernel{ctx};
  Mean mean_kernel{ctx};
  Sum sum_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 5.0f, 3.0f, 7.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});

  {
    SimpleRawBufferAllocator alloc(1);
    RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
    Tensor zmin = min_kernel({x, y}, &rt);
    EXPECT_TRUE(zmin.has_allocation());
    EXPECT_EQ(alloc.allocated_count(), 1u);
    EXPECT_EQ(zmin.data.size(), 0u);
    const float *pmin = zmin.AsFloat();
    EXPECT_FLOAT_EQ(pmin[0], 1.0f);
    EXPECT_FLOAT_EQ(pmin[1], 4.0f);
    EXPECT_FLOAT_EQ(pmin[2], 3.0f);
    EXPECT_FLOAT_EQ(pmin[3], 7.0f);
  }

  {
    SimpleRawBufferAllocator alloc(1);
    RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
    Tensor zmax = max_kernel({x, y}, &rt);
    EXPECT_TRUE(zmax.has_allocation());
    EXPECT_EQ(alloc.allocated_count(), 1u);
    EXPECT_EQ(zmax.data.size(), 0u);
    const float *pmax = zmax.AsFloat();
    EXPECT_FLOAT_EQ(pmax[0], 2.0f);
    EXPECT_FLOAT_EQ(pmax[1], 5.0f);
    EXPECT_FLOAT_EQ(pmax[2], 6.0f);
    EXPECT_FLOAT_EQ(pmax[3], 8.0f);
  }
  {
    SimpleRawBufferAllocator alloc(1);
    RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
    Tensor zmean = mean_kernel({x, y}, &rt);
    EXPECT_TRUE(zmean.has_allocation());
    EXPECT_EQ(alloc.allocated_count(), 1u);
    EXPECT_EQ(zmean.data.size(), 0u);
    const float *pmean = zmean.AsFloat();
    EXPECT_FLOAT_EQ(pmean[0], 1.5f);
    EXPECT_FLOAT_EQ(pmean[1], 4.5f);
    EXPECT_FLOAT_EQ(pmean[2], 4.5f);
    EXPECT_FLOAT_EQ(pmean[3], 7.5f);
  }
  {
    SimpleRawBufferAllocator alloc(1);
    RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});
    Tensor zsum = sum_kernel({x, y}, &rt);
    EXPECT_TRUE(zsum.has_allocation());
    EXPECT_EQ(alloc.allocated_count(), 1u);
    EXPECT_EQ(zsum.data.size(), 0u);
    const float *psum = zsum.AsFloat();
    EXPECT_FLOAT_EQ(psum[0], 3.0f);
    EXPECT_FLOAT_EQ(psum[1], 9.0f);
    EXPECT_FLOAT_EQ(psum[2], 9.0f);
    EXPECT_FLOAT_EQ(psum[3], 15.0f);
  }
}

TEST(KernelClass, CumSumClassUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(14)};
  CumSum cumsum_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor axis = Tensor::FromInt32("", {}, {1});

  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = cumsum_kernel(x, axis, false, false, &rt);
  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
  EXPECT_FLOAT_EQ(py[3], 7.0f);
}

TEST(KernelClass, CumProdClassUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(14)};
  CumProd cumprod_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor axis = Tensor::FromInt32("", {}, {1});

  SimpleRawBufferAllocator alloc(1);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = cumprod_kernel(x, axis, false, false, &rt);
  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
  EXPECT_FLOAT_EQ(py[3], 12.0f);
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

TEST(KernelClass, AddClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {4}, {10, 0, -3, 7});
  Tensor y = Tensor::FromInt32("", {4}, {3, 0, 2, -1});
  Tensor z = add_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT32));
  const int32_t *pz = z.AsInt32();
  EXPECT_EQ(pz[0], 13);
  EXPECT_EQ(pz[1], 0);
  EXPECT_EQ(pz[2], -1);
  EXPECT_EQ(pz[3], 6);
}

TEST(KernelClass, AddClassMatchesReferenceInt64) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {4}, {10, 0, -3, 7});
  Tensor y = Tensor::FromInt64("", {4}, {3, 0, 2, -1});
  Tensor z = add_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  const int64_t *pz = z.AsInt64();
  EXPECT_EQ(pz[0], 13);
  EXPECT_EQ(pz[1], 0);
  EXPECT_EQ(pz[2], -1);
  EXPECT_EQ(pz[3], 6);
}

TEST(KernelClass, AddUsesTypedParallelTuningForBroadcasting) {
  const KernelContext ctx{DefaultOpset(14)};
  Add add_kernel{ctx};
  const auto float_key = add_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  const auto float16_key = add_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT16));

  EXPECT_EQ(float_key.kernel, "Add");
  EXPECT_NE(float_key, float16_key);
  EXPECT_EQ(add_kernel.TuningKey(static_cast<int32_t>(DataType::STRING)).device,
            core::symbolic::Device::kUndefined);
  const auto schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->portable_defaults().Get<int64_t>("parallel.minimum_elements"),
            core::runtime::kParallelForGrainSize);

  add_kernel.Configure({float_key, {{"parallel.minimum_elements", int64_t{1}}}});
  EXPECT_EQ(add_kernel.tuning().parallel_minimum_elements, 1);
  Tensor x = Tensor::FromFloat("", {2, 1}, {1.0f, 2.0f});
  Tensor y = Tensor::FromFloat("", {1, 3}, {10.0f, 20.0f, 30.0f});
  Tensor z = add_kernel(x, y);
  ASSERT_EQ(z.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(std::vector<float>(z.AsFloat(), z.AsFloat() + 6),
            (std::vector<float>{11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f}));
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
  Tensor y("out", core::runtime::DataType::FLOAT, {3},
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
  Tensor z("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  Tensor y("", core::runtime::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
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
  Tensor y("", core::runtime::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
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
  Tensor y("", core::runtime::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  hamming_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], static_cast<float>(4.0 / 46.0), 1e-6f);
}

TEST(KernelClass, InPlaceRejectsMismatchedShapeOrType) {
  const KernelContext ctx{DefaultOpset(13)};
  Abs abs_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});

  // Wrong dtype.
  Tensor bad_dtype("", core::runtime::DataType::INT32, {3},
                   std::vector<uint8_t>(3 * sizeof(int32_t)));
  EXPECT_THROW(abs_kernel(x, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", core::runtime::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", core::runtime::DataType::FLOAT, {3},
                   std::vector<uint8_t>(1 * sizeof(float)));
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
  Tensor z("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT8));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::UINT32));
  const uint32_t *pz = reinterpret_cast<const uint32_t *>(z.data.data());
  EXPECT_EQ(pz[0], 7u);
  EXPECT_EQ(pz[1], 0u);
  EXPECT_EQ(pz[2], 2u);
  EXPECT_EQ(pz[3], 50u);
}

TEST(KernelClass, SubClassMatchesReferenceInt64) {
  // ``test_cc_flexattention_relative_positional`` exercises Sub on INT64
  // index tensors (q_idx - k_idx), so the kernel must handle INT64.
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {4}, {10, 0, -3, 7});
  Tensor y = Tensor::FromInt64("", {4}, {3, 0, 2, -1});
  Tensor z = sub_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  const int64_t *pz = z.AsInt64();
  EXPECT_EQ(pz[0], 7);
  EXPECT_EQ(pz[1], 0);
  EXPECT_EQ(pz[2], -5);
  EXPECT_EQ(pz[3], 8);
}

TEST(KernelClass, SubRejectsUnsupportedDtype) {
  // BOOL inputs are not in the supported dtype set (FLOAT/INT8/INT16/INT32/
  // INT64/UINT8/UINT16/UINT32/UINT64) so the kernel must reject them.
  const KernelContext ctx{DefaultOpset(14)};
  Sub sub_kernel{ctx};
  Tensor x("", core::runtime::DataType::BOOL, {2}, {1, 0});
  Tensor y("", core::runtime::DataType::BOOL, {2}, {0, 1});
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

TEST(KernelClass, MulUsesTypedParallelTuningForEqualShapes) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  const auto float_key = mul_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));

  EXPECT_EQ(float_key.kernel, "Mul");
  const auto schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->portable_defaults().Get<int64_t>("parallel.minimum_elements"),
            core::runtime::kParallelForGrainSize);

  mul_kernel.Configure({float_key, {{"parallel.minimum_elements", int64_t{1}}}});
  EXPECT_EQ(mul_kernel.tuning().parallel_minimum_elements, 1);
  Tensor x = Tensor::FromFloat("", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = Tensor::FromFloat("", {6}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  Tensor z = mul_kernel(x, y);
  EXPECT_EQ(std::vector<float>(z.AsFloat(), z.AsFloat() + 6),
            (std::vector<float>{2.0f, 6.0f, 12.0f, 20.0f, 30.0f, 42.0f}));
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
  Tensor z("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  mul_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 3.0f);
  EXPECT_FLOAT_EQ(pz[1], 6.0f);
  EXPECT_FLOAT_EQ(pz[2], 9.0f);
  EXPECT_FLOAT_EQ(pz[3], 12.0f);
}

TEST(KernelClass, MulClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {3}, {1, -2, 3});
  Tensor y = Tensor::FromInt32("", {3}, {4, 5, -6});
  Tensor z = mul_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT32));
  const int32_t *pz = z.AsInt32();
  EXPECT_EQ(pz[0], 4);
  EXPECT_EQ(pz[1], -10);
  EXPECT_EQ(pz[2], -18);
}

TEST(KernelClass, MulClassMatchesReferenceInt64) {
  const KernelContext ctx{DefaultOpset(14)};
  Mul mul_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {3}, {1, -2, 3});
  Tensor y = Tensor::FromInt64("", {3}, {4, 5, -6});
  Tensor z = mul_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  const int64_t *pz = z.AsInt64();
  EXPECT_EQ(pz[0], 4);
  EXPECT_EQ(pz[1], -10);
  EXPECT_EQ(pz[2], -18);
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

TEST(KernelClass, DivClassBroadcastsScalarOverEmptyAxis) {
  // Regression test: broadcasting a scalar (or size-1) operand against an
  // input whose shape contains a zero-length axis must yield a zero-element
  // output, not over-estimate the element count. The earlier ``max(dx, dy)``
  // broadcast rule produced an output dim of 1 for a (0, 1) pair, so the
  // iteration read past the empty input buffer and crashed.
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x("", core::runtime::DataType::FLOAT, {1, 512, 0}, std::vector<uint8_t>(0));
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z = div_kernel(x, y);
  EXPECT_EQ(z.element_count(), 0);
  const std::vector<int64_t> expected_shape{1, 512, 0};
  EXPECT_EQ(z.shape, expected_shape);
}

TEST(KernelClass, DivInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(14)};
  Div div_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  {
    Tensor x = Tensor::FromInt64("", {4}, {-3, 3, -3, 3});
    Tensor y = Tensor::FromInt64("", {4}, {2, 2, -2, -2});
    Tensor z = div_kernel(x, y);
    const int64_t *pz = z.AsInt64();
    EXPECT_EQ(pz[0], -1);
    EXPECT_EQ(pz[1], 1);
    EXPECT_EQ(pz[2], 1);
    EXPECT_EQ(pz[3], -1);
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
    Tensor x("", static_cast<int32_t>(core::runtime::DataType::FLOAT16), {6}, {});
    Tensor y("", static_cast<int32_t>(core::runtime::DataType::FLOAT16), {6}, {});
    // Bit patterns of float16(-4.3, 7.2, 5.0, 4.3, -7.2, 8.0) and
    // float16(2.1, -3.4, 8.0, -2.1, 3.4, 5.0).
    const std::vector<uint16_t> xb = {0xc44dU, 0x4733U, 0x4500U, 0x444dU, 0xc733U, 0x4800U};
    const std::vector<uint16_t> yb = {0x4033U, 0xc2cdU, 0x4800U, 0xc033U, 0x42cdU, 0x4500U};
    x.data.assign(reinterpret_cast<const uint8_t *>(xb.data()),
                  reinterpret_cast<const uint8_t *>(xb.data() + xb.size()));
    y.data.assign(reinterpret_cast<const uint8_t *>(yb.data()),
                  reinterpret_cast<const uint8_t *>(yb.data() + yb.size()));
    Tensor z = mod_kernel(x, y, /*fmod=*/1);
    ASSERT_EQ(z.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
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
    Tensor x("", static_cast<int32_t>(core::runtime::DataType::FLOAT16), {1},
             std::vector<uint8_t>(sizeof(uint16_t), 0));
    Tensor y("", static_cast<int32_t>(core::runtime::DataType::FLOAT16), {1},
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
  Tensor y("", core::runtime::DataType::UINT32, {2, 2}, std::vector<uint8_t>(4 * sizeof(uint32_t)));
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
  Tensor a_zp("", core::runtime::DataType::UINT8, {1}, std::vector<uint8_t>{12});
  Tensor b_zp("", core::runtime::DataType::UINT8, {1}, std::vector<uint8_t>{0});
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
  Tensor a_zp("", core::runtime::DataType::INT8, {},
              std::vector<uint8_t>{static_cast<uint8_t>(static_cast<int8_t>(1))});
  Tensor b_zp("", core::runtime::DataType::INT8, {},
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
  Tensor y("", core::runtime::DataType::INT32, {2, 2}, std::vector<uint8_t>(4 * sizeof(int32_t)));
  mmi(a, b, a_zp, b_zp, y);
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], 58);
  EXPECT_EQ(py[1], 64);
  EXPECT_EQ(py[2], 139);
  EXPECT_EQ(py[3], 154);
}

TEST(KernelClass, MatMulAndMatMulIntegerShareOutputShapeRules) {
  const KernelContext matmul_ctx{DefaultOpset(13)};
  const KernelContext mmi_ctx{DefaultOpset(10)};
  MatMul matmul_kernel{matmul_ctx};
  MatMulInteger mmi{mmi_ctx};
  Tensor a_zero_point;
  Tensor b_zero_point;

  {
    Tensor a = Tensor::FromInt32("", {3}, {2, 3, 4});
    Tensor b = Tensor::FromInt32("", {3, 2}, {1, 5, 2, 6, 3, 7});
    Tensor a_q = Tensor::FromUint8("", {3}, {2, 3, 4});
    Tensor b_q = Tensor::FromUint8("", {3, 2}, {1, 5, 2, 6, 3, 7});
    const Tensor y = matmul_kernel(a, b);
    const Tensor y_q = mmi(a_q, b_q, a_zero_point, b_zero_point);
    EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(y.shape, y_q.shape);
  }
  {
    Tensor a = Tensor::FromInt32("", {2, 1, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::FromInt32("", {1, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor a_q = Tensor::FromUint8("", {2, 1, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b_q = Tensor::FromUint8("", {1, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const Tensor y = matmul_kernel(a, b);
    const Tensor y_q = mmi(a_q, b_q, a_zero_point, b_zero_point);
    EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 1, 4}));
    EXPECT_EQ(y.shape, y_q.shape);
  }
  {
    Tensor a = Tensor::FromInt32("", {3}, {1, 2, 3});
    Tensor b = Tensor::FromInt32("", {3}, {4, 5, 6});
    Tensor a_q = Tensor::FromUint8("", {3}, {1, 2, 3});
    Tensor b_q = Tensor::FromUint8("", {3}, {4, 5, 6});
    const Tensor y = matmul_kernel(a, b);
    const Tensor y_q = mmi(a_q, b_q, a_zero_point, b_zero_point);
    EXPECT_EQ(y.shape, (std::vector<int64_t>{}));
    EXPECT_EQ(y.shape, y_q.shape);
  }
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

TEST(KernelClass, MatMulIntegerWithPerColumnBZeroPoint) {
  // Per-column b_zero_point: b_zp[j] is subtracted from each element in column j of B.
  // A [2x3] UINT8, B [3x2] UINT8, b_zp = [1, 2] (one per output column).
  // Y[i][j] = sum_k A[i][k] * (B[k][j] - b_zp[j])
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {2, 3}, {11, 7, 3, 10, 6, 2});
  Tensor b = Tensor::FromUint8("", {3, 2}, {1, 4, 2, 5, 3, 6});
  Tensor a_zp;
  Tensor b_zp("", core::runtime::DataType::UINT8, {2}, std::vector<uint8_t>{1, 2});
  Tensor y = mmi(a, b, a_zp, b_zp);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int32_t *py = y.AsInt32();
  // Y[0][0] = 11*(1-1) + 7*(2-1) + 3*(3-1) = 0 + 7 + 6 = 13
  // Y[0][1] = 11*(4-2) + 7*(5-2) + 3*(6-2) = 22 + 21 + 12 = 55
  // Y[1][0] = 10*(1-1) + 6*(2-1) + 2*(3-1) = 0 + 6 + 4 = 10
  // Y[1][1] = 10*(4-2) + 6*(5-2) + 2*(6-2) = 20 + 18 + 8 = 46
  EXPECT_EQ(py[0], 13);
  EXPECT_EQ(py[1], 55);
  EXPECT_EQ(py[2], 10);
  EXPECT_EQ(py[3], 46);
}

TEST(KernelClass, MatMulIntegerWithPerRowAZeroPoint) {
  // Per-row a_zero_point: a_zp[i] is subtracted from each element in row i of A.
  // A [2x3] UINT8, B [3x2] UINT8, a_zp = [1, 2] (one per input row).
  // Y[i][j] = sum_k (A[i][k] - a_zp[i]) * B[k][j]
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {2, 3}, {11, 7, 3, 10, 6, 2});
  Tensor b = Tensor::FromUint8("", {3, 2}, {1, 4, 2, 5, 3, 6});
  Tensor a_zp("", core::runtime::DataType::UINT8, {2}, std::vector<uint8_t>{1, 2});
  Tensor b_zp;
  Tensor y = mmi(a, b, a_zp, b_zp);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int32_t *py = y.AsInt32();
  // Y[0][0] = (11-1)*1 + (7-1)*2 + (3-1)*3 = 10 + 12 + 6 = 28
  // Y[0][1] = (11-1)*4 + (7-1)*5 + (3-1)*6 = 40 + 30 + 12 = 82
  // Y[1][0] = (10-2)*1 + (6-2)*2 + (2-2)*3 = 8 + 8 + 0 = 16
  // Y[1][1] = (10-2)*4 + (6-2)*5 + (2-2)*6 = 32 + 20 + 0 = 52
  EXPECT_EQ(py[0], 28);
  EXPECT_EQ(py[1], 82);
  EXPECT_EQ(py[2], 16);
  EXPECT_EQ(py[3], 52);
}

TEST(KernelClass, MatMulIntegerUsesAllocatorForZeroPoints) {
  // When the RuntimeContext carries an allocator, the transient per-row and
  // per-column zero-point buffers are drawn from it (via TemporaryTypedBuffer)
  // and freed before returning, leaving only the output allocation alive.
  const KernelContext ctx{DefaultOpset(10)};
  MatMulInteger mmi{ctx};
  Tensor a = Tensor::FromUint8("", {2, 3}, {11, 7, 3, 10, 6, 2});
  Tensor b = Tensor::FromUint8("", {3, 2}, {1, 4, 2, 5, 3, 6});
  Tensor a_zp("", core::runtime::DataType::UINT8, {2}, std::vector<uint8_t>{1, 2});
  Tensor b_zp("", core::runtime::DataType::UINT8, {2}, std::vector<uint8_t>{1, 2});

  // Capacity 3: the persistent output plus the two transient zero-point buffers
  // that must be alive simultaneously during the computation.
  SimpleRawBufferAllocator alloc(3);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = mmi(a, b, a_zp, b_zp, &rt);
  EXPECT_TRUE(y.has_allocation());
  // The two transient zero-point buffers have been freed, leaving only the output.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int32_t *py = y.AsInt32();
  // Y[i][j] = sum_k (A[i][k] - a_zp[i]) * (B[k][j] - b_zp[j])
  EXPECT_EQ(py[0], 10);
  EXPECT_EQ(py[1], 46);
  EXPECT_EQ(py[2], 4);
  EXPECT_EQ(py[3], 28);
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

TEST(KernelClass, EinsumUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor a = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromFloat("", {3, 2}, {7, 8, 9, 10, 11, 12});
  // Three concurrent slots are needed during the call: the output tensor plus
  // the transient ``ix`` and ``in_ptrs`` scratch buffers routed through the
  // allocator. The scratch buffers are freed before returning, leaving only
  // the output allocation alive.
  SimpleRawBufferAllocator alloc(3);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = einsum_kernel({a, b}, "ij,jk->ik", &rt);

  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1 * 7 + 2 * 9 + 3 * 11);
  EXPECT_FLOAT_EQ(py[1], 1 * 8 + 2 * 10 + 3 * 12);
  EXPECT_FLOAT_EQ(py[2], 4 * 7 + 5 * 9 + 6 * 11);
  EXPECT_FLOAT_EQ(py[3], 4 * 8 + 5 * 10 + 6 * 12);
}

TEST(KernelClass, EinsumScalarUsesAllocatorWithoutError) {
  // The scalar equation "->" produces an empty ``all_labels`` set (zero
  // iteration labels). The ``ix`` scratch buffer must still be created safely
  // when an allocator is attached; verify the reduction runs and the result is
  // correct.
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {}, {5.0f});
  SimpleRawBufferAllocator alloc(3);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = einsum_kernel({x}, "->", &rt);

  EXPECT_TRUE(y.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);
  ASSERT_TRUE(y.shape.empty());
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 5.0f);
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

TEST(KernelClass, EinsumReusesKernelAcrossEquationsAndShapes) {
  // A single kernel instance caches its contraction plan; reusing it with a
  // different equation or different input shapes must rebuild the plan and
  // still produce correct results.
  const KernelContext ctx{DefaultOpset(13)};
  Einsum einsum_kernel{ctx};

  Tensor a = Tensor::FromFloat("", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = Tensor::FromFloat("", {3, 2}, {7, 8, 9, 10, 11, 12});

  // First call builds the plan for the matmul equation.
  Tensor mm = einsum_kernel({a, b}, "ij,jk->ik");
  ASSERT_EQ(mm.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_FLOAT_EQ(mm.AsFloat()[0], 1 * 7 + 2 * 9 + 3 * 11);

  // Same equation and shapes again: cached plan must yield the same result.
  Tensor mm2 = einsum_kernel({a, b}, "ij,jk->ik");
  ASSERT_EQ(mm2.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_FLOAT_EQ(mm2.AsFloat()[0], 1 * 7 + 2 * 9 + 3 * 11);
  EXPECT_FLOAT_EQ(mm2.AsFloat()[3], 4 * 8 + 5 * 10 + 6 * 12);

  // Different equation on the same instance: plan must be rebuilt.
  Tensor t = einsum_kernel({a}, "ij->ji");
  ASSERT_EQ(t.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 4.0f);

  // Same equation as the first call but different input shapes: the cached
  // plan keyed on shapes must be invalidated and rebuilt.
  Tensor c = Tensor::FromFloat("", {1, 2}, {1, 2});
  Tensor d = Tensor::FromFloat("", {2, 1}, {3, 4});
  Tensor mm3 = einsum_kernel({c, d}, "ij,jk->ik");
  ASSERT_EQ(mm3.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(mm3.AsFloat()[0], 1 * 3 + 2 * 4);
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

TEST(KernelClass, TopKAcceptsSortedFalse) {
  const KernelContext ctx{DefaultOpset(11)};
  TopK topk_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {1, 5}, {5.0f, 1.0f, 4.0f, 2.0f, 3.0f});
  auto [values, indices] = topk_kernel(x, /*k=*/3, /*axis=*/-1, /*largest=*/true, /*sorted=*/false);
  ASSERT_EQ(values.shape, (std::vector<int64_t>{1, 3}));
  EXPECT_FLOAT_EQ(values.AsFloat()[0], 5.0f);
  EXPECT_FLOAT_EQ(values.AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(values.AsFloat()[2], 3.0f);
  EXPECT_EQ(indices.AsInt64()[0], 0);
  EXPECT_EQ(indices.AsInt64()[1], 2);
  EXPECT_EQ(indices.AsInt64()[2], 4);
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

TEST(KernelClass, TopKClassUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(11)};
  TopK topk_kernel{ctx};

  Tensor x = Tensor::FromFloat("", {1, 4}, {3.0f, 1.0f, 4.0f, 2.0f});
  // Capacity 3: the two persistent outputs (values, indices) plus one transient
  // scratch index buffer that TopK draws from the allocator and frees before
  // returning.
  SimpleRawBufferAllocator alloc(3);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  auto [values, indices] =
      topk_kernel(x, /*k=*/2, /*axis=*/-1, /*largest=*/true, /*sorted=*/true, &rt);
  EXPECT_TRUE(values.has_allocation());
  EXPECT_TRUE(indices.has_allocation());
  // The transient scratch buffer has been freed, leaving only the two outputs.
  EXPECT_EQ(alloc.allocated_count(), 2u);
  ASSERT_EQ(values.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_FLOAT_EQ(values.AsFloat()[0], 4.0f);
  EXPECT_FLOAT_EQ(values.AsFloat()[1], 3.0f);
  EXPECT_EQ(indices.AsInt64()[0], 2);
  EXPECT_EQ(indices.AsInt64()[1], 0);
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
  Tensor y("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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

// ---------------------------------------------------------------------------
// LeakyRelu kernel tests
// ---------------------------------------------------------------------------

TEST(KernelClass, LeakyReluClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(16)};
  LeakyRelu leakyrelu_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
  Tensor y = leakyrelu_kernel(x, 0.1f);
  ASSERT_EQ(y.element_count(), 6);
  ASSERT_EQ(y.data_type, core::runtime::DataType::FLOAT);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.3f);
  EXPECT_FLOAT_EQ(py[1], -0.1f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
  EXPECT_FLOAT_EQ(py[3], 1.0f);
  EXPECT_FLOAT_EQ(py[4], 2.0f);
  EXPECT_FLOAT_EQ(py[5], 3.0f);
}

TEST(KernelClass, LeakyReluClassDefaultAlpha) {
  const KernelContext ctx{DefaultOpset(16)};
  LeakyRelu leakyrelu_kernel{ctx};
  // Default alpha is 0.01f to match the ONNX schema.
  Tensor x = Tensor::FromFloat("", {3}, {-2.0f, 0.0f, 5.0f});
  Tensor y = leakyrelu_kernel(x);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.02f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 5.0f);
}

TEST(KernelClass, LeakyReluClassSupportsDouble) {
  const KernelContext ctx{DefaultOpset(16)};
  LeakyRelu leakyrelu_kernel{ctx};
  Tensor x("", core::runtime::DataType::DOUBLE, {3}, std::vector<uint8_t>(3 * sizeof(double)));
  double *px = reinterpret_cast<double *>(x.data.data());
  px[0] = -4.0;
  px[1] = 0.0;
  px[2] = 7.0;
  Tensor y = leakyrelu_kernel(x, 0.25f);
  ASSERT_EQ(y.data_type, core::runtime::DataType::DOUBLE);
  const double *py = reinterpret_cast<const double *>(y.data.data());
  EXPECT_DOUBLE_EQ(py[0], -1.0);
  EXPECT_DOUBLE_EQ(py[1], 0.0);
  EXPECT_DOUBLE_EQ(py[2], 7.0);
}

TEST(KernelClass, LeakyReluInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(16)};
  LeakyRelu leakyrelu_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {-1.0f, -2.0f, 3.0f, -4.0f});
  Tensor y("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  leakyrelu_kernel(x, 0.5f, y);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -0.5f);
  EXPECT_FLOAT_EQ(py[1], -1.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
  EXPECT_FLOAT_EQ(py[3], -2.0f);
}

TEST(KernelClass, LeakyReluRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(16)};
  LeakyRelu leakyrelu_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {2}, {-1, 2});
  EXPECT_THROW(leakyrelu_kernel(x, 0.1f), std::invalid_argument);
}

TEST(KernelClass, PowClassMatchesReferenceFloat) {
  // Matches the upstream ``test_pow_example`` reference case.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  Tensor z = pow_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  ASSERT_EQ(z.data_type, core::runtime::DataType::FLOAT);
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

TEST(KernelClass, PowUsesTypedParallelTuningForGeneralBroadcasting) {
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  const auto key = pow_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  pow_kernel.Configure({key, {{"parallel.minimum_elements", int64_t{1}}}});

  EXPECT_EQ(pow_kernel.tuning().parallel_minimum_elements, 1);
  Tensor x = Tensor::FromFloat("", {2, 1}, {2.0f, 3.0f});
  Tensor y = Tensor::FromInt64("", {1, 3}, {1, 2, 3});
  Tensor z = pow_kernel(x, y);
  ASSERT_EQ(z.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(std::vector<float>(z.AsFloat(), z.AsFloat() + 6),
            (std::vector<float>{2.0f, 4.0f, 8.0f, 3.0f, 9.0f, 27.0f}));
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
    ASSERT_EQ(z.data_type, core::runtime::DataType::FLOAT);
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
    ASSERT_EQ(z.data_type, core::runtime::DataType::INT64);
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
    ASSERT_EQ(z.data_type, core::runtime::DataType::INT32);
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
    ASSERT_EQ(z.data_type, core::runtime::DataType::FLOAT);
    const float *pz = z.AsFloat();
    EXPECT_FLOAT_EQ(pz[0], 1.0f);
    EXPECT_FLOAT_EQ(pz[1], 32.0f);
    EXPECT_FLOAT_EQ(pz[2], 729.0f);
  }
}

TEST(KernelClass, PowUsesAllocatorForFloat16ExponentDecode) {
  // FLOAT16 base with a FLOAT16 exponent decodes the exponent into a transient
  // float buffer. When the RuntimeContext carries an allocator that buffer must
  // be drawn from it (via TemporaryTypedBuffer) and freed before returning,
  // leaving only the output allocation alive.
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x32 = Tensor::FromFloat("", {3}, {2.0f, 3.0f, 4.0f});
  Tensor y32 = Tensor::FromFloat("", {3}, {2.0f, 2.0f, 2.0f});
  Tensor x16 =
      onnx_kernels::DemoteFromFloat32(x32, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  Tensor y16 =
      onnx_kernels::DemoteFromFloat32(y32, static_cast<int32_t>(core::runtime::DataType::FLOAT16));

  // Capacity 2: the persistent output plus the transient decode buffer that
  // must be alive simultaneously during the computation.
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor z16 = pow_kernel(x16, y16, &rt);
  EXPECT_TRUE(z16.has_allocation());
  // The transient exponent decode buffer has been freed, leaving only the output.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  ASSERT_EQ(z16.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
  const Tensor z = onnx_kernels::PromoteToFloat32(z16);
  const float *pz = z.AsFloat();
  EXPECT_NEAR(pz[0], 4.0f, 1e-2f);
  EXPECT_NEAR(pz[1], 9.0f, 1e-2f);
  EXPECT_NEAR(pz[2], 16.0f, 1e-2f);
}

TEST(KernelClass, PowInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(15)};
  Pow pow_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 3.0f, 2.0f, 3.0f});
  Tensor z("", core::runtime::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
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
  Tensor x("", core::runtime::DataType::DOUBLE, {static_cast<int64_t>(values.size())}, bytes);
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

namespace {

// Builds a half-precision tensor (FLOAT16 or BFLOAT16) from a flattened list
// of float32 sample values by promoting/demoting through
// :cpp:func:`DemoteFromFloat32`.
Tensor MakeHalfTensor(int32_t target_dtype, const std::vector<int64_t> &shape,
                      const std::vector<float> &values) {
  Tensor f = Tensor::FromFloat("", shape, values);
  return onnx_kernels::DemoteFromFloat32(f, target_dtype);
}

// Decodes a half-precision tensor back into a std::vector<float> by promoting
// to FLOAT32. Used to verify the bit-pattern of kernel outputs against the
// equivalent FLOAT computation rounded to the same half-precision dtype.
std::vector<float> DecodeHalfTensor(const Tensor &t) {
  Tensor f = onnx_kernels::PromoteToFloat32(t);
  const float *p = f.AsFloat();
  return std::vector<float>(p, p + f.element_count());
}

} // namespace

TEST(KernelClass, GemmPreparedConstantBMatchesReferenceForBothTransposeModes) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  const Tensor a = Tensor::FromFloat("A", {1, 3}, {1.0f, 2.0f, 3.0f});

  for (const int64_t trans_b : {int64_t{0}, int64_t{1}}) {
    core::runtime::PreparedExecutionState state(1, 1);
    const Tensor b = trans_b == 0
                         ? Tensor::FromFloat("B", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
                         : Tensor::FromFloat("B", {2, 3}, {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f});
    const auto prepared = gemm_kernel.PrepareConstantB(b, trans_b, state);
    const auto shared = gemm_kernel.PrepareConstantB(b, trans_b, state);

    ASSERT_TRUE(prepared.IsReady());
    ASSERT_TRUE(shared.IsReady());
    EXPECT_EQ(state.prepared_arena().allocated_count(), 1u);
    const Tensor expected = gemm_kernel(a, b, nullptr, 1.0f, 0.0f, /*transA=*/0, trans_b);
    const Tensor got = gemm_kernel(a, prepared, nullptr, 1.0f, 0.0f, /*transA=*/0);
    ASSERT_EQ(got.shape, expected.shape);
    EXPECT_EQ(
        std::vector<float>(got.AsFloat(), got.AsFloat() + got.element_count()),
        std::vector<float>(expected.AsFloat(), expected.AsFloat() + expected.element_count()));
  }
}

TEST(KernelClass, GemmPreparedConstantBAllocationFailureCanRetry) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  core::runtime::PreparedExecutionState state(1, 1);
  const Tensor b = Tensor::FromFloat("B", {1, 1}, {2.0f});
  core::runtime::AllocationHandle occupied(&state.prepared_arena(),
                                           state.prepared_arena().Allocate(sizeof(float)));

  EXPECT_THROW(gemm_kernel.PrepareConstantB(b, /*transB=*/0, state), std::bad_alloc);
  occupied.Reset();
  const auto prepared = gemm_kernel.PrepareConstantB(b, /*transB=*/0, state);
  EXPECT_TRUE(prepared.IsReady());
}

TEST(KernelClass, PreparedPlanOverlapsDependentGemmsAndReusesWeights) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  core::runtime::PreparedExecutionState state(2, 2);
  const Tensor input = Tensor::FromFloat("A", {1, 2}, {1.0f, 2.0f});
  const Tensor first_weight = Tensor::FromFloat("B1", {2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});
  const Tensor second_weight = Tensor::FromFloat("B2", {2, 1}, {3.0f, 4.0f});
  PreparedGemmB first_prepared;
  PreparedGemmB second_prepared;
  std::atomic<int> preparation_count{0};
  std::atomic<bool> second_weight_finished{false};
  core::runtime::PreparedExecutionPlan plan({
      core::runtime::TaskDescriptor{core::runtime::TaskId{1}, core::runtime::TaskScope::kSession,
                                    core::runtime::TaskKind::kPrepare,
                                    core::runtime::ResourceClass::kCpu},
      core::runtime::TaskDescriptor{core::runtime::TaskId{2}, core::runtime::TaskScope::kSession,
                                    core::runtime::TaskKind::kPrepare,
                                    core::runtime::ResourceClass::kCpu},
      core::runtime::TaskDescriptor{core::runtime::TaskId{3},
                                    core::runtime::TaskScope::kInvocation,
                                    core::runtime::TaskKind::kExecute,
                                    core::runtime::ResourceClass::kCpu,
                                    {core::runtime::TaskId{1}}},
      core::runtime::TaskDescriptor{core::runtime::TaskId{4},
                                    core::runtime::TaskScope::kInvocation,
                                    core::runtime::TaskKind::kExecute,
                                    core::runtime::ResourceClass::kCpu,
                                    {core::runtime::TaskId{2}, core::runtime::TaskId{3}}},
  });

  auto run = [&]() {
    Tensor first_output;
    Tensor second_output;
    bool first_executed_before_second_weight = false;
    const core::runtime::PreparedExecutionResult result =
        plan.RunSequential(state, [&](const core::runtime::TaskDescriptor &task,
                                      core::runtime::PreparedExecutionState &) {
          switch (task.id.value) {
          case 1:
            ++preparation_count;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            first_prepared = gemm_kernel.PrepareConstantB(first_weight, 0, state);
            break;
          case 2:
            ++preparation_count;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            second_prepared = gemm_kernel.PrepareConstantB(second_weight, 0, state);
            second_weight_finished = true;
            break;
          case 3:
            first_executed_before_second_weight = !second_weight_finished;
            first_output = gemm_kernel(input, first_prepared, nullptr, 1.0f, 0.0f, 0, nullptr);
            break;
          case 4:
            second_output =
                gemm_kernel(first_output, second_prepared, nullptr, 1.0f, 0.0f, 0, nullptr);
            break;
          default:
            FAIL() << "Unexpected prepared task ID " << task.id.value;
          }
        });
    EXPECT_TRUE(first_executed_before_second_weight);
    EXPECT_EQ(result.diagnostics[2].status, core::runtime::TaskStatus::kSucceeded);
    EXPECT_EQ(result.diagnostics[3].status, core::runtime::TaskStatus::kSucceeded);
    ASSERT_EQ(second_output.shape, (Shape{1, 1}));
    EXPECT_FLOAT_EQ(second_output.As<float>()[0], 19.0f);
  };

  run();
  second_weight_finished = false;
  run();
  EXPECT_EQ(preparation_count, 2);
}

// Verifies that the half-precision path of ``kernel::Gemm`` matches the
// FLOAT path rounded through the same half-precision dtype, for both
// FLOAT16 and BFLOAT16 inputs (with and without the optional ``C`` bias).
TEST(KernelClass, GemmHalfPrecisionMatchesFloatReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  const std::vector<float> a_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b_vals = {0.5f, -1.0f, 1.5f, 2.0f, -0.5f, 0.25f};
  const std::vector<float> c_vals = {0.125f, -0.5f};
  const Tensor a_f = Tensor::FromFloat("", {2, 3}, a_vals);
  const Tensor b_f = Tensor::FromFloat("", {3, 2}, b_vals);
  const Tensor c_f = Tensor::FromFloat("", {2}, c_vals);
  const float alpha = 0.5f;
  const float beta = 1.5f;
  const Tensor ref = gemm_kernel(a_f, b_f, &c_f, alpha, beta, 0, 0);
  ASSERT_EQ(ref.shape, (std::vector<int64_t>{2, 2}));

  for (int32_t target : {static_cast<int32_t>(core::runtime::DataType::FLOAT16),
                         static_cast<int32_t>(core::runtime::DataType::BFLOAT16)}) {
    const Tensor a_h = MakeHalfTensor(target, {2, 3}, a_vals);
    const Tensor b_h = MakeHalfTensor(target, {3, 2}, b_vals);
    const Tensor c_h = MakeHalfTensor(target, {2}, c_vals);
    const Tensor y_h = gemm_kernel(a_h, b_h, &c_h, alpha, beta, 0, 0);
    ASSERT_EQ(y_h.data_type, target);
    ASSERT_EQ(y_h.shape, ref.shape);
    // Compare via the matching half rounding of the FLOAT reference.
    const Tensor expected =
        onnx_kernels::DemoteFromFloat32(onnx_kernels::PromoteToFloat32(ref), target);
    // Bit-for-bit equality is too strict because the inner computation is
    // performed in FLOAT32 — compare numerical values after promoting both.
    const std::vector<float> got = DecodeHalfTensor(y_h);
    const std::vector<float> exp = DecodeHalfTensor(expected);
    ASSERT_EQ(got.size(), exp.size());
    const float tol =
        target == static_cast<int32_t>(core::runtime::DataType::FLOAT16) ? 1e-2f : 5e-2f;
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_NEAR(got[i], exp[i], tol) << "i=" << i;
    }
  }

  // In-place overload preserves the half-precision dtype of the output.
  const Tensor a_h =
      MakeHalfTensor(static_cast<int32_t>(core::runtime::DataType::FLOAT16), {2, 3}, a_vals);
  const Tensor b_h =
      MakeHalfTensor(static_cast<int32_t>(core::runtime::DataType::FLOAT16), {3, 2}, b_vals);
  Tensor y_h("", core::runtime::DataType::FLOAT16, {2, 2},
             std::vector<uint8_t>(4 * sizeof(uint16_t)));
  gemm_kernel(a_h, b_h, /*c=*/nullptr, 1.0f, 0.0f, 0, 0, y_h);
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT16));
}

TEST(KernelClass, GemmUsesTypedTilingPackingTaskAndConversionTuning) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  const auto float_key = gemm_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  const auto half_key = gemm_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT16));

  EXPECT_EQ(float_key.library, "onnx_light");
  EXPECT_EQ(float_key.kernel, "Gemm");
  EXPECT_EQ(float_key.implementation, "portable");
  EXPECT_NE(float_key, half_key);
  EXPECT_EQ(gemm_kernel.TuningKey(static_cast<int32_t>(DataType::STRING)).device,
            core::symbolic::Device::kUndefined);

  const auto schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  ASSERT_NE(schema, nullptr);
  const auto &defaults = schema->portable_defaults();
  EXPECT_EQ(defaults.Get<int64_t>("algorithm.tile_m"), 64);
  EXPECT_EQ(defaults.Get<int64_t>("algorithm.tile_n"), 256);
  EXPECT_EQ(defaults.Get<int64_t>("algorithm.tile_k"), 256);
  EXPECT_EQ(defaults.Get<int64_t>("algorithm.pack_b_minimum_elements"), 16384);
  EXPECT_EQ(defaults.Get<int64_t>("algorithm.skinny_m_limit"), 8);
  EXPECT_EQ(defaults.Get<int64_t>("parallel.fmas_per_work_unit"), 256);
  EXPECT_EQ(defaults.Get<int64_t>("parallel.minimum_tasks"), 2);
  EXPECT_EQ(defaults.Get<int64_t>("conversion.parallel_minimum_elements"), 1048576);

  core::runtime::KernelTuningParameters tuned = defaults;
  tuned.values["algorithm.tile_m"] = int64_t{2};
  tuned.values["algorithm.tile_n"] = int64_t{2};
  tuned.values["algorithm.tile_k"] = int64_t{2};
  tuned.values["algorithm.pack_b_minimum_elements"] = int64_t{1};
  tuned.values["algorithm.skinny_m_limit"] = int64_t{1};
  tuned.values["parallel.fmas_per_work_unit"] = int64_t{1};
  tuned.values["parallel.minimum_tasks"] = int64_t{1};
  tuned.values["conversion.parallel_minimum_elements"] = int64_t{1};
  gemm_kernel.Configure(tuned);

  const Tensor a = Tensor::FromFloat("", {3, 2}, {1, 4, 2, 5, 3, 6});
  const Tensor b = Tensor::FromFloat("", {3, 2}, {7, 8, 9, 10, 11, 12});
  const Tensor y = gemm_kernel(a, b, nullptr, 1.0f, 0.0f, 1, 0);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 58.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 64.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 139.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 154.0f);

  EXPECT_EQ(gemm_kernel.tuning().pack_b_minimum_elements, 1);
  tuned.values["algorithm.tile_k"] = int64_t{0};
  EXPECT_THROW(gemm_kernel.Configure(tuned), std::invalid_argument);
  tuned.values["algorithm.tile_k"] = int64_t{2};
  tuned.key.library = "other_library";
  EXPECT_THROW(gemm_kernel.Configure(tuned), std::invalid_argument);
}

TEST(KernelClass, GemmParallelTilesPreserveSerialReductionBits) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm serial_kernel{ctx};
  Gemm parallel_kernel{ctx};
  const auto key = serial_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));
  const auto schema = core::runtime::GetKernelTuningRegistry().FindSchema(key);
  ASSERT_NE(schema, nullptr);

  core::runtime::KernelTuningParameters serial = schema->portable_defaults();
  serial.values["algorithm.tile_m"] = int64_t{1};
  serial.values["algorithm.tile_n"] = int64_t{2};
  serial.values["algorithm.tile_k"] = int64_t{2};
  serial.values["algorithm.pack_b_minimum_elements"] = int64_t{1};
  serial.values["algorithm.skinny_m_limit"] = int64_t{1};
  serial.values["parallel.fmas_per_work_unit"] = int64_t{1};
  serial.values["parallel.minimum_tasks"] = std::numeric_limits<int64_t>::max();
  serial_kernel.Configure(serial);

  core::runtime::KernelTuningParameters parallel = serial;
  parallel.values["parallel.minimum_tasks"] = int64_t{1};
  parallel_kernel.Configure(parallel);

  const Tensor a = Tensor::FromFloat(
      "", {4, 7},
      {1e20f, 1.0f,   -1e20f, 3.0f,   1e-10f,  -2.0f,    7.0f,      -1e20f,    5.0f, 1e20f,
       -3.0f, 2e-10f, 4.0f,   -9.0f,  1.0f,    -2.0f,    3.0f,      -4.0f,     5.0f, -6.0f,
       7.0f,  0.5f,   0.25f,  0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f});
  const Tensor b = Tensor::FromFloat("", {7, 4}, {1.0f, -1.0f, 0.5f,  2.0f, 2.0f,   0.5f,  -1.0f,
                                                  3.0f, -1.0f, 2.0f,  1.5f, -0.5f,  0.25f, -2.0f,
                                                  4.0f, 1.0f,  3.0f,  1.0f, -0.25f, 2.0f,  -2.0f,
                                                  3.0f, 0.75f, -1.0f, 1.5f, -0.5f,  2.0f,  0.25f});

  const Tensor expected = serial_kernel(a, b, nullptr, 1.0f, 0.0f, 0, 0);
  const Tensor actual = parallel_kernel(a, b, nullptr, 1.0f, 0.0f, 0, 0);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
}

// Exercises the calibration function ``Gemm::RegisterTuningSchemas()``
// registers for ``parallel.minimum_tasks``: it runs the bounded crossover
// search against the portable tile defaults and must publish a validated
// candidate for the FLOAT tuning key.
TEST(KernelClass, GemmCalibratesParallelMinimumTasksThreshold) {
  const KernelContext ctx{DefaultOpset(13)};
  Gemm gemm_kernel{ctx};
  const auto float_key = gemm_kernel.TuningKey(static_cast<int32_t>(DataType::FLOAT));

  core::runtime::KernelCalibrationSelection selection;
  selection.library = float_key.library;
  selection.kernels = {float_key.kernel};
  selection.element_types = {float_key.element_type};

  core::runtime::CalibrationOptions options;
  options.maximum_duration_ms = 1000;

  const core::runtime::CalibrationBatchReport report =
      core::runtime::CalibrateRegisteredKernels(selection, options);

  ASSERT_EQ(report.calibrated.size(), 1u);
  EXPECT_EQ(report.calibrated[0].key, float_key);
  EXPECT_TRUE(report.calibrated[0].Contains("parallel.minimum_tasks"));
  EXPECT_TRUE(report.unsupported.empty());

  const auto schema = core::runtime::GetKernelTuningRegistry().FindSchema(float_key);
  ASSERT_NE(schema, nullptr);
  EXPECT_NO_THROW(schema->Validate(report.calibrated[0]));
}

// Verifies that ``kernel::MatMul`` produces FLOAT16 / BFLOAT16 outputs that
// numerically match the FLOAT computation rounded through the same dtype.
TEST(KernelClass, MatMulHalfPrecisionMatchesFloatReference) {
  const KernelContext ctx{DefaultOpset(13)};
  MatMul matmul_kernel{ctx};
  const std::vector<float> a_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b_vals = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const Tensor a_f = Tensor::FromFloat("", {2, 3}, a_vals);
  const Tensor b_f = Tensor::FromFloat("", {3, 2}, b_vals);
  const Tensor ref = matmul_kernel(a_f, b_f);

  for (int32_t target : {static_cast<int32_t>(core::runtime::DataType::FLOAT16),
                         static_cast<int32_t>(core::runtime::DataType::BFLOAT16)}) {
    const Tensor a_h = MakeHalfTensor(target, {2, 3}, a_vals);
    const Tensor b_h = MakeHalfTensor(target, {3, 2}, b_vals);
    const Tensor y_h = matmul_kernel(a_h, b_h);
    ASSERT_EQ(y_h.data_type, target);
    ASSERT_EQ(y_h.shape, ref.shape);
    const std::vector<float> got = DecodeHalfTensor(y_h);
    const std::vector<float> exp = DecodeHalfTensor(
        onnx_kernels::DemoteFromFloat32(onnx_kernels::PromoteToFloat32(ref), target));
    ASSERT_EQ(got.size(), exp.size());
    const float tol =
        target == static_cast<int32_t>(core::runtime::DataType::FLOAT16) ? 1.0f : 4.0f;
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_NEAR(got[i], exp[i], tol) << "i=" << i;
    }
  }

  // In-place overload preserves the half-precision dtype.
  const Tensor a_h =
      MakeHalfTensor(static_cast<int32_t>(core::runtime::DataType::BFLOAT16), {2, 3}, a_vals);
  const Tensor b_h =
      MakeHalfTensor(static_cast<int32_t>(core::runtime::DataType::BFLOAT16), {3, 2}, b_vals);
  Tensor y_h("", core::runtime::DataType::BFLOAT16, {2, 2},
             std::vector<uint8_t>(4 * sizeof(uint16_t)));
  matmul_kernel(a_h, b_h, y_h);
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(core::runtime::DataType::BFLOAT16));
}

TEST(KernelClass, MelWeightMatrixUsesAllocatorWhenRuntimeContextHasOne) {
  const KernelContext ctx{DefaultOpset(17)};
  MelWeightMatrix kernel{ctx};

  const Tensor num_mel_bins = Tensor::FromInt64("", {}, {4});
  const Tensor dft_length = Tensor::FromInt64("", {}, {8});
  const Tensor sample_rate = Tensor::FromInt64("", {}, {8000});
  const Tensor lower_edge_hertz = Tensor::FromFloat("", {}, {0.0f});
  const Tensor upper_edge_hertz = Tensor::FromFloat("", {}, {4000.0f});

  // Two concurrent slots are needed: one for the output tensor and one for the
  // transient ``bin_indices`` scratch buffer routed through the allocator.
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = kernel(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz,
                    core::runtime::DataType::FLOAT, &rt);
  EXPECT_TRUE(y.has_allocation());
  // Only the output remains allocated; the scratch buffer was freed.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_EQ(y.data.size(), 0u);
  ASSERT_EQ(y.shape.size(), 2u);
  EXPECT_EQ(y.shape[0], 5); // floor(8/2) + 1
  EXPECT_EQ(y.shape[1], 4); // num_mel_bins
}

} // namespace Test
