// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/shapes/shapes/math/shape_math.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::DefaultOpset;
using core::runtime::Shape;
using core::runtime::Tensor;
using core::runtime::ValueStore;
using core::symbolic::SymDim;
using core::symbolic::SymShape;
using core::symbolic::SymTensor;
using core::symbolic::TensorType;
using onnx_kernels::kernel::Gemm;
using onnx_kernels::kernel::KernelContext;

TEST(Gemm29, TransposesFinalAxesWithPreparedWeightsAndPreallocatedOutput) {
  const Gemm kernel{KernelContext{DefaultOpset(29)}};
  for (const bool transA : {false, true}) {
    for (const bool transB : {false, true}) {
      Tensor a = transA ? Tensor::FromFloat("", {2, 2, 3}, {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12})
                        : Tensor::FromFloat("", {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
      Tensor b = Tensor::FromFloat(
          "", {2, 2}, transB ? std::vector<float>{2, 1, 3, 4} : std::vector<float>{2, 3, 1, 4});
      for (const Shape &bias_shape :
           {Shape{}, Shape{2}, Shape{3, 1}, Shape{2, 1, 2}, Shape{2, 3, 2}}) {
        std::vector<float> bias_values(bias_shape.product());
        for (size_t i = 0; i < bias_values.size(); ++i) {
          bias_values[i] = 2.0f * static_cast<float>(i);
        }
        Tensor c = Tensor::FromFloat("", bias_shape, bias_values);
        std::vector<float> expected;
        for (int64_t batch = 0; batch < 2; ++batch) {
          for (int64_t row = 0; row < 3; ++row) {
            for (int64_t col = 0; col < 2; ++col) {
              const float first = static_cast<float>((batch * 3 + row) * 2 + 1);
              const float second = first + 1.0f;
              size_t bias_index = 0;
              if (bias_shape == Shape{2})
                bias_index = col;
              else if (bias_shape == Shape{3, 1})
                bias_index = row;
              else if (bias_shape == Shape{2, 1, 2})
                bias_index = batch * 2 + col;
              else if (bias_shape == Shape{2, 3, 2})
                bias_index = (batch * 3 + row) * 2 + col;
              expected.push_back(0.25f * (col == 0 ? first * 2 + second : first * 3 + second * 4) +
                                 0.5f * bias_values[bias_index]);
            }
          }
        }
        const Tensor actual = kernel(a, b, &c, 0.25f, 0.5f, transA, transB);
        EXPECT_EQ(actual.shape, Shape({2, 3, 2}));
        EXPECT_EQ(std::vector<float>(actual.As<float>(), actual.As<float>() + expected.size()),
                  expected);
        ValueStore store;
        const auto prepared = kernel.PrepareConstantB(b, transB, store);
        const Tensor packed_actual = kernel(a, prepared, &c, 0.25f, 0.5f, transA);
        EXPECT_EQ(packed_actual.shape, actual.shape);
        EXPECT_EQ(std::vector<float>(packed_actual.As<float>(),
                                     packed_actual.As<float>() + expected.size()),
                  expected);
        Tensor output = Tensor::FromFloat("", {2, 3, 2}, std::vector<float>(12));
        kernel(a, b, &c, 0.25f, 0.5f, transA, transB, output);
        EXPECT_EQ(std::vector<float>(output.As<float>(), output.As<float>() + expected.size()),
                  expected);
      }
    }
  }
}

TEST(Gemm29, VectorTransposeAndZeroBeta) {
  const Gemm kernel{KernelContext{DefaultOpset(29)}};
  Tensor a = Tensor::FromFloat("", {2}, {1, 2});
  Tensor b = Tensor::FromFloat("", {2, 2}, {2, 3, 1, 4});
  Tensor c = Tensor::FromFloat(
      "", {2}, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()});
  for (const bool transA : {false, true}) {
    const Tensor y = kernel(a, b, &c, 1.0f, 0.0f, transA, 0);
    EXPECT_EQ(y.shape, Shape({2}));
    EXPECT_EQ(y.As<float>()[0], 4.0f);
    EXPECT_EQ(y.As<float>()[1], 11.0f);
  }
  Tensor bad_bias = Tensor::FromFloat("", {1, 2}, {0, 0});
  EXPECT_THROW(kernel(a, b, &bad_bias, 1, 0, 0, 0), std::invalid_argument);
  ValueStore store;
  const auto prepared = kernel.PrepareConstantB(b, 0, store);
  EXPECT_THROW(kernel(a, prepared, &bad_bias, 1, 0, 0), std::invalid_argument);
  const Gemm old_kernel{KernelContext{DefaultOpset(28)}};
  EXPECT_THROW(old_kernel(a, b, nullptr, 1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(old_kernel(a, prepared, nullptr, 1, 0, 0), std::invalid_argument);
}

TEST(Gemm29, RejectsInvalidRanksDimensionsAndBuffers) {
  const Gemm kernel{KernelContext{DefaultOpset(29)}};
  Tensor a = Tensor::FromFloat("", {2, 3, 2}, std::vector<float>(12, 1));
  Tensor b = Tensor::FromFloat("", {2, 2}, {2, 3, 1, 4});
  Tensor scalar = Tensor::FromFloat("", {}, {1});
  Tensor batched_b = Tensor::FromFloat("", {1, 2, 2}, {2, 3, 1, 4});
  Tensor bad_k = Tensor::FromFloat("", {3, 2}, std::vector<float>(6));
  EXPECT_THROW(kernel(scalar, b, nullptr, 1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(kernel(a, batched_b, nullptr, 1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(kernel(a, bad_k, nullptr, 1, 0, 0, 0), std::invalid_argument);
  Tensor short_output = Tensor::FromFloat("", {1}, {0});
  short_output.shape = {2, 3, 2};
  EXPECT_THROW(kernel(a, b, nullptr, 1, 0, 0, 0, short_output), std::invalid_argument);
  Tensor short_input = Tensor::FromFloat("", {1}, {1});
  short_input.shape = {2, 3, 2};
  EXPECT_THROW(kernel(short_input, b, nullptr, 1, 0, 0, 0), std::invalid_argument);
  Tensor overflowing_input;
  overflowing_input.data_type = TensorProto::FLOAT;
  overflowing_input.shape = {std::numeric_limits<int64_t>::max(), 2, 2};
  EXPECT_THROW(kernel(overflowing_input, b, nullptr, 1, 0, 0, 0), std::invalid_argument);
}

TEST(Gemm29, EmptyContractionAndOutput) {
  const Gemm kernel{KernelContext{DefaultOpset(29)}};
  Tensor a = Tensor::FromFloat("", {2, 3, 0}, {});
  Tensor b = Tensor::FromFloat("", {0, 2}, {});
  Tensor c = Tensor::FromFloat("", {2, 1, 2}, {1, 2, 3, 4});
  ValueStore store;
  const auto prepared = kernel.PrepareConstantB(b, 0, store);
  for (const bool packed : {false, true}) {
    const Tensor y = packed ? kernel(a, prepared, &c, 1, 1, 0) : kernel(a, b, &c, 1, 1, 0, 0);
    EXPECT_EQ(y.shape, Shape({2, 3, 2}));
    EXPECT_EQ(std::vector<float>(y.As<float>(), y.As<float>() + 12),
              (std::vector<float>{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}));
  }
  a = Tensor::FromFloat("", {2, 3, 2}, std::vector<float>(12, 1));
  b = Tensor::FromFloat("", {2, 0}, {});
  const Tensor empty = kernel(a, b, nullptr, 1, 0, 0, 0);
  EXPECT_EQ(empty.shape, Shape({2, 3, 0}));
  EXPECT_EQ(empty.size_bytes(), 0u);
}

TEST(Gemm29, NativeShapeInferencePreservesLeadingDimensionsAndValidatesBias) {
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 29);
  ctx.Set("A",
          SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim("batch"), SymDim(5), SymDim(3)}));
  ctx.Set("B", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(7), SymDim(5)}));
  ctx.Set("C", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(7)}));
  NodeProto node = MakeNode("Gemm", {"A", "B", "C"}, {"Y"});
  AddAttribute<int64_t>(node, "transA", 1);
  AddAttribute<int64_t>(node, "transB", 1);
  onnx_shapes::shapes::math::ComputeShapeGemm(ctx, node, "A", "B");
  EXPECT_EQ(ctx.Get("Y").Shape(), SymShape({SymDim("batch"), SymDim(3), SymDim(7)}));
  ctx.Set("C", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(2), SymDim(7)}));
  AddAttribute<float>(node, "beta", 0);
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeGemm(ctx, node, "A", "B"),
               std::invalid_argument);
  ctx.SetOpsetVersion("", 28);
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeGemm(ctx, node, "A", "B"),
               std::invalid_argument);
}
