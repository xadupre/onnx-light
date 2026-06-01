// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"
#include "onnx_backend_test/kernels/preview/include_preview_kernels.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/kernels/training/include_training_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::kernel::Abs;
using onnx_backend_test::kernel::Adam;
using onnx_backend_test::kernel::Add;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::ArrayFeatureExtractor;
using onnx_backend_test::kernel::AveragePool;
using onnx_backend_test::kernel::BatchNormalization;
using onnx_backend_test::kernel::Binarizer;
using onnx_backend_test::kernel::BlackmanWindow;
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::FlexAttention;
using onnx_backend_test::kernel::If;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::LabelEncoder;
using onnx_backend_test::kernel::LinearClassifier;
using onnx_backend_test::kernel::LinearRegressor;
using onnx_backend_test::kernel::Normalizer;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::QuantizeLinear;
using onnx_backend_test::kernel::ReduceSum;
using onnx_backend_test::kernel::RoiAlign;
using onnx_backend_test::kernel::Scaler;
using onnx_backend_test::kernel::SequenceConstruct;
using onnx_backend_test::kernel::StringConcat;
using onnx_backend_test::kernel::StringSplit;
using onnx_backend_test::kernel::SVMClassifier;
using onnx_backend_test::kernel::SVMRegressor;
using onnx_backend_test::kernel::Xor;
using onnx_backend_test::kernel::ZipMap;
using OptionalKernel = onnx_backend_test::kernel::Optional;

namespace Test {

// ---------------------------------------------------------------------------
// Framework-level tests for the backend kernel harness itself. Tests that
// exercise individual kernels live in kernels/<subfolder>/*.cc, mirroring the
// layout of onnx_backend_test/kernels/.
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, KernelContextStoresOpset) {
  KernelContext ctx(DefaultOpset(13));
  EXPECT_EQ(ctx.opset.domain, std::string());
  EXPECT_EQ(ctx.opset.version, 13);
}

TEST(BackendKernelClass, CanRunInPlaceReportsKernelCapability) {
  // Element-wise unary/binary kernels can write their output into an input
  // buffer (shape and dtype match by construction or when no broadcasting
  // expansion is needed for that input).
  EXPECT_TRUE(Abs::CanRunInPlace());
  EXPECT_TRUE(Add::CanRunInPlace());
  EXPECT_TRUE(And::CanRunInPlace());
  // BatchNormalization in inference mode produces ``Y`` with the same shape
  // and dtype as ``X`` and only reads the per-channel parameters once.
  EXPECT_TRUE(BatchNormalization::CanRunInPlace());
  EXPECT_TRUE(Binarizer::CanRunInPlace());
  EXPECT_TRUE(Or::CanRunInPlace());
  EXPECT_TRUE(Xor::CanRunInPlace());

  // If just copies the selected branch into the output.
  EXPECT_TRUE(If::CanRunInPlace());

  // Optional is a passthrough of its input.
  EXPECT_TRUE(OptionalKernel::CanRunInPlace());

  // Output buffer fundamentally cannot equal any input buffer for these.
  EXPECT_FALSE(Adam::CanRunInPlace());
  EXPECT_FALSE(ArrayFeatureExtractor::CanRunInPlace());
  EXPECT_FALSE(AveragePool::CanRunInPlace());
  EXPECT_FALSE(BlackmanWindow::CanRunInPlace());
  EXPECT_FALSE(Concat::CanRunInPlace());
  EXPECT_FALSE(FlexAttention::CanRunInPlace());
  EXPECT_FALSE(LabelEncoder::CanRunInPlace());
  EXPECT_FALSE(LinearClassifier::CanRunInPlace());
  EXPECT_FALSE(LinearRegressor::CanRunInPlace());
  // Normalizer always produces a float output, but the input may be
  // int32/int64, so the output buffer cannot share storage with the input
  // in the general case (different dtypes/byte widths).
  EXPECT_FALSE(Normalizer::CanRunInPlace());
  EXPECT_FALSE(QuantizeLinear::CanRunInPlace());
  EXPECT_FALSE(ReduceSum::CanRunInPlace());
  EXPECT_FALSE(RoiAlign::CanRunInPlace());
  // SequenceConstruct stacks inputs along a new outer axis, so its output
  // layout cannot share storage with any single input buffer.
  EXPECT_FALSE(SequenceConstruct::CanRunInPlace());
  EXPECT_FALSE(SVMClassifier::CanRunInPlace());
  EXPECT_FALSE(SVMRegressor::CanRunInPlace());
  // Scaler always produces a float output, but the input may be int32/int64,
  // so the output buffer cannot share storage with the input in the general
  // case (different dtypes/byte widths).
  EXPECT_FALSE(Scaler::CanRunInPlace());
  // StringConcat writes a freshly-built string whose bytes depend on both
  // inputs, so output cannot alias either input buffer.
  EXPECT_FALSE(StringConcat::CanRunInPlace());
  EXPECT_FALSE(StringSplit::CanRunInPlace());
  EXPECT_FALSE(ZipMap::CanRunInPlace());
}

} // namespace Test
