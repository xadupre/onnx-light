// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_kernels/kernels/optional/include_optional_kernels.h"
#include "onnx_kernels/kernels/preview/include_preview_kernels.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_kernels/test_case.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DefaultOpset;
using onnx_kernels::kernel::Abs;
using onnx_kernels::kernel::Adam;
using onnx_kernels::kernel::Add;
using onnx_kernels::kernel::And;
using onnx_kernels::kernel::ArrayFeatureExtractor;
using onnx_kernels::kernel::AveragePool;
using onnx_kernels::kernel::BatchNormalization;
using onnx_kernels::kernel::Binarizer;
using onnx_kernels::kernel::BlackmanWindow;
using onnx_kernels::kernel::Concat;
using onnx_kernels::kernel::Dropout;
using onnx_kernels::kernel::FlexAttention;
using onnx_kernels::kernel::HammingWindow;
using onnx_kernels::kernel::HannWindow;
using onnx_kernels::kernel::If;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::LabelEncoder;
using onnx_kernels::kernel::LinearClassifier;
using onnx_kernels::kernel::LinearRegressor;
using onnx_kernels::kernel::Normalizer;
using onnx_kernels::kernel::Or;
using onnx_kernels::kernel::QuantizeLinear;
using onnx_kernels::kernel::ReduceSum;
using onnx_kernels::kernel::RoiAlign;
using onnx_kernels::kernel::Scaler;
using onnx_kernels::kernel::SequenceConstruct;
using onnx_kernels::kernel::StringConcat;
using onnx_kernels::kernel::StringSplit;
using onnx_kernels::kernel::SVMClassifier;
using onnx_kernels::kernel::SVMRegressor;
using onnx_kernels::kernel::Xor;
using onnx_kernels::kernel::ZipMap;
using OptionalKernel = onnx_kernels::kernel::Optional;

namespace Test {

// ---------------------------------------------------------------------------
// Framework-level tests for the backend kernel harness itself. Tests that
// exercise individual kernels live in kernels/<subfolder>/*.cc, mirroring the
// layout of onnx_kernels/kernels/.
// ---------------------------------------------------------------------------

TEST(KernelClass, KernelContextStoresOpset) {
  KernelContext ctx(DefaultOpset(13));
  EXPECT_EQ(ctx.opset.domain, std::string());
  EXPECT_EQ(ctx.opset.version, 13);
}

TEST(KernelClass, CanRunInPlaceReportsKernelCapability) {
  // Element-wise unary/binary kernels can write their output into an input
  // buffer (shape and dtype match by construction or when no broadcasting
  // expansion is needed for that input).
  EXPECT_TRUE(Abs::CanRunInPlace());
  EXPECT_TRUE(Add::CanRunInPlace());
  EXPECT_TRUE(And::CanRunInPlace());
  // BatchNormalization in inference mode produces ``Y`` with the same shape
  // and dtype as ``X`` and only reads the per-channel parameters once.
  EXPECT_TRUE(BatchNormalization::CanRunInPlace());
  EXPECT_TRUE(Dropout::CanRunInPlace());
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
  EXPECT_FALSE(HannWindow::CanRunInPlace());
  EXPECT_FALSE(HammingWindow::CanRunInPlace());
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
