// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"
#include "onnx_backend_test/kernels/preview/include_preview_kernels.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::kernel::Abs;
using onnx_backend_test::kernel::Add;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::AveragePool;
using onnx_backend_test::kernel::BlackmanWindow;
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::FlexAttention;
using onnx_backend_test::kernel::If;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::LabelEncoder;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::QuantizeLinear;
using onnx_backend_test::kernel::ReduceSum;
using onnx_backend_test::kernel::SequenceConstruct;
using onnx_backend_test::kernel::StringConcat;
using onnx_backend_test::kernel::Xor;
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
  EXPECT_TRUE(Or::CanRunInPlace());
  EXPECT_TRUE(Xor::CanRunInPlace());

  // If just copies the selected branch into the output.
  EXPECT_TRUE(If::CanRunInPlace());

  // Optional is a passthrough of its input.
  EXPECT_TRUE(OptionalKernel::CanRunInPlace());

  // Output buffer fundamentally cannot equal any input buffer for these.
  EXPECT_FALSE(AveragePool::CanRunInPlace());
  EXPECT_FALSE(BlackmanWindow::CanRunInPlace());
  EXPECT_FALSE(Concat::CanRunInPlace());
  EXPECT_FALSE(FlexAttention::CanRunInPlace());
  EXPECT_FALSE(LabelEncoder::CanRunInPlace());
  EXPECT_FALSE(QuantizeLinear::CanRunInPlace());
  EXPECT_FALSE(ReduceSum::CanRunInPlace());
  // SequenceConstruct stacks inputs along a new outer axis, so its output
  // layout cannot share storage with any single input buffer.
  EXPECT_FALSE(SequenceConstruct::CanRunInPlace());
  // StringConcat writes a freshly-built string whose bytes depend on both
  // inputs, so output cannot alias either input buffer.
  EXPECT_FALSE(StringConcat::CanRunInPlace());
}

} // namespace Test
