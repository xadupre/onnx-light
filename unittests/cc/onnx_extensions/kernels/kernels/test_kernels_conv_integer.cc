// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::kernel::ConvInteger;

namespace Test {

TEST(KernelClass, ConvIntegerCanRunInPlaceIsFalse) { EXPECT_FALSE(ConvInteger::CanRunInPlace()); }

} // namespace Test
