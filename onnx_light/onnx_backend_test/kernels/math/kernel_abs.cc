// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include <cmath>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

template <typename T, int opset=0> Tensor Abs(const Tensor&);

template <float, 0> Tensor Abs(const Tensor &t) {

}

} // kernel namespace
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
