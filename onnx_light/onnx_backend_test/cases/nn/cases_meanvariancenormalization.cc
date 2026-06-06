// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterMeanVarianceNormalizationCases(std::vector<TestCase> &registry) {
  // ``mvn``: default axes [0,2,3].
  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::MeanVarianceNormalization mvn_kernel{ctx};

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                 {-1.0f, 0.0f,  0.5f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                  16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                  25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                                  34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                  43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f});

    Tensor y = mvn_kernel(x);

    Expect(node, {x}, {y}, "test_cc_mvn", {opset}, "backend-test", registry);
  }

  // ``mvn_expanded``: explicit axes [0,2,3].
  {
    const OpsetId opset = DefaultOpset(13);
    const kernel::KernelContext ctx{opset};
    const kernel::MeanVarianceNormalization mvn_kernel{ctx};

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "axes", {0, 2, 3});

    Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                 {-2.0f, -1.0f, 0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                  16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                  25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                                  34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                  43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f});

    Tensor y = mvn_kernel(x, {0, 2, 3});

    Expect(node, {x}, {y}, "test_cc_mvn_expanded", {opset}, "backend-test", registry);
  }

  // ``mvn_expanded_ver18``: same explicit axes with opset 18 import.
  {
    const OpsetId opset = DefaultOpset(18);
    const kernel::KernelContext ctx{opset};
    const kernel::MeanVarianceNormalization mvn_kernel{ctx};

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "axes", {0, 2, 3});

    Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                 {-3.0f, -2.0f, -1.0f, 0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                                  15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                                  24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                                  33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                                  42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f});

    Tensor y = mvn_kernel(x, {0, 2, 3});

    Expect(node, {x}, {y}, "test_cc_mvn_expanded_ver18", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
