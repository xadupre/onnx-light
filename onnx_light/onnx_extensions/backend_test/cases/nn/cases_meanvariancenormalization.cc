// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterMeanVarianceNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {

  if (mode == TestMode::BENCHMARK) {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");

    constexpr int64_t count = 32 * 64 * 64 * 1 * 16;
    Expect(registry, std::move(node), "test_cc_mvn_benchmark", {opset}, {count}, {count},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext mvn_kernel_ctx{opset};
             const onnx_kernels::kernel::MeanVarianceNormalization mvn_kernel{mvn_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {32, 64, 64, 1, 16}, 2401);
             Tensor y = mvn_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // ``mvn``: default axes [0,2,3].
  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_mvn", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mvn_kernel_ctx{opset};
      const onnx_kernels::kernel::MeanVarianceNormalization mvn_kernel{mvn_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                   {-1.0f, 0.0f,  0.5f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                    7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                    43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f});

      Tensor y = mvn_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // ``mvn_explicit_axes``: explicit axes [0,2,3].
  {
    const OpsetId opset = DefaultOpset(13);

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "axes", {0, 2, 3});
    Expect(registry, std::move(node), "test_cc_mvn_explicit_axes", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mvn_kernel_ctx{opset};
      const onnx_kernels::kernel::MeanVarianceNormalization mvn_kernel{mvn_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                   {-2.0f, -1.0f, 0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                    7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f,
                                    43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f});

      Tensor y = mvn_kernel(x, {0, 2, 3});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // ``mvn_explicit_axes_ver18``: same explicit axes with opset 18 import.
  {
    const OpsetId opset = DefaultOpset(18);

    NodeProto node;
    node.set_op_type("MeanVarianceNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "axes", {0, 2, 3});
    Expect(registry, std::move(node), "test_cc_mvn_explicit_axes_ver18", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(18);

      const KernelContext mvn_kernel_ctx{opset};
      const onnx_kernels::kernel::MeanVarianceNormalization mvn_kernel{mvn_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {3, 3, 3, 1, 2},
                                   {-3.0f, -2.0f, -1.0f, 0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                    6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                                    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                                    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                                    33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                                    42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f});

      Tensor y = mvn_kernel(x, {0, 2, 3});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
