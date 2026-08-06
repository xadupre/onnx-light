// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// MaxUnpool — y = max-unpool(X, I[, output_shape], kernel_shape[, strides,
// pads]) (since opset 9 in the ai.onnx domain). The cases below mirror the
// ``test_maxunpool_*`` reference cases in the ONNX test suite.
//
// Cases registered:
//
//   * ``test_cc_maxunpool_export_with_output_shape`` — three-input form,
//     ``output_shape`` provided.
//   * ``test_cc_maxunpool_export_without_output_shape`` — two-input form.
// ---------------------------------------------------------------------------
void RegisterMaxUnpoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::MaxUnpool maxunpool_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    // Two-input form: kernel 2x2, stride 2 unpools a [1, 1, 512, 512] map into
    // a [1, 1, 1024, 1024] output. Each pooled element is placed at the
    // top-left corner of its 2x2 window.
    const int64_t pooled = 512;
    const int64_t out = pooled * 2;
    NodeProto node;
    node.set_op_type("MaxUnpool");
    node.add_input("xT");
    node.add_input("xI");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_maxunpool_benchmark", {opset},
           {pooled * pooled, pooled * pooled}, {out * out},
           [maxunpool_kernel, pooled, out]() -> IoData {
             Tensor x = Tensor::FromFloat("", {1, 1, pooled, pooled},
                                          Randn<float>({pooled, pooled}, 2001));
             std::vector<int64_t> idx(static_cast<size_t>(pooled * pooled));
             for (int64_t i = 0; i < pooled; ++i) {
               for (int64_t j = 0; j < pooled; ++j) {
                 idx[static_cast<size_t>(i * pooled + j)] = (2 * i) * out + (2 * j);
               }
             }
             Tensor indices = Tensor::FromInt64("", {1, 1, pooled, pooled}, idx);
             Tensor y = maxunpool_kernel(x, indices, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});
             return IoData{{std::move(x), std::move(indices)}, {std::move(y)}};
           });
    return;
  }

  // Three-input form: ``output_shape = [1, 1, 5, 5]`` overrides the
  // inferred ``[1, 1, 4, 4]``; the inferred region is placed at the
  // top-left corner of the larger output.
  {
    NodeProto node;
    node.set_op_type("MaxUnpool");
    node.add_input("xT");
    node.add_input("xI");
    node.add_input("output_shape");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_maxunpool_export_with_output_shape", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
             Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
             Tensor output_shape = Tensor::FromInt64("", {4}, {1, 1, 5, 5});
             Tensor y = maxunpool_kernel(x, indices, output_shape, /*kernel_shape=*/{2, 2},
                                         /*strides=*/{2, 2});

             return IoData{{std::move(x), std::move(indices), std::move(output_shape)},
                           {std::move(y)}};
           });
  }

  // Two-input form: the output shape is fully determined by ``kernel_shape``
  // and ``strides`` to ``[1, 1, 4, 4]``.
  {
    NodeProto node;
    node.set_op_type("MaxUnpool");
    node.add_input("xT");
    node.add_input("xI");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_maxunpool_export_without_output_shape", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
             Tensor y = maxunpool_kernel(x, indices, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});

             return IoData{{std::move(x), std::move(indices)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
