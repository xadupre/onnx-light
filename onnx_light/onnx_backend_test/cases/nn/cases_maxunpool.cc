// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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
void RegisterMaxUnpoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::MaxUnpool maxunpool_kernel{ctx};

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

    Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
    Tensor output_shape = Tensor::FromInt64("", {4}, {1, 1, 5, 5});
    Tensor y = maxunpool_kernel(x, indices, output_shape, /*kernel_shape=*/{2, 2},
                                /*strides=*/{2, 2});

    Expect(node, {x, indices, output_shape}, {y}, "test_cc_maxunpool_export_with_output_shape",
           {opset}, "backend-test", registry);
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

    Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
    Tensor y = maxunpool_kernel(x, indices, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});

    Expect(node, {x, indices}, {y}, "test_cc_maxunpool_export_without_output_shape", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
