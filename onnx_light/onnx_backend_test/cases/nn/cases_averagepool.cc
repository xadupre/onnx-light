// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Helper that appends a single INTS attribute (``name`` -> ``values``) to
// ``node``. Used to encode AveragePool's ``kernel_shape``, ``strides`` and
// ``pads`` attributes.
void AddIntsAttribute(NodeProto &node, const char *name, const std::vector<int64_t> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : values) {
    attr->ints().push_back(v);
  }
}

// Helper that appends a single INT attribute (``name`` -> ``value``) to
// ``node``. Used to encode AveragePool's ``ceil_mode`` and
// ``count_include_pad`` attributes.
void AddIntAttribute(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

} // namespace

// ---------------------------------------------------------------------------
// AveragePool — y = avg-pool(x, kernel_shape[, strides, pads, ceil_mode,
// count_include_pad]) (since opset 19 in the ai.onnx domain; the subset of
// attributes exercised here has been stable since opset 11). The kernel
// supports any number of spatial dimensions; the cases below exercise the
// 2-D NCHW shape that mirrors the ``test_averagepool_2d_*`` reference cases
// in the ONNX test suite.
//
// Three cases are registered:
//
//   * ``test_cc_averagepool_2d_default`` — 2x2 kernel, default strides (1),
//     no padding, ``ceil_mode = 0``, ``count_include_pad = 0``.
//   * ``test_cc_averagepool_2d_strides`` — 3x3 kernel, strides ``(2, 2)``,
//     no padding.
//   * ``test_cc_averagepool_2d_pads_count_include_pad`` — 3x3 kernel,
//     strides ``(1, 1)``, padding ``(1, 1, 1, 1)`` with
//     ``count_include_pad = 1`` so padded zeros are counted in the divisor.
// ---------------------------------------------------------------------------
void RegisterAveragePoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(19);
  const kernel::AveragePool average_pool_kernel{kernel::KernelContext(opset)};

  // Default 2x2 kernel on a 1x1x4x4 input.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_default", {opset}, "backend-test", registry);
  }

  // 3x3 kernel with strides (2, 2) on a 1x1x5x5 input.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {3, 3});
    AddIntsAttribute(node, "strides", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_strides", {opset}, "backend-test", registry);
  }

  // 3x3 kernel with explicit padding (1, 1, 1, 1) and
  // ``count_include_pad = 1`` so padded zeros contribute to the divisor.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {3, 3});
    AddIntsAttribute(node, "pads", {1, 1, 1, 1});
    AddIntAttribute(node, "count_include_pad", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                                   /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/false,
                                   /*count_include_pad=*/true);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_pads_count_include_pad", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
