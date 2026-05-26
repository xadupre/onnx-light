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
// supports any number of spatial dimensions; the cases below mirror the
// ``test_averagepool_*`` reference cases in the ONNX test suite that do not
// rely on attributes not yet supported by this kernel (``dilations`` and
// ``auto_pad``).
//
// Cases registered (each is the C++ analogue of the like-named ONNX
// reference case, with the ``test_cc_`` prefix):
//
//   * ``test_cc_averagepool_1d_default`` — 1-D, 2-wide kernel.
//   * ``test_cc_averagepool_2d_default`` — 2x2 kernel, default strides (1),
//     no padding.
//   * ``test_cc_averagepool_2d_strides`` — 3x3 kernel, strides ``(2, 2)``,
//     no padding.
//   * ``test_cc_averagepool_2d_ceil`` — 3x3 kernel, strides ``(2, 2)``,
//     ``ceil_mode = 1``.
//   * ``test_cc_averagepool_2d_ceil_last_window_starts_on_pad`` — 3x3
//     kernel, strides ``(3, 3)``, pads ``(1, 1, 1, 1)``, ``ceil_mode = 1``
//     and ``count_include_pad = 1`` (the last window starts on a padded
//     position).
//   * ``test_cc_averagepool_2d_pads`` — 3x3 kernel, pads ``(2, 2, 2, 2)``,
//     default ``count_include_pad = 0``.
//   * ``test_cc_averagepool_2d_pads_count_include_pad`` — 3x3 kernel,
//     padding ``(1, 1, 1, 1)`` with ``count_include_pad = 1``.
//   * ``test_cc_averagepool_2d_precomputed_pads`` — 5x5 kernel,
//     pads ``(2, 2, 2, 2)``.
//   * ``test_cc_averagepool_2d_precomputed_pads_count_include_pad`` — 5x5
//     kernel, pads ``(2, 2, 2, 2)`` with ``count_include_pad = 1``.
//   * ``test_cc_averagepool_2d_precomputed_strides`` — 2x2 kernel, strides
//     ``(2, 2)``.
//   * ``test_cc_averagepool_3d_default`` — 3-D, 2x2x2 kernel.
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

  // 1-D AveragePool with a 2-wide kernel (mirrors
  // ``test_averagepool_1d_default``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {2});

    Tensor x = Tensor::FromFloat("", {1, 1, 8}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2});

    Expect(node, {x}, {y}, "test_cc_averagepool_1d_default", {opset}, "backend-test", registry);
  }

  // 3x3 kernel, strides (2, 2) with ``ceil_mode = 1`` (mirrors
  // ``test_averagepool_2d_ceil``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {3, 3});
    AddIntsAttribute(node, "strides", {2, 2});
    AddIntAttribute(node, "ceil_mode", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2},
                                   /*pads=*/{}, /*ceil_mode=*/true,
                                   /*count_include_pad=*/false);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_ceil", {opset}, "backend-test", registry);
  }

  // 3x3 kernel, strides (3, 3), pads (1, 1, 1, 1), ``ceil_mode = 1`` and
  // ``count_include_pad = 1`` so the last window starts on a padded
  // position (mirrors ``test_averagepool_2d_ceil_last_window_starts_on_pad``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {3, 3});
    AddIntsAttribute(node, "strides", {3, 3});
    AddIntsAttribute(node, "pads", {1, 1, 1, 1});
    AddIntAttribute(node, "ceil_mode", 1);
    AddIntAttribute(node, "count_include_pad", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{3, 3},
                                   /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/true,
                                   /*count_include_pad=*/true);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_ceil_last_window_starts_on_pad", {opset},
           "backend-test", registry);
  }

  // 3x3 kernel with pads (2, 2, 2, 2), default ``count_include_pad = 0``
  // (mirrors ``test_averagepool_2d_pads``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {3, 3});
    AddIntsAttribute(node, "pads", {2, 2, 2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                                   /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                   /*count_include_pad=*/false);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_pads", {opset}, "backend-test", registry);
  }

  // 5x5 kernel with pads (2, 2, 2, 2) on a 5x5 input (mirrors
  // ``test_averagepool_2d_precomputed_pads``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {5, 5});
    AddIntsAttribute(node, "pads", {2, 2, 2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                                   /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                   /*count_include_pad=*/false);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_precomputed_pads", {opset}, "backend-test",
           registry);
  }

  // 5x5 kernel with pads (2, 2, 2, 2) and ``count_include_pad = 1`` on a
  // 5x5 input (mirrors ``test_averagepool_2d_precomputed_pads_count_include_pad``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {5, 5});
    AddIntsAttribute(node, "pads", {2, 2, 2, 2});
    AddIntAttribute(node, "count_include_pad", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                                   /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                   /*count_include_pad=*/true);

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_precomputed_pads_count_include_pad", {opset},
           "backend-test", registry);
  }

  // 2x2 kernel with strides (2, 2) on a 5x5 input (mirrors
  // ``test_averagepool_2d_precomputed_strides``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {2, 2});
    AddIntsAttribute(node, "strides", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_precomputed_strides", {opset}, "backend-test",
           registry);
  }

  // 3-D AveragePool with a 2x2x2 kernel (mirrors
  // ``test_averagepool_3d_default``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddIntsAttribute(node, "kernel_shape", {2, 2, 2});

    std::vector<float> data(1 * 1 * 3 * 3 * 3);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 1, 3, 3, 3}, data);
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_3d_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
