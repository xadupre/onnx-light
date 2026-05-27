// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// AveragePool — y = avg-pool(x, kernel_shape[, strides, pads, ceil_mode,
// count_include_pad, dilations, auto_pad]) (since opset 19 in the ai.onnx
// domain; the subset of attributes exercised here has been stable since
// opset 11, with ``dilations`` added in opset 19). The kernel supports any
// number of spatial dimensions; the cases below mirror the
// ``test_averagepool_*`` reference cases in the ONNX test suite.
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
//   * ``test_cc_averagepool_2d_precomputed_same_upper`` — 3x3 kernel,
//     strides ``(2, 2)``, ``auto_pad = SAME_UPPER``.
//   * ``test_cc_averagepool_2d_precomputed_strides`` — 2x2 kernel, strides
//     ``(2, 2)``.
//   * ``test_cc_averagepool_2d_same_upper`` — 2x2 kernel,
//     ``auto_pad = SAME_UPPER``.
//   * ``test_cc_averagepool_2d_same_lower`` — 2x2 kernel,
//     ``auto_pad = SAME_LOWER``.
//   * ``test_cc_averagepool_2d_dilations`` — 2x2 kernel, dilations
//     ``(2, 2)``, ``ceil_mode = 1``.
//   * ``test_cc_averagepool_3d_default`` — 3-D, 2x2x2 kernel.
//   * ``test_cc_averagepool_3d_dilations_small`` — 3-D, 2x2x2 kernel,
//     dilations ``(2, 2, 2)``, ``ceil_mode = 1``.
//   * ``test_cc_averagepool_3d_dilations_large_count_include_pad_is_{0,1}_ceil_mode_is_{True,False}``
//     — 3-D, 5x5x5 kernel, strides ``(3, 3, 3)``, dilations ``(2, 2, 2)`` on
//     a deterministic random 1x1x32x32x32 input (mirrors the four
//     ``test_averagepool_3d_dilations_large_*`` reference cases; inputs are
//     drawn from :cpp:func:`Randn` with a fixed seed instead of
//     ``np.random.randn`` so the values are reproducible).
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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<int64_t>(node, "count_include_pad", 1);

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<int64_t>(node, "ceil_mode", 1);
    AddAttribute<int64_t>(node, "count_include_pad", 1);

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});
    AddAttribute<int64_t>(node, "count_include_pad", 1);

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});

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
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});

    std::vector<float> data(1 * 1 * 3 * 3 * 3);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 1, 3, 3, 3}, data);
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_3d_default", {opset}, "backend-test", registry);
  }

  // 3x3 kernel with strides (2, 2) and ``auto_pad = SAME_UPPER`` on a
  // 1x1x5x5 input (mirrors ``test_averagepool_2d_precomputed_same_upper``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", std::string("SAME_UPPER"));

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                  19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                                   /*ceil_mode=*/false, /*count_include_pad=*/false,
                                   /*dilations=*/{}, /*auto_pad=*/"SAME_UPPER");

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_precomputed_same_upper", {opset}, "backend-test",
           registry);
  }

  // 2x2 kernel with ``auto_pad = SAME_UPPER`` on a deterministic 1x1x4x4
  // input (mirrors ``test_averagepool_2d_same_upper``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", std::string("SAME_UPPER"));

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                                   /*ceil_mode=*/false, /*count_include_pad=*/false,
                                   /*dilations=*/{}, /*auto_pad=*/"SAME_UPPER");

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_same_upper", {opset}, "backend-test", registry);
  }

  // 2x2 kernel with ``auto_pad = SAME_LOWER`` on a deterministic 1x1x4x4
  // input (mirrors ``test_averagepool_2d_same_lower``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", std::string("SAME_LOWER"));

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                                   /*ceil_mode=*/false, /*count_include_pad=*/false,
                                   /*dilations=*/{}, /*auto_pad=*/"SAME_LOWER");

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_same_lower", {opset}, "backend-test", registry);
  }

  // 2x2 kernel, dilations (2, 2), strides (1, 1), ``ceil_mode = 1`` on a
  // 1x1x4x4 input (mirrors ``test_averagepool_2d_dilations``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                                   /*ceil_mode=*/true, /*count_include_pad=*/false,
                                   /*dilations=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_2d_dilations", {opset}, "backend-test", registry);
  }

  // 3-D AveragePool with a 2x2x2 kernel, dilations (2, 2, 2), strides
  // (1, 1, 1), ``ceil_mode = 1`` on a 1x1x4x4x4 input (mirrors
  // ``test_averagepool_3d_dilations_small``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

    // Four identical 4x4 planes filled with 1..16.
    std::vector<float> data;
    data.reserve(1 * 1 * 4 * 4 * 4);
    for (int64_t p = 0; p < 4; ++p) {
      for (int64_t i = 1; i <= 16; ++i) {
        data.push_back(static_cast<float>(i));
      }
    }
    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4, 4}, data);
    Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2, 2}, /*strides=*/{1, 1, 1},
                                   /*pads=*/{}, /*ceil_mode=*/true,
                                   /*count_include_pad=*/false, /*dilations=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_averagepool_3d_dilations_small", {opset}, "backend-test",
           registry);
  }

  // 3-D AveragePool with a 5x5x5 kernel, strides (3, 3, 3) and dilations
  // (2, 2, 2) on a deterministic random 1x1x32x32x32 input, exercising all
  // four (count_include_pad, ceil_mode) combinations (mirrors the
  // ``test_averagepool_3d_dilations_large_*`` reference cases — the upstream
  // cases use ``np.random.randn`` which is non-deterministic across
  // installations; here we use :cpp:func:`Randn` with a fixed seed instead so
  // the reference outputs are reproducible).
  {
    uint64_t seed = 31;
    for (int64_t cip : {int64_t{0}, int64_t{1}}) {
      for (bool ceil_mode : {true, false}) {
        NodeProto node;
        node.set_op_type("AveragePool");
        node.add_input("x");
        node.add_output("y");
        AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5, 5});
        AddAttribute<std::vector<int64_t>>(node, "strides", {3, 3, 3});
        AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2, 2});
        AddAttribute<int64_t>(node, "count_include_pad", cip);
        AddAttribute<int64_t>(node, "ceil_mode", ceil_mode ? 1 : 0);

        Tensor x = Tensor::FromFloat("", {1, 1, 32, 32, 32},
                                     Randn<float>({1, 1, 32, 32, 32}, /*seed=*/seed++));
        Tensor y = average_pool_kernel(x, /*kernel_shape=*/{5, 5, 5}, /*strides=*/{3, 3, 3},
                                       /*pads=*/{}, /*ceil_mode=*/ceil_mode,
                                       /*count_include_pad=*/cip != 0, /*dilations=*/{2, 2, 2});

        std::string name = "test_cc_averagepool_3d_dilations_large_count_include_pad_is_" +
                           std::to_string(cip) + "_ceil_mode_is_" + (ceil_mode ? "True" : "False");
        Expect(node, {x}, {y}, name, {opset}, "backend-test", registry);
      }
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
