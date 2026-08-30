// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

using onnx_kernels::kernel::AutoPad;

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
//   * ``test_cc_averagepool_18_ceil_count_include_pad_1d`` — opset-18, 1-D,
//     7-wide kernel, strides ``(3)``, pads ``(3, 3)``, ``ceil_mode = 1`` and
//     ``count_include_pad = 1``.
//   * ``test_cc_averagepool_2d_default`` — 2x2 kernel, default strides (1),
//     no padding.
//   * ``test_cc_averagepool_2d_strides`` — 3x3 kernel, strides ``(2, 2)``,
//     no padding.
//   * ``test_cc_averagepool_2d_ceil`` — 3x3 kernel, strides ``(2, 2)``,
//     ``ceil_mode = 1``.
//   * ``test_cc_averagepool_18_ceil_count_include_pad_2d`` — opset-18, 3x3
//     kernel, strides ``(2, 2)``, pads ``(1, 1, 1, 1)``, ``ceil_mode = 1``
//     and ``count_include_pad = 1``.
//   * ``test_cc_averagepool_18_ceil_count_exclude_pad_2d`` — opset-18, 3x3
//     kernel, strides ``(2, 2)``, pads ``(1, 1, 1, 1)`` and
//     ``ceil_mode = 1`` with the default ``count_include_pad = 0``.
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
//   * ``test_cc_averagepool_2d_dilations_valid`` — 3x3 kernel, dilations
//     ``(2, 2)``, ``auto_pad = VALID``.
//   * ``test_cc_averagepool_3d_default`` — 3-D, 2x2x2 kernel.
//   * ``test_cc_averagepool_18_ceil_count_include_pad_3d`` — opset-18, 3-D,
//     3x3x3 kernel, strides ``(2, 2, 2)``, pads ``(1, 1, 1, 1, 1, 1)``,
//     ``ceil_mode = 1`` and ``count_include_pad = 1``.
//   * ``test_cc_averagepool_3d_dilations_small`` — 3-D, 2x2x2 kernel,
//     dilations ``(2, 2, 2)``, ``ceil_mode = 1``.
//   * ``test_cc_averagepool_3d_dilations_large_count_include_pad_is_{0,1}_ceil_mode_is_{True,False}``
//     — 3-D, 5x5x5 kernel, strides ``(3, 3, 3)``, dilations ``(2, 2, 2)`` on
//     a deterministic random 1x1x32x32x32 input (mirrors the four
//     ``test_averagepool_3d_dilations_large_*`` reference cases; inputs are
//     drawn from :cpp:func:`Randn` with a fixed seed instead of
//     ``np.random.randn`` so the values are reproducible).
// ---------------------------------------------------------------------------
void RegisterAveragePoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset18 = DefaultOpset(18);

  const OpsetId opset = DefaultOpset(19);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});

    constexpr int64_t in_count = 1 * 32 * 128 * 128;
    constexpr int64_t out_count = 1 * 32 * 127 * 127;
    Expect(registry, std::move(node), "test_cc_averagepool_2d_default_benchmark", {opset},
           {in_count}, {out_count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {1, 32, 128, 128}, 1101);
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Default 2x2 kernel on a 1x1x4x4 input.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    Expect(registry, std::move(node), "test_cc_averagepool_2d_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 3x3 kernel with strides (2, 2) on a 1x1x5x5 input.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_averagepool_2d_strides", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                                   {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_pads_count_include_pad", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                                            /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/false,
                                            /*count_include_pad=*/true);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 1-D AveragePool with a 2-wide kernel (mirrors
  // ``test_averagepool_1d_default``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});
    Expect(registry, std::move(node), "test_cc_averagepool_1d_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 8}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Opset-18 clone of ``test_averagepool_19_ceil_count_include_pad_1d``,
  // covering the legacy ceil_mode + count_include_pad path mirrored in
  // onnxruntime PR #29629.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {7});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {3, 3});
    AddAttribute<int64_t>(node, "ceil_mode", 1);
    AddAttribute<int64_t>(node, "count_include_pad", 1);
    Expect(
        registry, std::move(node), "test_cc_averagepool_18_ceil_count_include_pad_1d", {opset18},
        []() -> IoData {
          const OpsetId opset18 = DefaultOpset(18);

          const KernelContext average_pool_kernel18_ctx{opset18};
          const onnx_kernels::kernel::AveragePool average_pool_kernel18{average_pool_kernel18_ctx};

          Tensor x = Tensor::FromFloat("", {1, 2, 9},
                                       {2.0903f, 4.6493f, 1.6320f, -3.2051f, 4.6975f, 4.7296f,
                                        3.3653f, -1.5815f, -2.3832f, 0.9628f, -1.5899f, -2.6820f,
                                        5.7529f, 7.7346f, -0.8910f, -2.0151f, 0.1313f, -0.5374f});
          Tensor y =
              average_pool_kernel18(x, /*kernel_shape=*/{7}, /*strides=*/{3}, /*pads=*/{3, 3},
                                    /*ceil_mode=*/true, /*count_include_pad=*/true);

          return IoData{{std::move(x)}, {std::move(y)}};
        });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_ceil", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2},
                                     /*pads=*/{}, /*ceil_mode=*/true,
                                     /*count_include_pad=*/false);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Opset-18 clone of the ceil_mode + count_include_pad regression case from
  // onnxruntime PR #29629.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<int64_t>(node, "ceil_mode", 1);
    AddAttribute<int64_t>(node, "count_include_pad", 1);
    Expect(registry, std::move(node), "test_cc_averagepool_18_ceil_count_include_pad_2d", {opset18},
           []() -> IoData {
             const OpsetId opset18 = DefaultOpset(18);

             const KernelContext average_pool_kernel18_ctx{opset18};
             const onnx_kernels::kernel::AveragePool average_pool_kernel18{
                 average_pool_kernel18_ctx};

             Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                           10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
             Tensor y = average_pool_kernel18(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2},
                                              /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/true,
                                              /*count_include_pad=*/true);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // No-regression opset-18 ceil_mode case: excluding padding must remain
  // correct on the legacy path as well.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    AddAttribute<int64_t>(node, "ceil_mode", 1);
    AddAttribute<int64_t>(node, "count_include_pad", 0);
    Expect(registry, std::move(node), "test_cc_averagepool_18_ceil_count_exclude_pad_2d", {opset18},
           []() -> IoData {
             const OpsetId opset18 = DefaultOpset(18);

             const KernelContext average_pool_kernel18_ctx{opset18};
             const onnx_kernels::kernel::AveragePool average_pool_kernel18{
                 average_pool_kernel18_ctx};

             Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                           10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
             Tensor y = average_pool_kernel18(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2},
                                              /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/true,
                                              /*count_include_pad=*/false);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_ceil_last_window_starts_on_pad",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{3, 3},
                                            /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/true,
                                            /*count_include_pad=*/true);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_pads", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                                     /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                     /*count_include_pad=*/false);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_precomputed_pads", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                                            /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                            /*count_include_pad=*/false);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_precomputed_pads_count_include_pad",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                                            /*pads=*/{2, 2, 2, 2}, /*ceil_mode=*/false,
                                            /*count_include_pad=*/true);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_precomputed_strides", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
             Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 3-D AveragePool with a 2x2x2 kernel (mirrors
  // ``test_averagepool_3d_default``).
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});
    Expect(registry, std::move(node), "test_cc_averagepool_3d_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      std::vector<float> data(1 * 1 * 3 * 3 * 3);
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {1, 1, 3, 3, 3}, data);
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2, 2});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Opset-18 clone of the 3-D ceil_mode + count_include_pad regression case
  // from onnxruntime PR #29629.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1, 1, 1});
    AddAttribute<int64_t>(node, "ceil_mode", 1);
    AddAttribute<int64_t>(node, "count_include_pad", 1);
    Expect(registry, std::move(node), "test_cc_averagepool_18_ceil_count_include_pad_3d", {opset18},
           []() -> IoData {
             const OpsetId opset18 = DefaultOpset(18);

             const KernelContext average_pool_kernel18_ctx{opset18};
             const onnx_kernels::kernel::AveragePool average_pool_kernel18{
                 average_pool_kernel18_ctx};

             std::vector<float> data(27);
             for (size_t i = 0; i < data.size(); ++i) {
               data[i] = static_cast<float>(i + 1);
             }
             Tensor x = Tensor::FromFloat("", {1, 1, 3, 3, 3}, data);
             Tensor y = average_pool_kernel18(x, /*kernel_shape=*/{3, 3, 3}, /*strides=*/{2, 2, 2},
                                              /*pads=*/{1, 1, 1, 1, 1, 1}, /*ceil_mode=*/true,
                                              /*count_include_pad=*/true);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_precomputed_same_upper", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
             Tensor y =
                 average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                                     /*ceil_mode=*/false, /*count_include_pad=*/false,
                                     /*dilations=*/{}, /*auto_pad=*/AutoPad::kSameUpper);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_same_upper", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                                     /*ceil_mode=*/false, /*count_include_pad=*/false,
                                     /*dilations=*/{}, /*auto_pad=*/AutoPad::kSameUpper);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_same_lower", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                                     /*ceil_mode=*/false, /*count_include_pad=*/false,
                                     /*dilations=*/{}, /*auto_pad=*/AutoPad::kSameLower);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_averagepool_2d_dilations", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(19);

      const KernelContext average_pool_kernel_ctx{opset};
      const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = average_pool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                                     /*ceil_mode=*/true, /*count_include_pad=*/false,
                                     /*dilations=*/{2, 2});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 3x3 kernel, dilations (2, 2), strides (1, 1), ``auto_pad = VALID`` on a
  // 1x1x7x7 input. Covers the dilated effective-kernel output-shape path fixed
  // in ONNX PR #8174.
  {
    NodeProto node;
    node.set_op_type("AveragePool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", std::string("VALID"));
    Expect(registry, std::move(node), "test_cc_averagepool_2d_dilations_valid", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

             Tensor x = Tensor::FromFloat(
                 "", {1, 1, 7, 7},
                 {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                  21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                  31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                  41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f});
             Tensor y =
                 average_pool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1}, /*pads=*/{},
                                     /*ceil_mode=*/false, /*count_include_pad=*/false,
                                     /*dilations=*/{2, 2}, /*auto_pad=*/AutoPad::kValid);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_averagepool_3d_dilations_small", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(19);

             const KernelContext average_pool_kernel_ctx{opset};
             const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

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

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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

        std::string name = "test_cc_averagepool_3d_dilations_large_count_include_pad_is_" +
                           std::to_string(cip) + "_ceil_mode_is_" + (ceil_mode ? "True" : "False");
        const uint64_t captured_seed = seed++;
        Expect(
            registry, std::move(node), name, {opset}, [captured_seed, ceil_mode, cip]() -> IoData {
              const OpsetId opset = DefaultOpset(19);

              const KernelContext average_pool_kernel_ctx{opset};
              const onnx_kernels::kernel::AveragePool average_pool_kernel{average_pool_kernel_ctx};

              Tensor x = RandnTensor(DataType::FLOAT, {1, 1, 32, 32, 32}, /*seed=*/captured_seed);
              Tensor y =
                  average_pool_kernel(x, /*kernel_shape=*/{5, 5, 5}, /*strides=*/{3, 3, 3},
                                      /*pads=*/{}, /*ceil_mode=*/ceil_mode,
                                      /*count_include_pad=*/cip != 0, /*dilations=*/{2, 2, 2});
              return IoData{{std::move(x)}, {std::move(y)}};
            });
      }
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
