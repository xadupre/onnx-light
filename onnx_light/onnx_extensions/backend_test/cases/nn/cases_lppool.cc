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
// LpPool — y = (sum |x_i|^p)^(1/p) over the pooling window (since opset 22
// in the ai.onnx domain; the subset of attributes exercised here has been
// stable since opset 18 — ``dilations`` was added at opset 18).
//
// Cases registered (each is the C++ analogue of the like-named ONNX
// reference case, with the ``test_cc_`` prefix):
//
//   * ``test_cc_lppool_1d_default`` — 1-D, 2-wide kernel, ``p = 3``.
//   * ``test_cc_lppool_2d_default`` — 2x2 kernel, ``p = 4``.
//   * ``test_cc_lppool_3d_default`` — 3-D, 2x2x2 kernel, ``p = 3``.
//   * ``test_cc_lppool_2d_same_upper`` — 2x2 kernel,
//     ``auto_pad = SAME_UPPER``, ``p = 2``.
//   * ``test_cc_lppool_2d_same_lower`` — 2x2 kernel,
//     ``auto_pad = SAME_LOWER``, ``p = 4``.
//   * ``test_cc_lppool_2d_pads`` — 3x3 kernel, pads ``(2, 2, 2, 2)``,
//     ``p = 3``.
//   * ``test_cc_lppool_2d_strides`` — 5x5 kernel, strides ``(3, 3)``,
//     ``p = 2``.
//   * ``test_cc_lppool_2d_dilations`` — 2x2 kernel, dilations ``(2, 2)``,
//     ``p = 2`` (exact-value parity case with the upstream reference, which
//     ships a precomputed deterministic expected output).
// ---------------------------------------------------------------------------
void RegisterLpPoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const auto lp_pool_kernel = MakeReferenceKernel<onnx_kernels::kernel::LpPool>(opset);

  uint64_t seed = 137;

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1});
    AddAttribute<int64_t>(node, "p", 3);

    constexpr int64_t in_count = 1 * 32 * 16384;
    constexpr int64_t out_count = 1 * 32 * 16383;
    const uint64_t benchmark_seed = seed;
    Expect(registry, std::move(node), "test_cc_lppool_1d_default_benchmark", {opset}, {in_count},
           {out_count}, [lp_pool_kernel, benchmark_seed]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 32, 16384}, benchmark_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2}, /*strides=*/{1},
                             /*pads=*/{}, /*p=*/3);
             });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 1-D LpPool with a 2-wide kernel, p = 3 on a 1x3x32 input (mirrors
  // ``test_lppool_1d_default``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1});
    AddAttribute<int64_t>(node, "p", 3);
    Expect(registry, std::move(node), "test_cc_lppool_1d_default", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2}, /*strides=*/{1}, /*pads=*/{},
                             /*p=*/3);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool, 2x2 kernel, p = 4 on a 1x3x32x32 input (mirrors
  // ``test_lppool_2d_default``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<int64_t>(node, "p", 4);
    Expect(registry, std::move(node), "test_cc_lppool_2d_default", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{}, /*p=*/4);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 3-D LpPool, 2x2x2 kernel, p = 3 on a 1x3x32x32x32 input (mirrors
  // ``test_lppool_3d_default``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});
    AddAttribute<int64_t>(node, "p", 3);
    Expect(registry, std::move(node), "test_cc_lppool_3d_default", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32, 32, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2, 2, 2}, /*strides=*/{}, /*pads=*/{},
                             /*p=*/3);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool, 2x2 kernel, auto_pad = SAME_UPPER, p = 2 (mirrors
  // ``test_lppool_2d_same_upper``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_cc_lppool_2d_same_upper", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                             /*p=*/2, /*ceil_mode=*/false, /*dilations=*/{},
                             /*auto_pad=*/AutoPad::kSameUpper);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool, 2x2 kernel, auto_pad = SAME_LOWER, p = 4 (mirrors
  // ``test_lppool_2d_same_lower``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", "SAME_LOWER");
    AddAttribute<int64_t>(node, "p", 4);
    Expect(registry, std::move(node), "test_cc_lppool_2d_same_lower", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                             /*p=*/4, /*ceil_mode=*/false, /*dilations=*/{},
                             /*auto_pad=*/AutoPad::kSameLower);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool, 3x3 kernel, pads = (2, 2, 2, 2), p = 3 on a 1x3x28x28 input
  // (mirrors ``test_lppool_2d_pads``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});
    AddAttribute<int64_t>(node, "p", 3);
    Expect(registry, std::move(node), "test_cc_lppool_2d_pads", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 28, 28}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{},
                             /*pads=*/{2, 2, 2, 2}, /*p=*/3);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool, 5x5 kernel, strides (3, 3), p = 2 on a 1x3x32x32 input
  // (mirrors ``test_lppool_2d_strides``).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 3});
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_cc_lppool_2d_strides", {opset},
           [=, case_seed = seed++]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 3, 32, 32}, /*seed=*/case_seed);
             Tensor y = lp_pool_kernel.Invoke([&](const auto &kernel) {
               return kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{3, 3}, /*pads=*/{},
                             /*p=*/2);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // 2-D LpPool with dilations (mirrors ``test_lppool_2d_dilations`` — uses
  // the exact deterministic input and expected output shipped with the
  // upstream ONNX reference case).
  {
    NodeProto node;
    node.set_op_type("LpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_cc_lppool_2d_dilations", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
      Tensor y = Tensor::FromFloat(
          "", {1, 1, 2, 2},
          {14.560219778561036f, 16.24807680927192f, 21.633307652783937f, 23.49468024894146f});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
