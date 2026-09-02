// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// GlobalAveragePool — output = mean over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globalaveragepool — 2-D spatial (N=1, C=3, H=5, W=5).
//   * test_cc_globalaveragepool_precomputed — precomputed 1x1x3x3 example.
// ---------------------------------------------------------------------------
void RegisterGlobalAveragePoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("GlobalAveragePool");
    node.add_input("x");
    node.add_output("y");

    constexpr int64_t in_count = 1 * 64 * 128 * 128;
    constexpr int64_t out_count = 1 * 64 * 1 * 1;
    Expect(registry, std::move(node), "test_cc_globalaveragepool_benchmark", {opset}, {in_count},
           {out_count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::GlobalAveragePool kernel{kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {1, 64, 128, 128}, 1801);
             Tensor y = kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 1 x 3 x 5 x 5 input — mirrors test_globalaveragepool.
  {
    NodeProto node;
    node.set_op_type("GlobalAveragePool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globalaveragepool", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalAveragePool kernel{kernel_ctx};

      std::vector<float> x_data(1 * 3 * 5 * 5);
      for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
      Tensor y = kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 1 x 1 x 3 x 3 precomputed example.
  {
    NodeProto node;
    node.set_op_type("GlobalAveragePool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globalaveragepool_precomputed", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::GlobalAveragePool kernel{kernel_ctx};

             Tensor x = Tensor::FromFloat("", {1, 1, 3, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
             Tensor y = kernel(x); // expected: 5.0

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

// ---------------------------------------------------------------------------
// GlobalMaxPool — output = max over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globalmaxpool — 2-D spatial (N=1, C=3, H=5, W=5).
//   * test_cc_globalmaxpool_3d — 1-D spatial (N=2, C=4, W=10).
//   * test_cc_globalmaxpool_precomputed — precomputed 1x1x3x3 example.
// ---------------------------------------------------------------------------
void RegisterGlobalMaxPoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");

    constexpr int64_t in_count = 1 * 64 * 128 * 128;
    constexpr int64_t out_count = 1 * 64 * 1 * 1;
    Expect(registry, std::move(node), "test_cc_globalmaxpool_benchmark", {opset}, {in_count},
           {out_count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::GlobalMaxPool kernel{kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {1, 64, 128, 128}, 1802);
             Tensor y = kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 1 x 3 x 5 x 5 input — mirrors test_globalmaxpool.
  {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globalmaxpool", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalMaxPool kernel{kernel_ctx};

      std::vector<float> x_data(1 * 3 * 5 * 5);
      for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
      Tensor y = kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 2 x 4 x 10 input — mirrors test_globalmaxpool_3d.
  {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globalmaxpool_3d", {opset}, []() -> IoData {
      std::vector<float> x_data(2 * 4 * 10);
      for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {2, 4, 10}, x_data);
      Tensor y = Tensor::FromFloat("", {2, 4, 1},
                                   {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 1 x 1 x 3 x 3 precomputed example.
  {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globalmaxpool_precomputed", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalMaxPool kernel{kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 3, 3},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
      Tensor y = kernel(x); // expected: 9.0

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

// ---------------------------------------------------------------------------
// GlobalLpPool — output = Lp norm over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globallppool_lp1 — 2-D spatial, p=1 (L1 norm).
//   * test_cc_globallppool_lp2 — 2-D spatial, p=2 (default L2 norm).
//   * test_cc_globallppool_default — 1x1x3x3, default p=2.
// ---------------------------------------------------------------------------
void RegisterGlobalLpPoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", 1);

    constexpr int64_t in_count = 1 * 64 * 128 * 128;
    constexpr int64_t out_count = 1 * 64 * 1 * 1;
    Expect(registry, std::move(node), "test_cc_globallppool_lp1_benchmark", {opset}, {in_count},
           {out_count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::GlobalLpPool kernel{kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {1, 64, 128, 128}, 1803);
             Tensor y = kernel(x, /*p=*/1);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 1 x 3 x 5 x 5 input, p=1.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_cc_globallppool_lp1", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalLpPool kernel{kernel_ctx};

      std::vector<float> x_data(1 * 3 * 5 * 5);
      for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
      Tensor y = kernel(x, /*p=*/1);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 1 x 3 x 5 x 5 input, p=2.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_cc_globallppool_lp2", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalLpPool kernel{kernel_ctx};

      std::vector<float> x_data(1 * 3 * 5 * 5);
      for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = static_cast<float>(i + 1);
      }
      Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
      Tensor y = kernel(x, /*p=*/2);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // 1 x 1 x 3 x 3, default p=2.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_globallppool_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::GlobalLpPool kernel{kernel_ctx};

      Tensor x = Tensor::FromFloat("", {1, 1, 3, 3},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
      Tensor y = kernel(x); // expected: sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285)

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
