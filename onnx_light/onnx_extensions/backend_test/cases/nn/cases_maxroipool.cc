// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a deterministic ``N=1, C=2, H=6, W=6`` feature map whose values
// run from 0/36 to 71/36, so each of the two channels has a different,
// easy-to-inspect pattern.
Tensor MakeFeatureMap() {
  std::vector<float> values(2 * 6 * 6);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i) / 36.0f;
  }
  return Tensor::FromFloat("", {1, 2, 6, 6}, values);
}

} // namespace

// ---------------------------------------------------------------------------
// MaxRoiPool — Fast R-CNN style region-of-interest max pooling (since opset 1
// in the ai.onnx domain). The kernel is restricted to FLOAT inputs and
// supports the attribute set:
//
//   * ``pooled_shape`` (INTS, required): pooled (height, width) per RoI.
//   * ``spatial_scale`` (FLOAT, default 1.0).
//
// Cases registered (each prefixed with ``test_cc_``):
//
//   * ``test_cc_maxroipool_default`` — opset 22 with ``pooled_shape =
//     {2, 2}`` and the default ``spatial_scale = 1.0`` on a 1x2x6x6 feature
//     map for two RoIs (one full-extent, one interior).
//   * ``test_cc_maxroipool_spatial_scale`` — opset 22 with ``pooled_shape =
//     {3, 3}`` and ``spatial_scale = 0.5`` on the same feature map for one
//     RoI whose coordinates address the full feature map after scaling.
// ---------------------------------------------------------------------------
void RegisterMaxRoiPoolCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const auto maxroipool_kernel = MakeReferenceKernel<onnx_kernels::kernel::MaxRoiPool>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("MaxRoiPool");
    node.add_input("X");
    node.add_input("rois");
    node.add_output("Y");
    AddAttribute<std::vector<int64_t>>(node, "pooled_shape", {2, 2});

    constexpr int64_t x_count = 1 * 32 * 128 * 128;
    constexpr int64_t rois_count = 64 * 5;
    constexpr int64_t y_count = 64 * 32 * 2 * 2;
    Expect(registry, std::move(node), "test_cc_maxroipool_default_benchmark", {opset},
           {x_count, rois_count}, {y_count}, [maxroipool_kernel]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 32, 128, 128}, 2801);
             const std::vector<int64_t> rois_shape = {64, 5};
             std::vector<float> rois_values;
             rois_values.reserve(64 * 5);
             for (int64_t i = 0; i < 64; ++i) {
               const float start = static_cast<float>(i % 32);
               rois_values.insert(rois_values.end(),
                                  {0.0f, start, start, start + 63.0f, start + 63.0f});
             }
             Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);
             onnx_kernels::kernel::MaxRoiPool::Attributes attrs;
             attrs.pooled_shape = {2, 2};
             attrs.spatial_scale = 1.0f;
             Tensor y = maxroipool_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(x, rois, attrs); });
             return IoData{{std::move(x), std::move(rois)}, {std::move(y)}};
           });
    return;
  }

  // Case 1: default spatial_scale, two RoIs, 2x2 pooled output.
  {
    NodeProto node;
    node.set_op_type("MaxRoiPool");
    node.add_input("X");
    node.add_input("rois");
    node.add_output("Y");
    AddAttribute<std::vector<int64_t>>(node, "pooled_shape", {2, 2});
    Expect(registry, std::move(node), "test_cc_maxroipool_default", {opset}, [=]() -> IoData {
      Tensor x = MakeFeatureMap();
      const std::vector<int64_t> rois_shape = {2, 5};
      const std::vector<float> rois_values = {
          0.0f, 0.0f, 0.0f, 5.0f, 5.0f, // full 6x6 extent
          0.0f, 1.0f, 1.0f, 4.0f, 4.0f, // interior 4x4 extent
      };
      Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);

      onnx_kernels::kernel::MaxRoiPool::Attributes attrs;
      attrs.pooled_shape = {2, 2};
      attrs.spatial_scale = 1.0f;
      Tensor y =
          maxroipool_kernel.Invoke([&](const auto &kernel) { return kernel(x, rois, attrs); });

      return IoData{{std::move(x), std::move(rois)}, {std::move(y)}};
    });
  }

  // Case 2: explicit spatial_scale = 0.5, single RoI, 3x3 pooled output.
  {
    NodeProto node;
    node.set_op_type("MaxRoiPool");
    node.add_input("X");
    node.add_input("rois");
    node.add_output("Y");
    AddAttribute<std::vector<int64_t>>(node, "pooled_shape", {3, 3});
    AddAttribute<float>(node, "spatial_scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_maxroipool_spatial_scale", {opset}, [=]() -> IoData {
      Tensor x = MakeFeatureMap();
      const std::vector<int64_t> rois_shape = {1, 5};
      // Scaled by 0.5 the corners (0, 0)..(10, 10) cover the whole 6x6 map.
      const std::vector<float> rois_values = {0.0f, 0.0f, 0.0f, 10.0f, 10.0f};
      Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);

      onnx_kernels::kernel::MaxRoiPool::Attributes attrs;
      attrs.pooled_shape = {3, 3};
      attrs.spatial_scale = 0.5f;
      Tensor y =
          maxroipool_kernel.Invoke([&](const auto &kernel) { return kernel(x, rois, attrs); });

      return IoData{{std::move(x), std::move(rois)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
