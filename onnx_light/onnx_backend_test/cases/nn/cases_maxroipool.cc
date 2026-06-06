// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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
void RegisterMaxRoiPoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::MaxRoiPool maxroipool_kernel{ctx};

  // Case 1: default spatial_scale, two RoIs, 2x2 pooled output.
  {
    NodeProto node;
    node.set_op_type("MaxRoiPool");
    node.add_input("X");
    node.add_input("rois");
    node.add_output("Y");
    AddAttribute<std::vector<int64_t>>(node, "pooled_shape", {2, 2});

    Tensor x = MakeFeatureMap();
    const std::vector<int64_t> rois_shape = {2, 5};
    const std::vector<float> rois_values = {
        0.0f, 0.0f, 0.0f, 5.0f, 5.0f, // full 6x6 extent
        0.0f, 1.0f, 1.0f, 4.0f, 4.0f, // interior 4x4 extent
    };
    Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);

    kernel::MaxRoiPool::Attributes attrs;
    attrs.pooled_shape = {2, 2};
    attrs.spatial_scale = 1.0f;
    Tensor y = maxroipool_kernel(x, rois, attrs);

    Expect(node, {x, rois}, {y}, "test_cc_maxroipool_default", {opset}, "backend-test", registry);
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

    Tensor x = MakeFeatureMap();
    const std::vector<int64_t> rois_shape = {1, 5};
    // Scaled by 0.5 the corners (0, 0)..(10, 10) cover the whole 6x6 map.
    const std::vector<float> rois_values = {0.0f, 0.0f, 0.0f, 10.0f, 10.0f};
    Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);

    kernel::MaxRoiPool::Attributes attrs;
    attrs.pooled_shape = {3, 3};
    attrs.spatial_scale = 0.5f;
    Tensor y = maxroipool_kernel(x, rois, attrs);

    Expect(node, {x, rois}, {y}, "test_cc_maxroipool_spatial_scale", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
