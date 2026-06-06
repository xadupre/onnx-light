// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_kernels/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Builds a deterministic ``N=1, C=1, H=10, W=10`` feature map whose values
// run from 0/100 to 99/100. Shared by both registered cases so test inputs
// are easy to inspect.
Tensor MakeFeatureMap() {
  std::vector<float> values(100);
  for (int i = 0; i < 100; ++i) {
    values[i] = static_cast<float>(i) / 100.0f;
  }
  return Tensor::FromFloat("", {1, 1, 10, 10}, values);
}

// Builds the deterministic 1x1x10x10 feature map used by the upstream
// ``test_roialign_*`` reference cases (see
// ``onnx/backend/test/case/node/roialign.py``).
Tensor MakeUpstreamFeatureMap() {
  const std::vector<float> values = {
      0.2764f, 0.7150f, 0.1958f, 0.3416f, 0.4638f, 0.0259f, 0.2963f, 0.6518f, 0.4856f, 0.7250f,
      0.9637f, 0.0895f, 0.2919f, 0.6753f, 0.0234f, 0.6132f, 0.8085f, 0.5324f, 0.8992f, 0.4467f,
      0.3265f, 0.8479f, 0.9698f, 0.2471f, 0.9336f, 0.1878f, 0.4766f, 0.4308f, 0.3400f, 0.2162f,
      0.0206f, 0.1720f, 0.2155f, 0.4394f, 0.0653f, 0.3406f, 0.7724f, 0.3921f, 0.2541f, 0.5799f,
      0.4062f, 0.2194f, 0.4473f, 0.4687f, 0.7109f, 0.9327f, 0.9815f, 0.6320f, 0.1728f, 0.6119f,
      0.3097f, 0.1283f, 0.4984f, 0.5068f, 0.4279f, 0.0173f, 0.4388f, 0.0430f, 0.4671f, 0.7119f,
      0.1011f, 0.8477f, 0.4726f, 0.1777f, 0.9923f, 0.4042f, 0.1869f, 0.7795f, 0.9946f, 0.9689f,
      0.1366f, 0.3671f, 0.7011f, 0.6234f, 0.9867f, 0.5585f, 0.6985f, 0.5609f, 0.8788f, 0.9928f,
      0.5697f, 0.8511f, 0.6711f, 0.9406f, 0.8751f, 0.7496f, 0.1650f, 0.1049f, 0.1559f, 0.2514f,
      0.7012f, 0.4056f, 0.7879f, 0.3461f, 0.0415f, 0.2998f, 0.5094f, 0.3727f, 0.5482f, 0.0502f,
  };
  return Tensor::FromFloat("", {1, 1, 10, 10}, values);
}

} // namespace

// ---------------------------------------------------------------------------
// RoiAlign — region-of-interest aligned pooling described in the Mask R-CNN
// paper (since opset 10 in the ai.onnx domain; the
// ``coordinate_transformation_mode`` attribute was added in opset 16).
//
// Cases registered (each prefixed with ``test_cc_``):
//
//   * ``test_cc_roialign`` — opset 16 with the default
//     ``coordinate_transformation_mode = "half_pixel"`` and explicit
//     ``mode = "avg"``, exercising bilinear sampling with
//     ``sampling_ratio = 2`` and ``spatial_scale = 1.0`` on a 10x10 feature
//     map for two RoIs producing a 5x5 output per RoI.
//   * ``test_cc_roialign_max`` — opset 16 with
//     ``coordinate_transformation_mode = "output_half_pixel"`` (matching
//     opset 10 behaviour) and ``mode = "max"`` on the same feature map.
//
// The cases below mirror the upstream ONNX reference suite
// (``onnx/backend/test/case/node/roialign.py``); the inputs (X, rois,
// batch_indices) and node attributes match ``test_roialign_*`` exactly and
// the expected outputs are produced by our reference kernel:
//
//   * ``test_cc_roialign_aligned_false`` — opset 16, default
//     ``mode = "avg"``, ``coordinate_transformation_mode =
//     "output_half_pixel"`` (mirrors ``test_roialign_aligned_false``).
//   * ``test_cc_roialign_aligned_true`` — opset 16, default
//     ``mode = "avg"``, ``coordinate_transformation_mode = "half_pixel"``
//     (mirrors ``test_roialign_aligned_true``).
//   * ``test_cc_roialign_mode_max`` — opset 16, ``mode = "max"``,
//     ``coordinate_transformation_mode = "output_half_pixel"`` (mirrors
//     ``test_roialign_mode_max``).
// ---------------------------------------------------------------------------
void RegisterRoiAlignCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::RoiAlign roialign_kernel{ctx};

  // Two RoIs over the 10x10 feature map.
  const std::vector<int64_t> rois_shape = {2, 4};
  const std::vector<float> rois_values = {
      0.0f, 0.0f, 9.0f, 9.0f, // covers the full feature map
      2.0f, 2.0f, 7.0f, 7.0f, // covers an interior 5x5 region
  };
  const std::vector<int64_t> batch_indices_values = {0, 0};

  // Case 1: avg mode with the default coordinate_transformation_mode.
  {
    NodeProto node;
    node.set_op_type("RoiAlign");
    node.add_input("X");
    node.add_input("rois");
    node.add_input("batch_indices");
    node.add_output("Y");

    AttributeProto *mode_attr = node.add_attribute();
    mode_attr->set_name("mode");
    mode_attr->set_type(AttributeProto::AttributeType::STRING);
    mode_attr->set_s("avg");

    AttributeProto *oh = node.add_attribute();
    oh->set_name("output_height");
    oh->set_type(AttributeProto::AttributeType::INT);
    oh->set_i(5);

    AttributeProto *ow = node.add_attribute();
    ow->set_name("output_width");
    ow->set_type(AttributeProto::AttributeType::INT);
    ow->set_i(5);

    AttributeProto *sr = node.add_attribute();
    sr->set_name("sampling_ratio");
    sr->set_type(AttributeProto::AttributeType::INT);
    sr->set_i(2);

    AttributeProto *ss = node.add_attribute();
    ss->set_name("spatial_scale");
    ss->set_type(AttributeProto::AttributeType::FLOAT);
    ss->set_f(1.0f);

    Tensor x = MakeFeatureMap();
    Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);
    Tensor batch_indices = Tensor::FromInt64("", {2}, batch_indices_values);

    kernel::RoiAlign::Attributes attrs;
    attrs.mode = "avg";
    attrs.output_height = 5;
    attrs.output_width = 5;
    attrs.sampling_ratio = 2;
    attrs.spatial_scale = 1.0f;
    attrs.coordinate_transformation_mode = "half_pixel";
    Tensor y = roialign_kernel(x, rois, batch_indices, attrs);

    Expect(node, {x, rois, batch_indices}, {y}, "test_cc_roialign", {opset}, "backend-test",
           registry);
  }

  // Case 2: max mode with output_half_pixel (legacy opset-10 behaviour).
  {
    NodeProto node;
    node.set_op_type("RoiAlign");
    node.add_input("X");
    node.add_input("rois");
    node.add_input("batch_indices");
    node.add_output("Y");

    AttributeProto *mode_attr = node.add_attribute();
    mode_attr->set_name("mode");
    mode_attr->set_type(AttributeProto::AttributeType::STRING);
    mode_attr->set_s("max");

    AttributeProto *ctm = node.add_attribute();
    ctm->set_name("coordinate_transformation_mode");
    ctm->set_type(AttributeProto::AttributeType::STRING);
    ctm->set_s("output_half_pixel");

    AttributeProto *oh = node.add_attribute();
    oh->set_name("output_height");
    oh->set_type(AttributeProto::AttributeType::INT);
    oh->set_i(5);

    AttributeProto *ow = node.add_attribute();
    ow->set_name("output_width");
    ow->set_type(AttributeProto::AttributeType::INT);
    ow->set_i(5);

    AttributeProto *sr = node.add_attribute();
    sr->set_name("sampling_ratio");
    sr->set_type(AttributeProto::AttributeType::INT);
    sr->set_i(2);

    AttributeProto *ss = node.add_attribute();
    ss->set_name("spatial_scale");
    ss->set_type(AttributeProto::AttributeType::FLOAT);
    ss->set_f(1.0f);

    Tensor x = MakeFeatureMap();
    Tensor rois = Tensor::FromFloat("", rois_shape, rois_values);
    Tensor batch_indices = Tensor::FromInt64("", {2}, batch_indices_values);

    kernel::RoiAlign::Attributes attrs;
    attrs.mode = "max";
    attrs.output_height = 5;
    attrs.output_width = 5;
    attrs.sampling_ratio = 2;
    attrs.spatial_scale = 1.0f;
    attrs.coordinate_transformation_mode = "output_half_pixel";
    Tensor y = roialign_kernel(x, rois, batch_indices, attrs);

    Expect(node, {x, rois, batch_indices}, {y}, "test_cc_roialign_max", {opset}, "backend-test",
           registry);
  }

  // Inputs shared by the three upstream ``test_roialign_*`` cases (see
  // ``onnx/backend/test/case/node/roialign.py``). The same X, rois and
  // batch_indices feed all three cases; only the attributes differ.
  const std::vector<int64_t> upstream_rois_shape = {3, 4};
  const std::vector<float> upstream_rois_values = {
      0.0f, 0.0f, 9.0f, 9.0f, //
      0.0f, 5.0f, 4.0f, 9.0f, //
      5.0f, 5.0f, 9.0f, 9.0f, //
  };
  const std::vector<int64_t> upstream_batch_indices_values = {0, 0, 0};

  // Helper lambda that registers one upstream case. Builds the node with the
  // requested attributes, computes the expected output using the reference
  // kernel, and emits it under ``case_name``.
  auto register_upstream = [&](const std::string &case_name, const std::string &mode,
                               const std::string &coordinate_transformation_mode) {
    NodeProto node;
    node.set_op_type("RoiAlign");
    node.add_input("X");
    node.add_input("rois");
    node.add_input("batch_indices");
    node.add_output("Y");

    if (!mode.empty()) {
      AddAttribute<std::string>(node, "mode", mode);
    }
    AddAttribute<std::string>(node, "coordinate_transformation_mode",
                              coordinate_transformation_mode);
    AddAttribute<int64_t>(node, "output_height", 5);
    AddAttribute<int64_t>(node, "output_width", 5);
    AddAttribute<int64_t>(node, "sampling_ratio", 2);
    AddAttribute<float>(node, "spatial_scale", 1.0f);

    Tensor x = MakeUpstreamFeatureMap();
    Tensor rois = Tensor::FromFloat("", upstream_rois_shape, upstream_rois_values);
    Tensor batch_indices = Tensor::FromInt64("", {3}, upstream_batch_indices_values);

    kernel::RoiAlign::Attributes attrs;
    attrs.mode = mode.empty() ? "avg" : mode;
    attrs.output_height = 5;
    attrs.output_width = 5;
    attrs.sampling_ratio = 2;
    attrs.spatial_scale = 1.0f;
    attrs.coordinate_transformation_mode = coordinate_transformation_mode;
    Tensor y = roialign_kernel(x, rois, batch_indices, attrs);

    Expect(node, {x, rois, batch_indices}, {y}, case_name, {opset}, "backend-test", registry);
  };

  // Upstream ``test_roialign_aligned_false``: avg mode (default) with
  // ``coordinate_transformation_mode = "output_half_pixel"``.
  register_upstream("test_cc_roialign_aligned_false", /*mode=*/"", "output_half_pixel");

  // Upstream ``test_roialign_aligned_true``: avg mode (default) with
  // ``coordinate_transformation_mode = "half_pixel"``.
  register_upstream("test_cc_roialign_aligned_true", /*mode=*/"", "half_pixel");

  // Upstream ``test_roialign_mode_max``: max mode with
  // ``coordinate_transformation_mode = "output_half_pixel"``.
  register_upstream("test_cc_roialign_mode_max", "max", "output_half_pixel");
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
