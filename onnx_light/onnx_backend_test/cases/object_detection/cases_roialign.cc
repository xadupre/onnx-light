// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

} // namespace

// ---------------------------------------------------------------------------
// RoiAlign — region-of-interest aligned pooling described in the Mask R-CNN
// paper (since opset 10 in the ai.onnx domain; the
// ``coordinate_transformation_mode`` attribute was added in opset 16).
//
// Two cases are registered:
//
//   * ``test_cc_roialign`` — opset 16 with the default
//     ``coordinate_transformation_mode = "half_pixel"`` and explicit
//     ``mode = "avg"``, exercising bilinear sampling with
//     ``sampling_ratio = 2`` and ``spatial_scale = 1.0`` on a 10x10 feature
//     map for two RoIs producing a 5x5 output per RoI.
//   * ``test_cc_roialign_max`` — opset 16 with
//     ``coordinate_transformation_mode = "output_half_pixel"`` (matching
//     opset 10 behaviour) and ``mode = "max"`` on the same feature map.
// ---------------------------------------------------------------------------
void RegisterRoiAlignCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::RoiAlign roialign_kernel{kernel::KernelContext(opset)};

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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
