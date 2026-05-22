// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_object_detection_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace object_detection {

std::string MakeRoiAlignDoc() {
  return R"DOC(
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
)DOC";
}

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
