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

std::string MakeNonMaxSuppressionDoc() {
  return R"DOC(
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Boxes are suppressed if their IOU with a previously selected box is strictly greater than iou_threshold (i.e., boxes with IOU exactly equal to the threshold are kept).
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
)DOC";
}

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
