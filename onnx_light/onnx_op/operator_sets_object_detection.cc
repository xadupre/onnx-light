// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_object_detection.h"
#include "onnx_op/operator_sets_object_detection_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace object_detection {

namespace {

std::vector<TensorType> RoiAlignFloatTypes() {
  return {
      TensorType::kBfloat16,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

LightOpSchema MakeRoiAlignSchema(int since_version) {
  return LightOpSchema(
      "RoiAlign", kOnnxDomain, since_version, MakeRoiAlignDoc(),
      {
          {"X",
           "Input data tensor from the previous operator; "
           "4-D feature map of shape (N, C, H, W), "
           "where N is the batch size, C is the number of channels, "
           "and H and W are the height and the width of the data.",
           "T1"},
          {"rois",
           "RoIs (Regions of Interest) to pool over; rois is "
           "2-D input of shape (num_rois, 4) given as "
           "[[x1, y1, x2, y2], ...]. "
           "The RoIs' coordinates are in the coordinate system of the input image. "
           "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.",
           "T1"},
          {"batch_indices",
           "1-D tensor of shape (num_rois,) with each element denoting "
           "the index of the corresponding image in the batch.",
           "T2"},
      },
      {
          {"Y",
           "RoI pooled output, 4-D tensor of shape "
           "(num_rois, C, output_height, output_width). The r-th batch element Y[r-1] "
           "is a pooled feature map corresponding to the r-th RoI X[r-1].",
           "T1"},
      },
      {
          {"T1", RoiAlignFloatTypes(), "Constrain types to float tensors."},
          {"T2", {TensorType::kInt64}, "Constrain types to int tensors."},
      });
}

} // namespace

LightOpSchema MakeNonMaxSuppressionSchema(int since_version) {
  return LightOpSchema(
      "NonMaxSuppression", kOnnxDomain, since_version, MakeNonMaxSuppressionDoc(),
      {
          {"boxes",
           "An input tensor with shape [num_batches, spatial_dimension, 4]. The single box "
           "data format is indicated by center_point_box.",
           "tensor(float)"},
          {"scores", "An input tensor with shape [num_batches, num_classes, spatial_dimension]",
           "tensor(float)"},
          {"max_output_boxes_per_class",
           "Integer representing the maximum number of boxes to be selected per batch per "
           "class. It is a scalar. Default to 0, which means no output.",
           "tensor(int64)"},
          {"iou_threshold",
           since_version >= 11
               ? "Float representing the threshold for deciding whether boxes overlap too much "
                 "with respect to IOU. Boxes with IoU strictly greater than this threshold are "
                 "suppressed. It is scalar. Value range [0, 1]. Default to 0."
               : "Float representing the threshold for deciding whether boxes overlap too much "
                 "with respect to IOU. It is scalar. Value range [0, 1]. Default to 0.",
           "tensor(float)"},
          {"score_threshold",
           "Float representing the threshold for deciding when to remove boxes based on score. "
           "It is a scalar.",
           "tensor(float)"},
      },
      {
          {"selected_indices",
           "selected indices from the boxes tensor. [num_selected_indices, 3], the selected "
           "index format is [batch_index, class_index, box_index].",
           "tensor(int64)"},
      },
      // NonMaxSuppression uses literal type strings on inputs/outputs instead of named
      // type-parameter constraints, so no TypeConstraintParam entries are required.
      {},
      {
          AttributeParam{"center_point_box",
                         "Integer indicate the format of the box data. The default is 0. "
                         "0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and "
                         "(y2, x2) are the coordinates of any diagonal pair of box corners "
                         "and the coordinates can be provided as normalized (i.e., lying in the "
                         "interval [0, 1]) or absolute. Mostly used for TF models. "
                         "1 - the box data is supplied as [x_center, y_center, width, height]. "
                         "Mostly used for Pytorch models.",
                         AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      });
}

std::vector<LightOpSchema> GetAllOnnxOpObjectDetectionSchemasWithHistory(const std::string &op_type,
                                                                         bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"RoiAlign",
       [] {
         return std::vector<LightOpSchema>{
             MakeRoiAlignSchema(22),
             MakeRoiAlignSchema(16),
             MakeRoiAlignSchema(10),
         };
       }},
      {"NonMaxSuppression",
       [] {
         return std::vector<LightOpSchema>{
             MakeNonMaxSuppressionSchema(11),
             MakeNonMaxSuppressionSchema(10),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
