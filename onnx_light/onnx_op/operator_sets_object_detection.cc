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

std::vector<LightOpSchema>
GetAllOnnxOpObjectDetectionSchemasWithHistory(bool init_doc, const std::string &op_type) {
  std::vector<LightOpSchema> schemas{
      MakeRoiAlignSchema(22),
      MakeRoiAlignSchema(16),
      MakeRoiAlignSchema(10),
  };
  return FilterSchemasByOpType(init_doc ? schemas : StripDocs(schemas), op_type);
}

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
