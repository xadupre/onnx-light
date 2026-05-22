// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets.h"

#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_image.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_nn.h"
#include "onnx_op/operator_sets_object_detection.h"
#include "onnx_op/operator_sets_preview.h"
#include "onnx_op/operator_sets_quantization.h"
#include "onnx_op/operator_sets_reduction.h"
#include "onnx_op/operator_sets_sequence.h"
#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_text.h"
#include "onnx_op/operator_sets_traditionalml.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

std::vector<LightOpSchema> GetAllOnnxOpSchemasWithHistory(bool init_doc) {
  const std::vector<LightOpSchema> math_schemas =
      math::GetAllOnnxOpMathSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> controlflow_schemas =
      controlflow::GetAllOnnxOpControlflowSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> generator_schemas =
      generator::GetAllOnnxOpGeneratorSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> image_schemas =
      image::GetAllOnnxOpImageSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> logical_schemas =
      logical::GetAllOnnxOpLogicalSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> nn_schemas = nn::GetAllOnnxOpNnSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> object_detection_schemas =
      object_detection::GetAllOnnxOpObjectDetectionSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> preview_schemas =
      preview::GetAllOnnxOpPreviewSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> quantization_schemas =
      quantization::GetAllOnnxOpQuantizationSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> reduction_schemas =
      reduction::GetAllOnnxOpReductionSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> sequence_schemas =
      sequence::GetAllOnnxOpSequenceSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> tensor_schemas =
      tensor::GetAllOnnxOpTensorSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> text_schemas =
      text::GetAllOnnxOpTextSchemasWithHistory(init_doc);
  const std::vector<LightOpSchema> traditionalml_schemas =
      traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory(init_doc);

  std::vector<LightOpSchema> all_schemas;
  all_schemas.reserve(math_schemas.size() + controlflow_schemas.size() + generator_schemas.size() +
                      image_schemas.size() + logical_schemas.size() + nn_schemas.size() +
                      object_detection_schemas.size() + preview_schemas.size() +
                      quantization_schemas.size() + reduction_schemas.size() +
                      sequence_schemas.size() + tensor_schemas.size() + text_schemas.size() +
                      traditionalml_schemas.size());

  all_schemas.insert(all_schemas.end(), math_schemas.begin(), math_schemas.end());
  all_schemas.insert(all_schemas.end(), controlflow_schemas.begin(), controlflow_schemas.end());
  all_schemas.insert(all_schemas.end(), generator_schemas.begin(), generator_schemas.end());
  all_schemas.insert(all_schemas.end(), image_schemas.begin(), image_schemas.end());
  all_schemas.insert(all_schemas.end(), logical_schemas.begin(), logical_schemas.end());
  all_schemas.insert(all_schemas.end(), nn_schemas.begin(), nn_schemas.end());
  all_schemas.insert(all_schemas.end(), object_detection_schemas.begin(),
                     object_detection_schemas.end());
  all_schemas.insert(all_schemas.end(), preview_schemas.begin(), preview_schemas.end());
  all_schemas.insert(all_schemas.end(), quantization_schemas.begin(), quantization_schemas.end());
  all_schemas.insert(all_schemas.end(), reduction_schemas.begin(), reduction_schemas.end());
  all_schemas.insert(all_schemas.end(), sequence_schemas.begin(), sequence_schemas.end());
  all_schemas.insert(all_schemas.end(), tensor_schemas.begin(), tensor_schemas.end());
  all_schemas.insert(all_schemas.end(), text_schemas.begin(), text_schemas.end());
  all_schemas.insert(all_schemas.end(), traditionalml_schemas.begin(), traditionalml_schemas.end());
  return all_schemas;
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
