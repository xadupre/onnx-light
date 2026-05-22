// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_training.h"
#include "onnx_op/operator_sets_training_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace training {

namespace {

LightOpSchema MakeGradientSchema() {
  return LightOpSchema("Gradient", kOnnxPreviewTrainingDomain, 1, MakeGradientDoc(),
                       {
                           {"Inputs",
                            "The values fed into graph identified by the attributes. "
                            "The i-th input is the value of the i-th tensor specified in the "
                            "concatenated list of the attribute \"xs\" and the attribute "
                            " \"zs\". For example, if xs=[\"A\", \"B\"] and zs=[\"C\"], the "
                            "first input is used as the value of symbol \"A\" and the 3rd "
                            "input is substituted for all the occurrences of \"C\".",
                            "T1"},
                       },
                       {
                           {"Outputs",
                            "The gradient of the tensor specified by the attribute \"y\" "
                            "with respect to each of tensors specified in the "
                            "attribute \"xs\". The i-th output is the gradient of \"y\" with "
                            "respect to the i-th tensor specified in the attribute \"xs\".",
                            "T2"},
                       },
                       {
                           {"T1", AllTensorTypes(), "Allow outputs to be any kind of tensor."},
                           {"T2",
                            {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
                            "Allow inputs to be any kind of floating-point tensor."},
                       });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTrainingSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeGradientSchema(),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace training
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
