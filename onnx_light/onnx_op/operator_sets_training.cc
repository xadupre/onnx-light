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

LightOpSchema MakeAdamSchema() {
  return LightOpSchema(
      "Adam", kOnnxPreviewTrainingDomain, 1, MakeAdamDoc(),
      {
          {"R", "The initial learning rate.", "T1"},
          {"T", "The update count of \"X\". It should be a scalar.", "T2"},
          {"inputs",
           "The tensors to be optimized, followed by their respective gradients, "
           "followed by their respective accumulated gradients (aka momentum), "
           "followed by their respective accumulated squared gradients. For example, "
           "to optimize tensors \"X_1\" and \"X_2,\", the input list would be "
           "[\"X_1\", \"X_2\", "
           "gradient of \"X_1\", gradient of \"X_2\", "
           "accumulated gradient of \"X_1\", accumulated gradient of \"X_2\", "
           "accumulated squared gradient of \"X_1\", accumulated squared gradient of \"X_2\"].",
           "T3"},
      },
      {
          {"outputs",
           "New values of optimized tensors, "
           "followed by their respective new accumulated gradients, "
           "followed by their respective new accumulated squared gradients. "
           "For example, if two tensors \"X_1\" and \"X_2\" are optimized, "
           "the outputs list would be "
           "[new value of \"X_1\", new value of \"X_2\", "
           "new accumulated gradient of \"X_1\", "
           "new accumulated gradient of \"X_2\", "
           "new accumulated squared gradient of \"X_1\", "
           "new accumulated squared gradient of \"X_2\"].",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kDouble},
           "Constrain input types to float scalars."},
          {"T2", {TensorType::kInt64}, "Constrain input types to 64-bit integer scalars."},
          {"T3",
           {TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTrainingSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeAdamSchema(),
      MakeGradientSchema(),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace training
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
