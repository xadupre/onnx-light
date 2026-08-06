// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_training.h"
#include "onnx_op/operator_sets_training_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_op::training {

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
                       },
                       {
                           {"xs",
                            "Input tensor variables' names. The variables identified by all "
                            "xs and zs will be inputs to the (sub)graph specified by this op. "
                            "The value of this attribute determines the number of inputs of "
                            "Gradient. Please note that xs and zs together define all inputs "
                            "of the concerned graph.",
                            AttributeType::STRINGS, true, std::monostate{}},
                           {"y",
                            "The targeted tensor. It must be one of the outputs of the "
                            "subgraph.",
                            AttributeType::STRING, true, std::monostate{}},
                           {"zs",
                            "Input tensor variables' names. The variables identified by all "
                            "xs and zs will be inputs to the (sub)graph specified by this op. "
                            "The value of this attribute determines the number of inputs of "
                            "Gradient. Please note that xs and zs together define all inputs "
                            "of the concerned graph.",
                            AttributeType::STRINGS, false, std::monostate{}},
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
      },
      {
          {"alpha",
           "Coefficient of previously accumulated gradient in running average. Default to 0.9.",
           AttributeType::FLOAT, false, double{0.9}},
          {"beta",
           "Coefficient of previously accumulated squared-gradient in running average. "
           "Default to 0.999.",
           AttributeType::FLOAT, false, double{0.999}},
          {"epsilon", "Small scalar to avoid dividing by 0. Default to 1e-6.", AttributeType::FLOAT,
           false, double{1e-6}},
          {"norm_coefficient",
           "Regularization coefficient in 0.5 * norm_coefficient * ||X||_F^2. Default to 0, "
           "which means no regularization.",
           AttributeType::FLOAT, false, double{0.0}},
          {"norm_coefficient_post",
           "Regularization coefficient post computation, i.e., "
           "grad = grad + norm_coefficient_post * X. Default to 0, "
           "which means no regularization.",
           AttributeType::FLOAT, false, double{0.0}},
      });
}

LightOpSchema MakeAdagradSchema() {
  return LightOpSchema(
      "Adagrad", kOnnxPreviewTrainingDomain, 1, MakeAdagradDoc(),
      {
          {"R", "The initial learning rate.", "T1"},
          {"T", "The update count of \"X\". It should be a scalar.", "T2"},
          {"inputs",
           "The current values of optimized tensors, followed by their "
           "respective gradients, followed by their respective accumulated squared gradients."
           "For example, if two tensor \"X_1\" and \"X_2\" "
           "are optimized, "
           "The input list would be "
           "[\"X_1\", \"X_2\", "
           "gradient of \"X_1\", "
           "gradient of \"X_2\", "
           "accumulated squared gradient of \"X_1\", "
           "accumulated squared gradient of \"X_2\"].",
           "T3"},
      },
      {
          {"outputs",
           "Updated values of optimized tensors, followed by their updated "
           "values of accumulated squared gradients. For example, "
           "if two tensor \"X_1\" and \"X_2\" are "
           "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
           "new accumulated squared gradient of \"X_1\", new accumulated squared gradient of "
           "\"X_2\"].",
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
      },
      {
          {"decay_factor",
           "Coefficient of previously accumulated squared-gradient in running average. "
           "Default to 0, which means no decay.",
           AttributeType::FLOAT, false, double{0.0}},
          {"epsilon", "Small scalar to avoid dividing by 0. Default to 1e-6.", AttributeType::FLOAT,
           false, double{1e-6}},
          {"norm_coefficient",
           "Regularization coefficient in 0.5 * norm_coefficient * ||X||_F^2. Default to 0, "
           "which means no regularization.",
           AttributeType::FLOAT, false, double{0.0}},
      });
}

LightOpSchema MakeMomentumSchema() {
  return LightOpSchema(
      "Momentum", kOnnxPreviewTrainingDomain, 1, MakeMomentumDoc(),
      {
          {"R", "The learning rate.", "T1"},
          {"T", "Update count of \"X\". It should be a scalar.", "T2"},
          {"inputs",
           "It sequentially contains the current values of optimized tensors, then their "
           "gradient tensors, and finally their momentum tensors. For example, if two tensors "
           "\"X_1\" and \"X_2\" are optimized, The expected input list would be "
           "[\"X_1\", \"X_2\", gradient of \"X_1\", gradient of \"X_2\", momentum of \"X_1\", "
           "momentum of \"X_2\"].",
           "T3"},
      },
      {
          {"outputs",
           "It sequentially contains the new values of optimized tensors and then the new "
           "values of their momentum tensors. For example, if two tensors \"X_1\" and \"X_2\" are "
           "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
           "new momentum of \"X_1\", new momentum of \"X_2\"].",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kDouble},
           "Constrain input types to float scalars."},
          {"T2", {TensorType::kInt64}, "Constrain input types to 64-bit integer scalars."},
          {"T3",
           {TensorType::kFloat, TensorType::kDouble},
           "Constrain input types to float tensors."},
      },
      {
          {"alpha",
           "Coefficient of previously accumulated gradient in running average. This attribute "
           "must be specified.",
           AttributeType::FLOAT, true, std::monostate{}},
          {"beta", "Dampening factor for updated gradient. This attribute must be specified.",
           AttributeType::FLOAT, true, std::monostate{}},
          {"mode",
           "Momentum algorithm to be used. It must be one of \"nesterov\" and \"standard\". "
           "This attribute must be specified.",
           AttributeType::STRING, true, std::monostate{}},
          {"norm_coefficient",
           "Regularization coefficient in 0.5 * norm_coefficient * ||X||_F^2. This attribute "
           "must be specified.",
           AttributeType::FLOAT, true, std::monostate{}},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTrainingSchemasWithHistory(const std::string &op_type,
                                                                  bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"Adagrad", [] { return std::vector<LightOpSchema>{MakeAdagradSchema()}; }},
      {"Adam", [] { return std::vector<LightOpSchema>{MakeAdamSchema()}; }},
      {"Gradient", [] { return std::vector<LightOpSchema>{MakeGradientSchema()}; }},
      {"Momentum", [] { return std::vector<LightOpSchema>{MakeMomentumSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::training
