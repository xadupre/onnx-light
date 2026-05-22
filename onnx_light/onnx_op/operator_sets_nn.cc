// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_nn.h"
#include "onnx_op/operator_sets_nn_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace nn {

namespace {

const char *const kPoolingInputDescription =
    "Input data tensor from the previous operator; "
    "dimensions for image case are (N x C x H x W), "
    "where N is the batch size, C is the number of "
    "channels, and H and W are the height and the "
    "width of the data. For non image case, the "
    "dimensions are in the form of "
    "(N x C x D1 x D2 ... Dn), where N is the batch "
    "size. Optionally, if dimension denotation is "
    "in effect, the operation expects the input "
    "data tensor to arrive with the dimension denotation "
    "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].";

const char *const kPoolingOutputDescription =
    "Output data tensor from average or max pooling across "
    "the input tensor. Dimensions will vary based "
    "on various kernel, stride, and pad sizes. Floor value of "
    "the dimension is used";

std::vector<TensorType> AveragePoolTypes(int since_version) {
  if (since_version >= 22) {
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  }
  return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeAveragePoolSchema(int since_version) {
  return LightOpSchema("AveragePool", kOnnxDomain, since_version, MakeAveragePoolDoc(since_version),
                       {
                           {"X", kPoolingInputDescription, "T"},
                       },
                       {
                           {"Y", kPoolingOutputDescription, "T"},
                       },
                       {
                           {"T", AveragePoolTypes(since_version),
                            "Constrain input and output types to float tensors."},
                       });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpNnSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      MakeAveragePoolSchema(22), MakeAveragePoolSchema(19), MakeAveragePoolSchema(11),
      MakeAveragePoolSchema(10), MakeAveragePoolSchema(7),  MakeAveragePoolSchema(1),
  };
}

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
