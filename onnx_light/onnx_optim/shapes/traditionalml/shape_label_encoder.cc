// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

// Returns the resolved output ``TensorType`` for every ``values_*``
// attribute present on ``node``. The vector size therefore matches the
// number of (mutually exclusive) value attributes the caller specified
// — exactly one is expected. Throws ``std::invalid_argument`` when a
// ``values_tensor`` attribute is present but does not carry a tensor
// value.
std::vector<TensorType> CollectLabelEncoderValueTypes(const NodeProto &node) {
  std::vector<TensorType> types;
  for (int i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    const std::string name = attr.name().as_string();
    if (name == "values_tensor") {
      if (!attr.has_t()) {
        throw std::invalid_argument(
            "ComputeShapeLabelEncoder: attribute 'values_tensor' must carry a tensor value.");
      }
      types.push_back(DataTypeToTensorType(attr.ref_t().data_type()));
    } else if (name == "values_strings") {
      types.push_back(TensorType::kString);
    } else if (name == "values_int64s") {
      types.push_back(TensorType::kInt64);
    } else if (name == "values_floats") {
      types.push_back(TensorType::kFloat);
    }
  }
  return types;
}

} // namespace

void ComputeShapeLabelEncoder(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LabelEncoder", "ComputeShapeLabelEncoder");

  const std::vector<TensorType> value_types = CollectLabelEncoderValueTypes(node);
  if (value_types.size() != 1) {
    throw std::invalid_argument(
        "ComputeShapeLabelEncoder: exactly one of the attributes 'values_tensor', "
        "'values_strings', 'values_int64s' or 'values_floats' must be specified for a "
        "LabelEncoder node.");
  }

  // LabelEncoder is a one-to-one mapping: the output shape always
  // matches the input shape, only the dtype is determined by the
  // selected ``values_*`` attribute.
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, value_types[0], input.Shape()));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
