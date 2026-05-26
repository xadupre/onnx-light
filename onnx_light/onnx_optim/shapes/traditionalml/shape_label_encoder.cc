// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

// Counts how many of the supported LabelEncoder ``values_*`` attributes
// are present on ``node``. Stores pointers to the first one of each
// kind in the corresponding out-parameter (``nullptr`` when absent).
int CollectLabelEncoderValueAttributes(const NodeProto &node, const AttributeProto *&values_tensor,
                                       const AttributeProto *&values_strings,
                                       const AttributeProto *&values_int64s,
                                       const AttributeProto *&values_floats) {
  values_tensor = values_strings = values_int64s = values_floats = nullptr;
  for (int i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    const std::string name = attr.name().as_string();
    if (name == "values_tensor" && values_tensor == nullptr) {
      values_tensor = &attr;
    } else if (name == "values_strings" && values_strings == nullptr) {
      values_strings = &attr;
    } else if (name == "values_int64s" && values_int64s == nullptr) {
      values_int64s = &attr;
    } else if (name == "values_floats" && values_floats == nullptr) {
      values_floats = &attr;
    }
  }
  int count = 0;
  for (const AttributeProto *a : {values_tensor, values_strings, values_int64s, values_floats}) {
    if (a != nullptr) {
      ++count;
    }
  }
  return count;
}

} // namespace

void ComputeShapeLabelEncoder(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LabelEncoder", "ComputeShapeLabelEncoder");

  const AttributeProto *values_tensor = nullptr;
  const AttributeProto *values_strings = nullptr;
  const AttributeProto *values_int64s = nullptr;
  const AttributeProto *values_floats = nullptr;
  const int non_null = CollectLabelEncoderValueAttributes(node, values_tensor, values_strings,
                                                          values_int64s, values_floats);
  if (non_null != 1) {
    throw std::invalid_argument(
        "ComputeShapeLabelEncoder: exactly one of the attributes 'values_tensor', "
        "'values_strings', 'values_int64s' or 'values_floats' must be specified for a "
        "LabelEncoder node.");
  }

  TensorType value_type = TensorType::kUndefined;
  if (values_tensor != nullptr) {
    if (!values_tensor->has_t()) {
      throw std::invalid_argument(
          "ComputeShapeLabelEncoder: attribute 'values_tensor' must carry a tensor value.");
    }
    value_type = DataTypeToTensorType(values_tensor->ref_t().data_type());
  } else if (values_strings != nullptr) {
    value_type = TensorType::kString;
  } else if (values_int64s != nullptr) {
    value_type = TensorType::kInt64;
  } else { // values_floats
    value_type = TensorType::kFloat;
  }

  // LabelEncoder is a one-to-one mapping: the output shape always
  // matches the input shape, only the dtype is determined by the
  // selected ``values_*`` attribute.
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, value_type, input.Shape()));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
