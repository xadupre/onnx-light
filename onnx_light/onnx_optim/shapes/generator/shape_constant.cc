// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

namespace {

// If ``tensor`` carries small integer content compatible with a shape
// value (rank ≤ 1, fewer than ``kConstantValueAsShapeMaxElements``
// elements, integer dtype), stamps it with a ``ValueAsShape``
// annotation reusing the integer values as dims.
void MaybeSetValueAsShapeFromTensorProto(OptimTensor &tensor, const TensorProto &tensor_proto) {
  if (!IsIntegerTensorType(tensor.Dtype())) {
    return;
  }
  if (tensor.Shape().Rank() > 1) {
    return;
  }
  int64_t count = 1;
  for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
    count *= tensor.Shape()[i].AsInt();
  }
  if (count < 0 || count >= kConstantValueAsShapeMaxElements) {
    return;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(tensor_proto, values)) {
    return;
  }
  // For a 0-D scalar the tensor proto stores a single element; for a
  // 1-D tensor the number of stored elements must match the dimension.
  if (static_cast<int64_t>(values.size()) != count) {
    return;
  }
  OptimShape value_shape;
  for (int64_t v : values) {
    value_shape.PushBack(OptimDim(v));
  }
  tensor.SetValueAsShape(std::move(value_shape));
}

// Counts how many of the supported Constant ``value*`` attributes are
// present on ``node``. Stores pointers to the first one of each kind
// in the corresponding out-parameter (``nullptr`` when absent).
int CollectConstantValueAttributes(
    const NodeProto &node, const AttributeProto *&value, const AttributeProto *&sparse_value,
    const AttributeProto *&value_int, const AttributeProto *&value_ints,
    const AttributeProto *&value_float, const AttributeProto *&value_floats,
    const AttributeProto *&value_string, const AttributeProto *&value_strings) {
  value = sparse_value = value_int = value_ints = value_float = value_floats = value_string =
      value_strings = nullptr;
  for (int i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    const std::string name = attr.name().as_string();
    if (name == "value" && value == nullptr) {
      value = &attr;
    } else if (name == "sparse_value" && sparse_value == nullptr) {
      sparse_value = &attr;
    } else if (name == "value_int" && value_int == nullptr) {
      value_int = &attr;
    } else if (name == "value_ints" && value_ints == nullptr) {
      value_ints = &attr;
    } else if (name == "value_float" && value_float == nullptr) {
      value_float = &attr;
    } else if (name == "value_floats" && value_floats == nullptr) {
      value_floats = &attr;
    } else if (name == "value_string" && value_string == nullptr) {
      value_string = &attr;
    } else if (name == "value_strings" && value_strings == nullptr) {
      value_strings = &attr;
    }
  }
  int count = 0;
  for (const AttributeProto *a : {value, sparse_value, value_int, value_ints, value_float,
                                  value_floats, value_string, value_strings}) {
    if (a != nullptr) {
      ++count;
    }
  }
  return count;
}

} // namespace

void ComputeShapeConstant(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Constant", "ComputeShapeConstant");

  const AttributeProto *value = nullptr;
  const AttributeProto *sparse_value = nullptr;
  const AttributeProto *value_int = nullptr;
  const AttributeProto *value_ints = nullptr;
  const AttributeProto *value_float = nullptr;
  const AttributeProto *value_floats = nullptr;
  const AttributeProto *value_string = nullptr;
  const AttributeProto *value_strings = nullptr;
  const int non_null =
      CollectConstantValueAttributes(node, value, sparse_value, value_int, value_ints, value_float,
                                     value_floats, value_string, value_strings);
  EXT_ENFORCE_INVALID(
      non_null == 1, "ComputeShapeConstant: exactly one of the attributes 'value', 'sparse_value', "
                     "'value_int', 'value_ints', 'value_float', 'value_floats', 'value_string' or "
                     "'value_strings' must be specified for a Constant node.");

  OptimTensor output;

  if (value != nullptr) {
    if (!value->has_t()) {
      throw std::invalid_argument(
          "ComputeShapeConstant: attribute 'value' must carry a tensor value.");
    }
    const TensorProto &tensor_proto = value->t();
    const TensorType dtype = DataTypeToTensorType(tensor_proto.data_type());
    OptimShape shape = ShapeFromTensorProtoDims(tensor_proto);
    output = OptimTensor(nullptr, dtype, std::move(shape));
    MaybeSetValueAsShapeFromTensorProto(output, tensor_proto);
  } else if (value_int != nullptr) {
    // Scalar INT64.
    output = OptimTensor(nullptr, TensorType::kInt64, OptimShape{});
    if (kConstantValueAsShapeMaxElements > 1) {
      OptimShape value_shape;
      value_shape.PushBack(OptimDim(value_int->i()));
      output.SetValueAsShape(std::move(value_shape));
    }
  } else if (value_ints != nullptr) {
    const auto &ints = value_ints->ints();
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(ints.size())));
    output = OptimTensor(nullptr, TensorType::kInt64, std::move(shape));
    if (static_cast<int64_t>(ints.size()) < kConstantValueAsShapeMaxElements) {
      OptimShape value_shape;
      for (int i = 0; i < ints.size(); ++i) {
        value_shape.PushBack(OptimDim(ints[i]));
      }
      output.SetValueAsShape(std::move(value_shape));
    }
  } else if (value_float != nullptr) {
    output = OptimTensor(nullptr, TensorType::kFloat, OptimShape{});
  } else if (value_floats != nullptr) {
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(value_floats->floats().size())));
    output = OptimTensor(nullptr, TensorType::kFloat, std::move(shape));
  } else if (value_string != nullptr) {
    output = OptimTensor(nullptr, TensorType::kString, OptimShape{});
  } else if (value_strings != nullptr) {
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(value_strings->strings().size())));
    output = OptimTensor(nullptr, TensorType::kString, std::move(shape));
  } else { // sparse_value
    EXT_ENFORCE_INVALID(
        !(!sparse_value->has_sparse_tensor()),
        "ComputeShapeConstant: attribute 'sparse_value' must carry a sparse tensor value.");
    const SparseTensorProto &sparse = sparse_value->sparse_tensor();
    const TensorType dtype = DataTypeToTensorType(sparse.values().data_type());
    OptimShape shape;
    for (int i = 0; i < sparse.dims().size(); ++i) {
      shape.PushBack(OptimDim(static_cast<int64_t>(sparse.dims()[i])));
    }
    output = OptimTensor(nullptr, dtype, std::move(shape));
  }

  ctx.Set(node.output(0), std::move(output));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
