// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <cstdint>
#include <vector>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceEmpty(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceEmpty", "ComputeShapeSequenceEmpty");

  // Resolve the optional 'dtype' attribute (an INT-valued
  // TensorProto::DataType). When absent the schema default is FLOAT.
  const AttributeProto *attr = FindAttribute(node, "dtype");
  TensorProto::DataType dtype = TensorProto::FLOAT;
  if (attr != nullptr) {
    EXT_ENFORCE_INVALID(attr->ref_type() == AttributeProto::AttributeType::INT,
                        "ComputeShapeSequenceEmpty: attribute 'dtype' must be of type INT.");
    dtype = static_cast<TensorProto::DataType>(attr->ref_i());
  }
  const TensorType elem_dtype = DataTypeToTensorType(dtype);

  // Output is a sequence of length 0 with the resolved element dtype
  // and an empty per-element shapes vector.
  ctx.SetSequence(node.output(0), OptimSequence(elem_dtype, std::vector<OptimShape>{}));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
