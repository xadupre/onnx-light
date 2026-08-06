// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

void ComputeShapeSequenceEmpty(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceEmpty", "ComputeShapeSequenceEmpty");

  // Resolve the optional 'dtype' attribute (an INT-valued
  // TensorProto::DataType). When absent the schema default is FLOAT.
  const AttributeProto *attr = FindAttribute(node, "dtype");
  TensorProto::DataType dtype = TensorProto::FLOAT;
  if (attr != nullptr) {
    EXT_ENFORCE_INVALID(attr->has_type() && attr->ref_type() == AttributeProto::AttributeType::INT,
                        "ComputeShapeSequenceEmpty: attribute 'dtype' must be of type INT.");
    dtype = static_cast<TensorProto::DataType>(attr->ref_i());
  }
  const TensorType elem_dtype = DataTypeToTensorType(dtype);

  // Output is a sequence of length 0 with the resolved element dtype
  // and an empty per-element shapes vector.
  ctx.SetSequence(node.output(0), SymSequence(elem_dtype, std::vector<SymShape>{}));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
