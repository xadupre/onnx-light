// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx_lib/defs/doc_strings.h"
#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace {

std::vector<std::string> TensorAndSequenceTypes(bool ir14) {
  auto types = ir14 ? OpSchema::all_tensor_types_ir14() : OpSchema::all_tensor_types();
  const auto &sequence_types =
      ir14 ? OpSchema::all_tensor_sequence_types_ir14() : OpSchema::all_tensor_sequence_types();
  types.insert(types.end(), sequence_types.begin(), sequence_types.end());
  return types;
}

std::vector<std::string> OptionalTypes(const std::vector<std::string> &types) {
  std::vector<std::string> optional_types;
  optional_types.reserve(types.size());
  for (const auto &type : types) {
    optional_types.emplace_back("optional(" + type + ")");
  }
  return optional_types;
}

std::vector<std::string> OptionalAndTensorTypes(bool ir14) {
  const auto types = TensorAndSequenceTypes(ir14);
  auto optional_types = ir14 ? OptionalTypes(types) : OpSchema::all_optional_types();
  optional_types.insert(optional_types.end(), types.begin(), types.end());
  return optional_types;
}

void OptionalInference(InferenceContext &ctx) {
  if (ctx.getNumOutputs() != 1) {
    fail_type_inference("Optional is expected to have an output.");
  }

  const size_t num_inputs = ctx.getNumInputs();
  const auto *attr_proto = ctx.getAttribute("type");
  if (num_inputs == 0 && attr_proto != nullptr) {
    if (!attr_proto->has_tp()) {
      fail_type_inference("Attribute 'type' should be a TypeProto and it should specify a type.");
    }
    ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(attr_proto->tp());
  } else if (num_inputs == 1) {
    const auto *input_type = ctx.getInputType(0);
    if (input_type == nullptr) {
      fail_type_inference("Input type is null. Type information is expected for the input.");
    }
    ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(*input_type);
  } else {
    fail_type_inference("Optional is expected to have either an input or the type attribute set.");
  }
}

void OptionalHasElementInference(InferenceContext &ctx) {
  const size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 0 && num_inputs != 1) {
    fail_type_inference("OptionalHasElement is expected to have 0 or 1 input.");
  }
  if (ctx.getNumOutputs() != 1) {
    fail_type_inference("OptionalHasElement is expected to have 1 output.");
  }
  auto *output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_tensor_type->set_elem_type(TensorProto::BOOL);
  output_tensor_type->mutable_shape()->Clear();
}

void OptionalGetElementInference(InferenceContext &ctx) {
  if (ctx.getNumInputs() != 1) {
    fail_type_inference("OptionalGetElement must have an input element.");
  }
  const auto *input_type = ctx.getInputType(0);
  if (input_type == nullptr) {
    fail_type_inference("Input type is null. Input must have Type information.");
  }
  if (input_type->has_optional_type()) {
    if (!input_type->optional_type().has_elem_type()) {
      fail_type_inference("Optional-type input must contain an element with type information.");
    }
    ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
  } else {
    propagateShapeAndTypeFromFirstInput(ctx);
  }
}
OpSchema MakeOptionalSchema(bool ir14) {
  const auto types = TensorAndSequenceTypes(ir14);
  return OpSchema()
      .SetDoc(kDoc_Optional_ver15)
      .Input(0, "input", "The input element.", "V", OpSchema::Optional)
      .Attr("type", "Type of the element in the optional output", AttributeProto::TYPE_PROTO,
            OPTIONAL_VALUE)
      .Output(0, "output", "The optional output enclosing the input element.", "O")
      .TypeConstraint("V", types, "Constrain input type to all tensor and sequence types.")
      .TypeConstraint("O", ir14 ? OptionalTypes(types) : OpSchema::all_optional_types(),
                      "Constrain output type to all optional tensor or optional sequence types.")
      .TypeAndShapeInferenceFunction(OptionalInference);
}

OpSchema MakeOptionalHasElementSchema(bool ir14) {
  return OpSchema()
      .SetDoc(kDoc_OptionalHasElement_ver18)
      .Input(0, "input", "The optional input.", "O", OpSchema::Optional)
      .Output(0, "output",
              "A scalar boolean tensor. If true, it indicates that optional-type input contains "
              "an element. Otherwise, it is empty.",
              "B")
      .TypeConstraint("O", OptionalAndTensorTypes(ir14),
                      "Constrain input type to optional, tensor and sequence types.")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
      .TypeAndShapeInferenceFunction(OptionalHasElementInference);
}

OpSchema MakeOptionalGetElementSchema(bool ir14) {
  const auto types = TensorAndSequenceTypes(ir14);
  auto input_types = ir14 ? OptionalTypes(types) : OpSchema::all_optional_types();
  input_types.insert(input_types.end(), types.begin(), types.end());
  return OpSchema()
      .SetDoc(kDoc_OptionalGetElement_ver18)
      .Input(0, "input", "The optional input.", "O")
      .Output(0, "output", "Output element in the optional input.", "V")
      .TypeConstraint("O", input_types,
                      "Constrain input type to optional, tensor and sequence types.")
      .TypeConstraint("V", types, "Constrain output type to all tensor or sequence types.")
      .TypeAndShapeInferenceFunction(OptionalGetElementInference);
}

} // namespace

ONNX_OPERATOR_SET_SCHEMA(Optional, 15, MakeOptionalSchema(false));
ONNX_OPERATOR_SET_SCHEMA(OptionalHasElement, 18, MakeOptionalHasElementSchema(false));
ONNX_OPERATOR_SET_SCHEMA(OptionalGetElement, 18, MakeOptionalGetElementSchema(false));

ONNX_OPERATOR_SET_SCHEMA(Optional, 28, MakeOptionalSchema(true));
ONNX_OPERATOR_SET_SCHEMA(OptionalHasElement, 28, MakeOptionalHasElementSchema(true));
ONNX_OPERATOR_SET_SCHEMA(OptionalGetElement, 28, MakeOptionalGetElementSchema(true));

} // namespace ONNX_LIGHT_NAMESPACE
