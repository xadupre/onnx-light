// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx_lib/defs/doc_strings.h"
#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

inline static void unaryLogicalOpInference(InferenceContext &ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

static void binaryLogicalOpInference(InferenceContext &ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(), ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::function<void(OpSchema &)> BinaryLogicDocGenerator(const char *name) {
  return [=](OpSchema &schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First input operand for the logical operator.", "T", OpSchema::Single,
                 true, 1, OpSchema::NonDifferentiable);
    schema.Input(1, "B", "Second input operand for the logical operator.", "T", OpSchema::Single,
                 true, 1, OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T1", OpSchema::Single, true, 1,
                  OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction(binaryLogicalOpInference);
  };
}

ONNX_OPERATOR_SET_SCHEMA(And, 7,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("and"))
                             .TypeConstraint("T", {"tensor(bool)"},
                                             "Constrain input to boolean tensor.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(Or, 7,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("or"))
                             .TypeConstraint("T", {"tensor(bool)"},
                                             "Constrain input to boolean tensor.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(Xor, 7,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("xor"))
                             .TypeConstraint("T", {"tensor(bool)"},
                                             "Constrain input to boolean tensor.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(Greater, 13,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("greater"))
                             .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                                             "Constrain input types to all numeric tensors.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(Less, 13,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("less"))
                             .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                                             "Constrain input types to all numeric tensors.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal, 19,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("equal"))
        .TypeConstraint("T",
                        {"tensor(bool)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                         "tensor(uint64)", "tensor(int8)", "tensor(int16)", "tensor(int32)",
                         "tensor(int64)", "tensor(float16)", "tensor(float)", "tensor(double)",
                         "tensor(bfloat16)", "tensor(string)"},
                        "Constrain input types to all (non-complex) tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(Not, 1,
                         OpSchema()
                             .SetDoc(kDoc_Not_ver1)
                             .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1,
                                    OpSchema::NonDifferentiable)
                             .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1,
                                     OpSchema::NonDifferentiable)
                             .TypeConstraint("T", {"tensor(bool)"},
                                             "Constrain input/output to boolean tensors.")
                             .TypeAndShapeInferenceFunction(unaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    BitShift, 28,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(kDoc_BitShift_ver28) + GenerateBroadcastingDocMul()))
        .Input(0, "X", "First operand, input to be shifted.", "T", OpSchema::Single, true, 1,
               OpSchema::NonDifferentiable)
        .Input(1, "Y", "Second operand, amounts of shift.", "T", OpSchema::Single, true, 1,
               OpSchema::NonDifferentiable)
        .Output(0, "Z", "Output tensor", "T", OpSchema::Single, true, 1,
                OpSchema::NonDifferentiable)
        .TypeConstraint("T",
                        {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
                         "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"},
                        "Constrain input and output types to integer tensors.")
        .Attr("direction",
              "Direction of moving bits. It can be either \"RIGHT\" (for right shift) "
              "or \"LEFT\" (for left shift).",
              AttributeProto::STRING)
        .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

ONNX_OPERATOR_SET_SCHEMA(LessOrEqual, 16,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("less_equal"))
                             .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                                             "Constrain input types to all numeric tensors.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor.")
                             .TypeAndShapeInferenceFunction(binaryLogicalOpInference)
                             .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(GreaterOrEqual, 16,
                         OpSchema()
                             .FillUsing(BinaryLogicDocGenerator("greater_equal"))
                             .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                                             "Constrain input types to all numeric tensors.")
                             .TypeConstraint("T1", {"tensor(bool)"},
                                             "Constrain output to boolean tensor.")
                             .TypeAndShapeInferenceFunction(binaryLogicalOpInference)
                             .FunctionBody(R"ONNX(
        {
            O1 = Greater (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseNot, 18,
    OpSchema()
        .SetDoc(kDoc_BitwiseNot_ver18)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1,
                OpSchema::NonDifferentiable)
        .TypeConstraint("T",
                        {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
                         "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"},
                        "Constrain input/output to integer tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema &)> BinaryBitwiseDocGenerator(const char *name) {
  return [=](OpSchema &schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulting from performing the bitwise `{name}` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First input operand for the bitwise operator.", "T", OpSchema::Single,
                 true, 1, OpSchema::NonDifferentiable);
    schema.Input(1, "B", "Second input operand for the bitwise operator.", "T", OpSchema::Single,
                 true, 1, OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T", OpSchema::Single, true, 1,
                  OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(), ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(BitwiseAnd, 18,
                         OpSchema()
                             .FillUsing(BinaryBitwiseDocGenerator("and"))
                             .TypeConstraint("T",
                                             {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                                              "tensor(uint64)", "tensor(int8)", "tensor(int16)",
                                              "tensor(int32)", "tensor(int64)"},
                                             "Constrain input to integer tensors."));

ONNX_OPERATOR_SET_SCHEMA(BitwiseOr, 18,
                         OpSchema()
                             .FillUsing(BinaryBitwiseDocGenerator("or"))
                             .TypeConstraint("T",
                                             {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                                              "tensor(uint64)", "tensor(int8)", "tensor(int16)",
                                              "tensor(int32)", "tensor(int64)"},
                                             "Constrain input to integer tensors."));

ONNX_OPERATOR_SET_SCHEMA(BitwiseXor, 18,
                         OpSchema()
                             .FillUsing(BinaryBitwiseDocGenerator("xor"))
                             .TypeConstraint("T",
                                             {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
                                              "tensor(uint64)", "tensor(int8)", "tensor(int16)",
                                              "tensor(int32)", "tensor(int64)"},
                                             "Constrain input to integer tensors."));

} // namespace ONNX_LIGHT_NAMESPACE
