// Auto-generated: minimal ONNX operator schema registration.
// This file registers just name/domain/version for every operator in the
// built-in opsets without requiring the full operator defs source files.
//
// DO NOT EDIT – regenerate from onnx_light/onnx/defs/operator_sets*.h

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {

void RegisterAllOnnxOperatorSchemas() {
  // ------------------------------------------------------------------
  // Register commonly-used operators with onnx_light-compatible
  // type-and-shape inference functions FIRST, so that the skeleton
  // loop below sees them as duplicates and skips those entries.
  // ------------------------------------------------------------------
  auto reg_infer = [](OpSchema schema) { RegisterSchema(std::move(schema), 0, false, false); };

  // Helper: propagate elem_type from input[0] to output[0].
  auto prop_in0 = [](InferenceContext &ctx) { propagateShapeAndTypeFromFirstInput(ctx); };

  // Helper: propagate elem_type from input[0] to output[0] (type only).
  auto prop_type_in0 = [](InferenceContext &ctx) { propagateElemTypeFromInputToOutput(ctx, 0, 0); };

  // Helper: binary op – propagate elem_type from input[0] to output[0], validate type match.
  auto binary_type_prop = [](InferenceContext &ctx) {
    const auto *in0 = ctx.getInputType(0);
    const auto *in1 = ctx.getInputType(1);
    if (in0 && in0->has_tensor_type() && in1 && in1->has_tensor_type()) {
      if (in0->tensor_type().has_elem_type() && in1->tensor_type().has_elem_type() &&
          in0->tensor_type().elem_type() != in1->tensor_type().elem_type()) {
        fail_type_inference("Binary op: input type mismatch.");
      }
    }
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (hasNInputShapes(ctx, 2)) {
      bidirectionalBroadcastShapeInference(
          ctx.getInputType(0)->tensor_type().shape(), ctx.getInputType(1)->tensor_type().shape(),
          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }
  };

  // Add / Sub / Mul / Div (versions 13 and 14 – both needed for opset lookups)
  for (const char *name : {"Add", "Sub", "Mul", "Div"}) {
    for (int ver : {13, 14}) {
      reg_infer(OpSchema()
                    .SetName(name)
                    .SetDomain(ONNX_DOMAIN)
                    .SinceVersion(ver)
                    .SetDoc("Performs element-wise binary operation on two input tensors with "
                            "Numpy-style broadcasting support.")
                    .Input(0, "A", "First operand.", "T")
                    .Input(1, "B", "Second operand.", "T")
                    .Output(0, "C", "Result tensor.", "T")
                    .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                                    "Constrain input and output types to all numeric tensors.")
                    .TypeAndShapeInferenceFunction(binary_type_prop));
    }
  }

  // Cast: the 'to' attribute determines the output type.
  // Register for all versions where Cast appears so GetSchema always finds one.
  for (int ver : {1, 6, 9, 13, 19, 21, 23, 24, 25}) {
    reg_infer(OpSchema()
                  .SetName("Cast")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .SetDoc("Casts the elements of an input tensor to a specified data type.")
                  .Attr("to", "The data type to which the elements of the input tensor are cast.",
                        AttributeProto::INT)
                  .Input(0, "input", "Input tensor to be cast.", "T1")
                  .Output(0, "output", "Output tensor with the same shape as input.", "T2")
                  .TypeConstraint("T1", OpSchema::all_non_complex_tensor_types_ir13(),
                                  "Constrain input types. Casting from complex is not supported.")
                  .TypeConstraint("T2", OpSchema::all_non_complex_tensor_types_ir13(),
                                  "Constrain output types. Casting to complex is not supported.")
                  .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                    const auto *to_attr = ctx.getAttribute("to");
                    if (to_attr && to_attr->type() == AttributeProto::INT) {
                      updateOutputElemType(ctx, 0, static_cast<int32_t>(to_attr->i()));
                    }
                    if (hasInputShape(ctx, 0)) {
                      updateOutputShape(ctx, 0, getInputShape(ctx, 0));
                    }
                  }));
  }

  // Clip (pass-through type) – register all known versions
  // x, min (optional), max (optional) must all have the same type.
  auto clip_infer = [](InferenceContext &ctx) {
    const auto *in0 = ctx.getInputType(0);
    if (!in0 || !in0->has_tensor_type()) {
      return;
    }
    // Validate that any non-null optional inputs share the same element type.
    for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
      const auto *ini = ctx.getInputType(i);
      if (!ini || !ini->has_tensor_type()) {
        continue; // optional input absent – OK
      }
      if (ini->tensor_type().has_elem_type() && in0->tensor_type().has_elem_type() &&
          ini->tensor_type().elem_type() != in0->tensor_type().elem_type()) {
        fail_type_inference("Clip: all inputs must have the same element type.");
      }
    }
    propagateShapeAndTypeFromFirstInput(ctx);
  };
  for (int ver : {1, 6, 11, 12, 13}) {
    reg_infer(OpSchema()
                  .SetName("Clip")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction(clip_infer));
  }

  // ReduceMax (versions 13, 18, 20 – propagate type and shape from first input)
  for (int ver : {13, 18, 20}) {
    reg_infer(OpSchema()
                  .SetName("ReduceMax")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction(prop_in0));
  }

  // QuantizeLinear: output type comes from zero_point (input 2) if present, else uint8
  for (int ver : {13, 19, 21, 23, 24, 25}) {
    reg_infer(OpSchema()
                  .SetName("QuantizeLinear")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                    const auto *zp_type = ctx.getInputType(2);
                    if (zp_type && zp_type->has_tensor_type() &&
                        zp_type->tensor_type().has_elem_type()) {
                      propagateElemTypeFromInputToOutput(ctx, 2, 0);
                    } else {
                      updateOutputElemType(ctx, 0, static_cast<int32_t>(TensorProto::UINT8));
                    }
                    if (hasInputShape(ctx, 0)) {
                      updateOutputShape(ctx, 0, getInputShape(ctx, 0));
                    }
                  }));
  }

  // Concat (version 13 – shape inference with axis handling)
  reg_infer(OpSchema()
                .SetName("Concat")
                .SetDomain(ONNX_DOMAIN)
                .SinceVersion(13)
                .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                  const auto *axis_attr = ctx.getAttribute("axis");
                  if (!axis_attr || !axis_attr->has_i()) {
                    return;
                  }
                  const auto *first_type = ctx.getInputType(0);
                  auto *output_type = ctx.getOutputType(0);
                  if (!first_type || !output_type || !first_type->has_tensor_type()) {
                    return;
                  }
                  output_type->tensor_type().set_elem_type(first_type->tensor_type().elem_type());
                  if (!first_type->tensor_type().has_shape()) {
                    return;
                  }
                  const auto &first_dims = first_type->tensor_type().shape().dim();
                  const int64_t rank = static_cast<int64_t>(first_dims.size());
                  int64_t axis = axis_attr->i();
                  if (axis < 0) {
                    axis += rank;
                  }
                  if (axis < 0 || axis >= rank) {
                    return;
                  }
                  bool axis_concrete = true;
                  int64_t axis_total = 0;
                  for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
                    const auto *in_type = ctx.getInputType(i);
                    if (!in_type || !in_type->has_tensor_type() ||
                        !in_type->tensor_type().has_shape()) {
                      axis_concrete = false;
                      break;
                    }
                    const auto &in_dims = in_type->tensor_type().shape().dim();
                    if (static_cast<int64_t>(in_dims.size()) <= axis) {
                      axis_concrete = false;
                      break;
                    }
                    const auto &in_dim = in_dims[static_cast<size_t>(axis)];
                    if (in_dim.has_dim_value()) {
                      axis_total += in_dim.dim_value();
                    } else {
                      axis_concrete = false;
                      break;
                    }
                  }
                  auto *out_shape = output_type->tensor_type().mutable_shape();
                  for (int64_t d = 0; d < rank; ++d) {
                    auto *out_dim = out_shape->add_dim();
                    if (d == axis) {
                      if (axis_concrete) {
                        out_dim->set_dim_value(axis_total);
                      }
                    } else {
                      out_dim->CopyFrom(first_dims[static_cast<size_t>(d)]);
                    }
                  }
                }));

  // FlexAttention (version 1): keep I/O and type signature aligned with preview schema.
  reg_infer(OpSchema()
                .SetName("FlexAttention")
                .SetDomain(AI_ONNX_PREVIEW_DOMAIN)
                .SinceVersion(1)
                .Input(0, "Q", "", "T1")
                .Input(1, "K", "", "T1")
                .Input(2, "V", "", "T1")
                .Output(0, "Y", "", "T1")
                .TypeConstraint("T1", OpSchema::all_float_types_ir4(),
                                "Constrain Q, K, V and Y to float tensors."));

  // Not: propagate shape and boolean type from input to output.
  for (int ver : {1}) {
    reg_infer(OpSchema()
                  .SetName("Not")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction(prop_in0));
  }

  // LessOrEqual, GreaterOrEqual: output is BOOL with broadcast shape.
  auto comparison_infer = [](InferenceContext &ctx) {
    updateOutputElemType(ctx, 0, TensorProto::BOOL);
    if (hasNInputShapes(ctx, 2)) {
      bidirectionalBroadcastShapeInference(ctx.getInputType(0)->tensor_type().shape(),
                                           ctx.getInputType(1)->tensor_type().shape(),
                                           *ctx.getOutputType(0)->tensor_type().mutable_shape());
    }
  };
  for (const char *name : {"LessOrEqual", "GreaterOrEqual"}) {
    for (int ver : {12, 16}) {
      reg_infer(OpSchema()
                    .SetName(name)
                    .SetDomain(ONNX_DOMAIN)
                    .SinceVersion(ver)
                    .TypeAndShapeInferenceFunction(comparison_infer));
    }
  }

  // ConstantOfShape: output shape from shape input data, type from 'value' attr.
  for (int ver : {9, 20, 23}) {
    reg_infer(OpSchema()
                  .SetName("ConstantOfShape")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                    const auto *val_attr = ctx.getAttribute("value");
                    if (val_attr && val_attr->has_t()) {
                      updateOutputElemType(ctx, 0, val_attr->t().data_type());
                    } else {
                      updateOutputElemType(ctx, 0, TensorProto::FLOAT);
                    }
                    const TensorProto *shape_data = ctx.getInputData(0);
                    if (!shape_data) {
                      return;
                    }
                    std::vector<int64_t> shape_vals;
                    const auto &i64_data = shape_data->int64_data();
                    if (!i64_data.empty()) {
                      shape_vals.assign(i64_data.begin(), i64_data.end());
                    } else if (!shape_data->raw_data().empty() &&
                               shape_data->raw_data().size() % sizeof(int64_t) == 0) {
                      const auto *p =
                          reinterpret_cast<const int64_t *>(shape_data->raw_data().data());
                      const size_t n = shape_data->raw_data().size() / sizeof(int64_t);
                      shape_vals.assign(p, p + n);
                    } else {
                      return;
                    }
                    auto *out_shape = ctx.getOutputType(0)->tensor_type().mutable_shape();
                    for (int64_t v : shape_vals) {
                      out_shape->add_dim()->set_dim_value(v);
                    }
                  }));
  }

  // Flatten: output is a rank-2 tensor [prod(0..axis-1), prod(axis..rank-1)].
  for (int ver : {1, 9, 11, 13, 21, 23}) {
    reg_infer(OpSchema()
                  .SetName("Flatten")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    const auto *in_type = ctx.getInputType(0);
                    if (!in_type || !in_type->has_tensor_type() ||
                        !in_type->tensor_type().has_shape()) {
                      return;
                    }
                    const auto &in_dims = in_type->tensor_type().shape().dim();
                    const int64_t rank = static_cast<int64_t>(in_dims.size());
                    int64_t axis = getAttribute(ctx, "axis", static_cast<int64_t>(1));
                    if (axis < 0) {
                      axis += rank;
                    }
                    // Compute product of dimensions before axis.
                    int64_t outer = 1;
                    bool outer_ok = true;
                    for (int64_t i = 0; i < axis; ++i) {
                      const auto &d = in_dims[static_cast<size_t>(i)];
                      if (d.has_dim_value()) {
                        outer *= d.dim_value();
                      } else {
                        outer_ok = false;
                        break;
                      }
                    }
                    // Compute product of dimensions from axis to end.
                    int64_t inner = 1;
                    bool inner_ok = true;
                    for (int64_t i = axis; i < rank; ++i) {
                      const auto &d = in_dims[static_cast<size_t>(i)];
                      if (d.has_dim_value()) {
                        inner *= d.dim_value();
                      } else {
                        inner_ok = false;
                        break;
                      }
                    }
                    auto *out_shape = ctx.getOutputType(0)->tensor_type().mutable_shape();
                    auto *d0 = out_shape->add_dim();
                    if (outer_ok) {
                      d0->set_dim_value(outer);
                    }
                    auto *d1 = out_shape->add_dim();
                    if (inner_ok) {
                      d1->set_dim_value(inner);
                    }
                  }));
  }

  // Shape: output is an INT64 rank-1 tensor whose size equals the input rank.
  for (int ver : {1, 13, 15, 19, 21, 23}) {
    reg_infer(
        OpSchema()
            .SetName("Shape")
            .SetDomain(ONNX_DOMAIN)
            .SinceVersion(ver)
            .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
              updateOutputElemType(ctx, 0, TensorProto::INT64);
              const auto *in_type = ctx.getInputType(0);
              if (in_type && in_type->has_tensor_type() && in_type->tensor_type().has_shape()) {
                auto *out_shape = ctx.getOutputType(0)->tensor_type().mutable_shape();
                auto *d = out_shape->add_dim();
                d->set_dim_value(static_cast<int64_t>(in_type->tensor_type().shape().dim().size()));
              }
            }));
  }

  // Expand: output has the same type as input and shape from the shape tensor.
  for (int ver : {8, 13}) {
    reg_infer(OpSchema()
                  .SetName("Expand")
                  .SetDomain(ONNX_DOMAIN)
                  .SinceVersion(ver)
                  .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    // Try to use the shape tensor data to determine the output shape.
                    const TensorProto *shape_data = ctx.getInputData(1);
                    if (!shape_data) {
                      return;
                    }
                    if (shape_data->data_type() != TensorProto::INT64) {
                      return;
                    }
                    // Extract shape values from the tensor.
                    std::vector<int64_t> shape_vals;
                    const auto &i64_data = shape_data->int64_data();
                    if (!i64_data.empty()) {
                      shape_vals.assign(i64_data.begin(), i64_data.end());
                    } else if (!shape_data->raw_data().empty() &&
                               shape_data->raw_data().size() % sizeof(int64_t) == 0) {
                      const auto *p =
                          reinterpret_cast<const int64_t *>(shape_data->raw_data().data());
                      const size_t n = shape_data->raw_data().size() / sizeof(int64_t);
                      shape_vals.assign(p, p + n);
                    } else {
                      return;
                    }
                    auto *out_shape = ctx.getOutputType(0)->tensor_type().mutable_shape();
                    for (int64_t v : shape_vals) {
                      out_shape->add_dim()->set_dim_value(v);
                    }
                  }));
  }

  // ------------------------------------------------------------------
  // Skeleton registrations for every other operator / version.
  // Versions already registered above are silently skipped.
  // ------------------------------------------------------------------
  auto reg = [](const char *name, const char *domain, int ver) {
    RegisterSchema(OpSchema().SetName(name).SetDomain(domain).SinceVersion(ver), 0, false, false);
  };
  reg("Abs", ONNX_DOMAIN, 1);
  reg("Add", ONNX_DOMAIN, 1);
  reg("And", ONNX_DOMAIN, 1);
  reg("ArgMax", ONNX_DOMAIN, 1);
  reg("ArgMin", ONNX_DOMAIN, 1);
  reg("AveragePool", ONNX_DOMAIN, 1);
  reg("BatchNormalization", ONNX_DOMAIN, 1);
  reg("Cast", ONNX_DOMAIN, 1);
  reg("Ceil", ONNX_DOMAIN, 1);
  reg("Clip", ONNX_DOMAIN, 1);
  reg("Concat", ONNX_DOMAIN, 1);
  reg("Constant", ONNX_DOMAIN, 1);
  reg("Conv", ONNX_DOMAIN, 1);
  reg("ConvTranspose", ONNX_DOMAIN, 1);
  reg("DepthToSpace", ONNX_DOMAIN, 1);
  reg("Div", ONNX_DOMAIN, 1);
  reg("Dropout", ONNX_DOMAIN, 1);
  reg("Elu", ONNX_DOMAIN, 1);
  reg("Equal", ONNX_DOMAIN, 1);
  reg("Exp", ONNX_DOMAIN, 1);
  reg("Flatten", ONNX_DOMAIN, 1);
  reg("Floor", ONNX_DOMAIN, 1);
  reg("GRU", ONNX_DOMAIN, 1);
  reg("Gather", ONNX_DOMAIN, 1);
  reg("Gemm", ONNX_DOMAIN, 1);
  reg("GlobalAveragePool", ONNX_DOMAIN, 1);
  reg("GlobalLpPool", ONNX_DOMAIN, 1);
  reg("GlobalMaxPool", ONNX_DOMAIN, 1);
  reg("Greater", ONNX_DOMAIN, 1);
  reg("HardSigmoid", ONNX_DOMAIN, 1);
  reg("Hardmax", ONNX_DOMAIN, 1);
  reg("Identity", ONNX_DOMAIN, 1);
  reg("If", ONNX_DOMAIN, 1);
  reg("InstanceNormalization", ONNX_DOMAIN, 1);
  reg("LRN", ONNX_DOMAIN, 1);
  reg("LSTM", ONNX_DOMAIN, 1);
  reg("LeakyRelu", ONNX_DOMAIN, 1);
  reg("Less", ONNX_DOMAIN, 1);
  reg("Log", ONNX_DOMAIN, 1);
  reg("LogSoftmax", ONNX_DOMAIN, 1);
  reg("Loop", ONNX_DOMAIN, 1);
  reg("LpNormalization", ONNX_DOMAIN, 1);
  reg("LpPool", ONNX_DOMAIN, 1);
  reg("MatMul", ONNX_DOMAIN, 1);
  reg("Max", ONNX_DOMAIN, 1);
  reg("MaxPool", ONNX_DOMAIN, 1);
  reg("MaxRoiPool", ONNX_DOMAIN, 1);
  reg("Mean", ONNX_DOMAIN, 1);
  reg("Min", ONNX_DOMAIN, 1);
  reg("Mul", ONNX_DOMAIN, 1);
  reg("Neg", ONNX_DOMAIN, 1);
  reg("Not", ONNX_DOMAIN, 1);
  reg("Or", ONNX_DOMAIN, 1);
  reg("PRelu", ONNX_DOMAIN, 1);
  reg("Pad", ONNX_DOMAIN, 1);
  reg("Pow", ONNX_DOMAIN, 1);
  reg("RNN", ONNX_DOMAIN, 1);
  reg("RandomNormal", ONNX_DOMAIN, 1);
  reg("RandomNormalLike", ONNX_DOMAIN, 1);
  reg("RandomUniform", ONNX_DOMAIN, 1);
  reg("RandomUniformLike", ONNX_DOMAIN, 1);
  reg("Reciprocal", ONNX_DOMAIN, 1);
  reg("ReduceL1", ONNX_DOMAIN, 1);
  reg("ReduceL2", ONNX_DOMAIN, 1);
  reg("ReduceLogSum", ONNX_DOMAIN, 1);
  reg("ReduceLogSumExp", ONNX_DOMAIN, 1);
  reg("ReduceMax", ONNX_DOMAIN, 1);
  reg("ReduceMean", ONNX_DOMAIN, 1);
  reg("ReduceMin", ONNX_DOMAIN, 1);
  reg("ReduceProd", ONNX_DOMAIN, 1);
  reg("ReduceSum", ONNX_DOMAIN, 1);
  reg("ReduceSumSquare", ONNX_DOMAIN, 1);
  reg("Relu", ONNX_DOMAIN, 1);
  reg("Reshape", ONNX_DOMAIN, 1);
  reg("Selu", ONNX_DOMAIN, 1);
  reg("Shape", ONNX_DOMAIN, 1);
  reg("Sigmoid", ONNX_DOMAIN, 1);
  reg("Size", ONNX_DOMAIN, 1);
  reg("Slice", ONNX_DOMAIN, 1);
  reg("Softmax", ONNX_DOMAIN, 1);
  reg("Softplus", ONNX_DOMAIN, 1);
  reg("Softsign", ONNX_DOMAIN, 1);
  reg("SpaceToDepth", ONNX_DOMAIN, 1);
  reg("Split", ONNX_DOMAIN, 1);
  reg("Sqrt", ONNX_DOMAIN, 1);
  reg("Squeeze", ONNX_DOMAIN, 1);
  reg("Sub", ONNX_DOMAIN, 1);
  reg("Sum", ONNX_DOMAIN, 1);
  reg("Tanh", ONNX_DOMAIN, 1);
  reg("Tile", ONNX_DOMAIN, 1);
  reg("TopK", ONNX_DOMAIN, 1);
  reg("Transpose", ONNX_DOMAIN, 1);
  reg("Unsqueeze", ONNX_DOMAIN, 1);
  reg("Upsample", ONNX_DOMAIN, 1);
  reg("Xor", ONNX_DOMAIN, 1);
  reg("GlobalLpPool", ONNX_DOMAIN, 2);
  reg("LpPool", ONNX_DOMAIN, 2);
  reg("Pad", ONNX_DOMAIN, 2);
  reg("Split", ONNX_DOMAIN, 2);
  reg("GRU", ONNX_DOMAIN, 3);
  reg("Concat", ONNX_DOMAIN, 4);
  reg("Reshape", ONNX_DOMAIN, 5);
  reg("Abs", ONNX_DOMAIN, 6);
  reg("Add", ONNX_DOMAIN, 6);
  reg("BatchNormalization", ONNX_DOMAIN, 6);
  reg("Cast", ONNX_DOMAIN, 6);
  reg("Ceil", ONNX_DOMAIN, 6);
  reg("Clip", ONNX_DOMAIN, 6);
  reg("Div", ONNX_DOMAIN, 6);
  reg("Dropout", ONNX_DOMAIN, 6);
  reg("Elu", ONNX_DOMAIN, 6);
  reg("Exp", ONNX_DOMAIN, 6);
  reg("Floor", ONNX_DOMAIN, 6);
  reg("Gemm", ONNX_DOMAIN, 6);
  reg("HardSigmoid", ONNX_DOMAIN, 6);
  reg("InstanceNormalization", ONNX_DOMAIN, 6);
  reg("LeakyRelu", ONNX_DOMAIN, 6);
  reg("Log", ONNX_DOMAIN, 6);
  reg("Max", ONNX_DOMAIN, 6);
  reg("Mean", ONNX_DOMAIN, 6);
  reg("Min", ONNX_DOMAIN, 6);
  reg("Mul", ONNX_DOMAIN, 6);
  reg("Neg", ONNX_DOMAIN, 6);
  reg("PRelu", ONNX_DOMAIN, 6);
  reg("Reciprocal", ONNX_DOMAIN, 6);
  reg("Relu", ONNX_DOMAIN, 6);
  reg("Selu", ONNX_DOMAIN, 6);
  reg("Sigmoid", ONNX_DOMAIN, 6);
  reg("Sqrt", ONNX_DOMAIN, 6);
  reg("Sub", ONNX_DOMAIN, 6);
  reg("Sum", ONNX_DOMAIN, 6);
  reg("Tanh", ONNX_DOMAIN, 6);
  reg("Tile", ONNX_DOMAIN, 6);
  reg("Acos", ONNX_DOMAIN, 7);
  reg("Add", ONNX_DOMAIN, 7);
  reg("And", ONNX_DOMAIN, 7);
  reg("Asin", ONNX_DOMAIN, 7);
  reg("Atan", ONNX_DOMAIN, 7);
  reg("AveragePool", ONNX_DOMAIN, 7);
  reg("BatchNormalization", ONNX_DOMAIN, 7);
  reg("Cos", ONNX_DOMAIN, 7);
  reg("Div", ONNX_DOMAIN, 7);
  reg("Dropout", ONNX_DOMAIN, 7);
  reg("Equal", ONNX_DOMAIN, 7);
  reg("Gemm", ONNX_DOMAIN, 7);
  reg("Greater", ONNX_DOMAIN, 7);
  reg("GRU", ONNX_DOMAIN, 7);
  reg("Less", ONNX_DOMAIN, 7);
  reg("LSTM", ONNX_DOMAIN, 7);
  reg("Mul", ONNX_DOMAIN, 7);
  reg("Or", ONNX_DOMAIN, 7);
  reg("Pow", ONNX_DOMAIN, 7);
  reg("RNN", ONNX_DOMAIN, 7);
  reg("Sin", ONNX_DOMAIN, 7);
  reg("Sub", ONNX_DOMAIN, 7);
  reg("Tan", ONNX_DOMAIN, 7);
  reg("Upsample", ONNX_DOMAIN, 7);
  reg("Multinomial", ONNX_DOMAIN, 7);
  reg("Xor", ONNX_DOMAIN, 7);
  reg("PRelu", ONNX_DOMAIN, 7);
  reg("Expand", ONNX_DOMAIN, 8);
  reg("Min", ONNX_DOMAIN, 8);
  reg("Max", ONNX_DOMAIN, 8);
  reg("Sum", ONNX_DOMAIN, 8);
  reg("Mean", ONNX_DOMAIN, 8);
  reg("MaxPool", ONNX_DOMAIN, 8);
  reg("Scan", ONNX_DOMAIN, 8);
  reg("BatchNormalization", ONNX_DOMAIN, 9);
  reg("Compress", ONNX_DOMAIN, 9);
  reg("ConstantOfShape", ONNX_DOMAIN, 9);
  reg("EyeLike", ONNX_DOMAIN, 9);
  reg("Greater", ONNX_DOMAIN, 9);
  reg("Less", ONNX_DOMAIN, 9);
  reg("Upsample", ONNX_DOMAIN, 9);
  reg("MaxUnpool", ONNX_DOMAIN, 9);
  reg("Constant", ONNX_DOMAIN, 9);
  reg("MatMul", ONNX_DOMAIN, 9);
  reg("OneHot", ONNX_DOMAIN, 9);
  reg("PRelu", ONNX_DOMAIN, 9);
  reg("Gemm", ONNX_DOMAIN, 9);
  reg("Flatten", ONNX_DOMAIN, 9);
  reg("Scatter", ONNX_DOMAIN, 9);
  reg("Sinh", ONNX_DOMAIN, 9);
  reg("Cosh", ONNX_DOMAIN, 9);
  reg("Asinh", ONNX_DOMAIN, 9);
  reg("Acosh", ONNX_DOMAIN, 9);
  reg("Atanh", ONNX_DOMAIN, 9);
  reg("Shrink", ONNX_DOMAIN, 9);
  reg("IsNaN", ONNX_DOMAIN, 9);
  reg("Sign", ONNX_DOMAIN, 9);
  reg("Scan", ONNX_DOMAIN, 9);
  reg("Erf", ONNX_DOMAIN, 9);
  reg("Cast", ONNX_DOMAIN, 9);
  reg("Where", ONNX_DOMAIN, 9);
  reg("NonZero", ONNX_DOMAIN, 9);
  reg("TfIdfVectorizer", ONNX_DOMAIN, 9);
  reg("MeanVarianceNormalization", ONNX_DOMAIN, 9);
  reg("Upsample", ONNX_DOMAIN, 10);
  reg("Resize", ONNX_DOMAIN, 10);
  reg("StringNormalizer", ONNX_DOMAIN, 10);
  reg("TopK", ONNX_DOMAIN, 10);
  reg("MaxPool", ONNX_DOMAIN, 10);
  reg("Mod", ONNX_DOMAIN, 10);
  reg("AveragePool", ONNX_DOMAIN, 10);
  reg("Slice", ONNX_DOMAIN, 10);
  reg("ThresholdedRelu", ONNX_DOMAIN, 10);
  reg("Dropout", ONNX_DOMAIN, 10);
  reg("MatMulInteger", ONNX_DOMAIN, 10);
  reg("QLinearMatMul", ONNX_DOMAIN, 10);
  reg("ConvInteger", ONNX_DOMAIN, 10);
  reg("QLinearConv", ONNX_DOMAIN, 10);
  reg("QuantizeLinear", ONNX_DOMAIN, 10);
  reg("DequantizeLinear", ONNX_DOMAIN, 10);
  reg("IsInf", ONNX_DOMAIN, 10);
  reg("NonMaxSuppression", ONNX_DOMAIN, 10);
  reg("ReverseSequence", ONNX_DOMAIN, 10);
  reg("RoiAlign", ONNX_DOMAIN, 10);
  reg("Loop", ONNX_DOMAIN, 11);
  reg("BitShift", ONNX_DOMAIN, 11);
  reg("Unique", ONNX_DOMAIN, 11);
  reg("CumSum", ONNX_DOMAIN, 11);
  reg("Round", ONNX_DOMAIN, 11);
  reg("TopK", ONNX_DOMAIN, 11);
  reg("DepthToSpace", ONNX_DOMAIN, 11);
  reg("Equal", ONNX_DOMAIN, 11);
  reg("Constant", ONNX_DOMAIN, 11);
  reg("DynamicQuantizeLinear", ONNX_DOMAIN, 11);
  reg("GatherElements", ONNX_DOMAIN, 11);
  reg("ScatterElements", ONNX_DOMAIN, 11);
  reg("Scatter", ONNX_DOMAIN, 11);
  reg("Clip", ONNX_DOMAIN, 11);
  reg("Resize", ONNX_DOMAIN, 11);
  reg("Range", ONNX_DOMAIN, 11);
  reg("Det", ONNX_DOMAIN, 11);
  reg("ScatterND", ONNX_DOMAIN, 11);
  reg("GatherND", ONNX_DOMAIN, 11);
  reg("Gather", ONNX_DOMAIN, 11);
  reg("OneHot", ONNX_DOMAIN, 11);
  reg("Slice", ONNX_DOMAIN, 11);
  reg("Squeeze", ONNX_DOMAIN, 11);
  reg("Unsqueeze", ONNX_DOMAIN, 11);
  reg("Flatten", ONNX_DOMAIN, 11);
  reg("ArgMin", ONNX_DOMAIN, 11);
  reg("ArgMax", ONNX_DOMAIN, 11);
  reg("ReduceL1", ONNX_DOMAIN, 11);
  reg("ReduceL2", ONNX_DOMAIN, 11);
  reg("ReduceLogSum", ONNX_DOMAIN, 11);
  reg("ReduceLogSumExp", ONNX_DOMAIN, 11);
  reg("ReduceMax", ONNX_DOMAIN, 11);
  reg("ReduceMean", ONNX_DOMAIN, 11);
  reg("ReduceMin", ONNX_DOMAIN, 11);
  reg("ReduceProd", ONNX_DOMAIN, 11);
  reg("ReduceSum", ONNX_DOMAIN, 11);
  reg("ReduceSumSquare", ONNX_DOMAIN, 11);
  reg("Compress", ONNX_DOMAIN, 11);
  reg("Concat", ONNX_DOMAIN, 11);
  reg("Hardmax", ONNX_DOMAIN, 11);
  reg("LogSoftmax", ONNX_DOMAIN, 11);
  reg("Softmax", ONNX_DOMAIN, 11);
  reg("Scan", ONNX_DOMAIN, 11);
  reg("Split", ONNX_DOMAIN, 11);
  reg("AveragePool", ONNX_DOMAIN, 11);
  reg("MaxPool", ONNX_DOMAIN, 11);
  reg("MaxUnpool", ONNX_DOMAIN, 11);
  reg("LpPool", ONNX_DOMAIN, 11);
  reg("Conv", ONNX_DOMAIN, 11);
  reg("ConvTranspose", ONNX_DOMAIN, 11);
  reg("SequenceEmpty", ONNX_DOMAIN, 11);
  reg("SequenceConstruct", ONNX_DOMAIN, 11);
  reg("SequenceInsert", ONNX_DOMAIN, 11);
  reg("SequenceAt", ONNX_DOMAIN, 11);
  reg("SequenceErase", ONNX_DOMAIN, 11);
  reg("SequenceLength", ONNX_DOMAIN, 11);
  reg("SplitToSequence", ONNX_DOMAIN, 11);
  reg("ConcatFromSequence", ONNX_DOMAIN, 11);
  reg("Pad", ONNX_DOMAIN, 11);
  reg("Gemm", ONNX_DOMAIN, 11);
  reg("If", ONNX_DOMAIN, 11);
  reg("NonMaxSuppression", ONNX_DOMAIN, 11);
  reg("ArgMax", ONNX_DOMAIN, 12);
  reg("ArgMin", ONNX_DOMAIN, 12);
  reg("Clip", ONNX_DOMAIN, 12);
  reg("Einsum", ONNX_DOMAIN, 12);
  reg("MaxPool", ONNX_DOMAIN, 12);
  reg("ReduceMax", ONNX_DOMAIN, 12);
  reg("ReduceMin", ONNX_DOMAIN, 12);
  reg("GatherND", ONNX_DOMAIN, 12);
  reg("NegativeLogLikelihoodLoss", ONNX_DOMAIN, 12);
  reg("Dropout", ONNX_DOMAIN, 12);
  reg("Constant", ONNX_DOMAIN, 12);
  reg("Celu", ONNX_DOMAIN, 12);
  reg("Max", ONNX_DOMAIN, 12);
  reg("Min", ONNX_DOMAIN, 12);
  reg("LessOrEqual", ONNX_DOMAIN, 12);
  reg("GreaterOrEqual", ONNX_DOMAIN, 12);
  reg("SoftmaxCrossEntropyLoss", ONNX_DOMAIN, 12);
  reg("Pow", ONNX_DOMAIN, 12);
  reg("Constant", ONNX_DOMAIN, 13);
  reg("Greater", ONNX_DOMAIN, 13);
  reg("Less", ONNX_DOMAIN, 13);
  reg("Equal", ONNX_DOMAIN, 13);
  reg("Add", ONNX_DOMAIN, 13);
  reg("Sub", ONNX_DOMAIN, 13);
  reg("Mul", ONNX_DOMAIN, 13);
  reg("Div", ONNX_DOMAIN, 13);
  reg("Softmax", ONNX_DOMAIN, 13);
  reg("LogSoftmax", ONNX_DOMAIN, 13);
  reg("Hardmax", ONNX_DOMAIN, 13);
  reg("Mod", ONNX_DOMAIN, 13);
  reg("Neg", ONNX_DOMAIN, 13);
  reg("Abs", ONNX_DOMAIN, 13);
  reg("Reciprocal", ONNX_DOMAIN, 13);
  reg("Floor", ONNX_DOMAIN, 13);
  reg("Ceil", ONNX_DOMAIN, 13);
  reg("Sqrt", ONNX_DOMAIN, 13);
  reg("Relu", ONNX_DOMAIN, 13);
  reg("Exp", ONNX_DOMAIN, 13);
  reg("Log", ONNX_DOMAIN, 13);
  reg("Tanh", ONNX_DOMAIN, 13);
  reg("Pow", ONNX_DOMAIN, 13);
  reg("Sigmoid", ONNX_DOMAIN, 13);
  reg("Max", ONNX_DOMAIN, 13);
  reg("Min", ONNX_DOMAIN, 13);
  reg("Sum", ONNX_DOMAIN, 13);
  reg("Mean", ONNX_DOMAIN, 13);
  reg("Clip", ONNX_DOMAIN, 13);
  reg("Gemm", ONNX_DOMAIN, 13);
  reg("MatMul", ONNX_DOMAIN, 13);
  reg("Expand", ONNX_DOMAIN, 13);
  reg("Sign", ONNX_DOMAIN, 13);
  reg("Erf", ONNX_DOMAIN, 13);
  reg("SoftmaxCrossEntropyLoss", ONNX_DOMAIN, 13);
  reg("NegativeLogLikelihoodLoss", ONNX_DOMAIN, 13);
  reg("Dropout", ONNX_DOMAIN, 13);
  reg("Flatten", ONNX_DOMAIN, 13);
  reg("LRN", ONNX_DOMAIN, 13);
  reg("MeanVarianceNormalization", ONNX_DOMAIN, 13);
  reg("ReduceMax", ONNX_DOMAIN, 13);
  reg("ReduceMin", ONNX_DOMAIN, 13);
  reg("ReduceSum", ONNX_DOMAIN, 13);
  reg("ReduceSumSquare", ONNX_DOMAIN, 13);
  reg("ReduceMean", ONNX_DOMAIN, 13);
  reg("ReduceProd", ONNX_DOMAIN, 13);
  reg("ReduceLogSum", ONNX_DOMAIN, 13);
  reg("ReduceLogSumExp", ONNX_DOMAIN, 13);
  reg("ReduceL1", ONNX_DOMAIN, 13);
  reg("ReduceL2", ONNX_DOMAIN, 13);
  reg("ArgMax", ONNX_DOMAIN, 13);
  reg("ArgMin", ONNX_DOMAIN, 13);
  reg("Cast", ONNX_DOMAIN, 13);
  reg("Reshape", ONNX_DOMAIN, 13);
  reg("Shape", ONNX_DOMAIN, 13);
  reg("Size", ONNX_DOMAIN, 13);
  reg("Concat", ONNX_DOMAIN, 13);
  reg("Split", ONNX_DOMAIN, 13);
  reg("Slice", ONNX_DOMAIN, 13);
  reg("Transpose", ONNX_DOMAIN, 13);
  reg("ScatterND", ONNX_DOMAIN, 13);
  reg("ScatterElements", ONNX_DOMAIN, 13);
  reg("Gather", ONNX_DOMAIN, 13);
  reg("GatherElements", ONNX_DOMAIN, 13);
  reg("Squeeze", ONNX_DOMAIN, 13);
  reg("Unsqueeze", ONNX_DOMAIN, 13);
  reg("SpaceToDepth", ONNX_DOMAIN, 13);
  reg("DepthToSpace", ONNX_DOMAIN, 13);
  reg("Tile", ONNX_DOMAIN, 13);
  reg("Resize", ONNX_DOMAIN, 13);
  reg("Identity", ONNX_DOMAIN, 13);
  reg("IsNaN", ONNX_DOMAIN, 13);
  reg("NonZero", ONNX_DOMAIN, 13);
  reg("GatherND", ONNX_DOMAIN, 13);
  reg("Pad", ONNX_DOMAIN, 13);
  reg("QuantizeLinear", ONNX_DOMAIN, 13);
  reg("DequantizeLinear", ONNX_DOMAIN, 13);
  reg("Loop", ONNX_DOMAIN, 13);
  reg("If", ONNX_DOMAIN, 13);
  reg("CumSum", ONNX_DOMAIN, 14);
  reg("Relu", ONNX_DOMAIN, 14);
  reg("Reshape", ONNX_DOMAIN, 14);
  reg("GRU", ONNX_DOMAIN, 14);
  reg("LSTM", ONNX_DOMAIN, 14);
  reg("RNN", ONNX_DOMAIN, 14);
  reg("Trilu", ONNX_DOMAIN, 14);
  reg("BatchNormalization", ONNX_DOMAIN, 14);
  reg("HardSwish", ONNX_DOMAIN, 14);
  reg("Add", ONNX_DOMAIN, 14);
  reg("Sub", ONNX_DOMAIN, 14);
  reg("Mul", ONNX_DOMAIN, 14);
  reg("Div", ONNX_DOMAIN, 14);
  reg("Identity", ONNX_DOMAIN, 14);
  reg("BatchNormalization", ONNX_DOMAIN, 15);
  reg("Bernoulli", ONNX_DOMAIN, 15);
  reg("Pow", ONNX_DOMAIN, 15);
  reg("Optional", ONNX_DOMAIN, 15);
  reg("OptionalHasElement", ONNX_DOMAIN, 15);
  reg("OptionalGetElement", ONNX_DOMAIN, 15);
  reg("CastLike", ONNX_DOMAIN, 15);
  reg("Shape", ONNX_DOMAIN, 15);
  reg("RoiAlign", ONNX_DOMAIN, 16);
  reg("ScatterND", ONNX_DOMAIN, 16);
  reg("ScatterElements", ONNX_DOMAIN, 16);
  reg("If", ONNX_DOMAIN, 16);
  reg("Loop", ONNX_DOMAIN, 16);
  reg("Identity", ONNX_DOMAIN, 16);
  reg("Where", ONNX_DOMAIN, 16);
  reg("GridSample", ONNX_DOMAIN, 16);
  reg("Scan", ONNX_DOMAIN, 16);
  reg("LessOrEqual", ONNX_DOMAIN, 16);
  reg("GreaterOrEqual", ONNX_DOMAIN, 16);
  reg("LeakyRelu", ONNX_DOMAIN, 16);
  reg("PRelu", ONNX_DOMAIN, 16);
  reg("LayerNormalization", ONNX_DOMAIN, 17);
  reg("SequenceMap", ONNX_DOMAIN, 17);
  reg("DFT", ONNX_DOMAIN, 17);
  reg("HannWindow", ONNX_DOMAIN, 17);
  reg("HammingWindow", ONNX_DOMAIN, 17);
  reg("BlackmanWindow", ONNX_DOMAIN, 17);
  reg("MelWeightMatrix", ONNX_DOMAIN, 17);
  reg("STFT", ONNX_DOMAIN, 17);
  reg("Pad", ONNX_DOMAIN, 18);
  reg("CenterCropPad", ONNX_DOMAIN, 18);
  reg("Resize", ONNX_DOMAIN, 18);
  reg("Mish", ONNX_DOMAIN, 18);
  reg("OptionalGetElement", ONNX_DOMAIN, 18);
  reg("Split", ONNX_DOMAIN, 18);
  reg("OptionalHasElement", ONNX_DOMAIN, 18);
  reg("Col2Im", ONNX_DOMAIN, 18);
  reg("ScatterND", ONNX_DOMAIN, 18);
  reg("ScatterElements", ONNX_DOMAIN, 18);
  reg("ReduceSumSquare", ONNX_DOMAIN, 18);
  reg("ReduceLogSum", ONNX_DOMAIN, 18);
  reg("ReduceLogSumExp", ONNX_DOMAIN, 18);
  reg("ReduceL1", ONNX_DOMAIN, 18);
  reg("ReduceL2", ONNX_DOMAIN, 18);
  reg("ReduceMax", ONNX_DOMAIN, 18);
  reg("ReduceMin", ONNX_DOMAIN, 18);
  reg("ReduceMean", ONNX_DOMAIN, 18);
  reg("ReduceProd", ONNX_DOMAIN, 18);
  reg("BitwiseAnd", ONNX_DOMAIN, 18);
  reg("BitwiseOr", ONNX_DOMAIN, 18);
  reg("BitwiseXor", ONNX_DOMAIN, 18);
  reg("BitwiseNot", ONNX_DOMAIN, 18);
  reg("GroupNormalization", ONNX_DOMAIN, 18);
  reg("LpPool", ONNX_DOMAIN, 18);
  reg("Equal", ONNX_DOMAIN, 19);
  reg("AveragePool", ONNX_DOMAIN, 19);
  reg("Cast", ONNX_DOMAIN, 19);
  reg("CastLike", ONNX_DOMAIN, 19);
  reg("Constant", ONNX_DOMAIN, 19);
  reg("DeformConv", ONNX_DOMAIN, 19);
  reg("DequantizeLinear", ONNX_DOMAIN, 19);
  reg("Identity", ONNX_DOMAIN, 19);
  reg("If", ONNX_DOMAIN, 19);
  reg("Loop", ONNX_DOMAIN, 19);
  reg("Pad", ONNX_DOMAIN, 19);
  reg("QuantizeLinear", ONNX_DOMAIN, 19);
  reg("Reshape", ONNX_DOMAIN, 19);
  reg("Resize", ONNX_DOMAIN, 19);
  reg("Scan", ONNX_DOMAIN, 19);
  reg("Shape", ONNX_DOMAIN, 19);
  reg("Size", ONNX_DOMAIN, 19);
  reg("AffineGrid", ONNX_DOMAIN, 20);
  reg("ConstantOfShape", ONNX_DOMAIN, 20);
  reg("DFT", ONNX_DOMAIN, 20);
  reg("Gelu", ONNX_DOMAIN, 20);
  reg("GridSample", ONNX_DOMAIN, 20);
  reg("ImageDecoder", ONNX_DOMAIN, 20);
  reg("IsInf", ONNX_DOMAIN, 20);
  reg("IsNaN", ONNX_DOMAIN, 20);
  reg("ReduceMax", ONNX_DOMAIN, 20);
  reg("ReduceMin", ONNX_DOMAIN, 20);
  reg("RegexFullMatch", ONNX_DOMAIN, 20);
  reg("StringConcat", ONNX_DOMAIN, 20);
  reg("StringSplit", ONNX_DOMAIN, 20);
  reg("Cast", ONNX_DOMAIN, 21);
  reg("CastLike", ONNX_DOMAIN, 21);
  reg("Constant", ONNX_DOMAIN, 21);
  reg("ConstantOfShape", ONNX_DOMAIN, 21);
  reg("DequantizeLinear", ONNX_DOMAIN, 21);
  reg("Flatten", ONNX_DOMAIN, 21);
  reg("GroupNormalization", ONNX_DOMAIN, 21);
  reg("Identity", ONNX_DOMAIN, 21);
  reg("If", ONNX_DOMAIN, 21);
  reg("Loop", ONNX_DOMAIN, 21);
  reg("Pad", ONNX_DOMAIN, 21);
  reg("QLinearMatMul", ONNX_DOMAIN, 21);
  reg("QuantizeLinear", ONNX_DOMAIN, 21);
  reg("Reshape", ONNX_DOMAIN, 21);
  reg("Scan", ONNX_DOMAIN, 21);
  reg("Shape", ONNX_DOMAIN, 21);
  reg("Size", ONNX_DOMAIN, 21);
  reg("Squeeze", ONNX_DOMAIN, 21);
  reg("Transpose", ONNX_DOMAIN, 21);
  reg("Unsqueeze", ONNX_DOMAIN, 21);
  reg("EyeLike", ONNX_DOMAIN, 22);
  reg("RandomUniform", ONNX_DOMAIN, 22);
  reg("RandomNormal", ONNX_DOMAIN, 22);
  reg("RandomUniformLike", ONNX_DOMAIN, 22);
  reg("RandomNormalLike", ONNX_DOMAIN, 22);
  reg("Multinomial", ONNX_DOMAIN, 22);
  reg("Bernoulli", ONNX_DOMAIN, 22);
  reg("ThresholdedRelu", ONNX_DOMAIN, 22);
  reg("Selu", ONNX_DOMAIN, 22);
  reg("Elu", ONNX_DOMAIN, 22);
  reg("Mish", ONNX_DOMAIN, 22);
  reg("HardSigmoid", ONNX_DOMAIN, 22);
  reg("HardSwish", ONNX_DOMAIN, 22);
  reg("Softsign", ONNX_DOMAIN, 22);
  reg("Softplus", ONNX_DOMAIN, 22);
  reg("Sin", ONNX_DOMAIN, 22);
  reg("Cos", ONNX_DOMAIN, 22);
  reg("Tan", ONNX_DOMAIN, 22);
  reg("Asin", ONNX_DOMAIN, 22);
  reg("Acos", ONNX_DOMAIN, 22);
  reg("Atan", ONNX_DOMAIN, 22);
  reg("Sinh", ONNX_DOMAIN, 22);
  reg("Cosh", ONNX_DOMAIN, 22);
  reg("Asinh", ONNX_DOMAIN, 22);
  reg("Acosh", ONNX_DOMAIN, 22);
  reg("Atanh", ONNX_DOMAIN, 22);
  reg("Round", ONNX_DOMAIN, 22);
  reg("Det", ONNX_DOMAIN, 22);
  reg("NegativeLogLikelihoodLoss", ONNX_DOMAIN, 22);
  reg("AveragePool", ONNX_DOMAIN, 22);
  reg("MaxPool", ONNX_DOMAIN, 22);
  reg("MaxUnpool", ONNX_DOMAIN, 22);
  reg("LpPool", ONNX_DOMAIN, 22);
  reg("MaxRoiPool", ONNX_DOMAIN, 22);
  reg("Conv", ONNX_DOMAIN, 22);
  reg("ConvTranspose", ONNX_DOMAIN, 22);
  reg("DeformConv", ONNX_DOMAIN, 22);
  reg("GlobalAveragePool", ONNX_DOMAIN, 22);
  reg("GlobalMaxPool", ONNX_DOMAIN, 22);
  reg("GlobalLpPool", ONNX_DOMAIN, 22);
  reg("InstanceNormalization", ONNX_DOMAIN, 22);
  reg("LpNormalization", ONNX_DOMAIN, 22);
  reg("Dropout", ONNX_DOMAIN, 22);
  reg("RoiAlign", ONNX_DOMAIN, 22);
  reg("RNN", ONNX_DOMAIN, 22);
  reg("GRU", ONNX_DOMAIN, 22);
  reg("LSTM", ONNX_DOMAIN, 22);
  reg("GridSample", ONNX_DOMAIN, 22);
  reg("Attention", ONNX_DOMAIN, 23);
  reg("Cast", ONNX_DOMAIN, 23);
  reg("CastLike", ONNX_DOMAIN, 23);
  reg("Constant", ONNX_DOMAIN, 23);
  reg("ConstantOfShape", ONNX_DOMAIN, 23);
  reg("DequantizeLinear", ONNX_DOMAIN, 23);
  reg("Flatten", ONNX_DOMAIN, 23);
  reg("Identity", ONNX_DOMAIN, 23);
  reg("If", ONNX_DOMAIN, 23);
  reg("Loop", ONNX_DOMAIN, 23);
  reg("Pad", ONNX_DOMAIN, 23);
  reg("QuantizeLinear", ONNX_DOMAIN, 23);
  reg("Reshape", ONNX_DOMAIN, 23);
  reg("RMSNormalization", ONNX_DOMAIN, 23);
  reg("RotaryEmbedding", ONNX_DOMAIN, 23);
  reg("Scan", ONNX_DOMAIN, 23);
  reg("Shape", ONNX_DOMAIN, 23);
  reg("Size", ONNX_DOMAIN, 23);
  reg("Squeeze", ONNX_DOMAIN, 23);
  reg("Transpose", ONNX_DOMAIN, 23);
  reg("Unsqueeze", ONNX_DOMAIN, 23);
  reg("Attention", ONNX_DOMAIN, 24);
  reg("Cast", ONNX_DOMAIN, 24);
  reg("CastLike", ONNX_DOMAIN, 24);
  reg("Constant", ONNX_DOMAIN, 24);
  reg("ConstantOfShape", ONNX_DOMAIN, 24);
  reg("DequantizeLinear", ONNX_DOMAIN, 24);
  reg("Flatten", ONNX_DOMAIN, 24);
  reg("Identity", ONNX_DOMAIN, 24);
  reg("If", ONNX_DOMAIN, 24);
  reg("Loop", ONNX_DOMAIN, 24);
  reg("Pad", ONNX_DOMAIN, 24);
  reg("QuantizeLinear", ONNX_DOMAIN, 24);
  reg("Reshape", ONNX_DOMAIN, 24);
  reg("Scan", ONNX_DOMAIN, 24);
  reg("Shape", ONNX_DOMAIN, 24);
  reg("Size", ONNX_DOMAIN, 24);
  reg("Swish", ONNX_DOMAIN, 24);
  reg("SplitToSequence", ONNX_DOMAIN, 24);
  reg("Squeeze", ONNX_DOMAIN, 24);
  reg("TopK", ONNX_DOMAIN, 24);
  reg("Transpose", ONNX_DOMAIN, 24);
  reg("Unsqueeze", ONNX_DOMAIN, 24);
  reg("TensorScatter", ONNX_DOMAIN, 24);
  reg("Cast", ONNX_DOMAIN, 25);
  reg("CastLike", ONNX_DOMAIN, 25);
  reg("Constant", ONNX_DOMAIN, 25);
  reg("ConstantOfShape", ONNX_DOMAIN, 25);
  reg("DequantizeLinear", ONNX_DOMAIN, 25);
  reg("Flatten", ONNX_DOMAIN, 25);
  reg("Identity", ONNX_DOMAIN, 25);
  reg("Reshape", ONNX_DOMAIN, 25);
  reg("Shape", ONNX_DOMAIN, 25);
  reg("Size", ONNX_DOMAIN, 25);
  reg("If", ONNX_DOMAIN, 25);
  reg("Loop", ONNX_DOMAIN, 25);
  reg("Scan", ONNX_DOMAIN, 25);
  reg("Pad", ONNX_DOMAIN, 25);
  reg("Squeeze", ONNX_DOMAIN, 25);
  reg("Transpose", ONNX_DOMAIN, 25);
  reg("Unsqueeze", ONNX_DOMAIN, 25);
  reg("QuantizeLinear", ONNX_DOMAIN, 25);
  reg("BitCast", ONNX_DOMAIN, 26);
  reg("CumProd", ONNX_DOMAIN, 26);
  reg("ArrayFeatureExtractor", AI_ONNX_ML_DOMAIN, 1);
  reg("Binarizer", AI_ONNX_ML_DOMAIN, 1);
  reg("CastMap", AI_ONNX_ML_DOMAIN, 1);
  reg("CategoryMapper", AI_ONNX_ML_DOMAIN, 1);
  reg("DictVectorizer", AI_ONNX_ML_DOMAIN, 1);
  reg("FeatureVectorizer", AI_ONNX_ML_DOMAIN, 1);
  reg("Imputer", AI_ONNX_ML_DOMAIN, 1);
  reg("LabelEncoder", AI_ONNX_ML_DOMAIN, 1);
  reg("LinearClassifier", AI_ONNX_ML_DOMAIN, 1);
  reg("LinearRegressor", AI_ONNX_ML_DOMAIN, 1);
  reg("Normalizer", AI_ONNX_ML_DOMAIN, 1);
  reg("OneHotEncoder", AI_ONNX_ML_DOMAIN, 1);
  reg("SVMClassifier", AI_ONNX_ML_DOMAIN, 1);
  reg("SVMRegressor", AI_ONNX_ML_DOMAIN, 1);
  reg("Scaler", AI_ONNX_ML_DOMAIN, 1);
  reg("TreeEnsembleClassifier", AI_ONNX_ML_DOMAIN, 1);
  reg("TreeEnsembleRegressor", AI_ONNX_ML_DOMAIN, 1);
  reg("ZipMap", AI_ONNX_ML_DOMAIN, 1);
  reg("LabelEncoder", AI_ONNX_ML_DOMAIN, 2);
  reg("TreeEnsembleClassifier", AI_ONNX_ML_DOMAIN, 3);
  reg("TreeEnsembleRegressor", AI_ONNX_ML_DOMAIN, 3);
  reg("LabelEncoder", AI_ONNX_ML_DOMAIN, 4);
  reg("TreeEnsemble", AI_ONNX_ML_DOMAIN, 5);
  reg("TreeEnsembleRegressor", AI_ONNX_ML_DOMAIN, 5);
  reg("TreeEnsembleClassifier", AI_ONNX_ML_DOMAIN, 5);
  reg("Gradient", AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1);
  reg("Momentum", AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1);
  reg("Adagrad", AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1);
  reg("Adam", AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1);
  reg("FlexAttention", AI_ONNX_PREVIEW_DOMAIN, 1);
}

} // namespace ONNX_LIGHT_NAMESPACE
