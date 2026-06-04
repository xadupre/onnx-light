// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_math_doc.h"

#include <iterator>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

std::vector<LightOpSchema> BuildElementwiseMathSchemaForVersion(const char *op_type) {
  return std::vector<LightOpSchema>{
      LightOpSchema(op_type, kOnnxDomain, 14, MakeElementwiseMathDoc(op_type, 14),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", AllNumericTypesIr4(),
                         "Constrain input and output types to all numeric tensors."},
                    }),

      LightOpSchema(op_type, kOnnxDomain, 13, MakeElementwiseMathDoc(op_type, 13),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReductionIr4(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, 7, MakeElementwiseMathDoc(op_type, 7),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReduction(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, 6, MakeElementwiseMathDoc(op_type, 6),
                    {
                        {"A", "First operand, should share the type with the second operand.", "T"},
                        {"B",
                         "Second operand. With broadcasting can be of smaller size than A. If "
                         "broadcasting is disabled it should be of the same size.",
                         "T"},
                    },
                    {
                        {"C", "Result, has same dimensions and type as A", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReduction(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(
          op_type, kOnnxDomain, 1, MakeElementwiseMathDoc(op_type, 1),
          {
              {"A", "First operand, should share the type with the second operand.", "T"},
              {"B",
               "Second operand. With broadcasting can be of smaller size than A. If broadcasting "
               "is disabled it should be of the same size.",
               "T"},
          },
          {
              {"C", "Result, has same dimensions and type as A", "T"},
          },
          {
              {"T", FloatTypes(), "Constrain input and output types to float tensors."},
          })};
}

std::vector<LightOpSchema> BuildModSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("Mod", kOnnxDomain, 13, "Performs an element-wise binary modulo operation.",
                    {
                        {"A", "Dividend tensor", "T"},
                        {"B", "Divisor tensor", "T"},
                    },
                    {
                        {"C", "Remainder tensor", "T"},
                    },
                    {
                        {"T", AllNumericTypesIr4(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Mod", kOnnxDomain, 10, "Performs element-wise binary modulus.",
                    {
                        {"A", "Dividend tensor", "T"},
                        {"B", "Divisor tensor", "T"},
                    },
                    {
                        {"C", "Remainder tensor", "T"},
                    },
                    {
                        {"T", AllNumericTypes(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildPowSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("Pow", kOnnxDomain, 7, MakePowDoc(),
                    {
                        {"X", "First operand, base of the exponent.", "T"},
                        {"Y", "Second operand, power of the exponent.", "T"},
                    },
                    {
                        {"Z", "Output tensor.", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Pow", kOnnxDomain, 1, MakePowDoc(),
                    {
                        {"X", "Input tensor of any shape, base of the exponent.", "T"},
                        {"Y",
                         "Input tensor of any shape broadcastable to X shape, the exponent "
                         "component.",
                         "T"},
                    },
                    {
                        {"Z", "Output tensor (same size as X)", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildDetSchemas() {
  static constexpr const char *kDetDoc = R"DOC(
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(LightOpSchema(
      "Det", kOnnxDomain, 22, kDetDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to floating-point tensors."},
      }));
  schemas.push_back(LightOpSchema(
      "Det", kOnnxDomain, 11, kDetDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T", FloatTypes(), "Constrain input and output types to floating-point tensors."},
      }));
  return schemas;
}

std::vector<LightOpSchema> BuildNegSchemas() {
  static constexpr const char *kNegDoc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";
  // ``Neg`` accepts only signed numeric tensors. The exact type set is
  // inlined here because no shared helper exposes it.
  const std::vector<TensorType> kNegTypesV13 = {
      TensorType::kFloat, TensorType::kInt32,   TensorType::kInt8,   TensorType::kInt16,
      TensorType::kInt64, TensorType::kFloat16, TensorType::kDouble, TensorType::kBfloat16,
  };
  const std::vector<TensorType> kNegTypesV6 = {
      TensorType::kFloat, TensorType::kInt32,   TensorType::kInt8,   TensorType::kInt16,
      TensorType::kInt64, TensorType::kFloat16, TensorType::kDouble,
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Neg", kOnnxDomain, 13, kNegDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T", kNegTypesV13, "Constrain input and output types to signed numeric tensors."},
      }));
  schemas.push_back(LightOpSchema(
      "Neg", kOnnxDomain, 6, kNegDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T", kNegTypesV6, "Constrain input and output types to signed numeric tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Neg", kOnnxDomain, 1, kNegDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildAbsSchemas() {
  static constexpr const char *kAbsDocV13 = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.
)DOC";
  static constexpr const char *kAbsDocV6 = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Abs", kOnnxDomain, 13, kAbsDocV13,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T", AllNumericTypesIr4(), "Constrain input and output types to all numeric tensors."},
      }));
  schemas.push_back(LightOpSchema(
      "Abs", kOnnxDomain, 6, kAbsDocV6,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T", AllNumericTypes(), "Constrain input and output types to all numeric tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Abs", kOnnxDomain, 1, kAbsDocV6,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildTanhSchemas() {
  static constexpr const char *kTanhDoc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";
  const std::string output_description = MakeUnaryMathOutputDescription("Tanh");
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Tanh", kOnnxDomain, 13, kTanhDoc,
      {
          {"input", "Input tensor", "T"},
      },
      {
          {"output", output_description, "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Tanh", kOnnxDomain, 6, kTanhDoc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Tanh", kOnnxDomain, 1, kTanhDoc,
                    {
                        {"input", "1-D input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildSeluSchemas() {
  static constexpr const char *kSeluDoc = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Selu", kOnnxDomain, 22, kSeluDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      },
      {
          {"alpha",
           "Coefficient of SELU default to 1.67326319217681884765625 "
           "(i.e., float32 approximation of 1.6732632423543772848170429916717).",
           AttributeType::FLOAT, /*required=*/false, 1.67326319217681884765625},
          {"gamma",
           "Coefficient of SELU default to 1.05070102214813232421875 "
           "(i.e., float32 approximation of 1.0507009873554804934193349852946).",
           AttributeType::FLOAT, /*required=*/false, 1.05070102214813232421875},
      },
      /*has_function_implementation=*/true));
  schemas.push_back(
      LightOpSchema("Selu", kOnnxDomain, 6, kSeluDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    {
                        {"alpha",
                         "Coefficient of SELU default to 1.67326319217681884765625 "
                         "(i.e., float32 approximation of 1.6732632423543772848170429916717).",
                         AttributeType::FLOAT, /*required=*/false, 1.67326319217681884765625},
                        {"gamma",
                         "Coefficient of SELU default to 1.05070102214813232421875 "
                         "(i.e., float32 approximation of 1.0507009873554804934193349852946).",
                         AttributeType::FLOAT, /*required=*/false, 1.05070102214813232421875},
                    },
                    /*has_function_implementation=*/true));
  schemas.push_back(
      LightOpSchema("Selu", kOnnxDomain, 1, kSeluDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    {
                        {"alpha", "Coefficient of SELU default to 1.6732.", AttributeType::FLOAT,
                         /*required=*/false, 1.6732},
                        {"gamma", "Coefficient of SELU default to 1.0507.", AttributeType::FLOAT,
                         /*required=*/false, 1.0507},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildSigmoidSchemas() {
  static constexpr const char *kSigmoidDoc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied
to the tensor element-wise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Sigmoid", kOnnxDomain, 13, kSigmoidDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Sigmoid", kOnnxDomain, 6, kSigmoidDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Sigmoid", kOnnxDomain, 1, kSigmoidDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildSoftplusSchemas() {
  static constexpr const char *kSoftplusDoc = R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(LightOpSchema(
      "Softplus", kOnnxDomain, 22, kSoftplusDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      },
      /*has_function_implementation=*/true));
  schemas.push_back(
      LightOpSchema("Softplus", kOnnxDomain, 1, kSoftplusDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    /*has_function_implementation=*/true));
  return schemas;
}

std::vector<LightOpSchema> BuildSoftsignSchemas() {
  static constexpr const char *kSoftsignDoc = R"DOC(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)DOC";
  static constexpr const char *kSoftsignOutputDescription =
      "The softsign (x/(1+|x|)) values of the input tensor computed element-wise";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(LightOpSchema(
      "Softsign", kOnnxDomain, 22, kSoftsignDoc,
      {
          {"input", "Input tensor", "T"},
      },
      {
          {"output", kSoftsignOutputDescription, "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      },
      /*has_function_implementation=*/true));
  schemas.push_back(
      LightOpSchema("Softsign", kOnnxDomain, 1, kSoftsignDoc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", kSoftsignOutputDescription, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    /*has_function_implementation=*/true));
  return schemas;
}

std::vector<LightOpSchema> BuildThresholdedReluSchemas() {
  static constexpr const char *kThresholdedReluDoc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(LightOpSchema(
      "ThresholdedRelu", kOnnxDomain, 22, kThresholdedReluDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      },
      {
          {"alpha", "Threshold value", AttributeType::FLOAT, /*required=*/false, 1.0},
      },
      /*has_function_implementation=*/true));
  schemas.push_back(
      LightOpSchema("ThresholdedRelu", kOnnxDomain, 10, kThresholdedReluDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    {
                        {"alpha", "Threshold value", AttributeType::FLOAT, /*required=*/false, 1.0},
                    },
                    /*has_function_implementation=*/true));
  return schemas;
}

std::vector<LightOpSchema> BuildPReluSchemas() {
  static constexpr const char *kPReluDoc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";
  static const std::vector<FormalParameter> v1_inputs = {
      {"X", "Input tensor", "T"},
      {"slope",
       "Slope tensor. If `Slope` is of size 1, the value is shared"
       "across different channels",
       "T"},
  };
  static const std::vector<FormalParameter> v7_inputs = {
      {"X", "Input tensor", "T"},
      {"slope",
       "Slope tensor. The shape of slope can be smaller than first input X; "
       "if so, its shape must be unidirectional broadcastable to X",
       "T"},
  };
  static const std::vector<FormalParameter> v7_outputs = {
      {"Y", "Output tensor (same size as X)", "T"},
  };
  static const std::vector<FormalParameter> v1_outputs = {
      {"Y", "Output tensor", "T"},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(5);
  schemas.push_back(LightOpSchema(
      "PRelu", kOnnxDomain, 16, kPReluDoc, v7_inputs, v7_outputs,
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble,
            TensorType::kUint32, TensorType::kUint64, TensorType::kInt32, TensorType::kInt64},
           "Constrain input and output types to float/int tensors."},
      },
      /*has_function_implementation=*/true));
  schemas.push_back(LightOpSchema(
      "PRelu", kOnnxDomain, 9, kPReluDoc, v7_inputs, v7_outputs,
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kUint32,
            TensorType::kUint64, TensorType::kInt32, TensorType::kInt64},
           "Constrain input and output types to float/int tensors."},
      }));
  schemas.push_back(
      LightOpSchema("PRelu", kOnnxDomain, 7, kPReluDoc, v7_inputs, v7_outputs,
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("PRelu", kOnnxDomain, 6, kPReluDoc, v1_inputs, v1_outputs,
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("PRelu", kOnnxDomain, 1, kPReluDoc, v1_inputs, v1_outputs,
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    },
                    std::vector<AttributeParam>{
                        AttributeParam{"consumed_inputs", "legacy optimization attribute.",
                                       AttributeType::INTS, false, std::monostate{}},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildSwishSchemas() {
  static constexpr const char *kSwishDoc = R"DOC(
Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
where $Swish(x) = x * sigmoid(alpha * x)$.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(1);
  schemas.push_back(LightOpSchema(
      "Swish", kOnnxDomain, 24, kSwishDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kBfloat16, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      },
      {
          {"alpha", "Coefficient to multiply with input before sigmoid.", AttributeType::FLOAT,
           /*required=*/false, 1.0},
      },
      /*has_function_implementation=*/true));
  return schemas;
}

std::vector<LightOpSchema> BuildSqrtSchemas() {
  static constexpr const char *kSqrtDoc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Sqrt", kOnnxDomain, 13, kSqrtDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Sqrt", kOnnxDomain, 6, kSqrtDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Sqrt", kOnnxDomain, 1, kSqrtDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

AttributeParam MakeSoftmaxAxisAttr(int64_t default_axis) {
  return AttributeParam{"axis",
                        "Describes the dimension Softmax will be performed on. "
                        "Negative value means counting dimensions from the back.",
                        AttributeType::INT, false, default_axis};
}

std::vector<LightOpSchema> BuildSoftmaxSchemas() {
  static constexpr const char *kSoftmaxDocV13 = R"DOC(
The operator computes the normalized exponential values for the given input.
The "axis" attribute indicates the dimension along which Softmax is
performed. The output tensor has the same shape as the input tensor.
)DOC";
  static constexpr const char *kSoftmaxDocV11 = R"DOC(
The operator computes the normalized exponential values for the given input.
The "axis" attribute indicates the dimension along which Softmax is
performed. The output tensor has the same shape as the input tensor.
)DOC";
  static constexpr const char *kSoftmaxDocV1 = R"DOC(
The operator computes the normalized exponential values for the given input.
Inputs are conceptually coerced to a 2D matrix and Softmax is applied on the
second dimension. The output tensor has the same shape as the input tensor.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Softmax", kOnnxDomain, 13, kSoftmaxDocV13,
      {
          {"input", "The input tensor of rank >= axis.", "T"},
      },
      {
          {"output", "The output values with the same shape as the input tensor.", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      },
      {
          MakeSoftmaxAxisAttr(-1),
      }));
  schemas.push_back(LightOpSchema(
      "Softmax", kOnnxDomain, 11, kSoftmaxDocV11,
      {
          {"input",
           "The input tensor that's coerced into a 2D matrix of size (NxD) as described above.",
           "T"},
      },
      {
          {"output",
           "The output values with the same shape as input tensor (the original size without "
           "coercion).",
           "T"},
      },
      {
          {"T", FloatTypes(), "Constrain input and output types to float tensors."},
      },
      {
          MakeSoftmaxAxisAttr(1),
      }));
  schemas.push_back(LightOpSchema(
      "Softmax", kOnnxDomain, 1, kSoftmaxDocV1,
      {
          {"input",
           "The input tensor that's coerced into a 2D matrix of size (NxD) as described above.",
           "T"},
      },
      {
          {"output",
           "The output values with the same shape as input tensor (the original size without "
           "coercion).",
           "T"},
      },
      {
          {"T", FloatTypes(), "Constrain input and output types to float tensors."},
      },
      {
          MakeSoftmaxAxisAttr(1),
      }));
  return schemas;
}

// --- SoftmaxCrossEntropyLoss -------------------------------------------------

static constexpr const char *kSoftmaxCrossEntropyLossDocV13 =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

* shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.
* shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
```
or
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
```

loss is zero for the case when label-value equals ignore_index.
```
l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
```

where:
```
p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
```

Finally, L is optionally reduced:

* If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
* If reduction = 'sum', the output is scalar: Sum(L).
* If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
  where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
)DOC";

static constexpr const char *kSoftmaxCrossEntropyLossDocV12 =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
)DOC";

LightOpSchema MakeSoftmaxCrossEntropyLossSchema(int since_version) {
  const char *doc =
      since_version >= 13 ? kSoftmaxCrossEntropyLossDocV13 : kSoftmaxCrossEntropyLossDocV12;
  std::vector<TensorType> t_types = {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  if (since_version >= 13) {
    t_types.push_back(TensorType::kBfloat16);
  }
  return LightOpSchema(
      "SoftmaxCrossEntropyLoss", kOnnxDomain, since_version, doc,
      {
          {"scores",
           "The predicted outputs with shape [batch_size, class_size], or "
           "[batch_size, class_size, D1, D2 , ..., Dk], where K is the number of dimensions.",
           "T"},
          {"labels",
           "The ground truth output tensor, with shape [batch_size], or "
           "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions. "
           "Labels element value shall be in range of [0, C). "
           "If ignore_index is specified, it may have a value outside [0, C) and the label "
           "values should either be "
           "in the range [0, C) or have the value ignore_index.",
           "Tind"},
          {"weights",
           "A manual rescaling weight given to each class. If given, it has to "
           "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
           "it is treated as if having all ones.",
           "T"},
      },
      {
          {"output",
           "Weighted loss float Tensor. If reduction is 'none', this has the "
           "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
           "K-dimensional loss. Otherwise, it is a scalar.",
           "T"},
          {"log_prob",
           "Log probability tensor. If the output of softmax is prob, its value is log(prob).",
           "T"},
      },
      {
          {"T", t_types, "Constrain input and output types to float tensors."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain target to integer types"},
      },
      {
          {"reduction",
           since_version >= 13 ? "Type of reduction to apply to loss: none, sum, mean(default). "
                                 "'none': no reduction will be applied, "
                                 "'sum': the output will be summed. "
                                 "'mean': the sum of the output will be divided by the number of "
                                 "elements in the output."
                               : "Type of reduction to apply to loss: none, sum, mean(default). "
                                 "'none': no reduction will be applied, "
                                 "'sum': the output will be summed. "
                                 "'mean': the sum of the output will be divided by the number of "
                                 "elements in the output.",
           AttributeType::STRING, /*required=*/false, std::string("mean")},
          {"ignore_index",
           "Specifies a target value that is ignored and does not contribute to the input "
           "gradient. It's an optional value.",
           AttributeType::INT, /*required=*/false},
      },
      /*has_function_implementation=*/true);
}

std::vector<LightOpSchema> BuildSoftmaxCrossEntropyLossSchemas() {
  return std::vector<LightOpSchema>{
      MakeSoftmaxCrossEntropyLossSchema(13),
      MakeSoftmaxCrossEntropyLossSchema(12),
  };
}

// --- NegativeLogLikelihoodLoss -----------------------------------------------

static constexpr const char *kNegativeLogLikelihoodLossDoc = R"DOC(
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```
)DOC";

LightOpSchema MakeNegativeLogLikelihoodLossSchema(int since_version) {
  std::vector<TensorType> t_types =
      since_version >= 22
          ? std::vector<TensorType>{TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,
                                    TensorType::kDouble}
          : std::vector<TensorType>{TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  return LightOpSchema(
      "NegativeLogLikelihoodLoss", kOnnxDomain, since_version, kNegativeLogLikelihoodLossDoc,
      {
          {"input", "Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).", "T"},
          {"target",
           "Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value shall be "
           "in range of [0, C). "
           "If ignore_index is specified, it may have a value outside [0, C) and the target "
           "values should either be "
           "in the range [0, C) or have the value ignore_index.",
           "Tind"},
          {"weight",
           "Optional rescaling weight tensor. "
           "If given, it has to be a tensor of size C. Otherwise, it is treated as if having "
           "all ones.",
           "T"},
      },
      {
          {"loss", "The negative log likelihood loss", "T"},
      },
      {
          {"T", t_types, "Constrain input, weight, and output types to floating-point tensors."},
          {"Tind", {TensorType::kInt32, TensorType::kInt64}, "Constrain target to integer types"},
      },
      {
          {"reduction",
           "Type of reduction to apply to loss: none, sum, mean (default). "
           "'none': the output is the loss for each sample. "
           "'sum': the output will be summed. "
           "'mean': the sum of the output will be divided by the sum of applied weights.",
           AttributeType::STRING, /*required=*/false, std::string("mean")},
          {"ignore_index",
           "Specifies a target value that is ignored and does not contribute to the input "
           "gradient. It's an optional value.",
           AttributeType::INT, /*required=*/false},
      },
      /*has_function_implementation=*/true);
}

std::vector<LightOpSchema> BuildNegativeLogLikelihoodLossSchemas() {
  return std::vector<LightOpSchema>{
      MakeNegativeLogLikelihoodLossSchema(22),
      MakeNegativeLogLikelihoodLossSchema(13),
      MakeNegativeLogLikelihoodLossSchema(12),
  };
}

std::vector<LightOpSchema> BuildUnaryFloatMathSchemas(const char *op_type, int latest_version,
                                                      int previous_version) {
  const std::string doc = MakeUnaryMathDoc(op_type);
  const std::string output_description = MakeUnaryMathOutputDescription(op_type);
  return std::vector<LightOpSchema>{
      LightOpSchema(op_type, kOnnxDomain, latest_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T",
                         {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,
                          TensorType::kDouble},
                         "Constrain input and output types to float tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, previous_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    })};
}

std::vector<LightOpSchema> BuildErfSchemas() {
  const std::string doc = MakeUnaryMathDoc("Erf");
  const std::string output_description = MakeUnaryMathOutputDescription("Erf");
  return std::vector<LightOpSchema>{
      LightOpSchema("Erf", kOnnxDomain, 13, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T",
                         {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,
                          TensorType::kDouble},
                         "Constrain input and output types to float tensors."},
                    }),
      LightOpSchema(
          "Erf", kOnnxDomain, 9, doc,
          {
              {"input", "Input tensor", "T"},
          },
          {
              {"output", output_description, "T"},
          },
          {
              {"T", AllNumericTypes(), "Constrain input and output types to all numeric tensors."},
          })};
}

std::vector<LightOpSchema> BuildFloorSchemas() {
  static constexpr const char *kFloorDocV13 = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
)DOC";
  static constexpr const char *kFloorDocV6 = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Floor", kOnnxDomain, 13, kFloorDocV13,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Floor", kOnnxDomain, 6, kFloorDocV6,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Floor", kOnnxDomain, 1, kFloorDocV6,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildClipSchemas() {
  static constexpr const char *kClipDocV11 = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";
  static constexpr const char *kClipDocV13 = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
When 'min' is greater than 'max', the clip operator sets all the 'input' values to
the value of 'max'. Thus, this is equivalent to 'Min(max, Max(input, min))'.
)DOC";
  static constexpr const char *kClipDocV1 = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC";

  const std::vector<FormalParameter> v11plus_inputs = {
      {"input", "Input tensor whose elements to be clipped", "T"},
      {"min",
       "Minimum value, under which element is replaced by min. "
       "It must be a scalar(tensor of empty shape).",
       "T"},
      {"max",
       "Maximum value, above which element is replaced by max. "
       "It must be a scalar(tensor of empty shape).",
       "T"},
  };
  const std::vector<FormalParameter> v6_inputs = {
      {"input", "Input tensor whose elements to be clipped", "T"},
  };
  const std::vector<FormalParameter> outputs = {
      {"output", "Output tensor with clipped input elements", "T"},
  };

  std::vector<LightOpSchema> schemas;
  schemas.reserve(5);
  schemas.push_back(LightOpSchema(
      "Clip", kOnnxDomain, 13, kClipDocV13, v11plus_inputs, outputs,
      {
          {"T", AllNumericTypesIr4(), "Constrain input and output types to all numeric tensors."},
      },
      /*attributes=*/{},
      /*has_function_implementation=*/true));
  schemas.push_back(LightOpSchema(
      "Clip", kOnnxDomain, 12, kClipDocV11, v11plus_inputs, outputs,
      {
          {"T", AllNumericTypes(), "Constrain input and output types to all numeric tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Clip", kOnnxDomain, 11, kClipDocV11, v11plus_inputs, outputs,
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(LightOpSchema(
      "Clip", kOnnxDomain, 6, kClipDocV1, v6_inputs, outputs,
      {
          {"T", FloatTypes(), "Constrain input and output types to float tensors."},
      },
      std::vector<AttributeParam>{
          AttributeParam{"min", "Minimum value, under which element is replaced by min",
                         AttributeType::FLOAT, false,
                         static_cast<double>(std::numeric_limits<float>::lowest())},
          AttributeParam{"max", "Maximum value, above which element is replaced by max",
                         AttributeType::FLOAT, false,
                         static_cast<double>(std::numeric_limits<float>::max())},
      }));
  schemas.push_back(LightOpSchema(
      "Clip", kOnnxDomain, 1, kClipDocV1, v6_inputs, outputs,
      {
          {"T", FloatTypes(), "Constrain input and output types to float tensors."},
      },
      std::vector<AttributeParam>{
          AttributeParam{"min", "Minimum value, under which element is replaced by min",
                         AttributeType::FLOAT, false, std::monostate{}},
          AttributeParam{"max", "Maximum value, above which element is replaced by max",
                         AttributeType::FLOAT, false, std::monostate{}},
          AttributeParam{"consumed_inputs", "legacy optimization attribute.", AttributeType::INTS,
                         false, std::monostate{}},
      }));
  return schemas;
}

std::vector<LightOpSchema> BuildCeilSchemas() {
  static constexpr const char *kCeilDocV13 = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
)DOC";
  static constexpr const char *kCeilDocV6 = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Ceil", kOnnxDomain, 13, kCeilDocV13,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Ceil", kOnnxDomain, 6, kCeilDocV6,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Ceil", kOnnxDomain, 1, kCeilDocV6,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildRoundSchemas() {
  static constexpr const char *kRoundDoc = R"DOC(
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
)DOC";
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(LightOpSchema(
      "Round", kOnnxDomain, 22, kRoundDoc,
      {
          {"X", "Input tensor", "T"},
      },
      {
          {"Y", "Output tensor", "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Round", kOnnxDomain, 11, kRoundDoc,
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildBlackmanWindowSchemas() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "BlackmanWindow", kOnnxDomain, 17, MakeBlackmanWindowDoc(),
          {
              {"size", "A scalar value indicating the length of the window.", "T1"},
          },
          {
              {"output", "A Blackman window with length: size. The output has the shape: [size].",
               "T2"},
          },
          {
              {"T1",
               {TensorType::kInt32, TensorType::kInt64},
               "Constrain the input size to int32_t or int64_t."},
              {"T2", AllNumericTypesIr4(), "Constrain output types to numeric tensors."},
          }),
  };
}

std::vector<LightOpSchema> BuildHannWindowSchemas() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "HannWindow", kOnnxDomain, 17, MakeHannWindowDoc(),
          {
              {"size", "A scalar value indicating the length of the window.", "T1"},
          },
          {
              {"output", "A Hann window with length: size. The output has the shape: [size].",
               "T2"},
          },
          {
              {"T1",
               {TensorType::kInt32, TensorType::kInt64},
               "Constrain the input size to int32_t or int64_t."},
              {"T2", AllNumericTypesIr4(), "Constrain output types to numeric tensors."},
          }),
  };
}

std::vector<LightOpSchema> BuildHammingWindowSchemas() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "HammingWindow", kOnnxDomain, 17, MakeHammingWindowDoc(),
          {
              {"size", "A scalar value indicating the length of the window.", "T1"},
          },
          {
              {"output", "A Hamming window with length: size. The output has the shape: [size].",
               "T2"},
          },
          {
              {"T1",
               {TensorType::kInt32, TensorType::kInt64},
               "Constrain the input size to int32_t or int64_t."},
              {"T2", AllNumericTypesIr4(), "Constrain output types to numeric tensors."},
          }),
  };
}

std::vector<LightOpSchema> BuildMelWeightMatrixSchemas() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "MelWeightMatrix", kOnnxDomain, 17, MakeMelWeightMatrixDoc(),
          {
              {"num_mel_bins", "The number of bands in the mel spectrum.", "T1"},
              {"dft_length",
               "The size of the original DFT. "
               "The size of the original DFT is used to infer the size of the onesided DFT, which "
               "is understood to be floor(dft_length/2) + 1, i.e. the spectrogram only contains "
               "the nonredundant DFT bins.",
               "T1"},
              {"sample_rate",
               "Samples per second of the input signal used to create the spectrogram. Used to "
               "figure out the frequencies corresponding to each spectrogram bin, which dictates "
               "how they are mapped into the mel scale.",
               "T1"},
              {"lower_edge_hertz",
               "Lower bound on the frequencies to be included in the mel spectrum. This "
               "corresponds to the lower edge of the lowest triangular band.",
               "T2"},
              {"upper_edge_hertz", "The desired top edge of the highest frequency band.", "T2"},
          },
          {
              {"output",
               "The Mel Weight Matrix. "
               "The output has the shape: [floor(dft_length/2) + 1][num_mel_bins].",
               "T3"},
          },
          {
              {"T1",
               {TensorType::kInt32, TensorType::kInt64},
               "Constrain to integer tensors."},
              {"T2",
               {TensorType::kFloat, TensorType::kFloat16, TensorType::kDouble,
                TensorType::kBfloat16},
               "Constrain to float tensors"},
              {"T3", AllNumericTypesIr4(), "Constrain to any numerical types."},
          }),
  };
}

std::vector<TensorType> MatMulGemmTypes(int since_version) {
  if (since_version >= 13) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kUint32,
            TensorType::kUint64,  TensorType::kInt32, TensorType::kInt64,  TensorType::kBfloat16};
  }
  if (since_version >= 9) {
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kUint32,
            TensorType::kUint64,  TensorType::kInt32, TensorType::kInt64};
  }
  return FloatTypes();
}

const char *MatMulGemmTypeDescription(int since_version) {
  return since_version >= 9 ? "Constrain input and output types to float/int tensors."
                            : "Constrain input and output types to float tensors.";
}

std::vector<LightOpSchema> BuildMatMulSchemas() {
  const std::string doc = MakeMatMulDoc();
  std::vector<LightOpSchema> schemas;
  for (int version : {13, 9, 1}) {
    schemas.push_back(
        LightOpSchema("MatMul", kOnnxDomain, version, doc,
                      {
                          {"A", "N-dimensional matrix A", "T"},
                          {"B", "N-dimensional matrix B", "T"},
                      },
                      {
                          {"Y", "Matrix multiply results from A * B", "T"},
                      },
                      {
                          {"T", MatMulGemmTypes(version), MatMulGemmTypeDescription(version)},
                      }));
  }
  return schemas;
}

std::vector<LightOpSchema> BuildUnaryFloatMathSchemasWithV1(const char *op_type, int latest_version,
                                                            int previous_version,
                                                            int oldest_version) {
  const std::string doc = MakeUnaryMathDoc(op_type);
  const std::string output_description = MakeUnaryMathOutputDescription(op_type);
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, latest_version, doc,
      {
          {"input", "Input tensor", "T"},
      },
      {
          {"output", output_description, "T"},
      },
      {
          {"T",
           {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema(op_type, kOnnxDomain, previous_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema(op_type, kOnnxDomain, oldest_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildLogSchemas() {
  const std::string doc = MakeUnaryMathDoc("Log");
  const std::string output_description = MakeUnaryMathOutputDescription("Log");
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(LightOpSchema(
      "Log", kOnnxDomain, 13, doc,
      {
          {"input", "Input tensor", "T"},
      },
      {
          {"output", output_description, "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      }));
  schemas.push_back(
      LightOpSchema("Log", kOnnxDomain, 6, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Log", kOnnxDomain, 1, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildGemmSchemas() {
  std::vector<LightOpSchema> schemas;
  // Gemm v13: optional C, bfloat16 added.
  // Gemm v11: optional C.
  for (int version : {13, 11}) {
    schemas.push_back(LightOpSchema(
        "Gemm", kOnnxDomain, version, MakeGemmDoc(version),
        {
            {"A",
             "Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA "
             "is non-zero.",
             "T"},
            {"B",
             "Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB "
             "is non-zero.",
             "T"},
            {"C",
             "Optional input tensor C. If not specified, the computation is done as if C is a "
             "scalar 0. The shape of C should be unidirectional broadcastable to (M, N).",
             "T"},
        },
        {
            {"Y", "Output tensor of shape (M, N).", "T"},
        },
        {
            {"T", MatMulGemmTypes(version), MatMulGemmTypeDescription(version)},
        }));
  }
  // Gemm v9, v7: required C, broadcast doc, shape A/B descriptions.
  for (int version : {9, 7}) {
    schemas.push_back(LightOpSchema(
        "Gemm", kOnnxDomain, version, MakeGemmDoc(version),
        {
            {"A",
             "Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA "
             "is non-zero.",
             "T"},
            {"B",
             "Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB "
             "is non-zero.",
             "T"},
            {"C",
             "Input tensor C. The shape of C should be unidirectional broadcastable to (M, N).",
             "T"},
        },
        {
            {"Y", "Output tensor of shape (M, N).", "T"},
        },
        {
            {"T", MatMulGemmTypes(version), MatMulGemmTypeDescription(version)},
        }));
  }
  // Gemm v6.
  schemas.push_back(LightOpSchema("Gemm", kOnnxDomain, 6, MakeGemmDoc(6),
                                  {
                                      {"A", "Input tensor A", "T"},
                                      {"B", "Input tensor B", "T"},
                                      {"C", "Input tensor C", "T"},
                                  },
                                  {
                                      {"Y", "Output tensor.", "T"},
                                  },
                                  {
                                      {"T", MatMulGemmTypes(6), MatMulGemmTypeDescription(6)},
                                  }));
  // Gemm v1.
  schemas.push_back(LightOpSchema("Gemm", kOnnxDomain, 1, MakeGemmDoc(1),
                                  {
                                      {"A", "Input tensor A", "T"},
                                      {"B", "Input tensor B", "T"},
                                      {"C", "Input tensor C, can be inplace.", "T"},
                                  },
                                  {
                                      {"Y", "Output tensor.", "T"},
                                  },
                                  {
                                      {"T", MatMulGemmTypes(1), MatMulGemmTypeDescription(1)},
                                  }));
  return schemas;
}

std::vector<LightOpSchema> BuildEinsumSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(1);

  // Einsum v12: introduced. Variadic homogeneous input "Inputs" (min arity 1),
  // single output "Output". The "equation" attribute is required and carries
  // the Einstein summation string. Type constraint ``T`` admits all numeric
  // tensor types.
  schemas.push_back(
      LightOpSchema("Einsum", kOnnxDomain, 12, MakeEinsumDoc(),
                    {
                        {"Inputs", "Operands", "T"},
                    },
                    {
                        {"Output", "Output tensor", "T"},
                    },
                    {
                        {"T", AllNumericTypes(),
                         "Constrain input and output types to all numerical tensor types."},
                    },
                    {AttributeParam{"equation", "Einsum expression string.", AttributeType::STRING,
                                    /*required=*/true, std::monostate{}}}));

  return schemas;
}

std::vector<LightOpSchema> BuildSumSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(4);

  const std::vector<TensorType> float_types_with_bf16 = {
      TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
  const std::vector<TensorType> float_types_no_bf16 = FloatTypes();

  // Sum v13: adds bfloat16 to the type constraint; multidirectional
  // (NumPy-style) broadcasting (since v8). Single variadic input
  // ``data_0``.
  schemas.push_back(LightOpSchema(
      "Sum", kOnnxDomain, 13, MakeSumDoc(13),
      {
          {"data_0", "List of tensors for sum.", "T"},
      },
      {
          {"sum", "Output tensor.", "T"},
      },
      {
          {"T", float_types_with_bf16, "Constrain input and output types to float tensors."},
      }));

  // Sum v8: introduces multidirectional broadcasting.
  schemas.push_back(LightOpSchema(
      "Sum", kOnnxDomain, 8, MakeSumDoc(8),
      {
          {"data_0", "List of tensors for sum.", "T"},
      },
      {
          {"sum", "Output tensor.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // Sum v6: same wording as v1; no broadcasting (all inputs must share shape).
  schemas.push_back(LightOpSchema(
      "Sum", kOnnxDomain, 6, MakeSumDoc(6),
      {
          {"data_0", "List of tensors for Sum.", "T"},
      },
      {
          {"sum", "Output tensor. Same dimension as inputs.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // Sum v1: original schema, no broadcasting.
  schemas.push_back(LightOpSchema(
      "Sum", kOnnxDomain, 1, MakeSumDoc(1),
      {
          {"data_0", "List of tensors for Sum.", "T"},
      },
      {
          {"sum", "Output tensor. Same dimension as inputs.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      },
      {AttributeParam{"consumed_inputs", "legacy optimization attribute.", AttributeType::INTS,
                      false, std::monostate{}}}));

  return schemas;
}

namespace {

// Shared schema builder for the variadic element-wise ``Max``/``Min``
// operators. Both share the same shape, version history (v1, v6, v8, v12,
// v13) and the same type-constraint progression -- only the docs and the
// output description differ.
std::vector<LightOpSchema> BuildVariadicMinMaxSchemas(const char *op_type, const char *output_name,
                                                      const std::string &input_doc_v1,
                                                      const std::string &output_doc_v1,
                                                      const std::string &input_doc_v8,
                                                      std::string (*doc_fn)(int)) {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(5);

  const std::vector<TensorType> numeric_types_ir4 = AllNumericTypesIr4();
  const std::vector<TensorType> numeric_types = AllNumericTypes();
  const std::vector<TensorType> float_types_no_bf16 = FloatTypes();

  // v13: type constraint widens to ``all_numeric_types_ir4`` (adds bfloat16).
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, 13, doc_fn(13),
      {
          {"data_0", input_doc_v8, "T"},
      },
      {
          {output_name, "Output tensor.", "T"},
      },
      {
          {"T", numeric_types_ir4, "Constrain input and output types to numeric tensors."},
      }));

  // v12: type constraint widens to all numeric types (introduced in opset 12).
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, 12, doc_fn(12),
      {
          {"data_0", input_doc_v8, "T"},
      },
      {
          {output_name, "Output tensor.", "T"},
      },
      {
          {"T", numeric_types, "Constrain input and output types to numeric tensors."},
      }));

  // v8: introduces multidirectional broadcasting; type constraint stays at
  // the float types from v6.
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, 8, doc_fn(8),
      {
          {"data_0", input_doc_v8, "T"},
      },
      {
          {output_name, "Output tensor.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // v6: same wording as v1; no broadcasting (all inputs must share shape).
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, 6, doc_fn(6),
      {
          {"data_0", input_doc_v1, "T"},
      },
      {
          {output_name, output_doc_v1, "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // v1: original schema, no broadcasting, legacy ``consumed_inputs`` attr.
  schemas.push_back(LightOpSchema(
      op_type, kOnnxDomain, 1, doc_fn(1),
      {
          {"data_0", input_doc_v1, "T"},
      },
      {
          {output_name, output_doc_v1, "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      },
      {AttributeParam{"consumed_inputs", "legacy optimization attribute.", AttributeType::INTS,
                      false, std::monostate{}}}));

  return schemas;
}

} // namespace

std::vector<LightOpSchema> BuildMaxSchemas() {
  return BuildVariadicMinMaxSchemas("Max", "max", "List of tensors for Max.",
                                    "Output tensor. Same dimension as inputs.",
                                    "List of tensors for max.", &MakeMaxDoc);
}

std::vector<LightOpSchema> BuildMinSchemas() {
  return BuildVariadicMinMaxSchemas("Min", "min", "List of tensors for Min",
                                    "Output tensor. Same dimension as inputs.",
                                    "List of tensors for min.", &MakeMinDoc);
}

std::vector<LightOpSchema> BuildMeanSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(4);

  const std::vector<TensorType> float_types_with_bf16 = {
      TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16};
  const std::vector<TensorType> float_types_no_bf16 = FloatTypes();

  // Mean v13: adds bfloat16 to the type constraint; multidirectional
  // (NumPy-style) broadcasting (since v8). Single variadic input ``data_0``.
  schemas.push_back(LightOpSchema(
      "Mean", kOnnxDomain, 13, MakeMeanDoc(13),
      {
          {"data_0", "List of tensors for mean.", "T"},
      },
      {
          {"mean", "Output tensor.", "T"},
      },
      {
          {"T", float_types_with_bf16, "Constrain input and output types to float tensors."},
      }));

  // Mean v8: introduces multidirectional broadcasting.
  schemas.push_back(LightOpSchema(
      "Mean", kOnnxDomain, 8, MakeMeanDoc(8),
      {
          {"data_0", "List of tensors for mean.", "T"},
      },
      {
          {"mean", "Output tensor.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // Mean v6: same wording as v1; no broadcasting (all inputs must share shape).
  schemas.push_back(LightOpSchema(
      "Mean", kOnnxDomain, 6, MakeMeanDoc(6),
      {
          {"data_0", "List of tensors for Mean.", "T"},
      },
      {
          {"mean", "Output tensor. Same dimension as inputs.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      }));

  // Mean v1: original schema, no broadcasting.
  schemas.push_back(LightOpSchema(
      "Mean", kOnnxDomain, 1, MakeMeanDoc(1),
      {
          {"data_0", "List of tensors for Mean.", "T"},
      },
      {
          {"mean", "Output tensor. Same dimension as inputs.", "T"},
      },
      {
          {"T", float_types_no_bf16, "Constrain input and output types to float tensors."},
      },
      {AttributeParam{"consumed_inputs", "legacy optimization attribute.", AttributeType::INTS,
                      false, std::monostate{}}}));

  return schemas;
}

std::vector<LightOpSchema> BuildCumSumSchemas() {
  const std::string doc = MakeCumSumDoc();
  const std::vector<AttributeParam> attrs = {
      AttributeParam{"exclusive",
                     "If set to 1 will return exclusive sum in which the top element is not "
                     "included. In other terms, if set to 1, the j-th output element would be the "
                     "sum of the first (j-1) elements. Otherwise, it would be the sum of the first "
                     "j elements.",
                     AttributeType::INT, false, static_cast<int64_t>(0)},
      AttributeParam{"reverse", "If set to 1 will perform the sums in reverse direction.",
                     AttributeType::INT, false, static_cast<int64_t>(0)},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  // CumSum v14 widens T with bfloat16 (numeric_types_for_math_reduction_ir4).
  schemas.push_back(LightOpSchema(
      "CumSum", kOnnxDomain, 14, doc,
      {
          {"x", "An input tensor that is to be processed.", "T"},
          {"axis",
           "A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means "
           "counting dimensions from the back.",
           "T2"},
      },
      {
          {"y", "Output tensor of the same type as 'x' with cumulative sums of the x's elements",
           "T"},
      },
      {
          {"T", NumericTypesForMathReductionIr4(),
           "Constrain input and output types to numeric tensors."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "axis tensor can be int32 or int64 only"},
      },
      attrs));
  schemas.push_back(LightOpSchema(
      "CumSum", kOnnxDomain, 11, doc,
      {
          {"x", "An input tensor that is to be processed.", "T"},
          {"axis",
           "A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means "
           "counting dimensions from the back.",
           "T2"},
      },
      {
          {"y", "Output tensor of the same type as 'x' with cumulative sums of the x's elements",
           "T"},
      },
      {
          {"T",
           {TensorType::kUint32, TensorType::kUint64, TensorType::kInt32, TensorType::kInt64,
            TensorType::kFloat, TensorType::kDouble},
           "Input can be of any tensor type."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "axis tensor can be int32 or int64 only"},
      },
      attrs));
  return schemas;
}

std::vector<LightOpSchema> BuildCumProdSchemas() {
  const std::string doc = MakeCumProdDoc();
  const std::vector<AttributeParam> attrs = {
      AttributeParam{"exclusive",
                     "If set to 1 will return exclusive product in which the top element is not "
                     "included. In other terms, if set to 1, the j-th output element would be the "
                     "product of the first (j-1) elements. Otherwise, it would be the product of "
                     "the first j elements.",
                     AttributeType::INT, false, static_cast<int64_t>(0)},
      AttributeParam{"reverse", "If set to 1 will perform the products in reverse direction.",
                     AttributeType::INT, false, static_cast<int64_t>(0)},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(1);
  schemas.push_back(LightOpSchema(
      "CumProd", kOnnxDomain, 26, doc,
      {
          {"x", "An input tensor that is to be processed.", "T"},
          {"axis",
           "A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means "
           "counting dimensions from the back.",
           "T2"},
      },
      {
          {"y",
           "Output tensor of the same type as 'x' with cumulative products of the x's elements",
           "T"},
      },
      {
          {"T", NumericTypesForMathReductionIr4(),
           "Constrain input and output types to numeric tensors."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "axis tensor can be int32 or int64 only"},
      },
      attrs));
  return schemas;
}

std::vector<LightOpSchema> BuildTopKSchemas() {
  const std::vector<TensorType> float_types = FloatTypes();
  const std::vector<FormalParameter> outputs = {
      {"Values",
       "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing top K "
       "values from the input tensor",
       "T"},
      {"Indices",
       "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing the "
       "corresponding input tensor indices for the top K values.",
       "I"},
  };
  const std::vector<FormalParameter> v10plus_inputs = {
      {"X", "Tensor of shape [a_0, a_1, ..., a_{n-1}]", "T"},
      {"K",
       "A 1-D tensor containing a single positive value corresponding to the number of top "
       "elements to retrieve",
       "tensor(int64)"},
  };
  const std::vector<FormalParameter> v1_inputs = {
      {"X", "Tensor of shape [a_0, a_1, ..., a_{n-1}]", "T"},
  };

  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);

  // TopK v11: K is a 1-D tensor input; supports all numeric types and adds the
  // ``largest`` and ``sorted`` attributes.
  schemas.push_back(LightOpSchema(
      "TopK", kOnnxDomain, 11, MakeTopKDoc(11), v10plus_inputs, outputs,
      {
          {"T", AllNumericTypes(), "Constrain input and output types to numeric tensors."},
          {"I", {TensorType::kInt64}, "Constrain index tensor to int64"},
      },
      std::vector<AttributeParam>{
          AttributeParam{"axis",
                         "Dimension on which to do the sort. Negative value means "
                         "counting dimensions from the back. Accepted range is [-r, r-1] "
                         "where r = rank(input).",
                         AttributeType::INT, false, static_cast<int64_t>(-1)},
          AttributeParam{"largest", "Whether to return the top-K largest or smallest elements.",
                         AttributeType::INT, false, static_cast<int64_t>(1)},
          AttributeParam{"sorted", "Whether to return the elements in sorted order.",
                         AttributeType::INT, false, static_cast<int64_t>(1)},
      }));

  // TopK v10: K is a 1-D tensor input; float types only.
  schemas.push_back(
      LightOpSchema("TopK", kOnnxDomain, 10, MakeTopKDoc(10), v10plus_inputs, outputs,
                    {
                        {"T", float_types, "Constrain input and output types to float tensors."},
                        {"I", {TensorType::kInt64}, "Constrain index tensor to int64"},
                    },
                    std::vector<AttributeParam>{
                        AttributeParam{"axis", "Dimension on which to do the sort.",
                                       AttributeType::INT, false, static_cast<int64_t>(-1)},
                    }));

  // TopK v1: ``k`` is a required integer attribute; float types only.
  schemas.push_back(LightOpSchema(
      "TopK", kOnnxDomain, 1, MakeTopKDoc(1), v1_inputs, outputs,
      {
          {"T", float_types, "Constrain input and output types to float tensors."},
          {"I", {TensorType::kInt64}, "Constrain index tensor to int64"},
      },
      std::vector<AttributeParam>{
          AttributeParam{"k", "Number of top elements to retrieve", AttributeType::INT,
                         /*required=*/true, std::monostate{}},
          AttributeParam{"axis", "Dimension on which to do the sort.", AttributeType::INT, false,
                         static_cast<int64_t>(-1)},
      }));

  return schemas;
}

// Mirrors OpSchema::all_float_types_ir4() ordering used by the upstream DFT v20 schema (T1).
std::vector<TensorType> DFTFloatTypesVer20() {
  return {
      TensorType::kBfloat16,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

// Mirrors the explicit type list used by the upstream DFT v17 schema (T1), which keeps
// bfloat16 last rather than first.
std::vector<TensorType> DFTFloatTypesVer17() {
  return {
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  };
}

std::vector<LightOpSchema> BuildDFTSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);

  const std::string input_desc_v20 =
      "For real input, the following shape is expected: "
      "`[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]`. "
      "For complex input, the following shape is expected: "
      "`[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]`. "
      "The final dimension represents the real and imaginary parts of the value in that order.";
  const std::string dft_length_desc_v20 =
      "The length of the signal as a scalar. "
      "If greater than the axis dimension, the signal will be zero-padded up to `dft_length`. "
      "If less than the axis dimension, only the first `dft_length` values will be used as the "
      "signal. "
      "If not provided, the default `dft_length = signal_dim_axis`, except for the IRFFT case "
      "(`onesided=1`, `inverse=1`), in which case the default dft_length is "
      "`2 * (signal_dim_axis - 1)`.";
  const std::string output_desc_v20 =
      "The Fourier Transform of the input vector. "
      "For standard DFT (`onesided=0`), the output shape is: "
      "`[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]` (complex), with "
      "`signal_dim_axis = dft_length`. "
      "For RFFT (`onesided=1`, `inverse=0`), the output shape is: "
      "`[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]` (one-sided complex), "
      "with `signal_dim_axis = floor(dft_length/2) + 1`. "
      "For IRFFT (`onesided=1`, `inverse=1`), the output shape is: "
      "`[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]` (real), where "
      "`signal_dim_axis = dft_length`.";
  const std::string input_desc_v17 =
      "For real input, the following shape is expected: "
      "[batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]. "
      "For complex input, the following shape is expected: "
      "[batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. "
      "The first dimension is the batch dimension. "
      "The following N dimensions correspond to the signal's dimensions. "
      "The final dimension represents the real and imaginary parts of the value in that order.";
  const std::string dft_length_desc_v17 =
      "The length of the signal as a scalar. "
      "If greater than the axis dimension, the signal will be zero-padded up to dft_length. "
      "If less than the axis dimension, only the first dft_length values will be used as the "
      "signal. "
      "If not provided, the default dft_length = signal_dim_axis, except for the IRFFT case "
      "(onesided=1, inverse=1), in which case the default dft_length is 2 * (signal_dim_axis - "
      "1). "
      "It's an optional value.";
  const std::string output_desc_v17 =
      "The Fourier Transform of the input vector. "
      "For standard DFT (onesided=0), the output shape is: "
      "[batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2] (complex), with "
      "signal_dim_axis = dft_length. "
      "For RFFT (onesided=1, inverse=0), the output shape is: "
      "[batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2] (one-sided complex), "
      "with signal_dim_axis = floor(dft_length/2) + 1. "
      "For IRFFT (onesided=1, inverse=1), the output shape is: "
      "[batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1] (real), where "
      "signal_dim_axis = dft_length.";
  const std::string onesided_desc =
      "If `onesided` is `1`, only values for `k` in `[0, 1, 2, ..., floor(n_fft/2) + 1]` are "
      "used or returned because the real-to-complex Fourier transform satisfies the conjugate "
      "symmetry, i.e., `X[m, k] = X[m, n_fft-k]*`, where `m` denotes \"all other dimensions\" "
      "DFT was not applied on. When `onesided=1` and `inverse=0` (forward DFT), only real input "
      "is supported and a one-sided complex spectrum is returned (RFFT). When `onesided=1` and "
      "`inverse=1` (inverse DFT), only complex input is supported and a full real signal is "
      "returned (IRFFT). Value can be `0` or `1`. Default is `0`.";
  const std::string inverse_desc =
      "Whether to perform the inverse discrete Fourier Transform. Default is 0, which "
      "corresponds to `false`.";

  // DFT v20: axis becomes an optional input. Only ``onesided`` and ``inverse``
  // remain as attributes.
  schemas.push_back(LightOpSchema(
      "DFT", kOnnxDomain, 20, MakeDFTDoc(20),
      {
          {"input", input_desc_v20, "T1"},
          {"dft_length", dft_length_desc_v20, "T2"},
          {"axis",
           "The axis as a scalar on which to perform the DFT. Default is `-2` (last signal "
           "axis). Negative value means counting dimensions from the back. Accepted range is "
           "$[-r, -2] \\cup [0, r-2]$ where `r = rank(input)`. The last dimension is for "
           "representing complex numbers and thus is an invalid axis.",
           "tensor(int64)"},
      },
      {
          {"output", output_desc_v20, "T1"},
      },
      {
          {"T1", DFTFloatTypesVer20(), "Constrain input and output types to float tensors."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "Constrain scalar length types to integers."},
      },
      {
          AttributeParam{"onesided", onesided_desc, AttributeType::INT, /*required=*/false,
                         static_cast<int64_t>(0)},
          AttributeParam{"inverse", inverse_desc, AttributeType::INT, /*required=*/false,
                         static_cast<int64_t>(0)},
      }));

  // DFT v17: axis is an INT attribute (default 1).
  schemas.push_back(LightOpSchema(
      "DFT", kOnnxDomain, 17, MakeDFTDoc(17),
      {
          {"input", input_desc_v17, "T1"},
          {"dft_length", dft_length_desc_v17, "T2"},
      },
      {
          {"output", output_desc_v17, "T1"},
      },
      {
          {"T1", DFTFloatTypesVer17(), "Constrain input and output types to float tensors."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "Constrain scalar length types to int64_t."},
      },
      {
          AttributeParam{"axis",
                         "The axis on which to perform the DFT. By default this value is set to "
                         "1, which corresponds to the first dimension after the batch index. "
                         "Negative value means counting dimensions from the back. Accepted range "
                         "is $[-r, -2] \\cup [0, r-2]$ where `r = rank(input)`. The last "
                         "dimension is for representing complex numbers and thus is an invalid "
                         "axis.",
                         AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
          AttributeParam{"inverse",
                         "Whether to perform the inverse discrete fourier transform. By default "
                         "this value is set to 0, which corresponds to false.",
                         AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
          AttributeParam{"onesided",
                         "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) "
                         "+ 1] are used or returned because the real-to-complex Fourier "
                         "transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m, "
                         "n_fft-w]*. When onesided=1 and inverse=0 (forward DFT), only real "
                         "input is supported and a one-sided complex spectrum is returned "
                         "(RFFT). When onesided=1 and inverse=1 (inverse DFT), only complex "
                         "input is supported and a full real signal is returned (IRFFT). When "
                         "invoked with real or complex valued input, the default value is 0. "
                         "Values can be 0 or 1.",
                         AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
      }));

  return schemas;
}

std::vector<LightOpSchema> BuildSTFTSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(1);

  const std::string signal_desc =
      "Input tensor representing a real or complex valued signal. "
      "For real input, the following shape is expected: [batch_size][signal_length][1]. "
      "For complex input, the following shape is expected: "
      "[batch_size][signal_length][2], where "
      "[batch_size][signal_length][0] represents the real component and "
      "[batch_size][signal_length][1] represents the imaginary component of the signal.";
  const std::string frame_step_desc = "The number of samples to step between successive DFTs.";
  const std::string window_desc =
      "A tensor representing the window that will be slid over the signal."
      "The window must have rank 1 with shape: [window_shape]. "
      "It's an optional value. ";
  const std::string frame_length_desc = "A scalar representing the size of the DFT. "
                                        "It's an optional value.";
  const std::string output_desc =
      "The Short-time Fourier Transform of the signals."
      "If onesided is 1, the output has the shape: [batch_size][frames][dft_unique_bins][2], "
      "where dft_unique_bins is frame_length // 2 + 1 (the unique components of the DFT) "
      "If onesided is 0, the output has the shape: [batch_size][frames][frame_length][2], "
      "where frame_length is the length of the DFT.";
  const std::string onesided_desc =
      "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are "
      "returned because "
      "the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, "
      "w] = X[m,w]=X[m,n_fft-w]*. "
      "Note if the input or window tensors are complex, then onesided output is not "
      "possible. "
      "Enabling onesided with real inputs performs a Real-valued fast Fourier transform "
      "(RFFT)."
      "When invoked with real or complex valued input, the default value is 1. "
      "Values can be 0 or 1.";

  schemas.push_back(LightOpSchema(
      "STFT", kOnnxDomain, 17, MakeSTFTDoc(17),
      {
          {"signal", signal_desc, "T1"},
          {"frame_step", frame_step_desc, "T2"},
          {"window", window_desc, "T1"},
          {"frame_length", frame_length_desc, "T2"},
      },
      {
          {"output", output_desc, "T1"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain signal and output to float tensors."},
          {"T2",
           {TensorType::kInt32, TensorType::kInt64},
           "Constrain scalar length types to int64_t."},
      },
      {
          AttributeParam{"onesided", onesided_desc, AttributeType::INT, /*required=*/false,
                         static_cast<int64_t>(1)},
      }));

  return schemas;
}

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory(const std::string &op_type,
                                                              bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"Abs", [] { return BuildAbsSchemas(); }},
      {"Acos", [] { return BuildUnaryFloatMathSchemas("Acos", 22, 7); }},
      {"Acosh", [] { return BuildUnaryFloatMathSchemas("Acosh", 22, 9); }},
      {"Add", [] { return BuildElementwiseMathSchemaForVersion("Add"); }},
      {"Asin", [] { return BuildUnaryFloatMathSchemas("Asin", 22, 7); }},
      {"Asinh", [] { return BuildUnaryFloatMathSchemas("Asinh", 22, 9); }},
      {"Atan", [] { return BuildUnaryFloatMathSchemas("Atan", 22, 7); }},
      {"Atanh", [] { return BuildUnaryFloatMathSchemas("Atanh", 22, 9); }},
      {"BlackmanWindow", [] { return BuildBlackmanWindowSchemas(); }},
      {"Ceil", [] { return BuildCeilSchemas(); }},
      {"Clip", [] { return BuildClipSchemas(); }},
      {"Cos", [] { return BuildUnaryFloatMathSchemas("Cos", 22, 7); }},
      {"Cosh", [] { return BuildUnaryFloatMathSchemas("Cosh", 22, 9); }},
      {"CumProd", [] { return BuildCumProdSchemas(); }},
      {"CumSum", [] { return BuildCumSumSchemas(); }},
      {"DFT", [] { return BuildDFTSchemas(); }},
      {"Det", [] { return BuildDetSchemas(); }},
      {"Div", [] { return BuildElementwiseMathSchemaForVersion("Div"); }},
      {"Einsum", [] { return BuildEinsumSchemas(); }},
      {"Erf", [] { return BuildErfSchemas(); }},
      {"Exp", [] { return BuildUnaryFloatMathSchemasWithV1("Exp", 13, 6, 1); }},
      {"Floor", [] { return BuildFloorSchemas(); }},
      {"Gemm", [] { return BuildGemmSchemas(); }},
      {"HammingWindow", [] { return BuildHammingWindowSchemas(); }},
      {"HannWindow", [] { return BuildHannWindowSchemas(); }},
      {"Log", [] { return BuildLogSchemas(); }},
      {"MatMul", [] { return BuildMatMulSchemas(); }},
      {"Max", [] { return BuildMaxSchemas(); }},
      {"Mean", [] { return BuildMeanSchemas(); }},
      {"MelWeightMatrix", [] { return BuildMelWeightMatrixSchemas(); }},
      {"Min", [] { return BuildMinSchemas(); }},
      {"Mod", [] { return BuildModSchemas(); }},
      {"Mul", [] { return BuildElementwiseMathSchemaForVersion("Mul"); }},
      {"Neg", [] { return BuildNegSchemas(); }},
      {"NegativeLogLikelihoodLoss", [] { return BuildNegativeLogLikelihoodLossSchemas(); }},
      {"PRelu", [] { return BuildPReluSchemas(); }},
      {"Pow", [] { return BuildPowSchemas(); }},
      {"Round", [] { return BuildRoundSchemas(); }},
      {"Selu", [] { return BuildSeluSchemas(); }},
      {"Sigmoid", [] { return BuildSigmoidSchemas(); }},
      {"Sin", [] { return BuildUnaryFloatMathSchemas("Sin", 22, 7); }},
      {"Sinh", [] { return BuildUnaryFloatMathSchemas("Sinh", 22, 9); }},
      {"Softmax", [] { return BuildSoftmaxSchemas(); }},
      {"SoftmaxCrossEntropyLoss", [] { return BuildSoftmaxCrossEntropyLossSchemas(); }},
      {"Softplus", [] { return BuildSoftplusSchemas(); }},
      {"Softsign", [] { return BuildSoftsignSchemas(); }},
      {"Sqrt", [] { return BuildSqrtSchemas(); }},
      {"STFT", [] { return BuildSTFTSchemas(); }},
      {"Sub", [] { return BuildElementwiseMathSchemaForVersion("Sub"); }},
      {"Sum", [] { return BuildSumSchemas(); }},
      {"Swish", [] { return BuildSwishSchemas(); }},
      {"Tan", [] { return BuildUnaryFloatMathSchemas("Tan", 22, 7); }},
      {"Tanh", [] { return BuildTanhSchemas(); }},
      {"ThresholdedRelu", [] { return BuildThresholdedReluSchemas(); }},
      {"TopK", [] { return BuildTopKSchemas(); }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
