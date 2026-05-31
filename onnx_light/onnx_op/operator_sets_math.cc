// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_math_doc.h"

#include <iterator>
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
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
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
      {"Cos", [] { return BuildUnaryFloatMathSchemas("Cos", 22, 7); }},
      {"Cosh", [] { return BuildUnaryFloatMathSchemas("Cosh", 22, 9); }},
      {"Div", [] { return BuildElementwiseMathSchemaForVersion("Div"); }},
      {"Gemm", [] { return BuildGemmSchemas(); }},
      {"MatMul", [] { return BuildMatMulSchemas(); }},
      {"Mod", [] { return BuildModSchemas(); }},
      {"Mul", [] { return BuildElementwiseMathSchemaForVersion("Mul"); }},
      {"Pow", [] { return BuildPowSchemas(); }},
      {"Sigmoid", [] { return BuildSigmoidSchemas(); }},
      {"Sin", [] { return BuildUnaryFloatMathSchemas("Sin", 22, 7); }},
      {"Sinh", [] { return BuildUnaryFloatMathSchemas("Sinh", 22, 9); }},
      {"Softmax", [] { return BuildSoftmaxSchemas(); }},
      {"Sub", [] { return BuildElementwiseMathSchemaForVersion("Sub"); }},
      {"Tan", [] { return BuildUnaryFloatMathSchemas("Tan", 22, 7); }},
      {"Tanh", [] { return BuildTanhSchemas(); }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
