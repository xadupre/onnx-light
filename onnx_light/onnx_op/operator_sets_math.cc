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

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas;
  for (const auto &op_type : {"Add", "Div", "Mul", "Sub"}) {
    std::vector<LightOpSchema> bin_schemas = BuildElementwiseMathSchemaForVersion(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_schemas.begin()),
                   std::make_move_iterator(bin_schemas.end()));
  }
  std::vector<LightOpSchema> mod_schemas = BuildModSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(mod_schemas.begin()),
                 std::make_move_iterator(mod_schemas.end()));
  std::vector<LightOpSchema> pow_schemas = BuildPowSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(pow_schemas.begin()),
                 std::make_move_iterator(pow_schemas.end()));
  std::vector<LightOpSchema> abs_schemas = BuildAbsSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(abs_schemas.begin()),
                 std::make_move_iterator(abs_schemas.end()));
  std::vector<LightOpSchema> sin_schemas = BuildUnaryFloatMathSchemas("Sin", 22, 7);
  schemas.insert(schemas.end(), std::make_move_iterator(sin_schemas.begin()),
                 std::make_move_iterator(sin_schemas.end()));
  std::vector<LightOpSchema> cos_schemas = BuildUnaryFloatMathSchemas("Cos", 22, 7);
  schemas.insert(schemas.end(), std::make_move_iterator(cos_schemas.begin()),
                 std::make_move_iterator(cos_schemas.end()));
  std::vector<LightOpSchema> sinh_schemas = BuildUnaryFloatMathSchemas("Sinh", 22, 9);
  schemas.insert(schemas.end(), std::make_move_iterator(sinh_schemas.begin()),
                 std::make_move_iterator(sinh_schemas.end()));
  std::vector<LightOpSchema> cosh_schemas = BuildUnaryFloatMathSchemas("Cosh", 22, 9);
  schemas.insert(schemas.end(), std::make_move_iterator(cosh_schemas.begin()),
                 std::make_move_iterator(cosh_schemas.end()));
  std::vector<LightOpSchema> blackman_window_schemas = BuildBlackmanWindowSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(blackman_window_schemas.begin()),
                 std::make_move_iterator(blackman_window_schemas.end()));
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
