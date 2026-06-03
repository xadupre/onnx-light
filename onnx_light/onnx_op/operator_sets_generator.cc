// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_generator_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace generator {

namespace {

std::vector<TensorType> AllTensorTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,     TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,      TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,     TensorType::kString,
      TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<TensorType> AllTensorTypesIr4() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<TensorType> AllTensorTypesIr9() {
  return {
      TensorType::kUint8,      TensorType::kUint16,         TensorType::kUint32,
      TensorType::kUint64,     TensorType::kInt8,           TensorType::kInt16,
      TensorType::kInt32,      TensorType::kInt64,          TensorType::kBfloat16,
      TensorType::kFloat16,    TensorType::kFloat,          TensorType::kDouble,
      TensorType::kString,     TensorType::kBool,           TensorType::kComplex64,
      TensorType::kComplex128, TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
  };
}

std::vector<TensorType> AllTensorTypesIr10() {
  std::vector<TensorType> types = AllTensorTypesIr9();
  types.push_back(TensorType::kUint4);
  types.push_back(TensorType::kInt4);
  return types;
}

std::vector<TensorType> AllTensorTypesIr11() {
  std::vector<TensorType> types = AllTensorTypesIr10();
  types.push_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<TensorType> AllTensorTypesIr12() {
  std::vector<TensorType> types = AllTensorTypesIr11();
  types.push_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<TensorType> AllTensorTypesIr13() {
  std::vector<TensorType> types = AllTensorTypesIr12();
  types.push_back(TensorType::kUint2);
  types.push_back(TensorType::kInt2);
  return types;
}

std::vector<TensorType> ConstantTypes(int since_version) {
  switch (since_version) {
  case 25:
    return AllTensorTypesIr13();
  case 24:
    return AllTensorTypesIr12();
  case 23:
    return AllTensorTypesIr11();
  case 21:
    return AllTensorTypesIr10();
  case 19:
    return AllTensorTypesIr9();
  case 13:
    return AllTensorTypesIr4();
  case 12:
  case 11:
  case 9:
    return AllTensorTypes();
  case 1:
    return FloatTypes();
  default:
    throw SchemaError("Unsupported Constant since_version: " + std::to_string(since_version));
  }
}

// Bernoulli T1 (input) and T2 (output) type-constraint sets, indexed by
// since_version. Mirrors the upstream ``onnx`` ``Bernoulli`` schema history
// (see ``onnx_light/onnx_lib/defs/generator/{defs,old}.cc``).
std::vector<TensorType> BernoulliT1(int since_version) {
  switch (since_version) {
  case 22:
    // OpSchema::all_float_types_ir4()
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  case 15:
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  default:
    throw SchemaError("Unsupported Bernoulli since_version: " + std::to_string(since_version));
  }
}

std::vector<TensorType> BernoulliT2(int since_version) {
  switch (since_version) {
  case 22:
    // OpSchema::all_non_complex_numeric_types_plus_bool_ir4()
    return {
        TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64,
        TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,
        TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
        TensorType::kBool,
    };
  case 15:
    return {
        TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kBfloat16,
        TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
        TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
        TensorType::kBool,
    };
  default:
    throw SchemaError("Unsupported Bernoulli since_version: " + std::to_string(since_version));
  }
}

LightOpSchema MakeBernoulliSchema(int since_version) {
  return LightOpSchema(
      "Bernoulli", kOnnxDomain, since_version, MakeBernoulliDoc(),
      {
          {"input", "All values in input have to be in the range:[0, 1].", "T1"},
      },
      {
          {"output",
           "The returned output tensor only has values 0 or 1, same shape as input tensor.", "T2"},
      },
      {
          {"T1", BernoulliT1(since_version), "Constrain input types to float tensors."},
          {"T2", BernoulliT2(since_version),
           "Constrain output types to all numeric tensors and bool tensors."},
      },
      {
          AttributeParam{"seed",
                         "(Optional) Seed to the random generator, if not specified we will auto "
                         "generate one.",
                         AttributeType::FLOAT, /*required=*/false, std::monostate{}},
          AttributeParam{"dtype",
                         "The data type for the elements of the output tensor. if not specified, "
                         "we will use the data type of the input tensor.",
                         AttributeType::INT, /*required=*/false, std::monostate{}},
      });
}

LightOpSchema MakeConstantSchema(int since_version) {
  return LightOpSchema(
      "Constant", kOnnxDomain, since_version, MakeConstantDoc(since_version), {},
      {
          {"output", "Output tensor containing the same value of the provided tensor.", "T"},
      },
      {
          {"T", ConstantTypes(since_version),
           since_version == 1 ? "Constrain input and output types to float tensors."
                              : "Constrain input and output types to all tensor types."},
      });
}

// ConstantOfShape T2 type-constraint sets, indexed by since_version. Mirrors
// the upstream ``onnx`` ``ConstantOfShape`` schema history (see
// ``onnx_light/onnx_lib/defs/generator/{defs,old}.cc``).
std::vector<TensorType> ConstantOfShapeTypesV9() {
  return {
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
      TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
      TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool,
  };
}

std::vector<TensorType> ConstantOfShapeTypesV20() {
  std::vector<TensorType> types = ConstantOfShapeTypesV9();
  types.push_back(TensorType::kBfloat16);
  types.push_back(TensorType::kFloat8e4m3fn);
  types.push_back(TensorType::kFloat8e4m3fnuz);
  types.push_back(TensorType::kFloat8e5m2);
  types.push_back(TensorType::kFloat8e5m2fnuz);
  return types;
}

// From v21 upstream inserts ``uint4, int4`` *before* ``bool`` in the T2 list.
std::vector<TensorType> ConstantOfShapeTypesV21() {
  return {
      TensorType::kFloat16,        TensorType::kFloat,          TensorType::kDouble,
      TensorType::kInt8,           TensorType::kInt16,          TensorType::kInt32,
      TensorType::kInt64,          TensorType::kUint8,          TensorType::kUint16,
      TensorType::kUint32,         TensorType::kUint64,         TensorType::kUint4,
      TensorType::kInt4,           TensorType::kBool,           TensorType::kBfloat16,
      TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
      TensorType::kFloat8e5m2fnuz,
  };
}

std::vector<TensorType> ConstantOfShapeTypesV23() {
  std::vector<TensorType> types = ConstantOfShapeTypesV21();
  types.push_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<TensorType> ConstantOfShapeTypesV24() {
  std::vector<TensorType> types = ConstantOfShapeTypesV23();
  types.push_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<TensorType> ConstantOfShapeTypesV25() {
  std::vector<TensorType> types = ConstantOfShapeTypesV24();
  types.push_back(TensorType::kUint2);
  types.push_back(TensorType::kInt2);
  return types;
}

std::vector<TensorType> ConstantOfShapeTypes(int since_version) {
  switch (since_version) {
  case 25:
    return ConstantOfShapeTypesV25();
  case 24:
    return ConstantOfShapeTypesV24();
  case 23:
    return ConstantOfShapeTypesV23();
  case 21:
    return ConstantOfShapeTypesV21();
  case 20:
    return ConstantOfShapeTypesV20();
  case 9:
    return ConstantOfShapeTypesV9();
  default:
    throw SchemaError("Unsupported ConstantOfShape since_version: " +
                      std::to_string(since_version));
  }
}

std::vector<TensorType> EyeLikeTypesV9() {
  return {
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
      TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
      TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool,
  };
}

std::vector<TensorType> EyeLikeTypesV22() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBool,
  };
}

std::vector<TensorType> EyeLikeTypes(int since_version) {
  switch (since_version) {
  case 22:
    return EyeLikeTypesV22();
  case 9:
    return EyeLikeTypesV9();
  default:
    throw SchemaError("Unsupported EyeLike since_version: " + std::to_string(since_version));
  }
}

LightOpSchema MakeConstantOfShapeSchema(int since_version) {
  const char *t2_desc = (since_version >= 21) ? "Constrain output types to be numerics or boolean."
                                              : "Constrain output types to be numerics.";
  return LightOpSchema(
      "ConstantOfShape", kOnnxDomain, since_version, MakeConstantOfShapeDoc(since_version),
      {
          {"input",
           "1D tensor. The shape of the expected output tensor. If empty tensor is given, the "
           "output would be a scalar. All values must be >= 0.",
           "T1"},
      },
      {
          {"output",
           "Output tensor of shape specified by 'input'."
           "If attribute 'value' is specified, the value and datatype of the output tensor is "
           "taken from 'value'."
           "If attribute 'value' is not specified, the value in the output defaults to 0, and "
           "the datatype defaults to float32.",
           "T2"},
      },
      {
          {"T1", {TensorType::kInt64}, "Constrain input types."},
          {"T2", ConstantOfShapeTypes(since_version), t2_desc},
      },
      {
          AttributeParam{"value",
                         "(Optional) The value of the output elements."
                         "Should be a one-element tensor. If not specified, it defaults to a "
                         "tensor of value 0 and datatype float32",
                         AttributeType::TENSOR, /*required=*/false, std::monostate{}},
      });
}

LightOpSchema MakeEyeLikeSchema(int since_version) {
  const std::string dtype_doc =
      (since_version >= 22)
          ? "(Optional) The data type for the elements of the output tensor. If not specified,"
            " the data type of the input tensor T1 is used."
          : "(Optional) The data type for the elements of the output tensor. If not specified,"
            " the data type of the input tensor T1 is used. If input tensor T1 is also not"
            "specified, then type defaults to 'float'.";
  return LightOpSchema(
      "EyeLike", kOnnxDomain, since_version, MakeEyeLikeDoc(since_version),
      {
          {"input", "2D input tensor to copy shape, and optionally, type information from.", "T1"},
      },
      {
          {"output", "Output tensor, same shape as input tensor T1.", "T2"},
      },
      {
          {"T1", EyeLikeTypes(since_version),
           "Constrain input types. Strings and complex are not supported."},
          {"T2", EyeLikeTypes(since_version),
           "Constrain output types. Strings and complex are not supported."},
      },
      {
          {"k",
           "(Optional) Index of the diagonal to be populated with ones. Default is 0."
           " If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, "
           "k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(0)},
          {"dtype", dtype_doc, AttributeType::INT, /*required=*/false, std::monostate{}},
      });
}

// Output type constraint sets for the four ONNX Random* operators. v22
// promoted the constraint to ``all_float_types_ir4`` (adds ``bfloat16``);
// v1 used the legacy ``float16, float, double`` triple.
std::vector<TensorType> RandomFloatTypes(int since_version) {
  switch (since_version) {
  case 22:
    return {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  case 1:
    return {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble};
  default:
    throw SchemaError("Unsupported Random* since_version: " + std::to_string(since_version));
  }
}

// Input type constraint for the *Like variants. v22 promoted T1 to
// ``all_tensor_types_ir4``; v1 used ``all_tensor_types``.
std::vector<TensorType> RandomLikeT1Types(int since_version) {
  switch (since_version) {
  case 22:
    return {
        TensorType::kUint8,      TensorType::kUint16, TensorType::kUint32,
        TensorType::kUint64,     TensorType::kInt8,   TensorType::kInt16,
        TensorType::kInt32,      TensorType::kInt64,  TensorType::kBfloat16,
        TensorType::kFloat16,    TensorType::kFloat,  TensorType::kDouble,
        TensorType::kString,     TensorType::kBool,   TensorType::kComplex64,
        TensorType::kComplex128,
    };
  case 1:
    return {
        TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,     TensorType::kUint64,
        TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,      TensorType::kInt64,
        TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,     TensorType::kString,
        TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
    };
  default:
    throw SchemaError("Unsupported Random*Like since_version: " + std::to_string(since_version));
  }
}

LightOpSchema MakeRandomNormalSchema(int since_version) {
  return LightOpSchema(
      "RandomNormal", kOnnxDomain, since_version, MakeRandomNormalDoc(), {},
      {
          {"output", "Output tensor of random values drawn from normal distribution", "T"},
      },
      {
          {"T", RandomFloatTypes(since_version), "Constrain output types to float tensors."},
      },
      {
          {"mean", "The mean of the normal distribution.", AttributeType::FLOAT,
           /*required=*/false, 0.0},
          {"scale", "The standard deviation of the normal distribution.", AttributeType::FLOAT,
           /*required=*/false, 1.0},
          {"seed",
           "(Optional) Seed to the random generator, if not specified we will auto generate one.",
           AttributeType::FLOAT, /*required=*/false, std::monostate{}},
          {"dtype",
           "The data type for the elements of the output tensor. Default is TensorProto::FLOAT.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
          {"shape", "The shape of the output tensor.", AttributeType::INTS, /*required=*/true,
           std::monostate{}},
      });
}

LightOpSchema MakeRandomUniformSchema(int since_version) {
  return LightOpSchema(
      "RandomUniform", kOnnxDomain, since_version, MakeRandomUniformDoc(), {},
      {
          {"output", "Output tensor of random values drawn from uniform distribution", "T"},
      },
      {
          {"T", RandomFloatTypes(since_version), "Constrain output types to float tensors."},
      },
      {
          {"low", "Lower boundary of the output values.", AttributeType::FLOAT,
           /*required=*/false, 0.0},
          {"high", "Upper boundary of the output values.", AttributeType::FLOAT,
           /*required=*/false, 1.0},
          {"seed",
           "(Optional) Seed to the random generator, if not specified we will auto generate one.",
           AttributeType::FLOAT, /*required=*/false, std::monostate{}},
          {"dtype",
           "The data type for the elements of the output tensor. If not specified, default is "
           "TensorProto::FLOAT.",
           AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
          {"shape", "The shape of the output tensor.", AttributeType::INTS, /*required=*/true,
           std::monostate{}},
      });
}

LightOpSchema MakeRandomNormalLikeSchema(int since_version) {
  return LightOpSchema(
      "RandomNormalLike", kOnnxDomain, since_version, MakeRandomNormalLikeDoc(),
      {
          {"input", "Input tensor to copy shape and optionally type information from.", "T1"},
      },
      {
          {"output", "Output tensor of random values drawn from normal distribution", "T2"},
      },
      {
          {"T1", RandomLikeT1Types(since_version),
           "Constrain to any tensor type. If the dtype attribute is not provided this must be a "
           "valid output type."},
          {"T2", RandomFloatTypes(since_version), "Constrain output types to float tensors."},
      },
      {
          {"mean", "The mean of the normal distribution.", AttributeType::FLOAT,
           /*required=*/false, 0.0},
          {"scale", "The standard deviation of the normal distribution.", AttributeType::FLOAT,
           /*required=*/false, 1.0},
          {"seed",
           "(Optional) Seed to the random generator, if not specified we will auto generate one.",
           AttributeType::FLOAT, /*required=*/false, std::monostate{}},
          {"dtype",
           "(Optional) The data type for the elements of the output tensor, if not specified, "
           "we will use the data type of the input tensor.",
           AttributeType::INT, /*required=*/false, std::monostate{}},
      });
}

LightOpSchema MakeRandomUniformLikeSchema(int since_version) {
  return LightOpSchema(
      "RandomUniformLike", kOnnxDomain, since_version, MakeRandomUniformLikeDoc(),
      {
          {"input", "Input tensor to copy shape and optionally type information from.", "T1"},
      },
      {
          {"output", "Output tensor of random values drawn from uniform distribution", "T2"},
      },
      {
          {"T1", RandomLikeT1Types(since_version),
           "Constrain to any tensor type. If the dtype attribute is not provided this must be a "
           "valid output type."},
          {"T2", RandomFloatTypes(since_version), "Constrain output types to float tensors."},
      },
      {
          {"low", "Lower boundary of the output values.", AttributeType::FLOAT,
           /*required=*/false, 0.0},
          {"high", "Upper boundary of the output values.", AttributeType::FLOAT,
           /*required=*/false, 1.0},
          {"seed",
           "(Optional) Seed to the random generator, if not specified we will auto generate one.",
           AttributeType::FLOAT, /*required=*/false, std::monostate{}},
          {"dtype",
           "(Optional) The data type for the elements of the output tensor, if not specified, "
           "we will use the data type of the input tensor.",
           AttributeType::INT, /*required=*/false, std::monostate{}},
      });
}

LightOpSchema MakeRangeSchema(int since_version) {
  return LightOpSchema(
      "Range", kOnnxDomain, since_version, MakeRangeDoc(),
      {
          {"start", "Scalar. First entry for the range of output values.", "T"},
          {"limit", "Scalar. Exclusive upper limit for the range of output values.", "T"},
          {"delta", "Scalar. Value to step by.", "T"},
      },
      {
          {"output",
           "A 1-D tensor with same type as the inputs containing generated range of values.", "T"},
      },
      {
          {"T",
           {TensorType::kFloat, TensorType::kDouble, TensorType::kInt16, TensorType::kInt32,
            TensorType::kInt64},
           "Constrain input types to common numeric type tensors."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpGeneratorSchemasWithHistory(const std::string &op_type,
                                                                   bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"Bernoulli",
       [] {
         return std::vector<LightOpSchema>{
             MakeBernoulliSchema(22),
             MakeBernoulliSchema(15),
         };
       }},
      {"Constant",
       [] {
         return std::vector<LightOpSchema>{
             MakeConstantSchema(25), MakeConstantSchema(24), MakeConstantSchema(23),
             MakeConstantSchema(21), MakeConstantSchema(19), MakeConstantSchema(13),
             MakeConstantSchema(12), MakeConstantSchema(11), MakeConstantSchema(9),
             MakeConstantSchema(1),
         };
       }},
      {"ConstantOfShape",
       [] {
         return std::vector<LightOpSchema>{
             MakeConstantOfShapeSchema(25), MakeConstantOfShapeSchema(24),
             MakeConstantOfShapeSchema(23), MakeConstantOfShapeSchema(21),
             MakeConstantOfShapeSchema(20), MakeConstantOfShapeSchema(9),
         };
       }},
      {"EyeLike",
       [] {
         return std::vector<LightOpSchema>{
             MakeEyeLikeSchema(22),
             MakeEyeLikeSchema(9),
         };
       }},
      {"RandomNormal",
       [] {
         return std::vector<LightOpSchema>{
             MakeRandomNormalSchema(22),
             MakeRandomNormalSchema(1),
         };
       }},
      {"RandomNormalLike",
       [] {
         return std::vector<LightOpSchema>{
             MakeRandomNormalLikeSchema(22),
             MakeRandomNormalLikeSchema(1),
         };
       }},
      {"RandomUniform",
       [] {
         return std::vector<LightOpSchema>{
             MakeRandomUniformSchema(22),
             MakeRandomUniformSchema(1),
         };
       }},
      {"RandomUniformLike",
       [] {
         return std::vector<LightOpSchema>{
             MakeRandomUniformLikeSchema(22),
             MakeRandomUniformLikeSchema(1),
         };
       }},
      {"Range",
       [] {
         return std::vector<LightOpSchema>{
             MakeRangeSchema(11),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace generator
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
