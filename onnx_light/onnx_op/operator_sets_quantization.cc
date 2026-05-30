// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_quantization.h"
#include "onnx_op/operator_sets_quantization_doc.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace quantization {

namespace {

LightOpSchema MakeQuantizeLinearV25Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 25, MakeQuantizeLinearDoc(25),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get `y`. For per-tensor/layer quantization the "
           "scale is a scalar, for "
           "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the "
           "same shape as the "
           "input, except for one dimension in which blocking is performed.",
           "T2"},
          {"y_zero_point",
           "Zero point for doing quantization to get `y`. Shape must match `y_scale`. "
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T3"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input `x`.", "T3"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "The type of the input 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32,
            TensorType::kFloat8e8m0},
           "The type of the input 'y_scale'."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
            TensorType::kFloat8e5m2fnuz, TensorType::kUint4, TensorType::kInt4,
            TensorType::kFloat4e2m1, TensorType::kUint2, TensorType::kInt2},
           "The type of the input `y_zero_point` and the output `y`."},
      });
}

LightOpSchema MakeQuantizeLinearV24Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 24, MakeQuantizeLinearDoc(24),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get `y`. For per-tensor/layer quantization the "
           "scale is a scalar, for "
           "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the "
           "same shape as the "
           "input, except for one dimension in which blocking is performed.",
           "T2"},
          {"y_zero_point",
           "Zero point for doing quantization to get `y`. Shape must match `y_scale`. "
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T3"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input `x`.", "T3"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "The type of the input 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32,
            TensorType::kFloat8e8m0},
           "The type of the input 'y_scale'."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
            TensorType::kFloat8e5m2fnuz, TensorType::kUint4, TensorType::kInt4,
            TensorType::kFloat4e2m1},
           "The type of the input `y_zero_point` and the output `y`."},
      });
}

LightOpSchema MakeQuantizeLinearV23Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 23, MakeQuantizeLinearDoc(23),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get `y`. For per-tensor/layer quantization the "
           "scale is a scalar, for "
           "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the "
           "same shape as the "
           "input, except for one dimension in which blocking is performed.",
           "T2"},
          {"y_zero_point",
           "Zero point for doing quantization to get `y`. Shape must match `y_scale`."
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T3"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input `x`.", "T3"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "The type of the input 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "The type of the input 'y_scale'."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
            TensorType::kFloat8e5m2fnuz, TensorType::kUint4, TensorType::kInt4,
            TensorType::kFloat4e2m1},
           "The type of the input `y_zero_point` and the output `y`."},
      });
}

LightOpSchema MakeQuantizeLinearV21Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 21, MakeQuantizeLinearDoc(21),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get `y`. For per-tensor/layer quantization the "
           "scale is a scalar, for "
           "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the "
           "same shape as the "
           "input, except for one dimension in which blocking is performed.",
           "T1"},
          {"y_zero_point",
           "Zero point for doing quantization to get `y`. Shape must match `y_scale`."
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T2"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input `x`.", "T2"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "The type of the input 'x'."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2,
            TensorType::kFloat8e5m2fnuz, TensorType::kUint4, TensorType::kInt4},
           "The type of the input `y_zero_point` and the output `y`."},
      });
}

LightOpSchema MakeQuantizeLinearV19Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 19, MakeQuantizeLinearDoc(19),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get 'y'. It can be a scalar, which means "
           "per-tensor/layer quantization, "
           "or a 1-D Tensor for per-axis quantization.",
           "T1"},
          {"y_zero_point",
           "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T2"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16, TensorType::kInt32},
           "Constrain 'x' to float, float16, bfloat16 or int32 tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kFloat8e4m3fn,
            TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz},
           "Constrain 'y_zero_point' and 'y' to 8-bit integer/float tensor."},
      });
}

LightOpSchema MakeQuantizeLinearV13Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 13, MakeQuantizeLinearDoc(13),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get 'y'. It can be a scalar, which means "
           "per-tensor/layer quantization, "
           "or a 1-D Tensor for per-axis quantization.",
           "tensor(float)"},
          {"y_zero_point",
           "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
           "Default is uint8 with zero point of 0 if it's not specified.",
           "T2"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kInt32},
           "Constrain 'x' to float or int32 tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor."},
      });
}

LightOpSchema MakeQuantizeLinearV10Schema() {
  return LightOpSchema(
      "QuantizeLinear", kOnnxDomain, 10, MakeQuantizeLinearDoc(10),
      {
          {"x", "N-D full precision Input tensor to be quantized.", "T1"},
          {"y_scale",
           "Scale for doing quantization to get 'y'. It's a scalar, which means a "
           "per-tensor/layer quantization.",
           "tensor(float)"},
          {"y_zero_point",
           "Zero point for doing quantization to get 'y'. It's a scalar, which means a "
           "per-tensor/layer quantization. "
           "Default value is uint8 typed 0 if it's not specified.",
           "T2"},
      },
      {
          {"y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2"},
      },
      {
          {"T1",
           {TensorType::kFloat, TensorType::kInt32},
           "Constrain 'x' to float or int32 tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor."},
      });
}

LightOpSchema MakeDequantizeLinearV25Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 25, MakeDequantizeLinearDoc(25),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T1"},
          {"x_scale",
           "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
           "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has "
           "the same shape as "
           "the input, except for one dimension in which blocking is performed.",
           "T2"},
          {"x_zero_point",
           "Zero point for input `x`. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T1"},
      },
      {
          {"y",
           "N-D full precision output tensor. It has the same shape as input `x`. The data type "
           "is specified by the `output_dtype` attribute or, in its absence, the type of "
           "`x_scale`.",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kInt32, TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz,
            TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
            TensorType::kInt4, TensorType::kFloat4e2m1, TensorType::kUint2, TensorType::kInt2},
           "The type of the inputs 'x_zero_point' and 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16,
            TensorType::kFloat8e8m0},
           "The type of the input 'x_scale'."},
          {"T3",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "The type of the output 'y'."},
      });
}

LightOpSchema MakeDequantizeLinearV24Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 24, MakeDequantizeLinearDoc(24),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T1"},
          {"x_scale",
           "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
           "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has "
           "the same shape as "
           "the input, except for one dimension in which blocking is performed.",
           "T2"},
          {"x_zero_point",
           "Zero point for input `x`. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T1"},
      },
      {
          {"y",
           "N-D full precision output tensor. It has the same shape as input `x`. The data type "
           "is specified by the `output_dtype` attribute or, in its absence, the type of "
           "`x_scale`.",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kInt32, TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz,
            TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
            TensorType::kInt4, TensorType::kFloat4e2m1},
           "The type of the inputs 'x_zero_point' and 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16,
            TensorType::kFloat8e8m0},
           "The type of the input 'x_scale'."},
          {"T3",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "The type of the output 'y'."},
      });
}

LightOpSchema MakeDequantizeLinearV23Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 23, MakeDequantizeLinearDoc(23),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T1"},
          {"x_scale",
           "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
           "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has "
           "the same shape as "
           "the input, except for one dimension in which blocking is performed.",
           "T2"},
          {"x_zero_point",
           "Zero point for input `x`. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T1"},
      },
      {
          {"y",
           "N-D full precision output tensor. It has the same shape as input `x`. The data type "
           "is specified by the `output_dtype` attribute or, in its absence, the type of "
           "`x_scale`.",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kInt32, TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz,
            TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
            TensorType::kInt4, TensorType::kFloat4e2m1},
           "The type of the inputs 'x_zero_point' and 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "The type of the input 'x_scale'."},
          {"T3",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "The type of the output 'y'."},
      });
}

LightOpSchema MakeDequantizeLinearV21Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 21, MakeDequantizeLinearDoc(21),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T1"},
          {"x_scale",
           "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
           "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has "
           "the same shape as "
           "the input, except for one dimension in which blocking is performed.",
           "T2"},
          {"x_zero_point",
           "Zero point for input `x`. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T1"},
      },
      {
          {"y", "N-D full precision output tensor. It has same shape as input `x`.", "T2"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt16, TensorType::kUint16,
            TensorType::kInt32, TensorType::kFloat8e4m3fn, TensorType::kFloat8e4m3fnuz,
            TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
            TensorType::kInt4},
           "The type of the inputs 'x_zero_point' and 'x'."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "'x_scale' determines the output type."},
      });
}

LightOpSchema MakeDequantizeLinearV19Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 19, MakeDequantizeLinearDoc(19),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T1"},
          {"x_scale",
           "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer "
           "dequantization, "
           "or a 1-D tensor for per-axis dequantization.",
           "T2"},
          {"x_zero_point",
           "Zero point for input 'x'. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T1"},
      },
      {
          {"y", "N-D full precision output tensor. It has same shape as input 'x'.", "T2"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt32, TensorType::kFloat8e4m3fn,
            TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz},
           "Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit integer "
           "tensor."},
          {"T2",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "'x_scale' determines the output type."},
      });
}

LightOpSchema MakeDequantizeLinearV13Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 13, MakeDequantizeLinearDoc(13),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T"},
          {"x_scale",
           "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer "
           "dequantization, "
           "or a 1-D tensor for per-axis dequantization.",
           "tensor(float)"},
          {"x_zero_point",
           "Zero point for input 'x'. Shape must match x_scale. "
           "It's optional. Zero point is 0 when it's not specified.",
           "T"},
      },
      {
          {"y", "N-D full precision output tensor. It has same shape as input 'x'.",
           "tensor(float)"},
      },
      {
          {"T",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt32},
           "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor."},
      });
}

LightOpSchema MakeDequantizeLinearV10Schema() {
  return LightOpSchema(
      "DequantizeLinear", kOnnxDomain, 10, MakeDequantizeLinearDoc(10),
      {
          {"x", "N-D quantized input tensor to be de-quantized.", "T"},
          {"x_scale",
           "Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
           "tensor(float)"},
          {"x_zero_point",
           "Zero point for input 'x'. It's a scalar, which means a per-tensor/layer "
           "quantization. "
           "It's optional. 0 is the default value when it's not specified.",
           "T"},
      },
      {
          {"y", "N-D full precision output tensor. It has same shape as input 'x'.",
           "tensor(float)"},
      },
      {
          {"T",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kInt32},
           "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpQuantizationSchemasWithHistory(const std::string &op_type,
                                                                      bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"QuantizeLinear",
       [] {
         return std::vector<LightOpSchema>{
             MakeQuantizeLinearV25Schema(), MakeQuantizeLinearV24Schema(),
             MakeQuantizeLinearV23Schema(), MakeQuantizeLinearV21Schema(),
             MakeQuantizeLinearV19Schema(), MakeQuantizeLinearV13Schema(),
             MakeQuantizeLinearV10Schema(),
         };
       }},
      {"DequantizeLinear",
       [] {
         return std::vector<LightOpSchema>{
             MakeDequantizeLinearV25Schema(), MakeDequantizeLinearV24Schema(),
             MakeDequantizeLinearV23Schema(), MakeDequantizeLinearV21Schema(),
             MakeDequantizeLinearV19Schema(), MakeDequantizeLinearV13Schema(),
             MakeDequantizeLinearV10Schema(),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace quantization
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
