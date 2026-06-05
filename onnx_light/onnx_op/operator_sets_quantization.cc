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

// --- DynamicQuantizeLinear ---------------------------------------------------

LightOpSchema MakeDynamicQuantizeLinearV11Schema() {
  return LightOpSchema(
      "DynamicQuantizeLinear", kOnnxDomain, 11, MakeDynamicQuantizeLinearDoc(11),
      {
          {"x", "Input tensor", "T1"},
      },
      {
          {"y", "Quantized output tensor", "T2"},
          {"y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
           "tensor(float)"},
          {"y_zero_point",
           "Output zero point. It's a scalar, which means a per-tensor/layer quantization.", "T2"},
      },
      {
          {"T1", {TensorType::kFloat}, "Constrain 'x' to float tensor."},
          {"T2",
           {TensorType::kUint8},
           "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor."},
      });
}

// --- QLinearConv -------------------------------------------------------------

LightOpSchema MakeQLinearConvV10Schema() {
  return LightOpSchema(
      "QLinearConv", kOnnxDomain, 10, MakeQLinearConvDoc(10),
      {
          {"x",
           "Input data tensor from previous layer; "
           "has size (N x C x H x W), where N is the batch size, "
           "C is the number of channels, and H and W are the "
           "height and width. Note that this is for the 2D image. "
           "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
           "Optionally, if dimension denotation is "
           "in effect, the operation expects input data tensor "
           "to arrive with the dimension denotation of [DATA_BATCH, "
           "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
           "T1"},
          {"x_scale",
           "Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer "
           "quantization.",
           "tensor(float)"},
          {"x_zero_point",
           "Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer "
           "quantization.",
           "T1"},
          {"w",
           "The weight tensor that will be used in the "
           "convolutions; has size (M x C/group x kH x kW), where C "
           "is the number of channels, and kH and kW are the "
           "height and width of the kernel, and M is the number "
           "of feature maps. For more than 2 dimensions, the "
           "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
           "where (k1 x k2 x ... kn) is the dimension of the kernel. "
           "Optionally, if dimension denotation is in effect, "
           "the operation expects the weight tensor to arrive "
           "with the dimension denotation of [FILTER_OUT_CHANNEL, "
           "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
           "X.shape[1] == (W.shape[1] * group) == C "
           "(assuming zero based indices for the shape array). "
           "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
           "T2"},
          {"w_scale",
           "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a "
           "per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its "
           "number of elements should be equal to the number of output channels (M).",
           "tensor(float)"},
          {"w_zero_point",
           "Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor, which means "
           "a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its "
           "number of elements should be equal to the number of output channels (M).",
           "T2"},
          {"y_scale",
           "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer "
           "quantization.",
           "tensor(float)"},
          {"y_zero_point",
           "Zero point tensor for output 'y'. It's a scalar, which means a per-tensor/layer "
           "quantization.",
           "T3"},
          {"B",
           "Optional 1D bias to be added to the convolution, has size of M. "
           "Bias must be quantized using scale = x_scale * w_scale and zero_point = 0",
           "T4"},
      },
      {
          {"y",
           "Output data tensor that contains the result of the "
           "convolution. The output dimensions are functions "
           "of the kernel size, stride size, and pad lengths.",
           "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain input type to 8-bit integer tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain filter type to 8-bit integer tensor."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain output type to 8-bit integer tensor."},
          {"T4", {TensorType::kInt32}, "Constrain bias type to 32-bit integer tensor."},
      });
}

// --- QLinearMatMul -----------------------------------------------------------

LightOpSchema MakeQLinearMatMulV21Schema() {
  return LightOpSchema(
      "QLinearMatMul", kOnnxDomain, 21, MakeQLinearMatMulDoc(21),
      {
          {"a", "N-dimensional quantized matrix a", "T1"},
          {"a_scale", "scale of quantized input a", "TS"},
          {"a_zero_point", "zero point of quantized input a", "T1"},
          {"b", "N-dimensional quantized matrix b", "T2"},
          {"b_scale", "scale of quantized input b", "TS"},
          {"b_zero_point", "zero point of quantized input b", "T2"},
          {"y_scale", "scale of quantized output y", "TS"},
          {"y_zero_point", "zero point of quantized output y", "T3"},
      },
      {
          {"y", "Quantized matrix multiply results from a * b", "T3"},
      },
      {
          {"TS",
           {TensorType::kFloat, TensorType::kFloat16, TensorType::kBfloat16},
           "Constrain scales."},
          {"T1",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kFloat8e4m3fn,
            TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz},
           "The type of input a and its zeropoint."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kFloat8e4m3fn,
            TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz},
           "The type of input b and its zeropoint."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8, TensorType::kFloat8e4m3fn,
            TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz},
           "The type of the output and its zeropoint."},
      });
}

LightOpSchema MakeQLinearMatMulV10Schema() {
  return LightOpSchema(
      "QLinearMatMul", kOnnxDomain, 10, MakeQLinearMatMulDoc(10),
      {
          {"a", "N-dimensional quantized matrix a", "T1"},
          {"a_scale", "scale of quantized input a", "tensor(float)"},
          {"a_zero_point", "zero point of quantized input a", "T1"},
          {"b", "N-dimensional quantized matrix b", "T2"},
          {"b_scale", "scale of quantized input b", "tensor(float)"},
          {"b_zero_point", "zero point of quantized input b", "T2"},
          {"y_scale", "scale of quantized output y", "tensor(float)"},
          {"y_zero_point", "zero point of quantized output y", "T3"},
      },
      {
          {"y", "Quantized matrix multiply results from a * b", "T3"},
      },
      {
          {"T1",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain input a and its zero point data type to 8-bit integer tensor."},
          {"T2",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain input b and its zero point data type to 8-bit integer tensor."},
          {"T3",
           {TensorType::kInt8, TensorType::kUint8},
           "Constrain output y and its zero point data type to 8-bit integer tensor."},
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
      {"QLinearConv",
       [] {
         return std::vector<LightOpSchema>{
             MakeQLinearConvV10Schema(),
         };
       }},
      {"DynamicQuantizeLinear",
       [] {
         return std::vector<LightOpSchema>{
             MakeDynamicQuantizeLinearV11Schema(),
         };
       }},
      {"QLinearMatMul",
       [] {
         return std::vector<LightOpSchema>{
             MakeQLinearMatMulV21Schema(),
             MakeQLinearMatMulV10Schema(),
         };
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace quantization
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
