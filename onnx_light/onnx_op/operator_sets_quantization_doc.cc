// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_quantization_doc.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace quantization {

namespace {

constexpr const char *kQuantizeLinearVer25Doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]
- uint2: [0, 3]
- int2: [-2, 1]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
`x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
`y_scale`, unless the `precision` attribute is specified.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
  `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

constexpr const char *kQuantizeLinearVer24Doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
`x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
`y_scale`, unless the `precision` attribute is specified.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
  `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

constexpr const char *kQuantizeLinearVer21Doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.
Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]
For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
  `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

constexpr const char *kQuantizeLinearVer19Doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.
)DOC";

constexpr const char *kQuantizeLinearVer13Doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

constexpr const char *kQuantizeLinearVer10Doc = R"DOC(
The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

constexpr const char *kDequantizeLinearVer24Doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
is the same as `x_scale`. The output type also determines the precision of the multiplication operation.
)DOC";

constexpr const char *kDequantizeLinearVer21Doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.
`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 types quantization, but the dequantization formula remains the same
for consistency, and `x_scale` still determines the output type.
)DOC";

constexpr const char *kDequantizeLinearVer19Doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
)DOC";

constexpr const char *kDequantizeLinearVer13Doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

constexpr const char *kDequantizeLinearVer10Doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

} // namespace

std::string MakeQuantizeLinearDoc(int since_version) {
  switch (since_version) {
  case 25:
    return kQuantizeLinearVer25Doc;
  case 24:
  case 23:
    return kQuantizeLinearVer24Doc;
  case 21:
    return kQuantizeLinearVer21Doc;
  case 19:
    return kQuantizeLinearVer19Doc;
  case 13:
    return kQuantizeLinearVer13Doc;
  case 10:
    return kQuantizeLinearVer10Doc;
  default:
    return "";
  }
}

std::string MakeDequantizeLinearDoc(int since_version) {
  switch (since_version) {
  case 25:
  case 24:
  case 23:
    return kDequantizeLinearVer24Doc;
  case 21:
    return kDequantizeLinearVer21Doc;
  case 19:
    return kDequantizeLinearVer19Doc;
  case 13:
    return kDequantizeLinearVer13Doc;
  case 10:
    return kDequantizeLinearVer10Doc;
  default:
    return "";
  }
}

} // namespace quantization
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
