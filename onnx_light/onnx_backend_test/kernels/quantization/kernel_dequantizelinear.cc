// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

inline void RequireScalar(const Tensor &t, const char *name) {
  // A scalar is either a 0-D tensor (shape == {}) or a 1-D tensor with a
  // single element. Both are accepted for the per-tensor case to mirror
  // QuantizeLinear's behaviour.
  const int64_t n = t.element_count();
  EXT_ENFORCE_INVALID(n == 1, std::string("kernel::DequantizeLinear: ") + name +
                                  " must be a scalar (per-tensor dequantization).");
}

template <typename XT>
void DequantizeLoop(const Tensor &x, float x_scale, XT x_zero_point, Tensor &output) {
  const XT *px = reinterpret_cast<const XT *>(x.data.data());
  float *py = output.AsFloat();
  const int64_t n = x.element_count();
  const float zp = static_cast<float>(x_zero_point);
  for (int64_t i = 0; i < n; ++i) {
    py[i] = (static_cast<float>(px[i]) - zp) * x_scale;
  }
}

} // namespace

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale) const {
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, x_scale, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale, Tensor &output) const {
  EXT_ENFORCE_INVALID(x_scale.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::DequantizeLinear: x_scale must be FLOAT.");
  RequireScalar(x_scale, "x_scale");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::DequantizeLinear: output (no x_zero_point) must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = x_scale.AsFloat()[0];
  switch (x.data_type) {
  case static_cast<int32_t>(TensorProto::DataType::UINT8):
    DequantizeLoop<uint8_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::INT8):
    DequantizeLoop<int8_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::UINT16):
    DequantizeLoop<uint16_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::INT16):
    DequantizeLoop<int16_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::INT32):
    DequantizeLoop<int32_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  default:
    throw std::invalid_argument("kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16 and "
                                "INT32 inputs are supported.");
  }
}

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                    const Tensor &x_zero_point) const {
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, x_scale, x_zero_point, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                  const Tensor &x_zero_point, Tensor &output) const {
  EXT_ENFORCE_INVALID(x_scale.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::DequantizeLinear: x_scale must be FLOAT.");
  RequireScalar(x_scale, "x_scale");
  RequireScalar(x_zero_point, "x_zero_point");
  EXT_ENFORCE_INVALID(x.data_type == x_zero_point.data_type,
                      "kernel::DequantizeLinear: x_zero_point data_type must match x.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::DequantizeLinear: output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = x_scale.AsFloat()[0];
  switch (x.data_type) {
  case static_cast<int32_t>(TensorProto::DataType::UINT8):
    DequantizeLoop<uint8_t>(x, scale, static_cast<uint8_t>(x_zero_point.data[0]), output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::INT8):
    DequantizeLoop<int8_t>(x, scale, static_cast<int8_t>(x_zero_point.data[0]), output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::UINT16): {
    uint16_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(uint16_t));
    DequantizeLoop<uint16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(TensorProto::DataType::INT16): {
    int16_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(int16_t));
    DequantizeLoop<int16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(TensorProto::DataType::INT32): {
    int32_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(int32_t));
    DequantizeLoop<int32_t>(x, scale, zp, output);
    break;
  }
  default:
    throw std::invalid_argument("kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16 and "
                                "INT32 inputs are supported.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
