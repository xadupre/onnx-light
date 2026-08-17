// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Reads the 1-D FLOAT ``scales`` input tensor into an allocator-backed scratch
// buffer and validates it against the rank of the input. Mirrors the upstream
// Upsample v7/9/10 contract: the number of elements of ``scales`` must equal the
// rank of ``X`` and every entry must be >= 1. ``allocator`` (when non-null)
// supplies the scratch storage; otherwise the returned tensor falls back to
// inline storage.
Tensor ReadUpsampleScales(const Tensor &scales, std::size_t rank, RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(scales.data_type == DataType::FLOAT,
                      "kernel::Upsample: 'scales' input must be FLOAT.");
  EXT_ENFORCE_INVALID(scales.shape.size() == 1,
                      "kernel::Upsample: 'scales' input must be a 1-D tensor.");
  const int64_t n = scales.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == rank,
                      "kernel::Upsample: 'scales' length must equal the rank of 'X'.");
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(float);
  Tensor out = MakeOutputTensor(DataType::FLOAT, {n}, n_bytes, allocator);
  if (n > 0) {
    std::memcpy(out.mutable_bytes(), scales.bytes(), n_bytes);
  }
  const float *values = out.As<float>();
  for (int64_t i = 0; i < n; ++i) {
    EXT_ENFORCE_INVALID(values[i] >= 1.0f, "kernel::Upsample: 'scales' values must be >= 1.");
  }
  return out;
}

onnx_kernels::Shape ComputeUpsampleOutputShape(const onnx_kernels::Shape &in_shape,
                                               const float *scales) {
  onnx_kernels::Shape out_shape;
  out_shape.assign(in_shape.size(), 0);
  for (std::size_t k = 0; k < in_shape.size(); ++k) {
    const double scaled = static_cast<double>(in_shape[k]) * static_cast<double>(scales[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }
  return out_shape;
}

// Nearest-neighbor upsample for any rank, byte-element-wise copy.
void UpsampleNearest(const Tensor &input, const float *scales, const onnx_kernels::Shape &out_shape,
                     Tensor &output) {
  const std::size_t elem_size = ElementSize(input.data_type);
  const std::size_t rank = out_shape.size();

  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }

  onnx_kernels::Shape in_strides;
  in_strides.assign(rank, 0);
  onnx_kernels::Shape out_strides;
  out_strides.assign(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
      out_strides[k - 1] = out_strides[k] * out_shape[k];
    }
  }

  const uint8_t *const in_ptr = input.bytes();
  uint8_t *const out_ptr = output.mutable_bytes();

  for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t out_coord = remaining / out_strides[k];
      remaining %= out_strides[k];
      int64_t in_coord = static_cast<int64_t>(
          std::floor(static_cast<double>(out_coord) / static_cast<double>(scales[k])));
      if (in_coord >= input.shape[k]) {
        in_coord = input.shape[k] - 1;
      }
      if (in_coord < 0) {
        in_coord = 0;
      }
      in_idx += in_coord * in_strides[k];
    }
    std::memcpy(out_ptr + static_cast<std::size_t>(out_idx) * elem_size,
                in_ptr + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

// Reads a single element of ``input`` at flat index ``idx`` as a double,
// for the floating-point element types supported by the "linear" mode.
double LoadFloat(const Tensor &input, int64_t idx) {
  const uint8_t *const base = input.bytes();
  switch (input.data_type) {
  case DataType::FLOAT: {
    float v;
    std::memcpy(&v, base + static_cast<std::size_t>(idx) * sizeof(float), sizeof(float));
    return static_cast<double>(v);
  }
  case DataType::DOUBLE: {
    double v;
    std::memcpy(&v, base + static_cast<std::size_t>(idx) * sizeof(double), sizeof(double));
    return v;
  }
  default:
    EXT_THROW_INVALID("unsupported data type ", input.data_type, ", ",
                      "kernel::Upsample: linear mode only supports FLOAT/DOUBLE input types.");
  }
}

void StoreFloat(Tensor &output, int64_t idx, double value) {
  uint8_t *const base = output.mutable_bytes();
  switch (output.data_type) {
  case DataType::FLOAT: {
    float v = static_cast<float>(value);
    std::memcpy(base + static_cast<std::size_t>(idx) * sizeof(float), &v, sizeof(float));
    return;
  }
  case DataType::DOUBLE: {
    std::memcpy(base + static_cast<std::size_t>(idx) * sizeof(double), &value, sizeof(double));
    return;
  }
  default:
    EXT_THROW_INVALID("unsupported data type ", output.data_type, ", ",
                      "kernel::Upsample: linear mode only supports FLOAT/DOUBLE output types.");
  }
}

// Bilinear upsample of a 4-D NCHW tensor using the "asymmetric" coordinate
// transformation matching upstream Upsample v7/9/10 (in_coord = out_coord /
// scale, clamped to ``[0, in_dim - 1]``).
void UpsampleLinear4D(const Tensor &input, const float *scales,
                      const onnx_kernels::Shape &out_shape, Tensor &output) {
  EXT_ENFORCE_INVALID(input.shape.size() == 4,
                      "kernel::Upsample: linear mode requires a 4-D NCHW input.");
  EXT_ENFORCE_INVALID(scales[0] == 1.0f && scales[1] == 1.0f,
                      "kernel::Upsample: linear mode requires scales[0] == 1 and scales[1] == 1.");

  const int64_t N = input.shape[0];
  const int64_t C = input.shape[1];
  const int64_t H = input.shape[2];
  const int64_t W = input.shape[3];
  const int64_t H_out = out_shape[2];
  const int64_t W_out = out_shape[3];

  const double scale_h = static_cast<double>(scales[2]);
  const double scale_w = static_cast<double>(scales[3]);

  const int64_t in_stride_n = C * H * W;
  const int64_t in_stride_c = H * W;
  const int64_t in_stride_h = W;
  const int64_t out_stride_n = C * H_out * W_out;
  const int64_t out_stride_c = H_out * W_out;
  const int64_t out_stride_h = W_out;

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t h_out = 0; h_out < H_out; ++h_out) {
        const double in_h = static_cast<double>(h_out) / scale_h;
        int64_t h0 = static_cast<int64_t>(std::floor(in_h));
        int64_t h1 = h0 + 1;
        if (h0 < 0) {
          h0 = 0;
        }
        if (h0 > H - 1) {
          h0 = H - 1;
        }
        if (h1 < 0) {
          h1 = 0;
        }
        if (h1 > H - 1) {
          h1 = H - 1;
        }
        const double dh = in_h - std::floor(in_h);
        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
          const double in_w = static_cast<double>(w_out) / scale_w;
          int64_t w0 = static_cast<int64_t>(std::floor(in_w));
          int64_t w1 = w0 + 1;
          if (w0 < 0) {
            w0 = 0;
          }
          if (w0 > W - 1) {
            w0 = W - 1;
          }
          if (w1 < 0) {
            w1 = 0;
          }
          if (w1 > W - 1) {
            w1 = W - 1;
          }
          const double dw = in_w - std::floor(in_w);

          const int64_t base_in = n * in_stride_n + c * in_stride_c;
          const double v00 = LoadFloat(input, base_in + h0 * in_stride_h + w0);
          const double v01 = LoadFloat(input, base_in + h0 * in_stride_h + w1);
          const double v10 = LoadFloat(input, base_in + h1 * in_stride_h + w0);
          const double v11 = LoadFloat(input, base_in + h1 * in_stride_h + w1);

          const double v = v00 * (1.0 - dh) * (1.0 - dw) + v01 * (1.0 - dh) * dw +
                           v10 * dh * (1.0 - dw) + v11 * dh * dw;
          const int64_t out_idx =
              n * out_stride_n + c * out_stride_c + h_out * out_stride_h + w_out;
          StoreFloat(output, out_idx, v);
        }
      }
    }
  }
}

bool IsNearestMode(const std::string &mode) { return mode == "nearest"; }

bool IsLinearMode(const std::string &mode) { return mode == "linear" || mode == "bilinear"; }

} // namespace

Tensor Upsample::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                            RuntimeContext *rt) const {
  const Tensor scales_in =
      ReadUpsampleScales(scales, X.shape.size(), rt ? rt->execution_allocator() : nullptr);
  const float *scales_vec = scales_in.As<float>();
  const onnx_kernels::Shape out_shape = ComputeUpsampleOutputShape(X.shape, scales_vec);
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  const size_t output_n_bytes = PackedByteSize(X.data_type, total_elements);
  Tensor output = (rt ? rt->MakeOutputTensor(0, X.data_type, out_shape, output_n_bytes)
                      : MakeOutputTensor(X.data_type, out_shape, output_n_bytes, nullptr));
  (*this)(X, scales, attrs, output);
  return output;
}

void Upsample::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                          Tensor &output) const {
  const Tensor scales_in = ReadUpsampleScales(scales, X.shape.size(), nullptr);
  const float *scales_vec = scales_in.As<float>();
  const onnx_kernels::Shape out_shape = ComputeUpsampleOutputShape(X.shape, scales_vec);

  EXT_ENFORCE_INVALID(output.data_type == X.data_type,
                      "kernel::Upsample: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Upsample: preallocated output shape mismatch.");

  if (IsNearestMode(attrs.mode)) {
    UpsampleNearest(X, scales_vec, out_shape, output);
    return;
  }
  if (IsLinearMode(attrs.mode)) {
    UpsampleLinear4D(X, scales_vec, out_shape, output);
    return;
  }
  EXT_THROW_INVALID("kernel::Upsample: unsupported mode '", attrs.mode,
                    "'. Supported modes: 'nearest', 'linear'/'bilinear'.");
}

void Upsample::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scales = GetInput(node, 1, rt.tensors());
  onnx_kernels::kernel::Upsample::Attributes attrs;
  attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
  onnx_kernels::kernel::Upsample k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scales, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
