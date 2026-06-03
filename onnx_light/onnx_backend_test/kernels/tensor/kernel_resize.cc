// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads the 1-D FLOAT ``scales`` input tensor and validates it against the
// rank of the input. Mirrors the contract of the upstream Resize schema:
// ``scales`` is a 1-D FLOAT tensor with one entry per input axis.
std::vector<float> ReadResizeScales(const Tensor &scales, std::size_t rank) {
  EXT_ENFORCE_INVALID(scales.data_type == DataType::FLOAT,
                      "kernel::Resize: 'scales' input must be FLOAT.");
  EXT_ENFORCE_INVALID(scales.shape.size() == 1,
                      "kernel::Resize: 'scales' input must be a 1-D tensor.");
  const int64_t n = scales.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == rank,
                      "kernel::Resize: 'scales' length must equal the rank of 'X'.");
  std::vector<float> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), scales.data.data(), static_cast<std::size_t>(n) * sizeof(float));
  }
  for (float s : out) {
    EXT_ENFORCE_INVALID(s > 0.0f, "kernel::Resize: 'scales' values must be > 0.");
  }
  return out;
}

// Reads the 1-D INT64 ``sizes`` input tensor and validates it against the
// rank of the input.
std::vector<int64_t> ReadResizeSizes(const Tensor &sizes, std::size_t rank) {
  EXT_ENFORCE_INVALID(sizes.data_type == DataType::INT64,
                      "kernel::Resize: 'sizes' input must be INT64.");
  EXT_ENFORCE_INVALID(sizes.shape.size() == 1,
                      "kernel::Resize: 'sizes' input must be a 1-D tensor.");
  const int64_t n = sizes.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == rank,
                      "kernel::Resize: 'sizes' length must equal the rank of 'X'.");
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), sizes.data.data(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  for (int64_t s : out) {
    EXT_ENFORCE_INVALID(s > 0, "kernel::Resize: 'sizes' values must be > 0.");
  }
  return out;
}

std::vector<int64_t> ComputeOutputShapeFromScales(const std::vector<int64_t> &in_shape,
                                                  const std::vector<float> &scales) {
  std::vector<int64_t> out_shape(in_shape.size());
  for (std::size_t k = 0; k < in_shape.size(); ++k) {
    const double scaled = static_cast<double>(in_shape[k]) * static_cast<double>(scales[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }
  return out_shape;
}

// Nearest-neighbor resize for any rank, byte-element-wise copy. Uses the
// ``asymmetric`` coordinate transformation (``in_coord = floor(out_coord /
// scale)``, clamped to ``[0, in_dim - 1]``).
void ResizeNearest(const Tensor &input, const std::vector<float> &scales,
                   const std::vector<int64_t> &out_shape, Tensor &output) {
  const std::size_t elem_size = ElementSize(input.data_type);
  const std::size_t rank = out_shape.size();

  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }

  std::vector<int64_t> in_strides(rank, 0);
  std::vector<int64_t> out_strides(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
      out_strides[k - 1] = out_strides[k] * out_shape[k];
    }
  }

  const uint8_t *const in_ptr = input.data.data();
  uint8_t *const out_ptr = output.data.data();

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

bool IsNearestMode(const std::string &mode) { return mode == "nearest"; }

} // namespace

Tensor Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs) const {
  const std::vector<float> scales_vec = ReadResizeScales(scales, X.shape.size());
  const std::vector<int64_t> out_shape = ComputeOutputShapeFromScales(X.shape, scales_vec);
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  Tensor output("", X.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(X.data_type, total_elements)));
  (*this)(X, scales, attrs, output);
  return output;
}

void Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                        Tensor &output) const {
  const std::vector<float> scales_vec = ReadResizeScales(scales, X.shape.size());
  const std::vector<int64_t> out_shape = ComputeOutputShapeFromScales(X.shape, scales_vec);

  EXT_ENFORCE_INVALID(output.data_type == X.data_type,
                      "kernel::Resize: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Resize: preallocated output shape mismatch.");

  if (!IsNearestMode(attrs.mode)) {
    throw std::invalid_argument("kernel::Resize: only 'nearest' interpolation mode is supported "
                                "in this reference implementation; got '" +
                                attrs.mode + "'.");
  }
  EXT_ENFORCE_INVALID(attrs.coordinate_transformation_mode == "asymmetric",
                      "kernel::Resize: only 'asymmetric' coordinate_transformation_mode is "
                      "supported in this reference implementation.");
  ResizeNearest(X, scales_vec, out_shape, output);
}

Tensor Resize::ResizeSizes(const Tensor &X, const Tensor &sizes, const Attributes &attrs) const {
  const std::vector<int64_t> sizes_vec = ReadResizeSizes(sizes, X.shape.size());
  // Derive per-axis scales from sizes so we can reuse the nearest path.
  std::vector<float> scales_vec(sizes_vec.size());
  for (std::size_t k = 0; k < sizes_vec.size(); ++k) {
    EXT_ENFORCE_INVALID(X.shape[k] > 0,
                        "kernel::Resize: input dim must be > 0 when using 'sizes'.");
    scales_vec[k] = static_cast<float>(sizes_vec[k]) / static_cast<float>(X.shape[k]);
  }
  int64_t total_elements = 1;
  for (int64_t d : sizes_vec) {
    total_elements *= d;
  }
  Tensor output("", X.data_type, sizes_vec,
                std::vector<uint8_t>(PackedByteSize(X.data_type, total_elements)));

  if (!IsNearestMode(attrs.mode)) {
    throw std::invalid_argument("kernel::Resize: only 'nearest' interpolation mode is supported "
                                "in this reference implementation; got '" +
                                attrs.mode + "'.");
  }
  EXT_ENFORCE_INVALID(attrs.coordinate_transformation_mode == "asymmetric",
                      "kernel::Resize: only 'asymmetric' coordinate_transformation_mode is "
                      "supported in this reference implementation.");
  ResizeNearest(X, scales_vec, sizes_vec, output);
  return output;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
