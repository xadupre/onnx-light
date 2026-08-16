// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns the per-axis normalised coordinate values for an axis of length
// ``dim_size`` as a 1-D FLOAT Tensor. The convention matches
// ``torch.nn.functional.affine_grid`` and the upstream ONNX reference
// (``op_affine_grid.py``):
//
//   * align_corners == 1: linearly maps integer indices ``[0, dim_size-1]``
//     to the closed interval ``[-1, +1]``; the corner pixel *centres* are
//     at -1 and +1.
//   * align_corners == 0: maps the centre of pixel ``i`` to
//     ``-1 + (2i+1)/dim_size``; the *outer edges* of the corner pixels are
//     at -1 and +1.
//
// A dim of size 1 collapses to a single coordinate at 0 (matches numpy's
// ``np.arange(start, stop, step)`` behaviour for both conventions, where
// the single produced value is ``start``).
//
// When ``allocator`` is non-null the backing buffer is acquired from it;
// otherwise inline storage is used.
Tensor NormalisedCoords(int64_t dim_size, bool align_corners, RawBufferAllocator *allocator) {
  if (dim_size <= 0) {
    return MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {0}, 0, allocator);
  }
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {dim_size},
                                static_cast<size_t>(dim_size) * sizeof(float), allocator);
  float *data = out.AsFloat();
  if (align_corners) {
    if (dim_size == 1) {
      data[0] = -1.0f;
      return out;
    }
    const float step = 2.0f / static_cast<float>(dim_size - 1);
    for (int64_t i = 0; i < dim_size; ++i) {
      data[i] = -1.0f + step * static_cast<float>(i);
    }
  } else {
    const float step = 2.0f / static_cast<float>(dim_size);
    const float start = -1.0f + step / 2.0f;
    for (int64_t i = 0; i < dim_size; ++i) {
      data[i] = start + step * static_cast<float>(i);
    }
  }
  return out;
}

void ValidateInputs(const Tensor &theta, const Tensor &size) {
  EXT_ENFORCE_INVALID(theta.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AffineGrid: theta must be FLOAT.");
  EXT_ENFORCE_INVALID(size.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::AffineGrid: size must be INT64.");
  EXT_ENFORCE_INVALID(
      theta.shape.size() == 3,
      "kernel::AffineGrid: theta must be 3-D (N, 2, 3) for 2D or (N, 3, 4) for 3D.");
  EXT_ENFORCE_INVALID(size.shape.size() == 1,
                      "kernel::AffineGrid: size must be 1-D with 4 (2D) or 5 (3D) entries.");
  EXT_ENFORCE_INVALID(size.shape[0] == 4 || size.shape[0] == 5,
                      "kernel::AffineGrid: size must have 4 (2D) or 5 (3D) entries.");
  if (size.shape[0] == 4) {
    EXT_ENFORCE_INVALID(theta.shape[1] == 2 && theta.shape[2] == 3,
                        "kernel::AffineGrid: theta must be (N, 2, 3) for 2D.");
  } else {
    EXT_ENFORCE_INVALID(theta.shape[1] == 3 && theta.shape[2] == 4,
                        "kernel::AffineGrid: theta must be (N, 3, 4) for 3D.");
  }
  const int64_t *size_data = reinterpret_cast<const int64_t *>(size.bytes());
  EXT_ENFORCE_INVALID(size_data[0] == theta.shape[0],
                      "kernel::AffineGrid: size[0] must equal theta's batch dim N.");
}

// Computes the output shape for an AffineGrid call given a fully validated
// ``size`` input (1-D INT64 of length 4 or 5).
onnx_kernels::Shape ComputeOutputShape(const Tensor &size) {
  const int64_t *size_data = reinterpret_cast<const int64_t *>(size.bytes());
  onnx_kernels::Shape out_shape;
  out_shape.push_back(size_data[0]); // N
  if (size.shape[0] == 4) {
    out_shape.push_back(size_data[2]); // H
    out_shape.push_back(size_data[3]); // W
    out_shape.push_back(2);
  } else {
    out_shape.push_back(size_data[2]); // D
    out_shape.push_back(size_data[3]); // H
    out_shape.push_back(size_data[4]); // W
    out_shape.push_back(3);
  }
  return out_shape;
}

// Applies an (out_dim x in_dim) affine matrix ``theta`` (read row-major) to
// the homogeneous coordinate ``coords`` of length ``in_dim`` and writes the
// resulting ``out_dim`` values to ``out``.
void ApplyAffine(const float *theta, int64_t out_dim, int64_t in_dim, const float *coords,
                 float *out) {
  for (int64_t k = 0; k < out_dim; ++k) {
    float acc = 0.0f;
    for (int64_t j = 0; j < in_dim; ++j) {
      acc += theta[k * in_dim + j] * coords[j];
    }
    out[k] = acc;
  }
}

} // namespace

Tensor AffineGrid::operator()(const Tensor &theta, const Tensor &size, const Attributes &attrs,
                              RuntimeContext *rt) const {
  ValidateInputs(theta, size);
  const onnx_kernels::Shape out_shape = ComputeOutputShape(size);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  RawBufferAllocator *allocator = (rt != nullptr) ? rt->allocator() : nullptr;
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape,
                                static_cast<size_t>(total) * sizeof(float), allocator);
  (*this)(theta, size, attrs, out, allocator);
  return out;
}

void AffineGrid::operator()(const Tensor &theta, const Tensor &size, const Attributes &attrs,
                            Tensor &output, RawBufferAllocator *allocator) const {
  (void)ctx_;
  ValidateInputs(theta, size);
  const onnx_kernels::Shape expected_shape = ComputeOutputShape(size);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AffineGrid: preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::AffineGrid: preallocated output has unexpected shape.");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::AffineGrid: preallocated output buffer has unexpected size.");

  const bool align_corners = attrs.align_corners != 0;
  const float *theta_data = reinterpret_cast<const float *>(theta.bytes());
  float *out_data = reinterpret_cast<float *>(output.mutable_bytes());

  if (size.shape[0] == 4) {
    // 2D case. Output indexed as [N, H, W, 2].
    const int64_t N = expected_shape[0];
    const int64_t H = expected_shape[1];
    const int64_t W = expected_shape[2];
    const Tensor y_coords = NormalisedCoords(H, align_corners, allocator);
    const Tensor x_coords = NormalisedCoords(W, align_corners, allocator);
    const float *y_data = y_coords.AsFloat();
    const float *x_data = x_coords.AsFloat();
    // Homogeneous coord per (y, x): [x, y, 1] (matches op_affine_grid.py
    // which prepends y for dim 0 then x for dim 1 and finally takes the
    // dot product with theta rows).
    for (int64_t n = 0; n < N; ++n) {
      const float *t = theta_data + n * 6; // (2 x 3) row-major
      float *out_n = out_data + n * H * W * 2;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          const float coords[3] = {x_data[w], y_data[h], 1.0f};
          ApplyAffine(t, /*out_dim=*/2, /*in_dim=*/3, coords, out_n + (h * W + w) * 2);
        }
      }
    }
  } else {
    // 3D case. Output indexed as [N, D, H, W, 3].
    const int64_t N = expected_shape[0];
    const int64_t D = expected_shape[1];
    const int64_t H = expected_shape[2];
    const int64_t W = expected_shape[3];
    const Tensor z_coords = NormalisedCoords(D, align_corners, allocator);
    const Tensor y_coords = NormalisedCoords(H, align_corners, allocator);
    const Tensor x_coords = NormalisedCoords(W, align_corners, allocator);
    const float *z_data = z_coords.AsFloat();
    const float *y_data = y_coords.AsFloat();
    const float *x_data = x_coords.AsFloat();
    for (int64_t n = 0; n < N; ++n) {
      const float *t = theta_data + n * 12; // (3 x 4) row-major
      float *out_n = out_data + n * D * H * W * 3;
      for (int64_t d = 0; d < D; ++d) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            const float coords[4] = {x_data[w], y_data[h], z_data[d], 1.0f};
            ApplyAffine(t, /*out_dim=*/3, /*in_dim=*/4, coords, out_n + ((d * H + h) * W + w) * 3);
          }
        }
      }
    }
  }
}

void AffineGrid::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &theta = GetInput(node, 0, rt.tensors());
  const Tensor &size = GetInput(node, 1, rt.tensors());
  onnx_kernels::kernel::AffineGrid::Attributes affine_grid_attrs;
  affine_grid_attrs.align_corners = GetAttributeIntOrDefault(node, "align_corners", 0);
  onnx_kernels::kernel::AffineGrid affine_grid_kernel(rt.kernel_ctx());
  SetOutput(node, 0, affine_grid_kernel(theta, size, affine_grid_attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
