// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

std::vector<int64_t> ComputeDepthToSpaceOutputShape(const std::vector<int64_t> &in_shape,
                                                    int64_t blocksize) {
  EXT_ENFORCE_INVALID(in_shape.size() == 4, "kernel::DepthToSpace: input must be a 4-D tensor.");
  EXT_ENFORCE_INVALID(blocksize > 0, "kernel::DepthToSpace: blocksize must be positive.");
  const int64_t bs2 = blocksize * blocksize;
  EXT_ENFORCE_INVALID(in_shape[1] % bs2 == 0,
                      "kernel::DepthToSpace: input channel dim must be divisible by "
                      "blocksize * blocksize.");
  return {in_shape[0], in_shape[1] / bs2, in_shape[2] * blocksize, in_shape[3] * blocksize};
}

} // namespace

Tensor DepthToSpace::operator()(const Tensor &input, const Attributes &attrs) const {
  const std::vector<int64_t> out_shape =
      ComputeDepthToSpaceOutputShape(input.shape, attrs.blocksize);
  Tensor output("", input.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(input.data_type, input.element_count())));
  (*this)(input, attrs, output);
  return output;
}

void DepthToSpace::operator()(const Tensor &input, const Attributes &attrs, Tensor &output) const {
  const std::vector<int64_t> out_shape =
      ComputeDepthToSpaceOutputShape(input.shape, attrs.blocksize);
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::DepthToSpace: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::DepthToSpace: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(attrs.mode == "DCR" || attrs.mode == "CRD",
                      "kernel::DepthToSpace: mode must be 'DCR' or 'CRD'.");

  const std::size_t elem_size = ElementSize(input.data_type);
  const int64_t blocksize = attrs.blocksize;
  const int64_t bs2 = blocksize * blocksize;
  const int64_t N = input.shape[0];
  const int64_t C_in = input.shape[1];
  const int64_t H = input.shape[2];
  const int64_t W = input.shape[3];
  const int64_t C_out = C_in / bs2;
  const int64_t H_out = H * blocksize;
  const int64_t W_out = W * blocksize;
  const bool is_dcr = (attrs.mode == "DCR");

  // Input strides (row-major over [N, C_in, H, W]).
  const int64_t in_stride_n = C_in * H * W;
  const int64_t in_stride_c = H * W;
  const int64_t in_stride_h = W;

  // Output strides (row-major over [N, C_out, H_out, W_out]).
  const int64_t out_stride_n = C_out * H_out * W_out;
  const int64_t out_stride_c = H_out * W_out;
  const int64_t out_stride_h = W_out;

  const uint8_t *const in_ptr = input.bytes();
  uint8_t *const out_ptr = output.data.data();

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c_out = 0; c_out < C_out; ++c_out) {
      for (int64_t h_out = 0; h_out < H_out; ++h_out) {
        const int64_t h = h_out / blocksize;
        const int64_t bh = h_out % blocksize;
        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
          const int64_t w = w_out / blocksize;
          const int64_t bw = w_out % blocksize;
          // DCR: c_in = bh * (blocksize * C_out) + bw * C_out + c_out
          //   tmp = reshape(x, [b, blocksize, blocksize, C_out, h, w])
          //   tmp = transpose(tmp, [0, 3, 4, 1, 5, 2])
          //   y   = reshape(tmp, [b, C_out, h*blocksize, w*blocksize])
          // CRD: c_in = c_out * bs2 + bh * blocksize + bw
          //   tmp = reshape(x, [b, C_out, blocksize, blocksize, h, w])
          //   tmp = transpose(tmp, [0, 1, 4, 2, 5, 3])
          const int64_t c_in = is_dcr ? (bh * blocksize * C_out + bw * C_out + c_out)
                                      : (c_out * bs2 + bh * blocksize + bw);
          const int64_t in_off = n * in_stride_n + c_in * in_stride_c + h * in_stride_h + w;
          const int64_t out_off =
              n * out_stride_n + c_out * out_stride_c + h_out * out_stride_h + w_out;
          std::memcpy(out_ptr + static_cast<std::size_t>(out_off) * elem_size,
                      in_ptr + static_cast<std::size_t>(in_off) * elem_size, elem_size);
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
