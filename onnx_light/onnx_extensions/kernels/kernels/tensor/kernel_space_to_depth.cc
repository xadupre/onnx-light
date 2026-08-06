// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ComputeSpaceToDepthOutputShape(const onnx_kernels::Shape &in_shape,
                                                   int64_t blocksize) {
  EXT_ENFORCE_INVALID(in_shape.size() == 4, "kernel::SpaceToDepth: input must be a 4-D tensor.");
  EXT_ENFORCE_INVALID(blocksize > 0, "kernel::SpaceToDepth: blocksize must be positive.");
  EXT_ENFORCE_INVALID(in_shape[2] % blocksize == 0,
                      "kernel::SpaceToDepth: input H must be divisible by blocksize.");
  EXT_ENFORCE_INVALID(in_shape[3] % blocksize == 0,
                      "kernel::SpaceToDepth: input W must be divisible by blocksize.");
  return {in_shape[0], in_shape[1] * blocksize * blocksize, in_shape[2] / blocksize,
          in_shape[3] / blocksize};
}

} // namespace

Tensor SpaceToDepth::operator()(const Tensor &input, const Attributes &attrs,
                                RuntimeContext *rt) const {
  const onnx_kernels::Shape out_shape =
      ComputeSpaceToDepthOutputShape(input.shape, attrs.blocksize);
  const size_t output_n_bytes = PackedByteSize(input.data_type, input.element_count());
  Tensor output =
      MakeOutputTensor(input.data_type, out_shape, output_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, attrs, output);
  return output;
}

void SpaceToDepth::operator()(const Tensor &input, const Attributes &attrs, Tensor &output) const {
  const onnx_kernels::Shape out_shape =
      ComputeSpaceToDepthOutputShape(input.shape, attrs.blocksize);
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::SpaceToDepth: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::SpaceToDepth: preallocated output shape mismatch.");

  const std::size_t elem_size = ElementSize(input.data_type);
  const int64_t blocksize = attrs.blocksize;
  const int64_t N = input.shape[0];
  const int64_t C = input.shape[1];
  const int64_t H = input.shape[2];
  const int64_t W = input.shape[3];
  const int64_t H_out = H / blocksize;
  const int64_t W_out = W / blocksize;
  const int64_t C_out = C * blocksize * blocksize;

  // Input strides (row-major over [N, C, H, W]).
  const int64_t in_stride_n = C * H * W;
  const int64_t in_stride_c = H * W;
  const int64_t in_stride_h = W;

  // Output strides (row-major over [N, C_out, H_out, W_out]).
  const int64_t out_stride_n = C_out * H_out * W_out;
  const int64_t out_stride_c = H_out * W_out;
  const int64_t out_stride_h = W_out;

  const uint8_t *const in_ptr = input.bytes();
  uint8_t *const out_ptr = output.mutable_bytes();

  // ONNX spec (SpaceToDepth):
  //   tmp = reshape(x, [N, C, H/bs, bs, W/bs, bs])
  //   tmp = transpose(tmp, [0, 3, 5, 1, 2, 4])
  //   y   = reshape(tmp, [N, C*bs*bs, H/bs, W/bs])
  // So for an output element y[n, c_out, h_out, w_out]:
  //   bh   = c_out / (bs * C)
  //   bw   = (c_out / C) % bs
  //   c    = c_out % C
  //   h_in = h_out * bs + bh
  //   w_in = w_out * bs + bw
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c_out = 0; c_out < C_out; ++c_out) {
      const int64_t bh = c_out / (blocksize * C);
      const int64_t bw = (c_out / C) % blocksize;
      const int64_t c = c_out % C;
      for (int64_t h_out = 0; h_out < H_out; ++h_out) {
        const int64_t h_in = h_out * blocksize + bh;
        for (int64_t w_out = 0; w_out < W_out; ++w_out) {
          const int64_t w_in = w_out * blocksize + bw;
          const int64_t in_off = n * in_stride_n + c * in_stride_c + h_in * in_stride_h + w_in;
          const int64_t out_off =
              n * out_stride_n + c_out * out_stride_c + h_out * out_stride_h + w_out;
          std::memcpy(out_ptr + static_cast<std::size_t>(out_off) * elem_size,
                      in_ptr + static_cast<std::size_t>(in_off) * elem_size, elem_size);
        }
      }
    }
  }
}

void SpaceToDepth::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
  const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
  EXT_ENFORCE_INVALID(blocksize_attr != nullptr,
                      "RunNode: SpaceToDepth requires attribute 'blocksize'.");
  EXT_ENFORCE_INVALID(!(blocksize_attr->type() != AttributeProto::AttributeType::INT),
                      "RunNode: SpaceToDepth attribute 'blocksize' must be INT.");
  attrs.blocksize = blocksize_attr->i();
  onnx_kernels::kernel::SpaceToDepth kernel(rt.kernel_ctx());
  SetOutput(node, 0, kernel(input, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
