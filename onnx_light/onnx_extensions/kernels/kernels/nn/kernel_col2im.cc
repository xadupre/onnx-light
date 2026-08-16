// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Resolves attribute defaults given the spatial rank inferred from the
// (already-validated) ``image_shape`` and ``block_shape`` inputs.
void ResolveAttributes(size_t n_spatial, Col2Im::Attributes &attrs) {
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(n_spatial, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == n_spatial,
                      "kernel::Col2Im: 'dilations' size must match spatial rank.");
  if (attrs.pads.empty()) {
    attrs.pads.assign(n_spatial * 2, 0);
  }
  EXT_ENFORCE_INVALID(attrs.pads.size() == n_spatial * 2,
                      "kernel::Col2Im: 'pads' size must be 2 * spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(n_spatial, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == n_spatial,
                      "kernel::Col2Im: 'strides' size must match spatial rank.");
}

// Reads a 1-D INT64 tensor into a fixed-capacity shape of int64s. Throws if the
// dtype or rank are wrong. The result is rank-sized (spatial rank) and therefore
// fits within ``onnx_kernels::Shape`` without heap allocation.
onnx_kernels::Shape ReadInt64Shape(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::INT64), "kernel::Col2Im: '",
                      name, "' must be INT64.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1, "kernel::Col2Im: '", name, "' must be a 1-D tensor.");
  const int64_t n = t.shape[0];
  const int64_t *src = t.AsInt64();
  onnx_kernels::Shape out;
  out.assign(src, src + n);
  return out;
}

} // namespace

Tensor Col2Im::operator()(const Tensor &input, const Tensor &image_shape, const Tensor &block_shape,
                          const Attributes &attrs, RuntimeContext *rt) const {
  const onnx_kernels::Shape image_shape_vec = ReadInt64Shape(image_shape, "image_shape");
  const onnx_kernels::Shape block_shape_vec = ReadInt64Shape(block_shape, "block_shape");
  EXT_ENFORCE_INVALID(image_shape_vec.size() == block_shape_vec.size(),
                      "kernel::Col2Im: 'image_shape' and 'block_shape' must have the same length.");

  onnx_kernels::Shape out_shape;
  out_shape.reserve(2 + image_shape_vec.size());
  EXT_ENFORCE_INVALID(input.shape.size() == 3, "kernel::Col2Im: 'input' must be rank 3.");
  out_shape.push_back(input.shape[0]);
  int64_t block_product = 1;
  for (int64_t b : block_shape_vec) {
    EXT_ENFORCE_INVALID(b > 0, "kernel::Col2Im: every 'block_shape' entry must be > 0.");
    block_product *= b;
  }
  EXT_ENFORCE_INVALID(input.shape[1] % block_product == 0,
                      "kernel::Col2Im: input.shape[1] must be divisible by product(block_shape).");
  out_shape.push_back(input.shape[1] / block_product);
  for (int64_t d : image_shape_vec) {
    EXT_ENFORCE_INVALID(d > 0, "kernel::Col2Im: every 'image_shape' entry must be > 0.");
    out_shape.push_back(d);
  }

  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const size_t out_n_bytes = static_cast<size_t>(total) * sizeof(float);
  Tensor out =
      MakeOutputTensor(input.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, image_shape, block_shape, attrs, out);
  return out;
}

void Col2Im::operator()(const Tensor &input, const Tensor &image_shape, const Tensor &block_shape,
                        const Attributes &attrs, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::Col2Im: 'input' must be FLOAT.");
  EXT_ENFORCE_INVALID(input.shape.size() == 3, "kernel::Col2Im: 'input' must be rank 3.");

  const onnx_kernels::Shape image_shape_vec = ReadInt64Shape(image_shape, "image_shape");
  const onnx_kernels::Shape block_shape_vec = ReadInt64Shape(block_shape, "block_shape");
  EXT_ENFORCE_INVALID(image_shape_vec.size() == block_shape_vec.size(),
                      "kernel::Col2Im: 'image_shape' and 'block_shape' must have the same length.");
  const size_t n_spatial = image_shape_vec.size();
  EXT_ENFORCE_INVALID(n_spatial >= 1,
                      "kernel::Col2Im: 'image_shape' must have at least one entry.");

  Attributes resolved = attrs;
  ResolveAttributes(n_spatial, resolved);

  // Compute the expected output shape and number of blocks per spatial axis.
  const int64_t N = input.shape[0];
  int64_t block_product = 1;
  for (int64_t b : block_shape_vec) {
    EXT_ENFORCE_INVALID(b > 0, "kernel::Col2Im: every 'block_shape' entry must be > 0.");
    block_product *= b;
  }
  EXT_ENFORCE_INVALID(input.shape[1] % block_product == 0,
                      "kernel::Col2Im: input.shape[1] must be divisible by product(block_shape).");
  const int64_t C = input.shape[1] / block_product;

  onnx_kernels::Shape n_blocks;
  n_blocks.assign(n_spatial, 0);
  int64_t L_expected = 1;
  for (size_t i = 0; i < n_spatial; ++i) {
    const int64_t eff_k = (block_shape_vec[i] - 1) * resolved.dilations[i] + 1;
    const int64_t padded = image_shape_vec[i] + resolved.pads[i] + resolved.pads[i + n_spatial];
    EXT_ENFORCE_INVALID(resolved.strides[i] > 0,
                        "kernel::Col2Im: every 'strides' entry must be > 0.");
    EXT_ENFORCE_INVALID(padded >= eff_k,
                        "kernel::Col2Im: padded image dim is smaller than the effective kernel.");
    n_blocks[i] = (padded - eff_k) / resolved.strides[i] + 1;
    L_expected *= n_blocks[i];
  }
  EXT_ENFORCE_INVALID(input.shape[2] == L_expected,
                      "kernel::Col2Im: input.shape[2] (L) does not match the number of blocks "
                      "implied by image_shape, block_shape, pads, strides and dilations.");

  onnx_kernels::Shape expected_out_shape;
  expected_out_shape.reserve(2 + n_spatial);
  expected_out_shape.push_back(N);
  expected_out_shape.push_back(C);
  for (int64_t d : image_shape_vec) {
    expected_out_shape.push_back(d);
  }
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::Col2Im preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::Col2Im preallocated output shape must be "
                      "(N, C, image_shape[0], ...).");
  int64_t total = 1;
  for (int64_t d : expected_out_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::Col2Im preallocated output buffer has unexpected size.");

  // Per-axis image stride into the output buffer.
  onnx_kernels::Shape image_strides;
  image_strides.assign(n_spatial, 0);
  {
    int64_t s = 1;
    for (size_t i = n_spatial; i-- > 0;) {
      image_strides[i] = s;
      s *= image_shape_vec[i];
    }
  }

  const int64_t image_total = total / (N * C);

  float *py = output.AsFloat();
  for (int64_t i = 0; i < total; ++i) {
    py[i] = 0.0f;
  }
  const float *px = input.AsFloat();

  // Iterate over (n, c, block index l, kernel index k) in the same
  // lex order used by the input layout, accumulating into the output.
  onnx_kernels::Shape block_idx;
  block_idx.assign(n_spatial, 0);
  onnx_kernels::Shape kernel_idx;
  kernel_idx.assign(n_spatial, 0);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      float *out_image = py + (n * C + c) * image_total;
      // Walk blocks in lex order: outer-most dim is slowest-changing.
      for (size_t i = 0; i < n_spatial; ++i) {
        block_idx[i] = 0;
      }
      for (int64_t l = 0; l < L_expected; ++l) {
        // Walk kernel positions in lex order: outer-most dim slowest.
        for (size_t i = 0; i < n_spatial; ++i) {
          kernel_idx[i] = 0;
        }
        for (int64_t k = 0; k < block_product; ++k) {
          // Compute the corresponding (unpadded) image position.
          int64_t flat_image_offset = 0;
          bool in_bounds = true;
          for (size_t i = 0; i < n_spatial; ++i) {
            const int64_t img_pos = block_idx[i] * resolved.strides[i] +
                                    kernel_idx[i] * resolved.dilations[i] - resolved.pads[i];
            if (img_pos < 0 || img_pos >= image_shape_vec[i]) {
              in_bounds = false;
              break;
            }
            flat_image_offset += img_pos * image_strides[i];
          }
          if (in_bounds) {
            const int64_t in_chan = c * block_product + k;
            const float v = px[(n * input.shape[1] + in_chan) * input.shape[2] + l];
            out_image[flat_image_offset] += v;
          }
          // Increment kernel_idx (rightmost / fastest first).
          for (size_t i = n_spatial; i-- > 0;) {
            if (++kernel_idx[i] < block_shape_vec[i]) {
              break;
            }
            kernel_idx[i] = 0;
          }
        }
        // Increment block_idx (rightmost / fastest first).
        for (size_t i = n_spatial; i-- > 0;) {
          if (++block_idx[i] < n_blocks[i]) {
            break;
          }
          block_idx[i] = 0;
        }
      }
    }
  }
}

void Col2Im::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const Tensor &image_shape = GetInput(node, 1, rt.tensors());
  const Tensor &block_shape = GetInput(node, 2, rt.tensors());
  onnx_kernels::kernel::Col2Im::Attributes attrs;
  attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
  attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
  attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
  onnx_kernels::kernel::Col2Im k(rt.kernel_ctx());
  SetOutput(node, 0, k(input, image_shape, block_shape, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
