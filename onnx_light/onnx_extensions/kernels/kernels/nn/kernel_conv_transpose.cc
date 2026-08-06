// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Resolves the spatial attribute vectors to canonical sizes. ``pads`` (when
// ``output_shape`` or ``auto_pad`` is set) is recomputed in
// ``ComputeOutputShape`` below.
void ResolveAttributes(const Tensor &x, const Tensor &w, ConvTranspose::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  if (attrs.kernel_shape.empty()) {
    attrs.kernel_shape.assign(w.shape.begin() + 2, w.shape.end());
  }
  EXT_ENFORCE_INVALID(attrs.kernel_shape.size() == spatial_rank,
                      "kernel::ConvTranspose: 'kernel_shape' size must match input spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == spatial_rank,
                      "kernel::ConvTranspose: 'strides' size must match input spatial rank.");
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == spatial_rank,
                      "kernel::ConvTranspose: 'dilations' size must match input spatial rank.");
  if (attrs.output_padding.empty()) {
    attrs.output_padding.assign(spatial_rank, 0);
  }
  EXT_ENFORCE_INVALID(
      attrs.output_padding.size() == spatial_rank,
      "kernel::ConvTranspose: 'output_padding' size must match input spatial rank.");
  if (attrs.pads.empty()) {
    attrs.pads.assign(spatial_rank * 2, 0);
  }
  EXT_ENFORCE_INVALID(attrs.pads.size() == spatial_rank * 2,
                      "kernel::ConvTranspose: 'pads' size must be 2 * spatial rank.");
  if (!attrs.output_shape.empty()) {
    EXT_ENFORCE_INVALID(
        attrs.output_shape.size() == spatial_rank,
        "kernel::ConvTranspose: 'output_shape' size must match input spatial rank.");
  }
  if (attrs.auto_pad != AutoPad::kNotSet) {
    EXT_ENFORCE_INVALID(attrs.auto_pad == AutoPad::kSameUpper ||
                            attrs.auto_pad == AutoPad::kSameLower ||
                            attrs.auto_pad == AutoPad::kValid,
                        "kernel::ConvTranspose: invalid 'auto_pad' value.");
  }
}

void ValidateInputs(const Tensor &x, const Tensor &w, const Tensor &b,
                    const ConvTranspose::Attributes &attrs) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ConvTranspose: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ConvTranspose: W must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 3, "kernel::ConvTranspose: X must have rank >= 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == x.shape.size(),
                      "kernel::ConvTranspose: W rank must match X rank.");
  EXT_ENFORCE_INVALID(attrs.group >= 1, "kernel::ConvTranspose: 'group' must be >= 1.");
  EXT_ENFORCE_INVALID(x.shape[1] == w.shape[0],
                      "kernel::ConvTranspose: X.shape[1] must equal W.shape[0].");
  const int64_t M = w.shape[1] * attrs.group;
  if (!b.shape.empty()) {
    EXT_ENFORCE_INVALID(b.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::ConvTranspose: B must be FLOAT when present.");
    EXT_ENFORCE_INVALID(b.shape.size() == 1 && b.shape[0] == M,
                        "kernel::ConvTranspose: B must be 1-D of length M.");
  }
}

// Computes the per-axis output spatial dimensions and finalizes ``pads``.
// Mirrors the upstream ``convTransposeShapeInference`` rules for the
// interaction between ``output_shape``, ``output_padding``, ``pads`` and
// ``auto_pad``.
onnx_kernels::Shape ComputeOutputShape(const Tensor &x, ConvTranspose::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  onnx_kernels::Shape out_spatial;
  out_spatial.assign(spatial_rank, 0);
  const bool has_output_shape = !attrs.output_shape.empty();
  const AutoPad auto_pad = attrs.auto_pad;

  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t iD = x.shape[i + 2];
    const int64_t k = attrs.kernel_shape[i];
    const int64_t s = attrs.strides[i];
    const int64_t d = attrs.dilations[i];
    const int64_t op = attrs.output_padding[i];
    const int64_t eff_k = (k - 1) * d + 1;

    if (has_output_shape) {
      const int64_t out = attrs.output_shape[i];
      const int64_t total_pad = std::max<int64_t>(0, s * (iD - 1) + op + eff_k - out);
      if (auto_pad == AutoPad::kSameLower) {
        attrs.pads[i + spatial_rank] = total_pad / 2;
        attrs.pads[i] = total_pad - total_pad / 2;
      } else {
        attrs.pads[i] = total_pad / 2;
        attrs.pads[i + spatial_rank] = total_pad - total_pad / 2;
      }
      out_spatial[i] = out;
    } else if (auto_pad == AutoPad::kSameUpper || auto_pad == AutoPad::kSameLower) {
      // No explicit output_shape: per upstream, compute total_padding from
      // stride. ``output_spatial = iD * stride`` is the upstream default.
      const int64_t out = iD * s;
      const int64_t total_pad = std::max<int64_t>(0, s * (iD - 1) + op + eff_k - out);
      if (auto_pad == AutoPad::kSameUpper) {
        attrs.pads[i] = total_pad / 2;
        attrs.pads[i + spatial_rank] = total_pad - total_pad / 2;
      } else {
        attrs.pads[i + spatial_rank] = total_pad / 2;
        attrs.pads[i] = total_pad - total_pad / 2;
      }
      out_spatial[i] = out;
    } else {
      // NOTSET (or VALID with empty pads): use the standard formula.
      if (auto_pad == AutoPad::kValid) {
        attrs.pads[i] = 0;
        attrs.pads[i + spatial_rank] = 0;
      }
      out_spatial[i] = s * (iD - 1) + op + eff_k - attrs.pads[i] - attrs.pads[i + spatial_rank];
    }
  }
  return out_spatial;
}

} // namespace

Tensor ConvTranspose::operator()(const Tensor &x, const Tensor &w, const Tensor &b,
                                 const Attributes &attrs, RuntimeContext *rt) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, b, resolved);
  onnx_kernels::Shape out_spatial = ComputeOutputShape(x, resolved);
  const int64_t M = w.shape[1] * resolved.group;
  onnx_kernels::Shape out_shape;
  out_shape.reserve(x.shape.size());
  out_shape.push_back(x.shape[0]);
  out_shape.push_back(M);
  for (int64_t d : out_spatial) {
    out_shape.push_back(d);
  }
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const size_t out_n_bytes = static_cast<size_t>(total) * sizeof(float);
  Tensor out =
      MakeOutputTensor(x.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, w, b, resolved, out);
  return out;
}

void ConvTranspose::operator()(const Tensor &x, const Tensor &w, const Tensor &b,
                               const Attributes &attrs, Tensor &output) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, b, resolved);
  onnx_kernels::Shape out_spatial = ComputeOutputShape(x, resolved);
  const int64_t M = w.shape[1] * resolved.group;
  onnx_kernels::Shape expected_shape;
  expected_shape.reserve(x.shape.size());
  expected_shape.push_back(x.shape[0]);
  expected_shape.push_back(M);
  for (int64_t d : out_spatial) {
    expected_shape.push_back(d);
  }
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ConvTranspose preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::ConvTranspose preallocated output shape mismatch.");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::ConvTranspose preallocated output buffer has unexpected size.");

  const size_t spatial_rank = x.shape.size() - 2;
  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t C_per_group = C / resolved.group;
  const int64_t M_per_group = w.shape[1];

  onnx_kernels::Shape iD;
  iD.assign(spatial_rank, 0);
  onnx_kernels::Shape oD = out_spatial;
  for (size_t i = 0; i < spatial_rank; ++i) {
    iD[i] = x.shape[i + 2];
  }

  auto compute_strides = [](const auto &dims) {
    onnx_kernels::Shape strides;
    strides.assign(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  };
  const onnx_kernels::Shape out_strides = compute_strides(oD);
  const onnx_kernels::Shape &k_shape = resolved.kernel_shape;

  int64_t in_spatial_size = 1;
  for (int64_t d : iD) {
    in_spatial_size *= d;
  }
  int64_t out_spatial_size = 1;
  for (int64_t d : oD) {
    out_spatial_size *= d;
  }
  int64_t kernel_size = 1;
  for (int64_t d : k_shape) {
    kernel_size *= d;
  }

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pb = b.shape.empty() ? nullptr : b.AsFloat();
  float *py = output.AsFloat();

  // Initialize output with bias (or zero).
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const float bias = pb ? pb[m] : 0.0f;
      float *plane = py + ((n * M + m) * out_spatial_size);
      std::fill(plane, plane + out_spatial_size, bias);
    }
  }

  // Scatter-add x * w into y. For each input spatial position and each
  // kernel position, the output position is
  // ``oa = ia * stride - pad_begin + ka * dilation``.
  onnx_kernels::Shape iidx;
  iidx.assign(spatial_rank, 0);
  onnx_kernels::Shape kidx;
  kidx.assign(spatial_rank, 0);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t g = 0; g < resolved.group; ++g) {
      for (int64_t m_in_group = 0; m_in_group < M_per_group; ++m_in_group) {
        const int64_t m = g * M_per_group + m_in_group;
        float *out_plane = py + ((n * M + m) * out_spatial_size);
        for (int64_t c_in_group = 0; c_in_group < C_per_group; ++c_in_group) {
          const int64_t c = g * C_per_group + c_in_group;
          const float *x_plane = px + ((n * C + c) * in_spatial_size);
          // W is laid out (C, M/group, k1..kk). Channel-first index 'c'.
          const float *w_plane = pw + ((c * M_per_group + m_in_group) * kernel_size);
          std::fill(iidx.begin(), iidx.end(), 0);
          for (int64_t ip = 0; ip < in_spatial_size; ++ip) {
            const float xv = x_plane[ip];
            std::fill(kidx.begin(), kidx.end(), 0);
            for (int64_t kp = 0; kp < kernel_size; ++kp) {
              bool in_bounds = true;
              int64_t out_off = 0;
              for (size_t a = 0; a < spatial_rank; ++a) {
                const int64_t oa = iidx[a] * resolved.strides[a] - resolved.pads[a] +
                                   kidx[a] * resolved.dilations[a];
                if (oa < 0 || oa >= oD[a]) {
                  in_bounds = false;
                  break;
                }
                out_off += oa * out_strides[a];
              }
              if (in_bounds) {
                out_plane[out_off] += xv * w_plane[kp];
              }
              for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
                if (++kidx[a] < k_shape[a]) {
                  break;
                }
                kidx[a] = 0;
              }
            }
            for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
              if (++iidx[a] < iD[a]) {
                break;
              }
              iidx[a] = 0;
            }
          }
        }
      }
    }
  }
}

void ConvTranspose::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 2);
  EXT_ENFORCE_INVALID(!(node.input_size() > 3),
                      "RunNode: op 'ConvTranspose' expects at most 3 inputs, got ",
                      node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &w = GetInput(node, 1, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 2, rt.tensors());
  onnx_kernels::kernel::ConvTranspose::Attributes attrs;
  attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
  attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
  attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
  attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
  attrs.output_padding = GetAttributeIntsOrDefault(node, "output_padding", {});
  attrs.output_shape = GetAttributeIntsOrDefault(node, "output_shape", {});
  attrs.group = GetAttributeIntOrDefault(node, "group", 1);
  attrs.auto_pad = onnx_kernels::kernel::AutoPadFromString(
      GetAttributeStringOrDefault(node, "auto_pad", "NOTSET"));
  onnx_kernels::kernel::ConvTranspose k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, w, b != nullptr ? *b : Tensor{}, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
