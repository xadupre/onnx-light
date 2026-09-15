// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Indices are global flattened offsets into the final output, including when
// output_shape overrides the dimensions inferred from the pooling attributes.
Tensor RunMaxUnpool(const Tensor &x, const Tensor &indices, const Shape &kernel_shape,
                    const Shape &strides_in, const Shape &pads_in,
                    const Shape *explicit_output_shape, RuntimeContext *rt) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::MaxUnpool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(indices.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::MaxUnpool: indices must be INT64.");
  EXT_ENFORCE_INVALID(x.shape == indices.shape,
                      "kernel::MaxUnpool: indices must have the same shape as x.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::MaxUnpool: x must have rank >= 2 (N, C, ...).");
  EXT_ENFORCE_INVALID(!kernel_shape.empty(), "kernel::MaxUnpool: kernel_shape must be non-empty.");
  const size_t k = kernel_shape.size();
  EXT_ENFORCE_INVALID(x.shape.size() == k + 2,
                      "kernel::MaxUnpool: x rank must equal kernel_shape.size() + 2.");

  Shape strides;
  if (strides_in.empty()) {
    strides.assign(k, 1);
  } else {
    strides = strides_in;
  }
  Shape pads;
  if (pads_in.empty()) {
    pads.assign(2 * k, 0);
  } else {
    pads = pads_in;
  }
  EXT_ENFORCE_INVALID(strides.size() == k,
                      "kernel::MaxUnpool: strides must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(pads.size() == 2 * k,
                      "kernel::MaxUnpool: pads must have 2 * k entries (begins then ends).");
  for (size_t i = 0; i < k; ++i) {
    EXT_ENFORCE_INVALID(kernel_shape[i] > 0,
                        "kernel::MaxUnpool: kernel_shape entries must be positive.");
    EXT_ENFORCE_INVALID(strides[i] > 0, "kernel::MaxUnpool: strides entries must be positive.");
    EXT_ENFORCE_INVALID(pads[i] >= 0 && pads[i + k] >= 0,
                        "kernel::MaxUnpool: pads entries must be non-negative.");
  }

  Shape out_shape = explicit_output_shape ? *explicit_output_shape : x.shape;
  EXT_ENFORCE_INVALID(out_shape.size() == x.shape.size(),
                      "kernel::MaxUnpool: output_shape rank must match x rank.");
  if (explicit_output_shape == nullptr) {
    for (size_t i = 0; i < k; ++i) {
      const int64_t in_dim = x.shape[i + 2];
      EXT_ENFORCE_INVALID(
          in_dim <= 1 ||
              in_dim - 1 <= (std::numeric_limits<int64_t>::max() - kernel_shape[i]) / strides[i],
          "kernel::MaxUnpool: inferred output dimension overflows.");
      out_shape[i + 2] = strides[i] * (in_dim - 1) + kernel_shape[i];
      EXT_ENFORCE_INVALID(out_shape[i + 2] >= pads[i],
                          "kernel::MaxUnpool: inferred output dimension is negative.");
      out_shape[i + 2] -= pads[i];
      EXT_ENFORCE_INVALID(out_shape[i + 2] >= pads[i + k],
                          "kernel::MaxUnpool: inferred output dimension is negative.");
      out_shape[i + 2] -= pads[i + k];
    }
  }
  for (int64_t d : out_shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::MaxUnpool: output_shape entries must be non-negative.");
  }
  int64_t out_total = std::find(out_shape.begin(), out_shape.end(), 0) != out_shape.end() ? 0 : 1;
  for (int64_t d : out_shape) {
    EXT_ENFORCE_INVALID(out_total == 0 || d <= std::numeric_limits<int64_t>::max() / out_total,
                        "kernel::MaxUnpool: output size overflows.");
    out_total *= d;
  }
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(out_total) <=
                          std::numeric_limits<size_t>::max() / sizeof(float),
                      "kernel::MaxUnpool: output byte size overflows.");
  const size_t x_total = x.element_count();
  const int64_t *pi = indices.AsInt64();
  for (size_t i = 0; i < x_total; ++i) {
    EXT_ENFORCE_INVALID(pi[i] >= 0 && pi[i] < out_total, "kernel::MaxUnpool: indices entry ",
                        std::to_string(pi[i]), " out of range for output of ",
                        std::to_string(out_total), " elements.");
  }
  const size_t out_n_bytes = static_cast<size_t>(out_total) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);
  if (out_total == 0) {
    return out;
  }
  float *po = reinterpret_cast<float *>(out.mutable_bytes());
  std::fill(po, po + static_cast<size_t>(out_total), 0.0f);
  const float *px = x.AsFloat();
  for (size_t i = 0; i < x_total; ++i) {
    po[static_cast<size_t>(pi[i])] = px[i];
  }

  return out;
}

} // namespace

Tensor MaxUnpool::operator()(const Tensor &x, const Tensor &indices, const Shape &kernel_shape,
                             const Shape &strides, const Shape &pads, RuntimeContext *rt) const {
  return RunMaxUnpool(x, indices, kernel_shape, strides, pads, /*explicit_output_shape=*/nullptr,
                      rt);
}

Tensor MaxUnpool::operator()(const Tensor &x, const Tensor &indices, const Tensor &output_shape,
                             const Shape &kernel_shape, const Shape &strides, const Shape &pads,
                             RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(output_shape.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::MaxUnpool: output_shape must be INT64.");
  EXT_ENFORCE_INVALID(output_shape.shape.size() == 1,
                      "kernel::MaxUnpool: output_shape must be a rank-1 tensor.");
  EXT_ENFORCE_INVALID(static_cast<size_t>(output_shape.shape[0]) == x.shape.size(),
                      "kernel::MaxUnpool: output_shape size must match x rank.");
  const int64_t *posh = output_shape.AsInt64();
  Shape shape_vec;
  shape_vec.assign(static_cast<size_t>(output_shape.shape[0]), 0);
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    shape_vec[i] = posh[i];
  }
  return RunMaxUnpool(x, indices, kernel_shape, strides, pads, &shape_vec, rt);
}

void MaxUnpool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputRange(node, 2, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &indices = GetInput(node, 1, rt.tensors());
  const onnx_kernels::Shape kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
  const onnx_kernels::Shape strides = GetAttributeIntsOrDefault(node, "strides", {});
  const onnx_kernels::Shape pads = GetAttributeIntsOrDefault(node, "pads", {});
  onnx_kernels::kernel::MaxUnpool k(rt.kernel_ctx());
  const Tensor *output_shape = GetOptionalInput(node, 2, rt.tensors());
  if (output_shape != nullptr) {
    SetOutput(node, 0, k(x, indices, *output_shape, kernel_shape, strides, pads, &rt), rt);
  } else {
    SetOutput(node, 0, k(x, indices, kernel_shape, strides, pads, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
