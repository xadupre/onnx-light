// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Mirrors the upstream ONNX reference (``onnx.reference.ops.op_max_unpool``)
// behaviour:
//   1. Compute the *inferred* output shape from ``kernel_shape``, ``strides``
//      and ``pads`` regardless of whether ``output_shape`` is provided.
//   2. Allocate a buffer of ``prod(inferred_shape)`` zeros and scatter
//      ``X.flat[i]`` to ``Y.flat[indices.flat[i]]`` (indices are global flat
//      offsets into that inferred-shape buffer, not per-channel offsets).
//   3. If ``output_shape`` is provided, copy the inferred region into the
//      top-left corner of a zero buffer of shape ``output_shape``.
Tensor RunMaxUnpool(const Tensor &x, const Tensor &indices, const Shape &kernel_shape,
                    const Shape &strides_in, const Shape &pads_in,
                    const Shape *explicit_output_shape, RawBufferAllocator *allocator) {
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

  Shape inferred_shape;
  inferred_shape.assign(x.shape.size(), 0);
  inferred_shape[0] = x.shape[0];
  inferred_shape[1] = x.shape[1];
  for (size_t i = 0; i < k; ++i) {
    inferred_shape[i + 2] =
        strides[i] * (x.shape[i + 2] - 1) + kernel_shape[i] - pads[i] - pads[i + k];
    EXT_ENFORCE_INVALID(inferred_shape[i + 2] > 0,
                        "kernel::MaxUnpool: inferred output spatial dimension is non-positive.");
  }
  int64_t inferred_total = 1;
  for (int64_t d : inferred_shape) {
    inferred_total *= d;
  }

  int64_t x_total = 1;
  for (int64_t d : x.shape) {
    x_total *= d;
  }

  // Draw the scatter buffer from the runtime allocator (falling back to a
  // std::vector when no allocator is attached). The allocator-backed path is
  // not guaranteed zeroed, so the buffer is explicitly zero-filled before the
  // scatter below.
  detail::TemporaryTypedBuffer<float> y_inferred_buf(static_cast<size_t>(inferred_total), allocator,
                                                     "kernel::MaxUnpool y_inferred");
  float *y_inferred = y_inferred_buf.data();
  std::fill(y_inferred, y_inferred + static_cast<size_t>(inferred_total), 0.0f);
  const float *px = x.AsFloat();
  const int64_t *pi = indices.AsInt64();
  for (int64_t i = 0; i < x_total; ++i) {
    const int64_t idx = pi[i];
    EXT_ENFORCE_INVALID(idx >= 0 && idx < inferred_total, "kernel::MaxUnpool: indices entry ",
                        std::to_string(idx), " out of range for inferred output of ",
                        std::to_string(inferred_total), " elements.");
    y_inferred[static_cast<size_t>(idx)] = px[i];
  }

  if (explicit_output_shape == nullptr) {
    Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), inferred_shape,
                                  static_cast<size_t>(inferred_total) * sizeof(float), allocator);
    std::memcpy(out.mutable_bytes(), y_inferred,
                static_cast<size_t>(inferred_total) * sizeof(float));
    return out;
  }

  EXT_ENFORCE_INVALID(explicit_output_shape->size() == x.shape.size(),
                      "kernel::MaxUnpool: output_shape rank must match x rank.");
  for (int64_t d : *explicit_output_shape) {
    EXT_ENFORCE_INVALID(d > 0, "kernel::MaxUnpool: output_shape entries must be positive.");
  }
  // The output dimensions must be at least as large as the inferred ones; the
  // inferred region is copied into the top-left corner.
  for (size_t i = 0; i < x.shape.size(); ++i) {
    EXT_ENFORCE_INVALID((*explicit_output_shape)[i] >= inferred_shape[i],
                        "kernel::MaxUnpool: output_shape must be >= inferred shape on every axis.");
  }
  int64_t out_total = 1;
  for (int64_t d : *explicit_output_shape) {
    out_total *= d;
  }
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), *explicit_output_shape,
                                static_cast<size_t>(out_total) * sizeof(float), allocator);
  float *po = reinterpret_cast<float *>(out.mutable_bytes());
  std::fill(po, po + static_cast<size_t>(out_total), 0.0f);

  // Compute strides for both layouts and copy the inferred region into the
  // top-left corner of the output.
  Shape in_strides;
  in_strides.assign(x.shape.size(), 1);
  Shape out_strides;
  out_strides.assign(x.shape.size(), 1);
  for (size_t i = x.shape.size(); i-- > 1;) {
    in_strides[i - 1] = in_strides[i] * inferred_shape[i];
    out_strides[i - 1] = out_strides[i] * (*explicit_output_shape)[i];
  }
  Shape idx;
  idx.assign(x.shape.size(), 0);
  for (int64_t flat = 0; flat < inferred_total; ++flat) {
    int64_t rem = flat;
    for (size_t i = inferred_shape.size(); i-- > 0;) {
      idx[i] = rem % inferred_shape[i];
      rem /= inferred_shape[i];
    }
    int64_t out_offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
      out_offset += idx[i] * out_strides[i];
    }
    po[out_offset] = y_inferred[static_cast<size_t>(flat)];
  }

  return out;
}

} // namespace

Tensor MaxUnpool::operator()(const Tensor &x, const Tensor &indices, const Shape &kernel_shape,
                             const Shape &strides, const Shape &pads, RuntimeContext *rt) const {
  return RunMaxUnpool(x, indices, kernel_shape, strides, pads, /*explicit_output_shape=*/nullptr,
                      rt != nullptr ? rt->allocator() : nullptr);
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
  return RunMaxUnpool(x, indices, kernel_shape, strides, pads, &shape_vec,
                      rt != nullptr ? rt->allocator() : nullptr);
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
