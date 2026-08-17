// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ComputeGatherNDOutputShape(const onnx_kernels::Shape &data_shape,
                                               const onnx_kernels::Shape &idx_shape,
                                               int64_t batch_dims) {
  const int64_t r = static_cast<int64_t>(data_shape.size());
  const int64_t q = static_cast<int64_t>(idx_shape.size());
  EXT_ENFORCE_INVALID(r >= 1 && q >= 1,
                      "kernel::GatherND: 'data' and 'indices' must have rank >= 1.");
  EXT_ENFORCE_INVALID(batch_dims >= 0 && batch_dims < r && batch_dims < q,
                      "kernel::GatherND: 'batch_dims' out of range.");
  for (int64_t k = 0; k < batch_dims; ++k) {
    EXT_ENFORCE_INVALID(data_shape[static_cast<std::size_t>(k)] ==
                            idx_shape[static_cast<std::size_t>(k)],
                        "kernel::GatherND: leading 'batch_dims' of data and indices must match.");
  }
  const int64_t k_last = idx_shape[static_cast<std::size_t>(q - 1)];
  EXT_ENFORCE_INVALID(k_last >= 1 && batch_dims + k_last <= r,
                      "kernel::GatherND: indices last dim out of range.");
  onnx_kernels::Shape out_shape;
  out_shape.reserve(static_cast<std::size_t>(q - 1 + (r - batch_dims - k_last)));
  for (int64_t k = 0; k < q - 1; ++k) {
    out_shape.push_back(idx_shape[static_cast<std::size_t>(k)]);
  }
  for (int64_t k = batch_dims + k_last; k < r; ++k) {
    out_shape.push_back(data_shape[static_cast<std::size_t>(k)]);
  }
  return out_shape;
}

} // namespace

Tensor GatherND::operator()(const Tensor &data, const Tensor &indices, int64_t batch_dims,
                            RuntimeContext *rt) const {
  onnx_kernels::Shape out_shape = ComputeGatherNDOutputShape(data.shape, indices.shape, batch_dims);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const std::size_t elem_size = ElementSize(data.data_type);
  const size_t out_n_bytes = static_cast<std::size_t>(total) * elem_size;
  Tensor out = (rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, out_n_bytes)
                   : MakeOutputTensor(data.data_type, out_shape, out_n_bytes, nullptr));
  (*this)(data, indices, batch_dims, out);
  return out;
}

void GatherND::operator()(const Tensor &data, const Tensor &indices, int64_t batch_dims,
                          Tensor &output) const {
  onnx_kernels::Shape out_shape = ComputeGatherNDOutputShape(data.shape, indices.shape, batch_dims);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::GatherND: output dtype must match data dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape, "kernel::GatherND: output shape mismatch.");
  EXT_ENFORCE_INVALID(indices.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::GatherND: 'indices' input must be INT64.");

  const int64_t *idx_values = reinterpret_cast<const int64_t *>(indices.bytes());
  const int64_t r = static_cast<int64_t>(data.shape.size());
  const int64_t q = static_cast<int64_t>(indices.shape.size());
  const int64_t k_last = indices.shape[static_cast<std::size_t>(q - 1)];
  const std::size_t elem_size = ElementSize(data.data_type);

  // Row-major strides for data (in elements).
  onnx_kernels::Shape data_strides;
  data_strides.assign(static_cast<std::size_t>(r), 1);
  for (int64_t k = r - 2; k >= 0; --k) {
    data_strides[static_cast<std::size_t>(k)] =
        data_strides[static_cast<std::size_t>(k + 1)] * data.shape[static_cast<std::size_t>(k + 1)];
  }

  // Compute number of index tuples (product of indices.shape[:-1]).
  int64_t num_tuples = 1;
  for (int64_t k = 0; k < q - 1; ++k) {
    num_tuples *= indices.shape[static_cast<std::size_t>(k)];
  }

  // Number of contiguous data elements per index tuple (the trailing dims of
  // data beyond batch_dims + k_last).
  int64_t slice_elems = 1;
  for (int64_t k = batch_dims + k_last; k < r; ++k) {
    slice_elems *= data.shape[static_cast<std::size_t>(k)];
  }
  const int64_t slice_bytes = slice_elems * static_cast<int64_t>(elem_size);

  // Strides through the leading 'q-1' dims of indices to recover batch coords
  // when batch_dims > 0.
  onnx_kernels::Shape idx_outer_strides;
  idx_outer_strides.assign(static_cast<std::size_t>(q - 1), 1);
  for (int64_t k = q - 3; k >= 0; --k) {
    idx_outer_strides[static_cast<std::size_t>(k)] =
        idx_outer_strides[static_cast<std::size_t>(k + 1)] *
        indices.shape[static_cast<std::size_t>(k + 1)];
  }

  for (int64_t t = 0; t < num_tuples; ++t) {
    // Decode coord of this tuple within indices' leading dims.
    int64_t remaining = t;
    int64_t data_offset = 0;
    for (int64_t k = 0; k < batch_dims; ++k) {
      const int64_t c = remaining / idx_outer_strides[static_cast<std::size_t>(k)];
      remaining %= idx_outer_strides[static_cast<std::size_t>(k)];
      data_offset += c * data_strides[static_cast<std::size_t>(k)];
    }
    // Read the k_last-component index tuple from indices.
    for (int64_t k = 0; k < k_last; ++k) {
      int64_t idx = idx_values[static_cast<std::size_t>(t * k_last + k)];
      const int64_t dim = data.shape[static_cast<std::size_t>(batch_dims + k)];
      if (idx < 0) {
        idx += dim;
      }
      EXT_ENFORCE_INVALID(idx >= 0 && idx < dim, "kernel::GatherND: index out of range.");
      data_offset += idx * data_strides[static_cast<std::size_t>(batch_dims + k)];
    }
    std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(t * slice_bytes),
                data.bytes() +
                    static_cast<std::size_t>(data_offset) * static_cast<std::size_t>(elem_size),
                static_cast<std::size_t>(slice_bytes));
  }
}

void GatherND::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const Tensor &indices = GetInput(node, 1, rt.tensors());
  const int64_t batch_dims = GetAttributeIntOrDefault(node, "batch_dims", 0);
  onnx_kernels::kernel::GatherND k(rt.kernel_ctx());
  SetOutput(node, 0, k(data, indices, batch_dims, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
