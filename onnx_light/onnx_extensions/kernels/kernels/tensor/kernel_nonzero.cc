// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstddef>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns true when the value at byte offset ``i * elem_size`` of ``bytes`` is
// not equal to zero, for an element of size ``elem_size``.
bool IsElementNonZero(const uint8_t *bytes, std::size_t elem_size) {
  for (std::size_t b = 0; b < elem_size; ++b) {
    if (bytes[b] != 0) {
      return true;
    }
  }
  return false;
}

} // namespace

Tensor NonZero::operator()(const Tensor &x, RuntimeContext *rt) const {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::INT16:
  case DataType::UINT16:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT64:
  case DataType::UINT64:
  case DataType::BOOL:
    break;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::NonZero: unsupported input dtype.");
  }

  const std::size_t elem_size = ElementSize(x.data_type);
  const onnx_kernels::Shape &shape = x.shape;
  const std::size_t rank = shape.size();
  const int64_t total = x.element_count();

  // Identify non-zero positions in row-major order. The scratch storage for the
  // indices is drawn from the runtime allocator when available (falling back to
  // inline storage otherwise), sized to the worst case of every element being
  // non-zero.
  detail::TemporaryTypedBuffer<int64_t> nz_indices_buf(
      static_cast<std::size_t>(total > 0 ? total : 1), ctx_.allocator, "kernel::NonZero");
  int64_t *nz_indices = nz_indices_buf.data();
  std::size_t nnz_count = 0;
  for (int64_t i = 0; i < total; ++i) {
    if (IsElementNonZero(x.bytes() + static_cast<std::size_t>(i) * elem_size, elem_size)) {
      nz_indices[nnz_count++] = i;
    }
  }

  const int64_t nnz = static_cast<int64_t>(nnz_count);
  // Output shape: (rank, nnz). For scalar input (rank == 0), shape is (0, nnz),
  // mirroring the upstream NonZero spec (different from numpy.nonzero).
  const onnx_kernels::Shape out_shape{static_cast<int64_t>(rank), nnz};

  // Build the row-major (rank, nnz) index matrix: row r lists the r-th
  // coordinate of every non-zero element, in row-major scan order of the input.
  // The output buffer is acquired from the runtime allocator via
  // ``MakeOutputTensor`` and written in place, avoiding a separate
  // std::vector allocated outside the allocator.
  const std::size_t out_count = static_cast<std::size_t>(rank) * static_cast<std::size_t>(nnz);
  Tensor result = MakeOutputTensor(static_cast<int32_t>(DataType::INT64), out_shape,
                                   out_count * sizeof(int64_t), ctx_.allocator);
  int64_t *values = result.AsInt64();
  for (std::size_t k = 0; k < nnz_count; ++k) {
    int64_t flat = nz_indices[k];
    for (std::size_t r = rank; r > 0; --r) {
      const int64_t dim = shape[r - 1];
      const int64_t coord = (dim > 0) ? (flat % dim) : 0;
      if (dim > 0) {
        flat /= dim;
      }
      values[(r - 1) * static_cast<std::size_t>(nnz) + k] = coord;
    }
  }

  return result;
}

void NonZero::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
