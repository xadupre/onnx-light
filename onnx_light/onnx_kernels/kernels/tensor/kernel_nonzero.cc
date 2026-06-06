// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

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

Tensor NonZero::operator()(const Tensor &x) const {
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
  const std::vector<int64_t> &shape = x.shape;
  const std::size_t rank = shape.size();
  const int64_t total = x.element_count();

  // Identify non-zero positions in row-major order.
  std::vector<int64_t> nz_indices;
  nz_indices.reserve(static_cast<std::size_t>(total));
  for (int64_t i = 0; i < total; ++i) {
    if (IsElementNonZero(x.data.data() + static_cast<std::size_t>(i) * elem_size, elem_size)) {
      nz_indices.push_back(i);
    }
  }

  const int64_t nnz = static_cast<int64_t>(nz_indices.size());
  // Output shape: (rank, nnz). For scalar input (rank == 0), shape is (0, nnz),
  // mirroring the upstream NonZero spec (different from numpy.nonzero).
  const std::vector<int64_t> out_shape{static_cast<int64_t>(rank), nnz};

  // Build the row-major (rank, nnz) index matrix: row r lists the r-th
  // coordinate of every non-zero element, in row-major scan order of the input.
  std::vector<int64_t> values(static_cast<std::size_t>(rank) * static_cast<std::size_t>(nnz));
  for (std::size_t k = 0; k < nz_indices.size(); ++k) {
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

  return Tensor::FromInt64("", out_shape, values);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
