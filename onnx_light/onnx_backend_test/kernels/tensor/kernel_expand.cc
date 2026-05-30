// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads the target shape from the 1-D INT64 ``shape`` input tensor.
std::vector<int64_t> ReadExpandShapeInput(const Tensor &shape) {
  EXT_ENFORCE_INVALID(shape.data_type == static_cast<int32_t>(TensorProto::DataType::INT64),
                      "kernel::Expand: 'shape' input must be INT64.");
  EXT_ENFORCE_INVALID(shape.shape.size() <= 1,
                      "kernel::Expand: 'shape' input must be a 1-D tensor.");
  if (shape.shape.empty()) {
    return {};
  }
  const int64_t n = shape.shape[0];
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), shape.data.data(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  return out;
}

// Computes the broadcast output shape from ``in_shape`` and ``target``.
// Follows numpy-style right-alignment: missing leading dims are treated as 1.
std::vector<int64_t> BroadcastOutputShape(const std::vector<int64_t> &in_shape,
                                          const std::vector<int64_t> &target) {
  const std::size_t ri = in_shape.size();
  const std::size_t rt = target.size();
  const std::size_t r = std::max(ri, rt);
  std::vector<int64_t> out(r);
  for (std::size_t k = 0; k < r; ++k) {
    const bool has_i = k + ri >= r;
    const bool has_t = k + rt >= r;
    const int64_t di = has_i ? in_shape[k - (r - ri)] : int64_t{1};
    const int64_t dt = has_t ? target[k - (r - rt)] : int64_t{1};
    EXT_ENFORCE_INVALID(di == dt || di == 1 || dt == 1,
                        "kernel::Expand: incompatible dimensions " + std::to_string(di) + " and " +
                            std::to_string(dt) + " at axis " + std::to_string(k) + ".");
    out[k] = std::max(di, dt);
  }
  return out;
}

struct ExpandLayout {
  std::vector<int64_t> out_shape;
  std::vector<int64_t> in_shape_aligned; // right-aligned, prepended with 1s
  std::vector<int64_t> out_strides;
  std::vector<int64_t> in_strides;
  size_t elem_size;
  int64_t total_elements;
};

ExpandLayout ComputeExpandLayout(const Tensor &input, const std::vector<int64_t> &target) {
  ExpandLayout layout;
  layout.out_shape = BroadcastOutputShape(input.shape, target);
  const int64_t out_rank = static_cast<int64_t>(layout.out_shape.size());
  const int64_t in_rank = static_cast<int64_t>(input.shape.size());

  // Right-align input shape by prepending 1s.
  layout.in_shape_aligned.assign(static_cast<std::size_t>(out_rank), int64_t{1});
  for (int64_t k = 0; k < in_rank; ++k) {
    layout.in_shape_aligned[static_cast<std::size_t>(out_rank - in_rank + k)] =
        input.shape[static_cast<std::size_t>(k)];
  }

  // Compute output strides (row-major).
  layout.out_strides.resize(static_cast<std::size_t>(out_rank));
  layout.out_strides[static_cast<std::size_t>(out_rank - 1)] = 1;
  for (int64_t k = out_rank - 2; k >= 0; --k) {
    layout.out_strides[static_cast<std::size_t>(k)] =
        layout.out_strides[static_cast<std::size_t>(k + 1)] *
        layout.out_shape[static_cast<std::size_t>(k + 1)];
  }

  // Compute input strides (row-major, based on aligned input shape).
  layout.in_strides.resize(static_cast<std::size_t>(out_rank));
  layout.in_strides[static_cast<std::size_t>(out_rank - 1)] = 1;
  for (int64_t k = out_rank - 2; k >= 0; --k) {
    layout.in_strides[static_cast<std::size_t>(k)] =
        layout.in_strides[static_cast<std::size_t>(k + 1)] *
        layout.in_shape_aligned[static_cast<std::size_t>(k + 1)];
  }

  layout.elem_size = ElementSize(input.data_type);
  layout.total_elements = 1;
  for (int64_t d : layout.out_shape) {
    layout.total_elements *= d;
  }
  return layout;
}

} // namespace

Tensor Expand::operator()(const Tensor &input, const Tensor &shape) const {
  const std::vector<int64_t> target = ReadExpandShapeInput(shape);
  const ExpandLayout layout = ComputeExpandLayout(input, target);
  Tensor out(
      "", input.data_type, layout.out_shape,
      std::vector<uint8_t>(static_cast<std::size_t>(layout.total_elements) * layout.elem_size));
  (*this)(input, shape, out);
  return out;
}

void Expand::operator()(const Tensor &input, const Tensor &shape, Tensor &output) const {
  const std::vector<int64_t> target = ReadExpandShapeInput(shape);
  const ExpandLayout layout = ComputeExpandLayout(input, target);

  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Expand: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == layout.out_shape,
                      "kernel::Expand: preallocated output shape must match broadcast shape.");

  const std::size_t out_rank = layout.out_shape.size();

  // For each output element, determine the corresponding input element
  // index using the broadcast mapping (dimension with size 1 maps to 0).
  for (int64_t out_idx = 0; out_idx < layout.total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < out_rank; ++k) {
      const int64_t out_coord = remaining / layout.out_strides[k];
      remaining %= layout.out_strides[k];
      if (layout.in_shape_aligned[k] > 1) {
        in_idx += out_coord * layout.in_strides[k];
      }
      // If in_shape_aligned[k] == 1, the input index for this axis is 0,
      // contributing nothing to in_idx.
    }
    std::memcpy(output.data.data() + static_cast<std::size_t>(out_idx) * layout.elem_size,
                input.data.data() + static_cast<std::size_t>(in_idx) * layout.elem_size,
                layout.elem_size);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
