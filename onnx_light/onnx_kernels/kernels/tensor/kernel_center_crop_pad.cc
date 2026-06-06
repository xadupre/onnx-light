// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads a 1-D INT32 or INT64 ``shape`` tensor into a ``std::vector<int64_t>``.
std::vector<int64_t> ReadShapeTensor(const Tensor &t) {
  EXT_ENFORCE_INVALID(t.shape.size() == 1,
                      "kernel::CenterCropPad: 'shape' input must be a 1-D tensor.");
  const std::size_t n = static_cast<std::size_t>(t.shape[0]);
  std::vector<int64_t> out(n);
  if (t.data_type == DataType::INT64) {
    if (n > 0) {
      std::memcpy(out.data(), t.data.data(), n * sizeof(int64_t));
    }
  } else if (t.data_type == DataType::INT32) {
    const int32_t *p = reinterpret_cast<const int32_t *>(t.data.data());
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = static_cast<int64_t>(p[i]);
    }
  } else {
    throw std::invalid_argument("kernel::CenterCropPad: 'shape' input must be INT32 or INT64.");
  }
  return out;
}

// Normalizes ``axes`` against ``rank``; throws on out-of-range axes.
std::vector<int64_t> NormalizeAxes(const std::vector<int64_t> &axes, int64_t rank) {
  std::vector<int64_t> out;
  out.reserve(axes.size());
  for (int64_t a : axes) {
    int64_t na = a < 0 ? a + rank : a;
    EXT_ENFORCE_INVALID(na >= 0 && na < rank, "kernel::CenterCropPad: axis is out of range.");
    out.push_back(na);
  }
  return out;
}

} // namespace

Tensor CenterCropPad::operator()(const Tensor &input_data, const Tensor &shape,
                                 const CenterCropPad::Attributes &attrs) const {
  // Resolve the output shape from the input shape, ``shape`` and ``axes``.
  const std::size_t rank = input_data.shape.size();
  std::vector<int64_t> out_shape = input_data.shape;
  const std::vector<int64_t> shape_values = ReadShapeTensor(shape);

  std::vector<int64_t> axes;
  if (attrs.axes_present) {
    axes = NormalizeAxes(attrs.axes, static_cast<int64_t>(rank));
  } else {
    axes.resize(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
  }
  EXT_ENFORCE_INVALID(
      shape_values.size() == axes.size(),
      "kernel::CenterCropPad: number of elements of 'shape' must match the number of axes.");
  for (std::size_t i = 0; i < axes.size(); ++i) {
    out_shape[static_cast<std::size_t>(axes[i])] = shape_values[i];
  }

  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor output(
      "", input_data.data_type, out_shape,
      std::vector<uint8_t>(PackedByteSize(input_data.data_type, total), static_cast<uint8_t>(0)));
  (*this)(input_data, shape, attrs, output);
  return output;
}

void CenterCropPad::operator()(const Tensor &input_data, const Tensor &shape,
                               const CenterCropPad::Attributes &attrs, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == input_data.data_type,
                      "kernel::CenterCropPad: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(
      input_data.data_type != static_cast<int32_t>(DataType::STRING),
      "kernel::CenterCropPad: STRING tensors are not supported by the reference kernel.");

  const std::size_t rank = input_data.shape.size();
  EXT_ENFORCE_INVALID(rank > 0, "kernel::CenterCropPad: input must have rank >= 1.");
  EXT_ENFORCE_INVALID(output.shape.size() == rank,
                      "kernel::CenterCropPad: output rank must match input rank.");

  // Resolve axes and per-axis target ``shape``.
  const std::vector<int64_t> shape_values = ReadShapeTensor(shape);
  std::vector<int64_t> axes;
  if (attrs.axes_present) {
    axes = NormalizeAxes(attrs.axes, static_cast<int64_t>(rank));
  } else {
    axes.resize(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
  }
  EXT_ENFORCE_INVALID(
      shape_values.size() == axes.size(),
      "kernel::CenterCropPad: number of elements of 'shape' must match the number of axes.");

  // For each input axis, compute the cropping window over the input and the
  // padding window in the output. Defaults (no change) correspond to
  // ``in_start = 0``, ``out_start = 0`` and ``copy_len = dim``.
  std::vector<int64_t> in_start(rank, 0);
  std::vector<int64_t> out_start(rank, 0);
  std::vector<int64_t> copy_len(rank, 0);
  std::vector<bool> axis_selected(rank, false);
  for (std::size_t i = 0; i < rank; ++i) {
    copy_len[i] = input_data.shape[i];
  }
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t a = static_cast<std::size_t>(axes[i]);
    const int64_t in_dim = input_data.shape[a];
    const int64_t sh = shape_values[i];
    EXT_ENFORCE_INVALID(sh >= 0, "kernel::CenterCropPad: 'shape' values must be non-negative.");
    EXT_ENFORCE_INVALID(output.shape[a] == sh,
                        "kernel::CenterCropPad: preallocated output shape mismatch on axis.");
    axis_selected[a] = true;
    if (sh == in_dim) {
      copy_len[a] = in_dim;
    } else if (sh < in_dim) {
      // Center crop: start at floor((in_dim - sh) / 2).
      const int64_t d = in_dim - sh;
      in_start[a] = d / 2;
      out_start[a] = 0;
      copy_len[a] = sh;
    } else {
      // Center pad: start at floor((sh - in_dim) / 2) in the output.
      const int64_t d = sh - in_dim;
      in_start[a] = 0;
      out_start[a] = d / 2;
      copy_len[a] = in_dim;
    }
  }
  for (std::size_t i = 0; i < rank; ++i) {
    if (!axis_selected[i]) {
      EXT_ENFORCE_INVALID(
          output.shape[i] == input_data.shape[i],
          "kernel::CenterCropPad: output shape must match input shape on unspecified axes.");
    }
  }

  const std::size_t elem_size = ElementSize(input_data.data_type);

  // Compute row-major strides for input and output (in elements).
  std::vector<int64_t> in_strides(rank, 1);
  std::vector<int64_t> out_strides(rank, 1);
  for (std::size_t i = rank - 1; i > 0; --i) {
    in_strides[i - 1] = in_strides[i] * input_data.shape[i];
    out_strides[i - 1] = out_strides[i] * output.shape[i];
  }

  int64_t total_out = 1;
  for (int64_t d : output.shape) {
    total_out *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<std::size_t>(total_out) * elem_size,
                      "kernel::CenterCropPad: preallocated output buffer size mismatch.");

  // Zero-fill the output for the padding regions.
  std::memset(output.data.data(), 0, output.data.size());

  // Bail out early when any copy_len is zero (no elements to copy).
  for (int64_t cl : copy_len) {
    if (cl == 0) {
      return;
    }
  }

  // Iterate over the copy window (n-D loop). The inner-most axis copy is
  // collapsed into a single ``memcpy`` for efficiency.
  std::vector<int64_t> idx(rank, 0);
  const int64_t inner_len = copy_len.back() * static_cast<int64_t>(elem_size);
  const int64_t inner_in_off_base = in_start.back();
  const int64_t inner_out_off_base = out_start.back();
  const uint8_t *const in_ptr = input_data.data.data();
  uint8_t *const out_ptr = output.data.data();

  while (true) {
    int64_t in_off = inner_in_off_base * in_strides.back();
    int64_t out_off = inner_out_off_base * out_strides.back();
    for (std::size_t i = 0; i + 1 < rank; ++i) {
      in_off += (idx[i] + in_start[i]) * in_strides[i];
      out_off += (idx[i] + out_start[i]) * out_strides[i];
    }
    std::memcpy(out_ptr + static_cast<std::size_t>(out_off) * elem_size,
                in_ptr + static_cast<std::size_t>(in_off) * elem_size,
                static_cast<std::size_t>(inner_len));

    // Increment the outer indices (everything except the inner-most axis).
    if (rank == 1) {
      break;
    }
    std::size_t axis = rank - 2;
    while (true) {
      idx[axis] += 1;
      if (idx[axis] < copy_len[axis]) {
        break;
      }
      idx[axis] = 0;
      if (axis == 0) {
        return;
      }
      --axis;
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
