// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Reads a 1-D INT64 tensor into a ``std::vector<int64_t>``.
std::vector<int64_t> ReadInt64Vector(const Tensor &t, const std::string &name) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::INT64,
                      "kernel::Pad: '" + name + "' input must be INT64.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1,
                      "kernel::Pad: '" + name + "' input must be a 1-D tensor.");
  const std::size_t n = static_cast<std::size_t>(t.shape[0]);
  std::vector<int64_t> out(n);
  if (n > 0) {
    std::memcpy(out.data(), t.data.data(), n * sizeof(int64_t));
  }
  return out;
}

// Resolves ``axes`` (possibly negative) and falls back to ``[0, 1, ..., rank-1]``
// when ``axes_tensor`` is ``nullptr``.
std::vector<int64_t> ResolveAxes(const Tensor *axes_tensor, std::size_t rank) {
  std::vector<int64_t> axes;
  if (axes_tensor == nullptr) {
    axes.resize(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
    return axes;
  }
  EXT_ENFORCE_INVALID(axes_tensor->shape.size() == 1,
                      "kernel::Pad: 'axes' input must be a 1-D tensor.");
  const std::size_t n = static_cast<std::size_t>(axes_tensor->shape[0]);
  axes.resize(n);
  if (axes_tensor->data_type == DataType::INT64) {
    if (n > 0) {
      std::memcpy(axes.data(), axes_tensor->data.data(), n * sizeof(int64_t));
    }
  } else if (axes_tensor->data_type == DataType::INT32) {
    const int32_t *p = reinterpret_cast<const int32_t *>(axes_tensor->data.data());
    for (std::size_t i = 0; i < n; ++i) {
      axes[i] = static_cast<int64_t>(p[i]);
    }
  } else {
    throw std::invalid_argument("kernel::Pad: 'axes' input must be INT32 or INT64.");
  }
  const int64_t r = static_cast<int64_t>(rank);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    int64_t a = axes[i];
    if (a < 0) {
      a += r;
    }
    EXT_ENFORCE_INVALID(a >= 0 && a < r, "kernel::Pad: axis is out of range.");
    axes[i] = a;
  }
  return axes;
}

// Reads the ``constant_value`` scalar tensor and returns its bytes as a
// per-element pattern of size ``elem_size``. When ``cv`` is ``nullptr`` (or an
// empty optional tensor), returns ``elem_size`` zero bytes.
std::vector<uint8_t> ResolveConstantBytes(const Tensor *cv, int32_t data_type,
                                          std::size_t elem_size) {
  if (cv == nullptr || cv->element_count() == 0) {
    return std::vector<uint8_t>(elem_size, 0);
  }
  EXT_ENFORCE_INVALID(cv->data_type == data_type,
                      "kernel::Pad: 'constant_value' dtype must match 'data' dtype.");
  EXT_ENFORCE_INVALID(cv->element_count() == 1,
                      "kernel::Pad: 'constant_value' must be a scalar tensor.");
  EXT_ENFORCE_INVALID(cv->data.size() == elem_size,
                      "kernel::Pad: 'constant_value' has unexpected byte size.");
  return std::vector<uint8_t>(cv->data.begin(), cv->data.end());
}

// Pre-computed row-major strides (in elements).
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  const std::size_t rank = shape.size();
  std::vector<int64_t> strides(rank, 0);
  if (rank > 0) {
    strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      strides[k - 1] = strides[k] * shape[k];
    }
  }
  return strides;
}

// Maps an output coordinate on a padded axis to the corresponding input
// coordinate. ``input_dim`` is the input dimension on that axis. Returns
// ``-1`` when the output position falls inside the padded ``constant`` region.
int64_t MapCoord(int64_t out_coord, int64_t pad_begin, int64_t input_dim, const std::string &mode) {
  const int64_t inside = out_coord - pad_begin;
  if (mode == "constant") {
    if (inside < 0 || inside >= input_dim) {
      return -1;
    }
    return inside;
  }
  if (mode == "wrap") {
    EXT_ENFORCE_INVALID(input_dim > 0,
                        "kernel::Pad: 'wrap' mode requires positive input dimension.");
    int64_t m = inside % input_dim;
    if (m < 0) {
      m += input_dim;
    }
    return m;
  }
  if (mode == "edge") {
    if (inside < 0) {
      return 0;
    }
    if (inside >= input_dim) {
      return input_dim - 1;
    }
    return inside;
  }
  if (mode == "reflect") {
    EXT_ENFORCE_INVALID(input_dim > 0,
                        "kernel::Pad: 'reflect' mode requires positive input dimension.");
    if (input_dim == 1) {
      return 0;
    }
    const int64_t period = 2 * (input_dim - 1);
    int64_t m = inside % period;
    if (m < 0) {
      m += period;
    }
    if (m >= input_dim) {
      m = period - m;
    }
    return m;
  }
  throw std::invalid_argument("kernel::Pad: unsupported mode '" + mode + "'.");
}

} // namespace

Tensor Pad::operator()(const Tensor &data, const Tensor &pads, const Tensor *constant_value,
                       const Tensor *axes, const std::string &mode) const {
  const std::size_t rank = data.shape.size();
  const std::vector<int64_t> axes_vec = ResolveAxes(axes, rank);
  const std::vector<int64_t> pads_vec = ReadInt64Vector(pads, "pads");
  const std::size_t num_axes = axes_vec.size();
  EXT_ENFORCE_INVALID(pads_vec.size() == 2 * num_axes,
                      "kernel::Pad: 'pads' must have length 2 * num_axes.");
  for (int64_t p : pads_vec) {
    EXT_ENFORCE_INVALID(
        p >= 0, "kernel::Pad: negative padding (cropping) is not supported by this kernel.");
  }

  // Per-axis pad_begin/pad_end (indexed by data axis).
  std::vector<int64_t> pad_begin(rank, 0);
  std::vector<int64_t> pad_end(rank, 0);
  for (std::size_t i = 0; i < num_axes; ++i) {
    const std::size_t axis = static_cast<std::size_t>(axes_vec[i]);
    pad_begin[axis] = pads_vec[i];
    pad_end[axis] = pads_vec[i + num_axes];
  }
  std::vector<int64_t> out_shape(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    out_shape[i] = data.shape[i] + pad_begin[i] + pad_end[i];
  }

  const std::size_t elem_size = ElementSize(data.data_type);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor out("", data.data_type, out_shape,
             std::vector<uint8_t>(static_cast<std::size_t>(total) * elem_size));
  (*this)(data, pads, constant_value, axes, mode, out);
  return out;
}

void Pad::operator()(const Tensor &data, const Tensor &pads, const Tensor *constant_value,
                     const Tensor *axes, const std::string &mode, Tensor &output) const {
  const std::size_t rank = data.shape.size();
  const std::vector<int64_t> axes_vec = ResolveAxes(axes, rank);
  const std::vector<int64_t> pads_vec = ReadInt64Vector(pads, "pads");
  const std::size_t num_axes = axes_vec.size();
  EXT_ENFORCE_INVALID(pads_vec.size() == 2 * num_axes,
                      "kernel::Pad: 'pads' must have length 2 * num_axes.");
  for (int64_t p : pads_vec) {
    EXT_ENFORCE_INVALID(
        p >= 0, "kernel::Pad: negative padding (cropping) is not supported by this kernel.");
  }

  std::vector<int64_t> pad_begin(rank, 0);
  std::vector<int64_t> pad_end(rank, 0);
  for (std::size_t i = 0; i < num_axes; ++i) {
    const std::size_t axis = static_cast<std::size_t>(axes_vec[i]);
    pad_begin[axis] = pads_vec[i];
    pad_end[axis] = pads_vec[i + num_axes];
  }
  std::vector<int64_t> expected_shape(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    expected_shape[i] = data.shape[i] + pad_begin[i] + pad_end[i];
  }
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Pad: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::Pad: preallocated output shape must match padded shape.");

  const std::size_t elem_size = ElementSize(data.data_type);
  EXT_ENFORCE_INVALID(elem_size > 0, "kernel::Pad: data dtype is not supported by this kernel.");
  const std::vector<uint8_t> constant_bytes =
      ResolveConstantBytes(constant_value, data.data_type, elem_size);

  const std::vector<int64_t> in_strides = RowMajorStrides(data.shape);
  const std::vector<int64_t> out_strides = RowMajorStrides(output.shape);

  int64_t total = 1;
  for (int64_t d : output.shape) {
    total *= d;
  }

  std::vector<int64_t> out_coord(rank, 0);
  for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < rank; ++k) {
      out_coord[k] = remaining / out_strides[k];
      remaining -= out_coord[k] * out_strides[k];
    }
    bool is_pad = false;
    int64_t in_idx = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t mapped = MapCoord(out_coord[k], pad_begin[k], data.shape[k], mode);
      if (mapped < 0) {
        is_pad = true;
        break;
      }
      in_idx += mapped * in_strides[k];
    }
    uint8_t *const dst = output.data.data() + static_cast<std::size_t>(out_idx) * elem_size;
    if (is_pad) {
      std::memcpy(dst, constant_bytes.data(), elem_size);
    } else {
      std::memcpy(dst, data.data.data() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
