// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Reads a 1-D INT64 tensor into a ``Shape``.
onnx_kernels::Shape ReadInt64Vector(const Tensor &t, const std::string &name) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::INT64, "kernel::Pad: '", name,
                      "' input must be INT64.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1, "kernel::Pad: '", name, "' input must be a 1-D tensor.");
  const std::size_t n = static_cast<std::size_t>(t.shape[0]);
  onnx_kernels::Shape out;
  out.assign(n, 0);
  if (n > 0) {
    std::memcpy(out.begin(), t.bytes(), n * sizeof(int64_t));
  }
  return out;
}

// Resolves ``axes`` (possibly negative) and falls back to ``[0, 1, ..., rank-1]``
// when ``axes_tensor`` is ``nullptr``.
onnx_kernels::Shape ResolveAxes(const Tensor *axes_tensor, std::size_t rank) {
  onnx_kernels::Shape axes;
  if (axes_tensor == nullptr) {
    axes.assign(rank, 0);
    for (std::size_t i = 0; i < rank; ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
    return axes;
  }
  EXT_ENFORCE_INVALID(axes_tensor->shape.size() == 1,
                      "kernel::Pad: 'axes' input must be a 1-D tensor.");
  const std::size_t n = static_cast<std::size_t>(axes_tensor->shape[0]);
  axes.assign(n, 0);
  if (axes_tensor->data_type == DataType::INT64) {
    if (n > 0) {
      std::memcpy(axes.begin(), axes_tensor->bytes(), n * sizeof(int64_t));
    }
  } else if (axes_tensor->data_type == DataType::INT32) {
    const int32_t *p = reinterpret_cast<const int32_t *>(axes_tensor->bytes());
    for (std::size_t i = 0; i < n; ++i) {
      axes[i] = static_cast<int64_t>(p[i]);
    }
  } else {
    EXT_THROW_INVALID("kernel::Pad: 'axes' input must be INT32 or INT64.");
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

// Reads the ``constant_value`` scalar tensor and writes its bytes as a
// per-element pattern of size ``elem_size`` into ``out``. When ``cv`` is
// ``nullptr`` (or an empty optional tensor), writes ``elem_size`` zero bytes.
void ResolveConstantBytes(const Tensor *cv, int32_t data_type, std::size_t elem_size,
                          uint8_t *out) {
  if (cv == nullptr || cv->element_count() == 0) {
    std::memset(out, 0, elem_size);
    return;
  }
  EXT_ENFORCE_INVALID(cv->data_type == data_type,
                      "kernel::Pad: 'constant_value' dtype must match 'data' dtype.");
  EXT_ENFORCE_INVALID(cv->element_count() == 1,
                      "kernel::Pad: 'constant_value' must be a scalar tensor.");
  EXT_ENFORCE_INVALID(cv->size_bytes() == elem_size,
                      "kernel::Pad: 'constant_value' has unexpected byte size.");
  std::memcpy(out, cv->bytes(), elem_size);
}

// Pre-computed row-major strides (in elements).
onnx_kernels::Shape RowMajorStrides(const onnx_kernels::Shape &shape) {
  const std::size_t rank = shape.size();
  onnx_kernels::Shape strides;
  strides.assign(rank, 0);
  if (rank > 0) {
    strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      strides[k - 1] = strides[k] * shape[k];
    }
  }
  return strides;
}

int64_t CheckedAdd(int64_t left, int64_t right) {
  EXT_ENFORCE_INVALID(
      (right <= 0 || left <= std::numeric_limits<int64_t>::max() - right) &&
          (right >= 0 || left >= std::numeric_limits<int64_t>::min() - right),
      "kernel::Pad: padding results in an output dimension outside the int64 range.");
  return left + right;
}

int64_t ComputeOutputDim(int64_t input_dim, int64_t pad_begin, int64_t pad_end) {
  const int64_t output_dim = CheckedAdd(CheckedAdd(input_dim, pad_begin), pad_end);
  EXT_ENFORCE_INVALID(output_dim >= 0,
                      "kernel::Pad: padding results in a negative output dimension.");
  return output_dim;
}

// Maps an output coordinate on a padded axis to the corresponding input
// coordinate after cropping. Returns ``-1`` when the output position falls
// inside the padded ``constant`` region.
int64_t MapCoord(int64_t out_coord, int64_t positive_pad_begin, int64_t crop_begin,
                 int64_t cropped_dim, const std::string &mode) {
  const int64_t inside = out_coord - positive_pad_begin;
  if (mode == "constant") {
    if (inside < 0 || inside >= cropped_dim) {
      return -1;
    }
    return crop_begin + inside;
  }
  if (mode == "wrap") {
    EXT_ENFORCE_INVALID(cropped_dim > 0,
                        "kernel::Pad: 'wrap' mode requires positive input dimension.");
    int64_t m = inside % cropped_dim;
    if (m < 0) {
      m += cropped_dim;
    }
    return crop_begin + m;
  }
  if (mode == "edge") {
    if (inside < 0) {
      return crop_begin;
    }
    if (inside >= cropped_dim) {
      return crop_begin + cropped_dim - 1;
    }
    return crop_begin + inside;
  }
  if (mode == "reflect") {
    EXT_ENFORCE_INVALID(cropped_dim > 0,
                        "kernel::Pad: 'reflect' mode requires positive input dimension.");
    if (cropped_dim == 1) {
      return crop_begin;
    }
    const int64_t period = 2 * (cropped_dim - 1);
    int64_t m = inside % period;
    if (m < 0) {
      m += period;
    }
    if (m >= cropped_dim) {
      m = period - m;
    }
    return crop_begin + m;
  }
  EXT_THROW_INVALID("kernel::Pad: unsupported mode '", mode, "'.");
}

// Validates ``output`` and fills it with the padded result. ``allocator`` (when
// non-null) supplies scratch storage for the per-element constant pattern;
// otherwise the buffer falls back to inline storage.
void PadInto(const Tensor &data, const Tensor &pads, const Tensor *constant_value,
             const Tensor *axes, const std::string &mode, Tensor &output,
             RawBufferAllocator *allocator) {
  const std::size_t rank = data.shape.size();
  const onnx_kernels::Shape axes_vec = ResolveAxes(axes, rank);
  const onnx_kernels::Shape pads_vec = ReadInt64Vector(pads, "pads");
  const std::size_t num_axes = axes_vec.size();
  EXT_ENFORCE_INVALID(pads_vec.size() == 2 * num_axes,
                      "kernel::Pad: 'pads' must have length 2 * num_axes.");

  onnx_kernels::Shape pad_begin;
  pad_begin.assign(rank, 0);
  onnx_kernels::Shape pad_end;
  pad_end.assign(rank, 0);
  for (std::size_t i = 0; i < num_axes; ++i) {
    const std::size_t axis = static_cast<std::size_t>(axes_vec[i]);
    pad_begin[axis] = pads_vec[i];
    pad_end[axis] = pads_vec[i + num_axes];
  }
  onnx_kernels::Shape expected_shape;
  expected_shape.assign(rank, 0);
  onnx_kernels::Shape crop_begin;
  crop_begin.assign(rank, 0);
  onnx_kernels::Shape cropped_shape;
  cropped_shape.assign(rank, 0);
  for (std::size_t i = 0; i < rank; ++i) {
    expected_shape[i] = ComputeOutputDim(data.shape[i], pad_begin[i], pad_end[i]);
    EXT_ENFORCE_INVALID(pad_begin[i] != std::numeric_limits<int64_t>::min() &&
                            pad_end[i] != std::numeric_limits<int64_t>::min(),
                        "kernel::Pad: padding values must be greater than INT64_MIN.");
    crop_begin[i] = std::max(-pad_begin[i], int64_t{0});
    const int64_t crop_end = std::max(-pad_end[i], int64_t{0});
    const bool overcropped =
        crop_begin[i] > data.shape[i] || crop_end > data.shape[i] - crop_begin[i];
    EXT_ENFORCE_INVALID(!overcropped || mode == "constant", "kernel::Pad: mode '", mode,
                        "' cannot pad an overcropped input.");
    cropped_shape[i] = overcropped ? 0 : data.shape[i] - crop_begin[i] - crop_end;
    const int64_t positive_begin = std::max(pad_begin[i], int64_t{0});
    const int64_t positive_end = std::max(pad_end[i], int64_t{0});
    EXT_ENFORCE_INVALID(mode == "constant" || cropped_shape[i] > 0 || expected_shape[i] == 0,
                        "kernel::Pad: mode '", mode, "' cannot pad an empty cropped input.");
    EXT_ENFORCE_INVALID(
        mode != "reflect" || (positive_begin == 0 && positive_end == 0) ||
            (cropped_shape[i] >= 2 && positive_begin < cropped_shape[i] &&
             positive_end < cropped_shape[i]),
        "kernel::Pad: reflect padding cannot exceed the cropped axis length minus 1.");
  }
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Pad: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::Pad: preallocated output shape must match padded shape.");

  const std::size_t elem_size = ElementSize(data.data_type);
  EXT_ENFORCE_INVALID(elem_size > 0, "kernel::Pad: data dtype is not supported by this kernel.");
  detail::TemporaryTypedBuffer<uint8_t> constant_buf(elem_size, allocator,
                                                     "kernel::Pad constant_value");
  ResolveConstantBytes(constant_value, data.data_type, elem_size, constant_buf.data());
  const uint8_t *const constant_bytes = constant_buf.data();

  const onnx_kernels::Shape in_strides = RowMajorStrides(data.shape);
  const onnx_kernels::Shape out_strides = RowMajorStrides(output.shape);

  int64_t total = 1;
  for (int64_t d : output.shape) {
    total *= d;
  }

  onnx_kernels::Shape out_coord;
  out_coord.assign(rank, 0);
  for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < rank; ++k) {
      out_coord[k] = remaining / out_strides[k];
      remaining -= out_coord[k] * out_strides[k];
    }
    bool is_pad = false;
    int64_t in_idx = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t mapped = MapCoord(out_coord[k], std::max(pad_begin[k], int64_t{0}),
                                      crop_begin[k], cropped_shape[k], mode);
      if (mapped < 0) {
        is_pad = true;
        break;
      }
      in_idx += mapped * in_strides[k];
    }
    uint8_t *const dst = output.mutable_bytes() + static_cast<std::size_t>(out_idx) * elem_size;
    if (is_pad) {
      std::memcpy(dst, constant_bytes, elem_size);
    } else {
      std::memcpy(dst, data.bytes() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
    }
  }
}

} // namespace

Tensor Pad::operator()(const Tensor &data, const Tensor &pads, const Tensor *constant_value,
                       const Tensor *axes, const std::string &mode, RuntimeContext *rt) const {
  const std::size_t rank = data.shape.size();
  const onnx_kernels::Shape axes_vec = ResolveAxes(axes, rank);
  const onnx_kernels::Shape pads_vec = ReadInt64Vector(pads, "pads");
  const std::size_t num_axes = axes_vec.size();
  EXT_ENFORCE_INVALID(pads_vec.size() == 2 * num_axes,
                      "kernel::Pad: 'pads' must have length 2 * num_axes.");

  // Per-axis pad_begin/pad_end (indexed by data axis).
  onnx_kernels::Shape pad_begin;
  pad_begin.assign(rank, 0);
  onnx_kernels::Shape pad_end;
  pad_end.assign(rank, 0);
  for (std::size_t i = 0; i < num_axes; ++i) {
    const std::size_t axis = static_cast<std::size_t>(axes_vec[i]);
    pad_begin[axis] = pads_vec[i];
    pad_end[axis] = pads_vec[i + num_axes];
  }
  onnx_kernels::Shape out_shape;
  out_shape.assign(rank, 0);
  for (std::size_t i = 0; i < rank; ++i) {
    out_shape[i] = ComputeOutputDim(data.shape[i], pad_begin[i], pad_end[i]);
  }

  const std::size_t elem_size = ElementSize(data.data_type);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const size_t out_n_bytes = static_cast<std::size_t>(total) * elem_size;
  Tensor out = (rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, out_n_bytes)
                   : MakeOutputTensor(data.data_type, out_shape, out_n_bytes, nullptr));
  PadInto(data, pads, constant_value, axes, mode, out, rt ? rt->execution_allocator() : nullptr);
  return out;
}

void Pad::operator()(const Tensor &data, const Tensor &pads, const Tensor *constant_value,
                     const Tensor *axes, const std::string &mode, Tensor &output) const {
  PadInto(data, pads, constant_value, axes, mode, output, nullptr);
}

void Pad::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 4), "RunNode: op 'Pad' expects at most 4 inputs.");
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const std::string mode = GetAttributeStringOrDefault(node, "mode", "constant");
  onnx_kernels::kernel::Pad k(rt.kernel_ctx());

  // Opset 11+: ``pads`` is the second input.
  if (node.input_size() >= 2) {
    const Tensor &pads = GetInput(node, 1, rt.tensors());
    const Tensor *constant_value = GetOptionalInput(node, 2, rt.tensors());
    const Tensor *axes = GetOptionalInput(node, 3, rt.tensors());
    SetOutput(node, 0, k(data, pads, constant_value, axes, mode, &rt), rt);
    return;
  }

  // Legacy opset (<11): ``pads`` is an INTS attribute and ``value`` is a
  // FLOAT attribute (default 0).
  const std::vector<int64_t> pads_attr = GetAttributeIntsOrDefault(node, "pads", {});
  const Tensor pads = Tensor::FromInt64("", {static_cast<int64_t>(pads_attr.size())}, pads_attr);
  const float value = GetAttributeFloatOrDefault(node, "value", 0.0f);
  if (data.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    const Tensor cv = Tensor::FromFloat("", /*shape=*/{}, {value});
    SetOutput(node, 0, k(data, pads, &cv, /*axes=*/nullptr, mode, &rt), rt);
  } else {
    // For non-float dtypes the legacy form's float ``value`` attribute is
    // ill-defined; fall back to a zero-initialized constant.
    SetOutput(node, 0, k(data, pads, /*constant_value=*/nullptr, /*axes=*/nullptr, mode, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
