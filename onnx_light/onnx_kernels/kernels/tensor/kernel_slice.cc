// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

std::vector<int64_t> ReadIntInput(const Tensor &t, const std::string &name) {
  EXT_ENFORCE_INVALID(t.shape.size() <= 1, "kernel::Slice: '" + name + "' input must be 1-D.");
  const int64_t n = t.shape.empty() ? 0 : t.shape[0];
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n == 0) {
    return out;
  }
  if (t.data_type == static_cast<int32_t>(DataType::INT64)) {
    std::memcpy(out.data(), t.bytes(), static_cast<std::size_t>(n) * sizeof(int64_t));
    return out;
  }
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::INT32),
                      "kernel::Slice: '" + name + "' input must be INT32 or INT64.");
  const int32_t *p = reinterpret_cast<const int32_t *>(t.bytes());
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = static_cast<int64_t>(p[i]);
  }
  return out;
}

void ProcessSliceInputs(const int64_t dim, int64_t &start, int64_t &end, int64_t step) {
  EXT_ENFORCE_INVALID(step != 0, "kernel::Slice: step cannot be 0.");
  if (dim == 0) {
    start = 0;
    end = 0;
    return;
  }
  if (start < 0) {
    start += dim;
  }
  if (end < 0) {
    end += dim;
  }
  if (step < 0) {
    start = std::clamp(start, static_cast<int64_t>(0), dim - 1);
    end = std::clamp(end, static_cast<int64_t>(-1), dim - 1);
  } else {
    start = std::clamp(start, static_cast<int64_t>(0), dim);
    end = std::clamp(end, static_cast<int64_t>(0), dim);
  }
}

int64_t SliceLength(int64_t start, int64_t end, int64_t step) {
  if (step > 0) {
    if (end <= start) {
      return 0;
    }
    return 1 + (end - start - 1) / step;
  }
  if (end >= start) {
    return 0;
  }
  const int64_t step_abs = -step;
  return 1 + (start - end - 1) / step_abs;
}

struct SliceLayout {
  std::vector<int64_t> starts;
  std::vector<int64_t> steps;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> in_strides;
  std::vector<int64_t> out_strides;
  int64_t total_elements = 1;
  std::size_t elem_size = 0;
};

SliceLayout ComputeSliceLayout(const Tensor &data, const Tensor &starts_t, const Tensor &ends_t,
                               const Tensor *axes_t, const Tensor *steps_t) {
  const std::vector<int64_t> starts_in = ReadIntInput(starts_t, "starts");
  const std::vector<int64_t> ends_in = ReadIntInput(ends_t, "ends");
  EXT_ENFORCE_INVALID(starts_in.size() == ends_in.size(),
                      "kernel::Slice: starts and ends inputs must have the same length.");

  const std::size_t n = starts_in.size();
  std::vector<int64_t> axes = axes_t ? ReadIntInput(*axes_t, "axes") : std::vector<int64_t>{};
  if (!axes_t) {
    axes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
  }
  EXT_ENFORCE_INVALID(axes.size() == n,
                      "kernel::Slice: axes input must have the same length as starts.");

  std::vector<int64_t> steps = steps_t ? ReadIntInput(*steps_t, "steps") : std::vector<int64_t>{};
  if (!steps_t) {
    steps.assign(n, static_cast<int64_t>(1));
  }
  EXT_ENFORCE_INVALID(steps.size() == n,
                      "kernel::Slice: steps input must have the same length as starts.");

  const int64_t rank = static_cast<int64_t>(data.shape.size());
  SliceLayout layout;
  layout.starts.assign(static_cast<std::size_t>(rank), static_cast<int64_t>(0));
  layout.steps.assign(static_cast<std::size_t>(rank), static_cast<int64_t>(1));
  layout.out_shape = data.shape;

  for (std::size_t i = 0; i < n; ++i) {
    int64_t axis = axes[i];
    if (axis < 0) {
      axis += rank;
    }
    EXT_ENFORCE_INVALID(axis >= 0 && axis < rank, "kernel::Slice: axis is out of range.");
    int64_t s = starts_in[i];
    int64_t e = ends_in[i];
    const int64_t st = steps[i];
    ProcessSliceInputs(data.shape[static_cast<std::size_t>(axis)], s, e, st);
    layout.starts[static_cast<std::size_t>(axis)] = s;
    layout.steps[static_cast<std::size_t>(axis)] = st;
    layout.out_shape[static_cast<std::size_t>(axis)] = SliceLength(s, e, st);
  }

  layout.elem_size = ElementSize(data.data_type);

  layout.in_strides.assign(static_cast<std::size_t>(rank), static_cast<int64_t>(1));
  layout.out_strides.assign(static_cast<std::size_t>(rank), static_cast<int64_t>(1));
  for (int64_t i = rank - 2; i >= 0; --i) {
    layout.in_strides[static_cast<std::size_t>(i)] =
        layout.in_strides[static_cast<std::size_t>(i + 1)] *
        data.shape[static_cast<std::size_t>(i + 1)];
    layout.out_strides[static_cast<std::size_t>(i)] =
        layout.out_strides[static_cast<std::size_t>(i + 1)] *
        layout.out_shape[static_cast<std::size_t>(i + 1)];
  }

  layout.total_elements = 1;
  for (int64_t d : layout.out_shape) {
    layout.total_elements *= d;
  }
  return layout;
}

} // namespace

Tensor Slice::operator()(const Tensor &data, const Tensor &starts, const Tensor &ends,
                         const Tensor *axes, const Tensor *steps) const {
  const SliceLayout layout = ComputeSliceLayout(data, starts, ends, axes, steps);
  Tensor out(
      "", data.data_type, layout.out_shape,
      std::vector<uint8_t>(static_cast<std::size_t>(layout.total_elements) * layout.elem_size));
  (*this)(data, starts, ends, axes, steps, out);
  return out;
}

void Slice::operator()(const Tensor &data, const Tensor &starts, const Tensor &ends,
                       const Tensor *axes, const Tensor *steps, Tensor &output) const {
  const SliceLayout layout = ComputeSliceLayout(data, starts, ends, axes, steps);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Slice: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == layout.out_shape,
                      "kernel::Slice: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.data.size() ==
                          static_cast<std::size_t>(layout.total_elements) * layout.elem_size,
                      "kernel::Slice: preallocated output byte-size mismatch.");

  const std::size_t rank = data.shape.size();
  for (int64_t out_idx = 0; out_idx < layout.total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const int64_t coord = layout.out_strides.empty() ? 0 : remaining / layout.out_strides[axis];
      if (!layout.out_strides.empty()) {
        remaining %= layout.out_strides[axis];
      }
      const int64_t in_coord = layout.starts[axis] + coord * layout.steps[axis];
      in_idx += in_coord * layout.in_strides[axis];
    }
    std::memcpy(output.data.data() + static_cast<std::size_t>(out_idx) * layout.elem_size,
                data.bytes() + static_cast<std::size_t>(in_idx) * layout.elem_size,
                layout.elem_size);
  }
}

} // namespace kernel

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
