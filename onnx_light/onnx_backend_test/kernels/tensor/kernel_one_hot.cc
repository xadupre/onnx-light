// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads ``index`` as an ``int64_t`` from a tensor of any supported numeric
// element type. Throws ``std::invalid_argument`` when the dtype is not
// supported by the reference kernel.
int64_t ReadScalarAsInt64(const Tensor &t, std::size_t index) {
  switch (static_cast<DataType>(t.data_type)) {
  case DataType::INT64:
    return t.AsInt64()[index];
  case DataType::INT32:
    return static_cast<int64_t>(t.AsInt32()[index]);
  case DataType::INT16:
    return static_cast<int64_t>(t.AsInt16()[index]);
  case DataType::INT8:
    return static_cast<int64_t>(t.AsInt8()[index]);
  case DataType::UINT64:
    return static_cast<int64_t>(t.AsUint64()[index]);
  case DataType::UINT32:
    return static_cast<int64_t>(t.AsUint32()[index]);
  case DataType::UINT16:
    return static_cast<int64_t>(t.AsUint16()[index]);
  case DataType::UINT8:
    return static_cast<int64_t>(t.AsUint8()[index]);
  case DataType::FLOAT:
    return static_cast<int64_t>(t.AsFloat()[index]);
  case DataType::DOUBLE:
    return static_cast<int64_t>(t.AsDouble()[index]);
  default:
    EXT_ENFORCE_INVALID(false, "kernel::OneHot: unsupported numeric dtype.");
  }
  return 0;
}

// Validates ``depth`` and returns its scalar value as int64. ``depth`` must
// be a scalar or a rank-1 tensor with exactly one element.
int64_t ReadDepth(const Tensor &depth) {
  EXT_ENFORCE_INVALID(depth.shape.size() == 0 ||
                          (depth.shape.size() == 1 && depth.shape[0] == 1),
                      "kernel::OneHot: input 'depth' must be a scalar or a rank-1 tensor with a "
                      "single element.");
  EXT_ENFORCE_INVALID(depth.element_count() == 1,
                      "kernel::OneHot: input 'depth' must have exactly one element.");
  const int64_t value = ReadScalarAsInt64(depth, 0);
  EXT_ENFORCE_INVALID(value > 0, "kernel::OneHot: input 'depth' must be > 0.");
  return value;
}

// Resolves ``axis`` (relative to ``out_rank``) into a non-negative value.
int64_t ResolveAxis(int64_t axis, int64_t out_rank) {
  EXT_ENFORCE_INVALID(axis >= -out_rank && axis < out_rank,
                      "kernel::OneHot: 'axis' is out of range.");
  return axis < 0 ? axis + out_rank : axis;
}

// Computes the output shape: ``indices.shape`` with ``depth`` inserted at
// position ``axis`` (already normalised to a non-negative value).
std::vector<int64_t> ComputeOneHotShape(const std::vector<int64_t> &indices_shape,
                                        int64_t axis_pos, int64_t depth) {
  std::vector<int64_t> out_shape;
  out_shape.reserve(indices_shape.size() + 1);
  for (int64_t i = 0; i < static_cast<int64_t>(indices_shape.size()) + 1; ++i) {
    if (i == axis_pos) {
      out_shape.push_back(depth);
    } else if (i < axis_pos) {
      out_shape.push_back(indices_shape[static_cast<std::size_t>(i)]);
    } else {
      out_shape.push_back(indices_shape[static_cast<std::size_t>(i - 1)]);
    }
  }
  return out_shape;
}

// Fills ``output`` with ``off_value`` repeated for every element, where
// ``off_value`` is a single element starting at ``values.data[]`` index 0.
// Uses ``memcpy`` to copy ``elem_size`` bytes per element so any numeric or
// ``BOOL`` element type is supported.
void FillScalarRepeat(const uint8_t *src, std::size_t elem_size, std::size_t count,
                      uint8_t *dst) {
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(dst + i * elem_size, src, elem_size);
  }
}

} // namespace

Tensor OneHot::operator()(const Tensor &indices, const Tensor &depth, const Tensor &values,
                          const OneHot::Attributes &attrs) const {
  EXT_ENFORCE_INVALID(values.shape.size() == 1 && values.element_count() == 2,
                      "kernel::OneHot: input 'values' must be a rank-1 tensor with exactly two "
                      "elements [off_value, on_value].");
  const int64_t depth_val = ReadDepth(depth);
  const int64_t out_rank = static_cast<int64_t>(indices.shape.size()) + 1;
  const int64_t axis_pos = ResolveAxis(attrs.axis, out_rank);
  const std::vector<int64_t> out_shape = ComputeOneHotShape(indices.shape, axis_pos, depth_val);

  Tensor output;
  output.name = "";
  output.data_type = values.data_type;
  output.shape = out_shape;
  const int64_t out_count = output.element_count();
  if (values.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data.assign(static_cast<std::size_t>(out_count), std::string());
  } else {
    output.data.assign(PackedByteSize(values.data_type, out_count), static_cast<uint8_t>(0));
  }
  (*this)(indices, depth, values, attrs, output);
  return output;
}

void OneHot::operator()(const Tensor &indices, const Tensor &depth, const Tensor &values,
                        const OneHot::Attributes &attrs, Tensor &output) const {
  EXT_ENFORCE_INVALID(values.shape.size() == 1 && values.element_count() == 2,
                      "kernel::OneHot: input 'values' must be a rank-1 tensor with exactly two "
                      "elements [off_value, on_value].");
  EXT_ENFORCE_INVALID(values.data_type != static_cast<int32_t>(DataType::STRING),
                      "kernel::OneHot: STRING element type is not supported.");
  EXT_ENFORCE_INVALID(output.data_type == values.data_type,
                      "kernel::OneHot: preallocated output dtype must match 'values' dtype.");

  const int64_t depth_val = ReadDepth(depth);
  const int64_t out_rank = static_cast<int64_t>(indices.shape.size()) + 1;
  const int64_t axis_pos = ResolveAxis(attrs.axis, out_rank);
  const std::vector<int64_t> expected_shape =
      ComputeOneHotShape(indices.shape, axis_pos, depth_val);
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::OneHot: preallocated output shape mismatch.");

  const std::size_t elem_size = ElementSize(values.data_type);
  const uint8_t *off_value = values.data.data();
  const uint8_t *on_value = values.data.data() + elem_size;
  const int64_t out_count = output.element_count();

  // Initialise the whole buffer with ``off_value``.
  FillScalarRepeat(off_value, elem_size, static_cast<std::size_t>(out_count), output.data.data());

  // Compute strides for the output and for the "indices half" of the output
  // (i.e. the output without the axis dimension). ``inner_size`` is the
  // product of dimensions of the output after the axis position; combined
  // with ``axis_stride = depth * inner_size`` it lets us index the axis
  // efficiently.
  int64_t inner_size = 1;
  for (int64_t i = axis_pos + 1; i < out_rank; ++i) {
    inner_size *= output.shape[static_cast<std::size_t>(i)];
  }
  const int64_t axis_stride = depth_val * inner_size;

  const int64_t total_indices = indices.element_count();
  // For every flat index ``flat`` of ``indices``, compute the corresponding
  // flat offset in ``output`` for axis position 0; the actual offset for
  // axis position ``k`` is ``base + k * inner_size``.
  // ``flat = outer * inner_size + inner`` (in the indices layout, which is
  // the output layout collapsed along the axis dimension).
  for (int64_t flat = 0; flat < total_indices; ++flat) {
    const int64_t outer = flat / inner_size;
    const int64_t inner = flat % inner_size;
    const int64_t base = outer * axis_stride + inner;

    int64_t k = ReadScalarAsInt64(indices, static_cast<std::size_t>(flat));
    // Negative indices are supported from opset 11 onwards. The opset 9
    // schema constrains entries to ``[0, depth)`` so a negative value
    // there leaves the row filled with ``off_value`` (matches the spec).
    if (k < 0) {
      k += depth_val;
    }
    if (k < 0 || k >= depth_val) {
      continue; // Leave the corresponding row filled with off_value.
    }
    std::memcpy(output.data.data() + (base + k * inner_size) * static_cast<int64_t>(elem_size),
                on_value, elem_size);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
