// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"

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

// Computes the product of the first ``count`` dimensions of ``shape``.
int64_t PrefixProduct(const std::vector<int64_t> &shape, std::size_t count) {
  int64_t product = 1;
  for (std::size_t i = 0; i < count; ++i) {
    product *= shape[i];
  }
  return product;
}

// Computes the product of dimensions ``[from, shape.size())`` of ``shape``.
int64_t SuffixProduct(const std::vector<int64_t> &shape, std::size_t from) {
  int64_t product = 1;
  for (std::size_t i = from; i < shape.size(); ++i) {
    product *= shape[i];
  }
  return product;
}

// Resolves ``axis``/``new_axis``, validates the inputs, and computes:
//   * ``resolved_axis``: non-negative axis in the output shape;
//   * ``out_shape``    : the output tensor shape;
//   * ``elem_size``    : the size in bytes of one input element.
void ResolveAndValidate(const std::vector<Tensor> &inputs, int64_t axis, int64_t new_axis,
                        int &resolved_axis, std::vector<int64_t> &out_shape, size_t &elem_size) {
  EXT_ENFORCE_INVALID(new_axis == 0 || new_axis == 1,
                      "kernel::ConcatFromSequence: new_axis must be either 0 or 1.");
  EXT_ENFORCE_INVALID(!inputs.empty(),
                      "kernel::ConcatFromSequence: input sequence must not be empty.");

  const Tensor &first = inputs[0];
  EXT_ENFORCE_INVALID(first.data_type != 0,
                      "kernel::ConcatFromSequence: input element type must be a defined "
                      "DataType.");
  const int rank = static_cast<int>(first.shape.size());
  elem_size = first.element_size();

  for (std::size_t i = 1; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == first.data_type,
                        "kernel::ConcatFromSequence: all inputs must share the same data_type.");
    EXT_ENFORCE_INVALID(static_cast<int>(inputs[i].shape.size()) == rank,
                        "kernel::ConcatFromSequence: all inputs must share the same rank.");
  }

  const int upper_bound = (new_axis == 1) ? rank : rank - 1;
  const int lower_bound = (new_axis == 1) ? -rank - 1 : -rank;
  EXT_ENFORCE_INVALID(axis >= lower_bound && axis <= upper_bound,
                      "kernel::ConcatFromSequence: axis " + std::to_string(axis) +
                          " is out of range [" + std::to_string(lower_bound) + ", " +
                          std::to_string(upper_bound) + "].");
  resolved_axis = static_cast<int>(axis < 0 ? axis + upper_bound + 1 : axis);

  if (new_axis == 1) {
    // All inputs must share an identical shape.
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      EXT_ENFORCE_INVALID(
          inputs[i].shape == first.shape,
          "kernel::ConcatFromSequence: with new_axis=1 all inputs must share the same shape.");
    }
    out_shape.clear();
    out_shape.reserve(static_cast<std::size_t>(rank + 1));
    for (int d = 0; d <= rank; ++d) {
      if (d == resolved_axis) {
        out_shape.push_back(static_cast<int64_t>(inputs.size()));
      } else {
        out_shape.push_back(first.shape[d > resolved_axis ? d - 1 : d]);
      }
    }
  } else {
    // All inputs must share every dimension except ``resolved_axis``.
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      for (int d = 0; d < rank; ++d) {
        if (d == resolved_axis)
          continue;
        EXT_ENFORCE_INVALID(
            inputs[i].shape[d] == first.shape[d],
            "kernel::ConcatFromSequence: all inputs must agree on every dimension other than "
            "the concat axis.");
      }
    }
    out_shape = first.shape;
    int64_t axis_total = 0;
    for (const Tensor &in : inputs) {
      axis_total += in.shape[resolved_axis];
    }
    out_shape[resolved_axis] = axis_total;
  }
}

// Copies the concatenated bytes of ``inputs`` into ``output_bytes`` in
// row-major layout. ``out_shape`` and ``resolved_axis`` describe the
// output tensor; ``elem_size`` is the per-element byte size.
void CopyConcatenated(const std::vector<Tensor> &inputs, int resolved_axis,
                      const std::vector<int64_t> &out_shape, size_t elem_size,
                      std::vector<uint8_t> &output_bytes) {
  // Block of consecutive bytes that the concat sees as a "row" of the
  // outer block: outer * inner * elem_size bytes per input contribute
  // a slab whose width along the concat axis equals the input's
  // ``shape[resolved_axis]``.
  const int64_t outer = PrefixProduct(out_shape, static_cast<std::size_t>(resolved_axis));
  const int64_t inner = SuffixProduct(out_shape, static_cast<std::size_t>(resolved_axis + 1));
  const int64_t out_axis_dim = out_shape[resolved_axis];
  const size_t out_row_bytes = static_cast<size_t>(out_axis_dim * inner) * elem_size;

  for (int64_t o = 0; o < outer; ++o) {
    size_t out_offset = static_cast<size_t>(o) * out_row_bytes;
    for (const Tensor &in : inputs) {
      // Rank equality across inputs has already been enforced by
      // ResolveAndValidate / the new_axis=1 wrapper, so this index is
      // always in range.
      const int64_t in_axis_dim = in.shape[resolved_axis];
      const size_t slab_bytes = static_cast<size_t>(in_axis_dim * inner) * elem_size;
      const size_t in_offset = static_cast<size_t>(o) * slab_bytes;
      if (slab_bytes > 0) {
        std::memcpy(output_bytes.data() + out_offset, in.data.data() + in_offset, slab_bytes);
      }
      out_offset += slab_bytes;
    }
  }
}

} // namespace

Tensor ConcatFromSequence::operator()(const std::vector<Tensor> &inputs, int64_t axis,
                                      int64_t new_axis) const {
  int resolved_axis = 0;
  std::vector<int64_t> out_shape;
  size_t elem_size = 0;
  ResolveAndValidate(inputs, axis, new_axis, resolved_axis, out_shape, elem_size);
  const size_t total_bytes = static_cast<size_t>(elem_size) *
                             static_cast<size_t>(PrefixProduct(out_shape, out_shape.size()));
  Tensor out("", inputs[0].data_type, out_shape, std::vector<uint8_t>(total_bytes));
  if (new_axis == 1) {
    // Stacking with identical input shapes is equivalent to concatenating
    // along ``resolved_axis`` after inserting a unit dim at that position
    // into every input. We can therefore reuse ``CopyConcatenated`` by
    // treating each input's "axis dim" as 1.
    std::vector<Tensor> reshaped;
    reshaped.reserve(inputs.size());
    for (const Tensor &in : inputs) {
      Tensor r = in;
      r.shape.insert(r.shape.begin() + resolved_axis, 1);
      reshaped.push_back(std::move(r));
    }
    CopyConcatenated(reshaped, resolved_axis, out_shape, elem_size, out.data);
  } else {
    CopyConcatenated(inputs, resolved_axis, out_shape, elem_size, out.data);
  }
  return out;
}

void ConcatFromSequence::operator()(const std::vector<Tensor> &inputs, int64_t axis,
                                    int64_t new_axis, Tensor &output) const {
  int resolved_axis = 0;
  std::vector<int64_t> out_shape;
  size_t elem_size = 0;
  ResolveAndValidate(inputs, axis, new_axis, resolved_axis, out_shape, elem_size);
  EXT_ENFORCE_INVALID(
      output.data_type == inputs[0].data_type,
      "kernel::ConcatFromSequence preallocated output data_type must match input data_type.");
  EXT_ENFORCE_INVALID(
      output.shape == out_shape,
      "kernel::ConcatFromSequence preallocated output shape does not match the expected shape.");
  const size_t total_bytes = static_cast<size_t>(elem_size) *
                             static_cast<size_t>(PrefixProduct(out_shape, out_shape.size()));
  EXT_ENFORCE_INVALID(
      output.data.size() == total_bytes,
      "kernel::ConcatFromSequence preallocated output buffer has unexpected size in bytes.");
  if (new_axis == 1) {
    std::vector<Tensor> reshaped;
    reshaped.reserve(inputs.size());
    for (const Tensor &in : inputs) {
      Tensor r = in;
      r.shape.insert(r.shape.begin() + resolved_axis, 1);
      reshaped.push_back(std::move(r));
    }
    CopyConcatenated(reshaped, resolved_axis, out_shape, elem_size, output.data);
  } else {
    CopyConcatenated(inputs, resolved_axis, out_shape, elem_size, output.data);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
