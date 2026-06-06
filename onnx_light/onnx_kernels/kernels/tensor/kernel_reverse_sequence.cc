// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor ReverseSequence::operator()(const Tensor &input, const Tensor &sequence_lens,
                                   const ReverseSequence::Attributes &attrs) const {
  Tensor output;
  output.name = "";
  output.data_type = input.data_type;
  output.shape = input.shape;
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data.assign(static_cast<std::size_t>(input.element_count()), std::string());
  } else {
    output.data.assign(PackedByteSize(input.data_type, input.element_count()),
                       static_cast<uint8_t>(0));
  }
  (*this)(input, sequence_lens, attrs, output);
  return output;
}

void ReverseSequence::operator()(const Tensor &input, const Tensor &sequence_lens,
                                 const ReverseSequence::Attributes &attrs, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.shape.size() >= 2,
                      "kernel::ReverseSequence: input tensor must have rank >= 2.");
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::ReverseSequence: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::ReverseSequence: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(sequence_lens.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReverseSequence: 'sequence_lens' must be a tensor(int64).");
  EXT_ENFORCE_INVALID(sequence_lens.shape.size() == 1,
                      "kernel::ReverseSequence: 'sequence_lens' must have rank of 1.");

  const int64_t time_axis = attrs.time_axis;
  const int64_t batch_axis = attrs.batch_axis;
  EXT_ENFORCE_INVALID((time_axis == 0 || time_axis == 1) && (batch_axis == 0 || batch_axis == 1) &&
                          time_axis != batch_axis,
                      "kernel::ReverseSequence: time_axis and batch_axis must be 0 or 1 and must "
                      "differ from each other.");

  const std::size_t rank = input.shape.size();
  const int64_t T = input.shape[static_cast<std::size_t>(time_axis)];
  const int64_t B = input.shape[static_cast<std::size_t>(batch_axis)];

  EXT_ENFORCE_INVALID(sequence_lens.shape[0] == B,
                      "kernel::ReverseSequence: 'sequence_lens' length must equal "
                      "input.shape[batch_axis].");

  // Number of inner elements per (time, batch) pair (the product of the
  // remaining dimensions, i.e. dimensions with index >= 2).
  int64_t inner = 1;
  for (std::size_t i = 2; i < rank; ++i) {
    inner *= input.shape[i];
  }
  const int64_t total = T * B * inner;

  const bool is_string = input.data_type == static_cast<int32_t>(DataType::STRING);
  if (is_string) {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(input.string_data.size()) == total,
                        "kernel::ReverseSequence: input string_data size does not match shape.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(output.string_data.size()) == total,
                        "kernel::ReverseSequence: output string_data size does not match shape.");
  } else {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(input.data.size()) ==
                            static_cast<int64_t>(PackedByteSize(input.data_type, total)),
                        "kernel::ReverseSequence: input data size does not match shape.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) ==
                            static_cast<int64_t>(PackedByteSize(input.data_type, total)),
                        "kernel::ReverseSequence: output data size does not match shape.");
  }

  const std::size_t elem_size = is_string ? 0 : ElementSize(input.data_type);

  // Strides for the canonical [dim0, dim1, inner] interpretation. With
  // ``time_axis`` and ``batch_axis`` both restricted to {0, 1}, the input is
  // either laid out as ``[T, B, inner]`` (time_axis=0, batch_axis=1) or
  // ``[B, T, inner]`` (time_axis=1, batch_axis=0). We compute flat positions
  // using the actual strides derived from ``input.shape``.
  const int64_t stride_dim1 = inner;
  const int64_t stride_dim0 = input.shape[1] * stride_dim1;

  const int64_t *seq_data = sequence_lens.AsInt64();
  for (int64_t b = 0; b < B; ++b) {
    const int64_t seq_len = seq_data[b];
    EXT_ENFORCE_INVALID(seq_len >= 0 && seq_len <= T,
                        "kernel::ReverseSequence: sequence_lens[i] must be in [0, T].");
    for (int64_t t = 0; t < T; ++t) {
      // Source time index: within the reversed prefix it mirrors to
      // ``seq_len - 1 - t``; past the prefix it is the identity ``t``.
      const int64_t src_t = (t < seq_len) ? (seq_len - 1 - t) : t;

      int64_t dst_dim0, dst_dim1, src_dim0, src_dim1;
      if (time_axis == 0) {
        // shape = [T, B, inner]
        dst_dim0 = t;
        dst_dim1 = b;
        src_dim0 = src_t;
        src_dim1 = b;
      } else {
        // shape = [B, T, inner]
        dst_dim0 = b;
        dst_dim1 = t;
        src_dim0 = b;
        src_dim1 = src_t;
      }
      const int64_t dst_off = dst_dim0 * stride_dim0 + dst_dim1 * stride_dim1;
      const int64_t src_off = src_dim0 * stride_dim0 + src_dim1 * stride_dim1;

      if (is_string) {
        for (int64_t i = 0; i < inner; ++i) {
          output.string_data[static_cast<std::size_t>(dst_off + i)] =
              input.string_data[static_cast<std::size_t>(src_off + i)];
        }
      } else {
        std::memcpy(output.data.data() + static_cast<std::size_t>(dst_off) * elem_size,
                    input.data.data() + static_cast<std::size_t>(src_off) * elem_size,
                    static_cast<std::size_t>(inner) * elem_size);
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
