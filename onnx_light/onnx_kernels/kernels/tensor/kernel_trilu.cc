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

// Reads the optional ``k`` input of Trilu. Returns 0 when ``k`` is null
// (the spec default). Requires ``k`` to be a 0-D INT64 tensor.
int64_t ReadK(const Tensor *k) {
  if (k == nullptr) {
    return 0;
  }
  EXT_ENFORCE_INVALID(k->data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Trilu: input 'k' must be a tensor(int64).");
  EXT_ENFORCE_INVALID(k->element_count() == 1,
                      "kernel::Trilu: input 'k' must be a 0-D tensor (single value).");
  return *k->AsInt64();
}

} // namespace

Tensor Trilu::operator()(const Tensor &input, const Tensor *k,
                         const Trilu::Attributes &attrs) const {
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
  (*this)(input, k, attrs, output);
  return output;
}

void Trilu::operator()(const Tensor &input, const Tensor *k, const Trilu::Attributes &attrs,
                       Tensor &output) const {
  EXT_ENFORCE_INVALID(input.shape.size() >= 2, "kernel::Trilu: input tensor must have rank >= 2.");
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Trilu: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Trilu: preallocated output shape mismatch.");

  const int64_t k_val = ReadK(k);
  const bool upper = attrs.upper != 0;

  const std::size_t rank = input.shape.size();
  const int64_t N = input.shape[rank - 2];
  const int64_t M = input.shape[rank - 1];

  int64_t batch = 1;
  for (std::size_t i = 0; i + 2 < rank; ++i) {
    batch *= input.shape[i];
  }
  const int64_t plane = N * M;
  const int64_t total = batch * plane;

  const bool is_string = input.data_type == static_cast<int32_t>(DataType::STRING);
  if (is_string) {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(input.string_data.size()) == total,
                        "kernel::Trilu: input string_data size does not match shape.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(output.string_data.size()) == total,
                        "kernel::Trilu: output string_data size does not match shape.");
  } else {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(input.data.size()) ==
                            static_cast<int64_t>(PackedByteSize(input.data_type, total)),
                        "kernel::Trilu: input data size does not match shape.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) ==
                            static_cast<int64_t>(PackedByteSize(input.data_type, total)),
                        "kernel::Trilu: output data size does not match shape.");
  }

  const std::size_t elem_size = is_string ? 0 : ElementSize(input.data_type);

  // A position (i, j) is kept when:
  //   upper == true  -> j >= i + k
  //   upper == false -> j <= i + k
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t base = b * plane;
    for (int64_t i = 0; i < N; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t flat = base + i * M + j;
        const bool keep = upper ? (j >= i + k_val) : (j <= i + k_val);
        if (is_string) {
          output.string_data[static_cast<std::size_t>(flat)] =
              keep ? input.string_data[static_cast<std::size_t>(flat)] : std::string();
        } else if (keep) {
          std::memcpy(output.data.data() + static_cast<std::size_t>(flat) * elem_size,
                      input.data.data() + static_cast<std::size_t>(flat) * elem_size, elem_size);
        } else {
          std::memset(output.data.data() + static_cast<std::size_t>(flat) * elem_size, 0,
                      elem_size);
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
