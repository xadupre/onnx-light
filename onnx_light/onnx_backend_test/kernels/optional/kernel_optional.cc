// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Optional::operator()(const Tensor &input) const {
  Tensor out("", input.data_type, input.shape, std::vector<uint8_t>(input.data.size()));
  (*this)(input, out);
  return out;
}

void Optional::operator()(const Tensor &input, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.data_type != 0,
                      "kernel::Optional: input element type must be a defined DataType.");
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Optional preallocated output data_type must match input data_type.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Optional preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == input.data.size(),
                      "kernel::Optional preallocated output buffer has unexpected size in bytes.");
  // Passthrough: the present optional wraps an exact copy of the input.
  // ``std::memmove``-style safety is required so the in-place overload may
  // alias ``input`` and ``output``.
  if (!output.data.empty() && output.data.data() != input.data.data()) {
    std::copy(input.data.begin(), input.data.end(), output.data.begin());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
