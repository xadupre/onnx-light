// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Materializes a scalar ``Tensor<bool, {}>`` carrying the given value.
Tensor MakeScalarBool(bool value) {
  Tensor out("", static_cast<int32_t>(DataType::BOOL), std::vector<int64_t>{},
             std::vector<uint8_t>(1, value ? uint8_t{1} : uint8_t{0}));
  return out;
}

} // namespace

Tensor OptionalHasElement::operator()(const Tensor &input) const {
  EXT_ENFORCE_INVALID(input.data_type != 0,
                      "kernel::OptionalHasElement: input element type must be a defined DataType.");
  // The runtime Tensor type cannot represent an "empty optional", so any
  // concrete tensor input is treated as the present element and the
  // operator returns true.
  return MakeScalarBool(true);
}

Tensor OptionalHasElement::operator()(const Sequence &input) const {
  EXT_ENFORCE_INVALID(
      input.elem_type != 0,
      "kernel::OptionalHasElement: input sequence elem_type must be a defined DataType.");
  return MakeScalarBool(true);
}

Tensor OptionalHasElement::operator()() const {
  // Opset 18: an omitted input is reported as empty.
  return MakeScalarBool(false);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
