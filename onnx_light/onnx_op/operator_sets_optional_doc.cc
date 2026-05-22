// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_optional_doc.h"

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace optional {

std::string MakeOptionalDoc(int since_version) {
  if (since_version == 15) {
    return R"DOC(
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
)DOC";
  }
  throw SchemaError("Unsupported Optional since_version: " + std::to_string(since_version));
}

std::string MakeOptionalHasElementDoc(int since_version) {
  if (since_version == 18) {
    return R"DOC(
Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
)DOC";
  }
  if (since_version == 15) {
    return R"DOC(
Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
)DOC";
  }
  throw SchemaError("Unsupported OptionalHasElement since_version: " +
                    std::to_string(since_version));
}

std::string MakeOptionalGetElementDoc(int since_version) {
  if (since_version == 18) {
    return R"DOC(
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
)DOC";
  }
  if (since_version == 15) {
    return R"DOC(
Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.
)DOC";
  }
  throw SchemaError("Unsupported OptionalGetElement since_version: " +
                    std::to_string(since_version));
}

} // namespace optional
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
