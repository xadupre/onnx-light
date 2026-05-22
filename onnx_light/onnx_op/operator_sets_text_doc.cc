// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_text_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace text {

std::string MakeStringConcatDoc(int since_version) {
  if (since_version == 20) {
    return R"DOC(StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support))DOC";
  }
  return "";
}

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
