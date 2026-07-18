// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_ort/schema/operator_sets_microsoft_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace microsoft {

std::string MakeBiasGeluDoc() {
  return R"DOC(
Bias Gelu.

This operator extends Gelu by adding bias input ``B`` to input ``A`` before
applying the Gelu activation.
)DOC";
}

std::string MakeBiasGeluGradDxDoc() {
  return R"DOC(
Computes ``dX`` for BiasGelu.

Input ``dY`` is the upstream gradient, ``X`` is the forward input, and ``B`` is
the bias used by BiasGelu. Output ``dX`` has the same shape and type as ``dY``
and ``X``.
)DOC";
}

} // namespace microsoft
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
