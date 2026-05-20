// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

const char *MakeCastDoc();
const char *MakeCastInputDescription();
const char *MakeCastOutputDescription();
const char *MakeCastLegacyInputConstraintDescription();
const char *MakeCastLegacyOutputConstraintDescription();
const char *MakeCastInputConstraintDescription();
const char *MakeCastOutputConstraintDescription();

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
