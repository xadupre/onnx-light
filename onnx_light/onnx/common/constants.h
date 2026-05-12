// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_pb.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE {

/// Domain for ONNX-ML operators.
constexpr const char *AI_ONNX_ML_DOMAIN = "ai.onnx.ml";
/// Domain for ONNX Training operators.
constexpr const char *AI_ONNX_TRAINING_DOMAIN = "ai.onnx.training";
/// Domain for preview ONNX operators.
constexpr const char *AI_ONNX_PREVIEW_DOMAIN = "ai.onnx.preview";
/// Domain for preview ONNX Training operators.
constexpr const char *AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training";
/// Canonical empty-string ONNX domain.
constexpr const char *ONNX_DOMAIN = "";
/// Legacy alias for the canonical ONNX domain.
constexpr const char *AI_ONNX_DOMAIN = "ai.onnx";

/// Returns the canonical ONNX domain for the provided domain value.
inline std::string NormalizeDomain(const std::string &domain) {
  return (domain == AI_ONNX_DOMAIN) ? ONNX_DOMAIN : domain;
}

/// Indicates whether a domain is the canonical ONNX domain or its legacy alias.
inline bool IsOnnxDomain(const std::string &domain) {
  return (domain == AI_ONNX_DOMAIN) || (domain == ONNX_DOMAIN);
}

/// Default optional-value flag used by schema helpers.
constexpr bool OPTIONAL_VALUE = false;

/// Symbolic batch axis label.
constexpr const char *DATA_BATCH = "DATA_BATCH";
/// Symbolic channel axis label.
constexpr const char *DATA_CHANNEL = "DATA_CHANNEL";
/// Symbolic time axis label.
constexpr const char *DATA_TIME = "DATA_TIME";
/// Symbolic feature axis label.
constexpr const char *DATA_FEATURE = "DATA_FEATURE";
/// Symbolic filter input-channel axis label.
constexpr const char *FILTER_IN_CHANNEL = "FILTER_IN_CHANNEL";
/// Symbolic filter output-channel axis label.
constexpr const char *FILTER_OUT_CHANNEL = "FILTER_OUT_CHANNEL";
/// Symbolic filter spatial axis label.
constexpr const char *FILTER_SPATIAL = "FILTER_SPATIAL";

/// Type string for tensor values.
constexpr const char *TENSOR = "TENSOR";
/// Type string for image values.
constexpr const char *IMAGE = "IMAGE";
/// Type string for audio values.
constexpr const char *AUDIO = "AUDIO";
/// Type string for text values.
constexpr const char *TEXT = "TEXT";

} // namespace ONNX_LIGHT_NAMESPACE
