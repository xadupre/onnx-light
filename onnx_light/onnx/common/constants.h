// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_pb.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE {

constexpr const char *AI_ONNX_ML_DOMAIN = "ai.onnx.ml";
constexpr const char *AI_ONNX_TRAINING_DOMAIN = "ai.onnx.training";
constexpr const char *AI_ONNX_PREVIEW_DOMAIN = "ai.onnx.preview";
constexpr const char *AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training";
constexpr const char *ONNX_DOMAIN = "";
constexpr const char *AI_ONNX_DOMAIN = "ai.onnx";

inline std::string NormalizeDomain(const std::string &domain) {
  return (domain == AI_ONNX_DOMAIN) ? ONNX_DOMAIN : domain;
}

inline bool IsOnnxDomain(const std::string &domain) {
  return (domain == AI_ONNX_DOMAIN) || (domain == ONNX_DOMAIN);
}

constexpr bool OPTIONAL_VALUE = false;

constexpr const char *DATA_BATCH = "DATA_BATCH";
constexpr const char *DATA_CHANNEL = "DATA_CHANNEL";
constexpr const char *DATA_TIME = "DATA_TIME";
constexpr const char *DATA_FEATURE = "DATA_FEATURE";
constexpr const char *FILTER_IN_CHANNEL = "FILTER_IN_CHANNEL";
constexpr const char *FILTER_OUT_CHANNEL = "FILTER_OUT_CHANNEL";
constexpr const char *FILTER_SPATIAL = "FILTER_SPATIAL";

constexpr const char *TENSOR = "TENSOR";
constexpr const char *IMAGE = "IMAGE";
constexpr const char *AUDIO = "AUDIO";
constexpr const char *TEXT = "TEXT";

// ONNX IR version history enum (mirrors the protobuf-generated Version enum).
// IR_VERSION is the current IR version supported by this build.
enum Version : int {
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION_2019_9_19 = 6,
  IR_VERSION_2020_5_8 = 7,
  IR_VERSION_2021_7_30 = 8,
  IR_VERSION_2022_10_28 = 9,
  IR_VERSION_2023_4_25 = 10,
  IR_VERSION_2024_3_25 = 11,
  IR_VERSION_2025_1_24 = 12,
  IR_VERSION = 13
};

} // namespace ONNX_LIGHT_NAMESPACE
