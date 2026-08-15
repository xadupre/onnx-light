// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/dispatch_table.h"

#include <memory>

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/canonicalization/cast_cast_binary_pattern.h"
#include "onnx_extensions/patterns/canonicalization/cast_cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

void RegisterPatterns() {
  static const bool registered = []() {
    core::builder::RegisterPattern("Cast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastPattern>();
                                   });
    core::builder::RegisterPattern("CastCast",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastCastPattern>();
                                   });
    core::builder::RegisterPattern("CastCastBinary",
                                   []() -> std::unique_ptr<core::builder::PatternOptimization> {
                                     return std::make_unique<CastCastBinaryPattern>();
                                   });
    return true;
  }();
  (void)registered;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
