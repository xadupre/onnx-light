// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file default_onnx_schema.h
 * @brief Default schema provider for GraphBuilder backed by the built-in
 *        ONNX operator schemas.
 *
 * This declaration lives in the ``onnx_op`` library (which owns the built-in
 * schemas). It is spelled with the literal ``onnx_light`` namespace so its
 * symbol identity matches callers compiled in ``onnx_light`` translation units,
 * while its definition can call ``onnx_op::GetAllOnnxOpSchemasWithHistory`` from
 * a translation unit that keeps ``ONNX_LIGHT_NAMESPACE`` as a literal token.
 */

#pragma once

#include <string>
#include <vector>

#include "onnx_core/builder/op_schema_info.h"

namespace onnx_light {
namespace core {
namespace builder {

/// Returns the versioned schema history of ``op_type`` across every domain,
/// distilled into :cpp:struct:`OpSchemaInfo` entries. An empty vector means the
/// operator is unknown to the built-in ONNX schemas.
std::vector<OpSchemaInfo> DefaultOnnxSchemaLookup(const std::string &op_type);

} // namespace builder
} // namespace core
} // namespace onnx_light
