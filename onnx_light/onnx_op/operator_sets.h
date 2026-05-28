// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"
#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_image.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_nn.h"
#include "onnx_op/operator_sets_object_detection.h"
#include "onnx_op/operator_sets_optional.h"
#include "onnx_op/operator_sets_preview.h"
#include "onnx_op/operator_sets_quantization.h"
#include "onnx_op/operator_sets_reduction.h"
#include "onnx_op/operator_sets_sequence.h"
#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_text.h"
#include "onnx_op/operator_sets_traditionalml.h"
#include "onnx_op/operator_sets_training.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

/**
 * Returns the complete versioned schema history for all supported ONNX
 * operator domains.
 *
 * Combines schemas from the controlflow, generator, image, logical, math, nn,
 * object_detection, optional, preview, quantization, reduction, sequence,
 * tensor, text, traditionalml, and training sub-namespaces into a single flat
 * list ordered by domain, operator name, and descending opset version.
 *
 * @param init_doc If true (default), each schema's documentation string is
 *        populated. When false, documentation strings are discarded (doc()
 *        returns ""), which can save memory when documentation is not needed.
 * @param op_type If non-empty, only schemas whose ``name()`` equals
 *        ``op_type`` are returned. When empty (default), schemas for every
 *        registered operator are returned. This is convenient for tests that
 *        only need the history of a single operator.
 * @return Vector of LightOpSchema objects covering all supported operators and
 *         their historic opset versions.
 */
std::vector<LightOpSchema> GetAllOnnxOpSchemasWithHistory(bool init_doc = true,
                                                          const std::string &op_type = "");

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
