// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace traditionalml {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported traditional ML
 * operators (e.g. Binarizer, LabelEncoder, Normalizer, OneHotEncoder, Scaler,
 * TreeEnsemble, TreeEnsembleClassifier, TreeEnsembleRegressor,
 * SVMClassifier, SVMRegressor, ZipMap) in the
 * ai.onnx.ml domain.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx.ml domain, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema>
GetAllOnnxOpTraditionalMLSchemasWithHistory(const std::string &op_type = "", bool init_doc = true);

} // namespace traditionalml
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
