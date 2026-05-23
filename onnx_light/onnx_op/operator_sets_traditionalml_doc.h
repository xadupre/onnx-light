// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

/**
 * Returns the documentation string for the LabelEncoder operator.
 *
 * @return Documentation string for the LabelEncoder operator.
 */
std::string MakeLabelEncoderDoc();

/**
 * Returns the documentation string for the ZipMap operator.
 *
 * @return Documentation string for the ZipMap operator.
 */
std::string MakeZipMapDoc();

/**
 * Returns the documentation string for the TreeEnsembleClassifier operator
 * at the requested opset version.
 *
 * @param since_version Opset version at which the schema was introduced
 *        (1, 3 or 5).
 * @return Documentation string for the TreeEnsembleClassifier operator.
 */
std::string MakeTreeEnsembleClassifierDoc(int since_version);

/**
 * Returns the documentation string for the TreeEnsembleRegressor operator
 * at the requested opset version.
 *
 * @param since_version Opset version at which the schema was introduced
 *        (1, 3 or 5).
 * @return Documentation string for the TreeEnsembleRegressor operator.
 */
std::string MakeTreeEnsembleRegressorDoc(int since_version);

/**
 * Returns the documentation string for the TreeEnsemble operator.
 *
 * @return Documentation string for the TreeEnsemble operator.
 */
std::string MakeTreeEnsembleDoc();

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
