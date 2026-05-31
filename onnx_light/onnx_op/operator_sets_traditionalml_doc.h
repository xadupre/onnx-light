// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

/**
 * Returns the documentation string for the ArrayFeatureExtractor operator.
 *
 * @return Documentation string for the ArrayFeatureExtractor operator.
 */
std::string MakeArrayFeatureExtractorDoc();

/**
 * Returns the documentation string for the Imputer operator.
 *
 * @return Documentation string for the Imputer operator.
 */
std::string MakeImputerDoc();

/**
 * Returns the documentation string for the Binarizer operator.
 *
 * @return Documentation string for the Binarizer operator.
 */
std::string MakeBinarizerDoc();

/**
 * Returns the documentation string for the LabelEncoder operator.
 *
 * @return Documentation string for the LabelEncoder operator.
 */
std::string MakeLabelEncoderDoc();

/**
 * Returns the documentation string for the LinearClassifier operator.
 *
 * @return Documentation string for the LinearClassifier operator.
 */
std::string MakeLinearClassifierDoc();

/**
 * Returns the documentation string for the LinearRegressor operator.
 *
 * @return Documentation string for the LinearRegressor operator.
 */
std::string MakeLinearRegressorDoc();

/**
 * Returns the documentation string for the OneHotEncoder operator.
 *
 * @return Documentation string for the OneHotEncoder operator.
 */
std::string MakeOneHotEncoderDoc();

/**
 * Returns the documentation string for the Scaler operator.
 *
 * @return Documentation string for the Scaler operator.
 */
std::string MakeScalerDoc();

/**
 * Returns the documentation string for the SVMClassifier operator.
 *
 * @return Documentation string for the SVMClassifier operator.
 */
std::string MakeSVMClassifierDoc();

/**
 * Returns the documentation string for the SVMRegressor operator.
 *
 * @return Documentation string for the SVMRegressor operator.
 */
std::string MakeSVMRegressorDoc();

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
