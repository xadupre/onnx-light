// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectTraditionalMLTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                   TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"ArrayFeatureExtractor", &RegisterArrayFeatureExtractorCases},
      {"Binarizer", &RegisterBinarizerCases},
      {"CastMap", &RegisterCastMapCases},
      {"CategoryMapper", &RegisterCategoryMapperCases},
      {"DictVectorizer", &RegisterDictVectorizerCases},
      {"FeatureVectorizer", &RegisterFeatureVectorizerCases},
      {"Imputer", &RegisterImputerCases},
      {"LabelEncoder", &RegisterLabelEncoderCases},
      {"LinearClassifier", &RegisterLinearClassifierCases},
      {"LinearRegressor", &RegisterLinearRegressorCases},
      {"Normalizer", &RegisterNormalizerCases},
      {"OneHotEncoder", &RegisterOneHotEncoderCases},
      {"SVMClassifier", &RegisterSVMClassifierCases},
      {"SVMRegressor", &RegisterSVMRegressorCases},
      {"Scaler", &RegisterScalerCases},
      {"TreeEnsemble", &RegisterTreeEnsembleCases},
      {"TreeEnsembleClassifier", &RegisterTreeEnsembleClassifierCases},
      {"TreeEnsembleRegressor", &RegisterTreeEnsembleRegressorCases},
      {"ZipMap", &RegisterZipMapCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
