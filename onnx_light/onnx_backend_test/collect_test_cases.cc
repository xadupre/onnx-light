// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Implements CollectTestCases and CollectTestCasesByName (declared in
// onnx_core/backend_test/test_case.h) by delegating to the per-category
// Collect* helpers that live in onnx_backend_test/cases/.  This file must be
// compiled as part of lib_onnx_backend_test (not lib_onnx_core) because it
// takes a direct dependency on every onnx_backend_test case category.

#include "onnx_core/backend_test/test_case.h"

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/cases/image/include_image_cases.h"
#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/cases_for_shapes/inplace/include_inplace_cases.h"
#include "onnx_backend_test/cases_for_shapes/release/include_release_cases.h"
#include "onnx_backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"
#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"

#include <regex>

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace backend_test {

using ::onnx_light::onnx_backend_test::CollectControlflowTestCases;
using ::onnx_light::onnx_backend_test::CollectEmptyShapeTestCases;
using ::onnx_light::onnx_backend_test::CollectGeneratorTestCases;
using ::onnx_light::onnx_backend_test::CollectImageTestCases;
using ::onnx_light::onnx_backend_test::CollectInPlaceTestCases;
using ::onnx_light::onnx_backend_test::CollectLogicalTestCases;
using ::onnx_light::onnx_backend_test::CollectMathTestCases;
using ::onnx_light::onnx_backend_test::CollectNanInfTestCases;
using ::onnx_light::onnx_backend_test::CollectNNTestCases;
using ::onnx_light::onnx_backend_test::CollectObjectDetectionTestCases;
using ::onnx_light::onnx_backend_test::CollectOptionalTestCases;
using ::onnx_light::onnx_backend_test::CollectPreviewTestCases;
using ::onnx_light::onnx_backend_test::CollectQuantizationTestCases;
using ::onnx_light::onnx_backend_test::CollectReductionTestCases;
using ::onnx_light::onnx_backend_test::CollectReleaseTestCases;
using ::onnx_light::onnx_backend_test::CollectSequenceTestCases;
using ::onnx_light::onnx_backend_test::CollectShapeInferenceTestCases;
using ::onnx_light::onnx_backend_test::CollectShapeTagTestCases;
using ::onnx_light::onnx_backend_test::CollectTensorTestCases;
using ::onnx_light::onnx_backend_test::CollectTextTestCases;
using ::onnx_light::onnx_backend_test::CollectTraditionalMLTestCases;
using ::onnx_light::onnx_backend_test::CollectTrainingTestCases;

std::vector<TestCase> CollectTestCases(const std::string &op_type, bool include_big,
                                       TestMode mode) {
  std::vector<TestCase> registry;
  CollectControlflowTestCases(registry, op_type, mode);
  CollectGeneratorTestCases(registry, op_type, mode);
  CollectImageTestCases(registry, op_type, mode);
  CollectLogicalTestCases(registry, op_type, mode);
  CollectMathTestCases(registry, op_type, mode);
  CollectNNTestCases(registry, op_type, mode);
  CollectObjectDetectionTestCases(registry, op_type, mode);
  CollectOptionalTestCases(registry, op_type, mode);
  CollectPreviewTestCases(registry, op_type, mode);
  CollectQuantizationTestCases(registry, op_type, mode);
  CollectReductionTestCases(registry, op_type, mode);
  CollectSequenceTestCases(registry, op_type, mode);
  CollectTensorTestCases(registry, op_type, mode);
  CollectTextTestCases(registry, op_type, mode);
  CollectTraditionalMLTestCases(registry, op_type, mode);
  CollectTrainingTestCases(registry, op_type, mode);
  CollectShapeInferenceTestCases(registry, op_type, include_big, mode);
  CollectEmptyShapeTestCases(registry, op_type, mode);
  CollectInPlaceTestCases(registry, op_type, mode);
  CollectReleaseTestCases(registry, op_type, mode);
  CollectShapeTagTestCases(registry, op_type, mode);
  CollectNanInfTestCases(registry, op_type, mode);
  if (include_big) {
    return registry;
  }
  std::vector<TestCase> filtered;
  filtered.reserve(registry.size());
  for (auto &tc : registry) {
    if (tc.name.find("_big_") == std::string::npos) {
      filtered.emplace_back(std::move(tc));
    }
  }
  return filtered;
}

std::vector<TestCase> CollectTestCasesByName(const std::string &name_regex, bool include_big,
                                             TestMode mode) {
  std::vector<TestCase> all_cases = CollectTestCases("", include_big, mode);
  if (name_regex.empty()) {
    return all_cases;
  }
  std::regex pattern(name_regex);
  std::vector<TestCase> filtered;
  filtered.reserve(all_cases.size());
  for (auto &tc : all_cases) {
    if (std::regex_search(tc.name, pattern)) {
      filtered.emplace_back(std::move(tc));
    }
  }
  return filtered;
}

} // namespace backend_test
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
