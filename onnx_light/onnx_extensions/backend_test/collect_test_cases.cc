// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Defines CollectTestCases and CollectTestCasesByName (declared in
// onnx_core/backend_test/test_case.h) and registers all per-category
// collector functions into the global registry defined in
// onnx_core/backend_test/test_case_registry.cc.  This file must be compiled as part of
// lib_onnx_backend_test (not lib_onnx_core) because it takes a direct
// dependency on every onnx_backend_test case category.

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"

#include "onnx_extensions/backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/backend_test/cases/image/include_image_cases.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_extensions/backend_test/cases/optional/include_optional_cases.h"
#include "onnx_extensions/backend_test/cases/preview/include_preview_cases.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_extensions/backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/backend_test/cases/text/include_text_cases.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/backend_test/cases/training/include_training_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inplace/include_inplace_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/peak_memory/include_peak_memory_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/release/include_release_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"
#include "onnx_extensions/backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"

#include <regex>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

// All 22 per-category collector functions are registered here in a single
// static initializer.  Because this variable is in the same translation unit
// as CollectTestCases / CollectTestCasesByName, C++ guarantees that it is
// fully initialized before either function body executes (namespace-scope
// statics within a TU are initialized in declaration order).  All
// registrations happen sequentially from this one initializer, so no mutex
// is needed.
// clang-format off
[[maybe_unused]] const int kRegisterAllCollectors = []() {
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectControlflowTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectGeneratorTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectImageTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectLogicalTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectMathTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectNNTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectObjectDetectionTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectOptionalTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectPreviewTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectQuantizationTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectReductionTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectSequenceTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectTensorTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectTextTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectTraditionalMLTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectTrainingTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool big, TestMode m) {
    onnx_backend_test::CollectShapeInferenceTestCases(r, op, big, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectEmptyShapeTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectInPlaceTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectPeakMemoryTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectReleaseTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectShapeTagTestCases(r, op, m);
  });
  RegisterTestCasesCollector([](std::vector<TestCase> &r, const std::string &op, bool, TestMode m) {
    onnx_backend_test::CollectNanInfTestCases(r, op, m);
  });
  return 0;
}();
// clang-format on

} // namespace

std::vector<TestCase> CollectTestCases(const std::string &op_type, bool include_big,
                                       TestMode mode) {
  std::vector<TestCase> registry;
  for (const auto &fn : GetRegisteredCollectors()) {
    fn(registry, op_type, include_big, mode);
  }
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

std::vector<TestCase> GetTestCaseByName(const std::string &name, bool include_big, TestMode mode) {
  std::vector<TestCase> all_cases = CollectTestCases("", include_big, mode);
  std::vector<TestCase> result;
  for (auto &tc : all_cases) {
    if (tc.name == name) {
      result.emplace_back(std::move(tc));
      break;
    }
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
