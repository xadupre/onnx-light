// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(BackendTestCase, TreeEnsembleSoftmaxZeroUsesUniformZeroSumFallback) {
  std::vector<core::backend_test::TestCase> cases;
  onnx_backend_test::CollectTraditionalMLTestCases(cases, "TreeEnsemble");
  for (int64_t targets : {1, 2, 3, 5}) {
    for (const std::string type : {"float", "double"}) {
      const std::string name =
          "test_cc_treeensemble_softmax_zero_sum_zero_" + std::to_string(targets) + "_" + type;
      SCOPED_TRACE(name);
      const auto it = std::find_if(cases.begin(), cases.end(),
                                   [&name](const auto &tc) { return tc.name == name; });
      ASSERT_NE(it, cases.end());
      const auto &node = it->model().ref_graph().ref_node()[0];
      bool found_transform = false;
      for (const auto &attribute : node.ref_attribute()) {
        if (attribute.name() == "post_transform") {
          EXPECT_EQ(attribute.i(), 3);
          found_transform = true;
        }
      }
      EXPECT_TRUE(found_transform);
      ASSERT_EQ(it->data_sets().size(), 1u);
      const auto &data = it->data_sets()[0];
      ASSERT_EQ(data.inputs.size(), 1u);
      ASSERT_EQ(data.outputs.size(), 1u);
      const auto &output = data.outputs[0];
      EXPECT_EQ(output.data_type, data.inputs[0].data_type);
      ASSERT_EQ(output.shape, (std::vector<int64_t>{2, targets}));
      for (int64_t row = 0; row < 2; ++row) {
        double sum = 0.0;
        for (int64_t target = 0; target < targets; ++target) {
          const int64_t index = row * targets + target;
          const double value =
              type == "float" ? output.As<float>()[index] : output.As<double>()[index];
          EXPECT_TRUE(std::isfinite(value));
          EXPECT_NEAR(value, 1.0 / static_cast<double>(targets), 1e-7);
          sum += value;
        }
        EXPECT_NEAR(sum, 1.0, 1e-7);
      }
    }
  }
}

} // namespace Test
