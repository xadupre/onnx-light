// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "test_case_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <set>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;

namespace Test {

TEST(BackendExpm1, RegisteredCasesRunThroughNativeSessionAtOpset29) {
  auto cases = core::backend_test::CollectTestCases("Expm1");
  ASSERT_FALSE(cases.empty());
  std::set<int32_t> dtypes;
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const ModelProto &model = test_case.model();
    ASSERT_EQ(model.opset_import(0).version(), 29);
    ASSERT_EQ(model.graph().node(0).op_type(), "Expm1");
    if (test_case.name.find("test_cc_supported_dtype_Expm1_small_inputs_") == 0) {
      EXPECT_EQ(test_case.atol, 0.0);
    }
    for (const auto &data : test_case.data_sets()) {
      ASSERT_EQ(data.inputs.size(), 1u);
      ASSERT_EQ(data.outputs.size(), 1u);
      const Tensor &input = data.inputs[0];
      const Tensor &expected = data.outputs[0];
      dtypes.insert(input.data_type);
      RuntimeContext rt;
      rt.Set(input.name, input);
      RuntimeSession session(model);
      session.Run(rt);
      const Tensor &actual = rt.Get(expected.name);
      EXPECT_EQ(actual.data_type, expected.data_type);
      EXPECT_EQ(actual.shape, expected.shape);
      core::backend_test::ExpectBackendCaseTensorEqual(actual, expected, test_case);
      const bool is_double = actual.data_type == DataType::DOUBLE;
      const Tensor actual_float = PromoteToFloat32(actual);
      const Tensor expected_float = PromoteToFloat32(expected);
      for (int64_t i = 0; i < expected.element_count(); ++i) {
        const double reference = is_double ? expected.AsDouble()[i] : expected_float.AsFloat()[i];
        if (reference == 0) {
          const double value = is_double ? actual.AsDouble()[i] : actual_float.AsFloat()[i];
          EXPECT_EQ(value, 0.0);
          EXPECT_EQ(std::signbit(value), std::signbit(reference));
        }
      }
    }
  }
  EXPECT_EQ(dtypes, (std::set<int32_t>{static_cast<int32_t>(DataType::FLOAT),
                                       static_cast<int32_t>(DataType::DOUBLE),
                                       static_cast<int32_t>(DataType::FLOAT16),
                                       static_cast<int32_t>(DataType::BFLOAT16)}));
}

} // namespace Test
