// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectQuantizationTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectQuantizationTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, QuantizeLinearCaseIsPresent) {
  auto cases = CollectTestCases("QuantizeLinear");
  const TestCase *uint8_case = nullptr;
  const TestCase *int8_case = nullptr;
  const TestCase *uint16_case = nullptr;
  const TestCase *int16_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_quantizelinear") {
      uint8_case = &c;
    } else if (c.name == "test_cc_quantizelinear_int8") {
      int8_case = &c;
    } else if (c.name == "test_quantizelinear_uint16") {
      uint16_case = &c;
    } else if (c.name == "test_quantizelinear_int16") {
      int16_case = &c;
    }
  }
  ASSERT_NE(uint8_case, nullptr);
  ASSERT_NE(int8_case, nullptr);
  ASSERT_NE(uint16_case, nullptr);
  ASSERT_NE(int16_case, nullptr);

  // Default UINT8 case: two inputs (x, y_scale), single UINT8 output.
  {
    const GraphProto &graph = uint8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "QuantizeLinear");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(uint8_case->data_sets.size(), 1u);
    const auto &ds = uint8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::UINT8));
    const std::vector<int64_t> expected_shape = {6};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    EXPECT_EQ(static_cast<int>(ds.outputs[0].data[0]), 0);
    EXPECT_EQ(static_cast<int>(ds.outputs[0].data[3]), 255);
  }

  // INT8 case: three inputs (x, y_scale, y_zero_point), single INT8 output.
  {
    const GraphProto &graph = int8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(int8_case->data_sets.size(), 1u);
    const auto &ds = int8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT8));
    const int8_t *py = reinterpret_cast<const int8_t *>(ds.outputs[0].data.data());
    EXPECT_EQ(static_cast<int>(py[0]), -10);
    EXPECT_EQ(static_cast<int>(py[3]), 127);
  }

  // Upstream UINT16 case.
  {
    ASSERT_EQ(uint16_case->data_sets.size(), 1u);
    const auto &ds = uint16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(TensorProto::DataType::UINT16));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::UINT16));
    const uint16_t *py = reinterpret_cast<const uint16_t *>(ds.outputs[0].data.data());
    EXPECT_EQ(py[0], static_cast<uint16_t>(32767));
    EXPECT_EQ(py[3], static_cast<uint16_t>(65535));
  }

  // Upstream INT16 case.
  {
    ASSERT_EQ(int16_case->data_sets.size(), 1u);
    const auto &ds = int16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(TensorProto::DataType::INT16));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT16));
    const int16_t *py = reinterpret_cast<const int16_t *>(ds.outputs[0].data.data());
    EXPECT_EQ(py[0], static_cast<int16_t>(-1024));
    EXPECT_EQ(py[3], std::numeric_limits<int16_t>::min());
  }
}

TEST(BackendTestCase, DequantizeLinearCaseIsPresent) {
  auto cases = CollectTestCases("DequantizeLinear");
  const TestCase *uint8_case = nullptr;
  const TestCase *int8_case = nullptr;
  const TestCase *upstream_uint8_case = nullptr;
  const TestCase *upstream_uint16_case = nullptr;
  const TestCase *upstream_int16_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_dequantizelinear") {
      uint8_case = &c;
    } else if (c.name == "test_cc_dequantizelinear_int8") {
      int8_case = &c;
    } else if (c.name == "test_dequantizelinear") {
      upstream_uint8_case = &c;
    } else if (c.name == "test_dequantizelinear_uint16") {
      upstream_uint16_case = &c;
    } else if (c.name == "test_dequantizelinear_int16") {
      upstream_int16_case = &c;
    }
  }
  ASSERT_NE(uint8_case, nullptr);
  ASSERT_NE(int8_case, nullptr);
  ASSERT_NE(upstream_uint8_case, nullptr);
  ASSERT_NE(upstream_uint16_case, nullptr);
  ASSERT_NE(upstream_int16_case, nullptr);

  // Default UINT8 case: two inputs (x, x_scale), single FLOAT output.
  {
    const GraphProto &graph = uint8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "DequantizeLinear");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(uint8_case->data_sets.size(), 1u);
    const auto &ds = uint8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    const std::vector<int64_t> expected_shape = {4};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[3], 510.0f);
  }

  // INT8 case: three inputs (x, x_scale, x_zero_point), single FLOAT output.
  {
    const GraphProto &graph = int8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(int8_case->data_sets.size(), 1u);
    const auto &ds = int8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[3], 274.0f);
  }

  // Upstream UINT8 case (test_dequantizelinear): zero_point=128, expected
  // outputs [-256, -250, 0, 254].
  {
    ASSERT_EQ(upstream_uint8_case->data_sets.size(), 1u);
    const auto &ds = upstream_uint8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(TensorProto::DataType::UINT8));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -256.0f);
    EXPECT_FLOAT_EQ(py[1], -250.0f);
    EXPECT_FLOAT_EQ(py[2], 0.0f);
    EXPECT_FLOAT_EQ(py[3], 254.0f);
  }

  // Upstream UINT16 case.
  {
    ASSERT_EQ(upstream_uint16_case->data_sets.size(), 1u);
    const auto &ds = upstream_uint16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::UINT16));
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(TensorProto::DataType::UINT16));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -5534.0f);
    EXPECT_FLOAT_EQ(py[3], 466.0f);
  }

  // Upstream INT16 case.
  {
    ASSERT_EQ(upstream_int16_case->data_sets.size(), 1u);
    const auto &ds = upstream_int16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT16));
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(TensorProto::DataType::INT16));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 1448.0f);
    EXPECT_FLOAT_EQ(py[3], 4588.0f);
  }
}

} // namespace Test
