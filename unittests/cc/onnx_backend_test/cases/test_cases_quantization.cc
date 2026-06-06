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
  const TestCase *upstream_uint8_case = nullptr;
  const TestCase *axis_case = nullptr;
  const TestCase *e4m3fn_case = nullptr;
  const TestCase *e5m2_case = nullptr;
  const TestCase *uint4_case = nullptr;
  const TestCase *int4_case = nullptr;
  const TestCase *uint2_case = nullptr;
  const TestCase *int2_case = nullptr;
  const TestCase *float4e2m1_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_quantizelinear") {
      uint8_case = &c;
    } else if (c.name == "test_cc_quantizelinear_int8") {
      int8_case = &c;
    } else if (c.name == "test_quantizelinear_uint16") {
      uint16_case = &c;
    } else if (c.name == "test_quantizelinear_int16") {
      int16_case = &c;
    } else if (c.name == "test_quantizelinear") {
      upstream_uint8_case = &c;
    } else if (c.name == "test_quantizelinear_axis") {
      axis_case = &c;
    } else if (c.name == "test_quantizelinear_e4m3fn") {
      e4m3fn_case = &c;
    } else if (c.name == "test_quantizelinear_e5m2") {
      e5m2_case = &c;
    } else if (c.name == "test_quantizelinear_uint4") {
      uint4_case = &c;
    } else if (c.name == "test_quantizelinear_int4") {
      int4_case = &c;
    } else if (c.name == "test_quantizelinear_uint2") {
      uint2_case = &c;
    } else if (c.name == "test_quantizelinear_int2") {
      int2_case = &c;
    } else if (c.name == "test_quantizelinear_float4e2m1") {
      float4e2m1_case = &c;
    }
  }
  ASSERT_NE(uint8_case, nullptr);
  ASSERT_NE(int8_case, nullptr);
  ASSERT_NE(uint16_case, nullptr);
  ASSERT_NE(int16_case, nullptr);
  ASSERT_NE(upstream_uint8_case, nullptr);
  ASSERT_NE(axis_case, nullptr);
  ASSERT_NE(e4m3fn_case, nullptr);
  ASSERT_NE(e5m2_case, nullptr);
  ASSERT_NE(uint4_case, nullptr);
  ASSERT_NE(int4_case, nullptr);
  ASSERT_NE(uint2_case, nullptr);
  ASSERT_NE(int2_case, nullptr);
  ASSERT_NE(float4e2m1_case, nullptr);

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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT8));
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
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT16));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT16));
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
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT16));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT16));
    const int16_t *py = reinterpret_cast<const int16_t *>(ds.outputs[0].data.data());
    EXPECT_EQ(py[0], static_cast<int16_t>(-1024));
    EXPECT_EQ(py[3], std::numeric_limits<int16_t>::min());
  }

  // Upstream default UINT8 case (test_quantizelinear): y_zero_point=128.
  {
    ASSERT_EQ(upstream_uint8_case->data_sets.size(), 1u);
    const auto &ds = upstream_uint8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
    EXPECT_EQ(ds.outputs[0].data[0], 128u);
    EXPECT_EQ(ds.outputs[0].data[3], 255u);
    EXPECT_EQ(ds.outputs[0].data[5], 0u);
  }

  // Upstream per-axis UINT8 case (test_quantizelinear_axis).
  {
    const GraphProto &graph = axis_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &n = graph.ref_node()[0];
    ASSERT_EQ(n.ref_attribute().size(), 1u);
    const auto &attr = n.ref_attribute()[0];
    const std::string attr_name(attr.ref_name().data(), attr.ref_name().size());
    EXPECT_EQ(attr_name, "axis");
    EXPECT_EQ(attr.i(), static_cast<int64_t>(1));

    ASSERT_EQ(axis_case->data_sets.size(), 1u);
    const auto &ds = axis_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> scale_shape = {3};
    EXPECT_EQ(ds.inputs[1].shape, scale_shape);
    EXPECT_EQ(ds.inputs[2].shape, scale_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
    EXPECT_EQ(ds.outputs[0].data[0], 3u);
    EXPECT_EQ(ds.outputs[0].data[1], 89u);
    EXPECT_EQ(ds.outputs[0].data[12], 245u);
    EXPECT_EQ(ds.outputs[0].data[17], 102u);
  }

  // Upstream FLOAT8E4M3FN case (test_quantizelinear_e4m3fn): inputs
  // (x, y_scale, y_zero_point) and a 5-element FLOAT8E4M3FN output.
  {
    ASSERT_EQ(e4m3fn_case->data_sets.size(), 1u);
    const auto &ds = e4m3fn_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    EXPECT_EQ(ds.outputs[0].data.size(), 5u);
  }

  // Upstream FLOAT8E5M2 case (test_quantizelinear_e5m2).
  {
    ASSERT_EQ(e5m2_case->data_sets.size(), 1u);
    const auto &ds = e5m2_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E5M2));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E5M2));
    EXPECT_EQ(ds.outputs[0].data.size(), 5u);
  }

  // Sub-byte upstream cases: UINT4/INT4 are packed two nibbles per byte
  // (low nibble first), UINT2/INT2 four pairs per byte, FLOAT4E2M1 like
  // UINT4. The 3x4 output therefore fits in 6 bytes (4-bit) or 3 bytes
  // (2-bit).
  for (auto *c : {uint4_case, int4_case, float4e2m1_case}) {
    ASSERT_EQ(c->data_sets.size(), 1u);
    const auto &ds = c->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> shape = {3, 4};
    EXPECT_EQ(ds.outputs[0].shape, shape);
    EXPECT_EQ(ds.outputs[0].data.size(), 6u);
  }
  for (auto *c : {uint2_case, int2_case}) {
    ASSERT_EQ(c->data_sets.size(), 1u);
    const auto &ds = c->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> shape = {3, 4};
    EXPECT_EQ(ds.outputs[0].shape, shape);
    EXPECT_EQ(ds.outputs[0].data.size(), 3u);
  }

  // Spot-check the exact packed bytes for UINT4 and INT4 against the
  // upstream expected values.
  {
    const auto &ds = uint4_case->data_sets[0];
    // Expected nibbles: 1,2,3,5,0,0,3,4,4,5,5,11 →
    // bytes (low nibble first): 0x21, 0x53, 0x00, 0x43, 0x54, 0xB5.
    const std::vector<uint8_t> expected = {0x21, 0x53, 0x00, 0x43, 0x54, 0xB5};
    EXPECT_EQ(ds.outputs[0].data, expected);
  }
  {
    const auto &ds = int4_case->data_sets[0];
    // Expected nibbles: 1,2,3,5,-8,-6,3,4,4,5,5,7 →
    // bytes: 0x21, 0x53, 0xA8, 0x43, 0x54, 0x75.
    const std::vector<uint8_t> expected = {0x21, 0x53, 0xA8, 0x43, 0x54, 0x75};
    EXPECT_EQ(ds.outputs[0].data, expected);
  }
}

TEST(BackendTestCase, DequantizeLinearCaseIsPresent) {
  auto cases = CollectTestCases("DequantizeLinear");
  const TestCase *uint8_case = nullptr;
  const TestCase *int8_case = nullptr;
  const TestCase *upstream_uint8_case = nullptr;
  const TestCase *upstream_uint16_case = nullptr;
  const TestCase *upstream_int16_case = nullptr;
  const TestCase *upstream_e4m3fn_case = nullptr;
  const TestCase *upstream_e5m2_case = nullptr;
  const TestCase *upstream_e4m3fn_zp_case = nullptr;
  const TestCase *axis_case = nullptr;
  const TestCase *blocked_case = nullptr;
  const TestCase *e4m3fn_float16_case = nullptr;
  const TestCase *uint4_case = nullptr;
  const TestCase *int4_case = nullptr;
  const TestCase *uint2_case = nullptr;
  const TestCase *int2_case = nullptr;
  const TestCase *float4e2m1_case = nullptr;
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
    } else if (c.name == "test_dequantizelinear_e4m3fn") {
      upstream_e4m3fn_case = &c;
    } else if (c.name == "test_dequantizelinear_e5m2") {
      upstream_e5m2_case = &c;
    } else if (c.name == "test_dequantizelinear_e4m3fn_zero_point") {
      upstream_e4m3fn_zp_case = &c;
    } else if (c.name == "test_dequantizelinear_axis") {
      axis_case = &c;
    } else if (c.name == "test_dequantizelinear_blocked") {
      blocked_case = &c;
    } else if (c.name == "test_dequantizelinear_e4m3fn_float16") {
      e4m3fn_float16_case = &c;
    } else if (c.name == "test_dequantizelinear_uint4") {
      uint4_case = &c;
    } else if (c.name == "test_dequantizelinear_int4") {
      int4_case = &c;
    } else if (c.name == "test_dequantizelinear_uint2") {
      uint2_case = &c;
    } else if (c.name == "test_dequantizelinear_int2") {
      int2_case = &c;
    } else if (c.name == "test_dequantizelinear_float4e2m1") {
      float4e2m1_case = &c;
    }
  }
  ASSERT_NE(uint8_case, nullptr);
  ASSERT_NE(int8_case, nullptr);
  ASSERT_NE(upstream_uint8_case, nullptr);
  ASSERT_NE(upstream_uint16_case, nullptr);
  ASSERT_NE(upstream_int16_case, nullptr);
  ASSERT_NE(upstream_e4m3fn_case, nullptr);
  ASSERT_NE(upstream_e5m2_case, nullptr);
  ASSERT_NE(upstream_e4m3fn_zp_case, nullptr);
  ASSERT_NE(axis_case, nullptr);
  ASSERT_NE(blocked_case, nullptr);
  ASSERT_NE(e4m3fn_float16_case, nullptr);
  ASSERT_NE(uint4_case, nullptr);
  ASSERT_NE(int4_case, nullptr);
  ASSERT_NE(uint2_case, nullptr);
  ASSERT_NE(int2_case, nullptr);
  ASSERT_NE(float4e2m1_case, nullptr);

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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
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
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT16));
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT16));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -5534.0f);
    EXPECT_FLOAT_EQ(py[3], 466.0f);
  }

  // Upstream INT16 case.
  {
    ASSERT_EQ(upstream_int16_case->data_sets.size(), 1u);
    const auto &ds = upstream_int16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT16));
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT16));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 1448.0f);
    EXPECT_FLOAT_EQ(py[3], 4588.0f);
  }

  // Upstream FLOAT8E4M3FN case (test_dequantizelinear_e4m3fn): two inputs
  // and expected outputs [0, 1, 2, 896, -208].
  {
    ASSERT_EQ(upstream_e4m3fn_case->data_sets.size(), 1u);
    const auto &ds = upstream_e4m3fn_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[1], 1.0f);
    EXPECT_FLOAT_EQ(py[2], 2.0f);
    EXPECT_FLOAT_EQ(py[3], 896.0f);
    EXPECT_FLOAT_EQ(py[4], -208.0f);
  }

  // Upstream FLOAT8E5M2 case (test_dequantizelinear_e5m2): two inputs and
  // expected outputs [0, 1, 2, 98304, -192].
  {
    ASSERT_EQ(upstream_e5m2_case->data_sets.size(), 1u);
    const auto &ds = upstream_e5m2_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E5M2));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[1], 1.0f);
    EXPECT_FLOAT_EQ(py[2], 2.0f);
    EXPECT_FLOAT_EQ(py[3], 98304.0f);
    EXPECT_FLOAT_EQ(py[4], -192.0f);
  }

  // Upstream FLOAT8E4M3FN with explicit FLOAT8E4M3FN zero_point (1-D shape
  // [1]) case (test_dequantizelinear_e4m3fn_zero_point): three inputs and
  // expected outputs [0, 1, 2, 896, -208] (same as the no-zero-point case
  // because zero_point == 0).
  {
    ASSERT_EQ(upstream_e4m3fn_zp_case->data_sets.size(), 1u);
    const auto &ds = upstream_e4m3fn_zp_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    EXPECT_EQ(ds.inputs[2].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    const std::vector<int64_t> zp_shape = {1};
    EXPECT_EQ(ds.inputs[2].shape, zp_shape);
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[3], 896.0f);
    EXPECT_FLOAT_EQ(py[4], -208.0f);
  }

  // Upstream per-axis UINT8 case (test_dequantizelinear_axis): per-channel
  // form with 3-element x_scale and x_zero_point. The upstream node relies on
  // the default ``axis`` (1), so the saved NodeProto has no explicit ``axis``
  // attribute; the per-channel layout is inferred from the scale shape.
  {
    const GraphProto &graph = axis_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &n = graph.ref_node()[0];
    EXPECT_EQ(n.ref_attribute().size(), 0u);

    ASSERT_EQ(axis_case->data_sets.size(), 1u);
    const auto &ds = axis_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> scale_shape = {3};
    EXPECT_EQ(ds.inputs[1].shape, scale_shape);
    EXPECT_EQ(ds.inputs[2].shape, scale_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    const std::vector<int64_t> y_shape = {1, 3, 3, 2};
    EXPECT_EQ(ds.outputs[0].shape, y_shape);
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -162.0f);
    EXPECT_FLOAT_EQ(py[1], 10.0f);
    EXPECT_FLOAT_EQ(py[12], 245.0f);
    EXPECT_FLOAT_EQ(py[17], -470.0f);
  }

  // Upstream blocked UINT8 case (test_dequantizelinear_blocked): axis=1,
  // block_size=2.
  {
    const GraphProto &graph = blocked_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &n = graph.ref_node()[0];
    ASSERT_EQ(n.ref_attribute().size(), 2u);
    bool saw_axis = false;
    bool saw_block_size = false;
    for (const auto &attr : n.ref_attribute()) {
      const std::string attr_name(attr.ref_name().data(), attr.ref_name().size());
      if (attr_name == "axis") {
        EXPECT_EQ(attr.i(), static_cast<int64_t>(1));
        saw_axis = true;
      } else if (attr_name == "block_size") {
        EXPECT_EQ(attr.i(), static_cast<int64_t>(2));
        saw_block_size = true;
      }
    }
    EXPECT_TRUE(saw_axis);
    EXPECT_TRUE(saw_block_size);

    ASSERT_EQ(blocked_case->data_sets.size(), 1u);
    const auto &ds = blocked_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> y_shape = {1, 4, 3, 2};
    EXPECT_EQ(ds.outputs[0].shape, y_shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 6.0f);
    EXPECT_FLOAT_EQ(py[1], 178.0f);
    EXPECT_FLOAT_EQ(py[18], 1210.0f);
    EXPECT_FLOAT_EQ(py[23], 200.0f);
  }

  // Upstream FLOAT8E4M3FN -> FLOAT16 case (test_dequantizelinear_e4m3fn_float16):
  // axis=0, scalar FLOAT16 x_scale, FLOAT16 output.
  {
    const GraphProto &graph = e4m3fn_float16_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &n = graph.ref_node()[0];
    ASSERT_EQ(n.ref_attribute().size(), 1u);
    const auto &attr = n.ref_attribute()[0];
    const std::string attr_name(attr.ref_name().data(), attr.ref_name().size());
    EXPECT_EQ(attr_name, "axis");
    EXPECT_EQ(attr.i(), static_cast<int64_t>(0));

    ASSERT_EQ(e4m3fn_float16_case->data_sets.size(), 1u);
    const auto &ds = e4m3fn_float16_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN));
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT16));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT16));
    EXPECT_EQ(ds.inputs[1].shape, std::vector<int64_t>{});
    EXPECT_EQ(ds.outputs[0].data.size(), 5u * sizeof(uint16_t));
  }

  // Sub-byte upstream cases (UINT4/INT4/FLOAT4E2M1 pack two values per byte,
  // UINT2/INT2 pack four values per byte). All use axis=0, scalar x_scale and
  // a 1-element x_zero_point.
  for (auto *c : {uint4_case, int4_case, float4e2m1_case}) {
    ASSERT_EQ(c->data_sets.size(), 1u);
    const auto &ds = c->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> shape = {5};
    EXPECT_EQ(ds.outputs[0].shape, shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].data.size(), 5u * sizeof(float));
  }
  for (auto *c : {uint2_case, int2_case}) {
    ASSERT_EQ(c->data_sets.size(), 1u);
    const auto &ds = c->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    const std::vector<int64_t> shape = {4};
    EXPECT_EQ(ds.outputs[0].shape, shape);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].data.size(), 4u * sizeof(float));
  }

  // Spot-check the exact dequantized float outputs for the sub-byte cases
  // against the upstream values from onnx.backend.test.case.node.dequantizelinear.
  {
    const auto &ds = uint4_case->data_sets[0];
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT4));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -2.0f);
    EXPECT_FLOAT_EQ(py[1], 0.0f);
    EXPECT_FLOAT_EQ(py[2], 12.0f);
    EXPECT_FLOAT_EQ(py[3], 18.0f);
    EXPECT_FLOAT_EQ(py[4], 28.0f);
  }
  {
    const auto &ds = int4_case->data_sets[0];
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT4));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -2.0f);
    EXPECT_FLOAT_EQ(py[1], 0.0f);
    EXPECT_FLOAT_EQ(py[2], 12.0f);
    EXPECT_FLOAT_EQ(py[3], -10.0f);
    EXPECT_FLOAT_EQ(py[4], -18.0f);
  }
  {
    const auto &ds = uint2_case->data_sets[0];
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT2));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -2.0f);
    EXPECT_FLOAT_EQ(py[1], 0.0f);
    EXPECT_FLOAT_EQ(py[2], 2.0f);
    EXPECT_FLOAT_EQ(py[3], 4.0f);
  }
  {
    const auto &ds = int2_case->data_sets[0];
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT2));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], -2.0f);
    EXPECT_FLOAT_EQ(py[1], 0.0f);
    EXPECT_FLOAT_EQ(py[2], -4.0f);
    EXPECT_FLOAT_EQ(py[3], -6.0f);
  }
  {
    const auto &ds = float4e2m1_case->data_sets[0];
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT4E2M1));
    const float *py = reinterpret_cast<const float *>(ds.outputs[0].data.data());
    EXPECT_FLOAT_EQ(py[0], 0.0f);
    EXPECT_FLOAT_EQ(py[1], 2.0f);
    EXPECT_FLOAT_EQ(py[2], -2.0f);
    EXPECT_FLOAT_EQ(py[3], 3.0f);
    EXPECT_FLOAT_EQ(py[4], -8.0f);
  }
}

} // namespace Test
