// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

void AddStringAttr(NodeProto &node, const char *name, const std::string &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s(value);
}

void AddIntAttr(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

} // namespace

// ---------------------------------------------------------------------------
// CastMap — converts a ``map(int64, X)`` input into a 1-D output tensor whose
// length is either the number of keys (``DENSE``) or ``max_map`` (``SPARSE``).
// The output element type is controlled by the ``cast_to`` attribute. Mirrors
// the upstream ONNX ``ai.onnx.ml::CastMap`` operator (since opset 1).
// ---------------------------------------------------------------------------
void RegisterCastMapCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::CastMap cast_map{ctx};

  // DENSE map(int64, float) -> tensor(float). Keys are not sorted on input;
  // the operator must sort them ascending in the output.
  //
  // The map is represented at runtime as two tensor inputs: "x_keys" (INT64)
  // and "x_values" (FLOAT).
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x_keys");
    node.add_input("x_values");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_FLOAT");
    AddStringAttr(node, "map_form", "DENSE");

    const std::vector<int64_t> keys{2, 0, 1};
    const std::vector<float> values{2.5f, 0.5f, 1.5f};
    Tensor x_keys = Tensor::FromInt64("", {static_cast<int64_t>(keys.size())}, keys);
    Tensor x_values = Tensor::FromFloat("", {static_cast<int64_t>(values.size())}, values);
    Tensor y = cast_map.operator()<float, float>(keys, values, "TO_FLOAT", "DENSE", 0);

    Expect(node, {x_keys, x_values}, {y}, "test_cc_cast_map_int64_float_dense",
           {default_opset, opset}, "backend-test", registry);
  }

  // SPARSE map(int64, float) -> tensor(float). Missing positions are zero.
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x_keys");
    node.add_input("x_values");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_FLOAT");
    AddStringAttr(node, "map_form", "SPARSE");
    AddIntAttr(node, "max_map", 5);

    const std::vector<int64_t> keys{1, 3};
    const std::vector<float> values{10.0f, 30.0f};
    Tensor x_keys = Tensor::FromInt64("", {static_cast<int64_t>(keys.size())}, keys);
    Tensor x_values = Tensor::FromFloat("", {static_cast<int64_t>(values.size())}, values);
    Tensor y = cast_map.operator()<float, float>(keys, values, "TO_FLOAT", "SPARSE", 5);

    Expect(node, {x_keys, x_values}, {y}, "test_cc_cast_map_int64_float_sparse",
           {default_opset, opset}, "backend-test", registry);
  }

  // DENSE map(int64, string) -> tensor(string).
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x_keys");
    node.add_input("x_values");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_STRING");
    AddStringAttr(node, "map_form", "DENSE");

    const std::vector<int64_t> keys{1, 0};
    const std::vector<std::string> values{"b", "a"};
    Tensor x_keys = Tensor::FromInt64("", {static_cast<int64_t>(keys.size())}, keys);
    Tensor x_values = Tensor::FromStrings("", {static_cast<int64_t>(values.size())}, values);
    Tensor y = cast_map.operator()<std::string, std::string>(keys, values, "TO_STRING", "DENSE", 0);

    Expect(node, {x_keys, x_values}, {y}, "test_cc_cast_map_int64_string_dense",
           {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
