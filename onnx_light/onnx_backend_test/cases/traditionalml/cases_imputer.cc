// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

void AddFloatsAttr(NodeProto &node, const char *name, const std::vector<float> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOATS);
  for (float v : values) {
    attr->add_floats(v);
  }
}

void AddIntsAttr(NodeProto &node, const char *name, const std::vector<int64_t> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : values) {
    attr->add_ints(v);
  }
}

void AddFloatAttr(NodeProto &node, const char *name, float value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(value);
}

void AddIntAttr(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

} // namespace

// ---------------------------------------------------------------------------
// Imputer — replaces elements matching a sentinel value with a replacement
// value, element-wise. The replacement may be broadcast (length 1) or
// per-feature (length == last dimension). Mirrors the upstream ONNX
// ``ai.onnx.ml::Imputer`` operator (since opset 1 in the ``ai.onnx.ml``
// domain).
// ---------------------------------------------------------------------------
void RegisterImputerCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Imputer imputer{ctx};

  // Float case: replace 0.0 with a per-feature replacement value.
  {
    NodeProto node;
    node.set_op_type("Imputer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const float replaced_value = 0.0f;
    const std::vector<float> imputed_values{1.0f, 2.0f, 3.0f};
    AddFloatAttr(node, "replaced_value_float", replaced_value);
    AddFloatsAttr(node, "imputed_value_floats", imputed_values);

    Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 0.0f, 5.0f, 0.0f, 6.0f});
    Tensor y = imputer.operator()<float>(x, imputed_values, replaced_value);

    Expect(node, {x}, {y}, "test_cc_imputer_float", {default_opset, opset}, "backend-test",
           registry);
  }

  // Float case: broadcast replacement (imputed_values length 1).
  {
    NodeProto node;
    node.set_op_type("Imputer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const float replaced_value = -1.0f;
    const std::vector<float> imputed_values{0.0f};
    AddFloatAttr(node, "replaced_value_float", replaced_value);
    AddFloatsAttr(node, "imputed_value_floats", imputed_values);

    Tensor x = Tensor::FromFloat("", {4}, {-1.0f, 2.0f, -1.0f, 4.0f});
    Tensor y = imputer.operator()<float>(x, imputed_values, replaced_value);

    Expect(node, {x}, {y}, "test_cc_imputer_float_broadcast", {default_opset, opset},
           "backend-test", registry);
  }

  // Float case: NaN replacement.
  {
    NodeProto node;
    node.set_op_type("Imputer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const float replaced_value = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> imputed_values{0.0f};
    AddFloatAttr(node, "replaced_value_float", replaced_value);
    AddFloatsAttr(node, "imputed_value_floats", imputed_values);

    Tensor x = Tensor::FromFloat("", {3}, {replaced_value, 2.0f, replaced_value});
    Tensor y = imputer.operator()<float>(x, imputed_values, replaced_value);

    Expect(node, {x}, {y}, "test_cc_imputer_float_nan", {default_opset, opset}, "backend-test",
           registry);
  }

  // Int64 case: replace 0 with per-feature imputed values.
  {
    NodeProto node;
    node.set_op_type("Imputer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const int64_t replaced_value = 0;
    const std::vector<int64_t> imputed_values{10, 20};
    AddIntAttr(node, "replaced_value_int64", replaced_value);
    AddIntsAttr(node, "imputed_value_int64s", imputed_values);

    Tensor x = Tensor::FromInt64("", {3, 2}, {0, 0, 1, 2, 0, 3});
    Tensor y = imputer.operator()<int64_t>(x, imputed_values, replaced_value);

    Expect(node, {x}, {y}, "test_cc_imputer_int64", {default_opset, opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
