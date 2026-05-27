// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// LabelEncoder — y[i] = values_*[k] where keys_*[k] == x[i], else default_*
// (since opset 4 in the ``ai.onnx.ml`` domain). The following variants are
// registered:
//
//   * int64 keys → float values (the canonical "label id to float" mapping).
//   * float keys → int64 values (the inverse "float bucket to label" case).
//   * string keys → int64 values, with and without an explicit ``default_int64``
//     attribute — mirrors the upstream ONNX node tests
//     ``test_ai_onnx_ml_label_encoder_string_int`` and
//     ``test_ai_onnx_ml_label_encoder_string_int_no_default``
//     (see ``onnx/backend/test/case/node/ai_onnx_ml/label_encoder.py``).
//   * string keys → int16 values via ``keys_tensor``/``values_tensor``/
//     ``default_tensor`` tensor-typed attributes and via the
//     ``keys_strings``/``values_tensor`` mixed attributes — mirrors the upstream
//     ``test_ai_onnx_ml_label_encoder_tensor_mapping`` and
//     ``test_ai_onnx_ml_label_encoder_tensor_value_only_mapping`` tests.
// ---------------------------------------------------------------------------
void RegisterLabelEncoderCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 4);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::LabelEncoder label_encoder{kernel::KernelContext(opset)};

  // int64 -> float variant.
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> keys{0, 1, 2};
    const std::vector<float> values{0.5f, 1.5f, 2.5f};
    const float default_value = -1.0f;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_int64s");
    keys_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : keys) {
      keys_attr->ints().push_back(v);
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_floats");
    values_attr->set_type(AttributeProto::AttributeType::FLOATS);
    for (float v : values) {
      values_attr->floats().push_back(v);
    }

    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_float");
    default_attr->set_type(AttributeProto::AttributeType::FLOAT);
    default_attr->set_f(default_value);

    Tensor x = Tensor::FromInt64("", {4}, {0, 1, 2, 7});
    Tensor y = label_encoder.operator()<int64_t, float>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_cc_label_encoder_int64_to_float", {default_opset, opset},
           "backend-test", registry);
  }

  // float -> int64 variant.
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<float> keys{1.0f, 2.0f, 3.0f};
    const std::vector<int64_t> values{10, 20, 30};
    const int64_t default_value = -1;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_floats");
    keys_attr->set_type(AttributeProto::AttributeType::FLOATS);
    for (float v : keys) {
      keys_attr->floats().push_back(v);
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_int64s");
    values_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : values) {
      values_attr->ints().push_back(v);
    }

    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_int64");
    default_attr->set_type(AttributeProto::AttributeType::INT);
    default_attr->set_i(default_value);

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 9.0f});
    Tensor y = label_encoder.operator()<float, int64_t>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_cc_label_encoder_float_to_int64", {default_opset, opset},
           "backend-test", registry);
  }

  // string -> int64 with explicit default (upstream
  // ``test_ai_onnx_ml_label_encoder_string_int``).
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> keys{"a", "b", "c"};
    const std::vector<int64_t> values{0, 1, 2};
    const int64_t default_value = 42;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_strings");
    keys_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : keys) {
      keys_attr->strings().push_back(utils::String(v));
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_int64s");
    values_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : values) {
      values_attr->ints().push_back(v);
    }

    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_int64");
    default_attr->set_type(AttributeProto::AttributeType::INT);
    default_attr->set_i(default_value);

    Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
    Tensor y = label_encoder.operator()<std::string, int64_t>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_ai_onnx_ml_label_encoder_string_int", {default_opset, opset},
           "backend-test", registry);
  }

  // string -> int64 without ``default_int64`` (upstream
  // ``test_ai_onnx_ml_label_encoder_string_int_no_default``). The spec uses -1
  // as the implicit int64 default.
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> keys{"a", "b", "c"};
    const std::vector<int64_t> values{0, 1, 2};
    const int64_t default_value = -1;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_strings");
    keys_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : keys) {
      keys_attr->strings().push_back(utils::String(v));
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_int64s");
    values_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : values) {
      values_attr->ints().push_back(v);
    }

    Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
    Tensor y = label_encoder.operator()<std::string, int64_t>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_ai_onnx_ml_label_encoder_string_int_no_default",
           {default_opset, opset}, "backend-test", registry);
  }

  // string -> int16 via tensor-typed attributes (upstream
  // ``test_ai_onnx_ml_label_encoder_tensor_mapping``).
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> keys{"a", "b", "c"};
    const std::vector<int16_t> values{0, 1, 2};
    const int16_t default_value = 42;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_tensor");
    keys_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *kt = keys_attr->add_t();
    kt->set_data_type(TensorProto::DataType::STRING);
    kt->add_dims(static_cast<uint64_t>(keys.size()));
    for (const std::string &v : keys) {
      kt->string_data().push_back(utils::String(v));
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_tensor");
    values_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *vt = values_attr->add_t();
    vt->set_data_type(TensorProto::DataType::INT16);
    vt->add_dims(static_cast<uint64_t>(values.size()));
    {
      std::vector<uint8_t> raw(values.size() * sizeof(int16_t));
      std::memcpy(raw.data(), values.data(), raw.size());
      vt->set_raw_data(utils::ByteSpan(raw));
    }

    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_tensor");
    default_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *dt = default_attr->add_t();
    dt->set_data_type(TensorProto::DataType::INT16);
    dt->add_dims(1);
    {
      std::vector<uint8_t> raw(sizeof(int16_t));
      std::memcpy(raw.data(), &default_value, raw.size());
      dt->set_raw_data(utils::ByteSpan(raw));
    }

    Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
    Tensor y = label_encoder.operator()<std::string, int16_t>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_ai_onnx_ml_label_encoder_tensor_mapping",
           {default_opset, opset}, "backend-test", registry);
  }

  // string -> int16 via ``keys_strings`` + ``values_tensor``/``default_tensor``
  // (upstream ``test_ai_onnx_ml_label_encoder_tensor_value_only_mapping``).
  {
    NodeProto node;
    node.set_op_type("LabelEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> keys{"a", "b", "c"};
    const std::vector<int16_t> values{0, 1, 2};
    const int16_t default_value = 42;

    AttributeProto *keys_attr = node.add_attribute();
    keys_attr->set_name("keys_strings");
    keys_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : keys) {
      keys_attr->strings().push_back(utils::String(v));
    }

    AttributeProto *values_attr = node.add_attribute();
    values_attr->set_name("values_tensor");
    values_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *vt = values_attr->add_t();
    vt->set_data_type(TensorProto::DataType::INT16);
    vt->add_dims(static_cast<uint64_t>(values.size()));
    {
      std::vector<uint8_t> raw(values.size() * sizeof(int16_t));
      std::memcpy(raw.data(), values.data(), raw.size());
      vt->set_raw_data(utils::ByteSpan(raw));
    }

    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_tensor");
    default_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *dt = default_attr->add_t();
    dt->set_data_type(TensorProto::DataType::INT16);
    dt->add_dims(1);
    {
      std::vector<uint8_t> raw(sizeof(int16_t));
      std::memcpy(raw.data(), &default_value, raw.size());
      dt->set_raw_data(utils::ByteSpan(raw));
    }

    Tensor x = Tensor::FromStrings("", {5}, {"a", "b", "d", "c", "g"});
    Tensor y = label_encoder.operator()<std::string, int16_t>(x, keys, values, default_value);

    Expect(node, {x}, {y}, "test_ai_onnx_ml_label_encoder_tensor_value_only_mapping",
           {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
