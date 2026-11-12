// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
namespace {

void TensorInfo(ValueInfoProto &info, const char *name, int32_t dtype) {
  info.set_name(name);
  auto *type = info.mutable_type()->mutable_tensor_type();
  type->set_elem_type(dtype);
  type->mutable_shape()->add_dim()->set_dim_value(3);
}

void RegisterCodecCases(std::vector<TestCase> &registry, bool quantize, TestMode mode) {
  if (mode == TestMode::BENCHMARK)
    return;
  for (auto format : {QuantizationFormat::kInt4, QuantizationFormat::kNf4,
                      QuantizationFormat::kTernary, QuantizationFormat::kTiledFloat}) {
    const std::string name = std::string(quantize ? "test_cc_quantize_" : "test_cc_dequantize_") +
                             std::string(QuantizationFormatName(format));
    TestCase test(name, name, quantize ? TestCaseKind::MODEL : TestCaseKind::NODE,
                  TestCaseTag::AI_RT);
    test.build = [format, quantize, name](bool) {
      BuiltCase result;
      auto &model = result.model;
      model.set_ir_version(10);
      model.set_producer_name("backend-test");
      model.add_opset_import()->set_version(21);
      auto *opset = model.add_opset_import();
      opset->set_domain("ai.rt");
      opset->set_version(1);
      auto *graph = model.mutable_graph();
      graph->set_name(name);
      auto plan = MakeQuantizationPlan(format, 3);
      Tensor input = Tensor::FromFloat("X", {3}, {-1, 0, 1});
      DataSet data;
      if (quantize) {
        TensorInfo(*graph->add_input(), "X", TensorProto::FLOAT);
        auto *node = graph->add_node();
        node->set_domain("ai.rt");
        node->set_op_type("Quantize");
        node->add_input("X");
        node->add_output("Q");
        auto *type = node->add_attribute();
        type->set_name("type");
        type->set_type(AttributeProto::TYPE_PROTO);
        *type->mutable_tp()->mutable_struct_type() = MakeQuantizationType(plan);
        auto *intermediate = graph->add_value_info();
        intermediate->set_name("Q");
        *intermediate->mutable_type() = type->tp();
        data.inputs.push_back(input);
      } else {
        auto encoded = QuantizeTensor(input, plan).Encoded();
        encoded.set_name("Q");
        *graph->add_extended_initializer()->mutable_encoded_value() = std::move(encoded);
      }
      auto *node = graph->add_node();
      node->set_domain("ai.rt");
      node->set_op_type("Dequantize");
      node->add_input("Q");
      node->add_output("Y");
      AddAttribute<int64_t>(*node, "dtype", TensorProto::DOUBLE);
      TensorInfo(*graph->add_output(), "Y", TensorProto::DOUBLE);
      data.outputs.push_back(Tensor::FromDouble("Y", {3}, {-1, 0, 1}));
      result.data_sets.push_back(std::move(data));
      return result;
    };
    registry.push_back(std::move(test));
  }
}

void RegisterBitsAndBytesCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK)
    return;
  for (const std::string format : {"nf4", "fp4", "int8"}) {
    for (bool nested : {false, true}) {
      if (format == "int8" && nested)
        continue;
      const std::string name =
          "test_cc_dequantize_bitsandbytes_" + format + (nested ? "_nested" : "");
      TestCase test(name, name, TestCaseKind::NODE, TestCaseTag::AI_RT);
      test.build = [format, nested, name](bool) {
        BuiltCase result;
        auto &model = result.model;
        model.set_ir_version(10);
        model.set_producer_name("backend-test");
        model.add_opset_import()->set_version(21);
        auto *opset = model.add_opset_import();
        opset->set_domain("ai.rt");
        opset->set_version(1);
        auto *graph = model.mutable_graph();
        graph->set_name(name);
        BitsAndBytesWeights weights;
        std::vector<float> expected(33);
        if (format == "int8") {
          constexpr int codes[] = {-128, -127, -1, 0, 1, 127};
          std::vector<uint8_t> bytes(33);
          for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(codes[i % 6]);
            expected[i] = codes[i % 6] * (i < 11 ? 1.f : i < 22 ? 2.f : 0.5f);
          }
          weights.tensors.emplace("weight",
                                  Tensor("weight", TensorProto::INT8, {3, 11}, std::move(bytes)));
          weights.tensors.emplace("weight_scb",
                                  Tensor::FromFloat("weight_scb", {3}, {127, 254, 63.5f}));
        } else {
          // Pinned bitsandbytes 0.50.2 get_4bit_type tables, not decoder-produced references.
          const std::vector<float> table =
              format == "nf4"
                  ? std::vector<float>{-1.f,
                                       -0.6961928009986877f,
                                       -0.5250730514526367f,
                                       -0.39491748809814453f,
                                       -0.28444138169288635f,
                                       -0.18477343022823334f,
                                       -0.09105003625154495f,
                                       0.f,
                                       0.07958029955625534f,
                                       0.16093020141124725f,
                                       0.24611230194568634f,
                                       0.33791524171829224f,
                                       0.44070982933044434f,
                                       0.5626170039176941f,
                                       0.7229568362236023f,
                                       1.f}
                  : std::vector<float>{0,        1.f / 192, 2.f / 3,  1,          1.f / 3,  0.5f,
                                       1.f / 6,  0.25f,     0,        -1.f / 192, -2.f / 3, -1,
                                       -1.f / 3, -0.5f,     -1.f / 6, -0.25f};
          std::vector<uint8_t> bytes(17);
          for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<uint8_t>(((2 * i % 16) << 4) | ((2 * i + 1) % 16));
          bytes.back() = 0x0d;
          weights.tensors.emplace("weight",
                                  Tensor("weight", TensorProto::UINT8, {17, 1}, std::move(bytes)));
          weights.tensors.emplace("weight_quant_map",
                                  Tensor::FromFloat("weight_quant_map", {16}, table));
          std::string json = "{\"quant_type\":\"" + format +
                             "\",\"blocksize\":32,\"dtype\":\"float32\",\"shape\":[3,11]";
          if (nested) {
            weights.tensors.emplace("weight_absmax", Tensor("weight_absmax", TensorProto::UINT8,
                                                            {2}, std::vector<uint8_t>{0, 255}));
            weights.tensors.emplace("weight_nested_absmax",
                                    Tensor::FromFloat("weight_nested_absmax", {1}, {0.5f}));
            std::vector<float> nested_table(256, 0.f);
            nested_table.front() = -1.f;
            nested_table.back() = 1.f;
            weights.tensors.emplace(
                "weight_nested_quant_map",
                Tensor::FromFloat("weight_nested_quant_map", {256}, nested_table));
            json += ",\"nested_blocksize\":256,\"nested_dtype\":\"float32\",\"nested_offset\":2.0";
          } else
            weights.tensors.emplace("weight_absmax",
                                    Tensor::FromFloat("weight_absmax", {2}, {2.f, 0.5f}));
          json += "}";
          weights.tensors.emplace("weight_quant_state",
                                  Tensor("weight_quant_state", TensorProto::UINT8,
                                         {static_cast<int64_t>(json.size())},
                                         std::vector<uint8_t>(json.begin(), json.end())));
          for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = table[i % 16] * (i < 32 ? (nested ? 1.5f : 2.f) : (nested ? 2.5f : 0.5f));
        }
        auto encoded = ImportBitsAndBytesWeights(weights);
        encoded.set_name("Q");
        *graph->add_extended_initializer()->mutable_encoded_value() = std::move(encoded);
        auto *node = graph->add_node();
        node->set_domain("ai.rt");
        node->set_op_type("Dequantize");
        node->add_input("Q");
        node->add_output("Y");
        AddAttribute<int64_t>(*node, "dtype", TensorProto::FLOAT);
        auto *output = graph->add_output();
        output->set_name("Y");
        auto *type = output->mutable_type()->mutable_tensor_type();
        type->set_elem_type(TensorProto::FLOAT);
        type->mutable_shape()->add_dim()->set_dim_value(3);
        type->mutable_shape()->add_dim()->set_dim_value(11);
        DataSet data;
        data.outputs.push_back(Tensor::FromFloat("Y", {3, 11}, expected));
        result.data_sets.push_back(std::move(data));
        return result;
      };
      registry.push_back(std::move(test));
    }
  }
}

} // namespace

void RegisterQuantizeCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterCodecCases(registry, true, mode);
}

void RegisterDequantizeCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterCodecCases(registry, false, mode);
  RegisterBitsAndBytesCases(registry, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
