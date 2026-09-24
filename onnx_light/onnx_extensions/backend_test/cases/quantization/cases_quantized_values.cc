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
        *graph->add_encoded_initializer() = std::move(encoded);
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

} // namespace

void RegisterQuantizeCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterCodecCases(registry, true, mode);
}

void RegisterDequantizeCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterCodecCases(registry, false, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
