// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/cases/image/include_image_cases.h"
#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"

#include <regex>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Default IR version stamped on test models. Matches ``Version::IR_VERSION``
// in ``onnx_lib/onnx-data.pb.h`` but is duplicated here so this library does
// not need to depend on ``lib_onnx_lib``.
constexpr int64_t kDefaultIrVersion = 13;

// Filters node.input/node.output, keeping only entries with a non-empty name.
std::vector<std::string> NonEmpty(const utils::RepeatedField<utils::String> &names) {
  std::vector<std::string> out;
  out.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const auto &s = names[i];
    if (s.size() != 0) {
      out.emplace_back(s.data(), s.size());
    }
  }
  return out;
}

} // namespace

void InitModel(ModelProto &model, int64_t ir_version, const std::vector<OpsetId> &opset_imports,
               const std::string &producer_name) {
  model.set_ir_version(ir_version);
  model.set_producer_name(producer_name);
  for (const auto &osid : opset_imports) {
    OperatorSetIdProto proto;
    proto.set_domain(osid.domain);
    proto.set_version(osid.version);
    model.add_opset_import(proto);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &shape) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (int64_t d : shape) {
    sh->add_dim()->set_dim_value(d);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendDataSet(TestCase &tc, std::vector<Tensor> inputs, std::vector<Tensor> outputs) {
  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
  tc.data_sets.emplace_back(std::move(ds));
}

void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OpsetId> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry, const std::string &tag) {
  const auto present_inputs = NonEmpty(node.ref_input());
  const auto present_outputs = NonEmpty(node.ref_output());
  EXT_ENFORCE_INVALID(
      present_inputs.size() == inputs.size(),
      "Expect: number of input tensors does not match the non-empty inputs of the node.");
  EXT_ENFORCE_INVALID(
      present_outputs.size() == outputs.size(),
      "Expect: number of output tensors does not match the non-empty outputs of the node.");

  TestCase tc(name, name, "node", tag);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, opset_imports, producer_name);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);
  graph->add_node(node);

  for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor tensor = inputs[i];
    tensor.name = present_inputs[i];
    FillValueInfo(tensor, *graph->add_input());
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor tensor = outputs[i];
    tensor.name = present_outputs[i];
    FillValueInfo(tensor, *graph->add_output());
  }

  DataSet ds;
  ds.inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor t = inputs[i];
    t.name = present_inputs[i];
    ds.inputs.emplace_back(std::move(t));
  }
  ds.outputs.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor t = outputs[i];
    t.name = present_outputs[i];
    ds.outputs.emplace_back(std::move(t));
  }
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry);
  }
}

std::vector<TestCase> CollectTestCases(const std::string &op_type) {
  std::vector<TestCase> registry;
  CollectControlflowTestCases(registry, op_type);
  CollectGeneratorTestCases(registry, op_type);
  CollectImageTestCases(registry, op_type);
  CollectLogicalTestCases(registry, op_type);
  CollectMathTestCases(registry, op_type);
  CollectNNTestCases(registry, op_type);
  CollectObjectDetectionTestCases(registry, op_type);
  CollectOptionalTestCases(registry, op_type);
  CollectPreviewTestCases(registry, op_type);
  CollectQuantizationTestCases(registry, op_type);
  CollectReductionTestCases(registry, op_type);
  CollectSequenceTestCases(registry, op_type);
  CollectTensorTestCases(registry, op_type);
  CollectTextTestCases(registry, op_type);
  CollectTraditionalMLTestCases(registry, op_type);
  CollectTrainingTestCases(registry, op_type);
  CollectShapeInferenceTestCases(registry, op_type);
  CollectEmptyShapeTestCases(registry, op_type);
  CollectNanInfTestCases(registry, op_type);
  return registry;
}

std::vector<TestCase> CollectTestCasesByName(const std::string &name_regex) {
  std::vector<TestCase> all_cases = CollectTestCases();
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
