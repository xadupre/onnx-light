// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"

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

void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OpsetId> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry) {
  const auto present_inputs = NonEmpty(node.ref_input());
  const auto present_outputs = NonEmpty(node.ref_output());
  if (present_inputs.size() != inputs.size()) {
    throw std::invalid_argument(
        "Expect: number of input tensors does not match the non-empty inputs of the node.");
  }
  if (present_outputs.size() != outputs.size()) {
    throw std::invalid_argument(
        "Expect: number of output tensors does not match the non-empty outputs of the node.");
  }

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "node";
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name(producer_name);
  for (const auto &osid : opset_imports) {
    OperatorSetIdProto proto;
    proto.set_domain(osid.domain);
    proto.set_version(osid.version);
    model.add_opset_import(proto);
  }

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

std::vector<TestCase> CollectTestCases() {
  std::vector<TestCase> registry;
  CollectControlflowTestCases(registry);
  CollectGeneratorTestCases(registry);
  CollectLogicalTestCases(registry);
  CollectMathTestCases(registry);
  CollectNNTestCases(registry);
  CollectOptionalTestCases(registry);
  CollectQuantizationTestCases(registry);
  CollectReductionTestCases(registry);
  CollectSequenceTestCases(registry);
  CollectTensorTestCases(registry);
  CollectTextTestCases(registry);
  CollectTraditionalMLTestCases(registry);
  return registry;
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
