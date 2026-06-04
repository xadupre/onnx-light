// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 13;

// Adds a ``type`` TYPE_PROTO attribute wrapping
// ``Optional<Tensor<elem_type, shape>>`` onto ``node`` (used by the
// ``Optional`` node we chain in front of ``OptionalGetElement`` /
// ``OptionalHasElement``).
void AddOptionalOfTensorTypeAttr(NodeProto &node, int32_t elem_type,
                                 const std::vector<int64_t> &shape) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Optional *opt = tp->add_optional_type();
  TypeProto *elem = opt->add_elem_type();
  TypeProto::Tensor *tt = elem->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sp = tt->add_shape();
  for (int64_t d : shape) {
    sp->add_dim()->set_dim_value(d);
  }
}

// Builds a single-input model and registers it as a test case named
// ``name``. The model graph contains two nodes:
//
//   1. ``Optional`` that wraps the graph input ``input`` (a tensor) into
//      an ``Optional<Tensor<elem_type, shape>>`` value named
//      ``opt_value``;
//   2. ``op_type`` (one of ``OptionalGetElement`` or
//      ``OptionalHasElement``) that consumes ``opt_value`` and produces
//      ``output``.
//
// ``expected_output`` is materialised as the expected DataSet output.
void RegisterOptionalInputCase(const std::string &name, const std::string &op_type,
                               const Tensor &input, const Tensor &expected_output,
                               const std::vector<int64_t> &input_shape, int32_t input_elem_type,
                               const OpsetId &opset, std::vector<TestCase> &registry) {
  TestCase tc(name, name);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Node 1: Optional input -> opt_value
  NodeProto *opt_node = graph->add_node();
  opt_node->set_op_type("Optional");
  opt_node->add_input("input");
  opt_node->add_output("opt_value");
  AddOptionalOfTensorTypeAttr(*opt_node, input_elem_type, input_shape);

  // Node 2: <op_type> opt_value -> output
  NodeProto *get_node = graph->add_node();
  get_node->set_op_type(op_type);
  get_node->add_input("opt_value");
  get_node->add_output("output");

  // Graph input/output value-infos (both are plain tensors).
  Tensor named_input = input;
  named_input.name = "input";
  FillValueInfo(named_input, *graph->add_input());
  Tensor named_output = expected_output;
  named_output.name = "output";
  FillValueInfo(named_output, *graph->add_output());

  DataSet ds;
  ds.inputs.push_back(named_input);
  ds.outputs.push_back(named_output);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

// Builds a single-node model that exercises the opset-18 tensor-input
// path of ``OptionalGetElement`` / ``OptionalHasElement``. The graph
// contains a single ``op_type`` node whose input is a plain tensor and
// whose output is ``expected_output``.
void RegisterTensorInputCase(const std::string &name, const std::string &op_type,
                             const Tensor &input, const Tensor &expected_output,
                             const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("input");
  node.add_output("output");
  Expect(node, {input}, {expected_output}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// OptionalGetElement — extracts the element from an optional input (since
// opset 15 in the ai.onnx domain). Since opset 18 the operator also accepts
// non-optional tensor or sequence inputs as a no-op. The cases registered
// here cover:
//
//   * test_cc_optional_get_element_optional_tensor: chains Optional →
//     OptionalGetElement on a FLOAT tensor (opset 15);
//   * test_cc_optional_get_element_tensor: single-node case with a plain
//     FLOAT tensor input (opset 18, passthrough).
// ---------------------------------------------------------------------------
void RegisterOptionalGetElementCases(std::vector<TestCase> &registry) {
  const std::vector<int64_t> shape = {2, 3};
  Tensor input = Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  // Opset 15: Optional → OptionalGetElement.
  {
    const OpsetId opset = DefaultOpset(15);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalGetElement(ctx)(input);
    RegisterOptionalInputCase("test_cc_optional_get_element_optional_tensor", "OptionalGetElement",
                              input, output, shape, static_cast<int32_t>(DataType::FLOAT), opset,
                              registry);
  }

  // Opset 18: single-node case with a plain tensor input (passthrough).
  {
    const OpsetId opset = DefaultOpset(18);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalGetElement(ctx)(input);
    RegisterTensorInputCase("test_cc_optional_get_element_tensor", "OptionalGetElement", input,
                            output, opset, registry);
  }
}

// ---------------------------------------------------------------------------
// OptionalHasElement — returns a scalar bool indicating whether the input
// optional contains an element (since opset 15 in the ai.onnx domain).
// Since opset 18 the input may also be a non-optional tensor/sequence and
// the input may be omitted entirely. The cases registered here cover:
//
//   * test_cc_optional_has_element_optional_input: chains Optional →
//     OptionalHasElement on a FLOAT tensor (opset 15), expecting true;
//   * test_cc_optional_has_element_tensor_input: single-node case with a
//     plain FLOAT tensor input (opset 18), expecting true.
// ---------------------------------------------------------------------------
void RegisterOptionalHasElementCases(std::vector<TestCase> &registry) {
  const std::vector<int64_t> shape = {2, 3};
  Tensor input = Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  // Opset 15: Optional → OptionalHasElement, expecting true.
  {
    const OpsetId opset = DefaultOpset(15);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalHasElement(ctx)(input);
    RegisterOptionalInputCase("test_cc_optional_has_element_optional_input", "OptionalHasElement",
                              input, output, shape, static_cast<int32_t>(DataType::FLOAT), opset,
                              registry);
  }

  // Opset 18: single-node case with a plain tensor input, expecting true.
  {
    const OpsetId opset = DefaultOpset(18);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalHasElement(ctx)(input);
    RegisterTensorInputCase("test_cc_optional_has_element_tensor_input", "OptionalHasElement",
                            input, output, opset, registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
