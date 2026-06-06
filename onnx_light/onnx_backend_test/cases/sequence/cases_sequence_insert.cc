// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 13;

void PromoteOutputToSequenceType(std::vector<TestCase> &registry, int32_t elem_type,
                                 const std::vector<int64_t> &elem_shape) {
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &out_vi = *graph.mutable_output(0);
  TypeProto &out_tp = out_vi.ref_type();
  TypeProto::Sequence *out_seq = out_tp.add_sequence_type();
  TypeProto *out_elem = out_seq->add_elem_type();
  TypeProto::Tensor *out_tensor = out_elem->add_tensor_type();
  out_tensor->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto *out_shape = out_tensor->add_shape();
  for (int64_t d : elem_shape) {
    out_shape->add_dim()->set_dim_value(d);
  }
  out_tp.reset_tensor_type();
}

void RegisterSequenceInsertCase(const std::string &name, const std::vector<Tensor> &inputs,
                                const Tensor &tensor_to_insert,
                                const std::vector<int64_t> &elem_shape, bool has_position,
                                int64_t position, const OpsetId &opset,
                                std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  const Sequence seq = kernel::SequenceConstruct(ctx).AsSequence(inputs);
  Tensor position_tensor;
  const Tensor *pos_ptr = nullptr;
  if (has_position) {
    position_tensor = Tensor::FromInt64("position", {}, {position});
    pos_ptr = &position_tensor;
  }
  const Sequence out_seq = kernel::SequenceInsert(ctx)(seq, tensor_to_insert, pos_ptr);

  std::vector<Tensor> out_values(out_seq.values.begin(), out_seq.values.end());
  Tensor stacked = kernel::SequenceConstruct(ctx)(out_values);
  stacked.name = "output_sequence";

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

  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  for (const Tensor &t : inputs) {
    seq_node->add_input(t.name);
  }
  seq_node->add_output("input_seq");

  NodeProto *insert_node = graph->add_node();
  insert_node->set_op_type("SequenceInsert");
  insert_node->add_input("input_seq");
  insert_node->add_input(tensor_to_insert.name);
  if (has_position) {
    insert_node->add_input("position");
  }
  insert_node->add_output("output_sequence");

  for (const Tensor &t : inputs) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(tensor_to_insert, *graph->add_input());
  if (has_position) {
    FillValueInfo(position_tensor, *graph->add_input());
  }

  FillValueInfo(stacked, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : inputs) {
    ds.inputs.push_back(t);
  }
  ds.inputs.push_back(tensor_to_insert);
  if (has_position) {
    ds.inputs.push_back(position_tensor);
  }
  ds.outputs.push_back(stacked);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
  PromoteOutputToSequenceType(registry, static_cast<int32_t>(out_seq.elem_type), elem_shape);
}

} // namespace

void RegisterSequenceInsertCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  const std::vector<int64_t> elem_shape = {2, 3};
  Tensor a = Tensor::FromFloat("a", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor b = Tensor::FromFloat("b", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor c = Tensor::FromFloat("c", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
  Tensor x = Tensor::FromFloat("x", elem_shape, {12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f});

  RegisterSequenceInsertCase("test_cc_sequence_insert_default", {a, b, c}, x, elem_shape,
                             /*has_position=*/false, /*position=*/0, opset, registry);
  RegisterSequenceInsertCase("test_cc_sequence_insert_pos1", {a, b, c}, x, elem_shape,
                             /*has_position=*/true, /*position=*/1, opset, registry);
  RegisterSequenceInsertCase("test_cc_sequence_insert_neg", {a, b, c}, x, elem_shape,
                             /*has_position=*/true, /*position=*/-1, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
