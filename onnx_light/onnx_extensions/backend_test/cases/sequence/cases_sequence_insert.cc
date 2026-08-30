// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 13;

void RegisterSequenceInsertCase(const std::string &name, const std::vector<Tensor> &inputs,
                                const Tensor &tensor_to_insert,
                                const std::vector<int64_t> &elem_shape, bool has_position,
                                int64_t position, const OpsetId &opset,
                                std::vector<TestCase> &registry) {
  TestCase lazy_case(name, name);
  lazy_case.rtol = 1e-3;
  lazy_case.atol = 1e-7;
  lazy_case.build = [opset, inputs, has_position, position, tensor_to_insert, name,
                     elem_shape](bool) -> BuiltCase {
    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::SequenceConstruct kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::SequenceInsert kernel_2{ctx_2};
    const KernelContext ctx_3{opset};
    const onnx_kernels::kernel::SequenceConstruct kernel_3{ctx_3};

    const Sequence seq = kernel_1.AsSequence(inputs);
    Tensor position_tensor;
    const Tensor *pos_ptr = nullptr;
    if (has_position) {
      position_tensor = Tensor::FromInt64("position", {}, {position});
      pos_ptr = &position_tensor;
    }
    const Sequence out_seq = kernel_2(seq, tensor_to_insert, pos_ptr);

    std::vector<Tensor> out_values(out_seq.values.begin(), out_seq.values.end());
    Tensor stacked = kernel_3(out_values);
    stacked.name = "output_sequence";

    TestCase tc(name, name);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
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

    AppendValueInfo(
        *graph->add_output(), stacked.name,
        SequenceTypeSpec(TensorTypeSpec(static_cast<int32_t>(out_seq.elem_type), elem_shape)));

    DataSet ds;
    for (const Tensor &t : inputs) {
      ds.inputs.push_back(t);
    }
    ds.inputs.push_back(tensor_to_insert);
    if (has_position) {
      ds.inputs.push_back(position_tensor);
    }
    ds.outputs.push_back(stacked);
    tc.data_sets().emplace_back(std::move(ds));

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace

void RegisterSequenceInsertCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> big_shape = {512, 512};
    std::vector<Tensor> inputs;
    inputs.reserve(8);
    for (int i = 0; i < 8; ++i) {
      inputs.push_back(
          Tensor::FromFloat("t" + std::to_string(i), big_shape, Randn<float>(big_shape, 2001 + i)));
    }
    Tensor to_insert = RandnTensor(DataType::FLOAT, big_shape, 3001);
    RegisterSequenceInsertCase("test_cc_sequence_insert_benchmark", inputs, to_insert, big_shape,
                               /*has_position=*/true, /*position=*/4, opset, registry);
    return;
  }

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

  // Cases mirroring upstream test_sequence_insert_at_back and
  // test_sequence_insert_at_front (INT64 tensors).
  {
    const std::vector<int64_t> int_elem_shape = {3};
    Tensor i0 = Tensor::FromInt64("i0", int_elem_shape, {1, 2, 3});
    Tensor i1 = Tensor::FromInt64("i1", int_elem_shape, {4, 5, 6});
    Tensor i2 = Tensor::FromInt64("i2", int_elem_shape, {7, 8, 9});
    Tensor ix = Tensor::FromInt64("ix", int_elem_shape, {10, 11, 12});

    RegisterSequenceInsertCase("test_cc_sequence_insert_at_back", {i0, i1, i2}, ix, int_elem_shape,
                               /*has_position=*/false, /*position=*/0, opset, registry);
    RegisterSequenceInsertCase("test_cc_sequence_insert_at_front", {i0, i1, i2}, ix, int_elem_shape,
                               /*has_position=*/true, /*position=*/0, opset, registry);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
