// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Promotes the registry's most recently appended test case so that its
// single graph output ``output_sequence`` is declared as a
// ``SequenceType<Tensor<elem_type, shape>>`` rather than the plain
// ``TensorType`` ``Expect()`` emits via ``FillValueInfo``. ONNX runtimes
// validate the declared output type against the operator schema and would
// reject the model otherwise.
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
  // Drop the now-redundant tensor_type oneof field so only sequence_type is
  // serialized for this value-info.
  out_tp.reset_tensor_type();
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceConstruct — builds a tensor sequence from N input tensors that
// share the same element type (since opset 11 in the ai.onnx domain).
//
// Because the project's runtime :ref:`Tensor` type does not natively model
// sequence values, the registered cases materialize the constructed
// sequence as a single stacked ``Tensor`` whose outer dimension is the
// sequence length. The graph's declared output type is then promoted to
// ``SequenceType<Tensor<elem_type, elem_shape>>`` to match the operator
// schema.
//
// Two cases are registered:
//   * ``test_cc_sequence_construct``: three FLOAT tensors of shape [2, 3]
//     stacked into a ``Tensor<FLOAT, [3, 2, 3]>``.
//   * ``test_cc_sequence_construct_int64_single``: a single INT64 tensor of
//     shape [4] stacked into a ``Tensor<INT64, [1, 4]>``.
// ---------------------------------------------------------------------------
void RegisterSequenceConstructCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};

  // Case 1: three FLOAT tensors of shape [2, 3].
  {
    NodeProto node;
    node.set_op_type("SequenceConstruct");
    node.add_input("a");
    node.add_input("b");
    node.add_input("c");
    node.add_output("output_sequence");

    const std::vector<int64_t> elem_shape = {2, 3};
    Tensor a = Tensor::FromFloat("", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor b = Tensor::FromFloat("", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor c = Tensor::FromFloat("", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    Tensor output = kernel::SequenceConstruct(ctx)(std::vector<Tensor>{a, b, c});

    Expect(node, {a, b, c}, {output}, "test_cc_sequence_construct", {opset}, "backend-test",
           registry);
    PromoteOutputToSequenceType(registry, static_cast<int32_t>(DataType::FLOAT), elem_shape);
  }

  // Case 2: a single INT64 tensor of shape [4].
  {
    NodeProto node;
    node.set_op_type("SequenceConstruct");
    node.add_input("a");
    node.add_output("output_sequence");

    const std::vector<int64_t> elem_shape = {4};
    Tensor a = Tensor::FromInt64("", elem_shape, {-1, 0, 1, 2});

    Tensor output = kernel::SequenceConstruct(ctx)(std::vector<Tensor>{a});

    Expect(node, {a}, {output}, "test_cc_sequence_construct_int64_single", {opset}, "backend-test",
           registry);
    PromoteOutputToSequenceType(registry, static_cast<int32_t>(DataType::INT64), elem_shape);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
