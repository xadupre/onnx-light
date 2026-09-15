// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// SequenceConstruct — builds a tensor sequence from N input tensors that
// share the same element type (since opset 11 in the ai.onnx domain).
//
// Because the project's runtime :ref:`Tensor` type does not natively model
// sequence values, the registered cases materialize the constructed
// sequence as a single stacked ``Tensor`` whose outer dimension is the
// sequence length. The graph's declared output type is then set to
// ``SequenceType<Tensor<elem_type, elem_shape>>`` (via the ``output_types``
// argument of :func:`Expect`) to match the operator schema.
//
// Two cases are registered:
//   * ``test_cc_sequence_construct``: three FLOAT tensors of shape [2, 3]
//     stacked into a ``Tensor<FLOAT, [3, 2, 3]>``.
//   * ``test_cc_sequence_construct_int64_single``: a single INT64 tensor of
//     shape [4] stacked into a ``Tensor<INT64, [1, 4]>``.
// ---------------------------------------------------------------------------
void RegisterSequenceConstructCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> elem_shape = {512, 512};
    NodeProto node;
    node.set_op_type("SequenceConstruct");
    for (int i = 0; i < 8; ++i) {
      node.add_input("t" + std::to_string(i));
    }
    node.add_output("output_sequence");
    Expect(registry, std::move(node), "test_cc_sequence_construct_benchmark", {opset},
           [elem_shape]() -> IoData {
             const OpsetId opset = DefaultOpset(11);

             const KernelContext ctx_1{opset};
             const onnx_kernels::kernel::SequenceConstruct kernel_1{ctx_1};

             std::vector<Tensor> tensors;
             tensors.reserve(8);
             for (int i = 0; i < 8; ++i) {
               tensors.push_back(RandnTensor(DataType::FLOAT, elem_shape, 2001 + i));
             }
             Tensor output = kernel_1(tensors);
             return IoData{std::move(tensors), {std::move(output)}};
           },
           "", TestCaseTag::NONE,
           {SequenceTypeSpec(TensorTypeSpec(static_cast<int32_t>(DataType::FLOAT), elem_shape))});
    return;
  }

  // Case 1: three FLOAT tensors of shape [2, 3].
  {
    const std::vector<int64_t> elem_shape = {2, 3};
    NodeProto node;
    node.set_op_type("SequenceConstruct");
    node.add_input("a");
    node.add_input("b");
    node.add_input("c");
    node.add_output("output_sequence");
    Expect(registry, std::move(node), "test_cc_sequence_construct", {opset},
           [elem_shape]() -> IoData {
             const OpsetId opset = DefaultOpset(11);

             const KernelContext ctx_2{opset};
             const onnx_kernels::kernel::SequenceConstruct kernel_2{ctx_2};

             Tensor a =
                 Tensor::FromFloat("", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
             Tensor b = Tensor::FromFloat("", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
             Tensor c = Tensor::FromFloat("", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

             Tensor output = kernel_2(std::vector<Tensor>{a, b, c});

             return IoData{{std::move(a), std::move(b), std::move(c)}, {std::move(output)}};
           },
           "", TestCaseTag::NONE,
           {SequenceTypeSpec(TensorTypeSpec(static_cast<int32_t>(DataType::FLOAT), elem_shape))});
  }

  // Case 2: a single INT64 tensor of shape [4].
  {
    const std::vector<int64_t> elem_shape = {4};
    NodeProto node;
    node.set_op_type("SequenceConstruct");
    node.add_input("a");
    node.add_output("output_sequence");
    Expect(registry, std::move(node), "test_cc_sequence_construct_int64_single", {opset},
           [elem_shape]() -> IoData {
             const OpsetId opset = DefaultOpset(11);

             const KernelContext ctx_3{opset};
             const onnx_kernels::kernel::SequenceConstruct kernel_3{ctx_3};

             Tensor a = Tensor::FromInt64("", elem_shape, {-1, 0, 1, 2});

             Tensor output = kernel_3(std::vector<Tensor>{a});

             return IoData{{std::move(a)}, {std::move(output)}};
           },
           "", TestCaseTag::NONE,
           {SequenceTypeSpec(TensorTypeSpec(static_cast<int32_t>(DataType::INT64), elem_shape))});
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
