// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeReverseSequenceNode(int64_t time_axis, int64_t batch_axis, bool set_time_attr,
                                  bool set_batch_attr) {
  NodeProto node;
  node.set_op_type("ReverseSequence");
  node.add_input("X");
  node.add_input("sequence_lens");
  node.add_output("Y");
  if (set_time_attr) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("time_axis");
    attr->set_type(AttributeProto::INT);
    attr->set_i(time_axis);
  }
  if (set_batch_attr) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("batch_axis");
    attr->set_type(AttributeProto::INT);
    attr->set_i(batch_axis);
  }
  return node;
}

} // namespace

void RegisterReverseSequenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::ReverseSequence reverse_seq_kernel{ctx};

  // test_cc_reversesequence_time: 4x4 input, time_axis=0, batch_axis=1.
  // Matches Example 1 in the ONNX spec.
  {
    const Tensor x = Tensor::FromFloat("X", {4, 4},
                                       {0.0f, 4.0f, 8.0f, 12.0f, 1.0f, 5.0f, 9.0f, 13.0f, 2.0f,
                                        6.0f, 10.0f, 14.0f, 3.0f, 7.0f, 11.0f, 15.0f});
    const Tensor seq = Tensor::FromInt64("sequence_lens", {4}, {4, 3, 2, 1});
    kernel::ReverseSequence::Attributes attrs;
    attrs.time_axis = 0;
    attrs.batch_axis = 1;
    const Tensor y = reverse_seq_kernel(x, seq, attrs);
    Expect(MakeReverseSequenceNode(0, 1, /*set_time_attr=*/true, /*set_batch_attr=*/true), {x, seq},
           {y}, "test_cc_reversesequence_time", {opset}, "backend-test", registry);
  }

  // test_cc_reversesequence_batch: 4x4 input, time_axis=1, batch_axis=0.
  // Matches Example 2 in the ONNX spec.
  {
    const Tensor x = Tensor::FromFloat("X", {4, 4},
                                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
    const Tensor seq = Tensor::FromInt64("sequence_lens", {4}, {1, 2, 3, 4});
    kernel::ReverseSequence::Attributes attrs;
    attrs.time_axis = 1;
    attrs.batch_axis = 0;
    const Tensor y = reverse_seq_kernel(x, seq, attrs);
    Expect(MakeReverseSequenceNode(1, 0, /*set_time_attr=*/true, /*set_batch_attr=*/true), {x, seq},
           {y}, "test_cc_reversesequence_batch", {opset}, "backend-test", registry);
  }

  // test_cc_reversesequence_default_attrs: defaults (time_axis=0, batch_axis=1).
  {
    const Tensor x = Tensor::FromFloat("X", {3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    const Tensor seq = Tensor::FromInt64("sequence_lens", {2}, {3, 2});
    kernel::ReverseSequence::Attributes attrs;
    const Tensor y = reverse_seq_kernel(x, seq, attrs);
    Expect(MakeReverseSequenceNode(0, 1, /*set_time_attr=*/false, /*set_batch_attr=*/false),
           {x, seq}, {y}, "test_cc_reversesequence_default_attrs", {opset}, "backend-test",
           registry);
  }

  // test_cc_reversesequence_with_inner_dim: rank-3 input exercising the
  // inner-dimension copy path (time_axis=0, batch_axis=1, inner=2).
  {
    // shape [3 (time), 2 (batch), 2 (inner)]
    const Tensor x = Tensor::FromFloat(
        "X", {3, 2, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
    const Tensor seq = Tensor::FromInt64("sequence_lens", {2}, {3, 1});
    kernel::ReverseSequence::Attributes attrs;
    attrs.time_axis = 0;
    attrs.batch_axis = 1;
    const Tensor y = reverse_seq_kernel(x, seq, attrs);
    Expect(MakeReverseSequenceNode(0, 1, /*set_time_attr=*/true, /*set_batch_attr=*/true), {x, seq},
           {y}, "test_cc_reversesequence_with_inner_dim", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
