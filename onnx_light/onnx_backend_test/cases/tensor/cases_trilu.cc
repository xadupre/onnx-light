// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeTriluNode(bool with_k, bool upper, bool set_upper_attr) {
  NodeProto node;
  node.set_op_type("Trilu");
  node.add_input("X");
  if (with_k) {
    node.add_input("K");
  }
  node.add_output("Y");
  if (set_upper_attr) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("upper");
    attr->set_type(AttributeProto::INT);
    attr->set_i(upper ? 1 : 0);
  }
  return node;
}

} // namespace

void RegisterTriluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Trilu trilu_kernel{ctx};

  // test_cc_trilu_upper_default: 3x3 upper triangle (k=0, default attrs).
  {
    const Tensor x =
        Tensor::FromFloat("X", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/true, /*set_upper_attr=*/false), {x}, {y},
           "test_cc_trilu_upper_default", {opset}, "backend-test", registry);
  }

  // test_cc_trilu_lower: 3x3 lower triangle (k=0).
  {
    const Tensor x =
        Tensor::FromFloat("X", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/false, /*set_upper_attr=*/true), {x}, {y},
           "test_cc_trilu_lower", {opset}, "backend-test", registry);
  }

  // test_cc_trilu_upper_k_positive: shifts diagonal up by 1.
  {
    const Tensor x = Tensor::FromInt64("X", {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const Tensor k = Tensor::FromInt64("K", {}, {1});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_trilu_upper_k_positive", {opset}, "backend-test", registry);
  }

  // test_cc_trilu_lower_k_negative: lower triangle excluding the main diagonal.
  {
    const Tensor x =
        Tensor::FromFloat("X", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    const Tensor k = Tensor::FromInt64("K", {}, {-1});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_trilu_lower_k_negative", {opset}, "backend-test", registry);
  }

  // test_cc_trilu_batched_upper: batch of 2 matrices, upper, default k.
  {
    const Tensor x =
        Tensor::FromFloat("X", {2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/true, /*set_upper_attr=*/false), {x}, {y},
           "test_cc_trilu_batched_upper", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
