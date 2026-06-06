// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
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

  // The cases below mirror the upstream ONNX backend tests for the Trilu op
  // (``test_triu_*`` and ``test_tril_*``). The substring-based coverage test
  // ``test_onnx_backend_test_names_found_in_onnx_light`` matches these names
  // against the upstream node tests.
  //
  // Inputs match the example tensors documented in
  // ``onnx/backend/test/case/node/trilu.py``; expected outputs are computed by
  // the reference ``kernel::Trilu`` so they always agree with the spec.

  // 4x5 input shared by the ``triu_*`` and ``tril_*`` (non-square) cases.
  const std::vector<int64_t> x45_data = {4, 7, 3, 7, 9, 1, 2, 8, 6, 9,
                                         9, 4, 0, 8, 7, 4, 3, 4, 2, 4};
  // 2x3x3 input shared by ``triu_square`` and ``triu_square_neg``.
  const std::vector<int64_t> x233_triu = {4, 6, 9, 7, 5, 4, 8, 1, 2, 1, 4, 9, 9, 6, 3, 8, 9, 8};
  // 2x3x3 input shared by ``tril_square`` and ``tril_square_neg``.
  const std::vector<int64_t> x233_tril = {0, 4, 3, 2, 0, 9, 8, 2, 5, 2, 7, 2, 2, 6, 0, 2, 6, 5};

  // -------- triu_* (upper triangular; ``upper`` attribute omitted) --------

  // test_cc_triu
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/true, /*set_upper_attr=*/false), {x}, {y},
           "test_cc_triu", {opset}, "backend-test", registry);
  }

  // test_cc_triu_neg: k = -1
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {-1});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_neg", {opset}, "backend-test", registry);
  }

  // test_cc_triu_out_neg_out: k = -7 (whole tensor kept)
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {-7});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_out_neg_out", {opset}, "backend-test", registry);
  }

  // test_cc_triu_pos: k = 2
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {2});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_pos", {opset}, "backend-test", registry);
  }

  // test_cc_triu_out_pos: k = 6 (whole tensor zeroed)
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {6});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_out_pos", {opset}, "backend-test", registry);
  }

  // test_cc_triu_square: 2x3x3 batched upper, default k.
  {
    const Tensor x = Tensor::FromInt64("X", {2, 3, 3}, x233_triu);
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/true, /*set_upper_attr=*/false), {x}, {y},
           "test_cc_triu_square", {opset}, "backend-test", registry);
  }

  // test_cc_triu_square_neg: 2x3x3 batched upper, k = -1.
  {
    const Tensor x = Tensor::FromInt64("X", {2, 3, 3}, x233_triu);
    const Tensor k = Tensor::FromInt64("K", {}, {-1});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_square_neg", {opset}, "backend-test", registry);
  }

  // test_cc_triu_one_row: shape [3, 1, 5], k = 1.
  {
    const Tensor x =
        Tensor::FromInt64("X", {3, 1, 5}, {1, 4, 9, 7, 1, 9, 2, 8, 8, 4, 3, 9, 7, 4, 2});
    const Tensor k = Tensor::FromInt64("K", {}, {1});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_one_row", {opset}, "backend-test", registry);
  }

  // test_cc_triu_zero: shape [0, 5], k = 6 (zero-sized input).
  {
    const Tensor x = Tensor::FromInt64("X", {0, 5}, {});
    const Tensor k = Tensor::FromInt64("K", {}, {6});
    kernel::Trilu::Attributes attrs;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/true, /*set_upper_attr=*/false), {x, k}, {y},
           "test_cc_triu_zero", {opset}, "backend-test", registry);
  }

  // -------- tril_* (lower triangular; ``upper`` attribute set to 0) --------

  // test_cc_tril_neg: k = -1.
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {-1});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_neg", {opset}, "backend-test", registry);
  }

  // test_cc_tril_out_neg: k = -7 (whole tensor zeroed).
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {-7});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_out_neg", {opset}, "backend-test", registry);
  }

  // test_cc_tril_pos: k = 2.
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {2});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_pos", {opset}, "backend-test", registry);
  }

  // test_cc_tril_out_pos: k = 6 (whole tensor kept).
  {
    const Tensor x = Tensor::FromInt64("X", {4, 5}, x45_data);
    const Tensor k = Tensor::FromInt64("K", {}, {6});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_out_pos", {opset}, "backend-test", registry);
  }

  // test_cc_tril_square: 2x3x3 batched lower, default k.
  {
    const Tensor x = Tensor::FromInt64("X", {2, 3, 3}, x233_tril);
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/false, /*set_upper_attr=*/true), {x}, {y},
           "test_cc_tril_square", {opset}, "backend-test", registry);
  }

  // test_cc_tril_square_neg: 2x3x3 batched lower, k = -1.
  {
    const Tensor x = Tensor::FromInt64("X", {2, 3, 3}, x233_tril);
    const Tensor k = Tensor::FromInt64("K", {}, {-1});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_square_neg", {opset}, "backend-test", registry);
  }

  // test_cc_tril_one_row_neg: shape [3, 1, 5], default k, lower.
  {
    const Tensor x =
        Tensor::FromInt64("X", {3, 1, 5}, {6, 2, 4, 1, 6, 8, 3, 8, 7, 0, 2, 2, 9, 5, 9});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, /*k=*/nullptr, attrs);
    Expect(MakeTriluNode(/*with_k=*/false, /*upper=*/false, /*set_upper_attr=*/true), {x}, {y},
           "test_cc_tril_one_row_neg", {opset}, "backend-test", registry);
  }

  // test_cc_tril_zero: shape [3, 0, 5], k = 6 (zero-sized input), lower.
  {
    const Tensor x = Tensor::FromInt64("X", {3, 0, 5}, {});
    const Tensor k = Tensor::FromInt64("K", {}, {6});
    kernel::Trilu::Attributes attrs;
    attrs.upper = 0;
    const Tensor y = trilu_kernel(x, &k, attrs);
    Expect(MakeTriluNode(/*with_k=*/true, /*upper=*/false, /*set_upper_attr=*/true), {x, k}, {y},
           "test_cc_tril_zero", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
