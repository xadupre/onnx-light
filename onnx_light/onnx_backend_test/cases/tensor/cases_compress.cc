// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeCompressNode(std::optional<int64_t> axis) {
  NodeProto node;
  node.set_op_type("Compress");
  node.add_input("input");
  node.add_input("condition");
  node.add_output("output");
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

void RegisterCompressCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  const kernel::KernelContext ctx{opset};
  const kernel::Compress compress_kernel{ctx};

  // test_cc_compress_no_axis — flatten then select elements.
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {6}, {1, 0, 1, 1, 0, 0});
    Tensor output = compress_kernel(input, condition, std::nullopt);
    Expect(MakeCompressNode(std::nullopt), {input, condition}, {output}, "test_cc_compress_no_axis",
           {opset}, "backend-test", registry);
  }

  // test_cc_compress_axis0 — select rows (axis=0).
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {3}, {1, 0, 1});
    Tensor output = compress_kernel(input, condition, 0);
    Expect(MakeCompressNode(0), {input, condition}, {output}, "test_cc_compress_axis0", {opset},
           "backend-test", registry);
  }

  // test_cc_compress_axis1 — select columns (axis=1).
  {
    Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {2}, {0, 1});
    Tensor output = compress_kernel(input, condition, 1);
    Expect(MakeCompressNode(1), {input, condition}, {output}, "test_cc_compress_axis1", {opset},
           "backend-test", registry);
  }

  // test_cc_compress_negative_axis — negative axis (-1 == last axis).
  {
    Tensor input = Tensor::FromFloat("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor condition = Tensor::FromBool("condition", {3}, {1, 0, 1});
    Tensor output = compress_kernel(input, condition, -1);
    Expect(MakeCompressNode(-1), {input, condition}, {output}, "test_cc_compress_negative_axis",
           {opset}, "backend-test", registry);
  }

  // test_cc_compress_short_condition — condition shorter than axis dim.
  {
    Tensor input =
        Tensor::FromFloat("input", {4, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor condition = Tensor::FromBool("condition", {2}, {1, 0});
    Tensor output = compress_kernel(input, condition, 0);
    Expect(MakeCompressNode(0), {input, condition}, {output}, "test_cc_compress_short_condition",
           {opset}, "backend-test", registry);
  }

  // test_cc_compress_int64 — non-float input dtype.
  {
    Tensor input = Tensor::FromInt64("input", {3}, {10, 20, 30});
    Tensor condition = Tensor::FromBool("condition", {3}, {0, 1, 1});
    Tensor output = compress_kernel(input, condition, 0);
    Expect(MakeCompressNode(0), {input, condition}, {output}, "test_cc_compress_int64", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
