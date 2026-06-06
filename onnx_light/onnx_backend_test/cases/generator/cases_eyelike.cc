// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterEyeLikeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};

  // Upstream-style case without dtype attribute: output dtype follows input.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    const Tensor x = Tensor::FromFloat(
        "x", {3, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    const Tensor y = kernel::EyeLike(ctx)(x, /*k=*/0, /*dtype=*/0);
    Expect(node, {x}, {y}, "test_eyelike_without_dtype", {opset}, "backend-test", registry);
  }

  // Upstream-style case with dtype override and non-zero diagonal offset.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    auto *k_attr = node.add_attribute();
    k_attr->set_name("k");
    k_attr->set_type(AttributeProto::AttributeType::INT);
    k_attr->set_i(1);
    auto *dtype_attr = node.add_attribute();
    dtype_attr->set_name("dtype");
    dtype_attr->set_type(AttributeProto::AttributeType::INT);
    dtype_attr->set_i(static_cast<int64_t>(DataType::INT64));

    const Tensor x =
        Tensor::FromFloat("x", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor y = kernel::EyeLike(ctx)(x, /*k=*/1, static_cast<int32_t>(DataType::INT64));
    Expect(node, {x}, {y}, "test_eyelike_with_dtype", {opset}, "backend-test", registry);
  }

  // Upstream-style case populating an off-main diagonal with explicit FLOAT dtype.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    auto *k_attr = node.add_attribute();
    k_attr->set_name("k");
    k_attr->set_type(AttributeProto::AttributeType::INT);
    k_attr->set_i(1);
    auto *dtype_attr = node.add_attribute();
    dtype_attr->set_name("dtype");
    dtype_attr->set_type(AttributeProto::AttributeType::INT);
    dtype_attr->set_i(static_cast<int64_t>(DataType::FLOAT));

    const Tensor x = Tensor::FromFloat("x", {4, 5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    const Tensor y = kernel::EyeLike(ctx)(x, /*k=*/1, static_cast<int32_t>(DataType::FLOAT));
    Expect(node, {x}, {y}, "test_eyelike_populate_off_main_diagonal", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
