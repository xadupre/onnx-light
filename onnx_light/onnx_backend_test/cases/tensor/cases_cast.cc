// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Cast — element-wise conversion to the dtype carried by the required
// integer attribute ``to`` (since opset 13 in the ai.onnx domain).
//
// The cases below mirror the upstream ``test_cast_FLOAT_to_*`` /
// ``test_cast_*_to_FLOAT`` node tests for the four numeric element types
// natively supported by the backend test ``Tensor`` library: FLOAT, DOUBLE,
// INT32 and INT64. Inputs are small, fully deterministic vectors so the
// expected outputs can be computed by the reference :ref:`kernel::Cast`.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastNode(int64_t to) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("input");
  node.add_output("output");
  AddAttribute<int64_t>(node, "to", to);
  return node;
}

} // namespace

void RegisterCastCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Cast cast_kernel{kernel::KernelContext(opset)};

  // FLOAT -> DOUBLE: widening conversion of a small 1-D vector.
  {
    const int64_t to = static_cast<int64_t>(TensorProto::DataType::DOUBLE);
    NodeProto node = MakeCastNode(to);
    Tensor input = Tensor::FromFloat("", {3}, {-1.5f, 0.0f, 2.25f});
    Tensor output = cast_kernel(input, static_cast<int32_t>(to));
    Expect(node, {input}, {output}, "test_cc_cast_FLOAT_to_DOUBLE", {opset}, "backend-test",
           registry);
  }

  // DOUBLE -> FLOAT: narrowing conversion.
  {
    const int64_t to = static_cast<int64_t>(TensorProto::DataType::FLOAT);
    NodeProto node = MakeCastNode(to);
    Tensor input = Tensor::FromDouble("", {3}, {-1.5, 0.0, 2.25});
    Tensor output = cast_kernel(input, static_cast<int32_t>(to));
    Expect(node, {input}, {output}, "test_cc_cast_DOUBLE_to_FLOAT", {opset}, "backend-test",
           registry);
  }

  // FLOAT -> INT32: truncation toward zero (C++ static_cast semantics).
  {
    const int64_t to = static_cast<int64_t>(TensorProto::DataType::INT32);
    NodeProto node = MakeCastNode(to);
    Tensor input = Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f});
    Tensor output = cast_kernel(input, static_cast<int32_t>(to));
    Expect(node, {input}, {output}, "test_cc_cast_FLOAT_to_INT32", {opset}, "backend-test",
           registry);
  }

  // INT64 -> FLOAT: integer widening to a floating-point type.
  {
    const int64_t to = static_cast<int64_t>(TensorProto::DataType::FLOAT);
    NodeProto node = MakeCastNode(to);
    Tensor input = Tensor::FromInt64("", {4}, {-3, 0, 7, 42});
    Tensor output = cast_kernel(input, static_cast<int32_t>(to));
    Expect(node, {input}, {output}, "test_cc_cast_INT64_to_FLOAT", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
