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
// The cases below mirror the upstream ``test_cast_<FROM>_to_<TO>`` node
// tests for the four numeric element types natively supported by the
// backend test ``Tensor`` library: FLOAT, DOUBLE, INT32 and INT64. The
// upstream ONNX test suite also exercises FLOAT16, BFLOAT16, FP8, FP4 and
// sub-byte integer dtypes, but those are not yet handled by the backend
// test ``Tensor`` storage nor by :ref:`kernel::Cast`, so they are
// intentionally omitted here. All twelve cross-dtype permutations among
// the supported four types are registered so that ``Cast`` coverage is
// complete with respect to what the reference kernel accepts. Inputs are
// small, fully deterministic vectors so the expected outputs can be
// computed by the reference :ref:`kernel::Cast`.
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

template <typename FromCtor>
void RegisterOneCastCase(std::vector<TestCase> &registry, const kernel::Cast &cast_kernel,
                         const OpsetId &opset, const std::string &from_name,
                         TensorProto::DataType to_dtype, const std::string &to_name,
                         FromCtor make_input) {
  const int64_t to = static_cast<int64_t>(to_dtype);
  NodeProto node = MakeCastNode(to);
  Tensor input = make_input();
  Tensor output = cast_kernel(input, static_cast<int32_t>(to));
  Expect(node, {input}, {output}, "test_cc_cast_" + from_name + "_to_" + to_name, {opset},
         "backend-test", registry);
}

} // namespace

void RegisterCastCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Cast cast_kernel{kernel::KernelContext(opset)};

  // Deterministic per-source-dtype inputs covering negatives, zero and
  // values with non-zero fractional parts so the narrowing-to-integer
  // semantics (truncation toward zero) are exercised.
  auto make_float_input = []() { return Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); };
  auto make_double_input = []() { return Tensor::FromDouble("", {4}, {-1.5, 0.0, 2.75, 4.0}); };
  auto make_int32_input = []() { return Tensor::FromInt32("", {4}, {-3, 0, 7, 42}); };
  auto make_int64_input = []() { return Tensor::FromInt64("", {4}, {-3, 0, 7, 42}); };

  // FLOAT -> {DOUBLE, INT32, INT64}.
  RegisterOneCastCase(registry, cast_kernel, opset, "FLOAT", TensorProto::DataType::DOUBLE,
                      "DOUBLE", make_float_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "FLOAT", TensorProto::DataType::INT32, "INT32",
                      make_float_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "FLOAT", TensorProto::DataType::INT64, "INT64",
                      make_float_input);

  // DOUBLE -> {FLOAT, INT32, INT64}.
  RegisterOneCastCase(registry, cast_kernel, opset, "DOUBLE", TensorProto::DataType::FLOAT, "FLOAT",
                      make_double_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "DOUBLE", TensorProto::DataType::INT32, "INT32",
                      make_double_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "DOUBLE", TensorProto::DataType::INT64, "INT64",
                      make_double_input);

  // INT32 -> {FLOAT, DOUBLE, INT64}.
  RegisterOneCastCase(registry, cast_kernel, opset, "INT32", TensorProto::DataType::FLOAT, "FLOAT",
                      make_int32_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "INT32", TensorProto::DataType::DOUBLE,
                      "DOUBLE", make_int32_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "INT32", TensorProto::DataType::INT64, "INT64",
                      make_int32_input);

  // INT64 -> {FLOAT, DOUBLE, INT32}.
  RegisterOneCastCase(registry, cast_kernel, opset, "INT64", TensorProto::DataType::FLOAT, "FLOAT",
                      make_int64_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "INT64", TensorProto::DataType::DOUBLE,
                      "DOUBLE", make_int64_input);
  RegisterOneCastCase(registry, cast_kernel, opset, "INT64", TensorProto::DataType::INT32, "INT32",
                      make_int64_input);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
