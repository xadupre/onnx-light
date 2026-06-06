// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// CastLike — element-wise conversion to the dtype carried by the second
// input ``target_type`` (since opset 15 in the ai.onnx domain). The first
// input ``input`` carries the values; the second input is consulted only
// for its element type. Functionally equivalent to ``Cast`` with
// ``to = target_type.data_type``.
//
// The cases below mirror the upstream ``test_castlike_<FROM>_to_<TO>`` node
// tests for every element type supported by the backend test ``Tensor``
// library and by :ref:`kernel::CastLike`. Inputs are short, fully
// deterministic vectors so the expected outputs can be computed directly
// by :ref:`kernel::CastLike`.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastLikeNode() {
  NodeProto node;
  node.set_op_type("CastLike");
  node.add_input("input");
  node.add_input("target_type");
  node.add_output("output");
  return node;
}

struct CastLikeDtype {
  DataType dtype;
  const char *name;
  std::function<Tensor()> make_input;
};

// Per-source-dtype deterministic inputs. Values mirror those used by the
// ``Cast`` backend test cases so the two operators are exercised over the
// same conversion matrix.
std::vector<CastLikeDtype> SupportedCastLikeDtypes() {
  return {
      {DataType::FLOAT, "FLOAT",
       []() { return Tensor::FromFloat("input", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      {DataType::DOUBLE, "DOUBLE",
       []() { return Tensor::FromDouble("input", {4}, {-1.5, 0.0, 2.75, 4.0}); }},
      {DataType::INT32, "INT32", []() { return Tensor::FromInt32("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT64, "INT64", []() { return Tensor::FromInt64("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::INT8, "INT8", []() { return Tensor::FromInt8("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT8, "UINT8", []() { return Tensor::FromUint8("input", {4}, {0, 1, 7, 42}); }},
      {DataType::INT16, "INT16", []() { return Tensor::FromInt16("input", {4}, {-3, 0, 7, 42}); }},
      {DataType::UINT16, "UINT16",
       []() { return Tensor::FromUint16("input", {4}, {0, 1, 7, 42}); }},
      {DataType::BOOL, "BOOL", []() { return Tensor::FromBool("input", {4}, {0, 1, 1, 0}); }},
      {DataType::STRING, "STRING",
       []() { return Tensor::FromStrings("input", {4}, {"-3", "0", "7", "42"}); }},
  };
}

// Builds a placeholder ``target_type`` input with the requested dtype. Its
// contents are unused by :ref:`kernel::CastLike` (only the dtype is read),
// but the tensor must still carry one element so the backend test harness
// can serialize it as a regular initializer.
Tensor MakeTargetTypeTensor(const CastLikeDtype &to) {
  // A single-element tensor is enough for the dtype to be propagated. The
  // value is irrelevant; pick the smallest representative value of the
  // target dtype.
  switch (to.dtype) {
  case DataType::FLOAT:
    return Tensor::FromFloat("target_type", {1}, {0.0f});
  case DataType::DOUBLE:
    return Tensor::FromDouble("target_type", {1}, {0.0});
  case DataType::INT32:
    return Tensor::FromInt32("target_type", {1}, {0});
  case DataType::INT64:
    return Tensor::FromInt64("target_type", {1}, {0});
  case DataType::INT8:
    return Tensor::FromInt8("target_type", {1}, {0});
  case DataType::UINT8:
    return Tensor::FromUint8("target_type", {1}, {0});
  case DataType::INT16:
    return Tensor::FromInt16("target_type", {1}, {0});
  case DataType::UINT16:
    return Tensor::FromUint16("target_type", {1}, {0});
  case DataType::BOOL:
    return Tensor::FromBool("target_type", {1}, {0});
  case DataType::STRING:
    return Tensor::FromStrings("target_type", {1}, {""});
  default:
    return Tensor("target_type", static_cast<int32_t>(to.dtype), {1}, std::vector<uint8_t>(0));
  }
}

} // namespace

void RegisterCastLikeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(15);
  const kernel::KernelContext ctx{opset};
  const kernel::CastLike castlike_kernel{ctx};

  const auto dtypes = SupportedCastLikeDtypes();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype) {
        // Skip identity casts: upstream ONNX node tests skip them as well.
        continue;
      }
      NodeProto node = MakeCastLikeNode();
      Tensor input = from.make_input();
      Tensor target_type = MakeTargetTypeTensor(to);
      Tensor output = castlike_kernel(input, target_type);
      Expect(node, {input, target_type}, {output},
             std::string("test_cc_castlike_") + from.name + "_to_" + to.name, {opset},
             "backend-test", registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
