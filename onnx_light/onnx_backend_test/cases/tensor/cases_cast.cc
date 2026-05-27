// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Cast — element-wise conversion to the dtype carried by the required
// integer attribute ``to`` (since opset 13 in the ai.onnx domain).
//
// The cases below mirror the upstream ``test_cast_<FROM>_to_<TO>`` node
// tests for every element type supported by the backend test ``Tensor``
// library and by :ref:`kernel::Cast`: the numeric types ``FLOAT``,
// ``DOUBLE``, ``INT32``, ``INT64``, ``INT8``, ``UINT8``, ``INT16``,
// ``UINT16``, ``BOOL`` and the variable-length ``STRING`` type. All
// cross-dtype permutations are registered so that ``Cast`` coverage is
// complete with respect to what the reference kernel accepts. Inputs are
// small, fully deterministic vectors so the expected outputs can be
// computed directly by :ref:`kernel::Cast`. Upstream cases over
// ``FLOAT16``, ``BFLOAT16``, the FP8/FP4 variants and the sub-byte
// integer dtypes are intentionally omitted: those element types are not
// supported by the backend test ``Tensor`` storage nor by
// :ref:`kernel::Cast`, so they would need to be added at the kernel
// layer first.
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

struct CastDtype {
  TensorProto::DataType dtype;
  const char *name;
  std::function<Tensor()> make_input;
};

// Per-source-dtype deterministic inputs. Values are chosen to fit in the
// narrowest representable integer dtype (``INT8`` / ``UINT8``) so that the
// reference kernel's narrowing casts do not depend on undefined-behaviour
// overflow paths. ``BOOL`` uses {0, 1, 1, 0} so that both truthy and falsy
// elements appear in every cast.
std::vector<CastDtype> SupportedCastDtypes() {
  return {
      {TensorProto::DataType::FLOAT, "FLOAT",
       []() { return Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f}); }},
      {TensorProto::DataType::DOUBLE, "DOUBLE",
       []() { return Tensor::FromDouble("", {4}, {-1.5, 0.0, 2.75, 4.0}); }},
      {TensorProto::DataType::INT32, "INT32",
       []() { return Tensor::FromInt32("", {4}, {-3, 0, 7, 42}); }},
      {TensorProto::DataType::INT64, "INT64",
       []() { return Tensor::FromInt64("", {4}, {-3, 0, 7, 42}); }},
      {TensorProto::DataType::INT8, "INT8",
       []() { return Tensor::FromInt8("", {4}, {-3, 0, 7, 42}); }},
      {TensorProto::DataType::UINT8, "UINT8",
       []() { return Tensor::FromUint8("", {4}, {0, 1, 7, 42}); }},
      {TensorProto::DataType::INT16, "INT16",
       []() { return Tensor::FromInt16("", {4}, {-3, 0, 7, 42}); }},
      {TensorProto::DataType::UINT16, "UINT16",
       []() { return Tensor::FromUint16("", {4}, {0, 1, 7, 42}); }},
      {TensorProto::DataType::BOOL, "BOOL",
       []() { return Tensor::FromBool("", {4}, {0, 1, 1, 0}); }},
      {TensorProto::DataType::STRING, "STRING",
       []() {
         return Tensor::FromStrings(
             "", {4}, {std::string("-3"), std::string("0"), std::string("7"), std::string("42")});
       }},
  };
}

} // namespace

void RegisterCastCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Cast cast_kernel{kernel::KernelContext(opset)};

  const auto dtypes = SupportedCastDtypes();
  for (const auto &from : dtypes) {
    for (const auto &to : dtypes) {
      if (from.dtype == to.dtype) {
        // Skip identity casts: upstream ONNX node tests skip them as well.
        continue;
      }
      const int64_t to_attr = static_cast<int64_t>(to.dtype);
      NodeProto node = MakeCastNode(to_attr);
      Tensor input = from.make_input();
      Tensor output = cast_kernel(input, static_cast<int32_t>(to_attr));
      Expect(node, {input}, {output}, std::string("test_cc_cast_") + from.name + "_to_" + to.name,
             {opset}, "backend-test", registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
