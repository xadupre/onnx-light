// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
namespace {

void RegisterPackingCases(std::vector<TestCase> &registry, TestMode mode, bool unpack) {
  if (mode == TestMode::BENCHMARK)
    return;
  const std::string op = unpack ? "Unpack" : "Pack";
  auto add = [&](const std::string &suffix, Tensor digits, Tensor words, int64_t bits,
                 bool ternary = false, int64_t axis = -1) {
    NodeProto node =
        unpack ? MakeNode(op.c_str(), {"X", "count"}, {"Y"}) : MakeNode(op.c_str(), {"X"}, {"Y"});
    if (ternary) {
      AddAttribute<int64_t>(node, "radix", 3);
      AddAttribute<int64_t>(node, "group_size", 5);
    } else {
      AddAttribute<int64_t>(node, "bits", bits);
    }
    AddAttribute<int64_t>(node, "axis", axis);
    if (!unpack)
      AddAttribute<int64_t>(node, "output_type", words.data_type);
    const int64_t count = digits.shape[axis < 0 ? axis + digits.shape.size() : axis];
    Expect(registry, std::move(node), "test_" + std::string(unpack ? "unpack_" : "pack_") + suffix,
           {DefaultOpset(29)}, [digits, words, count, unpack]() -> IoData {
             if (unpack)
               return IoData{{words, Tensor::FromInt64("", {}, {count})}, {digits}};
             return IoData{{digits}, {words}};
           });
  };
  add("3bit", Tensor::FromUint8("", {8}, {0, 1, 2, 3, 4, 5, 6, 7}),
      Tensor::FromUint8("", {3}, {0x88, 0xC6, 0xFA}), 3);
  add("rows_5bit", Tensor::FromUint8("", {2, 3}, {1, 2, 3, 31, 0, 16}),
      Tensor::FromUint8("", {2, 2}, {0x41, 0x0C, 0x1F, 0x40}), 5);
  add("empty", Tensor::FromUint8("", {2, 0}, {}), Tensor::FromUint8("", {2, 0}, {}), 3);
  add("ternary", Tensor::FromUint8("", {2, 5}, {0, 1, 2, 0, 1, 2, 2, 2, 2, 2}),
      Tensor::FromUint8("", {2, 1}, {102, 242}), 0, true);
  add("uint32_axis", Tensor::FromUint8("", {7, 2}, std::vector<uint8_t>(14, 31)),
      Tensor::FromUint32(
          "", {2, 2},
          {0xFFFFFFFF, 0xFFFFFFFF, unpack ? 0xFFFFFFFFu : 7u, unpack ? 0xFFFFFFFFu : 7u}),
      5, false, 0);
}

} // namespace

void RegisterPackCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterPackingCases(registry, mode, false);
}

void RegisterUnpackCases(std::vector<TestCase> &registry, TestMode mode) {
  RegisterPackingCases(registry, mode, true);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
