// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Pad — pads ``data`` along the requested axes with the chosen ``mode``.
// Mirrors the upstream ONNX node tests in
// ``onnx/backend/test/case/node/pad.py``. The case names are prefixed with
// ``test_cc_`` so they line up with the substring matches enforced by
// ``unittests/onnxl_vs_onnx/test_backend_test_names_onnx_vs_onnxlight.py``
// against the corresponding ONNX node-test names
// (``test_constant_pad``, ``test_constant_pad_axes``,
// ``test_constant_pad_negative_axes``, ``test_edge_pad``,
// ``test_reflect_pad``, ``test_wrap_pad``).
// ---------------------------------------------------------------------------

namespace {

NodeProto MakePadNode(const std::vector<std::string> &inputs, const std::string &mode) {
  NodeProto node;
  node.set_op_type("Pad");
  for (const std::string &i : inputs) {
    node.add_input(i);
  }
  node.add_output("y");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("mode");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(mode);
  return node;
}

Tensor MakeInt64Vector(const std::string &name, const std::vector<int64_t> &values) {
  const std::vector<int64_t> shape = {static_cast<int64_t>(values.size())};
  std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return Tensor(name, DataType::INT64, shape, std::move(bytes));
}

} // namespace

void RegisterPadCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(21);
  const kernel::KernelContext ctx{opset};
  const kernel::Pad pad_kernel{ctx};

  // test_cc_constant_pad — pad a small 2-D float tensor with a constant
  // value of 1.2 on the trailing axis (matches the layout exercised by the
  // upstream ``test_constant_pad`` test, scaled down to fixed values).
  //
  // x:        [[1, 2, 3],
  //            [4, 5, 6]]                      (shape [2, 3])
  // pads:     [0, 1, 0, 2]                     (begin: [0, 1], end: [0, 2])
  // value:    1.2
  // output:   [[1.2, 1, 2, 3, 1.2, 1.2],
  //            [1.2, 4, 5, 6, 1.2, 1.2]]       (shape [2, 6])
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor pads = MakeInt64Vector("pads", {0, 1, 0, 2});
    const Tensor value = Tensor::FromFloat("value", {}, {1.2f});
    const Tensor y = pad_kernel(x, pads, &value, /*axes=*/nullptr, "constant");
    Expect(MakePadNode({"x", "pads", "value"}, "constant"), {x, pads, value}, {y},
           "test_cc_constant_pad", {opset}, "backend-test", registry);
  }

  // test_cc_edge_pad — replicate the edge values on every padded position.
  //
  // x:        [[1, 2, 3],
  //            [4, 5, 6]]
  // pads:     [1, 1, 1, 1]
  // output:   [[1, 1, 2, 3, 3],
  //            [1, 1, 2, 3, 3],
  //            [4, 4, 5, 6, 6],
  //            [4, 4, 5, 6, 6]]               (shape [4, 5])
  {
    const Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor pads = MakeInt64Vector("pads", {1, 1, 1, 1});
    const Tensor y = pad_kernel(x, pads, /*constant_value=*/nullptr, /*axes=*/nullptr, "edge");
    Expect(MakePadNode({"x", "pads"}, "edge"), {x, pads}, {y}, "test_cc_edge_pad", {opset},
           "backend-test", registry);
  }

  // test_cc_reflect_pad — reflect values around the edge (excluding the edge
  // itself).
  //
  // x:        [[1, 2, 3],
  //            [4, 5, 6]]
  // pads:     [1, 1, 1, 1]
  // output:   [[5, 4, 5, 6, 5],
  //            [2, 1, 2, 3, 2],
  //            [5, 4, 5, 6, 5],
  //            [2, 1, 2, 3, 2]]               (shape [4, 5])
  {
    const Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor pads = MakeInt64Vector("pads", {1, 1, 1, 1});
    const Tensor y = pad_kernel(x, pads, /*constant_value=*/nullptr, /*axes=*/nullptr, "reflect");
    Expect(MakePadNode({"x", "pads"}, "reflect"), {x, pads}, {y}, "test_cc_reflect_pad", {opset},
           "backend-test", registry);
  }

  // test_cc_wrap_pad — wrap (periodic) padding.
  //
  // x:        [[1, 2, 3],
  //            [4, 5, 6]]
  // pads:     [1, 1, 1, 1]
  // output:   [[6, 4, 5, 6, 4],
  //            [3, 1, 2, 3, 1],
  //            [6, 4, 5, 6, 4],
  //            [3, 1, 2, 3, 1]]               (shape [4, 5])
  {
    const Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor pads = MakeInt64Vector("pads", {1, 1, 1, 1});
    const Tensor y = pad_kernel(x, pads, /*constant_value=*/nullptr, /*axes=*/nullptr, "wrap");
    Expect(MakePadNode({"x", "pads"}, "wrap"), {x, pads}, {y}, "test_cc_wrap_pad", {opset},
           "backend-test", registry);
  }

  // test_cc_constant_pad_axes — restrict padding to axes [1, 3] of a rank-4
  // tensor; the other axes keep their input dim unchanged.
  //
  // x shape:  [1, 2, 2, 2]  (filled with 1..8)
  // pads:     [0, 1, 0, 2]  (begin/end for axes [1, 3])
  // axes:     [1, 3]
  // output shape: [1, 2, 2, 5]  (axis 1 grows by 0+0; axis 3 grows by 1+2)
  // value:    7.0
  {
    const Tensor x =
        Tensor::FromFloat("x", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor pads = MakeInt64Vector("pads", {0, 1, 0, 2});
    const Tensor value = Tensor::FromFloat("value", {}, {7.0f});
    const Tensor axes = MakeInt64Vector("axes", {1, 3});
    const Tensor y = pad_kernel(x, pads, &value, &axes, "constant");
    Expect(MakePadNode({"x", "pads", "value", "axes"}, "constant"), {x, pads, value, axes}, {y},
           "test_cc_constant_pad_axes", {opset}, "backend-test", registry);
  }

  // test_cc_constant_pad_negative_axes — same as the previous case but the
  // axes are given as negative indices ([-3, -1]) which must resolve to the
  // same axes [1, 3] for a rank-4 input.
  {
    const Tensor x =
        Tensor::FromFloat("x", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor pads = MakeInt64Vector("pads", {0, 1, 0, 2});
    const Tensor value = Tensor::FromFloat("value", {}, {7.0f});
    const Tensor axes = MakeInt64Vector("axes", {-3, -1});
    const Tensor y = pad_kernel(x, pads, &value, &axes, "constant");
    Expect(MakePadNode({"x", "pads", "value", "axes"}, "constant"), {x, pads, value, axes}, {y},
           "test_cc_constant_pad_negative_axes", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
