// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

void AddInt(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *a = node.add_attribute();
  a->set_name(name);
  a->set_type(AttributeProto::AttributeType::INT);
  a->set_i(value);
}

NodeProto MakeFlattenNode(int64_t axis, bool include_axis) {
  NodeProto node;
  node.set_op_type("Flatten");
  node.add_input("a");
  node.add_output("b");
  if (include_axis) {
    AddInt(node, "axis", axis);
  }
  return node;
}

// Generates deterministic test data: a simple linear sequence shifted around
// zero so positive and negative values are exercised. The Flatten kernel
// reuses the input data byte-for-byte, so the exact content does not matter
// as long as the input and expected-output buffers stay bit-identical.
std::vector<float> SequentialFloats(size_t count) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = static_cast<float>(i) * 0.5f - static_cast<float>(count) / 4.0f;
  }
  return data;
}

} // namespace

// ---------------------------------------------------------------------------
// Flatten — reshape an N-D tensor into a 2-D matrix
// ``(prod(shape[:axis]), prod(shape[axis:]))``. The data buffer is unchanged.
//
// Cases (mirror upstream test_flatten_*):
//   * test_cc_flatten_default_axis   — default axis=1 on a (5, 4, 3, 2) input.
//   * test_cc_flatten_axis0..axis3   — explicit axis=0..3 on a (2, 3, 4, 5) input.
//   * test_cc_flatten_negative_axis1..4 — axis=-1..-4 on a (2, 3, 4, 5) input.
// ---------------------------------------------------------------------------
void RegisterFlattenCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(25);
  const kernel::KernelContext ctx{opset};
  const kernel::Flatten kernel{ctx};

  // Default axis (axis=1) on a (5, 4, 3, 2) input.
  {
    NodeProto node = MakeFlattenNode(/*axis=*/1, /*include_axis=*/false);
    std::vector<float> data = SequentialFloats(5 * 4 * 3 * 2);
    Tensor a = Tensor::FromFloat("", {5, 4, 3, 2}, data);
    Tensor b = kernel(a);
    Expect(node, {a}, {b}, "test_cc_flatten_default_axis", {opset}, "backend-test", registry);
  }

  // Explicit axis 0..3 on a (2, 3, 4, 5) input.
  {
    const std::vector<int64_t> shape{2, 3, 4, 5};
    std::vector<float> data = SequentialFloats(2 * 3 * 4 * 5);
    for (int64_t axis = 0; axis < 4; ++axis) {
      NodeProto node = MakeFlattenNode(axis, /*include_axis=*/true);
      Tensor a = Tensor::FromFloat("", shape, data);
      Tensor b = kernel(a, axis);
      Expect(node, {a}, {b}, "test_cc_flatten_axis" + std::to_string(axis), {opset}, "backend-test",
             registry);
    }
  }

  // Negative axis -1..-4 on a (2, 3, 4, 5) input.
  {
    const std::vector<int64_t> shape{2, 3, 4, 5};
    std::vector<float> data = SequentialFloats(2 * 3 * 4 * 5);
    for (int64_t axis = -4; axis < 0; ++axis) {
      NodeProto node = MakeFlattenNode(axis, /*include_axis=*/true);
      Tensor a = Tensor::FromFloat("", shape, data);
      Tensor b = kernel(a, axis);
      Expect(node, {a}, {b}, "test_cc_flatten_negative_axis" + std::to_string(-axis), {opset},
             "backend-test", registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
