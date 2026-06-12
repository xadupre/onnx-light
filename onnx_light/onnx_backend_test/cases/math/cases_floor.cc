// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Floor — y = floor(x) (since opset 6, widened to bfloat16 in opset 13).
// Registers both a deterministic ``test_cc_floor`` case and the upstream
// ONNX backend test cases (``test_floor_example`` and ``test_floor``)
// mirrored from ``onnx.backend.test.case.node.floor.Floor``.
// ---------------------------------------------------------------------------
void RegisterFloorCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Floor floor_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Floor");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.2f, 2.0f});
    Tensor y = floor_kernel(x);

    Expect(node, {x}, {y}, "test_cc_floor", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Floor`` operator (mirror the
  // ``onnx.backend.test.case.node.floor.Floor`` Python class).
  //
  // From Floor.export(): ``test_floor_example`` uses x = [-1.5, 1.2, 2].
  {
    NodeProto node;
    node.set_op_type("Floor");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.5f, 1.2f, 2.0f});
    Tensor y = floor_kernel(x);
    Expect(node, {x}, {y}, "test_floor_example", {opset}, "backend-test", registry);
  }
  // From Floor.export(): ``test_floor`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Floor");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, /*seed=*/1));
    Tensor y = floor_kernel(x);
    Expect(node, {x}, {y}, "test_floor", {opset}, "backend-test", registry);
  }
  // FLOAT16
  {
    NodeProto node;
    node.set_op_type("Floor");
    node.add_input("x");
    node.add_output("y");

    Tensor x = kernel::MakeFloat16Tensor("", {2, 3}, {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.7f});
    Tensor y = floor_kernel(x);
    Expect(node, {x}, {y}, "test_cc_floor_float16", {opset}, "backend-test", registry);
  }

  // BFLOAT16
  {
    NodeProto node;
    node.set_op_type("Floor");
    node.add_input("x");
    node.add_output("y");

    std::vector<float> vals = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.7f};
    std::vector<uint8_t> raw(vals.size() * sizeof(uint16_t));
    auto *dst = reinterpret_cast<uint16_t *>(raw.data());
    for (size_t i = 0; i < vals.size(); ++i)
      dst[i] = kernel::FloatToBfloat16Bits(vals[i]);
    Tensor x("", static_cast<int32_t>(DataType::BFLOAT16), {2, 3}, std::move(raw));
    Tensor y = floor_kernel(x);
    Expect(node, {x}, {y}, "test_cc_floor_bfloat16", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
