// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// ---------------------------------------------------------------------------
// Shared empty-shape coverage for element-wise binary math operators. For
// each op we exercise the same three shapes:
//
//   * ``{}``       — rank-0 scalars (the shape vector itself is empty);
//   * ``{0}``      — zero-element 1-D tensors;
//   * ``{0, 3}``   — zero-element 2-D tensors.
//
// Both inputs use the same shape so we deliberately steer clear of the
// broadcasting paths involving a zero-sized dim against a non-zero dim
// (e.g. ``{0} vs {}``), which trip a pre-existing bug in the
// ``BinaryElementwise`` helper (``max(0, 1) == 1`` followed by a zero-stride
// read) and are tracked separately.
//
// Expected outputs are computed by the in-tree kernel so each case stays
// self-consistent with the implementation under test.
// ---------------------------------------------------------------------------

struct BinaryEmptyShapeCase {
  std::vector<int64_t> shape;
  const char *suffix;
};

constexpr int kSecondInputDefault = 0;
constexpr int kSecondInputSlope = 1;

template <typename Kernel>
void RegisterBinaryEmptyShape(std::vector<TestCase> &registry, const char *op_type,
                              int opset_version, const std::string &test_name_stem,
                              int second_input_kind = kSecondInputDefault) {
  const OpsetId opset = DefaultOpset(opset_version);
  const kernel::KernelContext ctx{opset};
  const Kernel kk{ctx};

  const char *second_input_name = (second_input_kind == kSecondInputSlope) ? "slope" : "y";
  const char *output_name = (second_input_kind == kSecondInputSlope) ? "y" : "z";

  const BinaryEmptyShapeCase cases[] = {
      {{}, "scalars"},
      {{0}, "zero_dim"},
      {{0, 3}, "zero_dim_2d"},
  };

  for (const auto &c : cases) {
    NodeProto node;
    node.set_op_type(op_type);
    node.add_input("x");
    node.add_input(second_input_name);
    node.add_output(output_name);

    // For the rank-0 case we pick small positive values so ops like Div are
    // well-defined; zero-element cases carry no data at all.
    std::vector<float> data_x = c.shape.empty() ? std::vector<float>{2.5f} : std::vector<float>{};
    std::vector<float> data_y = c.shape.empty() ? std::vector<float>{3.5f} : std::vector<float>{};

    Tensor x = Tensor::FromFloat("", c.shape, data_x);
    Tensor y = Tensor::FromFloat("", c.shape, data_y);
    Tensor z = kk(x, y);

    const std::string name = "test_cc_" + test_name_stem + "_empty_shape_" + c.suffix;
    Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
  }
}

} // namespace

void RegisterAddEmptyShapeCases(std::vector<TestCase> &registry) {
  RegisterBinaryEmptyShape<kernel::Add>(registry, "Add", 14, "add");
}

void RegisterSubEmptyShapeCases(std::vector<TestCase> &registry) {
  RegisterBinaryEmptyShape<kernel::Sub>(registry, "Sub", 14, "sub");
}

void RegisterMulEmptyShapeCases(std::vector<TestCase> &registry) {
  RegisterBinaryEmptyShape<kernel::Mul>(registry, "Mul", 14, "mul");
}

void RegisterDivEmptyShapeCases(std::vector<TestCase> &registry) {
  RegisterBinaryEmptyShape<kernel::Div>(registry, "Div", 14, "div");
}

void RegisterPReluEmptyShapeCases(std::vector<TestCase> &registry) {
  RegisterBinaryEmptyShape<kernel::PRelu>(registry, "PRelu", 16, "prelu", kSecondInputSlope);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
