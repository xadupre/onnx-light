// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Convenience aliases for the non-finite IEEE-754 ``float`` constants.
constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kPosInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Shared 6-element ``x`` and ``y`` operands that, taken together, cover the
// canonical NaN/Inf interactions for an element-wise binary kernel:
//
//   index | x          | y          | example results
//   ------+------------+------------+------------------------------------
//     0   |  NaN       |  1.0       | NaN-propagation through a finite y
//     1   | +Inf       |  2.0       |  Inf + finite, Inf * finite
//     2   | -Inf       | -2.0       | -Inf with negative finite
//     3   | +Inf       | -Inf       | +Inf +/- -Inf, +Inf * -Inf
//     4   |  1.0       |  NaN       | NaN-propagation through a finite x
//     5   |  0.0       | +Inf       | 0 * Inf  -> NaN ; 0 / Inf -> 0
//
// Expected outputs are computed by the in-tree kernel under test so each
// case stays self-consistent with the implementation. The reference values
// follow IEEE-754 arithmetic on ``float``.
const std::vector<float> kXValues = {kNan, kPosInf, kNegInf, kPosInf, 1.0f, 0.0f};
const std::vector<float> kYValues = {1.0f, 2.0f, -2.0f, kNegInf, kNan, kPosInf};
const std::vector<int64_t> kBinaryShape = {6};

template <typename Kernel>
void RegisterBinaryNanInf(std::vector<TestCase> &registry, const char *op_type, int opset_version,
                          const std::string &test_name_stem) {
  const OpsetId opset = DefaultOpset(opset_version);
  const kernel::KernelContext ctx{opset};
  const Kernel kk{ctx};

  // Element-wise case using the shared NaN/Inf operand vectors.
  {
    NodeProto node = MakeNode(op_type, {"x", "y"}, {"z"});

    Tensor x = Tensor::FromFloat("x", kBinaryShape, kXValues);
    Tensor y = Tensor::FromFloat("y", kBinaryShape, kYValues);
    Tensor z = kk(x, y);

    const std::string name = "test_cc_" + test_name_stem + "_nan_inf";
    Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry, "nan_inf");
  }

  // Scalar-broadcast case: combining a NaN scalar against a vector of
  // finite values must propagate NaN everywhere.
  {
    NodeProto node = MakeNode(op_type, {"x", "y"}, {"z"});

    Tensor x = Tensor::FromFloat("x", {4}, {1.0f, -1.0f, 2.0f, -2.0f});
    Tensor y = Tensor::FromFloat("y", {}, {kNan});
    Tensor z = kk(x, y);

    const std::string name = "test_cc_" + test_name_stem + "_nan_inf_bcast_nan_scalar";
    Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry, "nan_inf");
  }

  // Scalar-broadcast case: combining a +Inf scalar against a vector of
  // finite values produces +/-Inf (sign depends on the operator and the
  // sign of the vector value); the kernel computes the reference.
  {
    NodeProto node = MakeNode(op_type, {"x", "y"}, {"z"});

    Tensor x = Tensor::FromFloat("x", {4}, {1.0f, -1.0f, 2.0f, -2.0f});
    Tensor y = Tensor::FromFloat("y", {}, {kPosInf});
    Tensor z = kk(x, y);

    const std::string name = "test_cc_" + test_name_stem + "_nan_inf_bcast_inf_scalar";
    Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry, "nan_inf");
  }
}

} // namespace

void RegisterAddNanInfCases(std::vector<TestCase> &registry) {
  RegisterBinaryNanInf<kernel::Add>(registry, "Add", 14, "add");
}

void RegisterSubNanInfCases(std::vector<TestCase> &registry) {
  RegisterBinaryNanInf<kernel::Sub>(registry, "Sub", 14, "sub");
}

void RegisterMulNanInfCases(std::vector<TestCase> &registry) {
  RegisterBinaryNanInf<kernel::Mul>(registry, "Mul", 14, "mul");
}

void RegisterDivNanInfCases(std::vector<TestCase> &registry) {
  RegisterBinaryNanInf<kernel::Div>(registry, "Div", 14, "div");
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
