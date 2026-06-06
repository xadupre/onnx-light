// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// RMSNormalization (opset 23) — divides ``X`` by its root-mean-square taken
// over the last ``rank(X) - axis`` dimensions and then multiplies the
// result by a broadcastable ``scale`` tensor. The cases below mirror the
// upstream ``test_rms_normalization_*`` ONNX reference cases (non-expanded
// variants only — the ``_expanded`` variants exercise the function-body
// expansion path and are tracked separately).
// ---------------------------------------------------------------------------

namespace {

// Generates a deterministic float tensor of shape ``shape`` whose data is
// ``i * scale + offset`` for the i-th element in row-major order. This avoids
// relying on platform-dependent RNGs and makes the recorded expected output
// stable.
Tensor MakeFloatTensor(const std::string &name, const std::vector<int64_t> &shape, float scale,
                       float offset) {
  int64_t total = 1;
  for (int64_t d : shape) {
    total *= d;
  }
  std::vector<float> data(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i) {
    data[static_cast<size_t>(i)] = static_cast<float>(i) * scale + offset;
  }
  return Tensor::FromFloat(name, shape, data);
}

// Registers a single RMSNormalization case named ``test_cc_<base>``.
void RegisterCase(std::vector<TestCase> &registry, const kernel::RMSNormalization &kernel,
                  const OpsetId &opset, const std::string &base,
                  const std::vector<int64_t> &x_shape, const std::vector<int64_t> &scale_shape,
                  int64_t axis, bool include_axis_attr, float epsilon, bool include_epsilon_attr) {
  NodeProto node;
  node.set_op_type("RMSNormalization");
  node.add_input("x");
  node.add_input("scale");
  node.add_output("y");
  if (include_axis_attr) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  if (include_epsilon_attr) {
    AddAttribute<float>(node, "epsilon", epsilon);
  }

  Tensor x = MakeFloatTensor("", x_shape, 0.05f, -0.5f);
  Tensor scale = MakeFloatTensor("", scale_shape, 0.02f, 0.5f);
  Tensor y = kernel(x, scale, axis, epsilon);
  Expect(node, {x, scale}, {y}, "test_cc_" + base, {opset}, "backend-test", registry);
}

} // namespace

void RegisterRMSNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(23);
  const kernel::KernelContext ctx{opset};
  const kernel::RMSNormalization rmsnorm_kernel{ctx};

  constexpr float kDefaultEpsilon = 1e-5f;
  constexpr float kAltEpsilon = 0.1f;

  // 2D cases (shape [3, 4]).
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_2d_axis0", {3, 4}, {3, 4},
               /*axis=*/0, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_2d_axis1", {3, 4}, {4},
               /*axis=*/1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_2d_axis_negative_1", {3, 4}, {4},
               /*axis=*/-1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_2d_axis_negative_2", {3, 4},
               {3, 4}, /*axis=*/-2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);

  // 3D cases (shape [2, 3, 5]) with explicit epsilon.
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis0_epsilon", {2, 3, 5},
               {2, 3, 5}, /*axis=*/0, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis1_epsilon", {2, 3, 5},
               {3, 5}, /*axis=*/1, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis2_epsilon", {2, 3, 5},
               {5}, /*axis=*/2, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis_negative_1_epsilon",
               {2, 3, 5}, {5}, /*axis=*/-1, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis_negative_2_epsilon",
               {2, 3, 5}, {3, 5}, /*axis=*/-2, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_3d_axis_negative_3_epsilon",
               {2, 3, 5}, {2, 3, 5}, /*axis=*/-3, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);

  // 4D cases (shape [2, 3, 4, 5]).
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis0", {2, 3, 4, 5},
               {2, 3, 4, 5}, /*axis=*/0, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis1", {2, 3, 4, 5},
               {3, 4, 5}, /*axis=*/1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis2", {2, 3, 4, 5}, {4, 5},
               /*axis=*/2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis3", {2, 3, 4, 5}, {5},
               /*axis=*/3, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis_negative_1",
               {2, 3, 4, 5}, {5}, /*axis=*/-1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis_negative_2",
               {2, 3, 4, 5}, {4, 5}, /*axis=*/-2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis_negative_3",
               {2, 3, 4, 5}, {3, 4, 5}, /*axis=*/-3, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_4d_axis_negative_4",
               {2, 3, 4, 5}, {2, 3, 4, 5}, /*axis=*/-4, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);

  // Default axis (-1) with default epsilon; no attributes set.
  RegisterCase(registry, rmsnorm_kernel, opset, "rms_normalization_default_axis", {2, 3, 4, 5}, {5},
               /*axis=*/-1, /*include_axis_attr=*/false, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
