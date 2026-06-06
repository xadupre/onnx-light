// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// LayerNormalization (opset 17) — standardizes ``X`` over the last
// ``rank(X) - axis`` dimensions, then applies a per-element affine transform
// using ``Scale`` and the optional bias ``B``. The cases below mirror the
// upstream ``test_layer_normalization_*`` ONNX reference cases (non-expanded
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

// Registers a single LayerNormalization case named ``test_cc_<base>`` with
// inputs ``[X, W, B]`` and outputs ``[Y, Mean, InvStdDev]``.
void RegisterCase(std::vector<TestCase> &registry, const kernel::LayerNormalization &kernel,
                  const OpsetId &opset, const std::string &base,
                  const std::vector<int64_t> &x_shape, int64_t axis, bool include_axis_attr,
                  float epsilon, bool include_epsilon_attr) {
  // Normalized shape = X.shape[axis:].
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  std::vector<int64_t> normalized_shape(x_shape.begin() + resolved_axis, x_shape.end());

  NodeProto node;
  node.set_op_type("LayerNormalization");
  node.add_input("X");
  node.add_input("W");
  node.add_input("B");
  node.add_output("Y");
  node.add_output("Mean");
  node.add_output("InvStdDev");
  if (include_axis_attr) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  if (include_epsilon_attr) {
    AddAttribute<float>(node, "epsilon", epsilon);
  }

  Tensor x = MakeFloatTensor("", x_shape, 0.05f, -0.5f);
  Tensor w = MakeFloatTensor("", normalized_shape, 0.02f, 0.5f);
  Tensor b = MakeFloatTensor("", normalized_shape, 0.01f, -0.25f);
  auto [y, mean, inv_std_dev] = kernel(x, w, b, axis, epsilon);
  Expect(node, {x, w, b}, {y, mean, inv_std_dev}, "test_cc_" + base, {opset}, "backend-test",
         registry);
}

} // namespace

void RegisterLayerNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(17);
  const kernel::KernelContext ctx{opset};
  const kernel::LayerNormalization layernorm_kernel{ctx};

  constexpr float kDefaultEpsilon = 1e-5f;
  constexpr float kAltEpsilon = 0.1f;

  // 2D cases (shape [3, 4]).
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_2d_axis0", {3, 4},
               /*axis=*/0, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_2d_axis1", {3, 4},
               /*axis=*/1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_2d_axis_negative_1", {3, 4},
               /*axis=*/-1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_2d_axis_negative_2", {3, 4},
               /*axis=*/-2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);

  // 3D cases (shape [2, 3, 5]) with explicit epsilon.
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis0_epsilon", {2, 3, 5},
               /*axis=*/0, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis1_epsilon", {2, 3, 5},
               /*axis=*/1, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis2_epsilon", {2, 3, 5},
               /*axis=*/2, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis_negative_1_epsilon",
               {2, 3, 5}, /*axis=*/-1, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis_negative_2_epsilon",
               {2, 3, 5}, /*axis=*/-2, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_3d_axis_negative_3_epsilon",
               {2, 3, 5}, /*axis=*/-3, /*include_axis_attr=*/true, kAltEpsilon,
               /*include_epsilon_attr=*/true);

  // 4D cases (shape [2, 3, 4, 5]).
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis0", {2, 3, 4, 5},
               /*axis=*/0, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis1", {2, 3, 4, 5},
               /*axis=*/1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis2", {2, 3, 4, 5},
               /*axis=*/2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis3", {2, 3, 4, 5},
               /*axis=*/3, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis_negative_1",
               {2, 3, 4, 5}, /*axis=*/-1, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis_negative_2",
               {2, 3, 4, 5}, /*axis=*/-2, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis_negative_3",
               {2, 3, 4, 5}, /*axis=*/-3, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_4d_axis_negative_4",
               {2, 3, 4, 5}, /*axis=*/-4, /*include_axis_attr=*/true, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);

  // Default axis (-1) with default epsilon; no attributes set.
  RegisterCase(registry, layernorm_kernel, opset, "layer_normalization_default_axis", {2, 3, 4, 5},
               /*axis=*/-1, /*include_axis_attr=*/false, kDefaultEpsilon,
               /*include_epsilon_attr=*/false);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
