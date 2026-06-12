// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Returns the deterministic batch of two 2-D affine matrices produced by
// ``create_theta_2d()`` in ``onnx/backend/test/case/node/affinegrid.py``.
// The values were computed once by the upstream Python helper and
// hard-coded here to avoid re-running numpy at build time. Shape (2, 2, 3).
Tensor MakeUpstreamTheta2D() {
  const std::vector<float> values = {
      1.0889444f,  -3.2880466f,  5.0f,  //
      2.0223253f,  1.0960155f,   -3.3f, //
      0.83578837f, -0.55442286f, 2.5f,  //
      0.78762794f, 0.8397114f,   1.1f,  //
  };
  return Tensor::FromFloat("", {2, 2, 3}, values);
}

// Returns the deterministic batch of two 3-D affine matrices produced by
// ``create_theta_3d()`` in ``onnx/backend/test/case/node/affinegrid.py``.
// Shape (2, 3, 4).
Tensor MakeUpstreamTheta3D() {
  const std::vector<float> values = {
      2.6830733f,   -0.7943316f,  0.21829216f,  5.0f,  //
      0.62225395f,  3.2880466f,   -0.53033006f, -3.3f, //
      0.24721935f,  1.7241772f,   0.07809311f,  -1.1f, //
      -0.35552558f, 1.0044229f,   1.3995191f,   2.5f,  //
      0.17578839f,  0.060288567f, -0.9240381f,  1.1f,  //
      -1.1f,        -0.45f,       0.45f,        2.2f,  //
  };
  return Tensor::FromFloat("", {2, 3, 4}, values);
}

} // namespace

// ---------------------------------------------------------------------------
// AffineGrid — generates a sampling flow field from a batch of affine
// matrices ``theta`` and a target ``size``. Since opset 20 in the ai.onnx
// domain (matches torch.nn.functional.affine_grid; see
// ``onnx/backend/test/case/node/affinegrid.py``).
//
// Cases registered (names mirror the upstream
// ``onnx/backend/test/data/node`` folders):
//
//   * ``test_affine_grid_2d`` — 2-D field with align_corners=0.
//   * ``test_affine_grid_2d_align_corners`` — 2-D field with
//     align_corners=1.
//   * ``test_affine_grid_3d`` — 3-D field with align_corners=0.
//   * ``test_affine_grid_3d_align_corners`` — 3-D field with
//     align_corners=1.
//
// The inputs (theta, size) and node attributes match the upstream cases
// exactly; the expected outputs are produced by our reference kernel,
// which mirrors the upstream Python reference in
// ``onnx/reference/ops/op_affine_grid.py``.
// ---------------------------------------------------------------------------
void RegisterAffineGridCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::AffineGrid ag_kernel{ctx};

  // Helper that registers one case for the requested rank
  // (``size_dims`` is {N, C, H, W} for 2D or {N, C, D, H, W} for 3D).
  auto register_case = [&](const std::string &case_name, const Tensor &theta,
                           const std::vector<int64_t> &size_dims, int64_t align_corners) {
    NodeProto node;
    node.set_op_type("AffineGrid");
    node.add_input("theta");
    node.add_input("size");
    node.add_output("grid");
    AddAttribute<int64_t>(node, "align_corners", align_corners);

    Tensor size = Tensor::FromInt64("", {static_cast<int64_t>(size_dims.size())}, size_dims);

    kernel::AffineGrid::Attributes attrs;
    attrs.align_corners = align_corners;
    Tensor grid = ag_kernel(theta, size, attrs);

    Expect(node, {theta, size}, {grid}, case_name, {opset}, "backend-test", registry);
  };

  // Upstream ``test_affine_grid_2d`` cases: theta=(2,2,3), size=(N=2, C=3,
  // H=5, W=6); produces a (2, 5, 6, 2) flow field.
  const Tensor theta_2d = MakeUpstreamTheta2D();
  register_case("test_affine_grid_2d", theta_2d, {2, 3, 5, 6}, /*align_corners=*/0);
  register_case("test_affine_grid_2d_align_corners", theta_2d, {2, 3, 5, 6},
                /*align_corners=*/1);

  // Upstream ``test_affine_grid_3d`` cases: theta=(2,3,4), size=(N=2, C=3,
  // D=4, H=5, W=6); produces a (2, 4, 5, 6, 3) flow field.
  const Tensor theta_3d = MakeUpstreamTheta3D();
  register_case("test_affine_grid_3d", theta_3d, {2, 3, 4, 5, 6}, /*align_corners=*/0);
  register_case("test_affine_grid_3d_align_corners", theta_3d, {2, 3, 4, 5, 6},
                /*align_corners=*/1);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
