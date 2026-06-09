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

// Inputs and outputs mirror the upstream ``GridSample`` reference tests
// in ``onnx/backend/test/case/node/gridsample.py``. The values are
// copied verbatim from that file (rounded to four decimal places where
// the upstream file does so) so that our registered cases match the
// upstream backend-test corpus exactly.

// 4x4 image used by ``test_gridsample``.
Tensor MakeX_4x4() {
  return Tensor::FromFloat("X", {1, 1, 4, 4},
                           {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                            11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
}

// 6x6 sampling grid used by ``test_gridsample``.
Tensor MakeGrid_6x6() {
  std::vector<float> g;
  g.reserve(1 * 6 * 6 * 2);
  const float ys[6] = {-1.0f, -0.6f, -0.2f, 0.2f, 0.6f, 1.0f};
  const float xs[6] = {-1.0f, -0.6f, -0.2f, 0.2f, 0.6f, 1.0f};
  for (float y : ys) {
    for (float x : xs) {
      g.push_back(x);
      g.push_back(y);
    }
  }
  return Tensor::FromFloat("Grid", {1, 6, 6, 2}, g);
}

// 3x2 image shared by all ``test_gridsample_*_padding`` and
// ``test_gridsample_bilinear`` / ``_nearest`` / ``_bicubic`` cases.
Tensor MakeX_3x2() {
  return Tensor::FromFloat("X", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
}

// 2x4 grid used by ``test_gridsample_*_padding``.
Tensor MakeGrid_2x4_Pad() {
  return Tensor::FromFloat("Grid", {1, 2, 4, 2},
                           {-10.0f, -10.0f, -5.0f, -5.0f, -0.2f, -0.2f, 10.0f, 10.0f, 10.0f, 10.0f,
                            -0.2f, -0.2f, 5.0f, 5.0f, 10.0f, 10.0f});
}

// 2x4 grid used by ``test_gridsample_bilinear`` / ``_aligncorners_true`` /
// ``_nearest`` / ``_bicubic``.
Tensor MakeGrid_2x4_Mode() {
  return Tensor::FromFloat("Grid", {1, 2, 4, 2},
                           {-1.0f, -1.0f, -0.5f, -0.5f, -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, -0.2f,
                            -0.2f, 0.5f, 0.5f, 1.0f, 1.0f});
}

// 2x4 grid used by ``test_gridsample_*_align_corners_*_additional_1``.
Tensor MakeGrid_2x4_Additional() {
  return Tensor::FromFloat("Grid", {1, 2, 4, 2},
                           {-1.0f, -0.8f, -0.6f, -0.5f, -0.1f, -0.2f, 0.7f, 0.0f, 0.0f, 0.4f, 0.2f,
                            -0.2f, -0.3f, 0.5f, -1.0f, 1.0f});
}

// 3-D volume used by ``test_gridsample_volumetric_*``. Shape [N, C, D, H, W]
// = [1, 1, 3, 2, 2].
Tensor MakeX_Volumetric() {
  return Tensor::FromFloat(
      "X", {1, 1, 3, 2, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
}

// 3-D sampling grid used by ``test_gridsample_volumetric_*``. Shape
// [N, D_out, H_out, W_out, 3] = [1, 2, 4, 2, 3].
Tensor MakeGrid_Volumetric() {
  return Tensor::FromFloat("Grid", {1, 2, 4, 2, 3},
                           {-1.0f, -1.0f, -1.0f, -1.0f, -0.5f, 0.3f,  -0.5f, -0.5f, -0.5f, 1.0f,
                            -0.6f, -1.0f, -0.2f, -0.2f, -0.2f, 0.4f,  0.2f,  0.6f,  0.0f,  0.0f,
                            0.0f,  -1.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  -1.0f, 1.0f,  0.0f,
                            -0.2f, -0.2f, -0.2f, 1.0f,  0.4f,  -0.2f, 0.5f,  0.5f,  0.5f,  -1.0f,
                            -0.8f, 0.8f,  1.0f,  1.0f,  1.0f,  0.4f,  0.6f,  -0.3f});
}

void AddCase(std::vector<TestCase> &registry, const OpsetId &opset, const std::string &name,
             const std::string &mode, const std::string &padding_mode, int64_t align_corners,
             const Tensor &X, const Tensor &Grid, const std::vector<int64_t> &y_shape,
             const std::vector<float> &y_values) {
  NodeProto node;
  node.set_op_type("GridSample");
  node.add_input("X");
  node.add_input("Grid");
  node.add_output("Y");
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  if (!padding_mode.empty()) {
    AddAttribute<std::string>(node, "padding_mode", padding_mode);
  }
  if (align_corners != 0) {
    AddAttribute<int64_t>(node, "align_corners", align_corners);
  }
  Tensor Y = Tensor::FromFloat("Y", y_shape, y_values);
  Expect(node, {X, Grid}, {Y}, name, {opset}, "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// GridSample — samples ``X`` at the positions given by ``Grid``. Available
// since opset 16 in the ai.onnx domain (extended to N-D in opset 20).
// Inputs/outputs match the upstream cases in
// ``onnx/backend/test/case/node/gridsample.py``.
// ---------------------------------------------------------------------------
void RegisterGridSampleCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);

  // ---- test_gridsample ----------------------------------------------------
  AddCase(registry, opset, "test_gridsample", "linear", "zeros", /*align_corners=*/0, MakeX_4x4(),
          MakeGrid_6x6(), {1, 1, 6, 6},
          {0.0000f, 0.1500f,  0.5500f,  0.9500f,  1.3500f,  0.7500f, //
           0.6000f, 1.5000f,  2.3000f,  3.1000f,  3.9000f,  2.1000f, //
           2.2000f, 4.7000f,  5.5000f,  6.3000f,  7.1000f,  3.7000f, //
           3.8000f, 7.9000f,  8.7000f,  9.5000f,  10.3000f, 5.3000f, //
           5.4000f, 11.1000f, 11.9000f, 12.7000f, 13.5000f, 6.9000f, //
           3.0000f, 6.1500f,  6.5500f,  6.9500f,  7.3500f,  3.7500f});

  // ---- padding-mode cases (test_gridsample_*_padding) --------------------
  AddCase(registry, opset, "test_gridsample_zeros_padding", "", "zeros", 0, MakeX_3x2(),
          MakeGrid_2x4_Pad(), {1, 1, 2, 4},
          {0.0000f, 0.0000f, 1.7000f, 0.0000f, 0.0000f, 1.7000f, 0.0000f, 0.0000f});
  AddCase(registry, opset, "test_gridsample_border_padding", "", "border", 0, MakeX_3x2(),
          MakeGrid_2x4_Pad(), {1, 1, 2, 4},
          {0.0000f, 0.0000f, 1.7000f, 5.0000f, 5.0000f, 1.7000f, 5.0000f, 5.0000f});
  AddCase(registry, opset, "test_gridsample_reflection_padding", "", "reflection", 0, MakeX_3x2(),
          MakeGrid_2x4_Pad(), {1, 1, 2, 4},
          {2.5000f, 0.0000f, 1.7000f, 2.5000f, 2.5000f, 1.7000f, 5.0000f, 2.5000f});

  // ---- mode / align_corners cases ----------------------------------------
  AddCase(registry, opset, "test_gridsample_bilinear", "linear", "", 0, MakeX_3x2(),
          MakeGrid_2x4_Mode(), {1, 1, 2, 4},
          {0.0000f, 0.5000f, 1.7000f, 2.5000f, 2.5000f, 1.7000f, 4.5000f, 1.2500f});
  AddCase(registry, opset, "test_gridsample_aligncorners_true", "linear", "", 1, MakeX_3x2(),
          MakeGrid_2x4_Mode(), {1, 1, 2, 4},
          {0.0000f, 1.2500f, 2.0000f, 2.5000f, 2.5000f, 2.0000f, 3.7500f, 5.0000f});
  AddCase(registry, opset, "test_gridsample_nearest", "nearest", "", 0, MakeX_3x2(),
          MakeGrid_2x4_Mode(), {1, 1, 2, 4}, {0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 2.0f, 5.0f, 0.0f});
  AddCase(registry, opset, "test_gridsample_bicubic", "cubic", "", 0, MakeX_3x2(),
          MakeGrid_2x4_Mode(), {1, 1, 2, 4},
          {-0.1406f, 0.3828f, 1.7556f, 2.9688f, 2.9688f, 1.7556f, 5.1445f, 1.3906f});

  // ---- additional align_corners cases -----------------------------------
  AddCase(registry, opset, "test_gridsample_nearest_align_corners_0_additional_1", "nearest", "", 0,
          MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
          {0.0f, 0.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 4.0f});
  AddCase(registry, opset, "test_gridsample_nearest_align_corners_1_additional_1", "nearest", "", 1,
          MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
          {0.0f, 0.0f, 2.0f, 3.0f, 2.0f, 3.0f, 4.0f, 4.0f});
  AddCase(registry, opset, "test_gridsample_bilinear_align_corners_0_additional_1", "linear", "", 0,
          MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
          {0.0000f, 0.4500f, 1.8000f, 2.4000f, 3.7000f, 2.1000f, 3.7000f, 1.0000f});
  AddCase(registry, opset, "test_gridsample_bilinear_align_corners_1_additional_1", "linear", "", 1,
          MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
          {0.4000f, 1.2000f, 2.0500f, 2.8500f, 3.3000f, 2.2000f, 3.3500f, 4.0000f});
  AddCase(
      registry, opset, "test_gridsample_bicubic_align_corners_0_additional_1", "cubic", "", 0,
      MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
      {-0.173250f, 0.284265f, 1.923106f, 2.568000f, 5.170375f, 2.284414f, 4.744844f, 1.046875f});
  AddCase(registry, opset, "test_gridsample_bicubic_align_corners_1_additional_1", "cubic", "", 1,
          MakeX_3x2(), MakeGrid_2x4_Additional(), {1, 1, 2, 4},
          {0.304001f, 1.128750f, 2.266270f, 3.144844f, 4.531500f, 2.455360f, 4.599819f, 4.000000f});

  // ---- volumetric (5-D) cases (opset 20+) -------------------------------
  AddCase(registry, opset, "test_gridsample_volumetric_nearest_align_corners_0", "nearest", "", 0,
          MakeX_Volumetric(), MakeGrid_Volumetric(), {1, 1, 2, 4, 2},
          {1.0f, 5.0f, 1.0f, 0.0f, 5.0f, 12.0f, 5.0f, 5.0f, 5.0f, 0.0f, 5.0f, 0.0f, 12.0f, 9.0f,
           0.0f, 8.0f});
  AddCase(registry, opset, "test_gridsample_volumetric_nearest_align_corners_1", "nearest", "", 1,
          MakeX_Volumetric(), MakeGrid_Volumetric(), {1, 1, 2, 4, 2},
          {1.0f, 5.0f, 1.0f, 2.0f, 5.0f, 12.0f, 5.0f, 5.0f, 5.0f, 7.0f, 5.0f, 8.0f, 12.0f, 9.0f,
           12.0f, 8.0f});
  AddCase(registry, opset, "test_gridsample_volumetric_bilinear_align_corners_0", "linear", "", 0,
          MakeX_Volumetric(), MakeGrid_Volumetric(), {1, 1, 2, 4, 2},
          {0.1250f, 3.4000f, 2.0000f, 0.4500f, 4.7000f, 10.9000f, 6.5000f, 3.0000f, 6.5000f,
           1.7500f, 4.7000f, 3.3000f, 11.0000f, 2.5200f, 1.5000f, 5.4900f});
  AddCase(registry, opset, "test_gridsample_volumetric_bilinear_align_corners_1", "linear", "", 1,
          MakeX_Volumetric(), MakeGrid_Volumetric(), {1, 1, 2, 4, 2},
          {1.0000f, 6.7000f, 3.7500f, 2.4000f, 5.4000f, 9.3000f, 6.5000f, 6.0000f, 6.5000f, 7.0000f,
           5.4000f, 6.6000f, 9.2500f, 8.4000f, 12.0000f, 6.1000f});
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
