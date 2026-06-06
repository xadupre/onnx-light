// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Resize — nearest interpolation. Available since opset 10 (replaces the
// deprecated ``Upsample``); the ``axes`` attribute and the
// ``keep_aspect_ratio_policy`` attribute were added in opset 18. Mirrors
// the subset of upstream resize node tests in
// ``onnx/backend/test/case/node/resize.py`` that the reference implementation
// supports (``nearest`` mode with the ``half_pixel`` / ``align_corners`` /
// ``asymmetric`` coordinate transformations).
// ---------------------------------------------------------------------------

namespace {

// Optional attribute helpers --------------------------------------------------

void MaybeAddString(NodeProto &node, const char *name, const std::string &value) {
  if (!value.empty()) {
    AddAttribute<std::string>(node, name, value);
  }
}

void MaybeAddInts(NodeProto &node, const char *name, const std::vector<int64_t> &value) {
  if (!value.empty()) {
    AddAttribute<std::vector<int64_t>>(node, name, value);
  }
}

// Builds a Resize node taking ``(X, roi, scales)`` with ``roi`` as the empty
// input name (matching the v11+ schema for the simple "scales only" case).
NodeProto MakeResizeNodeScales(const std::string &mode = "nearest",
                               const std::string &coord_mode = "asymmetric",
                               const std::string &nearest_mode = "",
                               const std::vector<int64_t> &axes = {}) {
  NodeProto node;
  node.set_op_type("Resize");
  node.add_input("X");
  node.add_input(""); // roi (unused)
  node.add_input("scales");
  node.add_output("Y");
  MaybeAddString(node, "mode", mode);
  MaybeAddString(node, "coordinate_transformation_mode", coord_mode);
  MaybeAddString(node, "nearest_mode", nearest_mode);
  MaybeAddInts(node, "axes", axes);
  return node;
}

// Builds a Resize node taking ``(X, roi, scales, sizes)`` with ``roi`` and
// ``scales`` as empty input names (matching the v11+ schema for the "sizes
// only" case).
NodeProto MakeResizeNodeSizes(const std::string &mode = "nearest",
                              const std::string &coord_mode = "asymmetric",
                              const std::string &nearest_mode = "",
                              const std::vector<int64_t> &axes = {},
                              const std::string &keep_aspect_ratio_policy = "") {
  NodeProto node;
  node.set_op_type("Resize");
  node.add_input("X");
  node.add_input(""); // roi (unused)
  node.add_input(""); // scales (unused)
  node.add_input("sizes");
  node.add_output("Y");
  MaybeAddString(node, "mode", mode);
  MaybeAddString(node, "coordinate_transformation_mode", coord_mode);
  MaybeAddString(node, "nearest_mode", nearest_mode);
  MaybeAddInts(node, "axes", axes);
  MaybeAddString(node, "keep_aspect_ratio_policy", keep_aspect_ratio_policy);
  return node;
}

Tensor MakeScalesTensor(const std::vector<float> &scales) {
  return Tensor::FromFloat("", {static_cast<int64_t>(scales.size())}, scales);
}

Tensor MakeSizesTensor(const std::vector<int64_t> &sizes) {
  return Tensor::FromInt64("", {static_cast<int64_t>(sizes.size())}, sizes);
}

} // namespace

void RegisterResizeCases(std::vector<TestCase> &registry) {
  const OpsetId opset13 = DefaultOpset(13);
  const OpsetId opset18 = DefaultOpset(18);
  const kernel::KernelContext ctx{opset13};
  const kernel::Resize resize_kernel{ctx};

  // test_cc_resize_upsample_scales_nearest_asymmetric — NCHW input shape
  // [1, 1, 2, 2] upsampled by [1, 1, 2, 3] using nearest mode and the
  // asymmetric coordinate transformation; expected output is the upstream
  // reference for ``test_resize_upsample_scales_nearest`` rerun with
  // coordinate_transformation_mode == "asymmetric".
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 2.0f, 3.0f});
    kernel::Resize::Attributes attrs;
    attrs.mode = "nearest";
    attrs.coordinate_transformation_mode = "asymmetric";
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", "asymmetric"), {X, scales}, {Y},
           "test_cc_resize_upsample_scales_nearest_asymmetric", {opset13}, "backend-test",
           registry);
  }

  // test_cc_resize_upsample_scales_nearest_1d — minimal 1-D nearest case
  // (asymmetric mode), exercising the strided index mapping independently
  // of the NCHW layout.
  {
    const Tensor X = Tensor::FromFloat("", {3}, {10.0f, 20.0f, 30.0f});
    const Tensor scales = MakeScalesTensor({2.0f});
    kernel::Resize::Attributes attrs;
    attrs.mode = "nearest";
    attrs.coordinate_transformation_mode = "asymmetric";
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", "asymmetric"), {X, scales}, {Y},
           "test_cc_resize_upsample_scales_nearest_1d", {opset13}, "backend-test", registry);
  }

  // test_cc_resize_upsample_sizes_nearest_asymmetric — same NCHW input, but
  // the target shape is given via the ``sizes`` input rather than via
  // ``scales``. Output is [1, 1, 4, 6] matching the scales [1, 1, 2, 3].
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 4, 6});
    kernel::Resize::Attributes attrs;
    attrs.mode = "nearest";
    attrs.coordinate_transformation_mode = "asymmetric";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", "asymmetric"), {X, sizes}, {Y},
           "test_cc_resize_upsample_sizes_nearest_asymmetric", {opset13}, "backend-test", registry);
  }

  // ---------------------------------------------------------------------------
  // The cases below mirror the upstream ONNX node-level tests in
  // ``onnx/backend/test/case/node/resize.py``. The ``mode="nearest"`` cases
  // are all supported by the reference kernel; we omit the linear/cubic/
  // antialias/tf_crop_and_resize cases for which only ``nearest`` interpolation
  // is implemented here.
  // ---------------------------------------------------------------------------

  // test_resize_upsample_scales_nearest — opset 13 defaults
  // (mode=nearest, coordinate_transformation_mode=half_pixel,
  // nearest_mode=round_prefer_floor).
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 2.0f, 3.0f});
    kernel::Resize::Attributes attrs; // defaults: half_pixel + round_prefer_floor
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", /*coord_mode=*/""), {X, scales}, {Y},
           "test_resize_upsample_scales_nearest", {opset13}, "backend-test", registry);
  }

  // test_resize_downsample_scales_nearest — half_pixel + round_prefer_floor.
  {
    const Tensor X =
        Tensor::FromFloat("", {1, 1, 2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 0.6f, 0.6f});
    kernel::Resize::Attributes attrs;
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", /*coord_mode=*/""), {X, scales}, {Y},
           "test_resize_downsample_scales_nearest", {opset13}, "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest — sizes path, half_pixel default.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 7, 8});
    kernel::Resize::Attributes attrs;
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/""), {X, sizes}, {Y},
           "test_resize_upsample_sizes_nearest", {opset13}, "backend-test", registry);
  }

  // test_resize_downsample_sizes_nearest — sizes path, half_pixel default.
  {
    const Tensor X =
        Tensor::FromFloat("", {1, 1, 2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 1, 3});
    kernel::Resize::Attributes attrs;
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/""), {X, sizes}, {Y},
           "test_resize_downsample_sizes_nearest", {opset13}, "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_floor_align_corners — align_corners +
  // floor.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 4, 4},
                                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                        11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 8, 8});
    kernel::Resize::Attributes attrs;
    attrs.coordinate_transformation_mode = "align_corners";
    attrs.nearest_mode = "floor";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", "align_corners", "floor"), {X, sizes}, {Y},
           "test_resize_upsample_sizes_nearest_floor_align_corners", {opset13}, "backend-test",
           registry);
  }

  // test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric — asymmetric +
  // round_prefer_ceil.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 4, 4},
                                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                        11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 8, 8});
    kernel::Resize::Attributes attrs;
    attrs.coordinate_transformation_mode = "asymmetric";
    attrs.nearest_mode = "round_prefer_ceil";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", "asymmetric", "round_prefer_ceil"), {X, sizes}, {Y},
           "test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", {opset13},
           "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_ceil_half_pixel — half_pixel + ceil.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 4, 4},
                                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                        11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    const Tensor sizes = MakeSizesTensor({1, 1, 8, 8});
    kernel::Resize::Attributes attrs;
    attrs.coordinate_transformation_mode = "half_pixel";
    attrs.nearest_mode = "ceil";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", "half_pixel", "ceil"), {X, sizes}, {Y},
           "test_resize_upsample_sizes_nearest_ceil_half_pixel", {opset13}, "backend-test",
           registry);
  }

  // test_resize_upsample_scales_nearest_axes_2_3 — opset 18 ``axes`` attribute.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({2.0f, 3.0f});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes),
           {X, scales}, {Y}, "test_resize_upsample_scales_nearest_axes_2_3", {opset18},
           "backend-test", registry);
  }

  // test_resize_upsample_scales_nearest_axes_3_2 — opset 18 ``axes`` attribute,
  // with axes specified in non-ascending order.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({3.0f, 2.0f});
    kernel::Resize::Attributes attrs;
    attrs.axes = {3, 2};
    const Tensor Y = resize_kernel(X, scales, attrs);
    Expect(MakeResizeNodeScales("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes),
           {X, scales}, {Y}, "test_resize_upsample_scales_nearest_axes_3_2", {opset18},
           "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_axes_2_3 — opset 18 sizes + axes.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({7, 8});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes),
           {X, sizes}, {Y}, "test_resize_upsample_sizes_nearest_axes_2_3", {opset18},
           "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_axes_3_2 — opset 18 sizes + axes
  // (non-ascending order).
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({8, 7});
    kernel::Resize::Attributes attrs;
    attrs.axes = {3, 2};
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes),
           {X, sizes}, {Y}, "test_resize_upsample_sizes_nearest_axes_3_2", {opset18},
           "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_not_larger — opset 18
  // ``keep_aspect_ratio_policy=not_larger``: scales by min(sizes / in_size)
  // across axes (7x7 output for a 2x2 input requested at 7x8).
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({7, 8});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.keep_aspect_ratio_policy = "not_larger";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes,
                               attrs.keep_aspect_ratio_policy),
           {X, sizes}, {Y}, "test_resize_upsample_sizes_nearest_not_larger", {opset18},
           "backend-test", registry);
  }

  // test_resize_upsample_sizes_nearest_not_smaller — opset 18
  // ``keep_aspect_ratio_policy=not_smaller``: scales by max(sizes / in_size)
  // across axes (8x8 output for a 2x2 input requested at 7x8).
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor sizes = MakeSizesTensor({7, 8});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.keep_aspect_ratio_policy = "not_smaller";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes,
                               attrs.keep_aspect_ratio_policy),
           {X, sizes}, {Y}, "test_resize_upsample_sizes_nearest_not_smaller", {opset18},
           "backend-test", registry);
  }

  // test_resize_downsample_sizes_nearest_not_larger — downsample variant of
  // ``not_larger``: 2x4 input requested at 1x3 yields 1x2 output.
  {
    const Tensor X =
        Tensor::FromFloat("", {1, 1, 2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor sizes = MakeSizesTensor({1, 3});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.keep_aspect_ratio_policy = "not_larger";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes,
                               attrs.keep_aspect_ratio_policy),
           {X, sizes}, {Y}, "test_resize_downsample_sizes_nearest_not_larger", {opset18},
           "backend-test", registry);
  }

  // test_resize_downsample_sizes_nearest_not_smaller — downsample variant of
  // ``not_smaller``: 2x4 input requested at 1x3 yields 2x3 output.
  {
    const Tensor X =
        Tensor::FromFloat("", {1, 1, 2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor sizes = MakeSizesTensor({1, 3});
    kernel::Resize::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.keep_aspect_ratio_policy = "not_smaller";
    const Tensor Y = resize_kernel.ResizeSizes(X, sizes, attrs);
    Expect(MakeResizeNodeSizes("nearest", /*coord_mode=*/"", /*nearest_mode=*/"", attrs.axes,
                               attrs.keep_aspect_ratio_policy),
           {X, sizes}, {Y}, "test_resize_downsample_sizes_nearest_not_smaller", {opset18},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
