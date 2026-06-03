// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Resize — nearest interpolation, output_dim[i] = floor(input_dim[i] *
// scales[i]) (or sizes[i] when ``sizes`` is used). Available since opset 10
// (replaces the deprecated ``Upsample``). Mirrors a subset of the upstream
// resize node tests in ``onnx/backend/test/case/node/resize.py`` that the
// reference implementation supports (``nearest`` mode with the
// ``asymmetric`` coordinate transformation).
// ---------------------------------------------------------------------------

namespace {

// Builds a Resize node taking ``(X, roi, scales)`` with ``roi`` as the empty
// input name (matching the v11+ schema for the simple "scales only" case).
NodeProto MakeResizeNodeScales(const std::string &mode = "nearest",
                               const std::string &coord_mode = "asymmetric") {
  NodeProto node;
  node.set_op_type("Resize");
  node.add_input("X");
  node.add_input(""); // roi (unused)
  node.add_input("scales");
  node.add_output("Y");
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  if (!coord_mode.empty()) {
    AddAttribute<std::string>(node, "coordinate_transformation_mode", coord_mode);
  }
  return node;
}

// Builds a Resize node taking ``(X, roi, scales, sizes)`` with ``roi`` and
// ``scales`` as empty input names (matching the v11+ schema for the "sizes
// only" case).
NodeProto MakeResizeNodeSizes(const std::string &mode = "nearest",
                              const std::string &coord_mode = "asymmetric") {
  NodeProto node;
  node.set_op_type("Resize");
  node.add_input("X");
  node.add_input(""); // roi (unused)
  node.add_input(""); // scales (unused)
  node.add_input("sizes");
  node.add_output("Y");
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  if (!coord_mode.empty()) {
    AddAttribute<std::string>(node, "coordinate_transformation_mode", coord_mode);
  }
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
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
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
           "test_cc_resize_upsample_scales_nearest_asymmetric", {opset}, "backend-test", registry);
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
           "test_cc_resize_upsample_scales_nearest_1d", {opset}, "backend-test", registry);
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
           "test_cc_resize_upsample_sizes_nearest_asymmetric", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
