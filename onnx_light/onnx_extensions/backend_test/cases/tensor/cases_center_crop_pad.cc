// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeCenterCropPadNode(const std::vector<int64_t> &axes) {
  NodeProto node;
  node.set_op_type("CenterCropPad");
  node.add_input("x");
  node.add_input("shape");
  node.add_output("y");
  if (!axes.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "axes", axes);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// CenterCropPad — centrally crops or pads an input tensor to a target shape
// (since opset 18 in the ai.onnx domain). The reference kernel matches the
// upstream Python implementation in onnx.reference.ops.op_center_crop_pad.
// ---------------------------------------------------------------------------
void RegisterCenterCropPadCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeCenterCropPadNode({});
    Expect(registry, std::move(node), "test_cc_center_crop_pad_crop_benchmark", {opset},
           {4096 * 2048, 3}, {2048 * 2048}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             Tensor input = RandnTensor(DataType::FLOAT, {4096, 2048, 1}, 2001);
             Tensor shape = Tensor::FromInt64("", {3}, std::vector<int64_t>{2048, 2048, 1});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_center_crop_pad_crop — strictly cropping on every axis,
  // matching upstream test_center_crop_pad_crop (without random values).
  {
    Expect(registry, MakeCenterCropPadNode({}), "test_cc_center_crop_pad_crop", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(20 * 10 * 3);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {20, 10, 3}, values);
             const Tensor shape = Tensor::FromInt64("", {3}, std::vector<int64_t>{10, 7, 3});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }

  // test_cc_center_crop_pad_pad — strictly padding on every axis.
  {
    Expect(registry, MakeCenterCropPadNode({}), "test_cc_center_crop_pad_pad", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(10 * 7 * 3);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i + 1);
             }
             const Tensor input = Tensor::FromFloat("", {10, 7, 3}, values);
             const Tensor shape = Tensor::FromInt64("", {3}, std::vector<int64_t>{20, 10, 3});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }

  // test_cc_center_crop_pad_crop_and_pad — crop on axis 0, pad on axis 1.
  {
    Expect(registry, MakeCenterCropPadNode({}), "test_cc_center_crop_pad_crop_and_pad", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(20 * 8 * 3);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {20, 8, 3}, values);
             const Tensor shape = Tensor::FromInt64("", {3}, std::vector<int64_t>{10, 10, 3});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }

  // test_cc_center_crop_pad_crop_axes_hwc — ``axes`` restricts the operation
  // to a subset of dimensions.
  {
    Expect(registry, MakeCenterCropPadNode({0, 1}), "test_cc_center_crop_pad_crop_axes_hwc",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(20 * 8 * 3);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {20, 8, 3}, values);
             const Tensor shape = Tensor::FromInt64("", {2}, std::vector<int64_t>{10, 9});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             attrs.axes = {0, 1};
             attrs.axes_present = true;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }

  // test_cc_center_crop_pad_crop_negative_axes_hwc — negative axes.
  {
    Expect(registry, MakeCenterCropPadNode({-3, -2}),
           "test_cc_center_crop_pad_crop_negative_axes_hwc", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(20 * 8 * 3);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {20, 8, 3}, values);
             const Tensor shape = Tensor::FromInt64("", {2}, std::vector<int64_t>{10, 9});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             attrs.axes = {-3, -2};
             attrs.axes_present = true;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }

  // test_cc_center_crop_pad_crop_axes_chw — axes target the trailing
  // dimensions (channels-first layout).
  {
    Expect(registry, MakeCenterCropPadNode({1, 2}), "test_cc_center_crop_pad_crop_axes_chw",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(18);

             const KernelContext op_ctx{opset};
             const onnx_kernels::kernel::CenterCropPad op{op_ctx};

             std::vector<float> values(3 * 20 * 8);
             for (std::size_t i = 0; i < values.size(); ++i) {
               values[i] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {3, 20, 8}, values);
             const Tensor shape = Tensor::FromInt64("", {2}, std::vector<int64_t>{10, 9});
             onnx_kernels::kernel::CenterCropPad::Attributes attrs;
             attrs.axes = {1, 2};
             attrs.axes_present = true;
             const Tensor output = op(input, shape, attrs);
             return IoData{{std::move(input), std::move(shape)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
