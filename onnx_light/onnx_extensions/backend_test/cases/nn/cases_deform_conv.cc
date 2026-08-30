// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a base ``DeformConv`` node template with the requested IO names.
// The ``inputs`` argument lists every position (entries equal to the empty
// string are still added as placeholders, matching upstream's convention
// for skipping an optional input).
NodeProto MakeDeformConvNode(const std::vector<std::string> &inputs,
                             const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("DeformConv");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

// Registers the ``DeformConv`` reference backend test node cases. Each case
// mirrors a node test from the upstream ``onnx/backend/test/case/node``
// definitions (since opset 19), evaluating the expected output through the
// in-tree ``kernel::DeformConv`` reference implementation so the recorded
// expectations stay self-consistent with this library.
void RegisterDeformConvCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeDeformConvNode({"X", "W", "offset"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {0, 0, 0, 0});
    constexpr int64_t x_count = 1 * 16 * 128 * 128;
    constexpr int64_t w_count = 16 * 16 * 2 * 2;
    constexpr int64_t offset_count = 1 * 8 * 127 * 127;
    constexpr int64_t y_count = 1 * 16 * 127 * 127;
    Expect(registry, std::move(node), "test_cc_basic_deform_conv_without_padding_benchmark",
           {opset}, {x_count, w_count, offset_count}, {y_count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext dc_ctx{opset};
             const onnx_kernels::kernel::DeformConv dc{dc_ctx};

             Tensor X = RandnTensor(DataType::FLOAT, {1, 16, 128, 128}, 1501);
             Tensor W = RandnTensor(DataType::FLOAT, {16, 16, 2, 2}, 1502);
             Tensor offset = RandnTensor(DataType::FLOAT, {1, 8, 127, 127}, 1503);
             Tensor B;
             Tensor mask;
             onnx_kernels::kernel::DeformConv::Attributes attrs;
             attrs.kernel_shape = {2, 2};
             attrs.pads = {0, 0, 0, 0};
             Tensor Y = dc(X, W, offset, B, mask, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(offset)}, {std::move(Y)}};
           });
    return;
  }

  // -------------------------------------------------------------------
  // Case 1: basic 2x2 kernel without padding (test vector from
  // ``test_basic_deform_conv_without_padding``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    std::vector<float> off(1 * 8 * 2 * 2, 0.0f);
    off[(0 * 2 + 0) * 2 + 0] = 0.5f;
    off[(5 * 2 + 0) * 2 + 1] = -0.1f;
    Tensor offset = Tensor::FromFloat("offset", {1, 8, 2, 2}, off);
    Tensor B;
    Tensor mask;
    onnx_kernels::kernel::DeformConv::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {0, 0, 0, 0};
    NodeProto node = MakeDeformConvNode({"X", "W", "offset"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {0, 0, 0, 0});
    Expect(registry, std::move(node), "test_cc_basic_deform_conv_without_padding", {opset},
           [attrs, off]() -> IoData {
             Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
             Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
             Tensor offset = Tensor::FromFloat("offset", {1, 8, 2, 2}, off);
             Tensor B;
             Tensor mask;

             const OpsetId opset = DefaultOpset(22);

             const KernelContext dc_ctx{opset};
             const onnx_kernels::kernel::DeformConv dc{dc_ctx};

             Tensor Y = dc(X, W, offset, B, mask, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(offset)}, {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Case 2: basic 2x2 kernel with padding (test vector from
  // ``test_basic_deform_conv_with_padding``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    std::vector<float> off(1 * 8 * 4 * 4, 0.0f);
    off[(0 * 4 + 0) * 4 + 0] = 0.5f;
    off[(5 * 4 + 1) * 4 + 2] = -0.1f;
    Tensor offset = Tensor::FromFloat("offset", {1, 8, 4, 4}, off);
    Tensor B;
    Tensor mask;
    onnx_kernels::kernel::DeformConv::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {1, 1, 1, 1};
    NodeProto node = MakeDeformConvNode({"X", "W", "offset"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {1, 1, 1, 1});
    Expect(registry, std::move(node), "test_cc_basic_deform_conv_with_padding", {opset},
           [attrs, off]() -> IoData {
             Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
             Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
             Tensor offset = Tensor::FromFloat("offset", {1, 8, 4, 4}, off);
             Tensor B;
             Tensor mask;

             const OpsetId opset = DefaultOpset(22);

             const KernelContext dc_ctx{opset};
             const onnx_kernels::kernel::DeformConv dc{dc_ctx};

             Tensor Y = dc(X, W, offset, B, mask, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(offset)}, {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Case 3: with explicit bias and mask inputs (test vector from
  // ``test_deform_conv_with_mask_bias``).
  {
    Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
    std::vector<float> off(1 * 8 * 2 * 2, 0.0f);
    off[(0 * 2 + 0) * 2 + 0] = 0.5f;
    off[(5 * 2 + 0) * 2 + 1] = -0.1f;
    Tensor offset = Tensor::FromFloat("offset", {1, 8, 2, 2}, off);
    Tensor B = Tensor::FromFloat("B", {1}, {1.0f});
    std::vector<float> mvec(1 * 4 * 2 * 2, 1.0f);
    mvec[(2 * 2 + 1) * 2 + 1] = 0.2f;
    Tensor mask = Tensor::FromFloat("mask", {1, 4, 2, 2}, mvec);
    onnx_kernels::kernel::DeformConv::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {0, 0, 0, 0};
    NodeProto node = MakeDeformConvNode({"X", "W", "offset", "B", "mask"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {0, 0, 0, 0});
    Expect(registry, std::move(node), "test_cc_deform_conv_with_mask_bias", {opset},
           [attrs, off, mvec]() -> IoData {
             Tensor X = Tensor::FromFloat("X", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
             Tensor W = Tensor::FromFloat("W", {1, 1, 2, 2}, {1, 1, 1, 1});
             Tensor offset = Tensor::FromFloat("offset", {1, 8, 2, 2}, off);
             Tensor B = Tensor::FromFloat("B", {1}, {1.0f});
             Tensor mask = Tensor::FromFloat("mask", {1, 4, 2, 2}, mvec);

             const OpsetId opset = DefaultOpset(22);

             const KernelContext dc_ctx{opset};
             const onnx_kernels::kernel::DeformConv dc{dc_ctx};

             Tensor Y = dc(X, W, offset, B, mask, attrs);
             Y.name = "Y";
             return IoData{
                 {std::move(X), std::move(W), std::move(offset), std::move(B), std::move(mask)},
                 {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Case 4: ``offset_group=2`` exercising the multi-group offset code path
  // (test vector from ``test_deform_conv_with_multiple_offset_groups``).
  {
    std::vector<float> Xv(1 * 2 * 3 * 3, 0.0f);
    for (int k = 0; k < 9; ++k) {
      Xv[k] = static_cast<float>(k);
      Xv[9 + k] = static_cast<float>(8 - k);
    }
    Tensor X = Tensor::FromFloat("X", {1, 2, 3, 3}, Xv);
    Tensor W = Tensor::FromFloat("W", {1, 2, 2, 2}, std::vector<float>(8, 1.0f));
    std::vector<float> off(1 * 16 * 2 * 2, 0.0f);
    off[(0 * 2 + 0) * 2 + 0] = 0.5f;
    off[(13 * 2 + 0) * 2 + 1] = -0.1f;
    Tensor offset = Tensor::FromFloat("offset", {1, 16, 2, 2}, off);
    Tensor B;
    Tensor mask;
    onnx_kernels::kernel::DeformConv::Attributes attrs;
    attrs.kernel_shape = {2, 2};
    attrs.pads = {0, 0, 0, 0};
    attrs.offset_group = 2;
    NodeProto node = MakeDeformConvNode({"X", "W", "offset"}, {"Y"});
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "pads", {0, 0, 0, 0});
    AddAttribute<int64_t>(node, "offset_group", 2);
    Expect(registry, std::move(node), "test_cc_deform_conv_with_multiple_offset_groups", {opset},
           [attrs, Xv, off]() -> IoData {
             Tensor X = Tensor::FromFloat("X", {1, 2, 3, 3}, Xv);
             Tensor W = Tensor::FromFloat("W", {1, 2, 2, 2}, std::vector<float>(8, 1.0f));
             Tensor offset = Tensor::FromFloat("offset", {1, 16, 2, 2}, off);
             Tensor B;
             Tensor mask;

             const OpsetId opset = DefaultOpset(22);

             const KernelContext dc_ctx{opset};
             const onnx_kernels::kernel::DeformConv dc{dc_ctx};

             Tensor Y = dc(X, W, offset, B, mask, attrs);
             Y.name = "Y";
             return IoData{{std::move(X), std::move(W), std::move(offset)}, {std::move(Y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
