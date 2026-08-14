// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeCol2ImNode(const std::vector<std::string> &inputs,
                         const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("Col2Im");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

// Registers the ``Col2Im`` reference backend test node cases. Each case
// mirrors a node test from the upstream ``onnx/backend/test/case/node``
// definitions (since opset 18), evaluating the expected output through the
// in-tree ``kernel::Col2Im`` reference implementation so the recorded
// expectations stay self-consistent with this library.
void RegisterCol2ImCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Col2Im op{ctx};

  if (mode == TestMode::BENCHMARK) {
    // Large 2-D Col2Im: image 64x64, block 4x4, 16 channels.
    // L = (64 - 4 + 1)^2 = 3721 blocks; input is [1, C*block, L].
    const int64_t channels = 16;
    const int64_t block = 4;
    const int64_t image = 64;
    const int64_t blocks = (image - block + 1) * (image - block + 1);
    const int64_t col = channels * block * block;
    const std::vector<int64_t> in_shape = {1, col, blocks};
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    Expect(registry, std::move(node), "test_cc_col2im_benchmark", {opset}, {col * blocks, 2, 2},
           {channels * image * image}, [op, in_shape, block, image]() -> IoData {
             Tensor input = RandnTensor(DataType::FLOAT, in_shape, 2001);
             Tensor image_shape = Tensor::FromInt64("image_shape", {2}, {image, image});
             Tensor block_shape = Tensor::FromInt64("block_shape", {2}, {block, block});
             onnx_kernels::kernel::Col2Im::Attributes attrs;
             Tensor output = op(input, image_shape, block_shape, attrs);
             output.name = "output";
             return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                           {std::move(output)}};
           });
    return;
  }

  // ---------------------------------------------------------------------
  // Case 1: 2-D, no padding, default stride/dilation. Mirrors upstream
  // ``test_col2im``.
  {
    std::vector<float> in_v(25);
    for (int i = 0; i < 25; ++i) {
      in_v[i] = static_cast<float>(i + 1);
    }
    Tensor input = Tensor::FromFloat("input", {1, 5, 5}, in_v);
    Tensor image_shape = Tensor::FromInt64("image_shape", {2}, {5, 5});
    Tensor block_shape = Tensor::FromInt64("block_shape", {2}, {1, 5});
    onnx_kernels::kernel::Col2Im::Attributes attrs;
    Tensor output = op(input, image_shape, block_shape, attrs);
    output.name = "output";
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    Expect(registry, std::move(node), "test_cc_col2im", {opset}, [=]() -> IoData {
      return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                    {std::move(output)}};
    });
  }

  // ---------------------------------------------------------------------
  // Case 2: 2-D with explicit pads. Mirrors upstream ``test_col2im_pads``.
  {
    std::vector<float> in_v(5 * 15);
    for (int i = 0; i < 5 * 15; ++i) {
      in_v[i] = static_cast<float>(i + 1);
    }
    Tensor input = Tensor::FromFloat("input", {1, 5, 15}, in_v);
    Tensor image_shape = Tensor::FromInt64("image_shape", {2}, {5, 5});
    Tensor block_shape = Tensor::FromInt64("block_shape", {2}, {1, 5});
    onnx_kernels::kernel::Col2Im::Attributes attrs;
    attrs.pads = {0, 1, 0, 1};
    Tensor output = op(input, image_shape, block_shape, attrs);
    output.name = "output";
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    AddAttribute<std::vector<int64_t>>(node, "pads", {0, 1, 0, 1});
    Expect(registry, std::move(node), "test_cc_col2im_pads", {opset}, [=]() -> IoData {
      return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                    {std::move(output)}};
    });
  }

  // ---------------------------------------------------------------------
  // Case 3: 2-D with strides. Mirrors upstream ``test_col2im_strides``.
  {
    std::vector<float> in_v(9 * 4);
    for (int i = 0; i < 9 * 4; ++i) {
      in_v[i] = static_cast<float>(i + 1);
    }
    Tensor input = Tensor::FromFloat("input", {1, 9, 4}, in_v);
    Tensor image_shape = Tensor::FromInt64("image_shape", {2}, {5, 5});
    Tensor block_shape = Tensor::FromInt64("block_shape", {2}, {3, 3});
    onnx_kernels::kernel::Col2Im::Attributes attrs;
    attrs.strides = {2, 2};
    Tensor output = op(input, image_shape, block_shape, attrs);
    output.name = "output";
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    Expect(registry, std::move(node), "test_cc_col2im_strides", {opset}, [=]() -> IoData {
      return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                    {std::move(output)}};
    });
  }

  // ---------------------------------------------------------------------
  // Case 4: 2-D with dilations. Mirrors upstream ``test_col2im_dilations``.
  {
    std::vector<float> in_v(4 * 5);
    for (int i = 0; i < 4 * 5; ++i) {
      in_v[i] = static_cast<float>(i + 1);
    }
    Tensor input = Tensor::FromFloat("input", {1, 4, 5}, in_v);
    Tensor image_shape = Tensor::FromInt64("image_shape", {2}, {6, 6});
    Tensor block_shape = Tensor::FromInt64("block_shape", {2}, {2, 2});
    onnx_kernels::kernel::Col2Im::Attributes attrs;
    attrs.dilations = {1, 5};
    Tensor output = op(input, image_shape, block_shape, attrs);
    output.name = "output";
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {1, 5});
    Expect(registry, std::move(node), "test_cc_col2im_dilations", {opset}, [=]() -> IoData {
      return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                    {std::move(output)}};
    });
  }

  // ---------------------------------------------------------------------
  // Case 5: 5-D Col2Im over a small batched volume. Mirrors upstream
  // ``test_col2im_5d``.
  {
    std::vector<float> in_v(10 * 12);
    for (int i = 0; i < 10 * 12; ++i) {
      in_v[i] = static_cast<float>(i + 1);
    }
    Tensor input = Tensor::FromFloat("input", {1, 10, 12}, in_v);
    Tensor image_shape = Tensor::FromInt64("image_shape", {3}, {3, 4, 5});
    Tensor block_shape = Tensor::FromInt64("block_shape", {3}, {1, 1, 5});
    onnx_kernels::kernel::Col2Im::Attributes attrs;
    Tensor output = op(input, image_shape, block_shape, attrs);
    output.name = "output";
    NodeProto node = MakeCol2ImNode({"input", "image_shape", "block_shape"}, {"output"});
    Expect(registry, std::move(node), "test_cc_col2im_5d", {opset}, [=]() -> IoData {
      return IoData{{std::move(input), std::move(image_shape), std::move(block_shape)},
                    {std::move(output)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
