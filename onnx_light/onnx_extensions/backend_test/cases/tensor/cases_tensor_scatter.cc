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

NodeProto MakeTensorScatterNode(const std::string &mode, bool set_mode_attr) {
  NodeProto node;
  node.set_op_type("TensorScatter");
  node.add_input("past_cache");
  node.add_input("update");
  node.add_input("write_indices");
  node.add_output("present_cache");
  if (set_mode_attr) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("mode");
    attr->set_type(AttributeProto::STRING);
    attr->set_s(mode);
  }
  return node;
}

} // namespace

void RegisterTensorScatterCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(24);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeTensorScatterNode("linear", /*set_mode_attr=*/true);
    Expect(registry, std::move(node), "test_cc_tensorscatter_benchmark", {opset},
           {4194304, 1024, 2}, {4194304}, []() -> IoData {
             const OpsetId opset = DefaultOpset(24);

             const KernelContext ts_kernel_ctx{opset};
             const onnx_kernels::kernel::TensorScatter ts_kernel{ts_kernel_ctx};

             Tensor past_cache = RandnTensor(DataType::FLOAT, {2, 1, 4096, 512}, 2001);
             Tensor update = RandnTensor(DataType::FLOAT, {2, 1, 1, 512}, 2002);
             Tensor write_indices = Tensor::FromInt64("write_indices", {2}, {2048, 3072});
             onnx_kernels::kernel::TensorScatter::Attributes attrs;
             Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
             return IoData{{std::move(past_cache), std::move(update), std::move(write_indices)},
                           {std::move(present_cache)}};
           });
    return;
  }

  // test_cc_tensorscatter — mirrors upstream ``test_tensorscatter`` (4-D
  // input, default axis=-2, mode="linear").
  {
    Expect(registry, MakeTensorScatterNode("linear", /*set_mode_attr=*/true),
           "test_cc_tensorscatter", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(24);

             const KernelContext ts_kernel_ctx{opset};
             const onnx_kernels::kernel::TensorScatter ts_kernel{ts_kernel_ctx};

             const Tensor past_cache =
                 Tensor::FromFloat("past_cache", {2, 1, 4, 5},
                                   {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                    1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});
             const Tensor update =
                 Tensor::FromFloat("update", {2, 1, 1, 5}, {5, 5, 5, 5, 5, 1, 1, 1, 1, 1});
             const Tensor write_indices = Tensor::FromInt64("write_indices", {2}, {1, 2});
             onnx_kernels::kernel::TensorScatter::Attributes attrs;
             const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
             return IoData{{std::move(past_cache), std::move(update), std::move(write_indices)},
                           {std::move(present_cache)}};
           });
  }

  // test_cc_tensorscatter_circular — mirrors upstream
  // ``test_tensorscatter_circular`` (write index 3 with seq_len 2 wraps
  // around max_sequence_length 4 for batch 1).
  {
    Expect(registry, MakeTensorScatterNode("circular", /*set_mode_attr=*/true),
           "test_cc_tensorscatter_circular", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(24);

             const KernelContext ts_kernel_ctx{opset};
             const onnx_kernels::kernel::TensorScatter ts_kernel{ts_kernel_ctx};

             const Tensor past_cache =
                 Tensor::FromFloat("past_cache", {2, 1, 4, 5},
                                   {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                    1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});
             const Tensor update =
                 Tensor::FromFloat("update", {2, 1, 2, 5},
                                   {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2});
             const Tensor write_indices = Tensor::FromInt64("write_indices", {2}, {1, 3});
             onnx_kernels::kernel::TensorScatter::Attributes attrs;
             attrs.mode = "circular";
             const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
             return IoData{{std::move(past_cache), std::move(update), std::move(write_indices)},
                           {std::move(present_cache)}};
           });
  }

  // test_cc_tensorscatter_3d — mirrors upstream ``test_tensorscatter_3d``
  // (3-D input, default axis=-2 == 1, mode default "linear").
  {
    Expect(registry, MakeTensorScatterNode("linear", /*set_mode_attr=*/false),
           "test_cc_tensorscatter_3d", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(24);

             const KernelContext ts_kernel_ctx{opset};
             const onnx_kernels::kernel::TensorScatter ts_kernel{ts_kernel_ctx};

             const Tensor past_cache =
                 Tensor::FromFloat("past_cache", {3, 4, 5},
                                   {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1,
                                    1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1,
                                    1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1});
             const Tensor update = Tensor::FromFloat("update", {3, 2, 5},
                                                     {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                                      7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
             const Tensor write_indices = Tensor::FromInt64("write_indices", {3}, {1, 2, 0});
             onnx_kernels::kernel::TensorScatter::Attributes attrs;
             const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
             return IoData{{std::move(past_cache), std::move(update), std::move(write_indices)},
                           {std::move(present_cache)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
