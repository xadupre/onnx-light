// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeSliceNode(bool with_axes, bool with_steps) {
  NodeProto node;
  node.set_op_type("Slice");
  node.add_input("data");
  node.add_input("starts");
  node.add_input("ends");
  if (with_axes) {
    node.add_input("axes");
  }
  if (with_steps) {
    if (!with_axes) {
      node.add_input("");
    }
    node.add_input("steps");
  }
  node.add_output("output");
  return node;
}

Tensor MakeInt64VectorTensor(const std::vector<int64_t> &values) {
  std::vector<uint8_t> data(values.size() * sizeof(int64_t));
  if (!values.empty()) {
    std::memcpy(data.data(), values.data(), data.size());
  }
  return Tensor("", static_cast<int32_t>(DataType::INT64), {static_cast<int64_t>(values.size())},
                std::move(data));
}

// Builds a deterministic 20x10x5 float tensor with values ``i`` at flat
// index ``i``. Used by the ONNX-parity test cases below in lieu of the
// random data the upstream ``onnx`` reference cases use, so that the
// expected output can be computed once by the kernel itself.
Tensor MakeRangeTensor20x10x5() {
  std::vector<float> values(20 * 10 * 5);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  return Tensor::FromFloat("", {20, 10, 5}, values);
}

} // namespace

void RegisterSliceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true);
    Expect(registry, std::move(node), "test_cc_slice_axes_steps_benchmark", {opset},
           {16777216, 2, 2, 2, 2}, {4194304}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             Tensor data = RandnTensor(DataType::FLOAT, {4096, 4096}, 2001);
             Tensor starts = MakeInt64VectorTensor({0, 0});
             Tensor ends = MakeInt64VectorTensor({4096, 2048});
             Tensor axes = MakeInt64VectorTensor({0, 1});
             Tensor steps = MakeInt64VectorTensor({1, 2});
             Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
    return;
  }

  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           "test_cc_slice_axes_steps", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = Tensor::FromFloat("", {2, 4},
                                                   {
                                                       1.0f,
                                                       2.0f,
                                                       3.0f,
                                                       4.0f,
                                                       5.0f,
                                                       6.0f,
                                                       7.0f,
                                                       8.0f,
                                                   });
             const Tensor starts = MakeInt64VectorTensor({1, 0});
             const Tensor ends = MakeInt64VectorTensor({2, 3});
             const Tensor axes = MakeInt64VectorTensor({0, 1});
             const Tensor steps = MakeInt64VectorTensor({1, 2});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  {
    Expect(registry, MakeSliceNode(/*with_axes=*/false, /*with_steps=*/false),
           "test_cc_slice_default_axes_steps", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = Tensor::FromFloat("", {2, 4},
                                                   {
                                                       1.0f,
                                                       2.0f,
                                                       3.0f,
                                                       4.0f,
                                                       5.0f,
                                                       6.0f,
                                                       7.0f,
                                                       8.0f,
                                                   });
             const Tensor starts = MakeInt64VectorTensor({0, 1});
             const Tensor ends = MakeInt64VectorTensor({-1, 1000});
             const Tensor output = slice_kernel(data, starts, ends);
             return IoData{{std::move(data), std::move(starts), std::move(ends)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice``: 2-D slice (axes=[0,1], steps=[1,1]) on the
  // first two dimensions of a 20x10x5 input.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true), "test_cc_slice",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({0, 0});
             const Tensor ends = MakeInt64VectorTensor({3, 10});
             const Tensor axes = MakeInt64VectorTensor({0, 1});
             const Tensor steps = MakeInt64VectorTensor({1, 1});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_neg``: slice axis 1 with a negative ``end``.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true), "test_cc_slice_neg",
           {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({0});
             const Tensor ends = MakeInt64VectorTensor({-1});
             const Tensor axes = MakeInt64VectorTensor({1});
             const Tensor steps = MakeInt64VectorTensor({1});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_start_out_of_bounds``: both ``start`` and
  // ``end`` are far past the axis length, producing an empty slice.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           "test_cc_slice_start_out_of_bounds", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({1000});
             const Tensor ends = MakeInt64VectorTensor({1000});
             const Tensor axes = MakeInt64VectorTensor({1});
             const Tensor steps = MakeInt64VectorTensor({1});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_end_out_of_bounds``: ``end`` is past the axis
  // length and is clamped to it.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           "test_cc_slice_end_out_of_bounds", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({1});
             const Tensor ends = MakeInt64VectorTensor({1000});
             const Tensor axes = MakeInt64VectorTensor({1});
             const Tensor steps = MakeInt64VectorTensor({1});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_default_steps``: 3-D slice with axes provided
  // but no ``steps`` input (defaults to 1).
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/false),
           "test_cc_slice_default_steps", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({0, 0, 3});
             const Tensor ends = MakeInt64VectorTensor({20, 10, 4});
             const Tensor axes = MakeInt64VectorTensor({0, 1, 2});
             const Tensor output = slice_kernel(data, starts, ends, &axes);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_neg_steps``: 3-D reverse slice with negative
  // ``steps`` on every axis.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/true),
           "test_cc_slice_neg_steps", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({20, 10, 4});
             const Tensor ends = MakeInt64VectorTensor({0, 0, 1});
             const Tensor axes = MakeInt64VectorTensor({0, 1, 2});
             const Tensor steps = MakeInt64VectorTensor({-1, -3, -2});
             const Tensor output = slice_kernel(data, starts, ends, &axes, &steps);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes),
                            std::move(steps)},
                           {std::move(output)}};
           });
  }

  // Mirrors ONNX ``test_slice_negative_axes``: 3-D slice using negative
  // ``axes`` values that count from the end of ``data``'s rank.
  {
    Expect(registry, MakeSliceNode(/*with_axes=*/true, /*with_steps=*/false),
           "test_cc_slice_negative_axes", {opset}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext slice_kernel_ctx{opset};
             const onnx_kernels::kernel::Slice slice_kernel{slice_kernel_ctx};

             const Tensor data = MakeRangeTensor20x10x5();
             const Tensor starts = MakeInt64VectorTensor({0, 0, 3});
             const Tensor ends = MakeInt64VectorTensor({20, 10, 4});
             const Tensor axes = MakeInt64VectorTensor({0, -2, -1});
             const Tensor output = slice_kernel(data, starts, ends, &axes);
             return IoData{{std::move(data), std::move(starts), std::move(ends), std::move(axes)},
                           {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
