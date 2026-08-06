// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a Unique NodeProto with all four declared outputs and the requested
// optional attributes (``sorted`` and/or ``axis``).
NodeProto MakeUniqueNode(std::optional<int64_t> sorted_attr, std::optional<int64_t> axis_attr) {
  NodeProto node;
  node.set_op_type("Unique");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("indices");
  node.add_output("inverse_indices");
  node.add_output("counts");
  if (sorted_attr.has_value()) {
    AddAttribute<int64_t>(node, "sorted", *sorted_attr);
  }
  if (axis_attr.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis_attr);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Unique — returns the unique elements (or subtensors) of the input tensor
// and three optional companion outputs (indices, inverse_indices, counts).
// Mirrors the upstream ``onnx.backend.test.case.node.unique`` cases.
// ---------------------------------------------------------------------------
void RegisterUniqueCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Unique unique_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/0, /*axis_attr=*/std::nullopt);
    // The reference kernel groups values in O(count * unique) time, so the
    // benchmark feeds a large input drawn from a small value set to keep the
    // number of distinct groups bounded while still processing many elements.
    constexpr int64_t kUniqueBenchmarkSize = 1 << 20;
    constexpr int64_t kDistinctValues = 256;
    Expect(registry, std::move(node), "test_cc_unique_not_sorted_without_axis_benchmark", {opset},
           {kUniqueBenchmarkSize},
           {kDistinctValues, kDistinctValues, kUniqueBenchmarkSize, kDistinctValues},
           [unique_kernel, kUniqueBenchmarkSize, kDistinctValues]() -> IoData {
             std::vector<float> values(static_cast<std::size_t>(kUniqueBenchmarkSize));
             for (int64_t i = 0; i < kUniqueBenchmarkSize; ++i) {
               values[static_cast<std::size_t>(i)] = static_cast<float>(i % kDistinctValues);
             }
             Tensor x = Tensor::FromFloat("X", {kUniqueBenchmarkSize}, values);
             onnx_kernels::kernel::Unique::Attributes attrs;
             attrs.sorted = false;
             auto out = unique_kernel(x, attrs);
             return IoData{{std::move(x)},
                           {std::move(out.y), std::move(out.indices),
                            std::move(out.inverse_indices), std::move(out.counts)}};
           });
    return;
  }

  // test_cc_unique_not_sorted_without_axis — 1-D float input, sorted=0.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/0, /*axis_attr=*/std::nullopt);
    Expect(registry, std::move(node), "test_cc_unique_not_sorted_without_axis", {opset},
           [=]() -> IoData {
             const Tensor x = Tensor::FromFloat("X", {6}, {2.f, 1.f, 1.f, 3.f, 4.f, 3.f});
             onnx_kernels::kernel::Unique::Attributes attrs;
             attrs.sorted = false;
             auto out = unique_kernel(x, attrs);
             return IoData{{std::move(x)},
                           {std::move(out.y), std::move(out.indices),
                            std::move(out.inverse_indices), std::move(out.counts)}};
           });
  }

  // test_cc_unique_sorted_without_axis — 1-D float input, default sorted=1.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/std::nullopt, /*axis_attr=*/std::nullopt);
    Expect(registry, std::move(node), "test_cc_unique_sorted_without_axis", {opset},
           [=]() -> IoData {
             const Tensor x = Tensor::FromFloat("X", {6}, {2.f, 1.f, 1.f, 3.f, 4.f, 3.f});
             auto out = unique_kernel(x);
             return IoData{{std::move(x)},
                           {std::move(out.y), std::move(out.indices),
                            std::move(out.inverse_indices), std::move(out.counts)}};
           });
  }

  // test_cc_unique_length_1 — single-element input.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/std::nullopt, /*axis_attr=*/std::nullopt);
    Expect(registry, std::move(node), "test_cc_unique_length_1", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("X", {1}, {7.f});
      auto out = unique_kernel(x);
      return IoData{{std::move(x)},
                    {std::move(out.y), std::move(out.indices), std::move(out.inverse_indices),
                     std::move(out.counts)}};
    });
  }

  // test_cc_unique_sorted_with_axis — 2-D float input, sorted=1, axis=0.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/1, /*axis_attr=*/0);
    Expect(registry, std::move(node), "test_cc_unique_sorted_with_axis", {opset}, [=]() -> IoData {
      const Tensor x =
          Tensor::FromFloat("X", {3, 3}, {1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 3.f, 4.f});
      onnx_kernels::kernel::Unique::Attributes attrs;
      attrs.sorted = true;
      attrs.axis = 0;
      auto out = unique_kernel(x, attrs);
      return IoData{{std::move(x)},
                    {std::move(out.y), std::move(out.indices), std::move(out.inverse_indices),
                     std::move(out.counts)}};
    });
  }

  // test_cc_unique_sorted_with_negative_axis — 2-D float input, sorted=1,
  // axis=-1.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/1, /*axis_attr=*/-1);
    Expect(registry, std::move(node), "test_cc_unique_sorted_with_negative_axis", {opset},
           [=]() -> IoData {
             const Tensor x =
                 Tensor::FromFloat("X", {2, 4}, {1.f, 1.f, 0.f, 2.f, 1.f, 1.f, 0.f, 2.f});
             onnx_kernels::kernel::Unique::Attributes attrs;
             attrs.sorted = true;
             attrs.axis = -1;
             auto out = unique_kernel(x, attrs);
             return IoData{{std::move(x)},
                           {std::move(out.y), std::move(out.indices),
                            std::move(out.inverse_indices), std::move(out.counts)}};
           });
  }

  // test_cc_unique_sorted_with_axis_3d — 3-D float input, sorted=1, axis=1.
  {
    NodeProto node = MakeUniqueNode(/*sorted_attr=*/1, /*axis_attr=*/1);
    Expect(registry, std::move(node), "test_cc_unique_sorted_with_axis_3d", {opset},
           [=]() -> IoData {
             const Tensor x = Tensor::FromFloat(
                 "X", {2, 4, 2},
                 {1.f, 1.f, 0.f, 1.f, 2.f, 1.f, 0.f, 1.f, 1.f, 1.f, 0.f, 1.f, 2.f, 1.f, 0.f, 1.f});
             onnx_kernels::kernel::Unique::Attributes attrs;
             attrs.sorted = true;
             attrs.axis = 1;
             auto out = unique_kernel(x, attrs);
             return IoData{{std::move(x)},
                           {std::move(out.y), std::move(out.indices),
                            std::move(out.inverse_indices), std::move(out.counts)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
