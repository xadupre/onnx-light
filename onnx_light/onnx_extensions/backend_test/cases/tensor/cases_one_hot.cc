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

// Builds a ``OneHot`` ``NodeProto`` with three inputs and a single output.
// When ``set_axis`` is true the ``axis`` attribute is added with value
// ``axis``; otherwise it is omitted (defaulting to ``-1`` per the schema).
NodeProto MakeOneHotNode(bool set_axis, int64_t axis) {
  NodeProto node;
  node.set_op_type("OneHot");
  node.add_input("indices");
  node.add_input("depth");
  node.add_input("values");
  node.add_output("y");
  if (set_axis) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("axis");
    attr->set_type(AttributeProto::INT);
    attr->set_i(axis);
  }
  return node;
}

} // namespace

void RegisterOneHotCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::OneHot one_hot_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    constexpr int64_t index_count = kBenchmarkElementwiseSize / 12;
    NodeProto node = MakeOneHotNode(/*set_axis=*/false, /*axis=*/-1);
    Expect(registry, std::move(node), "test_onehot_without_axis_benchmark", {opset},
           {index_count, 1, 2}, {index_count * 12}, [one_hot_kernel, index_count]() -> IoData {
             std::vector<int64_t> index_values(index_count);
             for (int64_t i = 0; i < index_count; ++i) {
               index_values[static_cast<std::size_t>(i)] = i % 12;
             }
             Tensor indices = Tensor::FromInt64("indices", {index_count}, index_values);
             Tensor depth = Tensor::FromFloat("depth", {}, {12.0f});
             Tensor values = Tensor::FromInt32("values", {2}, {2, 5});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
    return;
  }

  // test_onehot_without_axis: indices INT64 vector, depth FLOAT scalar,
  // INT32 values. Output rank = 2, axis = -1 (innermost).
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/false, /*axis=*/-1), "test_onehot_without_axis",
           {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromInt64("indices", {3}, {0, 7, 8});
             const Tensor depth = Tensor::FromFloat("depth", {}, {12.0f});
             const Tensor values = Tensor::FromInt32("values", {2}, {2, 5});
             onnx_kernels::kernel::OneHot::Attributes attrs; // axis = -1
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }

  // test_onehot_with_axis: indices FLOAT matrix, depth FLOAT scalar,
  // FLOAT values, axis = 1.
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/true, /*axis=*/1), "test_onehot_with_axis",
           {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromFloat("indices", {2, 2}, {1.0f, 9.0f, 2.0f, 4.0f});
             const Tensor depth = Tensor::FromFloat("depth", {}, {10.0f});
             const Tensor values = Tensor::FromFloat("values", {2}, {1.0f, 3.0f});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             attrs.axis = 1;
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }

  // test_onehot_negative_indices: indices INT64 vector with negative entries,
  // depth FLOAT scalar, FLOAT values, axis = 1.
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/true, /*axis=*/1), "test_onehot_negative_indices",
           {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromInt64("indices", {3}, {0, -7, -8});
             const Tensor depth = Tensor::FromFloat("depth", {}, {10.0f});
             const Tensor values = Tensor::FromFloat("values", {2}, {1.0f, 3.0f});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             attrs.axis = 1;
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }

  // test_onehot_out_of_range_indices: indices INT64 vector with out-of-range
  // positive and negative entries. Out-of-range positions stay at off_value.
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/true, /*axis=*/1),
           "test_onehot_out_of_range_indices", {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromInt64("indices", {3}, {5, -6, -1});
             const Tensor depth = Tensor::FromFloat("depth", {}, {5.0f});
             const Tensor values = Tensor::FromFloat("values", {2}, {1.0f, 3.0f});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             attrs.axis = 1;
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }

  // test_onehot_with_negative_axis: indices FLOAT matrix, depth FLOAT scalar,
  // FLOAT values, axis = -2.
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/true, /*axis=*/-2),
           "test_onehot_with_negative_axis", {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromFloat("indices", {2, 2}, {1.0f, 9.0f, 2.0f, 4.0f});
             const Tensor depth = Tensor::FromFloat("depth", {}, {10.0f});
             const Tensor values = Tensor::FromFloat("values", {2}, {1.0f, 3.0f});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             attrs.axis = -2;
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }

  // test_cc_onehot_default_axis_int64_indices: covers the default axis
  // (-1) with INT64 indices and depth.
  {
    Expect(registry, MakeOneHotNode(/*set_axis=*/false, /*axis=*/-1),
           "test_cc_onehot_default_axis_int64_indices", {opset}, [=]() -> IoData {
             const Tensor indices = Tensor::FromInt64("indices", {2}, {1, 0});
             const Tensor depth = Tensor::FromInt64("depth", {}, {3});
             const Tensor values = Tensor::FromFloat("values", {2}, {0.0f, 1.0f});
             onnx_kernels::kernel::OneHot::Attributes attrs;
             const Tensor y = one_hot_kernel(indices, depth, values, attrs);
             return IoData{{std::move(indices), std::move(depth), std::move(values)},
                           {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
