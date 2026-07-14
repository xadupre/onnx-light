// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterWhereCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::Where where_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    RegisterLazyBenchmarkCase(
        registry, std::move(node), "test_where_example_benchmark", {opset}, {count, count, count},
        {count}, [where_kernel, shape]() -> IoData {
          Tensor condition =
              Tensor::FromBool("condition", shape, RandUint<uint8_t>(2, shape, /*seed=*/9401));
          Tensor x = Tensor::FromFloat("x", shape, Randn<float>(shape, /*seed=*/9402));
          Tensor y = Tensor::FromFloat("y", shape, Randn<float>(shape, /*seed=*/9403));
          Tensor output = where_kernel(condition, x, y);
          return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
        });
    return;
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
    Tensor x = Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("y", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_where_example", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 1}, {1, 0});
    Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor y = Tensor::FromInt32("y", {1, 3}, {10, 20, 30});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_where_bcast", {opset}, "backend-test",
           registry);
  }

  // Mirrors upstream onnx ``test_where_long_example``: int64 inputs.
  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 1});
    Tensor x = Tensor::FromInt64("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromInt64("y", {2, 2}, {9, 8, 7, 6});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_where_long_example", {opset}, "backend-test",
           registry);
  }

  // Additional cases covering the remaining x/y dtypes supported by the Where
  // kernel. ``condition`` is always BOOL; ``x`` and ``y`` share the dtype.
  const Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromBool("x", {2, 2}, {1, 1, 0, 0});
    Tensor y = Tensor::FromBool("y", {2, 2}, {0, 0, 1, 1});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_bool", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromDouble("x", {2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor y = Tensor::FromDouble("y", {2, 2}, {5.0, 6.0, 7.0, 8.0});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_double", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromInt8("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromInt8("y", {2, 2}, {-5, -6, -7, -8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_int8", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromInt16("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromInt16("y", {2, 2}, {-5, -6, -7, -8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_int16", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromUint8("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromUint8("y", {2, 2}, {5, 6, 7, 8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_uint8", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromUint16("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromUint16("y", {2, 2}, {5, 6, 7, 8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_uint16", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromUint32("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromUint32("y", {2, 2}, {5, 6, 7, 8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_uint32", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromUint64("x", {2, 2}, {1, 2, 3, 4});
    Tensor y = Tensor::FromUint64("y", {2, 2}, {5, 6, 7, 8});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_uint64", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Tensor x = Tensor::FromStrings("x", {2, 2}, {"a", "b", "c", "d"});
    Tensor y = Tensor::FromStrings("y", {2, 2}, {"e", "f", "g", "h"});
    Tensor output = where_kernel(condition, x, y);
    Expect(node, {condition, x, y}, {output}, "test_cc_where_string", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
