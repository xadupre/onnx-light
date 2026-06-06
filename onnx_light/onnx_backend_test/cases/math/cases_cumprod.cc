// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeCumProdNode(bool exclusive, bool reverse) {
  NodeProto node;
  node.set_op_type("CumProd");
  node.add_input("x");
  node.add_input("axis");
  node.add_output("y");
  if (exclusive) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("exclusive");
    attr->set_type(AttributeProto::INT);
    attr->set_i(1);
  }
  if (reverse) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("reverse");
    attr->set_type(AttributeProto::INT);
    attr->set_i(1);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// CumProd — cumulative product of the input tensor along a selected axis.
// Mirrors the docstring example in the upstream ``CumProd`` schema (opset 26).
// ---------------------------------------------------------------------------
void RegisterCumProdCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(26);
  const kernel::KernelContext ctx{opset};
  const kernel::CumProd cumprod_kernel{ctx};

  // 1-D inclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis);
    Expect(node, {x, axis}, {y}, "test_cumprod_1d", {opset}, "backend-test", registry);
  }

  // 1-D exclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/false);
    Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis, /*exclusive=*/true);
    Expect(node, {x, axis}, {y}, "test_cumprod_1d_exclusive", {opset}, "backend-test", registry);
  }

  // 1-D reverse cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/true);
    Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis, /*exclusive=*/false, /*reverse=*/true);
    Expect(node, {x, axis}, {y}, "test_cumprod_1d_reverse", {opset}, "backend-test", registry);
  }

  // 1-D reverse + exclusive cumulative product (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/true);
    Tensor x = Tensor::FromDouble("", {3}, {1.0, 2.0, 3.0});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis, /*exclusive=*/true, /*reverse=*/true);
    Expect(node, {x, axis}, {y}, "test_cumprod_1d_reverse_exclusive", {opset}, "backend-test",
           registry);
  }

  // 2-D cumulative product along axis 0.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis);
    Expect(node, {x, axis}, {y}, "test_cumprod_2d_axis_0", {opset}, "backend-test", registry);
  }

  // 2-D cumulative product along axis 1.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor axis = Tensor::FromInt32("", {}, {1});
    Tensor y = cumprod_kernel(x, axis);
    Expect(node, {x, axis}, {y}, "test_cumprod_2d_axis_1", {opset}, "backend-test", registry);
  }

  // 2-D cumulative product along negative axis (-1) — axis input is INT64.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor axis = Tensor::FromInt64("", {}, {static_cast<int64_t>(-1)});
    Tensor y = cumprod_kernel(x, axis);
    Expect(node, {x, axis}, {y}, "test_cumprod_2d_negative_axis", {opset}, "backend-test",
           registry);
  }

  // 2-D cumulative product on INT32 input along axis 0.
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/false, /*reverse=*/false);
    Tensor x = Tensor::FromInt32("", {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis);
    Expect(node, {x, axis}, {y}, "test_cumprod_2d_int32", {opset}, "backend-test", registry);
  }

  // 1-D exclusive cumulative product on INT32 input (axis = 0).
  {
    NodeProto node = MakeCumProdNode(/*exclusive=*/true, /*reverse=*/false);
    Tensor x = Tensor::FromInt32("", {5}, {1, 2, 3, 4, 5});
    Tensor axis = Tensor::FromInt32("", {}, {0});
    Tensor y = cumprod_kernel(x, axis, /*exclusive=*/true);
    Expect(node, {x, axis}, {y}, "test_cumprod_1d_int32_exclusive", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
