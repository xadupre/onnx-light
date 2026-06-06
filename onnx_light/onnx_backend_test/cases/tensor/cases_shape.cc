// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeShapeNode(std::optional<int64_t> start, std::optional<int64_t> end) {
  NodeProto node;
  node.set_op_type("Shape");
  node.add_input("x");
  node.add_output("y");
  if (start.has_value()) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("start");
    attr->set_type(AttributeProto::INT);
    attr->set_i(*start);
  }
  if (end.has_value()) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("end");
    attr->set_type(AttributeProto::INT);
    attr->set_i(*end);
  }
  return node;
}

// Renames a tensor (copy with a new ``name``) — used so kernel-produced
// expected outputs match the ``y`` output name in :func:`MakeShapeNode`.
Tensor Rename(Tensor t, const std::string &name) {
  t.name = name;
  return t;
}

} // namespace

void RegisterShapeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(15);
  const kernel::KernelContext ctx{opset};
  const kernel::Shape shape_kernel{ctx};

  // 2-D example: mirrors ``test_shape_example`` upstream.
  const Tensor x2d = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // 3-D example used by the remaining slicing cases (matches upstream
  // ``np.random.randn(3, 4, 5)`` shape; only the shape is observed by the op).
  const Tensor x3d = Tensor::FromFloat("x", {3, 4, 5}, std::vector<float>(3 * 4 * 5, 0.0f));

  // test_cc_shape_example
  {
    kernel::Shape::Attributes attrs;
    const Tensor y = Rename(shape_kernel(x2d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/std::nullopt, /*end=*/std::nullopt), {x2d}, {y},
           "test_cc_shape_example", {opset}, "backend-test", registry);
  }

  // test_cc_shape (no attributes, 3-D input).
  {
    kernel::Shape::Attributes attrs;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/std::nullopt, /*end=*/std::nullopt), {x3d}, {y}, "test_cc_shape",
           {opset}, "backend-test", registry);
  }

  // test_cc_shape_start_1
  {
    kernel::Shape::Attributes attrs;
    attrs.start = 1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/1, /*end=*/std::nullopt), {x3d}, {y}, "test_cc_shape_start_1",
           {opset}, "backend-test", registry);
  }

  // test_cc_shape_end_1
  {
    kernel::Shape::Attributes attrs;
    attrs.end = 1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/std::nullopt, /*end=*/1), {x3d}, {y}, "test_cc_shape_end_1",
           {opset}, "backend-test", registry);
  }

  // test_cc_shape_start_negative_1
  {
    kernel::Shape::Attributes attrs;
    attrs.start = -1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/-1, /*end=*/std::nullopt), {x3d}, {y},
           "test_cc_shape_start_negative_1", {opset}, "backend-test", registry);
  }

  // test_cc_shape_end_negative_1
  {
    kernel::Shape::Attributes attrs;
    attrs.end = -1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/std::nullopt, /*end=*/-1), {x3d}, {y},
           "test_cc_shape_end_negative_1", {opset}, "backend-test", registry);
  }

  // test_cc_shape_start_1_end_negative_1
  {
    kernel::Shape::Attributes attrs;
    attrs.start = 1;
    attrs.end = -1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/1, /*end=*/-1), {x3d}, {y},
           "test_cc_shape_start_1_end_negative_1", {opset}, "backend-test", registry);
  }

  // test_cc_shape_start_1_end_2
  {
    kernel::Shape::Attributes attrs;
    attrs.start = 1;
    attrs.end = 2;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/1, /*end=*/2), {x3d}, {y}, "test_cc_shape_start_1_end_2",
           {opset}, "backend-test", registry);
  }

  // test_cc_shape_clip_start
  {
    kernel::Shape::Attributes attrs;
    attrs.start = -10;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/-10, /*end=*/std::nullopt), {x3d}, {y},
           "test_cc_shape_clip_start", {opset}, "backend-test", registry);
  }

  // test_cc_shape_clip_end
  {
    kernel::Shape::Attributes attrs;
    attrs.end = 10;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/std::nullopt, /*end=*/10), {x3d}, {y}, "test_cc_shape_clip_end",
           {opset}, "backend-test", registry);
  }

  // test_cc_shape_start_greater_than_end
  {
    kernel::Shape::Attributes attrs;
    attrs.start = 2;
    attrs.end = 1;
    const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
    Expect(MakeShapeNode(/*start=*/2, /*end=*/1), {x3d}, {y},
           "test_cc_shape_start_greater_than_end", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
