// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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

void RegisterShapeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(15);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Shape shape_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    // Shape only reads the input's dimensions, so the case exists mainly for
    // benchmark coverage. A large multi-dimensional input keeps the timed
    // input materialisation representative of real graphs.
    const std::vector<int64_t> shape = {2048, 2048};
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/std::nullopt),
           "test_cc_shape_benchmark", {opset}, {2048 * 2048}, {2},
           [shape_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, 2001);
             onnx_kernels::kernel::Shape::Attributes attrs;
             Tensor y = Rename(shape_kernel(x, attrs), "y");
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // 2-D example: mirrors ``test_shape_example`` upstream.
  const Tensor x2d = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // 3-D example used by the remaining slicing cases (matches upstream
  // ``np.random.randn(3, 4, 5)`` shape; only the shape is observed by the op).
  const Tensor x3d = Tensor::FromFloat("x", {3, 4, 5}, std::vector<float>(3 * 4 * 5, 0.0f));

  // test_cc_shape_example
  {
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/std::nullopt),
           "test_cc_shape_example", {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             const Tensor y = Rename(shape_kernel(x2d, attrs), "y");
             return IoData{{std::move(x2d)}, {std::move(y)}};
           });
  }

  // test_cc_shape (no attributes, 3-D input).
  {
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/std::nullopt), "test_cc_shape",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_start_1
  {
    Expect(registry, MakeShapeNode(/*start=*/1, /*end=*/std::nullopt), "test_cc_shape_start_1",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = 1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_end_1
  {
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/1), "test_cc_shape_end_1",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.end = 1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_start_negative_1
  {
    Expect(registry, MakeShapeNode(/*start=*/-1, /*end=*/std::nullopt),
           "test_cc_shape_start_negative_1", {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = -1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_end_negative_1
  {
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/-1),
           "test_cc_shape_end_negative_1", {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.end = -1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_start_1_end_negative_1
  {
    Expect(registry, MakeShapeNode(/*start=*/1, /*end=*/-1), "test_cc_shape_start_1_end_negative_1",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = 1;
             attrs.end = -1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_start_1_end_2
  {
    Expect(registry, MakeShapeNode(/*start=*/1, /*end=*/2), "test_cc_shape_start_1_end_2", {opset},
           [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = 1;
             attrs.end = 2;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_clip_start
  {
    Expect(registry, MakeShapeNode(/*start=*/-10, /*end=*/std::nullopt), "test_cc_shape_clip_start",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = -10;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_clip_end
  {
    Expect(registry, MakeShapeNode(/*start=*/std::nullopt, /*end=*/10), "test_cc_shape_clip_end",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.end = 10;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }

  // test_cc_shape_start_greater_than_end
  {
    Expect(registry, MakeShapeNode(/*start=*/2, /*end=*/1), "test_cc_shape_start_greater_than_end",
           {opset}, [=]() -> IoData {
             onnx_kernels::kernel::Shape::Attributes attrs;
             attrs.start = 2;
             attrs.end = 1;
             const Tensor y = Rename(shape_kernel(x3d, attrs), "y");
             return IoData{{std::move(x3d)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
