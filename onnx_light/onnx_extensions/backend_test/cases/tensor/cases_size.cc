// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeSizeNode() {
  NodeProto node;
  node.set_op_type("Size");
  node.add_input("x");
  node.add_output("y");
  return node;
}

// Renames a tensor (copy with a new ``name``) — used so kernel-produced
// expected outputs match the ``y`` output name in :func:`MakeSizeNode`.
Tensor Rename(Tensor t, const std::string &name) {
  t.name = name;
  return t;
}

} // namespace

void RegisterSizeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto size_kernel = MakeReferenceKernel<onnx_kernels::kernel::Size>(opset);

  if (mode == TestMode::BENCHMARK) {
    // Size only reads the input's element count, so the case exists mainly
    // for benchmark coverage; a large input keeps the timed materialisation
    // representative.
    const std::vector<int64_t> shape = {2048, 2048};
    Expect(registry, MakeSizeNode(), "test_cc_size_benchmark", {opset}, {2048 * 2048}, {1},
           [size_kernel, shape]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, shape, 2001);
             Tensor y =
                 Rename(size_kernel.Invoke([&](const auto &kernel) { return kernel(x); }), "y");
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // test_cc_size_example — mirrors upstream ``test_size_example`` (2-D
  // float input of shape [2, 3]).
  {
    Expect(registry, MakeSizeNode(), "test_cc_size_example", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      const Tensor y =
          Rename(size_kernel.Invoke([&](const auto &kernel) { return kernel(x); }), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_size — mirrors upstream ``test_size`` (3-D float input of shape
  // [3, 4, 5]). Only the shape (and hence the element count) is observed by
  // the op.
  {
    Expect(registry, MakeSizeNode(), "test_cc_size", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("x", {3, 4, 5}, std::vector<float>(3 * 4 * 5, 0.0f));
      const Tensor y =
          Rename(size_kernel.Invoke([&](const auto &kernel) { return kernel(x); }), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_size_scalar — 0-D (scalar) input has exactly one element.
  {
    Expect(registry, MakeSizeNode(), "test_cc_size_scalar", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("x", {}, {42.0f});
      const Tensor y =
          Rename(size_kernel.Invoke([&](const auto &kernel) { return kernel(x); }), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // test_cc_size_empty — input with a zero dimension has zero elements.
  {
    Expect(registry, MakeSizeNode(), "test_cc_size_empty", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat("x", {2, 0, 3}, {});
      const Tensor y =
          Rename(size_kernel.Invoke([&](const auto &kernel) { return kernel(x); }), "y");
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
