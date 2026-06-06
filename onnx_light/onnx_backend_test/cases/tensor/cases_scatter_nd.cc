// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeScatterNDNode(const std::string &reduction) {
  NodeProto node;
  node.set_op_type("ScatterND");
  node.add_input("data");
  node.add_input("indices");
  node.add_input("updates");
  node.add_output("y");
  if (!reduction.empty() && reduction != "none") {
    AddAttribute<std::string>(node, "reduction", reduction);
  }
  return node;
}

Tensor MakeData4x4x4() {
  return Tensor::FromFloat("", {4, 4, 4},
                           {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
                            7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
                            5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8});
}

Tensor MakeUpdates2x4x4() {
  return Tensor::FromFloat("", {2, 4, 4}, {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                           1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
}

} // namespace

void RegisterScatterNDCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::ScatterND snd_kernel{ctx};

  // test_cc_scatternd — mirrors upstream ``test_scatternd``.
  {
    Tensor data = MakeData4x4x4();
    Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 2});
    Tensor updates = MakeUpdates2x4x4();
    kernel::ScatterND::Attributes attrs;
    Tensor output = snd_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNDNode("none"), {data, indices, updates}, {output}, "test_cc_scatternd",
           {opset}, "backend-test", registry);
  }

  // test_cc_scatternd_add — mirrors upstream ``test_scatternd_add``.
  {
    Tensor data = MakeData4x4x4();
    Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
    Tensor updates = MakeUpdates2x4x4();
    kernel::ScatterND::Attributes attrs;
    attrs.reduction = "add";
    Tensor output = snd_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNDNode("add"), {data, indices, updates}, {output}, "test_cc_scatternd_add",
           {opset}, "backend-test", registry);
  }

  // test_cc_scatternd_multiply — mirrors upstream ``test_scatternd_multiply``.
  {
    Tensor data = MakeData4x4x4();
    Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
    Tensor updates = MakeUpdates2x4x4();
    kernel::ScatterND::Attributes attrs;
    attrs.reduction = "mul";
    Tensor output = snd_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNDNode("mul"), {data, indices, updates}, {output},
           "test_cc_scatternd_multiply", {opset}, "backend-test", registry);
  }

  // test_cc_scatternd_max — mirrors upstream ``test_scatternd_max``.
  {
    Tensor data = MakeData4x4x4();
    Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
    Tensor updates = MakeUpdates2x4x4();
    kernel::ScatterND::Attributes attrs;
    attrs.reduction = "max";
    Tensor output = snd_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNDNode("max"), {data, indices, updates}, {output}, "test_cc_scatternd_max",
           {opset}, "backend-test", registry);
  }

  // test_cc_scatternd_min — mirrors upstream ``test_scatternd_min``.
  {
    Tensor data = MakeData4x4x4();
    Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
    Tensor updates = MakeUpdates2x4x4();
    kernel::ScatterND::Attributes attrs;
    attrs.reduction = "min";
    Tensor output = snd_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNDNode("min"), {data, indices, updates}, {output}, "test_cc_scatternd_min",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
