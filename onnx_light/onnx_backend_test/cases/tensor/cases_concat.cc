// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Concat — y = concat(inputs..., axis) along the requested axis (since
// opset 13). Mirrors the upstream ONNX node tests in
// ``onnx/backend/test/case/node/concat.py``: for the 1d / 2d / 3d input
// shapes, every positive axis in ``[0, rank-1]`` and every negative axis
// in ``[-rank, -1]`` is registered.
// ---------------------------------------------------------------------------

namespace {

struct ConcatShapeCase {
  const char *label;
  std::vector<int64_t> shape;
  std::vector<float> values0;
  std::vector<float> values1;
};

NodeProto MakeConcatNode(int64_t axis) {
  NodeProto node;
  node.set_op_type("Concat");
  node.add_input("value0");
  node.add_input("value1");
  node.add_output("output");
  AddAttribute<int64_t>(node, "axis", axis);
  return node;
}

} // namespace

void RegisterConcatCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Concat concat_kernel{ctx};

  // Inputs mirror the upstream ``test_cases`` dict in
  // ``onnx/backend/test/case/node/concat.py``.
  const std::vector<ConcatShapeCase> shape_cases = {
      {"1d", {2}, {1.0f, 2.0f}, {3.0f, 4.0f}},
      {"2d", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}},
      {"3d",
       {2, 2, 2},
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}},
  };

  for (const ConcatShapeCase &sc : shape_cases) {
    const int64_t rank = static_cast<int64_t>(sc.shape.size());
    Tensor x0 = Tensor::FromFloat("", sc.shape, sc.values0);
    Tensor x1 = Tensor::FromFloat("", sc.shape, sc.values1);

    // Positive axes: 0 .. rank-1.
    for (int64_t axis = 0; axis < rank; ++axis) {
      NodeProto node = MakeConcatNode(axis);
      Tensor y = concat_kernel({x0, x1}, axis);
      Expect(node, {x0, x1}, {y},
             std::string("test_cc_concat_") + sc.label + "_axis_" + std::to_string(axis), {opset},
             "backend-test", registry);
    }

    // Negative axes: -rank .. -1.
    for (int64_t axis = -rank; axis < 0; ++axis) {
      NodeProto node = MakeConcatNode(axis);
      Tensor y = concat_kernel({x0, x1}, axis);
      Expect(node, {x0, x1}, {y},
             std::string("test_cc_concat_") + sc.label + "_axis_negative_" + std::to_string(-axis),
             {opset}, "backend-test", registry);
    }
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
