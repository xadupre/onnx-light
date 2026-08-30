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

void RegisterConcatCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize / 2};
    NodeProto node = MakeConcatNode(0);
    Expect(registry, std::move(node), "test_cc_concat_1d_axis_0_benchmark", {opset},
           {kBenchmarkElementwiseSize / 2, kBenchmarkElementwiseSize / 2},
           {kBenchmarkElementwiseSize}, [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext concat_kernel_ctx{opset};
             const onnx_kernels::kernel::Concat concat_kernel{concat_kernel_ctx};

             Tensor x0 = RandnTensor(DataType::FLOAT, shape, 2001);
             Tensor x1 = RandnTensor(DataType::FLOAT, shape, 2002);
             Tensor y = concat_kernel({x0, x1}, 0);
             return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
           });
    return;
  }

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
    const std::vector<int64_t> shape = sc.shape;
    const std::vector<float> values0 = sc.values0;
    const std::vector<float> values1 = sc.values1;
    const std::string label = sc.label;

    // Positive axes: 0 .. rank-1.
    for (int64_t axis = 0; axis < rank; ++axis) {
      Expect(registry, MakeConcatNode(axis),
             std::string("test_cc_concat_") + label + "_axis_" + std::to_string(axis), {opset},
             [shape, values0, values1, axis]() -> IoData {
               const OpsetId opset = DefaultOpset(13);

               const KernelContext concat_kernel_ctx{opset};
               const onnx_kernels::kernel::Concat concat_kernel{concat_kernel_ctx};

               Tensor x0 = Tensor::FromFloat("", shape, values0);
               Tensor x1 = Tensor::FromFloat("", shape, values1);
               Tensor y = concat_kernel({x0, x1}, axis);
               return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
             });
    }

    // Negative axes: -rank .. -1.
    for (int64_t axis = -rank; axis < 0; ++axis) {
      Expect(registry, MakeConcatNode(axis),
             std::string("test_cc_concat_") + label + "_axis_negative_" + std::to_string(-axis),
             {opset}, [shape, values0, values1, axis]() -> IoData {
               const OpsetId opset = DefaultOpset(13);

               const KernelContext concat_kernel_ctx{opset};
               const onnx_kernels::kernel::Concat concat_kernel{concat_kernel_ctx};

               Tensor x0 = Tensor::FromFloat("", shape, values0);
               Tensor x1 = Tensor::FromFloat("", shape, values1);
               Tensor y = concat_kernel({x0, x1}, axis);
               return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
             });
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
