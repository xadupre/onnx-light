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

void RegisterScatterNDCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::ScatterND snd_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeScatterNDNode("none");
    Expect(registry, std::move(node), "test_cc_scatternd_benchmark", {opset},
           {4194304, 256, 4194304}, {4194304}, [snd_kernel]() -> IoData {
             Tensor data =
                 Tensor::FromFloat("", {256, 128, 128}, Randn<float>({256, 128, 128}, 2001));
             std::vector<int64_t> index_values(256);
             for (int64_t i = 0; i < 256; ++i) {
               index_values[static_cast<std::size_t>(i)] = i;
             }
             Tensor indices = Tensor::FromInt64("", {256, 1}, index_values);
             Tensor updates =
                 Tensor::FromFloat("", {256, 128, 128}, Randn<float>({256, 128, 128}, 2002));
             onnx_kernels::kernel::ScatterND::Attributes attrs;
             Tensor output = snd_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
    return;
  }

  // test_cc_scatternd — mirrors upstream ``test_scatternd``.
  {
    Expect(registry, MakeScatterNDNode("none"), "test_cc_scatternd", {opset}, [=]() -> IoData {
      Tensor data = MakeData4x4x4();
      Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 2});
      Tensor updates = MakeUpdates2x4x4();
      onnx_kernels::kernel::ScatterND::Attributes attrs;
      Tensor output = snd_kernel(data, indices, updates, attrs);
      return IoData{{std::move(data), std::move(indices), std::move(updates)}, {std::move(output)}};
    });
  }

  // test_cc_scatternd_add — mirrors upstream ``test_scatternd_add``.
  {
    Expect(registry, MakeScatterNDNode("add"), "test_cc_scatternd_add", {opset}, [=]() -> IoData {
      Tensor data = MakeData4x4x4();
      Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
      Tensor updates = MakeUpdates2x4x4();
      onnx_kernels::kernel::ScatterND::Attributes attrs;
      attrs.reduction = "add";
      Tensor output = snd_kernel(data, indices, updates, attrs);
      return IoData{{std::move(data), std::move(indices), std::move(updates)}, {std::move(output)}};
    });
  }

  // test_cc_scatternd_multiply — mirrors upstream ``test_scatternd_multiply``.
  {
    Expect(registry, MakeScatterNDNode("mul"), "test_cc_scatternd_multiply", {opset},
           [=]() -> IoData {
             Tensor data = MakeData4x4x4();
             Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
             Tensor updates = MakeUpdates2x4x4();
             onnx_kernels::kernel::ScatterND::Attributes attrs;
             attrs.reduction = "mul";
             Tensor output = snd_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
  }

  // test_cc_scatternd_max — mirrors upstream ``test_scatternd_max``.
  {
    Expect(registry, MakeScatterNDNode("max"), "test_cc_scatternd_max", {opset}, [=]() -> IoData {
      Tensor data = MakeData4x4x4();
      Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
      Tensor updates = MakeUpdates2x4x4();
      onnx_kernels::kernel::ScatterND::Attributes attrs;
      attrs.reduction = "max";
      Tensor output = snd_kernel(data, indices, updates, attrs);
      return IoData{{std::move(data), std::move(indices), std::move(updates)}, {std::move(output)}};
    });
  }

  // test_cc_scatternd_min — mirrors upstream ``test_scatternd_min``.
  {
    Expect(registry, MakeScatterNDNode("min"), "test_cc_scatternd_min", {opset}, [=]() -> IoData {
      Tensor data = MakeData4x4x4();
      Tensor indices = Tensor::FromInt64("", {2, 1}, {0, 0});
      Tensor updates = MakeUpdates2x4x4();
      onnx_kernels::kernel::ScatterND::Attributes attrs;
      attrs.reduction = "min";
      Tensor output = snd_kernel(data, indices, updates, attrs);
      return IoData{{std::move(data), std::move(indices), std::move(updates)}, {std::move(output)}};
    });
  }

  // test_cc_scatternd_max_with_element_indices — mirrors upstream
  // ``test_scatternd_max_with_element_indices``.
  // Indices address individual elements (index rank == data rank), exercising
  // element-level reduction.
  {
    Expect(registry, MakeScatterNDNode("max"), "test_cc_scatternd_max_with_element_indices",
           {opset}, [=]() -> IoData {
             Tensor data = Tensor::FromFloat("", {2, 2}, {1, 2, 3, 4});
             Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 1});
             Tensor updates = Tensor::FromFloat("", {2}, {5, 1});
             onnx_kernels::kernel::ScatterND::Attributes attrs;
             attrs.reduction = "max";
             Tensor output = snd_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
  }

  // test_cc_scatternd_min_with_element_indices — mirrors upstream
  // ``test_scatternd_min_with_element_indices``.
  {
    Expect(registry, MakeScatterNDNode("min"), "test_cc_scatternd_min_with_element_indices",
           {opset}, [=]() -> IoData {
             Tensor data = Tensor::FromFloat("", {2, 2}, {1, 2, 3, 4});
             Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 1});
             Tensor updates = Tensor::FromFloat("", {2}, {5, 1});
             onnx_kernels::kernel::ScatterND::Attributes attrs;
             attrs.reduction = "min";
             Tensor output = snd_kernel(data, indices, updates, attrs);
             return IoData{{std::move(data), std::move(indices), std::move(updates)},
                           {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
