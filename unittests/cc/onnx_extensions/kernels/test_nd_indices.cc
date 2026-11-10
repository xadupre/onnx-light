// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::DefaultOpset;
using core::runtime::KernelContext;
using core::runtime::RunNode;
using core::runtime::RuntimeContext;
using core::runtime::Tensor;

namespace {

Tensor NDIndices(bool int32, const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &values) {
  if (int32) {
    std::vector<int32_t> indices;
    indices.reserve(values.size());
    for (int64_t value : values) {
      indices.push_back(static_cast<int32_t>(value));
    }
    return Tensor::FromInt32("", shape, indices);
  }
  return Tensor::FromInt64("", shape, values);
}

void ExpectNDValues(const Tensor &tensor, const std::vector<int64_t> &shape,
                    const std::vector<float> &values) {
  EXPECT_EQ(tensor.shape, shape);
  ASSERT_EQ(tensor.element_count(), static_cast<int64_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(tensor.AsFloat()[i], values[i]) << i;
  }
}

} // namespace

TEST(NDIndicesKernels, GatherScalarAndBounds) {
  const onnx_kernels::kernel::GatherND gather{KernelContext(DefaultOpset(29))};
  const Tensor data = Tensor::FromFloat("", {2, 3}, {0, 1, 2, 3, 4, 5});
  for (bool int32 : {false, true}) {
    const Tensor output = gather(data, NDIndices(int32, {2}, {-1, -1}));
    ExpectNDValues(output, {}, {5});
    for (int64_t index : std::vector<int64_t>{2, -3, std::numeric_limits<int32_t>::min(),
                                              std::numeric_limits<int32_t>::max()}) {
      EXPECT_THROW(gather(data, NDIndices(int32, {2}, {index, 0})), std::invalid_argument);
    }
  }
}

TEST(NDIndicesKernels, GatherEmptyBatchesAndSelections) {
  const onnx_kernels::kernel::GatherND gather{KernelContext(DefaultOpset(29))};
  for (bool int32 : {false, true}) {
    const Tensor empty = Tensor::FromFloat("", {0, 4, 3}, {});
    ExpectNDValues(gather(empty, NDIndices(int32, {0, 1, 2, 1}, {}), 1), {0, 1, 2, 3}, {});
    const Tensor data = Tensor::FromFloat("", {2, 4, 3}, std::vector<float>(24, 1));
    ExpectNDValues(gather(data, NDIndices(int32, {2, 1, 0, 1}, {}), 1), {2, 1, 0, 3}, {});
  }
}

TEST(NDIndicesKernels, GatherMultipleBatchDimensions) {
  const onnx_kernels::kernel::GatherND gather{KernelContext(DefaultOpset(29))};
  const Tensor data = Tensor::FromFloat("", {2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  for (bool int32 : {false, true}) {
    ExpectNDValues(gather(data, NDIndices(int32, {2, 2, 2, 1}, {0, -1, 0, -1, 0, -1, 0, -1}), 2),
                   {2, 2, 2}, {0, 2, 3, 5, 6, 8, 9, 11});
  }
}

TEST(NDIndicesKernels, ScatterReductionsAndEmptyUpdates) {
  const onnx_kernels::kernel::ScatterND scatter{KernelContext(DefaultOpset(29))};
  const Tensor data = Tensor::FromFloat("", {2, 3, 2}, std::vector<float>(12, 1));
  const std::vector<std::pair<std::string, std::vector<float>>> cases{
      {"none", {1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 4, 5}},
      {"add", {1, 1, 9, 11, 1, 1, 1, 1, 1, 1, 5, 6}},
      {"mul", {1, 1, 12, 21, 1, 1, 1, 1, 1, 1, 4, 5}},
      {"max", {1, 1, 6, 7, 1, 1, 1, 1, 1, 1, 4, 5}},
      {"min", std::vector<float>(12, 1)},
  };
  for (bool int32 : {false, true}) {
    for (const auto &[reduction, expected] : cases) {
      SCOPED_TRACE(reduction);
      onnx_kernels::kernel::ScatterND::Attributes attrs;
      attrs.reduction = reduction;
      const bool none = reduction == "none";
      const Tensor indices = NDIndices(int32, {none ? 2 : 3, 2},
                                       none ? std::vector<int64_t>{0, 1, 1, -1}
                                            : std::vector<int64_t>{0, 1, 1, -1, 0, -2});
      const Tensor updates = Tensor::FromFloat("", {none ? 2 : 3, 2},
                                               none ? std::vector<float>{2, 3, 4, 5}
                                                    : std::vector<float>{2, 3, 4, 5, 6, 7});
      ExpectNDValues(scatter(data, indices, updates, attrs), {2, 3, 2}, expected);
      ExpectNDValues(
          scatter(data, NDIndices(int32, {0, 2}, {}), Tensor::FromFloat("", {0, 2}, {}), attrs),
          {2, 3, 2}, std::vector<float>(12, 1));
    }
  }
}

TEST(NDIndicesKernels, ScatterBounds) {
  const onnx_kernels::kernel::ScatterND scatter{KernelContext(DefaultOpset(29))};
  const Tensor data = Tensor::FromFloat("", {2}, {0, 1});
  const Tensor updates = Tensor::FromFloat("", {1}, {2});
  const onnx_kernels::kernel::ScatterND::Attributes attrs;
  for (bool int32 : {false, true}) {
    for (int64_t index : std::vector<int64_t>{2, -3, std::numeric_limits<int32_t>::min(),
                                              std::numeric_limits<int32_t>::max()}) {
      EXPECT_THROW(scatter(data, NDIndices(int32, {1, 1}, {index}), updates, attrs),
                   std::invalid_argument);
    }
  }
}

TEST(NDIndicesKernels, RejectsInvalidTypesEvenForEmptyIndices) {
  const Tensor data = Tensor::FromFloat("", {2}, {0, 1});
  const Tensor updates = Tensor::FromFloat("", {0}, {});
  const onnx_kernels::kernel::ScatterND::Attributes attrs;
  for (int version : {28, 29}) {
    const onnx_kernels::kernel::GatherND gather{KernelContext(DefaultOpset(version))};
    const onnx_kernels::kernel::ScatterND scatter{KernelContext(DefaultOpset(version))};
    const Tensor floats = Tensor::FromFloat("", {0, 1}, {});
    EXPECT_THROW(gather(data, floats), std::invalid_argument);
    EXPECT_THROW(scatter(data, floats, updates, attrs), std::invalid_argument);
    if (version < 29) {
      const Tensor int32 = NDIndices(true, {0, 1}, {});
      EXPECT_THROW(gather(data, int32), std::invalid_argument);
      EXPECT_THROW(scatter(data, int32, updates, attrs), std::invalid_argument);
    }
  }
}

TEST(NDIndicesKernels, DispatchesInt32Indices) {
  onnx_kernels::RegisterKernelFunctions();
  RuntimeContext rt{KernelContext(DefaultOpset(29))};
  rt.tensors()["data"] = Tensor::FromFloat("", {2, 3}, {0, 1, 2, 3, 4, 5});
  rt.tensors()["indices"] = Tensor::FromInt32("", {1, 1}, {-1});
  NodeProto gather;
  gather.set_op_type("GatherND");
  gather.add_input("data");
  gather.add_input("indices");
  gather.add_output("selected");
  RunNode(gather, rt);
  ExpectNDValues(rt.tensors().at("selected"), {1, 3}, {3, 4, 5});

  rt.tensors()["updates"] = Tensor::FromFloat("", {1, 3}, {7, 8, 9});
  NodeProto scatter;
  scatter.set_op_type("ScatterND");
  scatter.add_input("data");
  scatter.add_input("indices");
  scatter.add_input("updates");
  scatter.add_output("updated");
  RunNode(scatter, rt);
  ExpectNDValues(rt.tensors().at("updated"), {2, 3}, {0, 1, 2, 7, 8, 9});
}
