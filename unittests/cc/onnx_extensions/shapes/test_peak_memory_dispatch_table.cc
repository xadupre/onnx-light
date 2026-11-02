// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx.h"

#include "onnx_core/shapes/dispatch_table.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(CoreShapesPeakMemory, DefaultsToUnresolvedMaximumWhenUnregistered) {
  const core::shapes::Device device = core::shapes::Device::kCPU;
  const std::vector<core::shapes::SymShape> input_shapes;
  const auto estimate =
      core::shapes::ComputePeakMemory("ai.onnx", "OpWithoutPeakMemoryFn", device, input_shapes);
  ASSERT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_FALSE(estimate.worst_case.AsExpr().empty());
  EXPECT_FALSE(estimate.average_case);
  EXPECT_EQ(estimate,
            core::shapes::ComputePeakMemory("", "OpWithoutPeakMemoryFn", device, input_shapes));
}

// A registered peak-memory function is invoked with the device and the input
// shapes, and its result is returned by ComputePeakMemory.
TEST(CoreShapesPeakMemory, RegisteredFunctionIsDispatched) {
  core::shapes::Device seen_device = core::shapes::Device::kUndefined;
  std::size_t seen_num_inputs = 0;
  core::shapes::RegisterComputePeakMemoryFn(
      "ai.onnx", "PeakMemoryRegisteredOp", core::shapes::Device::kCPU,
      [&](core::shapes::Device device,
          const std::vector<core::shapes::SymShape> &input_shapes) -> int64_t {
        seen_device = device;
        seen_num_inputs = input_shapes.size();
        return 4096;
      });

  const std::vector<core::shapes::SymShape> input_shapes(2);
  const auto estimate = core::shapes::ComputePeakMemory("ai.onnx", "PeakMemoryRegisteredOp",
                                                        core::shapes::Device::kCPU, input_shapes);
  EXPECT_EQ(estimate.worst_case.AsInt(), 4096);
  EXPECT_FALSE(estimate.average_case);
  EXPECT_EQ(seen_device, core::shapes::Device::kCPU);
  EXPECT_EQ(seen_num_inputs, 2u);

  const auto &table = core::shapes::PeakMemoryDispatchTable();
  EXPECT_NE(table.find("ai.onnx:PeakMemoryRegisteredOp"), table.end());
}

// An empty domain is normalised to the default ONNX domain, so registration
// and lookup with "" and "ai.onnx" refer to the same entry.
TEST(CoreShapesPeakMemory, EmptyDomainNormalisedToOnnxDomain) {
  core::shapes::RegisterComputePeakMemoryFn(
      "", "PeakMemoryEmptyDomainOp", core::shapes::Device::kCPU,
      [](core::shapes::Device, const std::vector<core::shapes::SymShape> &) -> int64_t {
        return 123;
      });

  const std::vector<core::shapes::SymShape> input_shapes;
  EXPECT_EQ(core::shapes::ComputePeakMemory("ai.onnx", "PeakMemoryEmptyDomainOp",
                                            core::shapes::Device::kCPU, input_shapes),
            123);
  EXPECT_EQ(core::shapes::ComputePeakMemory("", "PeakMemoryEmptyDomainOp",
                                            core::shapes::Device::kCPU, input_shapes),
            123);
}

TEST(CoreShapesPeakMemory, NegativeEstimateOmitsEmptyDomainPrefix) {
  core::shapes::RegisterComputePeakMemoryFn(
      "", "PeakMemoryNegativeEstimateOp", core::shapes::Device::kCPU,
      [](core::shapes::Device, const std::vector<core::shapes::SymShape> &) -> int64_t {
        return -1;
      });

  const std::vector<core::shapes::SymShape> input_shapes;
  try {
    core::shapes::ComputePeakMemory("", "PeakMemoryNegativeEstimateOp", core::shapes::Device::kCPU,
                                    input_shapes);
    FAIL() << "Expected a negative estimate to be rejected.";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what())
                  .find("Negative peak-memory estimate for PeakMemoryNegativeEstimateOp."),
              std::string::npos);
  }
}

// The device is part of a peak-memory function's identifier: a function
// registered for a GPU device is keyed separately from the CPU/default entry
// and is only resolved when ComputePeakMemory is called with that device.
TEST(CoreShapesPeakMemory, DeviceIsPartOfIdentifier) {
  const core::shapes::Device gpu = core::symbolic::MakeGPUDevice(0);
  core::shapes::RegisterComputePeakMemoryFn(
      "ai.onnx", "PeakMemoryDeviceOp", gpu,
      [](core::shapes::Device, const std::vector<core::shapes::SymShape> &) -> int64_t {
        return 777;
      });

  const std::vector<core::shapes::SymShape> input_shapes;
  // Resolved for the device it was registered with.
  EXPECT_EQ(core::shapes::ComputePeakMemory("ai.onnx", "PeakMemoryDeviceOp", gpu, input_shapes),
            777);
  EXPECT_TRUE(core::shapes::ComputePeakMemory("ai.onnx", "PeakMemoryDeviceOp",
                                              core::shapes::Device::kCPU, input_shapes)
                  .worst_case.IsExpr());

  const auto &table = core::shapes::PeakMemoryDispatchTable();
  EXPECT_EQ(table.find("ai.onnx:PeakMemoryDeviceOp"), table.end());
  EXPECT_NE(table.find("ai.onnx:PeakMemoryDeviceOp:" + std::to_string(static_cast<int32_t>(gpu))),
            table.end());
}

TEST(CoreShapesPeakMemory, NodeAwareRegistrationPreservesLegacyCallsAndReplacement) {
  const auto device = core::shapes::Device::kCPU;
  NodeProto node;
  node.set_domain("ai.onnx");
  node.set_op_type("NodeAwarePeakMemoryOp");
  auto *attr = node.add_attribute();
  attr->set_name("bytes");
  attr->set_type(AttributeProto::INT);
  attr->set_i(128);
  const core::shapes::ComputePeakMemoryNodeFn estimator =
      [](core::shapes::Device, const NodeProto &current,
         const std::vector<core::shapes::SymShape> &inputs) -> int64_t {
    EXPECT_EQ(current.op_type(), "NodeAwarePeakMemoryOp");
    EXPECT_EQ(inputs.size(), 2u);
    return current.attribute_size() == 0 ? 16 : current.attribute(0).i();
  };
  const std::vector<core::shapes::SymShape> inputs(2);
  core::shapes::RegisterComputePeakMemoryFn("", node.op_type(), device, estimator);
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs), 128);
  EXPECT_EQ(core::shapes::ComputePeakMemory("", node.op_type(), device, inputs), 16);
  EXPECT_EQ(core::shapes::ComputePeakMemory(core::symbolic::MakeGPUDevice(0), node, inputs),
            core::shapes::UnknownPeakMemory(node.domain(), node.op_type()));
  EXPECT_EQ(core::shapes::ComputePeakMemory(core::shapes::Device::kUndefined, node, inputs), 128);
  node.mutable_attribute(0)->set_i(-1);
  EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
  core::shapes::RegisterComputePeakMemoryFn(
      "", node.op_type(), device,
      [](core::shapes::Device, const std::vector<core::shapes::SymShape> &) -> int64_t {
        return 32;
      });
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs), 32);
  EXPECT_EQ(core::shapes::ComputePeakMemory("", node.op_type(), device, inputs), 32);
  node.mutable_attribute(0)->set_i(64);
  core::shapes::RegisterComputePeakMemoryFn("", node.op_type(), device, estimator);
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs), 64);
}

TEST(CoreShapesPeakMemory, ReturnsDistinctSymbolicWorstAndAverageCases) {
  using core::shapes::PeakMemoryEstimate;
  using core::symbolic::SymDim;
  const PeakMemoryEstimate expected{SymDim("128*N"), SymDim("64*N")};
  NodeProto node;
  node.set_op_type("SymbolicScratch");
  const auto device = core::shapes::Device::kCPU;
  for (bool node_aware : {false, true}) {
    SCOPED_TRACE(node_aware);
    if (node_aware) {
      core::shapes::RegisterComputePeakMemoryFn(
          "", node.op_type(), device,
          [expected](core::shapes::Device, const NodeProto &,
                     const std::vector<core::shapes::SymShape> &) { return expected; });
    } else {
      core::shapes::RegisterComputePeakMemoryFn(
          "", node.op_type(), device,
          [expected](core::shapes::Device, const std::vector<core::shapes::SymShape> &) {
            return expected;
          });
    }
    EXPECT_EQ(core::shapes::ComputePeakMemory("", node.op_type(), device, {}), expected);
    EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, {}), expected);
  }
}

TEST(CoreShapesPeakMemory, RejectsInvalidWorstAndAverageCases) {
  using core::shapes::PeakMemoryEstimate;
  using core::symbolic::SymDim;
  std::vector<PeakMemoryEstimate> invalid;
  invalid.emplace_back(int64_t{-1});
  invalid.emplace_back(SymDim(""), std::nullopt);
  invalid.emplace_back(SymDim(int64_t{10}), SymDim(int64_t{-1}));
  invalid.emplace_back(SymDim(int64_t{10}), SymDim(int64_t{11}));
  invalid.emplace_back(SymDim(int64_t{10}), SymDim(""));
  NodeProto node;
  node.set_op_type("InvalidScratch");
  const auto device = core::shapes::Device::kCPU;
  for (bool node_aware : {false, true}) {
    SCOPED_TRACE(node_aware);
    for (const auto &estimate : invalid) {
      if (node_aware) {
        core::shapes::RegisterComputePeakMemoryFn(
            "", node.op_type(), device,
            [estimate](core::shapes::Device, const NodeProto &,
                       const std::vector<core::shapes::SymShape> &) { return estimate; });
      } else {
        core::shapes::RegisterComputePeakMemoryFn(
            "", node.op_type(), device,
            [estimate](core::shapes::Device, const std::vector<core::shapes::SymShape> &) {
              return estimate;
            });
      }
      EXPECT_THROW(core::shapes::ComputePeakMemory("", node.op_type(), device, {}),
                   std::invalid_argument);
      EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, {}), std::invalid_argument);
    }
  }
}

} // namespace Test
