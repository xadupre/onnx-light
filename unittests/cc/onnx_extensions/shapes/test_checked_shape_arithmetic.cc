// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_proto/onnx_helper.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

constexpr int64_t kTwo32 = int64_t{1} << 32;
constexpr int64_t kTwo62 = int64_t{1} << 62;
using NamedShape = std::pair<std::string, std::vector<int64_t>>;

core::symbolic::SymShape InferCheckedShape(const NodeProto &node,
                                           const std::vector<NamedShape> &inputs,
                                           const std::vector<NamedShape> &constants, bool wrapped,
                                           const std::vector<NodeProto> &prefix = {}) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(25);
  auto *graph = model.add_graph();
  graph->set_name("checked_arithmetic");
  std::vector<std::string> names;
  for (const auto &[name, dims] : inputs) {
    auto *input = graph->add_input();
    input->set_name(name);
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::DataType::FLOAT);
    auto *shape = type->mutable_shape();
    for (int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
    names.push_back(name);
  }
  for (const auto &[name, values] : constants) {
    auto *tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(TensorProto::DataType::INT64);
    tensor->add_dims(static_cast<int64_t>(values.size()));
    for (int64_t value : values) {
      tensor->add_int64_data(value);
    }
    names.push_back(name);
  }
  for (const auto &name : node.output()) {
    auto *output = graph->add_output();
    output->set_name(name);
    output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  }
  if (wrapped) {
    auto *opset = model.add_opset_import();
    opset->set_domain("local");
    opset->set_version(1);
    auto *function = model.add_functions();
    function->set_domain("local");
    function->set_name("CheckedShape");
    function->add_opset_import()->set_version(25);
    for (const auto &name : names) {
      function->add_input(name);
    }
    for (const auto &name : node.output()) {
      function->add_output(name);
    }
    for (const auto &item : prefix) {
      *function->add_node() = item;
    }
    *function->add_node() = node;
    auto *call = graph->add_node();
    call->set_op_type("CheckedShape");
    call->set_domain("local");
    for (const auto &name : names) {
      call->add_input(name);
    }
    for (const auto &name : node.output()) {
      call->add_output(name);
    }
  } else {
    for (const auto &item : prefix) {
      *graph->add_node() = item;
    }
    *graph->add_node() = node;
  }
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  return context.Get(node.output(0)).Shape();
}

void ExpectOverflow(const NodeProto &node, const std::vector<NamedShape> &inputs,
                    const std::vector<NamedShape> &constants = {},
                    const std::vector<NodeProto> &prefix = {}) {
  for (bool wrapped : {false, true}) {
    SCOPED_TRACE(wrapped);
    try {
      InferCheckedShape(node, inputs, constants, wrapped, prefix);
      FAIL() << "Expected checked dimension arithmetic to reject overflow";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("Integer overflow"), std::string::npos)
          << error.what();
    }
  }
}

void ExpectShape(const NodeProto &node, const std::vector<NamedShape> &inputs,
                 const std::vector<NamedShape> &constants, const std::vector<int64_t> &expected) {
  for (bool wrapped : {false, true}) {
    SCOPED_TRACE(wrapped);
    const auto shape = InferCheckedShape(node, inputs, constants, wrapped);
    ASSERT_EQ(shape.Rank(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      ASSERT_TRUE(shape[i].IsInt());
      EXPECT_EQ(shape[i].AsInt(), expected[i]);
    }
  }
}

} // namespace

TEST(CheckedShapeArithmetic, SharedHelpersCheckSignedBoundaries) {
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  constexpr int64_t minimum = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(core::shapes::CheckedDimensionMultiply(maximum, 1), maximum);
  EXPECT_EQ(core::shapes::CheckedDimensionMultiply(minimum, 0), 0);
  EXPECT_EQ(core::shapes::CheckedDimensionMultiply(minimum, 1), minimum);
  EXPECT_EQ(core::shapes::CheckedDimensionAdd(maximum, 0), maximum);
  EXPECT_EQ(core::shapes::CheckedDimensionAdd(maximum - 1, 1), maximum);
  EXPECT_EQ(core::shapes::CheckedDimensionAdd(minimum, 0), minimum);
  EXPECT_THROW(core::shapes::CheckedDimensionMultiply(maximum, 2), std::invalid_argument);
  EXPECT_THROW(core::shapes::CheckedDimensionMultiply(minimum, -1), std::invalid_argument);
  EXPECT_THROW(core::shapes::CheckedDimensionAdd(maximum, 1), std::invalid_argument);
  EXPECT_THROW(core::shapes::CheckedDimensionAdd(minimum, -1), std::invalid_argument);
}

TEST(CheckedShapeArithmetic, TileRejectsConstantAndPropagatedOverflow) {
  const auto node = MakeNode("Tile", {"X", "repeats"}, {"Y"});
  ExpectOverflow(node, {{"X", {kTwo32}}}, {{"repeats", {kTwo32}}});
  ExpectOverflow(node, {{"X", {kTwo32}}, {"source", {kTwo32}}}, {},
                 {MakeNode("Shape", {"source"}, {"repeats"})});
  ExpectShape(node, {{"X", {0}}}, {{"repeats", {kTwo32}}}, {0});
  ExpectShape(node, {{"X", {3}}}, {{"repeats", {2}}}, {6});
  ExpectShape(node, {{"X", {1}}}, {{"repeats", {std::numeric_limits<int64_t>::max()}}},
              {std::numeric_limits<int64_t>::max()});
}

TEST(CheckedShapeArithmetic, ReshapeRejectsOverflowBeforeZeroOrInferredDimension) {
  auto node = MakeNode("Reshape", {"X", "target"}, {"Y"});
  AddAttribute<int64_t>(node, "allowzero", 1);
  for (int64_t last : {int64_t{0}, int64_t{-1}}) {
    SCOPED_TRACE(last);
    ExpectOverflow(node, {{"X", {0}}}, {{"target", {kTwo32, kTwo32, last}}});
  }
  ExpectShape(node, {{"X", {0}}}, {{"target", {2, 3, 0}}}, {2, 3, 0});
  node = MakeNode("Reshape", {"X", "target"}, {"Y"});
  ExpectOverflow(node, {{"X", {1, kTwo32, 0}}}, {{"target", {kTwo32, 0, 0}}});
  ExpectOverflow(node, {{"X", {kTwo32, kTwo32, 0}}}, {{"target", {-1}}});
  ExpectShape(node, {{"X", {1, 3, 0}}}, {{"target", {2, 0, 0}}}, {2, 3, 0});
  ExpectShape(node, {{"X", {2, 3}}}, {{"target", {-1, 2}}}, {3, 2});
}

TEST(CheckedShapeArithmetic, FlattenChecksEachOutputProduct) {
  auto node = MakeNode("Flatten", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "axis", 1);
  ExpectOverflow(node, {{"X", {0, kTwo32, kTwo32}}});
  ExpectShape(node, {{"X", {0, 2, 3}}}, {}, {0, 6});
  node = MakeNode("Flatten", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "axis", -1);
  ExpectOverflow(node, {{"X", {kTwo32, kTwo32, 0}}});
  ExpectShape(node, {{"X", {2, 3, 4}}}, {}, {6, 4});
}

TEST(CheckedShapeArithmetic, DepthToSpaceChecksBlockAndBothSpatialDimensions) {
  auto node = MakeNode("DepthToSpace", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "blocksize", kTwo32);
  ExpectOverflow(node, {{"X", {1, 0, 1, 1}}});
  node = MakeNode("DepthToSpace", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "blocksize", 2);
  ExpectOverflow(node, {{"X", {0, 4, kTwo62, 1}}});
  ExpectOverflow(node, {{"X", {0, 4, 1, kTwo62}}});
  ExpectShape(node, {{"X", {0, 4, 2, 3}}}, {}, {0, 1, 4, 6});
  ExpectShape(node, {{"X", {1, 0, 2, 3}}}, {}, {1, 0, 4, 6});
}

TEST(CheckedShapeArithmetic, PackedAttentionChecksOutputWidth) {
  for (const std::string op : {"Attention", "LinearAttention"}) {
    SCOPED_TRACE(op);
    auto node = MakeNode(op.c_str(), {"Q", "K", "V"}, {"Y"});
    AddAttribute<int64_t>(node, "q_num_heads", kTwo32);
    AddAttribute<int64_t>(node, "kv_num_heads", 1);
    ExpectOverflow(node, {{"Q", {1, 1, kTwo32}}, {"K", {1, 1, 1}}, {"V", {1, 1, kTwo32}}});
    if (op == "LinearAttention") {
      node.add_input("past");
      ExpectOverflow(node, {{"Q", {1, 1, kTwo32}},
                            {"K", {1, 1, 1}},
                            {"V", {1, 1, kTwo32}},
                            {"past", {1, 1, 1, kTwo32}}});
    }
    node = MakeNode(op.c_str(), {"Q", "K", "V"}, {"Y"});
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    ExpectShape(node, {{"Q", {1, 3, 8}}, {"K", {1, 3, 4}}, {"V", {1, 3, 6}}}, {}, {1, 3, 12});
    ExpectShape(node, {{"Q", {0, 3, 8}}, {"K", {0, 3, 4}}, {"V", {0, 3, 6}}}, {}, {0, 3, 12});
  }
}

TEST(CheckedShapeArithmetic, AttentionChecksCacheLengthForBothRanks) {
  for (bool packed : {false, true}) {
    SCOPED_TRACE(packed);
    auto node =
        MakeNode("Attention", {"Q", "K", "V", "", "past_key", "past_value"}, {"Y", "PK", "PV"});
    if (packed) {
      AddAttribute<int64_t>(node, "q_num_heads", 1);
      AddAttribute<int64_t>(node, "kv_num_heads", 1);
    }
    const std::vector<int64_t> shape =
        packed ? std::vector<int64_t>{0, kTwo62, 1} : std::vector<int64_t>{0, 1, kTwo62, 1};
    ExpectOverflow(node, {{"Q", shape},
                          {"K", shape},
                          {"V", shape},
                          {"past_key", {0, 1, kTwo62, 1}},
                          {"past_value", {0, 1, kTwo62, 1}}});
    const std::vector<int64_t> valid =
        packed ? std::vector<int64_t>{0, 3, 1} : std::vector<int64_t>{0, 1, 3, 1};
    ExpectShape(node,
                {{"Q", valid},
                 {"K", valid},
                 {"V", valid},
                 {"past_key", {0, 1, 2, 1}},
                 {"past_value", {0, 1, 2, 1}}},
                {}, valid);
  }
}

} // namespace Test
