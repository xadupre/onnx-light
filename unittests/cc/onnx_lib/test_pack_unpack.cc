// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/shape_inference/implementation.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

TensorShapeProto PackingShape(std::initializer_list<int64_t> dimensions) {
  TensorShapeProto shape;
  for (auto dimension : dimensions) {
    shape.add_dim()->set_dim_value(dimension);
  }
  return shape;
}

ModelProto PackingModel(const std::string &op, const TensorShapeProto *shape,
                        const std::vector<AttributeProto> &attributes,
                        int storage_type = TensorProto::UINT8,
                        std::optional<int64_t> count = std::nullopt) {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("packing");
  auto *input = graph->add_input();
  input->set_name("X");
  auto *input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(op == "Unpack" ? storage_type : TensorProto::UINT8);
  if (shape != nullptr) {
    *input_type->mutable_shape() = *shape;
  }
  auto *node = graph->add_node();
  node->set_op_type(op);
  node->add_input("X");
  node->add_output("Y");
  for (const auto &attribute : attributes) {
    *node->add_attribute() = attribute;
  }
  if (op == "Pack") {
    *node->add_attribute() = MakeAttribute("output_type", int64_t{storage_type});
  } else {
    node->add_input("count");
    auto *count_input = graph->add_input();
    count_input->set_name("count");
    auto *count_type = count_input->mutable_type()->mutable_tensor_type();
    count_type->set_elem_type(TensorProto::INT64);
    count_type->mutable_shape();
    if (count.has_value()) {
      auto *initializer = graph->add_initializer();
      initializer->set_name("count");
      initializer->set_data_type(TensorProto::INT64);
      initializer->add_int64_data(*count);
    }
  }
  graph->add_output()->set_name("Y");
  return model;
}

void InferPacking(ModelProto &model, bool data_propagation = false) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, data_propagation));
}

const TensorShapeProto &PackingOutputShape(const ModelProto &model) {
  return model.graph().output(0).type().tensor_type().shape();
}

std::vector<std::vector<AttributeProto>> PackingAttributes() {
  std::vector<std::vector<AttributeProto>> result;
  for (int64_t bits = 1; bits <= 8; ++bits) {
    result.push_back({MakeAttribute("bits", bits)});
  }
  for (int64_t group : {1, 5, 39}) {
    result.push_back({MakeAttribute("radix", int64_t{3}), MakeAttribute("group_size", group)});
  }
  result.push_back({MakeAttribute("radix", int64_t{2}), MakeAttribute("group_size", int64_t{62})});
  result.push_back({MakeAttribute("radix", int64_t{5}), MakeAttribute("group_size", int64_t{27})});
  result.push_back({MakeAttribute("radix", int64_t{256}), MakeAttribute("group_size", int64_t{7})});
  return result;
}

} // namespace

TEST(onnx_pack_unpack, InfersBitsShapesForBothStorageTypesAndAxes) {
  for (int storage : {TensorProto::UINT8, TensorProto::UINT32}) {
    const int64_t word_bits = storage == TensorProto::UINT32 ? 32 : 8;
    for (int64_t axis : {-1, 0}) {
      for (int64_t bits = 1; bits <= 8; ++bits) {
        for (int64_t count : {0, 1, 7, 8, 9, 17}) {
          SCOPED_TRACE(std::to_string(storage) + "/" + std::to_string(axis) + "/" +
                       std::to_string(bits) + "/" + std::to_string(count));
          auto shape = PackingShape({count, count});
          const int selected = axis == 0 ? 0 : 1;
          shape.mutable_dim(1 - selected)->set_dim_param("rows");
          auto packed = PackingModel(
              "Pack", &shape, {MakeAttribute("bits", bits), MakeAttribute("axis", axis)}, storage);
          InferPacking(packed);
          const auto &packed_shape = PackingOutputShape(packed);
          ASSERT_EQ(packed_shape.dim_size(), 2);
          EXPECT_EQ(packed_shape.dim(selected).dim_value(),
                    (count * bits + word_bits - 1) / word_bits);
          EXPECT_EQ(packed_shape.dim(1 - selected).dim_param(), "rows");
          EXPECT_EQ(packed.graph().output(0).type().tensor_type().elem_type(), storage);
          auto unpacked = PackingModel("Unpack", &packed_shape,
                                       {MakeAttribute("bits", bits), MakeAttribute("axis", axis)},
                                       storage, count);
          InferPacking(unpacked);
          EXPECT_EQ(PackingOutputShape(unpacked).SerializeAsString(), shape.SerializeAsString());
          EXPECT_EQ(unpacked.graph().output(0).type().tensor_type().elem_type(),
                    TensorProto::UINT8);
          EXPECT_NO_THROW(checker::check_model(packed));
          EXPECT_NO_THROW(checker::check_model(unpacked));
        }
      }
    }
  }
}

TEST(onnx_pack_unpack, InfersGroupedPartialAndZeroDimensions) {
  for (int storage : {TensorProto::UINT8, TensorProto::UINT32}) {
    const int64_t word_bits = storage == TensorProto::UINT32 ? 32 : 8;
    for (int64_t group : {1, 5, 39}) {
      int64_t limit = 1;
      for (int64_t i = 0; i < group; ++i) {
        limit *= 3;
      }
      int64_t width = 0;
      for (auto maximum = limit - 1; maximum != 0; maximum /= 2) {
        ++width;
      }
      for (int64_t count : {0, 1, 2, 5, 6, 17, 40}) {
        auto shape = PackingShape({0, count});
        std::vector<AttributeProto> attributes{MakeAttribute("radix", int64_t{3}),
                                               MakeAttribute("group_size", group)};
        auto packed = PackingModel("Pack", &shape, attributes, storage);
        InferPacking(packed);
        const auto &packed_shape = PackingOutputShape(packed);
        EXPECT_EQ(packed_shape.dim(0).dim_value(), 0);
        const auto groups = count / group + (count % group != 0);
        EXPECT_EQ(packed_shape.dim(1).dim_value(), (groups * width + word_bits - 1) / word_bits);
        auto unpacked = PackingModel("Unpack", &packed_shape, attributes, storage, count);
        InferPacking(unpacked);
        EXPECT_EQ(PackingOutputShape(unpacked).SerializeAsString(), shape.SerializeAsString());
      }
    }
  }
}

TEST(onnx_pack_unpack, PreservesSymbolicDimensionsOnlyForByteIdentity) {
  auto shape = PackingShape({2, 3});
  shape.mutable_dim(1)->set_dim_param("codes");
  for (const auto *op : {"Pack", "Unpack"}) {
    for (int storage : {TensorProto::UINT8, TensorProto::UINT32}) {
      for (int64_t bits : {3, 8}) {
        auto model = PackingModel(op, &shape, {MakeAttribute("bits", bits)}, storage);
        InferPacking(model);
        const auto &dimension = PackingOutputShape(model).dim(1);
        EXPECT_FALSE(dimension.has_dim_value());
        if (storage == TensorProto::UINT8 && bits == 8) {
          EXPECT_EQ(dimension.dim_param(), "codes");
        } else {
          EXPECT_NE(dimension.dim_param(), "codes");
        }
      }
    }
    auto model = PackingModel(op, nullptr, {MakeAttribute("bits", int64_t{3})});
    InferPacking(model);
    EXPECT_EQ(model.graph().output(0).type().tensor_type().elem_type(), TensorProto::UINT8);
    EXPECT_FALSE(model.graph().output(0).type().tensor_type().has_shape());
  }
}

TEST(onnx_pack_unpack, InfersConstantAndDynamicCounts) {
  for (int64_t count : {0, 1, 9}) {
    auto shape = PackingShape({2, (count * 3 + 7) / 8});
    auto dynamic = PackingModel("Unpack", &shape, {MakeAttribute("bits", int64_t{3})});
    InferPacking(dynamic);
    EXPECT_FALSE(PackingOutputShape(dynamic).dim(1).has_dim_value());
    auto constant = PackingModel("Unpack", &shape, {MakeAttribute("bits", int64_t{3})},
                                 TensorProto::UINT8, count);
    InferPacking(constant);
    EXPECT_EQ(PackingOutputShape(constant).dim(1).dim_value(), count);
    auto raw = PackingModel("Unpack", &shape, {MakeAttribute("bits", int64_t{3})},
                            TensorProto::UINT8, count);
    auto *initializer = raw.mutable_graph()->mutable_initializer(0);
    initializer->clear_int64_data();
    std::string bytes(sizeof(int64_t), '\0');
    bytes[0] = static_cast<char>(count);
    initializer->set_raw_data(bytes);
    InferPacking(raw);
    EXPECT_EQ(PackingOutputShape(raw).dim(1).dim_value(), count);
  }
}

TEST(onnx_pack_unpack, InfersPropagatedNumericAndSymbolicCounts) {
  for (bool symbolic : {false, true}) {
    auto shape = PackingShape({2, 17});
    shape.mutable_dim(1)->Clear();
    auto model = PackingModel("Unpack", &shape, {MakeAttribute("bits", int64_t{3})});
    auto *graph = model.mutable_graph();
    auto *weights = graph->add_input();
    weights->set_name("W");
    *weights->mutable_type() = graph->input(0).type();
    auto *dimension =
        weights->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
    if (symbolic) {
      dimension->set_dim_param("codes");
    } else {
      dimension->set_dim_value(17);
    }
    auto *axis = graph->add_initializer();
    axis->set_name("axis");
    axis->set_data_type(TensorProto::INT64);
    axis->add_int64_data(-1);
    const auto unpack = graph->node(0);
    graph->clear_node();
    auto *shape_node = graph->add_node();
    shape_node->set_op_type("Shape");
    shape_node->add_input("W");
    shape_node->add_output("shape");
    auto *gather = graph->add_node();
    gather->set_op_type("Gather");
    gather->add_input("shape");
    gather->add_input("axis");
    gather->add_output("count");
    *graph->add_node() = unpack;
    // count is produced by Gather rather than supplied as a graph input.
    graph->mutable_input()->erase(graph->mutable_input()->begin() + 1);
    InferPacking(model, true);
    const auto &actual = PackingOutputShape(model).dim(1);
    if (symbolic) {
      EXPECT_EQ(actual.dim_param(), "codes");
    } else {
      EXPECT_EQ(actual.dim_value(), 17);
    }
  }
}

TEST(onnx_pack_unpack, RejectsInvalidPackingAttributesEvenWithUnknownShape) {
  const std::vector<std::vector<AttributeProto>> invalid{
      {},
      {MakeAttribute("bits", int64_t{0})},
      {MakeAttribute("bits", int64_t{9})},
      {MakeAttribute("bits", int64_t{3}), MakeAttribute("radix", int64_t{3})},
      {MakeAttribute("bits", int64_t{3}), MakeAttribute("group_size", int64_t{5})},
      {MakeAttribute("radix", int64_t{3})},
      {MakeAttribute("group_size", int64_t{5})},
      {MakeAttribute("radix", int64_t{1}), MakeAttribute("group_size", int64_t{5})},
      {MakeAttribute("radix", int64_t{257}), MakeAttribute("group_size", int64_t{1})},
      {MakeAttribute("radix", int64_t{3}), MakeAttribute("group_size", int64_t{0})},
      {MakeAttribute("radix", int64_t{3}), MakeAttribute("group_size", int64_t{-1})},
      {MakeAttribute("radix", int64_t{3}), MakeAttribute("group_size", int64_t{40})},
      {MakeAttribute("radix", int64_t{2}), MakeAttribute("group_size", int64_t{63})},
      {MakeAttribute("radix", int64_t{256}),
       MakeAttribute("group_size", std::numeric_limits<int64_t>::max())}};
  for (const auto *op : {"Pack", "Unpack"}) {
    for (const auto &attributes : invalid) {
      auto model = PackingModel(op, nullptr, attributes);
      EXPECT_THROW(InferPacking(model), InferenceError);
    }
  }
}

TEST(onnx_pack_unpack, RejectsInvalidShapesCountsStorageAndAxes) {
  auto scalar = PackingShape({});
  auto shape = PackingShape({1});
  const std::vector<AttributeProto> attributes{MakeAttribute("bits", int64_t{3})};
  for (const auto *op : {"Pack", "Unpack"}) {
    auto model = PackingModel(op, &scalar, attributes);
    EXPECT_THROW(InferPacking(model), InferenceError);
    for (int64_t axis :
         std::initializer_list<int64_t>{-2, 1, std::numeric_limits<int64_t>::max()}) {
      model = PackingModel(op, &shape,
                           {MakeAttribute("bits", int64_t{3}), MakeAttribute("axis", axis)});
      EXPECT_THROW(InferPacking(model), InferenceError);
    }
    model = PackingModel(op, &shape, attributes, TensorProto::INT32);
    EXPECT_THROW(InferPacking(model), InferenceError);
    auto negative = PackingShape({-1});
    model = PackingModel(op, &negative, attributes);
    EXPECT_THROW(InferPacking(model), InferenceError);
  }
  for (int64_t count : {-1, 0, 3}) {
    auto model = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8, count);
    EXPECT_THROW(InferPacking(model), InferenceError);
  }
  for (bool constant : {false, true}) {
    auto model = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8,
                              constant ? std::optional<int64_t>{2} : std::nullopt);
    model.mutable_graph()
        ->mutable_input(1)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->add_dim()
        ->set_dim_value(1);
    EXPECT_THROW(InferPacking(model), InferenceError);
  }
  auto malformed = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8, 2);
  malformed.mutable_graph()->mutable_initializer(0)->clear_int64_data();
  EXPECT_THROW(InferPacking(malformed), InferenceError);
  malformed = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8, 2);
  malformed.mutable_graph()->mutable_initializer(0)->add_int64_data(2);
  EXPECT_THROW(InferPacking(malformed), InferenceError);
  malformed = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8, 2);
  malformed.mutable_graph()->mutable_initializer(0)->set_data_type(TensorProto::INT32);
  EXPECT_THROW(InferPacking(malformed), InferenceError);
  for (size_t size : {0, 7, 9, 16}) {
    malformed = PackingModel("Unpack", &shape, attributes, TensorProto::UINT8, 2);
    auto *count = malformed.mutable_graph()->mutable_initializer(0);
    count->clear_int64_data();
    count->set_raw_data(std::string(size, '\0'));
    EXPECT_THROW(InferPacking(malformed), InferenceError);
  }
}

TEST(onnx_pack_unpack, InfersNearInt64LimitWithoutOverflow) {
  const auto maximum = std::numeric_limits<int64_t>::max();
  auto shape = PackingShape({0, maximum});
  for (int storage : {TensorProto::UINT8, TensorProto::UINT32}) {
    const int64_t word_bits = storage == TensorProto::UINT32 ? 32 : 8;
    for (int64_t bits = 1; bits <= 8; ++bits) {
      auto model = PackingModel("Pack", &shape, {MakeAttribute("bits", bits)}, storage);
      InferPacking(model);
      const auto words =
          (maximum / word_bits) * bits + ((maximum % word_bits) * bits + word_bits - 1) / word_bits;
      EXPECT_EQ(PackingOutputShape(model).dim(1).dim_value(), words);
      const auto packed_shape = PackingOutputShape(model);
      auto unpacked =
          PackingModel("Unpack", &packed_shape, {MakeAttribute("bits", bits)}, storage, maximum);
      InferPacking(unpacked);
      EXPECT_EQ(PackingOutputShape(unpacked).dim(1).dim_value(), maximum);
    }
    for (const auto &radix_group : {std::pair<int64_t, int64_t>{3, 39}, {5, 27}}) {
      const auto [radix, group] = radix_group;
      const int64_t width = radix == 3 ? 62 : 63;
      const auto groups = maximum / group + (maximum % group != 0);
      const auto words =
          (groups / word_bits) * width + ((groups % word_bits) * width + word_bits - 1) / word_bits;
      auto model = PackingModel("Pack", &shape,
                                {MakeAttribute("radix", radix), MakeAttribute("group_size", group)},
                                storage);
      InferPacking(model);
      EXPECT_EQ(PackingOutputShape(model).dim(1).dim_value(), words);
    }
  }
  auto overflow =
      PackingModel("Pack", &shape,
                   {MakeAttribute("radix", int64_t{256}), MakeAttribute("group_size", int64_t{6})});
  EXPECT_THROW(InferPacking(overflow), InferenceError);
  overflow =
      PackingModel("Unpack", &shape,
                   {MakeAttribute("radix", int64_t{256}), MakeAttribute("group_size", int64_t{6})},
                   TensorProto::UINT8, maximum);
  EXPECT_THROW(InferPacking(overflow), InferenceError);
  shape.mutable_dim(1)->Clear();
  for (const auto *input_shape : {&shape, static_cast<TensorShapeProto *>(nullptr)}) {
    overflow = PackingModel(
        "Unpack", input_shape,
        {MakeAttribute("radix", int64_t{256}), MakeAttribute("group_size", int64_t{6})},
        TensorProto::UINT8, maximum);
    EXPECT_THROW(InferPacking(overflow), InferenceError);
  }
  auto wide =
      PackingModel("Unpack", &shape,
                   {MakeAttribute("radix", int64_t{256}), MakeAttribute("group_size", int64_t{6})},
                   TensorProto::UINT32, maximum);
  InferPacking(wide);
  EXPECT_EQ(PackingOutputShape(wide).dim(1).dim_value(), maximum);
}

TEST(onnx_pack_unpack, VerifiesFunctionsAndExpansionForEveryPackingMode) {
  auto shape = PackingShape({2, 17});
  for (const auto *op : {"Pack", "Unpack"}) {
    EXPECT_EQ(OpSchemaRegistry::Schema(op, 28, ""), nullptr);
    for (auto attributes : PackingAttributes()) {
      for (int storage : {TensorProto::UINT8, TensorProto::UINT32}) {
        for (int64_t axis : {-1, 0, -2, 1}) {
          auto model = PackingModel(op, &shape, attributes, storage);
          auto *graph = model.mutable_graph();
          auto node = graph->node(0);
          *node.add_attribute() = MakeAttribute("axis", axis);
          const auto *schema = OpSchemaRegistry::Schema(op, 29, "");
          ASSERT_NE(schema, nullptr);
          ASSERT_TRUE(schema->HasContextDependentFunction());
          schema->Verify(node);
          std::vector<TypeProto> types{graph->input(0).type()};
          if (std::string(op) == "Unpack") {
            types.push_back(graph->input(1).type());
          }
          FunctionBodyBuildContextImpl context(node, types);
          FunctionProto function;
          ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
          checker::CheckerContext checker_context;
          checker_context.set_ir_version(IR_VERSION);
          checker_context.set_is_main_graph(false);
          checker_context.set_opset_imports({{"", 29}});
          checker::LexicalScopeContext lexical_scope;
          EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));
          const std::vector<AttributeProto> function_attributes(node.attribute().begin(),
                                                                node.attribute().end());
          const auto inferred =
              shape_inference::InferFunctionOutputTypes(function, types, function_attributes);
          ASSERT_EQ(inferred.size(), 1);
          EXPECT_EQ(inferred[0].tensor_type().elem_type(),
                    std::string(op) == "Pack" ? storage : TensorProto::UINT8);
          auto *output = graph->mutable_output(0)->mutable_type()->mutable_tensor_type();
          output->set_elem_type(std::string(op) == "Pack" ? storage : TensorProto::UINT8);
          output->mutable_shape()->add_dim();
          output->mutable_shape()->add_dim();
          graph->clear_node();
          FunctionExpandHelper(node, function, *graph);
          EXPECT_NO_THROW(checker::check_model(model));
        }
      }
    }
  }
}

TEST(onnx_pack_unpack, FunctionBuilderRequiresStorageTypeAndRankForNondefaultAxis) {
  auto model = PackingModel("Unpack", nullptr, {MakeAttribute("bits", int64_t{3})});
  auto node = model.graph().node(0);
  const auto *schema = OpSchemaRegistry::Schema("Unpack", 29, "");
  ASSERT_NE(schema, nullptr);
  FunctionProto function;
  FunctionBodyBuildContextImpl unknown_type(node);
  EXPECT_FALSE(schema->BuildContextDependentFunction(unknown_type, function));
  *node.add_attribute() = MakeAttribute("axis", int64_t{0});
  FunctionBodyBuildContextImpl unknown_rank(node,
                                            std::vector<TypeProto>{model.graph().input(0).type()});
  EXPECT_FALSE(schema->BuildContextDependentFunction(unknown_rank, function));
  auto scalar = model.graph().input(0).type();
  scalar.mutable_tensor_type()->mutable_shape();
  FunctionBodyBuildContextImpl scalar_context(node, std::vector<TypeProto>{scalar});
  EXPECT_THROW(schema->BuildContextDependentFunction(scalar_context, function), InferenceError);
  node.mutable_attribute(1)->set_i(-1);
  FunctionBodyBuildContextImpl scalar_default_axis(node, std::vector<TypeProto>{scalar});
  EXPECT_THROW(schema->BuildContextDependentFunction(scalar_default_axis, function),
               InferenceError);
}

TEST(onnx_pack_unpack, ExpandsNestedCapturesAndPreservesEmptyOptionalInputs) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("Identity", 29, "");
  ASSERT_NE(schema, nullptr);
  FunctionProto function;
  FunctionBuilder(function).Add(R"ONNX(
    condition = Constant <value = bool {1}>()
    captured = Neg(input)
    output = If(condition) <
      then_branch = outer_then () => (outer_result) {
        outer_result = If(condition) <
          then_branch = inner_then () => (inner_result) {
            inner_result = Identity(captured)
          },
          else_branch = inner_else () => (inner_result) {
            inner_result = Identity(input)
          }
        >
      },
      else_branch = outer_else () => (captured) {
        captured = Identity(input)
      }
    >
  )ONNX");
  schema->BuildFunction(function);
  ModelProto model;
  ASSERT_TRUE(OnnxParser::Parse(model, R"ONNX(
    <ir_version: 14, opset_import: ["" : 29]>
    expanded (float[2] caller_input) => (float[2] caller_output) {}
  )ONNX")
                  .IsOK());
  NodeProto node;
  node.set_op_type("Identity");
  node.add_input("caller_input");
  node.add_output("caller_output");
  auto &graph = *model.mutable_graph();
  FunctionExpandHelper(node, function, graph);
  EXPECT_NO_THROW(checker::check_model(model));
  const auto &outer_if = graph.node(2);
  const auto &outer_then = outer_if.attribute(0).g();
  const auto &outer_else = outer_if.attribute(1).g();
  const auto &inner_then = outer_then.node(0).attribute(0).g();
  const auto &inner_else = outer_then.node(0).attribute(1).g();
  EXPECT_EQ(inner_then.node(0).input(0), graph.node(1).output(0));
  EXPECT_EQ(inner_else.node(0).input(0), "caller_input");
  EXPECT_NE(outer_else.node(0).output(0), graph.node(1).output(0));

  FunctionProto optional;
  FunctionBuilder(optional).Add("output = Clip(input, , optional_max)");
  schema->BuildFunction(optional);
  optional.add_input("optional_max");
  GraphProto optional_graph;
  FunctionExpandHelper(node, optional, optional_graph);
  ASSERT_EQ(optional_graph.node(0).input_size(), 3);
  EXPECT_EQ(optional_graph.node(0).input(0), "caller_input");
  EXPECT_TRUE(optional_graph.node(0).input(1).empty());
  EXPECT_TRUE(optional_graph.node(0).input(2).empty());
}

TEST(onnx_pack_unpack, ExpandsPassthroughLoopBody) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("Identity", 29, "");
  ASSERT_NE(schema, nullptr);
  NodeProto node;
  node.set_op_type("Identity");
  node.add_input("caller_input");
  node.add_output("caller_output");
  FunctionProto passthrough;
  FunctionBuilder(passthrough).Add(R"ONNX(
    trip = Constant <value = int64 {2}>()
    keep = Constant <value = bool {1}>()
    output = Loop(trip, keep, input) <
      body = passthrough_body (int64 iteration, bool cond, float[2] state) => (bool cond, float[2] state) {}
    >
  )ONNX");
  schema->BuildFunction(passthrough);
  ModelProto passthrough_model;
  ASSERT_TRUE(OnnxParser::Parse(passthrough_model, R"ONNX(
    <ir_version: 14, opset_import: ["" : 29]>
    expanded (float[2] caller_input) => (float[2] caller_output) {}
  )ONNX")
                  .IsOK());
  auto &passthrough_graph = *passthrough_model.mutable_graph();
  FunctionExpandHelper(node, passthrough, passthrough_graph);
  EXPECT_NO_THROW(checker::check_model(passthrough_model));
  const auto &body = passthrough_graph.node(2).attribute(0).g();
  ASSERT_EQ(body.input_size(), 3);
  ASSERT_EQ(body.output_size(), 2);
  EXPECT_EQ(body.output(0).name(), body.input(1).name());
  EXPECT_EQ(body.output(1).name(), body.input(2).name());
}
