// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/schema.h"
#include "onnx_lib/inliner/inliner.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace {

using core::runtime::DataType;
using core::runtime::Shape;
using core::runtime::Tensor;
using core::symbolic::SymDim;
using core::symbolic::SymShape;
using core::symbolic::SymTensor;
using core::symbolic::TensorType;
using onnx_kernels::kernel::CausalConvWithState;

Tensor Samples(bool half, const Shape &shape, const std::vector<float> &values) {
  return half ? core::runtime::MakeFloat16Tensor("", shape, values)
              : Tensor::FromFloat("", shape, values);
}

void ExpectBytes(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes() != 0) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

CausalConvWithState MakeKernel() {
  return CausalConvWithState(core::runtime::KernelContext{core::backend_test::DefaultOpset(29)});
}

TEST(CausalConvWithStateDilation, ExactRawHistoryAndTapSpacing) {
  const auto kernel = MakeKernel();
  for (const bool half : {false, true}) {
    const Tensor x = Samples(half, {1, 2, 3}, {1, 2, 3, 10, 20, 30});
    const Tensor w = Samples(half, {2, 1, 3}, {1, 2, 4, 1, 2, 4});
    const Tensor past = Samples(half, {1, 2, 4}, {5, 6, 7, 8, 50, 60, 70, 80});
    const Tensor bias = Samples(half, {2}, {1, -1});
    CausalConvWithState::Attributes attrs;
    attrs.dilation = 2;
    const auto [output, present] = kernel(x, w, bias, past, attrs);
    ExpectBytes(output, Samples(half, {1, 2, 3}, {24, 31, 22, 229, 299, 209}));
    ExpectBytes(present, Samples(half, {1, 2, 4}, {8, 1, 2, 3, 80, 10, 20, 30}));
    const auto [zero_output, zero_present] = kernel(x, w, {}, {}, attrs);
    ExpectBytes(zero_output, Samples(half, {1, 2, 3}, {4, 8, 14, 40, 80, 140}));
    ExpectBytes(zero_present, Samples(half, {1, 2, 4}, {0, 1, 2, 3, 0, 10, 20, 30}));
  }
}

TEST(CausalConvWithStateDilation, ChunkEquivalenceIncludesEmptyAndShortChunks) {
  const auto kernel = MakeKernel();
  for (const bool half : {false, true}) {
    for (const int64_t dilation : {1, 2, 3}) {
      CausalConvWithState::Attributes attrs;
      attrs.dilation = dilation;
      const Tensor w = Samples(half, {1, 1, 4}, {1, 2, -1, 0.5f});
      const Tensor x = Samples(half, {1, 1, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
      const auto [full, full_state] = kernel(x, w, {}, {}, attrs);
      Tensor state;
      size_t offset = 0;
      std::vector<uint8_t> bytes;
      for (const int64_t length : {0, 1, 0, 2, 1, 4, 0}) {
        std::vector<float> values;
        for (int64_t j = 0; j < length; ++j)
          values.push_back(static_cast<float>(offset + j + 1));
        const Tensor chunk = Samples(half, {1, 1, length}, values);
        auto [out, next_state] = kernel(chunk, w, {}, state, attrs);
        if (out.size_bytes() != 0)
          bytes.insert(bytes.end(), out.bytes(), out.bytes() + out.size_bytes());
        state = std::move(next_state);
        offset += static_cast<size_t>(length);
      }
      ASSERT_EQ(bytes.size(), full.size_bytes());
      EXPECT_EQ(std::memcmp(bytes.data(), full.bytes(), bytes.size()), 0);
      ExpectBytes(state, full_state);
    }
  }
}

TEST(CausalConvWithStateDilation, EmptyChunkPreservesPastOrZeros) {
  const auto kernel = MakeKernel();
  for (const bool half : {false, true}) {
    const Tensor x = Samples(half, {1, 1, 0}, {});
    const Tensor w = Samples(half, {1, 1, 3}, {1, 2, 3});
    const Tensor past = Samples(half, {1, 1, 4}, {1, 2, 3, 4});
    CausalConvWithState::Attributes attrs;
    attrs.dilation = 2;
    const auto [output, state] = kernel(x, w, {}, past, attrs);
    ExpectBytes(output, x);
    ExpectBytes(state, past);
    const auto [zero_output, zero_state] = kernel(x, w, {}, {}, attrs);
    ExpectBytes(zero_output, x);
    ExpectBytes(zero_state, Samples(half, {1, 1, 4}, {0, 0, 0, 0}));
  }
}

TEST(CausalConvWithStateDilation, KernelOneAcceptsMaximumDilationButRejectsNonpositive) {
  const auto kernel = MakeKernel();
  const Tensor x = Tensor::FromFloat("", {1, 1, 2}, {2, 3});
  const Tensor w = Tensor::FromFloat("", {1, 1, 1}, {2});
  CausalConvWithState::Attributes attrs;
  attrs.dilation = std::numeric_limits<int64_t>::max();
  const auto [output, state] = kernel(x, w, {}, {}, attrs);
  ExpectBytes(output, Tensor::FromFloat("", {1, 1, 2}, {4, 6}));
  ExpectBytes(state, Tensor::FromFloat("", {1, 1, 0}, {}));
  for (const int64_t dilation : {0, -1}) {
    attrs.dilation = dilation;
    EXPECT_THROW(kernel(x, w, {}, {}, attrs), std::invalid_argument);
    Tensor out;
    Tensor present;
    EXPECT_THROW(kernel(x, w, nullptr, nullptr, attrs, out, present), std::invalid_argument);
  }
}

TEST(CausalConvWithStateDilation, RejectsOverflowBeforeAllocationOrAccess) {
  const auto kernel = MakeKernel();
  const Tensor x = Tensor::FromFloat("", {1, 1, 1}, {1});
  CausalConvWithState::Attributes attrs;
  Tensor w = Tensor::FromFloat("", {1, 1, 2}, {1, 1});
  attrs.dilation = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(kernel(x, w, {}, {}, attrs), std::invalid_argument);
  attrs.dilation = std::numeric_limits<int64_t>::max() / 2 + 1;
  w.shape[2] = 3;
  EXPECT_THROW(kernel(x, w, {}, {}, attrs), std::invalid_argument);
  attrs.dilation = 2;
  Tensor huge = x;
  huge.shape[2] = std::numeric_limits<int64_t>::max() - 1;
  w.shape[2] = 2;
  EXPECT_THROW(kernel(huge, w, {}, {}, attrs), std::invalid_argument);
  Tensor out;
  Tensor present;
  EXPECT_THROW(kernel(huge, w, nullptr, nullptr, attrs, out, present), std::invalid_argument);
}

TEST(CausalConvWithStateDilation, AcceptsInt64EffectiveKernelBoundaryForEmptyBatch) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const auto selected = inliner::FunctionIdSet::Create({{"", "CausalConvWithState"}});
  const Tensor w = Tensor::FromFloat("", {1, 1, 2}, {1, 1});
  CausalConvWithState::Attributes attrs;
  attrs.dilation = std::numeric_limits<int64_t>::max() - 1;
  for (const int64_t length : {0, 1}) {
    for (const bool with_past : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "length=" << length << ", with_past=" << with_past);
      const Tensor x = Tensor::FromFloat("X", {0, 1, length}, {});
      const Tensor past = Tensor::FromFloat("P", {0, 1, attrs.dilation}, {});
      const auto [output, state] = MakeKernel()(x, w, {}, with_past ? past : Tensor{}, attrs);
      ExpectBytes(output, x);
      EXPECT_EQ(state.shape, (Shape{0, 1, attrs.dilation}));
      EXPECT_EQ(state.size_bytes(), 0u);

      ModelProto model;
      model.set_ir_version(IR_VERSION);
      model.add_opset("", 29);
      auto &graph = *model.mutable_graph();
      graph.set_name("boundary_causal_conv");
      const auto add_value = [](ValueInfoProto *value, const char *name, const Tensor &tensor) {
        value->set_name(name);
        auto *type = value->mutable_type()->mutable_tensor_type();
        type->set_elem_type(tensor.data_type);
        for (const auto dim : tensor.shape)
          type->mutable_shape()->add_dim()->set_dim_value(dim);
      };
      add_value(graph.add_input(), "X", x);
      add_value(graph.add_input(), "W", w);
      if (with_past)
        add_value(graph.add_input(), "P", past);
      add_value(graph.add_output(), "Y", output);
      add_value(graph.add_output(), "S", state);
      auto *node = graph.add_node();
      node->set_op_type("CausalConvWithState");
      node->add_input("X");
      node->add_input("W");
      if (with_past) {
        node->add_input("");
        node->add_input("P");
      }
      node->add_output("Y");
      node->add_output("S");
      AddAttribute<int64_t>(*node, "dilation", attrs.dilation);
      std::vector<Tensor> inputs{x, w};
      inputs[1].name = "W";
      if (with_past)
        inputs.push_back(past);
      inliner::InlineSelectedFunctions(model, *selected, OpSchemaRegistry::Instance());
      ASSERT_GT(graph.node_size(), 1);
      ASSERT_NO_THROW(checker::check_model(model, true));
      const auto expanded = core::runtime::RunModel(model, inputs);
      ASSERT_EQ(expanded.size(), 2u);
      ExpectBytes(expanded[0], output);
      ExpectBytes(expanded[1], state);
    }
  }
}

TEST(CausalConvWithStateDilation, RejectsMismatchedStateAndTruncatedBias) {
  const auto kernel = MakeKernel();
  for (const bool half : {false, true}) {
    const Tensor x = Samples(half, {1, 1, 1}, {1});
    const Tensor w = Samples(half, {1, 1, 3}, {1, 1, 1});
    const Tensor past = Samples(half, {1, 1, 2}, {1, 2});
    CausalConvWithState::Attributes attrs;
    attrs.dilation = 2;
    EXPECT_THROW(kernel(x, w, {}, past, attrs), std::invalid_argument);
    Tensor bias;
    bias.data_type = x.data_type;
    bias.shape = {1};
    EXPECT_THROW(kernel(x, w, bias, {}, attrs), std::invalid_argument);
  }
}

TEST(CausalConvWithStateDilation, RuntimeAndExpandedFunctionMatchExactResults) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const auto selected = inliner::FunctionIdSet::Create({{"", "CausalConvWithState"}});
  for (const bool half : {false, true}) {
    for (const int64_t length : {0, 1, 7}) {
      for (const bool with_past : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "half=" << half << ", length=" << length << ", with_past=" << with_past);
        const Tensor x =
            Samples(half, {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 2.0f));
        const Tensor w = Samples(half, {1, 1, 3}, {1, 2, 4});
        const Tensor past = Samples(half, {1, 1, 4}, {1, 2, 3, 4});
        CausalConvWithState::Attributes attrs;
        attrs.dilation = 2;
        const auto [expected, expected_state] =
            MakeKernel()(x, w, {}, with_past ? past : Tensor{}, attrs);
        ModelProto model;
        model.set_ir_version(IR_VERSION);
        model.add_opset("", 29);
        auto &graph = *model.mutable_graph();
        graph.set_name("dilated_causal_conv");
        const auto input = [&](const char *name, const Tensor &tensor) {
          auto *value = graph.add_input();
          value->set_name(name);
          auto *type = value->mutable_type()->mutable_tensor_type();
          type->set_elem_type(tensor.data_type);
          for (const int64_t dim : tensor.shape)
            type->mutable_shape()->add_dim()->set_dim_value(dim);
        };
        input("X", x);
        graph.mutable_input(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
            ->mutable_dim(2)
            ->clear_dim_value();
        input("W", w);
        if (with_past)
          input("P", past);
        for (const auto &[name, tensor] :
             std::vector<std::pair<std::string, Tensor>>{{"Y", expected}, {"S", expected_state}}) {
          auto *value = graph.add_output();
          value->set_name(name);
          auto *type = value->mutable_type()->mutable_tensor_type();
          type->set_elem_type(tensor.data_type);
          for (const int64_t dim : tensor.shape)
            type->mutable_shape()->add_dim()->set_dim_value(dim);
        }
        auto *node = graph.add_node();
        node->set_op_type("CausalConvWithState");
        node->add_input("X");
        node->add_input("W");
        if (with_past) {
          node->add_input("");
          node->add_input("P");
        }
        node->add_output("Y");
        node->add_output("S");
        AddAttribute<int64_t>(*node, "dilation", 2);
        std::vector<Tensor> inputs{x, w};
        inputs[0].name = "X";
        inputs[1].name = "W";
        if (with_past) {
          inputs.push_back(past);
          inputs.back().name = "P";
        }
        const auto direct = core::runtime::RunModel(model, inputs);
        ASSERT_EQ(direct.size(), 2u);
        ExpectBytes(direct[0], expected);
        ExpectBytes(direct[1], expected_state);
        inliner::InlineSelectedFunctions(model, *selected, OpSchemaRegistry::Instance());
        ASSERT_GT(graph.node_size(), 1);
        ASSERT_NO_THROW(checker::check_model(model, true));
        const auto expanded = core::runtime::RunModel(model, inputs);
        ASSERT_EQ(expanded.size(), 2u);
        ExpectBytes(expanded[0], expected);
        ExpectBytes(expanded[1], expected_state);
      }
    }
  }
}

NodeProto ShapeNode(int64_t dilation) {
  NodeProto node;
  node.set_op_type("CausalConvWithState");
  node.add_input("X");
  node.add_input("W");
  node.add_output("Y");
  node.add_output("S");
  AddAttribute<int64_t>(node, "dilation", dilation);
  return node;
}

void InferCausalShape(core::shapes::ShapesContext &ctx, const NodeProto &node) {
  onnx_shapes::shapes::nn::ComputeShapeCausalConvWithState(ctx, node, "X", "W");
}

TEST(CausalConvWithStateDilationShape, ConcreteSymbolicAndCompoundKernelDimensions) {
  for (const int64_t dilation : {1, 3}) {
    core::shapes::ShapesContext ctx;
    ctx.Set("X", SymTensor(nullptr, TensorType::kFloat,
                           SymShape{SymDim("B"), SymDim(2), SymDim(int64_t{0})}));
    ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(2), SymDim(1), SymDim(4)}));
    InferCausalShape(ctx, ShapeNode(dilation));
    EXPECT_EQ(ctx.Get("S").Shape()[2].AsInt(), 3 * dilation);
    EXPECT_EQ(ctx.Get("Y").Shape()[2].AsInt(), 0);
    EXPECT_EQ(ctx.Get("S").Shape()[0].AsExpr(), "B");
    ctx.Set("W",
            SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(2), SymDim(1), SymDim("K+2")}));
    InferCausalShape(ctx, ShapeNode(dilation));
    EXPECT_EQ(ctx.Get("S").Shape()[2].AsExpr(), dilation == 1 ? "((K+2)-1)" : "((K+2)-1)*3");
    ctx.Set("W",
            SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(2), SymDim(1), SymDim("")}));
    InferCausalShape(ctx, ShapeNode(dilation));
    EXPECT_EQ(ctx.Get("S").Shape()[2].AsExpr(), "");
  }
}

TEST(CausalConvWithStateDilationShape, RejectsInvalidAndOverflowingDimensions) {
  for (const auto &[k, dilation, length] : std::vector<std::tuple<int64_t, int64_t, int64_t>>{
           {1, 0, 1},
           {1, -1, 1},
           {0, 1, 1},
           {2, std::numeric_limits<int64_t>::max(), 0},
           {3, std::numeric_limits<int64_t>::max() / 2 + 1, 0},
           {2, 2, std::numeric_limits<int64_t>::max() - 1}}) {
    core::shapes::ShapesContext ctx;
    ctx.Set("X",
            SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(length)}));
    ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(k)}));
    EXPECT_THROW(InferCausalShape(ctx, ShapeNode(dilation)), std::invalid_argument);
    EXPECT_FALSE(ctx.Has("Y"));
  }
}

TEST(CausalConvWithStateDilationShape, ValidatesKnownPastWithUnknownKernel) {
  for (const int64_t history : std::vector<int64_t>{-1, 3, std::numeric_limits<int64_t>::max()}) {
    core::shapes::ShapesContext ctx;
    ctx.Set("X", SymTensor(nullptr, TensorType::kFloat,
                           SymShape{SymDim(1), SymDim(1), SymDim(int64_t{0})}));
    ctx.Set("W",
            SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim("K")}));
    ctx.Set("P", SymTensor(nullptr, TensorType::kFloat,
                           SymShape{SymDim(1), SymDim(1), SymDim(history)}));
    NodeProto node = ShapeNode(2);
    node.add_input("");
    node.add_input("P");
    EXPECT_THROW(InferCausalShape(ctx, node), std::invalid_argument);
  }
}

TEST(CausalConvWithStateDilationShape, KernelOneAndUnknownKernelWithKnownHistory) {
  core::shapes::ShapesContext ctx;
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(2)}));
  ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(1)}));
  InferCausalShape(ctx, ShapeNode(std::numeric_limits<int64_t>::max()));
  EXPECT_EQ(ctx.Get("S").Shape()[2].AsInt(), 0);
  ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim("K")}));
  ctx.Set("P", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(6)}));
  NodeProto node = ShapeNode(3);
  node.add_input("");
  node.add_input("P");
  InferCausalShape(ctx, node);
  EXPECT_EQ(ctx.Get("S").Shape()[2].AsInt(), 6);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE
