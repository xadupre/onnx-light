// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/shapes/shapes/math/shape_math.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/inliner/inliner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

using core::runtime::DataType;
using core::runtime::DefaultOpset;
using core::runtime::KernelContext;
using core::runtime::Tensor;
using onnx_kernels::kernel::LogCumSumExp;

namespace {

NodeProto MakeNode(int64_t axis = 0) {
  NodeProto node;
  node.set_op_type("LogCumSumExp");
  node.add_input("X");
  node.add_output("Y");
  auto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::INT);
  attr->set_i(axis);
  return node;
}

void ExpectSameValues(const Tensor &actual, const Tensor &expected, double tolerance) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.data_type, expected.data_type);
  for (int64_t i = 0; i < actual.element_count(); ++i) {
    const double a =
        actual.data_type == DataType::DOUBLE ? actual.AsDouble()[i] : actual.AsFloat()[i];
    const double e =
        expected.data_type == DataType::DOUBLE ? expected.AsDouble()[i] : expected.AsFloat()[i];
    if (std::isnan(e)) {
      EXPECT_TRUE(std::isnan(a)) << i;
    } else if (std::isinf(e)) {
      EXPECT_EQ(a, e) << i;
    } else {
      EXPECT_NEAR(a, e, tolerance * std::max(std::abs(e), 1e-30)) << i;
    }
  }
}

} // namespace

TEST(LogCumSumExpKernel, AxesAndIndependentLanes) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const Tensor x = Tensor::FromDouble("", {2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  for (int64_t axis = 0; axis < 3; ++axis) {
    const Tensor positive = kernel(x, axis);
    ExpectSameValues(kernel(x, axis - 3), positive, 0);
    int64_t inner = 1;
    for (int64_t d = axis + 1; d < 3; ++d)
      inner *= x.shape[static_cast<size_t>(d)];
    for (int64_t i = 0; i < x.element_count(); ++i) {
      const int64_t position = (i / inner) % x.shape[static_cast<size_t>(axis)];
      long double sum = 0;
      for (int64_t k = 0; k <= position; ++k)
        sum += std::exp(static_cast<long double>(x.AsDouble()[i - k * inner]));
      EXPECT_NEAR(positive.AsDouble()[i], static_cast<double>(std::log(sum)), 1e-13);
    }
  }
}

TEST(LogCumSumExpKernel, DynamicRangeAndSmallCorrections) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const Tensor x = Tensor::FromDouble("", {3}, {1000, -1000, 1000});
  const Tensor y = kernel(x);
  EXPECT_DOUBLE_EQ(y.AsDouble()[0], 1000);
  EXPECT_DOUBLE_EQ(y.AsDouble()[1], 1000);
  EXPECT_DOUBLE_EQ(y.AsDouble()[2], 1000 + std::log(2.0));
  const Tensor small = kernel(Tensor::FromFloat("", {3}, {0, -20, -40}));
  EXPECT_GT(small.AsFloat()[1], 0);
  EXPECT_NEAR(small.AsFloat()[1], std::log1p(std::exp(-20.0)), 1e-15);
  const Tensor precise = kernel(Tensor::FromDouble("", {3}, {0, -40, -80}));
  EXPECT_GT(precise.AsDouble()[1], 0);
  EXPECT_NEAR(precise.AsDouble()[1], std::log1p(std::exp(-40.0)), 1e-32);
  const double large = std::numeric_limits<double>::max();
  const Tensor extreme = kernel(Tensor::FromDouble("", {2}, {-large, large}));
  EXPECT_EQ(extreme.AsDouble()[0], -large);
  EXPECT_EQ(extreme.AsDouble()[1], large);
}

TEST(LogCumSumExpKernel, PromotesHalfAccumulatorWithoutOutputRounding) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const std::vector<float> zeros(4096, 0);
  const Tensor f32 = kernel(Tensor::FromFloat("", {4096}, zeros));
  EXPECT_NEAR(f32.AsFloat()[4095], std::log(4096.0f), 1e-5f);
  for (const auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    Tensor x = dtype == DataType::FLOAT16 ? core::runtime::MakeFloat16Tensor("", {4096}, zeros)
                                          : core::runtime::MakeBfloat16Tensor("", {4096}, zeros);
    const Tensor y = kernel(x);
    EXPECT_EQ(y.data_type, dtype);
    const auto *bits = reinterpret_cast<const uint16_t *>(y.bytes());
    const auto encode = dtype == DataType::FLOAT16 ? core::runtime::FloatToFloat16Bits
                                                   : core::runtime::FloatToBfloat16Bits;
    for (int64_t i = 0; i < y.element_count(); ++i)
      EXPECT_EQ(bits[i], encode(f32.AsFloat()[i])) << i;
  }
}

TEST(LogCumSumExpKernel, SpecialValuesDominateInOrder) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Tensor x = Tensor::FromDouble(
      "", {3, 5}, {-inf, -inf, 1, inf, nan, inf, -inf, nan, inf, 2, nan, -inf, inf, nan, 0});
  const Tensor expected = Tensor::FromDouble(
      "", {3, 5}, {-inf, -inf, 1, inf, nan, inf, inf, nan, nan, nan, nan, nan, nan, nan, nan});
  ExpectSameValues(kernel(x, -1), expected, 0);
}

TEST(LogCumSumExpKernel, EmptyAndValidation) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const Tensor empty = Tensor::FromFloat("", {2, 0, 3}, {});
  for (int64_t axis = -3; axis < 3; ++axis) {
    const Tensor y = kernel(empty, axis);
    EXPECT_EQ(y.shape, empty.shape);
    EXPECT_EQ(y.size_bytes(), 0u);
  }
  EXPECT_THROW(kernel(empty, 3), std::invalid_argument);
  EXPECT_THROW(kernel(empty, -4), std::invalid_argument);
  EXPECT_THROW(kernel(empty, std::numeric_limits<int64_t>::min()), std::invalid_argument);
  EXPECT_THROW(kernel(empty, std::numeric_limits<int64_t>::max()), std::invalid_argument);
  EXPECT_THROW(kernel(Tensor::FromFloat("", {}, {1})), std::invalid_argument);
  EXPECT_THROW(kernel(Tensor::FromInt64("", {0}, {})), std::invalid_argument);
  Tensor malformed = Tensor::FromFloat("", {1}, {0});
  malformed.shape = {2};
  EXPECT_THROW(kernel(malformed), std::invalid_argument);
  malformed.shape = {-1, 0};
  EXPECT_THROW(kernel(malformed), std::invalid_argument);
  malformed.shape = {std::numeric_limits<int64_t>::max(), 2};
  EXPECT_THROW(kernel(malformed), std::invalid_argument);
  malformed.shape = {std::numeric_limits<int64_t>::max(), 0, std::numeric_limits<int64_t>::max()};
  malformed.data.resize(0);
  EXPECT_NO_THROW(kernel(malformed, 1));
}

TEST(LogCumSumExpKernel, PreallocatedAndInPlace) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  Tensor x = Tensor::FromFloat("", {3}, {0, 1, 2});
  const Tensor expected = kernel(x);
  Tensor output = Tensor::FromFloat("", {3}, {0, 0, 0});
  kernel(x, 0, output);
  ExpectSameValues(output, expected, 0);
  kernel(x, 0, x);
  ExpectSameValues(x, expected, 0);
  Tensor wrong_type = Tensor::FromDouble("", {3}, {0, 0, 0});
  EXPECT_THROW(kernel(x, 0, wrong_type), std::invalid_argument);
  Tensor wrong_shape = Tensor::FromFloat("", {1, 3}, {0, 0, 0});
  EXPECT_THROW(kernel(x, 0, wrong_shape), std::invalid_argument);
  output.data.resize(0);
  EXPECT_THROW(kernel(x, 0, output), std::invalid_argument);
}

TEST(LogCumSumExpKernel, UsesOnlyOutputAllocationForEveryDtype) {
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const std::vector<Tensor> inputs = {Tensor::FromFloat("", {3}, {0, 0, 0}),
                                      Tensor::FromDouble("", {3}, {0, 0, 0}),
                                      core::runtime::MakeFloat16Tensor("", {3}, {0, 0, 0}),
                                      core::runtime::MakeBfloat16Tensor("", {3}, {0, 0, 0})};
  for (const Tensor &input : inputs) {
    core::runtime::SimpleRawBufferAllocator allocator(1);
    core::runtime::RuntimeContext runtime(
        core::runtime::RuntimeContextOptions{.allocator = &allocator});
    const Tensor output = kernel(input, 0, &runtime);
    EXPECT_EQ(output.data_type, input.data_type);
    EXPECT_TRUE(output.has_allocation());
    EXPECT_EQ(allocator.allocated_count(), 1u);
    EXPECT_EQ(output.size_bytes(), input.size_bytes());
  }
}

TEST(LogCumSumExpShape, PropagatesSymbolicAndEmptyShapesAndValidatesAxis) {
  using core::symbolic::SymDim;
  using core::symbolic::SymShape;
  using core::symbolic::SymTensor;
  using core::symbolic::TensorType;
  for (const auto dtype :
       {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (const auto &shape :
         {SymShape{SymDim("N"), SymDim(3)}, SymShape{SymDim(int64_t{0}), SymDim(3)}}) {
      core::shapes::ShapesContext ctx;
      ctx.Set("X", SymTensor(nullptr, dtype, shape));
      for (int64_t axis = -2; axis < 2; ++axis) {
        const NodeProto node = MakeNode(axis);
        onnx_shapes::shapes::math::ComputeShapeLogCumSumExp(ctx, node, "X");
        EXPECT_EQ(ctx.Get("Y"), SymTensor(nullptr, dtype, shape));
      }
      for (const int64_t axis : {int64_t{-3}, int64_t{2}, std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::max()}) {
        const NodeProto node = MakeNode(axis);
        EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeLogCumSumExp(ctx, node, "X"),
                     std::invalid_argument);
      }
      NodeProto node = MakeNode();
      node.clear_attribute();
      EXPECT_NO_THROW(onnx_shapes::shapes::math::ComputeShapeLogCumSumExp(ctx, node, "X"));
      node.set_op_type("Log");
      EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeLogCumSumExp(ctx, node, "X"),
                   std::invalid_argument);
    }
  }
  core::shapes::ShapesContext ctx;
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, {}));
  const NodeProto node = MakeNode();
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeLogCumSumExp(ctx, node, "X"),
               std::invalid_argument);
}

TEST(LogCumSumExpShape, BackendCasesUseExtensionInference) {
  auto cases = core::backend_test::CollectTestCases("LogCumSumExp");
  ASSERT_EQ(cases.size(), 13u);
  for (auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ModelProto model = test_case.model();
    const Tensor &expected = test_case.data_sets()[0].outputs[0];
    model.mutable_graph()->mutable_output(0)->clear_type();
    ASSERT_NO_THROW(core::shapes::InferShapesModel(model));
    const auto &output = model.graph().output(0).type().tensor_type();
    EXPECT_EQ(output.elem_type(), expected.data_type);
    ASSERT_EQ(output.shape().dim_size(), expected.shape.size());
    for (size_t i = 0; i < expected.shape.size(); ++i)
      EXPECT_EQ(output.shape().dim(i).dim_value(), expected.shape[i]);
  }
}

TEST(LogCumSumExpKernel, InlinedFunctionMatchesNativeWithInt64ShapeReduction) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const auto *schema = OpSchemaRegistry::Schema("LogCumSumExp", 29, "");
  ASSERT_NE(schema, nullptr);
  const auto selected = inliner::FunctionIdSet::Create({{"local", "LogCumSumExp"}});
  const LogCumSumExp kernel{KernelContext{DefaultOpset(29)}};
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<Tensor> inputs = {
      Tensor::FromFloat("X", {3}, {0, -20, -40}),
      Tensor::FromDouble("X", {3}, {0, -40, -80}),
      Tensor::FromDouble("X", {2, 3}, {1000, 1000, -1000, -1000, -1000, 1000}),
      Tensor::FromDouble("X", {2, 5}, {-inf, -inf, 0, inf, nan, nan, -inf, inf, 0, 1}),
      Tensor::FromFloat("X", {0}, {}),
      Tensor::FromDouble("X", {2, 0, 3}, {})};
  for (const Tensor &input : inputs) {
    for (const bool unknown_rank : {false, true}) {
      for (int64_t axis = -static_cast<int64_t>(input.shape.size()); axis < 0; ++axis) {
        SCOPED_TRACE(input.data_type);
        SCOPED_TRACE(input.shape.size());
        SCOPED_TRACE(axis);
        SCOPED_TRACE(unknown_rank);
        NodeProto node = MakeNode(axis);
        node.set_domain("local");
        ModelProto model;
        model.set_ir_version(IR_VERSION);
        model.add_opset_import()->set_version(29);
        auto *local_opset = model.add_opset_import();
        local_opset->set_domain("local");
        local_opset->set_version(1);
        auto *graph = model.mutable_graph();
        graph->set_name("expanded_logcumsumexp");
        auto *input_info = graph->add_input();
        input_info->set_name("actual_input");
        auto *tensor_type = input_info->mutable_type()->mutable_tensor_type();
        tensor_type->set_elem_type(input.data_type);
        if (!unknown_rank) {
          for (const int64_t dim : input.shape)
            tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
        }
        utils::RepeatedProtoField<TypeProto> input_types;
        input_types.push_back(input_info->type());
        FunctionBodyBuildContextImpl context(node, input_types);
        FunctionProto function;
        ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
        function.set_domain("local");
        *model.add_functions() = function;
        auto *output = graph->add_output();
        output->set_name("actual_output");
        *output->mutable_type() = input_info->type();
        NodeProto *call = graph->add_node();
        *call = node;
        call->ref_input()[0] = "actual_input";
        call->ref_output()[0] = "actual_output";
        inliner::InlineSelectedLocalFunctions(model, *selected);
        ASSERT_GT(graph->node_size(), 1);
        if (!unknown_rank) {
          ASSERT_NO_THROW(checker::check_model(model, true));
        }
        core::runtime::RuntimeContext runtime;
        runtime.Set("actual_input", input);
        core::runtime::RuntimeSession session(model);
        session.Run(runtime);
        ExpectSameValues(runtime.Get("actual_output"), kernel(input, axis),
                         input.data_type == DataType::DOUBLE ? 1e-13 : 1e-6);
      }
    }

  } // namespace Test
}
} // namespace Test
