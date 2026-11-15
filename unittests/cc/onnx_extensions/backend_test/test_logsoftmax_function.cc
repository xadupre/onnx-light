// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/operator_sets.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;

namespace {

Tensor RunLogSoftmaxFunction(const Tensor &input, int64_t axis) {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("logsoftmax_function");
  auto *value = graph->add_input();
  value->set_name("x");
  auto *type = value->mutable_type()->mutable_tensor_type();
  type->set_elem_type(input.data_type);
  for (const int64_t dim : input.shape) {
    type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  auto *output = graph->add_output();
  output->set_name("y");
  *output->mutable_type() = value->type();
  NodeProto node;
  node.set_op_type("LogSoftmax");
  node.add_input("x");
  node.add_output("y");
  auto *attribute = node.add_attribute();
  attribute->set_name("axis");
  attribute->set_type(AttributeProto::INT);
  attribute->set_i(axis);
  FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{value->type()});
  FunctionProto function;
  const auto *schema = OpSchemaRegistry::Schema("LogSoftmax", 29, ONNX_DOMAIN);
  if (!schema->BuildContextDependentFunction(context, function, 29)) {
    throw std::runtime_error("Cannot build LogSoftmax function.");
  }
  FunctionExpandHelper(node, function, *graph);
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
  checker::check_model(model, true);
  RuntimeContext rt;
  rt.Set("x", input);
  RuntimeSession session(model);
  session.Run(rt);
  return rt.Get("y");
}

Tensor MakeInput(int32_t dtype, const Shape &shape, const std::vector<double> &values) {
  if (dtype == DataType::DOUBLE) {
    return Tensor::FromDouble("x", shape, values);
  }
  const std::vector<float> floats(values.begin(), values.end());
  if (dtype == DataType::FLOAT16) {
    return MakeFloat16Tensor("x", shape, floats);
  }
  if (dtype == DataType::BFLOAT16) {
    return MakeBfloat16Tensor("x", shape, floats);
  }
  return Tensor::FromFloat("x", shape, floats);
}

void ExpectClose(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  const Tensor actual_float = PromoteToFloat32(actual);
  const Tensor expected_float = PromoteToFloat32(expected);
  for (int64_t i = 0; i < actual.element_count(); ++i) {
    const double a =
        actual.data_type == DataType::DOUBLE ? actual.AsDouble()[i] : actual_float.AsFloat()[i];
    const double e = expected.data_type == DataType::DOUBLE ? expected.AsDouble()[i]
                                                            : expected_float.AsFloat()[i];
    if (std::isnan(e)) {
      EXPECT_TRUE(std::isnan(a));
    } else if (std::isinf(e) || e == 0) {
      EXPECT_EQ(a, e);
    } else {
      EXPECT_NEAR(a, e, std::abs(e) * (actual.data_type == DataType::BFLOAT16 ? 0.01 : 0.001));
    }
  }
}

} // namespace

TEST(BackendLogSoftmaxFunction, PreservesSmallLogProbabilitiesAcrossAxesAndTypes) {
  for (const int32_t dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const double margin = dtype == DataType::DOUBLE ? 40 : dtype == DataType::FLOAT ? 18 : 10;
    const double normalizer = std::log1p(2 * std::exp(-margin));
    const std::vector<double> values = {0,       -margin, -margin, -margin, 0,
                                        -margin, -margin, -margin, 0};
    std::vector<double> expected = values;
    for (auto &value : expected) {
      value -= normalizer;
    }
    for (const int axis : {-1, 0, 1}) {
      SCOPED_TRACE("dtype=" + std::to_string(dtype) + " axis=" + std::to_string(axis));
      const auto input = MakeInput(dtype, {3, 3}, values);
      const auto actual = RunLogSoftmaxFunction(input, axis);
      ExpectClose(actual, MakeInput(dtype, {3, 3}, expected));
    }
  }
}

TEST(BackendLogSoftmaxFunction, HandlesTiesMasksSingletonsAndEmptyInputs) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (const int32_t dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const double normalizer = std::log(2 + std::exp(-2));
    const auto input = MakeInput(dtype, {3, 3}, {0, -2, 0, 4, 4, 4, 0, -infinity, -infinity});
    ExpectClose(RunLogSoftmaxFunction(input, -1),
                MakeInput(dtype, {3, 3},
                          {-normalizer, -2 - normalizer, -normalizer, -std::log(3), -std::log(3),
                           -std::log(3), 0, -infinity, -infinity}));
    for (const Shape &shape : {Shape{2, 1, 3}, Shape{0, 3}, Shape{2, 0, 3}}) {
      const auto zeros =
          MakeInput(dtype, shape, std::vector<double>(shape[0] == 2 && shape[1] == 1 ? 6 : 0));
      ExpectClose(RunLogSoftmaxFunction(zeros, 1), zeros);
    }
    const auto nonfinite =
        MakeInput(dtype, {3, 2},
                  {infinity, 0, -infinity, -infinity, std::numeric_limits<double>::quiet_NaN(), 0});
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ExpectClose(RunLogSoftmaxFunction(nonfinite, -1),
                MakeInput(dtype, {3, 2}, std::vector<double>(6, nan)));
  }
}

TEST(BackendLogSoftmaxFunction, PromotesHalfReductionsAndCountsTiedMaximaExactly) {
  for (const int32_t dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const int64_t classes : {264, 2052, 70000}) {
      const auto input = MakeInput(dtype, {1, classes}, std::vector<double>(classes));
      ExpectClose(RunLogSoftmaxFunction(input, -1),
                  MakeInput(dtype, {1, classes}, std::vector<double>(classes, -std::log(classes))));
    }
    std::vector<double> values(101, -18);
    values[0] = 0;
    std::vector<double> expected = values;
    for (auto &value : expected) {
      value -= std::log1p(100 * std::exp(-18));
    }
    ExpectClose(RunLogSoftmaxFunction(MakeInput(dtype, {1, 101}, values), -1),
                MakeInput(dtype, {1, 101}, expected));
  }
}
