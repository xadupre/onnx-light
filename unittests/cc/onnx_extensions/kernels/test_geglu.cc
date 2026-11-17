// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_lib/defs/function.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;
using core::backend_test::DefaultOpset;
using onnx_kernels::kernel::GeGLU;

namespace Test {
using core::runtime::DataType;
using core::runtime::Tensor;
namespace {
std::string TensorBytes(const Tensor &tensor) {
  return tensor.size_bytes() == 0
             ? std::string{}
             : std::string(reinterpret_cast<const char *>(tensor.bytes()), tensor.size_bytes());
}

Tensor TypedTensor(DataType dtype, const Shape &shape, const std::vector<float> &values) {
  if (dtype == DataType::FLOAT16)
    return MakeFloat16Tensor("", shape, values);
  if (dtype == DataType::BFLOAT16)
    return MakeBfloat16Tensor("", shape, values);
  if (dtype == DataType::DOUBLE)
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  return Tensor::FromFloat("", shape, values);
}
} // namespace

TEST(KernelGeGLU, IndependentNegativeGateReferenceAndModes) {
  const GeGLU kernel{KernelContext{DefaultOpset(29)}};
  const std::vector<float> av{-4, -3, -1, -0.5f, 0, 0.5f, 1, 4};
  const std::vector<float> bv{2, -1, 0.5f, 3, -2, -3, 0.5f, 1};
  const std::vector<double> exact{
      -0.00025336993466495983, 0.00404969409489031, -0.07932762696572854, -0.4628063080889803, -0.0,
      -1.0371936919110196,     0.42067237303427146, 3.9998733150326675};
  const std::vector<double> tanh{-0.00014049189638454251,
                                 0.0036373920817729943,
                                 -0.07940400469586162,
                                 -0.4628579705245682,
                                 -0.0,
                                 -1.0371420294754319,
                                 0.4205959953041384,
                                 3.9999297540518075};
  for (DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const std::string mode : {"none", "tanh"}) {
      SCOPED_TRACE(static_cast<int32_t>(dtype));
      SCOPED_TRACE(mode);
      Tensor a = TypedTensor(dtype, {2, 4}, av);
      Tensor b = TypedTensor(dtype, {2, 4}, bv);
      const Tensor y = kernel(a, b, mode);
      EXPECT_EQ(y.shape, a.shape);
      EXPECT_EQ(y.data_type, a.data_type);
      const auto &expected = mode == "none" ? exact : tanh;
      const Tensor promoted =
          dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16 ? PromoteToFloat32(y) : y;
      const double tolerance = dtype == DataType::DOUBLE    ? 2e-15
                               : dtype == DataType::FLOAT   ? 1e-6
                               : dtype == DataType::FLOAT16 ? 0.002
                                                            : 0.009;
      for (size_t i = 0; i < av.size(); ++i) {
        const double actual =
            dtype == DataType::DOUBLE ? promoted.AsDouble()[i] : promoted.AsFloat()[i];
        EXPECT_NEAR(actual, expected[i],
                    tolerance * std::abs(expected[i]) + (dtype == DataType::DOUBLE ? 1e-15 : 1e-7));
      }
      kernel(a, b, mode, a);
      EXPECT_EQ(TensorBytes(a), TensorBytes(y));
      kernel(TypedTensor(dtype, {2, 4}, av), b, mode, b);
      EXPECT_EQ(TensorBytes(b), TensorBytes(y));
    }
  }
}

TEST(KernelGeGLU, HalfComputesActivationAndProductBeforeSingleRounding) {
  const GeGLU kernel{KernelContext{DefaultOpset(29)}};
  const onnx_kernels::kernel::Gelu gelu{KernelContext{DefaultOpset(29)}};
  const std::vector<float> values{-4, -3.25f, -1.125f, -0.3f, 0.3f, 1.125f, 3.25f, 4};
  for (DataType dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto decode = dtype == DataType::FLOAT16 ? Float16BitsToFloat : Bfloat16BitsToFloat;
    const auto encode = dtype == DataType::FLOAT16 ? FloatToFloat16Bits : FloatToBfloat16Bits;
    bool differs_from_two_roundings = false;
    for (const std::string mode : {"none", "tanh"}) {
      const Tensor a = TypedTensor(dtype, {8}, values);
      const Tensor b = TypedTensor(dtype, {8}, std::vector<float>(8, 1.375f));
      const Tensor fa = PromoteToFloat32(a);
      const Tensor fb = PromoteToFloat32(b);
      const Tensor activation = gelu(fa, mode);
      const Tensor y = kernel(a, b, mode);
      for (int i = 0; i < 8; ++i) {
        const uint16_t expected = encode(activation.AsFloat()[i] * fb.AsFloat()[i]);
        EXPECT_EQ(reinterpret_cast<const uint16_t *>(y.bytes())[i], expected);
        const uint16_t twice = encode(decode(encode(activation.AsFloat()[i])) * fb.AsFloat()[i]);
        differs_from_two_roundings |= expected != twice;
      }
    }
    EXPECT_TRUE(differs_from_two_roundings);
  }
}

TEST(KernelGeGLU, ScalarsEmptyAndDefaultAttribute) {
  const GeGLU kernel{KernelContext{DefaultOpset(29)}};
  for (DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const Shape &shape : {Shape{}, Shape{2, 0, 3}}) {
      const std::vector<float> values =
          shape.empty() ? std::vector<float>{1} : std::vector<float>{};
      const Tensor a = TypedTensor(dtype, shape, values);
      const Tensor y = kernel(a, a);
      const Tensor explicit_none = kernel(a, a, "none");
      EXPECT_EQ(y.shape, shape);
      EXPECT_EQ(y.data_type, a.data_type);
      EXPECT_EQ(TensorBytes(y), TensorBytes(explicit_none));
      for (const std::string mode : {"none", "tanh"}) {
        Tensor out = TypedTensor(dtype, shape, values);
        EXPECT_NO_THROW(kernel(a, a, mode, out));
      }
    }
  }
  const Shape huge_empty{0, std::numeric_limits<int64_t>::max(), 2};
  const Tensor empty("", DataType::FLOAT, huge_empty, {});
  const Tensor y = kernel(empty, empty);
  EXPECT_EQ(y.shape, huge_empty);
  EXPECT_EQ(y.size_bytes(), 0u);
}

TEST(KernelGeGLU, SignedZerosSmallValuesAndSpecialValuesMatchGelu) {
  const GeGLU kernel{KernelContext{DefaultOpset(29)}};
  const float infinity = std::numeric_limits<float>::infinity();
  for (const std::string mode : {"none", "tanh"}) {
    const Tensor a = Tensor::FromFloat("", {9},
                                       {-0.0f, 0.0f, -1e-20f, 1e-20f, -100, 100, -infinity,
                                        infinity, std::numeric_limits<float>::quiet_NaN()});
    const Tensor b = Tensor::FromFloat("", {9}, std::vector<float>(9, 2));
    const Tensor y = kernel(a, b, mode);
    EXPECT_EQ(y.AsFloat()[0], 0);
    EXPECT_TRUE(std::signbit(y.AsFloat()[0]));
    EXPECT_FALSE(std::signbit(y.AsFloat()[1]));
    EXPECT_FLOAT_EQ(y.AsFloat()[2], -1e-20f);
    EXPECT_FLOAT_EQ(y.AsFloat()[3], 1e-20f);
    EXPECT_EQ(y.AsFloat()[4], 0);
    EXPECT_FLOAT_EQ(y.AsFloat()[5], 200);
    EXPECT_TRUE(std::isnan(y.AsFloat()[6]));
    EXPECT_EQ(y.AsFloat()[7], infinity);
    EXPECT_TRUE(std::isnan(y.AsFloat()[8]));
  }
}

TEST(KernelGeGLU, RejectsInvalidInputsBeforeAllocationAndInvalidOutputs) {
  const GeGLU kernel{KernelContext{DefaultOpset(29)}};
  RuntimeContext rt{KernelContext{DefaultOpset(29)}};
  const Tensor a = Tensor::FromFloat("", {2}, {1, 2});
  const Tensor broadcast = Tensor::FromFloat("", {1}, {1});
  const Tensor different_rank = Tensor::FromFloat("", {1, 2}, {1, 2});
  const Tensor different_type = Tensor::FromDouble("", {2}, {1, 2});
  const Tensor malformed("", DataType::FLOAT, {2}, std::vector<uint8_t>(sizeof(float)));
  const Tensor huge("", DataType::FLOAT, {std::numeric_limits<int64_t>::max(), 2}, {});
  const Tensor negative("", DataType::FLOAT, {-1}, {});
  const Tensor integer = Tensor::FromInt32("", {0}, {});
  for (const Tensor *bad : {&broadcast, &different_rank, &different_type, &malformed}) {
    EXPECT_THROW(kernel(a, *bad), std::invalid_argument);
    EXPECT_THROW(kernel(a, *bad, "none", &rt), std::invalid_argument);
    Tensor out = Tensor::FromFloat("", {2}, {0, 0});
    EXPECT_THROW(kernel(a, *bad, "none", out), std::invalid_argument);
  }
  for (const Tensor *bad : {&huge, &negative, &integer, &malformed}) {
    EXPECT_THROW(kernel(*bad, *bad), std::invalid_argument);
    EXPECT_THROW(kernel(*bad, *bad, "none", &rt), std::invalid_argument);
  }
  const Tensor empty = Tensor::FromFloat("", {0}, {});
  for (const std::string attr : {"", "TANH", "sigmoid"}) {
    EXPECT_THROW(kernel(a, a, attr), std::invalid_argument);
    EXPECT_THROW(kernel(empty, empty, attr, &rt), std::invalid_argument);
  }
  Tensor wrong_type = Tensor::FromDouble("", {2}, {0, 0});
  Tensor wrong_shape = Tensor::FromFloat("", {1, 2}, {0, 0});
  Tensor wrong_size("", DataType::FLOAT, {2}, std::vector<uint8_t>(sizeof(float)));
  for (Tensor *out : {&wrong_type, &wrong_shape, &wrong_size}) {
    EXPECT_THROW(kernel(a, a, "none", *out), std::invalid_argument);
  }
}

TEST(KernelGeGLU, RuntimeDispatchAttributesAndTuning) {
  onnx_kernels::RegisterKernelFunctions();
  GeGLU kernel{KernelContext{DefaultOpset(29)}};
  for (DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(dtype));
    EXPECT_EQ(key.kernel, "GeGLU");
    ASSERT_NE(GetKernelTuningRegistry().FindSchema(key), nullptr);
    kernel.Configure(KernelTuningParameters{key, {{"parallel.minimum_elements", int64_t{1}}}});
    EXPECT_EQ(kernel.tuning().parallel_minimum_elements, 1);
  }
  for (const std::string mode : {"", "none", "tanh", "invalid"}) {
    NodeProto node = MakeNode("GeGLU", {"A", "B"}, {"Y"});
    if (!mode.empty()) {
      auto *attr = node.add_attribute();
      attr->set_name("approximate");
      attr->set_type(AttributeProto::STRING);
      attr->set_s(mode);
    }
    RuntimeContext rt{KernelContext{DefaultOpset(29)}};
    rt.Set("A", Tensor::FromFloat("", {}, {-4}));
    rt.Set("B", Tensor::FromFloat("", {}, {2}));
    if (mode == "invalid") {
      EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
      EXPECT_FALSE(rt.Has("Y"));
    } else {
      RunNode(node, rt);
      const Tensor &y = rt.Get("Y");
      EXPECT_LT(y.AsFloat()[0], 0);
      EXPECT_NEAR(y.AsFloat()[0],
                  mode == "tanh" ? -0.00014049189638454251 : -0.00025336993466495983, 1e-7);
    }
  }
  NodeProto wrong_type = MakeNode("GeGLU", {"A", "B"}, {"Y"});
  auto *attr = wrong_type.add_attribute();
  attr->set_name("approximate");
  attr->set_type(AttributeProto::INT);
  attr->set_i(0);
  RuntimeContext rt{KernelContext{DefaultOpset(29)}};
  rt.Set("A", Tensor::FromFloat("", {0}, {}));
  rt.Set("B", Tensor::FromFloat("", {0}, {}));
  EXPECT_THROW(RunNode(wrong_type, rt), std::invalid_argument);
  EXPECT_FALSE(rt.Has("Y"));
}

TEST(KernelGeGLU, FunctionExpansionExecutesWithSingleLowPrecisionRounding) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const auto *schema = OpSchemaRegistry::Schema("GeGLU", 29, "");
  ASSERT_NE(schema, nullptr);
  for (const DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const std::string mode : {"none", "tanh"}) {
      SCOPED_TRACE(static_cast<int32_t>(dtype));
      SCOPED_TRACE(mode);
      ModelProto model;
      model.set_ir_version(IR_VERSION);
      model.add_opset_import()->set_version(29);
      GraphProto *graph = model.mutable_graph();
      graph->set_name("geglu_expanded");
      for (const auto *name : {"A", "B"}) {
        auto *input = graph->add_input();
        input->set_name(name);
        auto *type = input->mutable_type()->mutable_tensor_type();
        type->set_elem_type(static_cast<int32_t>(dtype));
        type->mutable_shape()->add_dim()->set_dim_value(4);
      }
      auto *output = graph->add_output();
      output->set_name("Y");
      output->mutable_type()->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(dtype));
      NodeProto node = MakeNode("GeGLU", {"A", "B"}, {"Y"});
      AddAttribute<std::string>(node, "approximate", mode);
      const std::vector<TypeProto> types{graph->input(0).type(), graph->input(1).type()};
      FunctionBodyBuildContextImpl context(node, types);
      FunctionProto function;
      ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, 29));
      FunctionExpandHelper(node, function, *graph);
      Tensor a = TypedTensor(dtype, {4}, {-4, -1.125f, 0.3f, 3.25f});
      Tensor b = TypedTensor(dtype, {4}, {1.375f, -2, 1.375f, 0.5f});
      a.name = "A";
      b.name = "B";
      const GeGLU kernel{KernelContext{DefaultOpset(29)}};
      const Tensor native = kernel(a, b, mode);
      const auto expanded = RunModel(model, {a, b});
      ASSERT_EQ(expanded.size(), 1u);
      EXPECT_EQ(expanded[0].data_type, native.data_type);
      EXPECT_EQ(TensorBytes(expanded[0]), TensorBytes(native));
    }
  }
}
} // namespace Test
