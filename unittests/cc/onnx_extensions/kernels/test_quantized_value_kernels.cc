// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;

namespace {

void ExpectValues(const Tensor &actual, const std::vector<double> &expected) {
  ASSERT_EQ(actual.element_count(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (actual.data_type == TensorProto::DOUBLE)
      EXPECT_DOUBLE_EQ(actual.As<double>()[i], expected[i]);
    else if (actual.data_type == TensorProto::FLOAT)
      EXPECT_DOUBLE_EQ(actual.AsFloat()[i], expected[i]);
    else {
      uint16_t bits;
      std::memcpy(&bits, actual.bytes() + 2 * i, sizeof(bits));
      EXPECT_DOUBLE_EQ(actual.data_type == TensorProto::FLOAT16 ? Float16BitsToFloat(bits)
                                                                : Bfloat16BitsToFloat(bits),
                       expected[i]);
    }
  }
}

NodeProto QuantizeNode(const StructTypeProto &type) {
  NodeProto node;
  node.set_domain("ai.rt");
  node.set_op_type("Quantize");
  node.add_input("X");
  node.add_output("Q");
  auto *attribute = node.add_attribute();
  attribute->set_name("type");
  attribute->set_type(AttributeProto::TYPE_PROTO);
  *attribute->mutable_tp()->mutable_struct_type() = type;
  return node;
}

} // namespace

TEST(QuantizedValueKernels, CalibratesAffineBlocksAndReusesKernel) {
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 8, 4);
  auto node = QuantizeNode(MakeQuantizationType(plan));
  onnx_kernels::kernel::Quantize kernel{KernelContext(OpsetId{"ai.rt", 1})};
  kernel.set_node(node);
  RuntimeContext context;
  for (float multiplier : {1.f, 2.f}) {
    context.Put("X", Tensor::FromFloat("", {8},
                                       {-8 * multiplier, -4 * multiplier, 0, 7 * multiplier,
                                        -16 * multiplier, 0, 8 * multiplier, 14 * multiplier}));
    kernel.Run(context);
    plan.runs[0].blocks[0].scale = multiplier;
    plan.runs[0].blocks[1].scale = 2 * multiplier;
    auto reference = QuantizeTensor(context.Get("X"), plan).Encoded();
    reference.set_name("Q");
    EXPECT_EQ(context.values().at("Q").Encoded().SerializeAsString(),
              reference.SerializeAsString());
  }
}

TEST(QuantizedValueKernels, OptionalParametersAndLearnedCodebooks) {
  const auto input = Tensor::FromFloat("", {3}, {-1, 0, 1});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kSqueezellm, 3);
  const auto type = MakeQuantizationType(plan);
  EXPECT_THROW(QuantizeTensor(input, type), std::invalid_argument);
  const auto table =
      Tensor::FromDouble("", {16}, {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  const auto scale = Tensor::FromDouble("", {}, {1});
  QuantizationParameters parameters;
  parameters.scales = &scale;
  parameters.codebooks = &table;
  ExpectValues(DequantizeTensor(QuantizeTensor(input, type, parameters)), {-1, 0, 1});
  const auto bad = Tensor::FromDouble("", {2}, {1, 2});
  parameters.scales = &bad;
  EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
  parameters.scales = &scale;
  parameters.codebooks = &bad;
  EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
}

TEST(QuantizedValueKernels, RequiresAndAppliesTransformsAndPermutation) {
  const auto input = Tensor::FromFloat("", {2}, {-4, 6});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kSmoothquant, 2);
  plan.permutation = {1, 0};
  plan.transform_size = 2;
  plan.forward = {2, 0, 0, 2};
  plan.inverse = {.5, 0, 0, .5};
  const auto type = MakeQuantizationType(plan);
  EXPECT_THROW(QuantizeTensor(input, type), std::invalid_argument);
  const auto permutation = Tensor::FromInt64("", {2}, {1, 0});
  const auto forward =
      Tensor::FromDouble("", {plan.transform_size, plan.transform_size}, plan.forward);
  const auto inverse =
      Tensor::FromDouble("", {plan.transform_size, plan.transform_size}, plan.inverse);
  const auto scale = Tensor::FromDouble("", {}, {2});
  QuantizationParameters parameters;
  parameters.permutation = &permutation;
  parameters.forward = &forward;
  parameters.inverse = &inverse;
  parameters.scales = &scale;
  ExpectValues(DequantizeTensor(QuantizeTensor(input, type, parameters)), {-4, 6});

  for (const Shape &shape : {Shape{4}, Shape{1, 4}, Shape{4, 1}, Shape{1, 2, 2}}) {
    const auto bad_forward = Tensor::FromDouble("", shape, plan.forward);
    const auto bad_inverse = Tensor::FromDouble("", shape, plan.inverse);
    parameters.forward = &bad_forward;
    EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
    parameters.forward = &forward;
    parameters.inverse = &bad_inverse;
    EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
    parameters.inverse = &inverse;
  }
  ExpectValues(DequantizeTensor(QuantizeTensor(input, type, parameters)), {-4, 6});
}

TEST(QuantizedValueKernels, RejectsScalarTransformForOneByOneMatrix) {
  const auto input = Tensor::FromFloat("", {1}, {2});
  auto plan = MakeQuantizationPlan(QuantizationFormat::kSmoothquant, 1);
  plan.transform_size = 1;
  plan.forward = {1};
  plan.inverse = {1};
  const auto type = MakeQuantizationType(plan);
  const auto matrix = Tensor::FromDouble("", {1, 1}, {1});
  const auto scalar = Tensor::FromDouble("", {}, {1});
  QuantizationParameters parameters;
  parameters.forward = &matrix;
  parameters.inverse = &matrix;
  ExpectValues(DequantizeTensor(QuantizeTensor(input, type, parameters)), {2});
  parameters.forward = &scalar;
  EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
  parameters.forward = &matrix;
  parameters.inverse = &scalar;
  EXPECT_THROW(QuantizeTensor(input, type, parameters), std::invalid_argument);
}

TEST(QuantizedValueKernels, DequantizesToEachRequestedDtypeAndRejectsOverflow) {
  const auto input = Tensor::FromFloat("", {3}, {-1, 0, 1});
  const auto encoded = QuantizeTensor(input, MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
  SimpleRawBufferAllocator allocator(4);
  onnx_kernels::kernel::Dequantize kernel{KernelContext(OpsetId{"ai.rt", 1}, &allocator)};
  for (int32_t type :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    const auto output = kernel(encoded.Encoded(), type);
    EXPECT_EQ(output.data_type, type);
    EXPECT_EQ(output.allocation_owner(), &allocator);
    ExpectValues(output, {-1, 0, 1});
  }
  EXPECT_THROW(kernel(encoded.Encoded(), TensorProto::INT8), std::invalid_argument);
  EXPECT_THROW(kernel(encoded.Encoded(), TensorProto::UNDEFINED), std::invalid_argument);
  const auto large = QuantizeTensor(Tensor::FromFloat("", {1}, {65536}),
                                    MakeQuantizationPlan(QuantizationFormat::kTiledFloat, 1));
  EXPECT_THROW(kernel(large.Encoded(), TensorProto::FLOAT16), std::invalid_argument);
  auto corrupt = encoded.Encoded();
  corrupt.mutable_raw_data()->resize(1);
  EXPECT_THROW(kernel(corrupt, TensorProto::FLOAT), std::invalid_argument);
}

TEST(QuantizedValueKernels, RejectsMalformedTypesAndInputs) {
  auto type = MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
  auto input = Tensor::FromFloat("", {3}, {-1, 0, 1});
  type.set_name("unknown");
  EXPECT_THROW(QuantizeTensor(input, type), std::invalid_argument);
  type = MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 4));
  EXPECT_THROW(QuantizeTensor(input, type), std::invalid_argument);
  type = MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
  input = Tensor::FromFloat("", {3}, {0, 1, std::numeric_limits<float>::infinity()});
  EXPECT_THROW(QuantizeTensor(input, type), std::invalid_argument);
  auto node = QuantizeNode(type);
  node.clear_attribute();
  onnx_kernels::kernel::Quantize kernel{KernelContext(OpsetId{"ai.rt", 1})};
  kernel.set_node(node);
  RuntimeContext context;
  EXPECT_THROW(kernel.Run(context), std::invalid_argument);
}

TEST(QuantizedValueKernels, SchemasAndBackendCases) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_shapes::RegisterShapeFunctions();
  for (const std::string op : {"Quantize", "Dequantize"}) {
    const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory(op);
    ASSERT_EQ(schemas.size(), 1);
    EXPECT_EQ(schemas[0].domain(), "ai.rt");
    EXPECT_EQ(schemas[0].since_version(), 1);
    const auto cases = core::backend_test::CollectTestCases(op);
    ASSERT_EQ(cases.size(), 4);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.name);
      const auto &model = test.model();
      auto inferred = model;
      EXPECT_NO_THROW(core::shapes::InferShapesModel(inferred));
      RuntimeSession session(model);
      RuntimeContext context;
      RegisterModelFunctions(model, context);
      for (const auto &input : test.data_sets()[0].inputs)
        context.Put(input.name, input);
      session.Run(context);
      const auto &expected = test.data_sets()[0].outputs[0];
      const auto &actual = context.Get(expected.name);
      ASSERT_EQ(actual.data_type, expected.data_type);
      ASSERT_EQ(actual.shape, expected.shape);
      ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
      EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
    }
  }
}

TEST(QuantizedValueKernels, SchemaValidatesStructuredInputAndRequiredAttributes) {
  const auto schema = onnx_op::GetAllOnnxOpSchemasWithHistory("Dequantize")[0];
  NodeProto node;
  node.set_op_type("Dequantize");
  node.set_domain("ai.rt");
  node.add_input("X");
  node.add_output("Y");
  EXPECT_THROW(schema.Verify(node), core::schema::SchemaError);
  AddAttribute<int64_t>(node, "dtype", TensorProto::FLOAT);
  ValueInfoProto input;
  input.set_name("X");
  input.mutable_type()->mutable_struct_type();
  std::vector<std::optional<core::schema::SchemaInputValue>> inputs{input};
  EXPECT_NO_THROW(schema.Verify(node, &inputs));
  input.mutable_type()->clear_struct_type();
  input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  inputs[0] = input;
  EXPECT_THROW(schema.Verify(node, &inputs), core::schema::SchemaError);
}

TEST(QuantizedValueKernels, ResolvesModelTypesInSessionsAndChildren) {
  onnx_kernels::RegisterKernelFunctions();
  auto type = MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
  ModelProto model;
  auto *opset = model.add_opset_import();
  opset->set_domain("ai.rt");
  opset->set_version(1);
  opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(21);
  type.set_type_id(1);
  *model.add_struct_types() = type;
  auto *graph = model.mutable_graph();
  graph->set_name("catalogue");
  StructTypeProto reference;
  reference.set_type_ref(1);
  *graph->add_node() = QuantizeNode(reference);
  graph->add_input()->set_name("X");
  graph->add_output()->set_name("Q");
  RuntimeSession session(model);
  RuntimeContext context;
  context.Put("X", Tensor::FromFloat("", {3}, {-1, 0, 1}));
  session.Run(context);
  EXPECT_EQ(context.struct_type_catalogue().size(), 1);
  auto child = context.MakeFunctionContext();
  EXPECT_EQ(child.struct_type_catalogue().size(), 1);
  auto value = context.values().at("Q").Encoded();
  *value.mutable_struct_type() = reference;
  onnx_kernels::kernel::Dequantize kernel{KernelContext(OpsetId{"ai.rt", 1})};
  ExpectValues(kernel(value, TensorProto::FLOAT, child.struct_type_catalogue()), {-1, 0, 1});
}

TEST(QuantizedValueKernels, SharedParametersStayIndependentOfTypeReferences) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_shapes::RegisterShapeFunctions();
  RuntimeValue retained;
  {
    const auto source = Tensor::FromFloat("X", {3}, {-2, 0, 2});
    auto type = MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
    ModelProto model;
    type.set_type_id(7);
    *model.add_struct_types() = type;
    auto *graph = model.mutable_graph();
    graph->set_name("shared");
    StructTypeProto reference;
    reference.set_type_ref(7);
    auto *annotation = graph->add_quantization_annotation();
    annotation->set_tensor_name("onnx_light.quantization.parameters:fixed");
    auto add = [&](const std::string &role, const std::string &bytes) {
      auto *tensor = graph->add_initializer();
      tensor->set_name(role);
      tensor->set_data_type(TensorProto::UINT8);
      tensor->add_dims(bytes.size());
      tensor->set_raw_data(bytes);
      auto *mapping = annotation->add_quant_parameter_tensor_names();
      mapping->set_key(role);
      mapping->set_value(role);
    };
    add("storage_type", reference.SerializeAsString());
    const auto encoded = QuantizeTensor(source, MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
    add("logical_type", encoded.Encoded().logical_type().SerializeAsString());
    auto *scales = graph->add_initializer();
    scales->set_name("scales");
    scales->set_data_type(TensorProto::DOUBLE);
    scales->add_double_data(2);
    auto *mapping = annotation->add_quant_parameter_tensor_names();
    mapping->set_key("scales");
    mapping->set_value("scales");
    auto quantize = QuantizeNode(reference);
    AddAttribute<std::string>(quantize, "parameter_ref", "fixed");
    *graph->add_node() = quantize;
    auto *identity = graph->add_node();
    identity->set_op_type("Identity");
    identity->add_input("Q");
    identity->add_output("I");
    auto *decode = graph->add_node();
    decode->set_op_type("Dequantize");
    decode->set_domain("ai.rt");
    decode->add_input("I");
    decode->add_output("Y");
    AddAttribute<int64_t>(*decode, "dtype", TensorProto::FLOAT);
    auto *input = graph->add_input();
    input->set_name("X");
    *input->mutable_type() = encoded.Encoded().logical_type();
    auto *output = graph->add_output();
    output->set_name("I");
    *output->mutable_type()->mutable_struct_type() = MakeSharedQuantizationType(type);
    output = graph->add_output();
    output->set_name("Y");
    *output->mutable_type() = encoded.Encoded().logical_type();
    auto inferred = model;
    EXPECT_NO_THROW(core::shapes::InferShapesModel(inferred));
    RuntimeSession session(model);
    RuntimeContext context;
    context.Put("X", source);
    session.Run(context);
    ExpectValues(context.Get("Y"), {-2, 0, 2});
    retained = context.values().at("I").DeepCopy();
    auto child = context.MakeFunctionContext();
    EXPECT_EQ(child.quantization_parameters(), context.quantization_parameters());
    EXPECT_EQ(retained.Encoded().raw_data().size(), 3);
    EXPECT_EQ(retained.Encoded().struct_type().SerializeAsString(),
              MakeSharedQuantizationType(type).SerializeAsString());
    quantize.add_input("X");
    onnx_kernels::kernel::Quantize kernel{KernelContext(OpsetId{"ai.rt", 1})};
    kernel.set_node(quantize);
    EXPECT_THROW(kernel.Run(context), std::invalid_argument);
  }
  ExpectValues(DequantizeTensor(retained), {-2, 0, 2});
}

TEST(QuantizedValueKernels, CalibratesOrtColumnsAndAllBitWidths) {
  const auto input = Tensor::FromFloat("", {3, 2}, {-1, -2, 0, 0, 1, 2});
  for (auto format :
       {QuantizationFormat::kOrtMatmulnbitsInt2, QuantizationFormat::kOrtMatmulnbitsInt4,
        QuantizationFormat::kOrtMatmulnbitsInt8}) {
    auto plan = MakeMatMulNBitsPlan(format, 3, 2, 16);
    const auto fixture = QuantizeTensor(input, plan);
    const auto result = QuantizeTensor(input, fixture.Encoded().struct_type());
    const auto exported = ExportMatMulNBitsInputs(result.Encoded());
    EXPECT_EQ(exported.bits, plan.runs[0].layout.bits);
    const auto decoded = DequantizeTensor(result);
    ASSERT_EQ(decoded.shape, input.shape);
    for (size_t i = 0; i < 6; ++i)
      EXPECT_NEAR(decoded.AsFloat()[i], input.AsFloat()[i], 1e-6);
  }
}

TEST(QuantizedValueKernels, ScalarsEmptyAndOutliers) {
  const auto scalar = Tensor::FromFloat("", {}, {0});
  ExpectValues(DequantizeTensor(QuantizeTensor(scalar, MakeQuantizationType(MakeQuantizationPlan(
                                                           QuantizationFormat::kInt4, 1)))),
               {0});
  const auto empty = Tensor::FromFloat("", {0}, {});
  const auto decoded = DequantizeTensor(QuantizeTensor(
      empty, MakeQuantizationType(MakeQuantizationPlan(QuantizationFormat::kInt4, 0))));
  EXPECT_EQ(decoded.shape, empty.shape);
  auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 3);
  plan.outliers = {0};
  const auto input = Tensor::FromFloat("", {3}, {1000, -1, 1});
  const auto indices = Tensor::FromInt64("", {1}, {0});
  QuantizationParameters parameters;
  parameters.outliers = &indices;
  ExpectValues(DequantizeTensor(QuantizeTensor(input, MakeQuantizationType(plan), parameters)),
               {1000, -1, 1});
}

TEST(QuantizedValueKernels, MatchesExplicitCodecPlansAcrossAllProfiles) {
  const auto input = Tensor::FromFloat("", {4, 2}, std::vector<float>(8, 1));
  const auto scale = Tensor::FromDouble("", {}, {1});
  for (auto format : QuantizationFormats()) {
    SCOPED_TRACE(QuantizationFormatName(format));
    const bool ort = format >= QuantizationFormat::kOrtMatmulnbitsInt2;
    auto plan = ort ? MakeMatMulNBitsPlan(format, 4, 2, 16) : MakeQuantizationPlan(format, 8, 4);
    std::vector<double> tables;
    for (auto &run : plan.runs)
      for (auto &block : run.blocks) {
        if (run.layout.method == QuantizationMethod::kCodebook && block.codebook.empty()) {
          block.codebook.resize(run.layout.books * run.layout.entries * run.layout.vector_size);
          std::fill_n(block.codebook.begin(), run.layout.entries * run.layout.vector_size, 1.);
        }
        tables.insert(tables.end(), block.codebook.begin(), block.codebook.end());
      }
    if (format == QuantizationFormat::kQuarot || format == QuantizationFormat::kQuipSharp ||
        format == QuantizationFormat::kSmoothquant) {
      plan.transform_size = 2;
      plan.forward = {1, 0, 0, 1};
      plan.inverse = plan.forward;
    }
    const auto expected = QuantizeTensor(input, plan);
    const auto codebooks = Tensor::FromDouble("", {static_cast<int64_t>(tables.size())}, tables);
    const auto forward =
        Tensor::FromDouble("", {plan.transform_size, plan.transform_size}, plan.forward);
    const auto inverse =
        Tensor::FromDouble("", {plan.transform_size, plan.transform_size}, plan.inverse);
    QuantizationParameters parameters;
    parameters.scales = &scale;
    if (!tables.empty())
      parameters.codebooks = &codebooks;
    if (plan.transform_size) {
      parameters.forward = &forward;
      parameters.inverse = &inverse;
    }
    const auto actual = QuantizeTensor(input, expected.Encoded().struct_type(), parameters);
    EXPECT_EQ(actual.Encoded().SerializeAsString(), expected.Encoded().SerializeAsString());
  }
}
