// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Additional C++ tests for onnx_light/onnx_lib/checker.cc to cover branches
// that the upstream-mirrored checker_test.cc does not exercise: validation of
// ValueInfo / Tensor / SparseTensor / Sequence / Map / Optional / Attribute /
// Node / Graph / Model protos as well as cycle detection in model-local
// functions and the experimental-op classifier.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "onnx_core/runtime/quantization.h"
#include "onnx_lib/checker.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

#ifndef ONNX_NO_EXCEPTIONS

namespace {

using checker::CheckerContext;
using checker::LexicalScopeContext;
using checker::ValidationError;

// Helper: build a default CheckerContext for the current IR version with an
// ONNX opset import so that node/graph checks have a domain to resolve.
CheckerContext MakeCtx(int ir_version = IR_VERSION) {
  CheckerContext ctx;
  ctx.set_ir_version(ir_version);
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  ctx.set_opset_imports(opsets);
  return ctx;
}

// Helper: build a minimal valid TensorProto holding a single float scalar.
TensorProto MakeFloatScalar(const std::string &name, float value) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.add_float_data(value);
  return t;
}

ModelProto MakeSharedParameterModel(uint64_t count = 8, uint64_t block_size = 4) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  for (const auto &domain : {"", "ai.rt", "local"}) {
    auto *opset = model.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(std::string(domain).empty() ? 21 : 1);
  }
  auto *graph = model.mutable_graph();
  graph->set_name("shared");
  const auto plan = core::runtime::MakeQuantizationPlan(core::runtime::QuantizationFormat::kInt4,
                                                        count, block_size);
  const auto storage = core::runtime::MakeQuantizationType(plan);
  TypeProto logical;
  logical.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  logical.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(count);
  auto *input = graph->add_input();
  input->set_name("X");
  *input->mutable_type() = logical;
  auto *output = graph->add_output();
  output->set_name("Q");
  *output->mutable_type()->mutable_struct_type() =
      core::runtime::MakeSharedQuantizationType(storage);
  auto *annotation = graph->add_quantization_annotation();
  annotation->set_tensor_name("onnx_light.quantization.parameters:common");
  auto descriptor = [&](const char *role, const std::string &bytes) {
    auto *tensor = graph->add_initializer();
    tensor->set_name(role);
    tensor->set_data_type(TensorProto::UINT8);
    tensor->add_dims(bytes.size());
    tensor->set_raw_data(bytes);
    auto *mapping = annotation->add_quant_parameter_tensor_names();
    mapping->set_key(role);
    mapping->set_value(role);
  };
  descriptor("storage_type", storage.SerializeAsString());
  descriptor("logical_type", logical.SerializeAsString());
  auto *scales = graph->add_initializer();
  scales->set_name("scales");
  scales->set_data_type(TensorProto::FLOAT);
  for (const auto &run : plan.runs)
    for (size_t i = 0; i < run.blocks.size(); ++i)
      scales->add_float_data(static_cast<float>(i + 1));
  scales->add_dims(scales->float_data().size());
  auto *mapping = annotation->add_quant_parameter_tensor_names();
  mapping->set_key("scales");
  mapping->set_value("scales");
  auto *node = graph->add_node();
  node->set_domain("ai.rt");
  node->set_op_type("Quantize");
  node->add_input("X");
  node->add_output("Q");
  auto *type = node->add_attribute();
  type->set_name("type");
  type->set_type(AttributeProto::TYPE_PROTO);
  *type->mutable_tp()->mutable_struct_type() = storage;
  auto *reference = node->add_attribute();
  reference->set_name("parameter_ref");
  reference->set_type(AttributeProto::STRING);
  reference->set_s("common");
  return model;
}

ModelProto MakeSharedParameterFunctionModel() {
  auto model = MakeSharedParameterModel();
  auto *function = model.add_functions();
  function->set_domain("local");
  function->set_name("Shared");
  function->add_input("X");
  function->add_output("Q");
  function->add_attribute("parameters");
  for (const auto &opset : model.opset_import())
    *function->add_opset_import() = opset;
  *function->add_node() = model.graph().node(0);
  auto *reference = function->mutable_node(0)->mutable_attribute(1);
  reference->clear_s();
  reference->set_ref_attr_name("parameters");
  model.mutable_graph()->clear_node();
  auto *call = model.mutable_graph()->add_node();
  call->set_domain("local");
  call->set_op_type("Shared");
  call->add_input("X");
  call->add_output("Q");
  auto *argument = call->add_attribute();
  argument->set_name("parameters");
  argument->set_type(AttributeProto::STRING);
  argument->set_s("common");
  return model;
}

} // namespace

TEST(CHECKER_COVERAGE, PagedCacheValidatesSharedParameterReferences) {
  auto model = MakeSharedParameterModel();
  TypeProto logical;
  logical.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : {1, 1, 2, 4})
    logical.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  for (auto &tensor : *model.mutable_graph()->mutable_initializer())
    if (tensor.name() == "logical_type") {
      tensor.set_raw_data(logical.SerializeAsString());
      (*tensor.mutable_dims())[0] = tensor.raw_data().size();
    }
  *model.mutable_graph()->mutable_input(0)->mutable_type() = logical;
  const auto parameters = core::runtime::QuantizationParameterCatalogue::Build(model);
  const auto plan =
      core::runtime::MakeQuantizationPlan(core::runtime::QuantizationFormat::kInt4, 8, 4);
  const auto source =
      core::runtime::Tensor::FromFloat("K", {1, 1, 2, 4}, std::vector<float>(8, 2.f));
  const auto shared = core::runtime::QuantizeTensorShared(
      source, core::runtime::MakeQuantizationType(plan), "common", parameters);
  auto *cache = model.mutable_graph()->add_paged_cache_initializer();
  cache->set_name("cache");
  auto *block = cache->add_blocks();
  block->set_start(0);
  block->set_length(1);
  *block->mutable_encoded_key() = shared.Encoded();
  *block->mutable_encoded_value() = shared.Encoded();
  auto *output = model.mutable_graph()->add_output();
  output->set_name("cache");
  *output->mutable_type() = PagedKVCacheTypeV1();
  ASSERT_NO_THROW(checker::check_model(model));
  block->mutable_encoded_key()->set_parameter_ref("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
  block->mutable_encoded_key()->set_parameter_ref("common");
  block->set_length(3);
  EXPECT_THROW(checker::check_model(model), ValidationError);
}

TEST(CHECKER_COVERAGE, SharedParameterDeclarationsMatchRuntimeValidation) {
  const auto original = MakeSharedParameterModel();
  ASSERT_NO_THROW(checker::check_model(original));
  for (const auto &failure :
       {"unknown_role", "descriptor_dtype", "descriptor_rank", "descriptor_wire", "logical_dtype",
        "logical_shape", "scales_dtype", "scales_count", "scales_value"}) {
    SCOPED_TRACE(failure);
    ModelProto model;
    model.CopyFrom(original);
    auto *graph = model.mutable_graph();
    auto *scales = graph->mutable_initializer(2);
    const std::string kind = failure;
    if (kind == "unknown_role") {
      auto *mapping = graph->mutable_quantization_annotation(0)->add_quant_parameter_tensor_names();
      mapping->set_key("unknown");
      mapping->set_value("scales");
    } else if (kind == "descriptor_dtype") {
      graph->mutable_initializer(0)->set_data_type(TensorProto::INT8);
    } else if (kind == "descriptor_rank") {
      graph->mutable_initializer(0)->add_dims(1);
    } else if (kind == "descriptor_wire") {
      auto *descriptor = graph->mutable_initializer(0);
      const auto &raw = descriptor->raw_data();
      std::string bytes(reinterpret_cast<const char *>(raw.data()), raw.size());
      bytes.push_back('\0');
      descriptor->clear_dims();
      descriptor->add_dims(bytes.size());
      descriptor->set_raw_data(bytes);
    } else if (kind == "logical_dtype" || kind == "logical_shape") {
      TypeProto logical;
      const auto &raw = graph->initializer(1).raw_data();
      logical.ParseFromString(std::string(reinterpret_cast<const char *>(raw.data()), raw.size()));
      if (kind == "logical_dtype")
        logical.mutable_tensor_type()->set_elem_type(TensorProto::INT64);
      else
        logical.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(7);
      const auto bytes = logical.SerializeAsString();
      auto *descriptor = graph->mutable_initializer(1);
      descriptor->clear_dims();
      descriptor->add_dims(bytes.size());
      descriptor->set_raw_data(bytes);
    } else if (kind == "scales_dtype") {
      scales->set_data_type(TensorProto::INT64);
      scales->clear_float_data();
      scales->add_int64_data(1);
      scales->add_int64_data(2);
    } else if (kind == "scales_count") {
      scales->clear_dims();
      scales->add_dims(3);
      scales->add_float_data(3);
    } else {
      scales->clear_float_data();
      scales->add_float_data(0);
      scales->add_float_data(2);
    }
    EXPECT_THROW(core::runtime::QuantizationParameterCatalogue::Build(model),
                 std::invalid_argument);
    EXPECT_THROW(checker::check_model(model), ValidationError);
  }
}

TEST(CHECKER_COVERAGE, SharedParameterValidationDoesNotMaterializeLargeLogicalShape) {
  const uint64_t count = uint64_t{1} << (sizeof(size_t) > 4 ? 30 : 27);
  auto model = MakeSharedParameterModel(count, count);
  EXPECT_NO_THROW(checker::check_model(model));
  TypeProto logical;
  logical.CopyFrom(model.graph().input(0).type());
  logical.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(count - 1);
  const auto bytes = logical.SerializeAsString();
  auto *descriptor = model.mutable_graph()->mutable_initializer(1);
  descriptor->clear_dims();
  descriptor->add_dims(bytes.size());
  descriptor->set_raw_data(bytes);
  EXPECT_THROW(checker::check_model(model), ValidationError);
}

TEST(CHECKER_COVERAGE, SharedParameterFunctionArgumentsAndDefaults) {
  auto model = MakeSharedParameterFunctionModel();
  ASSERT_NO_THROW(checker::check_model(model));
  auto *call = model.mutable_graph()->mutable_node(0);
  call->mutable_attribute(0)->set_s("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
  call->mutable_attribute(0)->clear_s();
  call->mutable_attribute(0)->set_type(AttributeProto::INT);
  call->mutable_attribute(0)->set_i(1);
  EXPECT_THROW(checker::check_model(model), ValidationError);
  call->clear_attribute();
  EXPECT_THROW(checker::check_model(model), ValidationError);
  auto *function = model.mutable_functions(0);
  function->clear_attribute();
  auto *default_value = function->add_attribute_proto();
  default_value->set_name("parameters");
  default_value->set_type(AttributeProto::STRING);
  default_value->set_s("common");
  call->clear_attribute();
  ASSERT_NO_THROW(checker::check_model(model));
  default_value->set_s("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
  auto *override_value = call->add_attribute();
  override_value->set_name("parameters");
  override_value->set_type(AttributeProto::STRING);
  override_value->set_s("common");
  EXPECT_NO_THROW(checker::check_model(model));
}

TEST(CHECKER_COVERAGE, SharedParameterAttributeReferencesRequireFunctionScope) {
  for (bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    auto model = MakeSharedParameterModel();
    auto *reference = model.mutable_graph()->mutable_node(0)->mutable_attribute(1);
    reference->clear_s();
    reference->set_ref_attr_name("parameters");
    if (nested) {
      GraphProto branch;
      branch.CopyFrom(model.graph());
      branch.clear_input();
      branch.clear_initializer();
      branch.clear_quantization_annotation();
      model.mutable_graph()->clear_node();
      auto *node = model.mutable_graph()->add_node();
      node->set_op_type("If");
      node->add_input("condition");
      node->add_output("Q");
      for (const auto &name : {"then_branch", "else_branch"}) {
        auto *attribute = node->add_attribute();
        attribute->set_name(name);
        attribute->set_type(AttributeProto::GRAPH);
        *attribute->mutable_g() = branch;
      }
      auto *condition = model.mutable_graph()->add_initializer();
      condition->set_name("condition");
      condition->set_data_type(TensorProto::BOOL);
      condition->add_int32_data(1);
    }
    try {
      checker::check_model(model);
      FAIL() << "Unbound graph attribute reference was accepted.";
    } catch (const ValidationError &error) {
      EXPECT_NE(std::string(error.what()).find("require an unbound function body"),
                std::string::npos);
    }
  }
}

TEST(CHECKER_COVERAGE, SharedParameterInitializersMatchTheirCatalogue) {
  auto original = MakeSharedParameterModel();
  const auto parameters = core::runtime::QuantizationParameterCatalogue::Build(original);
  const auto source = core::runtime::Tensor::FromFloat("Q", {8}, std::vector<float>(8, 1));
  const auto shared = core::runtime::QuantizeTensorShared(
      source, original.graph().node(0).attribute(0).tp().struct_type(), "common", parameters);
  original.mutable_graph()->clear_node();
  *original.mutable_graph()->add_encoded_initializer() = shared.Encoded();
  ASSERT_NO_THROW(checker::check_model(original));
  for (const auto &failure : {"dtype", "shape", "compact_type", "payload", "reserved"}) {
    SCOPED_TRACE(failure);
    ModelProto model;
    model.CopyFrom(original);
    auto *encoded = model.mutable_graph()->mutable_encoded_initializer(0);
    const std::string kind = failure;
    if (kind == "dtype") {
      encoded->mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    } else if (kind == "shape") {
      auto *shape = encoded->mutable_logical_type()->mutable_tensor_type()->mutable_shape();
      shape->mutable_dim(0)->set_dim_value(4);
      shape->add_dim()->set_dim_value(2);
    } else if (kind == "compact_type") {
      encoded->mutable_struct_type()->set_name("different_compact_format");
    } else if (kind == "payload") {
      encoded->mutable_raw_data()->resize(4);
      encoded->mutable_struct_type()
          ->mutable_structure()
          ->mutable_field(0)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(0)
          ->set_dim_value(4);
    } else {
      (*encoded->mutable_raw_data())[0] = 1;
    }
    EXPECT_NO_THROW(StructTypeCatalogue{}.ValidateEncodedValue(*encoded));
    EXPECT_THROW(checker::check_model(model), ValidationError);
  }
  auto *definition = original.add_struct_types();
  *definition = shared.Encoded().struct_type();
  definition->set_type_id(1);
  auto *encoded = original.mutable_graph()->mutable_encoded_initializer(0);
  *encoded->mutable_struct_type() = StructTypeProto{};
  encoded->mutable_struct_type()->set_type_ref(1);
  EXPECT_NO_THROW(checker::check_model(original));
}

TEST(CHECKER_COVERAGE, SharedParameterNestedFunctionBindings) {
  auto model = MakeSharedParameterFunctionModel();
  FunctionProto wrapper;
  wrapper.CopyFrom(model.functions(0));
  wrapper.set_name("Wrapper");
  wrapper.clear_node();
  auto *inner = wrapper.add_node();
  *inner = model.graph().node(0);
  inner->mutable_attribute(0)->clear_s();
  inner->mutable_attribute(0)->set_ref_attr_name("parameters");
  *model.add_functions() = wrapper;
  auto *call = model.mutable_graph()->mutable_node(0);
  call->set_op_type("Wrapper");
  ASSERT_NO_THROW(checker::check_model(model));
  call->mutable_attribute(0)->set_s("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
  call->mutable_attribute(0)->set_s("common");
  auto *second = model.mutable_graph()->add_node();
  *second = model.graph().node(0);
  second->clear_output();
  second->add_output("R");
  second->mutable_attribute(0)->set_s("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
  second->clear_attribute();
  EXPECT_THROW(checker::check_model(model), ValidationError);
}

TEST(CHECKER_COVERAGE, SharedParameterFunctionSubgraphBindings) {
  auto model = MakeSharedParameterFunctionModel();
  auto *function = model.mutable_functions(0);
  NodeProto quantize;
  quantize.CopyFrom(function->node(0));
  function->clear_node();
  function->add_input("condition");
  auto *branch = function->add_node();
  branch->set_op_type("If");
  branch->add_input("condition");
  branch->add_output("Q");
  for (const auto &name : {"then_branch", "else_branch"}) {
    auto *attribute = branch->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::GRAPH);
    auto *graph = attribute->mutable_g();
    graph->set_name(name);
    *graph->add_node() = quantize;
    *graph->add_output() = model.graph().output(0);
  }
  auto *condition = model.mutable_graph()->add_initializer();
  condition->set_name("condition");
  condition->set_data_type(TensorProto::BOOL);
  condition->add_int32_data(1);
  auto *call = model.mutable_graph()->mutable_node(0);
  call->add_input("condition");
  ASSERT_NO_THROW(checker::check_model(model));
  call->mutable_attribute(0)->set_s("missing");
  EXPECT_THROW(checker::check_model(model), ValidationError);
}

// ---------------------------------------------------------------------------
// check_value_info
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, ValueInfoEmptyNameRejected) {
  ValueInfoProto vi;
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoMissingTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorOK) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  tt->mutable_shape(); // empty shape (scalar) is fine
  EXPECT_NO_THROW(checker::check_value_info(vi, MakeCtx()));
}

TEST(CHECKER_COVERAGE, ValueInfoTensorNegativeElemTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(static_cast<TensorProto::DataType>(-1));
  tt->mutable_shape();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorElemTypeAboveLimitRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(static_cast<TensorProto::DataType>(2049));
  tt->mutable_shape();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorMissingShapeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  // No mutable_shape() call -> has_shape() is false.
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoSubgraphRelaxed) {
  // For a non-main graph, only the name is required.
  ValueInfoProto vi;
  vi.set_name("x");
  CheckerContext ctx = MakeCtx();
  ctx.set_is_main_graph(false);
  EXPECT_NO_THROW(checker::check_value_info(vi, ctx));
}

TEST(CHECKER_COVERAGE, ValueInfoSequenceMissingElemTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("seq");
  vi.mutable_type()->mutable_sequence_type();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoOptionalOK) {
  ValueInfoProto vi;
  vi.set_name("opt");
  auto *opt = vi.mutable_type()->mutable_optional_type();
  auto *inner_tt = opt->mutable_elem_type()->mutable_tensor_type();
  inner_tt->set_elem_type(TensorProto::FLOAT);
  inner_tt->mutable_shape();
  EXPECT_NO_THROW(checker::check_value_info(vi, MakeCtx()));
}

TEST(CHECKER_COVERAGE, ValueInfoMapMissingKeyTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("m");
  // mutable_map_type() with no key_type / value_type set
  vi.mutable_type()->mutable_map_type();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoSparseTensorMissingShapeRejected) {
  ValueInfoProto vi;
  vi.set_name("s");
  auto *st = vi.mutable_type()->mutable_sparse_tensor_type();
  st->set_elem_type(TensorProto::FLOAT);
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoOpaqueOK) {
  ValueInfoProto vi;
  vi.set_name("o");
  auto *op = vi.mutable_type()->mutable_opaque_type();
  op->set_domain("com.microsoft.test");
  op->set_name("ComplexOpaqueType");
  EXPECT_NO_THROW(checker::check_value_info(vi, MakeCtx()));
}

TEST(CHECKER_COVERAGE, ValueInfoOpaqueMissingNameRejected) {
  ValueInfoProto vi;
  vi.set_name("o");
  // mutable_opaque_type() with no name set
  vi.mutable_type()->mutable_opaque_type();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

// ---------------------------------------------------------------------------
// check_tensor
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, TensorUndefinedDataTypeRejected) {
  TensorProto t;
  // Default-constructed TensorProto has data_type==UNDEFINED.
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorFloatOK) {
  TensorProto t = MakeFloatScalar("ok", 1.0f);
  EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
}

TEST(CHECKER_COVERAGE, TensorRawDataOK) {
  TensorProto t;
  t.set_name("raw");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  float v = 1.5f;
  std::vector<uint8_t> bytes(sizeof(v));
  std::memcpy(bytes.data(), &v, sizeof(v));
  for (auto b : bytes) {
    t.ref_raw_data().push_back(b);
  }
  EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
}

TEST(CHECKER_COVERAGE, TensorMultipleValueFieldsRejected) {
  TensorProto t;
  t.set_name("multi");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.add_float_data(1.0f);
  t.add_int32_data(2);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorZeroElementsWithDataRejected) {
  TensorProto t;
  t.set_name("z");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(int64_t{0});
  t.add_float_data(1.0f);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorWrongFieldForDataTypeRejected) {
  // INT32 dtype but values stored in float_data field.
  TensorProto t;
  t.set_name("badfield");
  t.set_data_type(TensorProto::INT32);
  t.add_dims(1);
  t.add_float_data(1.0f);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorStringInRawDataRejected) {
  TensorProto t;
  t.set_name("s");
  t.set_data_type(TensorProto::STRING);
  t.add_dims(1);
  t.ref_raw_data().push_back('x');
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalMissingLocationRejected) {
  TensorProto t;
  t.set_name("ext");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  // No "location" entry in external_data.
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalWithEmbeddedDataRejected) {
  TensorProto t;
  t.set_name("ext_with_data");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  t.add_float_data(1.0f);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value("weights.bin");
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalInvalidLocationRejected) {
  TensorProto t;
  t.set_name("ext_bad_loc");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value(".."); // rejected by resolve_external_data_location
  CheckerContext ctx = MakeCtx();
  ctx.set_model_dir("localfolder");
  EXPECT_THROW(checker::check_tensor(t, ctx), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalInvalidLocationBypassed) {
  TensorProto t;
  t.set_name("ext_bad_loc");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value(".."); // would be rejected by resolve_external_data_location
  CheckerContext ctx = MakeCtx();
  ctx.set_model_dir("localfolder");
  ctx.set_skip_external_data_location_check(true);
  EXPECT_NO_THROW(checker::check_tensor(t, ctx));
}

TEST(CHECKER_COVERAGE, TensorExternalEmptyLocationBypassed) {
  TensorProto t;
  t.set_name("ext_empty_loc");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value(""); // would be rejected: location should not be empty
  CheckerContext ctx = MakeCtx();
  ctx.set_skip_external_data_location_check(true);
  EXPECT_NO_THROW(checker::check_tensor(t, ctx));
}

TEST(CHECKER_COVERAGE, TensorExternalMissingLocationBypassed) {
  TensorProto t;
  t.set_name("ext_no_loc");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  // No "location" entry in external_data.
  CheckerContext ctx = MakeCtx();
  ctx.set_skip_external_data_location_check(true);
  EXPECT_NO_THROW(checker::check_tensor(t, ctx));
}

TEST(CHECKER_COVERAGE, TensorUnrecognizedDataTypeRejected) {
  TensorProto t;
  t.set_name("bogus");
  // Pick a value not part of the switch in check_tensor (and not UNDEFINED).
  t.set_data_type(static_cast<TensorProto::DataType>(1234));
  t.add_dims(1);
  t.add_int32_data(1);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

// Reject packed sub-byte tensors whose raw_data payload is too small.
TEST(CHECKER_COVERAGE, TensorPackedSubByteRawDataTooSmall) {
  // 4-bit types: 2 elements per byte.
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    for (int i = 0; i < 4; ++i) {
      t.ref_raw_data().push_back('\0'); // 1 byte too short (need 5)
    }
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    for (int i = 0; i < 5; ++i) {
      ok.ref_raw_data().push_back('\0'); // ceil(10/2) = 5
    }
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }

  // 2-bit types: 4 elements per byte.
  for (TensorProto::DataType dtype : {TensorProto::INT2, TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    for (int i = 0; i < 2; ++i) {
      t.ref_raw_data().push_back('\0'); // 1 byte too short (need 3)
    }
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    for (int i = 0; i < 3; ++i) {
      ok.ref_raw_data().push_back('\0'); // ceil(10/4) = 3
    }
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }
}

TEST(CHECKER_COVERAGE, TensorFloat6RejectsNonCanonicalEncodings) {
  for (TensorProto::DataType dtype : {TensorProto::FLOAT6E2M3, TensorProto::FLOAT6E3M2}) {
    for (const auto &[num_elements, invalid_last_byte] :
         {std::pair<int64_t, uint8_t>{1, 0x40}, {2, 0x10}, {3, 0x04}}) {
      TensorProto tensor;
      tensor.set_name("t");
      tensor.set_data_type(dtype);
      tensor.add_dims(num_elements);
      const size_t byte_count = static_cast<size_t>((num_elements * 6 + 7) / 8);
      tensor.ref_raw_data().resize(byte_count);
      tensor.ref_raw_data()[byte_count - 1] = invalid_last_byte;
      EXPECT_THROW(checker::check_tensor(tensor, MakeCtx()), ValidationError);

      tensor.ref_raw_data()[byte_count - 1] = 0;
      EXPECT_NO_THROW(checker::check_tensor(tensor, MakeCtx()));
    }

    for (const int32_t invalid_value : {-1, 64}) {
      TensorProto tensor;
      tensor.set_name("t");
      tensor.set_data_type(dtype);
      tensor.add_dims(1);
      tensor.add_int32_data(invalid_value);
      EXPECT_THROW(checker::check_tensor(tensor, MakeCtx()), ValidationError);

      tensor.ref_int32_data()[0] = 63;
      EXPECT_NO_THROW(checker::check_tensor(tensor, MakeCtx()));
    }
  }
}

// Reject non-packed int32_data-stored tensors whose payload is too small.
TEST(CHECKER_COVERAGE, TensorUnpackedInt32DataTooSmall) {
  // BOOL, INT8, UINT8, INT16, UINT16, FLOAT16, BFLOAT16, and FLOAT8* types are
  // stored one element per int32_data entry (they are not bit-packed), unlike
  // INT4/UINT4/FLOAT4E2M1.
  for (TensorProto::DataType dtype :
       {TensorProto::BOOL, TensorProto::INT8, TensorProto::UINT8, TensorProto::INT16,
        TensorProto::UINT16, TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT8E4M3FN,
        TensorProto::FLOAT8E4M3FNUZ, TensorProto::FLOAT8E5M2, TensorProto::FLOAT8E5M2FNUZ,
        TensorProto::FLOAT8E8M0}) {
    // 4 int32_data entries for a 32-element tensor: the 4-bit packed formula
    // ceil(nelem/8) = ceil(32/8) = 4 would incorrectly accept this, but these
    // types require nelem=32 entries (one per element), not the packed count.
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(32);
    for (int i = 0; i < 4; ++i) {
      t.add_int32_data(0);
    }
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    // Exactly enough (unpacked) int32 values must pass.
    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(32);
    for (int i = 0; i < 32; ++i) {
      ok.add_int32_data(0);
    }
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }
}

// Reject packed sub-byte tensors whose int32_data payload is too small.
TEST(CHECKER_COVERAGE, TensorPackedSubByteInt32DataTooSmall) {
  // 4-bit types: 8 elements per int32.
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    t.add_int32_data(0); // 1 int32, need 2
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    ok.add_int32_data(0);
    ok.add_int32_data(0); // ceil(10/8) = 2
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }

  // 2-bit types: 16 elements per int32.
  for (TensorProto::DataType dtype : {TensorProto::INT2, TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(20);
    t.add_int32_data(0); // 1 int32, need 2
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(20);
    ok.add_int32_data(0);
    ok.add_int32_data(0); // ceil(20/16) = 2
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }
}

// Zero-element packed tensors with empty payload must be valid.
TEST(CHECKER_COVERAGE, TensorPackedSubByteZeroElems) {
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1, TensorProto::INT2,
        TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(int64_t{0});
    EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
  }
}

// ---------------------------------------------------------------------------
// check_sparse_tensor
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, SparseTensorDefaultRejected) {
  // Default-constructed SparseTensorProto: values has rank 0 but the checker
  // requires rank 1 — exercises the "must have rank 1" failure branch.
  SparseTensorProto s;
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorValuesRankNot1Rejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  v->add_float_data(3.0f);
  v->add_float_data(4.0f);
  s.add_dims(4);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorZeroDenseRankRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(int64_t{0});
  // No dense dims => rank 0.
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorNonPositiveDimRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(int64_t{0});
  s.add_dims(int64_t{0}); // not > 0
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorNonEmptyNoIndicesRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesWrongDtypeRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT32); // must be INT64
  idx->add_dims(2);
  idx->add_int32_data(0);
  idx->add_int32_data(1);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1OK) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_int64_data(0);
  idx->add_int64_data(2);
  EXPECT_NO_THROW(checker::check_sparse_tensor(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1OutOfRangeRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_int64_data(100); // dense_size is 4
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1UnsortedRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_int64_data(3);
  idx->add_int64_data(1);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesNnzMismatchRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1); // mismatched with NNZ=2
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank2OK) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(2);
  s.add_dims(3);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_dims(2);
  // [[0,0],[1,2]]
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  idx->add_int64_data(1);
  idx->add_int64_data(2);
  EXPECT_NO_THROW(checker::check_sparse_tensor(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank2RankMismatchRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(2);
  s.add_dims(3);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_dims(3); // dense_rank is 2, not 3
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank3Rejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(2);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_dims(1);
  idx->add_dims(1);
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

// ---------------------------------------------------------------------------
// check_sequence / check_optional / check_map
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, SequenceMissingElemTypeRejected) {
  SequenceProto s;
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SequenceUndefinedElemTypeRejected) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::UNDEFINED);
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SequenceTensorOK) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::TENSOR);
  *s.add_tensor_values() = MakeFloatScalar("v0", 1.0f);
  EXPECT_NO_THROW(checker::check_sequence(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SequenceTensorInvalidElementRejected) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::TENSOR);
  auto *bad = s.add_tensor_values();
  bad->set_data_type(TensorProto::UNDEFINED); // triggers check_tensor failure
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, OptionalDefaultUndefinedAccepted) {
  // Default OptionalProto has elem_type=UNDEFINED which is allowed (returns
  // early without further checks).
  OptionalProto o;
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalUndefinedAccepted) {
  OptionalProto o;
  o.set_elem_type(OptionalProto::UNDEFINED);
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalTensorOK) {
  OptionalProto o;
  o.set_elem_type(OptionalProto::TENSOR);
  *o.mutable_tensor_value() = MakeFloatScalar("v", 1.0f);
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalInvalidElemTypeRejected) {
  OptionalProto o;
  o.set_elem_type(static_cast<OptionalProto::DataType>(999));
  EXPECT_THROW(checker::check_optional(o, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapDefaultRejected) {
  // Default-constructed MapProto has key_type==UNDEFINED.
  MapProto m;
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapInvalidKeyTypeRejected) {
  MapProto m;
  m.set_key_type(TensorProto::FLOAT); // disallowed for map keys
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapBothKeyVariantsRejected) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  *m.add_string_keys() = utils::String("k");
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  *values->add_tensor_values() = MakeFloatScalar("v", 1.0f);
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapKeyValueLengthMismatchRejected) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  // 0 values, 1 key -> mismatch
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapInt64KeysOK) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  m.add_keys(2);
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  *values->add_tensor_values() = MakeFloatScalar("a", 1.0f);
  *values->add_tensor_values() = MakeFloatScalar("b", 2.0f);
  EXPECT_NO_THROW(checker::check_map(m, MakeCtx()));
}

// ---------------------------------------------------------------------------
// check_attribute
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, AttributeEmptyNameRejected) {
  AttributeProto attr;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

// Note: AttributeProto.type is a required FIELD that is default-initialised
// to UNDEFINED, so the "type required" branch in check_attribute is not
// reachable from valid proto construction in onnx-light. We therefore only
// test the "type field and data field mismatch" branch below.

TEST(CHECKER_COVERAGE, AttributeTypeFieldMismatchRejected) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::INT); // says INT
  attr.set_f(1.0f);                   // but stores a float
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeMultipleValueFieldsRejected) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_f(1.0f);
  attr.set_i(1);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeIntOK) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::INT);
  attr.set_i(7);
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_attribute(attr, MakeCtx(), lex));
}

TEST(CHECKER_COVERAGE, AttributeFunctionBodyRefWithValueRejected) {
  // Attributes of nodes inside function bodies that have ref_attr_name set
  // must not also have a value field set.
  AttributeProto attr;
  attr.set_name("a");
  attr.set_ref_attr_name("outer");
  attr.set_type(AttributeProto::INT);
  attr.set_i(1);
  LexicalScopeContext lex;
  CheckerContext ctx = MakeCtx();
  ctx.set_is_main_graph(false);
  EXPECT_THROW(checker::check_attribute(attr, ctx, lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeTensorPropagatesError) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::TENSOR);
  attr.mutable_t()->set_data_type(TensorProto::UNDEFINED); // invalid
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

// ---------------------------------------------------------------------------
// check_node
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, NodeEmptyOpTypeRejected) {
  NodeProto node;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeNoInputsOrOutputsRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeMissingOpsetImportRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeDuplicateAttributeNamesRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  *node.add_input() = "x";
  *node.add_output() = "y";
  auto *a1 = node.add_attribute();
  a1->set_name("dup");
  a1->set_type(AttributeProto::INT);
  a1->set_i(1);
  auto *a2 = node.add_attribute();
  a2->set_name("dup");
  a2->set_type(AttributeProto::INT);
  a2->set_i(2);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeUnknownOnnxOpRejected) {
  // Built-in domain with a name that has no registered schema must fail.
  NodeProto node;
  node.set_op_type("ThisOpDoesNotExist_xyz");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeUnknownCustomDomainAcceptedByDefault) {
  // Unknown ops in unknown domains are accepted unless check_custom_domain.
  NodeProto node;
  node.set_op_type("MyOp");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  CheckerContext ctx = MakeCtx();
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  opsets["custom.domain"] = 1;
  ctx.set_opset_imports(opsets);
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_node(node, ctx, lex));
}

TEST(CHECKER_COVERAGE, NodeUnknownCustomDomainRejectedWhenStrict) {
  NodeProto node;
  node.set_op_type("MyOp");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  CheckerContext ctx = MakeCtx();
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  opsets["custom.domain"] = 1;
  ctx.set_opset_imports(opsets);
  ctx.set_check_custom_domain(true);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, ctx, lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeExperimentalOpSkipsSchemaLookup) {
  // Experimental ops short-circuit and must not require a registered schema.
  NodeProto node;
  node.set_op_type("ATen");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_node(node, MakeCtx(), lex));
}

// ---------------------------------------------------------------------------
// check_graph
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, GraphEmptyNameRejected) {
  GraphProto g;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphUnconnectedOutputRejected) {
  GraphProto g;
  g.set_name("g");
  auto *in = g.add_input();
  in->set_name("x");
  auto *ttin = in->mutable_type()->mutable_tensor_type();
  ttin->set_elem_type(TensorProto::FLOAT);
  ttin->mutable_shape();
  // Output 'y' is not produced by any node nor declared as input.
  auto *out = g.add_output();
  out->set_name("y");
  auto *ttout = out->mutable_type()->mutable_tensor_type();
  ttout->set_elem_type(TensorProto::FLOAT);
  ttout->mutable_shape();
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphDuplicateInputNamesRejected) {
  GraphProto g;
  g.set_name("g");
  for (int i = 0; i < 2; ++i) {
    auto *in = g.add_input();
    in->set_name("x"); // duplicate
    auto *tt = in->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);
    tt->mutable_shape();
  }
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphDuplicateInitializerNamesRejected) {
  GraphProto g;
  g.set_name("g");
  for (int i = 0; i < 2; ++i) {
    *g.add_initializer() = MakeFloatScalar("w", 1.0f);
  }
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphInitializerNotInInputForOldIRRejected) {
  // For ir_version <= 3, every initializer must also be an input.
  GraphProto g;
  g.set_name("g");
  *g.add_initializer() = MakeFloatScalar("w", 1.0f);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(3), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphTopologicallyUnsortedNodeRejected) {
  // Node consumes 'z' which is never defined as input/initializer/prior output.
  GraphProto g;
  g.set_name("g");
  auto *node = g.add_node();
  node->set_op_type("Identity");
  *node->add_input() = "z";
  *node->add_output() = "y";
  // Declare an output so the graph can otherwise be valid.
  auto *out = g.add_output();
  out->set_name("y");
  auto *tt = out->mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  tt->mutable_shape();
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

// ---------------------------------------------------------------------------
// check_function_call_cycles
// ---------------------------------------------------------------------------

namespace {

// Add a callee node referring to function (local_domain::name) to function `f`.
void AddCalleeNode(FunctionProto &f, const std::string &domain, const std::string &op_type) {
  auto *n = f.add_node();
  n->set_op_type(op_type);
  n->set_domain(domain);
}

void InitFunction(FunctionProto &f, const std::string &name) {
  f.set_name(name);
  f.set_domain("local");
}

} // namespace

TEST(CHECKER_COVERAGE, FunctionCallCyclesAcyclicOK) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f1");
  AddCalleeNode(*f1, "local", "f2");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f2");
  EXPECT_NO_THROW(checker::check_function_call_cycles(model));
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesSelfLoopRejected) {
  ModelProto model;
  auto *f = model.add_functions();
  InitFunction(*f, "f");
  AddCalleeNode(*f, "local", "f"); // self-reference
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesMutualRejected) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f1");
  AddCalleeNode(*f1, "local", "f2");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f2");
  AddCalleeNode(*f2, "local", "f1");
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesDuplicateImplIdRejected) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f"); // same implementation id (domain + name)
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

// ---------------------------------------------------------------------------
// check_model
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, ModelMissingIrVersionRejected) {
  ModelProto m;
  m.set_ir_version(0); // explicit "unset" sentinel for the checker
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelIrVersionTooHighRejected) {
  ModelProto m;
  m.set_ir_version(IR_VERSION + 1);
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelMissingOpsetImportRejected) {
  ModelProto m;
  m.set_ir_version(7);
  m.mutable_graph()->set_name("g");
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelDuplicateMetadataKeysRejected) {
  ModelProto m;
  m.set_ir_version(7);
  auto *o = m.add_opset_import();
  o->set_domain("");
  o->set_version(21);
  m.mutable_graph()->set_name("g");
  for (int i = 0; i < 2; ++i) {
    auto *kv = m.add_metadata_props();
    kv->set_key("same_key");
    kv->set_value("v");
  }
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelMinimalValidAccepted) {
  ModelProto m;
  m.set_ir_version(7);
  auto *o = m.add_opset_import();
  o->set_domain("");
  o->set_version(21);
  auto *g = m.mutable_graph();
  g->set_name("g");
  EXPECT_NO_THROW(checker::check_model(m));
}

// ---------------------------------------------------------------------------
// check_is_experimental_op
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, IsExperimentalOpTrueForAtenInDefaultDomain) {
  NodeProto node;
  node.set_op_type("ATen");
  EXPECT_TRUE(checker::check_is_experimental_op(node));
}

TEST(CHECKER_COVERAGE, IsExperimentalOpFalseForCustomDomain) {
  NodeProto node;
  node.set_op_type("ATen");
  node.set_domain("custom");
  EXPECT_FALSE(checker::check_is_experimental_op(node));
}

TEST(CHECKER_COVERAGE, IsExperimentalOpFalseForNonExperimentalOp) {
  NodeProto node;
  node.set_op_type("Add");
  EXPECT_FALSE(checker::check_is_experimental_op(node));
}

#endif // ONNX_NO_EXCEPTIONS

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
