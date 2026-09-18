#include "onnx.h"
#include "onnx_ort_flatbuffers.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto OrtInputModel() {
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(18);
  model.set_producer_name("ort_native_test");
  auto *graph = model.mutable_graph();
  graph->set_name("original");
  auto *input = graph->add_input();
  input->set_name("X");
  auto *type = input->mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::FLOAT);
  type->mutable_shape()->add_dim()->set_dim_value(2);
  auto *output = graph->add_output();
  output->CopyFrom(*input);
  output->set_name("Y");
  auto *weight = graph->add_initializer();
  weight->set_name("W");
  weight->set_data_type(TensorProto::FLOAT);
  weight->add_dims(2);
  weight->add_float_data(1.0f);
  weight->add_float_data(2.0f);
  auto *node = graph->add_node();
  node->set_op_type("Add");
  node->add_input("X");
  node->add_input("W");
  node->add_output("A");
  node = graph->add_node();
  node->set_op_type("Relu");
  node->add_input("A");
  node->add_output("Y");
  return model;
}

ModelProto OrtFanoutModel(size_t consumers) {
  auto model = OrtInputModel();
  auto *graph = model.mutable_graph();
  graph->clr_node();
  auto *producer = graph->add_node();
  producer->set_op_type("Add");
  producer->add_input("X");
  producer->add_input("W");
  producer->add_output("A");
  for (size_t i = 0; i < consumers; ++i) {
    auto *node = graph->add_node();
    node->set_op_type("Identity");
    node->add_input("A");
    node->add_output(i + 1 == consumers ? "Y" : "Y" + std::to_string(i));
  }
  return model;
}

uint32_t OrtRead32(const std::string &data, size_t offset) {
  uint32_t result = 0;
  for (size_t i = 0; i < 4; ++i)
    result |= static_cast<uint32_t>(static_cast<unsigned char>(data.at(offset + i))) << (i * 8);
  return result;
}

void OrtWrite32(std::string &data, size_t offset, uint32_t value) {
  for (size_t i = 0; i < 4; ++i)
    data.at(offset + i) = static_cast<char>((value >> (i * 8)) & 255);
}

size_t OrtField(const std::string &data, size_t table, size_t index) {
  const int64_t vtable = static_cast<int64_t>(table) - static_cast<int32_t>(OrtRead32(data, table));
  size_t field = static_cast<size_t>(vtable) + 4 + index * 2;
  const auto offset = static_cast<unsigned char>(data.at(field)) |
                      static_cast<unsigned char>(data.at(field + 1)) << 8;
  return offset ? table + offset : 0;
}

size_t OrtFollow(const std::string &data, size_t offset) {
  return offset + OrtRead32(data, offset);
}

size_t OrtGraph(const std::string &data) {
  size_t root = OrtRead32(data, 0);
  size_t model = OrtFollow(data, OrtField(data, root, 1));
  return OrtFollow(data, OrtField(data, model, 7));
}

ModelProto ReadOrt(const std::string &bytes, ParseOptions options = {}) {
  utils::StringStream stream(bytes.data(), static_cast<int64_t>(bytes.size()));
  ModelProto result;
  ParseModelFromOrtFlatbuffers(result, stream, options);
  return result;
}

} // namespace

TEST(onnx_ort_parsing, ParsesNativeModelAndOwnsPayload) {
  ModelProto parsed;
  {
    auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
    ParseOptions options;
    options.no_copy = true;
    options.alignment = 64;
    options.num_threads = -1;
    parsed = ReadOrt(bytes, options);
    std::fill(bytes.begin(), bytes.end(), '\0');
  }
  EXPECT_EQ(parsed.producer_name(), "ort_native_test");
  EXPECT_EQ(parsed.graph().node().size(), 2u);
  EXPECT_EQ(parsed.graph().node()[0].op_type(), "Add");
  EXPECT_EQ(parsed.graph().node()[1].op_type(), "Relu");
  const auto &raw = parsed.graph().initializer()[0].raw_data();
  ASSERT_EQ(raw.size(), 8u);
  EXPECT_EQ(raw[2], 128);
  EXPECT_EQ(raw[3], 63);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(raw.data()) % 64, 0u);
  EXPECT_EQ(parsed.graph().input()[0].type().tensor_type().shape().dim()[0].dim_value(), 2);
}

TEST(onnx_ort_parsing, AcceptsForwardSharedVtablesAndReordersNodes) {
  auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  const size_t graph = OrtGraph(bytes);
  const size_t nodes = OrtFollow(bytes, OrtField(bytes, graph, 2));
  const size_t first = OrtFollow(bytes, nodes + 4);
  const size_t second = OrtFollow(bytes, nodes + 8);
  const size_t second_vtable = second - OrtRead32(bytes, second);
  ASSERT_GT(second_vtable, first);
  OrtWrite32(
      bytes, first,
      static_cast<uint32_t>(static_cast<int64_t>(first) - static_cast<int64_t>(second_vtable)));
  OrtWrite32(bytes, nodes + 4, static_cast<uint32_t>(second - nodes - 4));
  OrtWrite32(bytes, nodes + 8, static_cast<uint32_t>(first - nodes - 8));
  size_t args = OrtFollow(bytes, OrtField(bytes, graph, 1));
  const size_t arg_count = OrtRead32(bytes, args);
  size_t last = OrtFollow(bytes, args + 4 + 4 * (arg_count - 1));
  size_t shared_type = OrtFollow(bytes, OrtField(bytes, last, 2));
  for (size_t i = 1; i + 1 < arg_count; ++i) {
    size_t argument = OrtFollow(bytes, args + 4 + 4 * i);
    size_t type_field = OrtField(bytes, argument, 2);
    OrtWrite32(bytes, type_field, static_cast<uint32_t>(shared_type - type_field));
  }
  auto model = ReadOrt(bytes);
  EXPECT_EQ(model.graph().node()[0].op_type(), "Add");
  EXPECT_EQ(model.graph().node()[1].op_type(), "Relu");
  size_t version = OrtFollow(bytes, OrtField(bytes, OrtRead32(bytes, 0), 0));
  bytes.at(version + 4) = '6';
  EXPECT_EQ(ReadOrt(bytes).graph().node()[0].op_type(), "Add");
}

TEST(onnx_ort_parsing, RejectsMalformedOffsetsWithoutChangingDestination) {
  const auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  ModelProto destination;
  destination.set_producer_name("unchanged");
  auto reject = [&](const std::string &input) {
    utils::StringStream stream(input.data(), static_cast<int64_t>(input.size()));
    ParseOptions options;
    EXPECT_THROW(ParseModelFromOrtFlatbuffers(destination, stream, options), std::runtime_error);
    EXPECT_EQ(destination.producer_name(), "unchanged");
    EXPECT_FALSE(destination.has_graph());
  };
  reject(bytes.substr(0, 7));
  reject(bytes.substr(0, bytes.size() / 2));
  auto modified = bytes;
  OrtWrite32(modified, 0, UINT32_MAX);
  reject(modified);
  modified = bytes;
  OrtWrite32(modified, OrtRead32(modified, 0), 0x80000000u);
  reject(modified);
  modified = bytes;
  const size_t graph = OrtGraph(bytes);
  const size_t args = OrtFollow(bytes, OrtField(bytes, graph, 1));
  OrtWrite32(modified, args, UINT32_MAX);
  reject(modified);
  modified = bytes;
  OrtWrite32(modified, args + 4, 0);
  reject(modified);
  modified = bytes;
  OrtWrite32(modified, OrtField(bytes, graph, 1), UINT32_MAX);
  reject(modified);
}

TEST(onnx_ort_parsing, EnforcesOptionsAndLimits) {
  const auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  ParseOptions options;
  options.max_recursion_depth = 0;
  EXPECT_THROW(ReadOrt(bytes, options), std::runtime_error);
  options.max_recursion_depth = 2;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
  options = {};
  options.max_tensor_size_bytes = -1;
  EXPECT_THROW(ReadOrt(bytes, options), std::runtime_error);
  options.max_tensor_size_bytes = 4;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
  options.max_tensor_size_bytes = 8;
  EXPECT_EQ(ReadOrt(bytes, options).graph().initializer()[0].raw_data().size(), 8u);
  options = {};
  options.alignment = 3;
  EXPECT_THROW(ReadOrt(bytes, options), std::runtime_error);
  options = {};
  options.skip_raw_data = true;
  auto skipped = ReadOrt(bytes, options);
  EXPECT_FALSE(skipped.graph().initializer()[0].has_raw_data());
  EXPECT_EQ(skipped.graph().initializer()[0].dims()[0], 2);
}

TEST(onnx_ort_parsing, AppliesCallbacksAndMaintainsDeleterLifetime) {
  const auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  int raw_calls = 0, node_calls = 0, deletions = 0;
  ParseOptions options;
  options.raw_data_callback = [&](TensorProto &tensor, GraphProto *graph) {
    ++raw_calls;
    EXPECT_NE(graph, nullptr);
    EXPECT_EQ(tensor.name(), "W");
    EXPECT_EQ(tensor.raw_data().size(), 8u);
    return [&]() { ++deletions; };
  };
  options.node_callback = [&](NodeProto &node, GraphProto &graph) {
    ++node_calls;
    EXPECT_EQ(graph.node().size(), 2u);
    node.set_name("parsed");
  };
  {
    auto model = ReadOrt(bytes, options);
    EXPECT_EQ(raw_calls, 1);
    EXPECT_EQ(node_calls, 2);
    EXPECT_EQ(deletions, 0);
    EXPECT_EQ(model.graph().node()[0].name(), "parsed");
  }
  EXPECT_EQ(deletions, 1);
}

TEST(onnx_ort_parsing, ParsesStringTensorsAndChecksPayloadExtents) {
  auto model = OrtInputModel();
  auto *tensor = model.mutable_graph()->add_initializer();
  tensor->set_name("strings");
  tensor->set_data_type(TensorProto::STRING);
  tensor->add_dims(2);
  tensor->add_string_data("first");
  tensor->add_string_data(std::string("second\0value", 12));
  auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  auto parsed = ReadOrt(bytes);
  EXPECT_EQ(parsed.graph().initializer()[1].string_data()[1], std::string("second\0value", 12));
  size_t initializers = OrtFollow(bytes, OrtField(bytes, OrtGraph(bytes), 0));
  size_t numeric = OrtFollow(bytes, initializers + 4);
  size_t raw = OrtFollow(bytes, OrtField(bytes, numeric, 4));
  OrtWrite32(bytes, raw, 7);
  EXPECT_THROW(ReadOrt(bytes), std::runtime_error);
}

TEST(onnx_ort_parsing, RejectsExternalOffsetsAndTensorVectorOverflow) {
  const auto original = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  const size_t graph = OrtGraph(original);
  size_t initializers = OrtFollow(original, OrtField(original, graph, 0));
  size_t tensor = OrtFollow(original, initializers + 4);
  size_t vtable = tensor - OrtRead32(original, tensor);
  auto bytes = original;
  // The native writer reserves an eight-byte slot for the absent external offset.
  bytes.at(vtable + 4 + 2 * 6) = 56;
  bytes.at(vtable + 5 + 2 * 6) = 0;
  OrtWrite32(bytes, tensor + 56, 0);
  EXPECT_THROW(ReadOrt(bytes), std::runtime_error);
  bytes = original;
  size_t dims = OrtFollow(bytes, OrtField(bytes, tensor, 2));
  OrtWrite32(bytes, dims, UINT32_MAX);
  EXPECT_THROW(ReadOrt(bytes), std::runtime_error);
  bytes = original;
  OrtWrite32(bytes, dims + 4, UINT32_MAX);
  OrtWrite32(bytes, dims + 8, UINT32_MAX);
  EXPECT_THROW(ReadOrt(bytes), std::runtime_error);
}

TEST(onnx_ort_parsing, RejectsStringAllocationBeforeMaterialization) {
  auto model = OrtInputModel();
  auto *tensor = model.mutable_graph()->add_initializer();
  tensor->set_name("string");
  tensor->set_data_type(TensorProto::STRING);
  tensor->add_dims(1);
  tensor->add_string_data(std::string(256, 's'));
  const auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  ParseOptions options;
  options.max_tensor_size_bytes = 64;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
}

TEST(onnx_ort_parsing, EnforcesExactNumericPayloadBoundaryWithoutChargingMetadata) {
  auto model = OrtInputModel();
  auto *graph = model.mutable_graph();
  auto *weight = graph->mutable_initializer(0);
  weight->clr_dims();
  weight->add_dims(2);
  weight->add_dims(3);
  weight->clr_float_data();
  for (int i = 0; i < 6; ++i)
    weight->add_float_data(static_cast<float>(i));
  auto *input = graph->mutable_input(0)->mutable_type()->mutable_tensor_type();
  input->clear_shape();
  input->mutable_shape()->add_dim()->set_dim_value(1);
  input->mutable_shape()->add_dim()->set_dim_value(2);
  auto *output = graph->mutable_output(0)->mutable_type()->mutable_tensor_type();
  output->clear_shape();
  output->mutable_shape()->add_dim()->set_dim_value(1);
  output->mutable_shape()->add_dim()->set_dim_value(3);
  graph->mutable_node(0)->set_op_type("MatMul");
  const auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  ParseOptions options;
  options.max_tensor_size_bytes = 24;
  options.alignment = 64;
  const auto parsed = ReadOrt(bytes, options);
  EXPECT_EQ(parsed.graph().initializer()[0].raw_data().size(), 24u);
  options.max_tensor_size_bytes = 23;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
}

TEST(onnx_ort_parsing, EnforcesUtf8StringPayloadBoundaryWithoutChargingContainers) {
  auto model = OrtInputModel();
  auto *graph = model.mutable_graph();
  graph->clr_input();
  graph->clr_initializer();
  graph->clr_node();
  graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::STRING);
  auto *node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("strings");
  node->add_output("Y");
  auto *tensor = graph->add_initializer();
  tensor->set_name("strings");
  tensor->set_data_type(TensorProto::STRING);
  tensor->add_dims(2);
  tensor->add_string_data("abc");
  tensor->add_string_data("caf\xc3\xa9");
  const auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  ParseOptions options;
  options.max_tensor_size_bytes = 8;
  const auto parsed = ReadOrt(bytes, options);
  ASSERT_EQ(parsed.graph().initializer()[0].string_data().size(), 2u);
  EXPECT_EQ(parsed.graph().initializer()[0].string_data()[0], "abc");
  EXPECT_EQ(parsed.graph().initializer()[0].string_data()[1], "caf\xc3\xa9");
  options.max_tensor_size_bytes = 7;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
}

TEST(onnx_ort_parsing, PreservesModelWhenCallbackThrows) {
  auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  utils::StringStream stream(bytes.data(), static_cast<int64_t>(bytes.size()));
  ModelProto destination;
  destination.set_producer_name("unchanged");
  ParseOptions options;
  options.node_callback = [](NodeProto &, GraphProto &) { throw std::runtime_error("callback"); };
  EXPECT_THROW(ParseModelFromOrtFlatbuffers(destination, stream, options), std::runtime_error);
  EXPECT_EQ(destination.producer_name(), "unchanged");
  EXPECT_FALSE(destination.has_graph());
}

TEST(onnx_ort_parsing, ParsesCapturedSubgraphsAndEnforcesNestedDepth) {
  auto model = OrtInputModel();
  auto *graph = model.mutable_graph();
  graph->clr_node();
  auto *condition = graph->add_initializer();
  condition->set_name("condition");
  condition->set_data_type(TensorProto::BOOL);
  condition->add_int32_data(1);
  auto *node = graph->add_node();
  node->set_op_type("If");
  node->add_input("condition");
  node->add_output("Y");
  for (const char *name : {"then_branch", "else_branch"}) {
    auto *attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::GRAPH);
    auto *body = attribute->mutable_g();
    body->set_name(name);
    body->add_output()->CopyFrom(graph->output()[0]);
    auto *inner = body->add_node();
    inner->set_op_type("Add");
    inner->add_input("X");
    inner->add_input("W");
    inner->add_output("Y");
  }
  const auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  auto decoded = ReadOrt(bytes);
  ASSERT_EQ(decoded.graph().node()[0].attribute().size(), 2u);
  EXPECT_EQ(decoded.graph().node()[0].attribute()[0].g().node()[0].input()[1], "W");
  ParseOptions options;
  options.max_recursion_depth = 9;
  EXPECT_THROW(ReadOrt(bytes, options), onnx_light_helpers::ParseLimitExceeded);
}

TEST(onnx_ort_parsing, RejectsCyclicExecutableDependencies) {
  auto bytes = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  size_t nodes = OrtFollow(bytes, OrtField(bytes, OrtGraph(bytes), 2));
  size_t first = OrtFollow(bytes, nodes + 4);
  size_t inputs = OrtFollow(bytes, OrtField(bytes, first, 8));
  size_t outputs = OrtFollow(bytes, OrtField(bytes, first, 9));
  size_t output_name = OrtFollow(bytes, outputs + 4);
  OrtWrite32(bytes, inputs + 4, static_cast<uint32_t>(output_name - inputs - 4));
  EXPECT_THROW(ReadOrt(bytes), std::runtime_error);
}

TEST(onnx_ort_parsing, HandlesBoundedMalformedMutations) {
  const auto original = SerializeModelToOrtFlatbuffers(OrtInputModel(), {});
  uint32_t state = 7;
  ParseOptions options;
  options.max_recursion_depth = 32;
  options.max_tensor_size_bytes = 1024;
  for (size_t iteration = 0; iteration < 512; ++iteration) {
    auto bytes = original;
    state = state * 1664525u + 1013904223u;
    const size_t offset = state % (bytes.size() - 4);
    OrtWrite32(bytes, offset, iteration % 2 ? UINT32_MAX : 0);
    ModelProto destination;
    destination.set_producer_name("unchanged");
    utils::StringStream stream(bytes.data(), static_cast<int64_t>(bytes.size()));
    try {
      ParseModelFromOrtFlatbuffers(destination, stream, options);
      EXPECT_TRUE(destination.has_graph());
    } catch (const std::runtime_error &) {
      EXPECT_EQ(destination.producer_name(), "unchanged");
      EXPECT_FALSE(destination.has_graph());
    }
  }
}

TEST(onnx_ort_parsing, RejectsSharedDuplicateEdgeVectorsBeforeRepeatedTraversal) {
  constexpr size_t consumers = 256;
  constexpr size_t endpoints = 4096;
  auto bytes = SerializeModelToOrtFlatbuffers(OrtFanoutModel(consumers), {});
  const size_t edges = OrtFollow(bytes, OrtField(bytes, OrtGraph(bytes), 4));
  while (bytes.size() % 4)
    bytes.push_back('\0');
  const size_t shared = bytes.size();
  bytes.resize(bytes.size() + 4 + 12 * endpoints, '\0');
  OrtWrite32(bytes, shared, endpoints);
  // Every consumer takes A from node zero at source/destination argument zero.
  for (size_t i = 1; i <= consumers; ++i) {
    const size_t edge = OrtFollow(bytes, edges + 4 + 4 * i);
    const size_t inputs = OrtField(bytes, edge, 1);
    OrtWrite32(bytes, inputs, static_cast<uint32_t>(shared - inputs));
  }
  try {
    ReadOrt(bytes);
    FAIL() << "Expected duplicate edge rejection";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("duplicate edge endpoint"), std::string::npos);
  }
}

TEST(onnx_ort_parsing, BoundsRepeatedValidationOfSharedArgumentCountVectors) {
  constexpr size_t consumers = 192;
  constexpr size_t counts = 65536;
  auto bytes = SerializeModelToOrtFlatbuffers(OrtFanoutModel(consumers), {});
  const size_t nodes = OrtFollow(bytes, OrtField(bytes, OrtGraph(bytes), 2));
  while (bytes.size() % 4)
    bytes.push_back('\0');
  const size_t shared = bytes.size();
  bytes.resize(bytes.size() + 4 + 4 * counts, '\0');
  OrtWrite32(bytes, shared, counts);
  OrtWrite32(bytes, shared + 4, 1);
  for (size_t i = 1; i <= consumers; ++i) {
    const size_t node = OrtFollow(bytes, nodes + 4 + 4 * i);
    const size_t field = OrtField(bytes, node, 11);
    OrtWrite32(bytes, field, static_cast<uint32_t>(shared - field));
  }
  EXPECT_THROW(ReadOrt(bytes), onnx_light_helpers::ParseLimitExceeded);
}
