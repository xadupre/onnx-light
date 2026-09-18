#include "onnx.h"
#include "onnx_ort_flatbuffers.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto OrtModel() {
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(18);
  auto *graph = model.mutable_graph();
  graph->set_name("native_ort");
  auto *input = graph->add_input();
  input->set_name("X");
  auto *tensor = input->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  auto *output = graph->add_output();
  output->CopyFrom(*input);
  output->set_name("Y");
  auto *node = graph->add_node();
  node->set_op_type("Relu");
  node->add_input("X");
  node->add_output("Y");
  return model;
}

uint32_t Read32(const std::string &bytes, size_t position) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; ++i)
    value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes.at(position + i))) << (8 * i);
  return value;
}

size_t Field(const std::string &bytes, size_t table, size_t index) {
  size_t vtable = table - Read32(bytes, table);
  size_t entry = vtable + 4 + 2 * index;
  auto offset = static_cast<unsigned char>(bytes.at(entry)) |
                static_cast<unsigned char>(bytes.at(entry + 1)) << 8;
  return table + offset;
}

size_t Follow(const std::string &bytes, size_t position) {
  return position + Read32(bytes, position);
}

} // namespace

TEST(onnx_ort_serialization, NativeFlatbufferLayoutAndShape) {
  auto model = OrtModel();
  auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  ASSERT_EQ(bytes.substr(4, 4), "ORTM");
  size_t session = Read32(bytes, 0);
  size_t version = Follow(bytes, Field(bytes, session, 0));
  ASSERT_EQ(Read32(bytes, version), 1u);
  EXPECT_EQ(bytes.at(version + 4), '4');
  size_t ort_model = Follow(bytes, Field(bytes, session, 1));
  EXPECT_EQ(Read32(bytes, Field(bytes, ort_model, 0)), 9u);
  size_t graph = Follow(bytes, Field(bytes, ort_model, 7));
  EXPECT_EQ(Read32(bytes, Field(bytes, graph, 3)), 1u);
  size_t args = Follow(bytes, Field(bytes, graph, 1));
  ASSERT_EQ(Read32(bytes, args), 3u);
  // The map is ordered: omitted optional argument, X, Y.
  size_t output = Follow(bytes, args + 12);
  size_t type = Follow(bytes, Field(bytes, output, 2));
  size_t tensor = Follow(bytes, Field(bytes, type, 2));
  EXPECT_EQ(Read32(bytes, Field(bytes, tensor, 0)), static_cast<uint32_t>(TensorProto::FLOAT));
  size_t shape = Follow(bytes, Field(bytes, tensor, 1));
  size_t dims = Follow(bytes, Field(bytes, shape, 0));
  ASSERT_EQ(Read32(bytes, dims), 1u);
  size_t dimension = Follow(bytes, dims + 4);
  size_t value = Follow(bytes, Field(bytes, dimension, 0));
  EXPECT_EQ(Read32(bytes, Field(bytes, value, 1)), 2u);
}

TEST(onnx_ort_serialization, RejectsUnsupportedAndInvalidInputs) {
  auto model = OrtModel();
  SerializeOptions options;
  options.max_serialized_size_bytes = 16;
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::length_error);
  std::string output = "old output";
  EXPECT_FALSE(SerializeModelToOrtFlatbuffers(model, output, options));
  EXPECT_TRUE(output.empty());
  options.max_serialized_size_bytes = -1;
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::runtime_error);
  output = "old output";
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, output, options), std::runtime_error);
  EXPECT_TRUE(output.empty());
  options = {};
  options.skip_raw_data = true;
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::runtime_error);
  options = {};
  model.add_functions()->set_name("local");
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::invalid_argument);
  model.clr_functions();
  model.mutable_graph()->mutable_node(0)->ref_input()[0] = "missing";
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::invalid_argument);
}

TEST(onnx_ort_serialization, ValidatesInitializerPayload) {
  auto model = OrtModel();
  auto *tensor = model.mutable_graph()->add_initializer();
  tensor->set_name("weight");
  tensor->set_data_type(TensorProto::FLOAT);
  tensor->add_dims(2);
  tensor->add_float_data(1.0f);
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::invalid_argument);
  tensor->add_float_data(2.0f);
  EXPECT_NO_THROW(SerializeModelToOrtFlatbuffers(model, {}));
  tensor->clr_float_data();
  tensor->set_data_location(TensorProto::EXTERNAL);
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::runtime_error);
  tensor->ref_raw_data() = std::vector<uint8_t>(8);
  tensor->ref_raw_data().resize(0);
  ASSERT_TRUE(tensor->has_raw_data());
  ASSERT_TRUE(tensor->raw_data().empty());
  try {
    SerializeModelToOrtFlatbuffers(model, {});
    FAIL() << "Expected unloaded external tensor rejection";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("external tensor payload must be loaded"),
              std::string::npos);
  }
}

TEST(onnx_ort_serialization, CallbackDoesNotMutateModel) {
  auto model = OrtModel();
  SerializeOptions options;
  options.node_callback = [](NodeProto &node, GraphProto &) { node.set_name("callback"); };
  auto bytes = SerializeModelToOrtFlatbuffers(model, options);
  EXPECT_NE(bytes.find("callback"), std::string::npos);
  EXPECT_TRUE(model.graph().node(0).name().empty());
}

TEST(onnx_ort_serialization, RawCallbackInlinesLoadedExternalTensor) {
  auto model = OrtModel();
  auto *tensor = model.mutable_graph()->add_initializer();
  tensor->set_name("weight");
  tensor->set_data_type(TensorProto::FLOAT);
  tensor->add_dims(1);
  tensor->set_data_location(TensorProto::EXTERNAL);
  tensor->ref_raw_data() = std::vector<uint8_t>{0, 0, 128, 63};
  int calls = 0;
  SerializeOptions options;
  options.raw_data_callback = [&](TensorProto &, GraphProto *graph, uint8_t *data, size_t size,
                                  bool size_only) -> int64_t {
    ++calls;
    EXPECT_NE(graph, nullptr);
    if (!size_only) {
      EXPECT_EQ(size, 4u);
      data[0] = 0;
      data[1] = 0;
      data[2] = 0;
      data[3] = 64;
    }
    return 4;
  };
  auto bytes = SerializeModelToOrtFlatbuffers(model, options);
  EXPECT_EQ(calls, 2);
  size_t session = Read32(bytes, 0);
  size_t ort_model = Follow(bytes, Field(bytes, session, 1));
  size_t graph = Follow(bytes, Field(bytes, ort_model, 7));
  size_t initializers = Follow(bytes, Field(bytes, graph, 0));
  size_t initializer = Follow(bytes, initializers + 4);
  size_t data = Follow(bytes, Field(bytes, initializer, 4));
  EXPECT_EQ(Read32(bytes, data + 4), 0x40000000u);
  EXPECT_EQ(tensor->data_location(), TensorProto::EXTERNAL);
  EXPECT_EQ(tensor->raw_data()[2], 128);
}

TEST(onnx_ort_serialization, InfersIntermediateTypesAndWritesEdges) {
  auto model = OrtModel();
  auto *graph = model.mutable_graph();
  graph->mutable_node(0)->ref_output()[0] = "intermediate";
  auto *node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("intermediate");
  node->add_output("Y");
  auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  size_t session = Read32(bytes, 0);
  size_t ort_model = Follow(bytes, Field(bytes, session, 1));
  size_t ort_graph = Follow(bytes, Field(bytes, ort_model, 7));
  size_t edges = Follow(bytes, Field(bytes, ort_graph, 4));
  ASSERT_EQ(Read32(bytes, edges), 2u);
  size_t first = Follow(bytes, edges + 4);
  size_t outgoing = Follow(bytes, Field(bytes, first, 2));
  ASSERT_EQ(Read32(bytes, outgoing), 1u);
  EXPECT_EQ(Read32(bytes, outgoing + 4), 1u);
  EXPECT_EQ(Read32(bytes, outgoing + 8), 0u);
  EXPECT_EQ(Read32(bytes, outgoing + 12), 0u);
  graph->mutable_node(0)->set_op_type("Unrecognized");
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::invalid_argument);
}

TEST(onnx_ort_serialization, LiftsConstantsWithoutMutatingGraph) {
  auto model = OrtModel();
  auto *graph = model.mutable_graph();
  graph->clr_node();
  auto *node = graph->add_node();
  node->set_op_type("Constant");
  node->add_output("Y");
  auto *attr = node->add_attribute();
  attr->set_name("value_floats");
  attr->set_type(AttributeProto::FLOATS);
  attr->add_floats(1.0f);
  attr->add_floats(2.0f);
  auto bytes = SerializeModelToOrtFlatbuffers(model, {});
  size_t session = Read32(bytes, 0);
  size_t ort_model = Follow(bytes, Field(bytes, session, 1));
  size_t ort_graph = Follow(bytes, Field(bytes, ort_model, 7));
  size_t initializers = Follow(bytes, Field(bytes, ort_graph, 0));
  EXPECT_EQ(Read32(bytes, initializers), 1u);
  size_t nodes = Follow(bytes, Field(bytes, ort_graph, 2));
  EXPECT_EQ(Read32(bytes, nodes), 0u);
  EXPECT_TRUE(model.graph().initializer().empty());
  EXPECT_EQ(model.graph().node().size(), 1u);
}

TEST(onnx_ort_serialization, AlignsRawAndTypedTensorPayloads) {
  auto model = OrtModel();
  auto *tensor = model.mutable_graph()->add_initializer();
  tensor->set_name("weight");
  tensor->set_data_type(TensorProto::FLOAT);
  tensor->add_dims(2);
  tensor->add_float_data(1.0f);
  tensor->add_float_data(2.0f);
  SerializeOptions options;
  options.alignment = 4096;
  for (bool raw : {false, true}) {
    if (raw) {
      tensor->clr_float_data();
      tensor->ref_raw_data() = std::vector<uint8_t>{0, 0, 128, 63, 0, 0, 0, 64};
    }
    auto bytes = SerializeModelToOrtFlatbuffers(model, options);
    size_t session = Read32(bytes, 0);
    size_t ort_model = Follow(bytes, Field(bytes, session, 1));
    size_t ort_graph = Follow(bytes, Field(bytes, ort_model, 7));
    size_t initializers = Follow(bytes, Field(bytes, ort_graph, 0));
    size_t initializer = Follow(bytes, initializers + 4);
    size_t data = Follow(bytes, Field(bytes, initializer, 4));
    EXPECT_EQ((data + 4) % 4096, 0u);
    EXPECT_EQ(Read32(bytes, data), 8u);
  }
  options.alignment = 3;
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, options), std::runtime_error);
}

TEST(onnx_ort_serialization, PreservesOverridableInitializerDeclaredShape) {
  auto model = OrtModel();
  auto *graph = model.mutable_graph();
  auto *weight = graph->add_initializer();
  weight->set_name("X");
  weight->set_data_type(TensorProto::FLOAT);
  weight->add_dims(2);
  weight->add_float_data(1.0f);
  weight->add_float_data(2.0f);
  auto *type = graph->mutable_input(0)->mutable_type()->mutable_tensor_type();
  for (int kind = 0; kind < 3; ++kind) {
    type->clear_shape();
    if (kind != 0) {
      auto *dim = type->mutable_shape()->add_dim();
      if (kind == 2)
        dim->set_dim_param("batch");
    }
    graph->mutable_output(0)->mutable_type()->CopyFrom(graph->input(0).type());
    auto bytes = SerializeModelToOrtFlatbuffers(model, {});
    size_t session = Read32(bytes, 0);
    size_t ort_model = Follow(bytes, Field(bytes, session, 1));
    size_t ort_graph = Follow(bytes, Field(bytes, ort_model, 7));
    size_t args = Follow(bytes, Field(bytes, ort_graph, 1));
    size_t input = Follow(bytes, args + 8);
    size_t value_type = Follow(bytes, Field(bytes, input, 2));
    size_t tensor_type = Follow(bytes, Field(bytes, value_type, 2));
    if (kind == 0) {
      EXPECT_EQ(Field(bytes, tensor_type, 1), tensor_type);
    } else {
      size_t shape = Follow(bytes, Field(bytes, tensor_type, 1));
      size_t dims = Follow(bytes, Field(bytes, shape, 0));
      ASSERT_EQ(Read32(bytes, dims), 1u);
      size_t dim = Follow(bytes, dims + 4);
      size_t value = Follow(bytes, Field(bytes, dim, 0));
      EXPECT_EQ(Read32(bytes, Field(bytes, value, 0)), kind == 1 ? 0u : 2u);
      if (kind == 2) {
        size_t parameter = Follow(bytes, Field(bytes, value, 2));
        EXPECT_EQ(bytes.substr(parameter + 4, Read32(bytes, parameter)), "batch");
      }
    }
  }
  type->set_elem_type(TensorProto::DOUBLE);
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::invalid_argument);
  type->set_elem_type(TensorProto::FLOAT);
  type->clear_shape();
  type->mutable_shape()->add_dim()->set_dim_value(3);
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::invalid_argument);
  type->mutable_shape()->mutable_dim(0)->set_dim_value(2);
  type->mutable_shape()->add_dim()->set_dim_value(1);
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(model, {}), std::invalid_argument);
}
