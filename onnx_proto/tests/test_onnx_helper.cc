#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

TEST(onnx_helper, MakePipelineCopiesExistingProtos) {
  ModelProto prefill;
  prefill.set_producer_name("prefill");
  ModelProto decode;
  decode.set_producer_name("decode");
  ModelProto embedding;
  embedding.set_producer_name("embedding");
  PipelineProto pipeline = MakePipeline(prefill, &decode, &embedding);

  ASSERT_TRUE(pipeline.has_prefill());
  ASSERT_TRUE(pipeline.has_decode());
  ASSERT_TRUE(pipeline.has_embedding());
  EXPECT_EQ(pipeline.ref_prefill().ref_producer_name(), "prefill");
  EXPECT_EQ(pipeline.ref_decode().ref_producer_name(), "decode");
  EXPECT_EQ(pipeline.ref_embedding().ref_producer_name(), "embedding");

  prefill.set_producer_name("changed");
  EXPECT_EQ(pipeline.ref_prefill().ref_producer_name(), "prefill");

  PipelineProto prefill_only = MakePipeline(prefill);
  EXPECT_TRUE(prefill_only.has_prefill());
  EXPECT_FALSE(prefill_only.has_decode());
  EXPECT_FALSE(prefill_only.has_embedding());
}

TEST(onnx_helper, IteratorTensorProto) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  IteratorTensorProto itp(&model.ref_graph());
  std::vector<uint8_t> dt;
  while (itp.next()) {
    dt.push_back(itp->ref_raw_data()[0]);
  }
  EXPECT_EQ(dt.size(), 2);
  EXPECT_EQ(dt[0], 2);
  EXPECT_EQ(dt[1], 1);
}

TEST(onnx_helper, IteratorTensorProto_ModelFunctionsAndRewritings) {
  ModelProto model;
  std::vector<TensorProto *> tensors;
  const auto add_tensor = [&](TensorProto *tensor) {
    tensor->set_name("tensor_" + std::to_string(tensors.size()));
    tensor->set_data_type(TensorProto::DataType::FLOAT);
    tensor->ref_dims().push_back(1);
    for (int i = 0; i < 4; ++i)
      tensor->ref_raw_data().push_back(static_cast<uint8_t>(tensors.size() + 1));
    tensors.push_back(tensor);
  };
  add_tensor(model.ref_graph().add_initializer());
  auto *function = model.add_functions();
  add_tensor(function->add_node()->add_attribute()->add_t());
  add_tensor(function->add_attribute_proto()->add_t());
  auto &plan = model.ref_rewritings()["optimized"];
  auto *edit = plan.add_rewrites();
  add_tensor(edit->add_added_initializers());
  add_tensor(edit->add_added_nodes()->add_attribute()->add_t());
  add_tensor(edit->add_initializer_calls()->add_attribute()->add_t());
  auto *preparation = plan.add_added_functions();
  add_tensor(preparation->add_node()->add_attribute()->add_t());
  add_tensor(preparation->add_attribute_proto()->add_g()->add_initializer());
  auto *tensor_list = function->add_attribute_proto();
  add_tensor(tensor_list->add_tensors());
  add_tensor(tensor_list->add_tensors());
  auto *graphs = function->add_node()->add_attribute();
  add_tensor(graphs->add_graphs()->add_initializer());
  add_tensor(graphs->add_graphs()->add_node()->add_attribute()->add_t());
  auto *preparation_tensors = preparation->add_node()->add_attribute();
  add_tensor(preparation_tensors->add_tensors());
  auto *preparation_graphs = preparation->add_attribute_proto();
  add_tensor(preparation_graphs->add_graphs()->add_initializer());
  add_tensor(preparation_graphs->add_graphs()->add_initializer());
  const auto add_constant = [&](StructTypeProto &type) {
    add_tensor(type.ref_structure().add_field()->add_constant());
  };
  add_constant(model.add_struct_types()->ref_array().ref_element_type().ref_struct_type());
  add_constant(
      model.ref_struct_types()[0].ref_decoder().add_attribute_proto()->ref_tp().ref_struct_type());
  add_tensor(model.ref_struct_types()[0].ref_encoder().add_node()->add_attribute()->add_t());
  add_constant(function->add_value_info()->ref_type().ref_struct_type());
  add_constant(model.ref_graph().add_input()->ref_type().ref_struct_type());
  add_constant(model.ref_graph()
                   .add_output()
                   ->ref_type()
                   .ref_sequence_type()
                   .ref_elem_type()
                   .ref_struct_type());
  add_constant(model.ref_graph()
                   .add_value_info()
                   ->ref_type()
                   .ref_optional_type()
                   .ref_elem_type()
                   .ref_struct_type());
  add_constant(function->add_attribute_proto()->add_type_protos()->ref_struct_type());
  auto &encoded = model.ref_graph().add_extended_initializer()->ref_encoded_value();
  add_tensor(encoded.ref_affine().add_scale());
  add_tensor(encoded.ref_affine().add_zero_point());
  add_constant(encoded.ref_logical_type().ref_struct_type());
  add_constant(preparation_graphs->ref_graphs()[0]
                   .add_extended_initializer()
                   ->ref_encoded_value()
                   .ref_struct_type());

  IteratorTensorProto iterator(&model);
  std::unordered_map<TensorProto *, size_t> visits;
  while (iterator.next())
    ++visits[&*iterator];
  ASSERT_EQ(visits.size(), tensors.size());
  for (auto *tensor : tensors)
    EXPECT_EQ(visits.at(tensor), 1);

  std::string original;
  model.SerializeToString(original);
  EXPECT_EQ(PopulateExternalData(model, 0, "weights.bin"), 4 * tensors.size());
  for (auto *tensor : tensors) {
    EXPECT_EQ(tensor->ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    ASSERT_EQ(tensor->ref_external_data().size(), 3);
    EXPECT_EQ(tensor->ref_external_data()[0].ref_value(), "weights.bin");
  }
  ClearExternalData(model);
  std::string restored;
  model.SerializeToString(restored);
  EXPECT_EQ(restored, original);

  SerializeOptions options;
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  options.raw_data_threshold = 1024;
  ASSERT_TRUE(model.SerializeToString(serialized, external_files, 8, "weights", options));
  EXPECT_TRUE(external_files.empty());
  EXPECT_EQ(serialized, original);
  options.raw_data_threshold = 0;
  ASSERT_TRUE(model.SerializeToString(serialized, external_files, 8, "weights", options));
  ASSERT_EQ(external_files.size(), (tensors.size() + 1) / 2);
  ModelProto parsed;
  parsed.ParseFromString(serialized);
  IteratorTensorProto loaded(&parsed);
  size_t count = 0;
  while (loaded.next()) {
    ASSERT_EQ(loaded->ref_external_data().size(), 3);
    const auto &data = loaded->ref_external_data();
    const auto &payload = external_files.at(data[0].ref_value());
    const size_t offset = std::stoull(data[1].ref_value());
    ASSERT_LE(offset + 4, payload.size());
    const uint8_t expected = static_cast<uint8_t>(std::stoul(loaded->ref_name().substr(7)) + 1);
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(static_cast<uint8_t>(payload[offset + i]), expected);
    ++count;
  }
  EXPECT_EQ(count, tensors.size());
}

TEST(onnx_helper, IteratorTensorProto_AttributeLayoutOrder) {
  ModelProto model;
  auto &graph = model.ref_graph();
  auto *attribute = graph.add_node()->add_attribute();
  std::vector<std::pair<TensorProto *, GraphProto *>> expected;
  const auto add_tensor = [&](TensorProto *tensor, GraphProto *owner) {
    tensor->set_data_type(TensorProto::UINT8);
    tensor->set_raw_data(std::vector<uint8_t>{static_cast<uint8_t>(expected.size())});
    expected.emplace_back(tensor, owner);
  };
  add_tensor(attribute->add_t(), &graph);
  add_tensor(&attribute->ref_sparse_tensor().ref_values(), &graph);
  add_tensor(&attribute->ref_sparse_tensor().ref_indices(), &graph);
  auto &subgraph = attribute->ref_g();
  add_tensor(subgraph.add_initializer(), &subgraph);
  auto &type = attribute->ref_tp().ref_struct_type();
  auto *field = type.ref_structure().add_field();
  add_tensor(field->add_constant(), &graph);
  auto &decoder = type.ref_decoder();
  add_tensor(decoder.add_attribute_proto()->add_t(), nullptr);
  auto &decoder_graph = decoder.add_node()->add_attribute()->ref_g();
  add_tensor(decoder_graph.add_initializer(), &decoder_graph);
  add_tensor(attribute->add_tensors(), &graph);
  auto *sparse = attribute->add_sparse_tensors();
  add_tensor(&sparse->ref_values(), &graph);
  add_tensor(&sparse->ref_indices(), &graph);
  auto *repeated_graph = attribute->add_graphs();
  add_tensor(repeated_graph->add_initializer(), repeated_graph);
  add_tensor(
      attribute->add_type_protos()->ref_struct_type().ref_structure().add_field()->add_constant(),
      &graph);
  add_tensor(graph.add_initializer(), &graph);

  IteratorTensorProto iterator(&model);
  for (const auto &[tensor, owner] : expected) {
    ASSERT_TRUE(iterator.next());
    EXPECT_EQ(&*iterator, tensor);
    EXPECT_EQ(iterator.graph(), owner);
  }
  EXPECT_FALSE(iterator.next());
  EXPECT_EQ(PopulateExternalData(model, 0, "weights.bin"), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(expected[i].first->external_data()[1].value(), std::to_string(i));
  ClearExternalData(model);
}

TEST(onnx_helper, ExternalData_LayoutTensors) {
  for (bool affine : {false, true}) {
    for (bool use_locations : {false, true}) {
      for (int64_t alignment :
           (use_locations ? std::vector<int64_t>{0, 16} : std::vector<int64_t>{0})) {
        SCOPED_TRACE(::testing::Message() << "affine=" << affine << " use_locations="
                                          << use_locations << " alignment=" << alignment);
        ModelProto model;
        auto *weights = model.ref_graph().add_initializer();
        weights->set_name("W");
        weights->set_data_type(TensorProto::FLOAT);
        weights->add_dims(128);
        weights->set_raw_data(std::vector<uint8_t>(512, 7));
        auto &value = model.ref_graph().add_extended_initializer()->ref_encoded_value();
        value.set_name("encoded");
        const std::vector<uint8_t> encoded_raw(128, 3);
        value.set_raw_data(encoded_raw);
        std::vector<TensorProto *> tensors{weights};
        if (affine) {
          value.ref_affine().set_storage_type(TensorProto::UINT8);
          tensors.push_back(value.ref_affine().add_scale());
          tensors.push_back(value.ref_affine().add_zero_point());
          tensors[1]->set_data_type(TensorProto::FLOAT);
          tensors[1]->set_raw_data(std::vector<uint8_t>{0, 0, 128, 62});
          tensors[2]->set_data_type(TensorProto::UINT8);
          tensors[2]->set_raw_data(std::vector<uint8_t>{17});
        } else {
          tensors.push_back(value.ref_struct_type().ref_structure().add_field()->add_constant());
          tensors[1]->set_data_type(TensorProto::FLOAT);
          tensors[1]->add_dims(128);
          tensors[1]->set_raw_data(std::vector<uint8_t>(512, 9));
        }
        std::string original;
        ASSERT_TRUE(model.SerializeToString(original));
        IteratorTensorProto iterator(&model.ref_graph());
        for (auto *tensor : tensors) {
          ASSERT_TRUE(iterator.next());
          EXPECT_EQ(&*iterator, tensor);
          EXPECT_EQ(iterator.graph(), &model.ref_graph());
        }
        EXPECT_FALSE(iterator.next());

        SerializeOptions options;
        options.raw_data_threshold = 0;
        options.use_external_data_location = use_locations;
        options.alignment = alignment;
        const auto tmpdir =
            std::filesystem::temp_directory_path() /
            ("onnx_light_layout_export_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(tmpdir);
        const auto model_path = (tmpdir / "model.onnx").string();
        const auto weights_path = (tmpdir / "weights.bin").string();
        {
          utils::TwoFilesWriteStream stream(model_path, weights_path);
          ASSERT_TRUE(SerializeModelProtoToStream(model, stream, options));
        }
        ModelProto parsed;
        {
          utils::FileStream stream(model_path);
          ParseOptions parse_options;
          ASSERT_TRUE(parsed.ParseFromStream(stream, parse_options));
        }
        EXPECT_EQ(parsed.ref_graph().ref_extended_initializer()[0].ref_encoded_value().raw_data(),
                  value.raw_data());
        IteratorTensorProto loaded(&parsed);
        for (auto *tensor : tensors) {
          ASSERT_TRUE(loaded.next());
          EXPECT_EQ(loaded->data_location(), TensorProto::EXTERNAL);
          ASSERT_EQ(loaded->external_data().size(), 3);
          const auto &metadata = loaded->external_data();
          const size_t offset = std::stoull(metadata[1].value());
          EXPECT_EQ(offset % (alignment == 0 ? 1 : alignment), 0);
          EXPECT_EQ(std::stoull(metadata[2].value()), tensor->raw_data().size());
          std::ifstream file(tmpdir / metadata[0].value().c_str(), std::ios::binary);
          file.seekg(offset);
          std::vector<uint8_t> payload(tensor->raw_data().size());
          file.read(reinterpret_cast<char *>(payload.data()), payload.size());
          ASSERT_TRUE(file);
          EXPECT_EQ(payload,
                    std::vector<uint8_t>(tensor->raw_data().data(),
                                         tensor->raw_data().data() + tensor->raw_data().size()));
        }
        EXPECT_FALSE(loaded.next());
        LoadExternalDataForModel(parsed, tmpdir.string());
        IteratorTensorProto reloaded(&parsed);
        for (auto *tensor : tensors) {
          ASSERT_TRUE(reloaded.next());
          EXPECT_EQ(reloaded->raw_data(), tensor->raw_data());
          EXPECT_FALSE(reloaded->has_external_data());
          EXPECT_EQ(reloaded->data_location(), TensorProto::DEFAULT);
        }
        EXPECT_FALSE(reloaded.next());
        std::string restored;
        ASSERT_TRUE(model.SerializeToString(restored));
        EXPECT_EQ(restored, original);
        std::filesystem::remove_all(tmpdir);
      }
    }
  }
}

TEST(onnx_helper, IteratorTensorProto_NestedGraph) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto *nodeg = graph->add_node();
  nodeg->set_name("test_graph");
  nodeg->set_op_type("If");
  AttributeProto *attrg = nodeg->add_attribute();
  attrg->set_name("bias");
  GraphProto *nested = attrg->add_g();

  TensorProto *weights2 = nested->add_initializer();
  weights2->set_name("weights");
  weights2->set_data_type(TensorProto::DataType::FLOAT);
  weights2->ref_dims().push_back(1);
  weights2->ref_dims().push_back(1);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);

  NodeProto *node2 = nested->add_node();
  node2->set_name("test_node");
  node2->set_op_type("Add");
  AttributeProto *attr2 = node2->add_attribute();
  attr2->set_name("bias");
  TensorProto &biasw2 = attr2->ref_t();
  biasw.set_name("biasw");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  IteratorTensorProto itp(&model.ref_graph());
  std::vector<uint8_t> dt;
  while (itp.next()) {
    dt.push_back(itp->ref_raw_data()[0]);
  }
  EXPECT_EQ(dt.size(), 4);
  EXPECT_EQ(dt[0], 2);
  EXPECT_EQ(dt[1], 4);
  EXPECT_EQ(dt[2], 3);
  EXPECT_EQ(dt[3], 1);
}

TEST(onnx_helper, IteratorTensorProto_ExternalData) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto *nodeg = graph->add_node();
  nodeg->set_name("test_graph");
  nodeg->set_op_type("If");
  AttributeProto *attrg = nodeg->add_attribute();
  attrg->set_name("bias");
  GraphProto *nested = attrg->add_g();

  TensorProto *weights2 = nested->add_initializer();
  weights2->set_name("weights");
  weights2->set_data_type(TensorProto::DataType::FLOAT);
  weights2->ref_dims().push_back(1);
  weights2->ref_dims().push_back(1);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);

  NodeProto *node2 = nested->add_node();
  node2->set_name("test_node");
  node2->set_op_type("Add");
  AttributeProto *attr2 = node2->add_attribute();
  attr2->set_name("bias");
  TensorProto &biasw2 = attr2->ref_t();
  biasw.set_name("biasw");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  PopulateExternalData(model, 2, "external_data.bin");

  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    EXPECT_TRUE(it->has_external_data());
    EXPECT_EQ(it->ref_external_data().size(), 3);
  }

  ClearExternalData(model);
  while (it.next()) {
    EXPECT_FALSE(it->has_external_data());
  }
}

TEST(onnx_helper, ConvertModelToExternalData_AllToOneFile) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *small_w = graph->add_initializer();
  small_w->set_name("small");
  small_w->set_data_type(TensorProto::DataType::FLOAT);
  small_w->ref_raw_data().resize(8);

  TensorProto *big_w = graph->add_initializer();
  big_w->set_name("big");
  big_w->set_data_type(TensorProto::DataType::FLOAT);
  big_w->ref_raw_data().resize(64);

  ConvertModelToExternalData(model, /*all_tensors_to_one_file=*/true,
                             /*location=*/"weights.bin", /*size_threshold=*/16,
                             /*convert_attribute=*/false);

  EXPECT_FALSE(small_w->has_external_data());
  EXPECT_TRUE(big_w->has_external_data());
  ASSERT_EQ(big_w->ref_external_data().size(), 1);
  EXPECT_EQ(std::string(big_w->ref_external_data()[0].ref_key()), "location");
  EXPECT_EQ(std::string(big_w->ref_external_data()[0].ref_value()), "weights.bin");
  EXPECT_EQ(big_w->ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  // raw_data is preserved (the bytes are flushed by a later save call).
  EXPECT_EQ(big_w->ref_raw_data().size(), 64u);
}

TEST(onnx_helper, ConvertModelToExternalData_PerTensorFile) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(64);

  ConvertModelToExternalData(model, /*all_tensors_to_one_file=*/false,
                             /*location=*/"", /*size_threshold=*/0,
                             /*convert_attribute=*/false);

  ASSERT_EQ(t->ref_external_data().size(), 1);
  EXPECT_EQ(std::string(t->ref_external_data()[0].ref_value()), "W");
}

TEST(onnx_helper, ConvertModelToExternalData_AbsoluteLocationRejected) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(64);

#if defined(_WIN32)
  const std::string abs_location = "C:/tmp/forbidden.bin";
#else
  const std::string abs_location = "/tmp/forbidden.bin";
#endif
  EXPECT_THROW(ConvertModelToExternalData(model, true, abs_location, 0, false),
               std::invalid_argument);
}

TEST(onnx_helper, ConvertModelToExternalData_ExistingLocationRejected) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(64);

  std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / "onnx_light_convert_existing.bin";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << "x";
  }
  const std::string existing = tmp.string();
  EXPECT_THROW(ConvertModelToExternalData(model, true, existing, 0, false),
               ExternalDataLocationExistsError);
  std::filesystem::remove(tmp);
}

TEST(onnx_helper, LoadExternalDataForModel_RoundTrip) {
  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path() / "onnx_light_load_external_roundtrip";
  std::filesystem::create_directories(tmpdir);
  const std::string ext_name = "weights.bin";
  std::filesystem::path weights_path = tmpdir / ext_name;
  std::vector<uint8_t> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  {
    std::ofstream out(weights_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(bytes.size());
  std::memcpy(t->ref_raw_data().data(), bytes.data(), bytes.size());

  ConvertModelToExternalData(model, true, ext_name, 0, false);
  EXPECT_TRUE(t->has_external_data());
  EXPECT_EQ(t->ref_data_location(), TensorProto::DataLocation::EXTERNAL);

  // Simulate post-save state: drop inline raw_data.
  t->ref_raw_data().resize(0);

  LoadExternalDataForModel(model, tmpdir.string());
  EXPECT_FALSE(t->has_external_data());
  EXPECT_EQ(t->ref_data_location(), TensorProto::DataLocation::DEFAULT);
  ASSERT_EQ(t->ref_raw_data().size(), bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_EQ(t->ref_raw_data().data()[i], bytes[i]);
  }
  std::filesystem::remove_all(tmpdir);
}

TEST(onnx_helper, LoadExternalDataForModel_SymlinkRejected) {
#if !defined(_WIN32)
  namespace fs = std::filesystem;
  fs::path tmpdir = fs::temp_directory_path() / "onnx_light_load_external_symlink";
  fs::remove_all(tmpdir);
  fs::create_directories(tmpdir);

  // Create a real target file and a symlink pointing to it inside tmpdir.
  std::vector<uint8_t> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  fs::path target_path = tmpdir / "target.bin";
  {
    std::ofstream out(target_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  const std::string ext_name = "weights.bin";
  fs::create_symlink(target_path, tmpdir / ext_name);

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(bytes.size());
  std::memcpy(t->ref_raw_data().data(), bytes.data(), bytes.size());

  ConvertModelToExternalData(model, true, ext_name, 0, false);
  t->ref_raw_data().resize(0);

  // The external data location is a symbolic link and must be rejected.
  try {
    LoadExternalDataForModel(model, tmpdir.string());
    FAIL() << "Expected LoadExternalDataForModel to reject the symbolic link.";
  } catch (const std::runtime_error &ex) {
    EXPECT_NE(std::string(ex.what()).find("symbolic link"), std::string::npos)
        << "Unexpected error message: " << ex.what();
  }

  fs::remove_all(tmpdir);
#endif
}

TEST(onnx_helper, LoadExternalDataForModel_ParentDirSymlinkRejected) {
#if !defined(_WIN32)
  namespace fs = std::filesystem;
  fs::path tmpdir = fs::temp_directory_path() / "onnx_light_load_external_parent_symlink";
  fs::remove_all(tmpdir);
  fs::create_directories(tmpdir);

  // Create a target directory outside tmpdir holding sensitive data.
  fs::path outside = fs::temp_directory_path() / "onnx_light_load_external_outside";
  fs::remove_all(outside);
  fs::create_directories(outside);
  std::vector<uint8_t> bytes{9, 8, 7, 6};
  {
    std::ofstream out(outside / "secret.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }

  // A directory symlink inside tmpdir that points outside.
  fs::create_directory_symlink(outside, tmpdir / "subdir");

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("W");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_raw_data().resize(bytes.size());
  std::memcpy(t->ref_raw_data().data(), bytes.data(), bytes.size());

  // "subdir/secret.bin" resolves outside tmpdir through the parent symlink.
  // Set the external_data metadata directly to exercise LoadExternalData's
  // canonical containment check with a directory component.
  t->ref_raw_data().resize(0);
  t->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  StringStringEntryProto *loc = t->add_external_data();
  loc->set_key("location");
  loc->set_value("subdir/secret.bin");
  StringStringEntryProto *off = t->add_external_data();
  off->set_key("offset");
  off->set_value("0");
  StringStringEntryProto *len = t->add_external_data();
  len->set_key("length");
  len->set_value("4");

  try {
    LoadExternalDataForModel(model, tmpdir.string());
    FAIL() << "Expected LoadExternalDataForModel to reject the parent-dir symlink escape.";
  } catch (const std::runtime_error &ex) {
    EXPECT_NE(std::string(ex.what()).find("outside the base directory"), std::string::npos)
        << "Unexpected error message: " << ex.what();
  }

  fs::remove_all(tmpdir);
  fs::remove_all(outside);
#endif
}

TEST(onnx_helper, SerializeModelProtoToStream) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto *nodeg = graph->add_node();
  nodeg->set_name("test_graph");
  nodeg->set_op_type("If");
  AttributeProto *attrg = nodeg->add_attribute();
  attrg->set_name("bias");
  GraphProto *nested = attrg->add_g();

  TensorProto *weights2 = nested->add_initializer();
  weights2->set_name("weights2");
  weights2->set_data_type(TensorProto::DataType::FLOAT);
  weights2->ref_dims().push_back(1);
  weights2->ref_dims().push_back(1);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);

  NodeProto *node2 = nested->add_node();
  node2->set_name("test_node");
  node2->set_op_type("Add");
  AttributeProto *attr2 = node2->add_attribute();
  attr2->set_name("bias");
  TensorProto &biasw2 = attr2->ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  SerializeOptions options;
  options.raw_data_threshold = 2;
  utils::TwoFilesWriteStream stream("SerializeModelProtoToStream.onnx",
                                    "SerializeModelProtoToStream.data");
  SerializeModelProtoToStream(model, stream, options);
}

TEST(onnx_helper, SerializeModelProtoToStream_DoesNotMutateModel) {
  ModelProto model;
  const int64_t external_data_threshold = 0;
  const int64_t max_external_file_size = 8;

  GraphProto *model_graph = model.add_graph();
  model_graph->set_name("g");

  for (int i = 0; i < 3; ++i) {
    TensorProto *weights = model_graph->add_initializer();
    const std::vector<uint8_t> tensor_raw_data{1, 2, static_cast<uint8_t>(3 + i), 4};
    weights->set_name("weights" + std::to_string(i));
    weights->set_data_type(TensorProto::DataType::UINT8);
    weights->add_dims(4);
    weights->set_raw_data(tensor_raw_data);
  }

  std::string serialized_before_two_file_write;
  EXPECT_NO_THROW(model.SerializeToString(serialized_before_two_file_write));
  ASSERT_FALSE(serialized_before_two_file_write.empty());

  const std::string model_path = "SerializeModelProtoToStream_DoesNotMutateModel.onnx";
  const std::string weights_path = "SerializeModelProtoToStream_DoesNotMutateModel.data";
  {
    utils::TwoFilesWriteStream stream(model_path, weights_path);
    SerializeOptions options;
    options.raw_data_threshold = external_data_threshold;
    options.max_external_file_size = max_external_file_size;
    SerializeModelProtoToStream(model, stream, options);
  }

  EXPECT_TRUE(std::filesystem::exists(model_path));
  EXPECT_TRUE(std::filesystem::exists(weights_path));

  std::string serialized_after_two_file_write;
  EXPECT_NO_THROW(model.SerializeToString(serialized_after_two_file_write));
  EXPECT_EQ(serialized_before_two_file_write, serialized_after_two_file_write);

  IteratorTensorProto tensor_it(model_graph);
  while (tensor_it.next()) {
    EXPECT_TRUE(tensor_it->has_raw_data());
    EXPECT_FALSE(tensor_it->has_external_data());
    EXPECT_FALSE(tensor_it->has_data_location());
  }

  EXPECT_EQ(std::remove(model_path.c_str()), 0);
  EXPECT_EQ(std::remove(weights_path.c_str()), 0);
  const std::string second_weights_path = weights_path + ".1";
  const bool has_second_weights_file = std::filesystem::exists(second_weights_path);
  EXPECT_TRUE(has_second_weights_file);
  if (has_second_weights_file) {
    EXPECT_EQ(std::remove(second_weights_path.c_str()), 0);
  }
}

TEST(onnx_external_ressource, SaveWithExternalData) {
  // Build a model with a large weight in-memory, then serialize it to two files
  // (model + external weights) and verify the external weights land in the
  // separate data file.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("big_weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(256);
  weights->ref_raw_data() = std::vector<uint8_t>(256 * sizeof(float), 7);

  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path() /
      ("onnx_light_save_with_external_data_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::remove_all(tmpdir);
  std::filesystem::create_directories(tmpdir);

  auto serialized = (tmpdir / "test_onnx_file_save_with_external_data.onnx").string();
  auto weights_file = (tmpdir / "test_onnx_file_save_with_external_data.data").string();
  {
    utils::TwoFilesWriteStream wstream(serialized, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }
  std::error_code model_size_error;
  auto size = std::filesystem::file_size(serialized, model_size_error);
  EXPECT_FALSE(static_cast<bool>(model_size_error)) << model_size_error.message();
  std::error_code weights_size_error;
  auto weights_size = std::filesystem::file_size(weights_file, weights_size_error);
  EXPECT_FALSE(static_cast<bool>(weights_size_error)) << weights_size_error.message();
  EXPECT_GT(weights_size, 1000);
  EXPECT_GT(size, 10);

  ModelProto metadata;
  {
    utils::FileStream rstream(serialized);
    ParseOptions ropts;
    ParseProtoFromStream(metadata, rstream, ropts, /*clear_external_data=*/false);
  }
  ASSERT_EQ(metadata.ref_graph().ref_initializer().size(), 1u);
  const TensorProto &saved = metadata.ref_graph().ref_initializer()[0];
  EXPECT_FALSE(saved.has_raw_data());
  EXPECT_TRUE(saved.has_external_data());
  EXPECT_TRUE(saved.has_data_location());

  std::filesystem::remove_all(tmpdir);
}

TEST(onnx_external_ressource, SaveWithExternalDataMaxFileSize) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("graph");

  for (int i = 0; i < 3; ++i) {
    TensorProto *weights = graph->add_initializer();
    weights->set_name("weights" + std::to_string(i));
    weights->set_data_type(TensorProto::DataType::FLOAT);
    weights->ref_dims().push_back(1);
    weights->ref_raw_data() = std::vector<uint8_t>{1, 2, 3, 4};
  }

  std::string onnx_file = "test_split_external_file_size.onnx";
  std::string weights_file = "test_split_external_file_size.data";
  std::string weights_file_1 = "test_split_external_file_size.data.1";
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    wopts.max_external_file_size = 8;
    SerializeProtoToStream(model, wstream, wopts);
  }

  EXPECT_TRUE(std::filesystem::exists(weights_file));
  EXPECT_TRUE(std::filesystem::exists(weights_file_1));
  EXPECT_EQ(std::filesystem::file_size(weights_file), 8);
  EXPECT_EQ(std::filesystem::file_size(weights_file_1), 4);

  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
  std::remove(weights_file_1.c_str());
}

TEST(onnx_external_ressource, SaveWithExternalDataMaxFileSizeParallel) {
  // Exercise the parallel-write path with max_external_file_size > 0: the writer
  // must pre-allocate every weights file (default + extras) and dispatch tensor
  // writes to the thread pool.  The contents written to disk must round-trip
  // through ParseFromStream and reproduce the original tensor data.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("graph");

  constexpr int kNumTensors = 6;
  constexpr int64_t kTensorBytes = 4;
  std::vector<std::vector<uint8_t>> expected(kNumTensors);
  for (int i = 0; i < kNumTensors; ++i) {
    expected[i] = {static_cast<uint8_t>(10 + i * 4 + 0), static_cast<uint8_t>(10 + i * 4 + 1),
                   static_cast<uint8_t>(10 + i * 4 + 2), static_cast<uint8_t>(10 + i * 4 + 3)};
    TensorProto *weights = graph->add_initializer();
    weights->set_name("weights" + std::to_string(i));
    weights->set_data_type(TensorProto::DataType::FLOAT);
    weights->ref_dims().push_back(1);
    weights->ref_raw_data() = expected[i];
  }

  const std::string onnx_file = "test_split_external_file_size_parallel.onnx";
  const std::string weights_file = "test_split_external_file_size_parallel.data";
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    // 8 bytes per file -> 2 tensors per file -> 3 weights files total.
    wopts.max_external_file_size = kTensorBytes * 2;
    wopts.num_threads = 4;
    SerializeProtoToStream(model, wstream, wopts);
  }

  std::vector<std::string> expected_files = {weights_file, weights_file + ".1",
                                             weights_file + ".2"};
  for (const auto &f : expected_files) {
    ASSERT_TRUE(std::filesystem::exists(f)) << "missing weights file: " << f;
    EXPECT_EQ(std::filesystem::file_size(f), static_cast<std::uintmax_t>(kTensorBytes * 2)) << f;
  }
  // No extra split file should exist beyond .2.
  EXPECT_FALSE(std::filesystem::exists(weights_file + ".3"));

  // Round-trip parse and check tensor contents end-to-end.
  ModelProto parsed;
  {
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ParseModelProtoFromStream(parsed, rstream, ropts, /*clear_external_data=*/true);
  }
  ASSERT_TRUE(parsed.has_graph());
  ASSERT_EQ(parsed.ref_graph().ref_initializer().size(), kNumTensors);
  for (int i = 0; i < kNumTensors; ++i) {
    const TensorProto &t = parsed.ref_graph().ref_initializer()[i];
    ASSERT_TRUE(t.has_raw_data());
    const auto &got = t.ref_raw_data();
    ASSERT_EQ(got.size(), expected[i].size());
    for (size_t k = 0; k < expected[i].size(); ++k) {
      EXPECT_EQ(static_cast<uint8_t>(got[k]), expected[i][k]) << "tensor " << i << " byte " << k;
    }
  }

  for (const auto &f : expected_files) {
    std::remove(f.c_str());
  }
  std::remove(onnx_file.c_str());
}

TEST(onnx_external_ressource, SaveWithMultipleExternalDataFiles) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *w1 = graph->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(TensorProto::DataType::FLOAT);
  w1->ref_dims().push_back(1);
  w1->ref_raw_data() = {1, 2, 3, 4};
  w1->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto *w1_loc = w1->add_external_data();
  w1_loc->set_key("location");
  w1_loc->set_value("weights_1.data");
  auto *w1_off = w1->add_external_data();
  w1_off->set_key("offset");
  w1_off->set_value("0");
  auto *w1_len = w1->add_external_data();
  w1_len->set_key("length");
  w1_len->set_value("4");

  TensorProto *w2 = graph->add_initializer();
  w2->set_name("w2");
  w2->set_data_type(TensorProto::DataType::FLOAT);
  w2->ref_dims().push_back(1);
  w2->ref_raw_data() = {5, 6, 7, 8};
  w2->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto *w2_loc = w2->add_external_data();
  w2_loc->set_key("location");
  w2_loc->set_value("weights_2.data");
  auto *w2_off = w2->add_external_data();
  w2_off->set_key("offset");
  w2_off->set_value("0");
  auto *w2_len = w2->add_external_data();
  w2_len->set_key("length");
  w2_len->set_value("4");

  SerializeOptions wopts;
  wopts.raw_data_threshold = 1024;
  const size_t max_external_file_size = 4;
  const std::string external_file_prefix = "weights_part";
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  model.SerializeToString(serialized, external_files, max_external_file_size, external_file_prefix,
                          wopts);

  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(external_files.size(), 2U);
  EXPECT_EQ(external_files.at("weights_1.data").size(), 4U);
  EXPECT_EQ(external_files.at("weights_2.data").size(), 4U);
  EXPECT_EQ(external_files.count("weights_part_0.data"), 0U);

  ModelProto loaded;
  loaded.ParseFromString(serialized);
  EXPECT_EQ(loaded.ref_graph().ref_initializer().size(), 2);
  for (const auto &t : loaded.ref_graph().ref_initializer()) {
    EXPECT_EQ(t.has_raw_data(), false);
    EXPECT_EQ(t.ref_external_data().size(), 3);
  }
}

TEST(onnx_external_ressource, SaveWithExternalDataLocationOptionDisabled) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *w1 = graph->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(TensorProto::DataType::FLOAT);
  w1->ref_dims().push_back(1);
  w1->ref_raw_data() = {1, 2, 3, 4};
  w1->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto *w1_loc = w1->add_external_data();
  w1_loc->set_key("location");
  w1_loc->set_value("w1.data");
  auto *w1_off = w1->add_external_data();
  w1_off->set_key("offset");
  w1_off->set_value("0");
  auto *w1_len = w1->add_external_data();
  w1_len->set_key("length");
  w1_len->set_value("4");

  utils::TwoFilesWriteStream wstream("SerializeModelProtoToStreamOptionDisabled.onnx",
                                     "SerializeModelProtoToStreamOptionDisabled.data");
  SerializeOptions wopts;
  wopts.raw_data_threshold = 2;
  wopts.use_external_data_location = false;
  EXPECT_THROW(SerializeProtoToStream(model, wstream, wopts), std::exception);
}

TEST(onnx_external_ressource, SaveWithExternalDataWeightsFileMustBeNextToModel) {
  namespace fs = std::filesystem;

  fs::path root = fs::temp_directory_path() / "onnx_light_save_two_files_validation";
  fs::path model_dir = root / "model";
  fs::path weights_dir = root / "weights";
  fs::remove_all(root);
  fs::create_directories(model_dir);
  fs::create_directories(weights_dir);

  fs::path model_path = model_dir / "model.onnx";
  fs::path weights_path = weights_dir / "weights.data";
  EXPECT_THROW(
      { utils::TwoFilesWriteStream stream(model_path.string(), weights_path.string()); },
      std::exception);
  fs::create_directories(model_dir / ".weights");
  const uint8_t bytes[] = {1, 2, 3, 4};
  EXPECT_NO_THROW({
    utils::TwoFilesWriteStream stream(model_path.string(),
                                      (model_dir / ".weights/weights.data").string());
    EXPECT_EQ(stream.weights_size(), 0);
    stream.write_raw_bytes_in_second_stream(bytes, sizeof(bytes));
    EXPECT_EQ(stream.weights_size(), sizeof(bytes));
  });
  EXPECT_EQ(fs::file_size(model_dir / ".weights/weights.data"), 4U);

  fs::remove_all(root);
}

TEST(onnx_external_ressource, WriteLocationsCompareCaseAliasesAcrossFullPaths) {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "onnx_light_write_location_case_aliases";
  fs::remove_all(root);
  fs::create_directories(root / "data");
  fs::create_directories(root / "Data");
  fs::create_directories(root / "other");
  fs::create_directories(root / "Other");
  const uint8_t bytes[] = {1, 2, 3, 4};

  for (const std::string location : {"weights.bin", "data/weights.bin"}) {
    utils::TwoFilesWriteStream stream((root / "model.onnx").string(), (root / location).string());
    EXPECT_NO_THROW(stream.set_active_weights_location((root / location).string()));
    EXPECT_NO_THROW(stream.pre_allocate_weights((root / location).string(), sizeof(bytes)));
    EXPECT_NO_THROW(stream.set_active_weights_location("./other/./Weights.bin"));
    stream.write_raw_bytes_in_second_stream(bytes, sizeof(bytes));
    EXPECT_EQ(stream.weights_size(), sizeof(bytes));
    EXPECT_NO_THROW(stream.pre_allocate_weights("other/Weights.bin", sizeof(bytes)));
    EXPECT_THROW(stream.set_active_weights_location("Other/Weights.bin"), std::runtime_error);
    EXPECT_THROW(stream.pre_allocate_weights("Other/Weights.bin", sizeof(bytes)),
                 std::runtime_error);
    const std::string alias = location == "weights.bin" ? "Weights.bin" : "Data/weights.bin";
    EXPECT_THROW(stream.set_active_weights_location(alias), std::runtime_error);
    EXPECT_THROW(stream.pre_allocate_weights(alias, sizeof(bytes)), std::runtime_error);
  }

  fs::remove_all(root);
}

TEST(onnx_external_ressource, LocationsAllowAtMostOneSubdirectory) {
  for (const char *location : {"weights.bin", ".weights/weights.bin", "./data/./weights.bin"})
    EXPECT_TRUE(IsSafeExternalDataLocation(location)) << location;
  for (const char *location :
       {"", ".", "/weights.bin", "../weights.bin", "data/../weights.bin", "a/b/weights.bin"})
    EXPECT_FALSE(IsSafeExternalDataLocation(location)) << location;
}

TEST(onnx_external_ressource, SaveWithExternalDataLocationInSubdirectory) {
  namespace fs = std::filesystem;

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *w1 = graph->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(TensorProto::DataType::FLOAT);
  w1->ref_dims().push_back(1);
  w1->ref_raw_data() = {1, 2, 3, 4};
  w1->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto *w1_loc = w1->add_external_data();
  w1_loc->set_key("location");
  w1_loc->set_value("nested/weights_1.data");
  auto *w1_off = w1->add_external_data();
  w1_off->set_key("offset");
  w1_off->set_value("0");
  auto *w1_len = w1->add_external_data();
  w1_len->set_key("length");
  w1_len->set_value("4");

  fs::path root = fs::temp_directory_path() / "onnx_light_save_location_validation";
  fs::remove_all(root);
  fs::create_directories(root / "nested");
  fs::path model_path = root / "model.onnx";
  fs::path weights_path = root / "weights.data";

  {
    utils::TwoFilesWriteStream stream(model_path.string(), weights_path.string());
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    EXPECT_NO_THROW(SerializeProtoToStream(model, stream, wopts));
  }
  EXPECT_TRUE(fs::exists(root / "nested/weights_1.data"));

  fs::remove_all(root);
}

TEST(onnx_external_ressource, LoadSharedExternalDataInSubdirectory) {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "onnx_light_shared_subdir_weights";
  fs::remove_all(root);
  fs::create_directories(root / ".weights");
  const fs::path model_path = root / "model.onnx";
  const fs::path weights_path = root / ".weights/model.safetensors";
  {
    std::ofstream weights(weights_path, std::ios::binary);
    weights << "0123456789abcdefabcdefgh";
  }
  ModelProto model;
  auto *graph = model.add_graph();
  graph->set_name("g");
  for (int i = 0; i < 2; ++i) {
    auto *tensor = graph->add_initializer();
    tensor->set_name("w" + std::to_string(i));
    tensor->set_data_type(TensorProto::FLOAT);
    tensor->add_dims(1);
    tensor->set_data_location(TensorProto::EXTERNAL);
    auto add = [tensor](const char *key, const std::string &value) {
      auto *entry = tensor->add_external_data();
      entry->set_key(key);
      entry->set_value(value);
    };
    if (i == 0) {
      add("location", ".weights/model.safetensors");
      add("offset", "16");
      add("length", "4");
    } else {
      add("length", "4");
      add("offset", "20");
      add("location", ".weights/model.safetensors");
    }
  }
  {
    utils::FileWriteStream stream(model_path.string());
    SerializeOptions options;
    ASSERT_TRUE(SerializeProtoToStream(model, stream, options));
  }
  for (bool no_copy : {false, true}) {
    utils::TwoFilesStream stream(model_path.string(), weights_path.string());
    ParseOptions options;
    options.no_copy = no_copy;
    ModelProto parsed;
    ParseModelProtoFromStream(parsed, stream, options, false);
    ASSERT_EQ(parsed.graph().initializer().size(), 2U);
    for (size_t i = 0; i < 2; ++i) {
      const auto &bytes = parsed.graph().initializer()[i].raw_data();
      EXPECT_EQ(std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()),
                i == 0 ? "abcd" : "efgh");
    }
  }
  fs::remove_all(root);
}

TEST(onnx_external_ressource, LoadDefaultExternalDataRejectsParentSymlink) {
#if !defined(_WIN32)
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "onnx_light_default_weights_parent_symlink";
  fs::remove_all(root);
  fs::create_directories(root / "model");
  fs::create_directories(root / "outside");
  {
    std::ofstream weights(root / "outside/weights.bin", std::ios::binary);
    weights << "abcd";
  }
  fs::create_directory_symlink(root / "outside", root / "model/sub");
  const fs::path original_dir = fs::current_path();
  for (bool ordered : {false, true}) {
    ModelProto model;
    auto *tensor = model.add_graph()->add_initializer();
    tensor->set_name("w");
    tensor->set_data_type(TensorProto::FLOAT);
    tensor->add_dims(1);
    tensor->set_data_location(TensorProto::EXTERNAL);
    auto add = [tensor](const char *key, const char *value) {
      auto *entry = tensor->add_external_data();
      entry->set_key(key);
      entry->set_value(value);
    };
    if (ordered) {
      add("location", "sub/weights.bin");
      add("offset", "0");
      add("length", "4");
    } else {
      add("length", "4");
      add("offset", "0");
      add("location", "sub/weights.bin");
    }
    {
      utils::FileWriteStream stream((root / "model/model.onnx").string());
      SerializeOptions options;
      ASSERT_TRUE(SerializeProtoToStream(model, stream, options));
    }
    fs::current_path(root / "model");
    for (bool no_copy : {false, true}) {
      for (bool use_mmap : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "ordered=" << ordered << ", no_copy=" << no_copy
                                          << ", use_mmap=" << use_mmap);
        EXPECT_THROW(
            {
              utils::TwoFilesStream stream("model.onnx", "sub/weights.bin", use_mmap);
              ParseOptions options;
              options.no_copy = no_copy;
              ModelProto parsed;
              ParseModelProtoFromStream(parsed, stream, options, false);
            },
            std::runtime_error);
      }
    }
    fs::current_path(original_dir);
  }
  fs::remove_all(root);
#endif
}

TEST(onnx_external_ressource, SaveWithExternalDataLocationNextToModelSucceeds) {
  namespace fs = std::filesystem;

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  TensorProto *w1 = graph->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(TensorProto::DataType::FLOAT);
  w1->ref_dims().push_back(1);
  w1->ref_raw_data() = {1, 2, 3, 4};
  w1->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto *w1_loc = w1->add_external_data();
  w1_loc->set_key("location");
  w1_loc->set_value("weights_1.data");
  auto *w1_off = w1->add_external_data();
  w1_off->set_key("offset");
  w1_off->set_value("0");
  auto *w1_len = w1->add_external_data();
  w1_len->set_key("length");
  w1_len->set_value("4");

  fs::path root = fs::temp_directory_path() / "onnx_light_save_valid_location";
  fs::remove_all(root);
  fs::create_directories(root);
  fs::path model_path = root / "model.onnx";
  fs::path weights_path = root / "weights.data";

  {
    utils::TwoFilesWriteStream stream(model_path.string(), weights_path.string());
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    EXPECT_NO_THROW(SerializeProtoToStream(model, stream, wopts));
  }
  EXPECT_TRUE(fs::exists(model_path));
  EXPECT_TRUE(fs::exists(root / "weights_1.data"));

  fs::remove_all(root);
}

TEST(onnx_external_ressource, SerializeToStringWithSplitExternalFiles) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  for (int i = 0; i < 3; ++i) {
    TensorProto *w = graph->add_initializer();
    w->set_name("w" + std::to_string(i));
    w->set_data_type(TensorProto::DataType::FLOAT);
    w->ref_dims().push_back(1);
    w->ref_raw_data() = {static_cast<uint8_t>(1 + i), static_cast<uint8_t>(2 + i),
                         static_cast<uint8_t>(3 + i), static_cast<uint8_t>(4 + i)};
  }

  SerializeOptions opts;
  opts.raw_data_threshold = 0;
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  model.SerializeToString(serialized, external_files, 8, "weights_part", opts);

  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(external_files.size(), 2U);
  EXPECT_EQ(external_files.at("weights_part_0.data").size(), 8U);
  EXPECT_EQ(external_files.at("weights_part_1.data").size(), 4U);

  ModelProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_TRUE(parsed.has_graph());
  ASSERT_EQ(parsed.ref_graph().ref_initializer().size(), 3U);
  const auto &i0 = parsed.ref_graph().ref_initializer()[0];
  const auto &i1 = parsed.ref_graph().ref_initializer()[1];
  const auto &i2 = parsed.ref_graph().ref_initializer()[2];
  EXPECT_EQ(std::string(i0.ref_external_data()[0].ref_value()), "weights_part_0.data");
  EXPECT_EQ(std::string(i1.ref_external_data()[0].ref_value()), "weights_part_0.data");
  EXPECT_EQ(std::string(i2.ref_external_data()[0].ref_value()), "weights_part_1.data");
  EXPECT_EQ(std::string(i0.ref_external_data()[1].ref_value()), "0");
  EXPECT_EQ(std::string(i1.ref_external_data()[1].ref_value()), "4");
  EXPECT_EQ(std::string(i2.ref_external_data()[1].ref_value()), "0");
}

TEST(onnx_external_ressource, LoadWithExternalDataSplitFiles) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("graph");

  for (int i = 0; i < 3; ++i) {
    TensorProto *weights = graph->add_initializer();
    weights->set_name("weights" + std::to_string(i));
    weights->set_data_type(TensorProto::DataType::FLOAT);
    weights->ref_dims().push_back(1);
    weights->ref_raw_data() = std::vector<uint8_t>{1, 2, static_cast<uint8_t>(3 + i), 4};
  }

  std::string onnx_file = "test_split_load_external.onnx";
  std::string weights_file = "test_split_load_external.data";
  std::string weights_file_1 = "test_split_load_external.data.1";
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    wopts.max_external_file_size = 8;
    SerializeProtoToStream(model, wstream, wopts);
  }

  EXPECT_TRUE(std::filesystem::exists(weights_file));
  EXPECT_TRUE(std::filesystem::exists(weights_file_1));

  // Load the model back using TwoFilesStream with only the primary weights file.
  // The secondary file (weights_file_1) must be opened automatically.
  ModelProto model2;
  {
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ParseProtoFromStream(model2, rstream, ropts);
  }

  ASSERT_EQ(model2.ref_graph().ref_initializer().size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    const TensorProto &orig = model.ref_graph().ref_initializer()[i];
    const TensorProto &loaded = model2.ref_graph().ref_initializer()[i];
    EXPECT_EQ(orig.ref_raw_data(), loaded.ref_raw_data()) << "Mismatch at initializer " << i;
  }

  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
  std::remove(weights_file_1.c_str());
}

TEST(onnx_file, FileStream_ModelProto_Write) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto *nodeg = graph->add_node();
  nodeg->set_name("test_graph");
  nodeg->set_op_type("If");
  AttributeProto *attrg = nodeg->add_attribute();
  attrg->set_name("bias");
  GraphProto *nested = attrg->add_g();

  TensorProto *weights2 = nested->add_initializer();
  weights2->set_name("weights2");
  weights2->set_data_type(TensorProto::DataType::FLOAT);
  weights2->ref_dims().push_back(1);
  weights2->ref_dims().push_back(1);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);

  NodeProto *node2 = nested->add_node();
  node2->set_name("test_node");
  node2->set_op_type("Add");
  AttributeProto *attr2 = node2->add_attribute();
  attr2->set_name("bias");
  TensorProto &biasw2 = attr2->ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path() /
      ("onnx_light_file_stream_model_proto_write_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::remove_all(tmpdir);
  ASSERT_TRUE(std::filesystem::create_directories(tmpdir));
  std::string temp_filename = (tmpdir / "test_tensor_file_stream.tmp").string();
  std::string temp_filename2 = (tmpdir / "test_tensor_file_stream2.tmp").string();
  std::string temp_weights = (tmpdir / "test_tensor_file_stream3.tmp").string();

  {
    utils::TwoFilesWriteStream wstream(temp_filename2, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 1000000;
    SerializeProtoToStream(model, wstream, wopts);
  }

  {
    utils::FileWriteStream wstream(temp_filename);
    SerializeOptions wopts;
    SerializeProtoToStream(model, wstream, wopts);
  }

  std::error_code size_error;
  auto size = std::filesystem::file_size(temp_filename, size_error);
  EXPECT_FALSE(size_error) << temp_filename << ": " << size_error.message();
  std::error_code size2_error;
  auto size2 = std::filesystem::file_size(temp_filename2, size2_error);
  EXPECT_FALSE(size2_error) << temp_filename2 << ": " << size2_error.message();
  if (!size_error && !size2_error) {
    EXPECT_EQ(size, size2);
  }

  std::error_code remove_error;
  const auto removed = std::filesystem::remove_all(tmpdir, remove_error);
  EXPECT_FALSE(remove_error) << tmpdir << ": " << remove_error.message();
  EXPECT_GT(removed, 0u);
}

TEST(onnx_file, FileStream_ModelProto_WriteRead) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(1);
  weights->ref_dims().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);
  weights->ref_raw_data().push_back(1);

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("bias");
  TensorProto &biasw = attr->ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto *nodeg = graph->add_node();
  nodeg->set_name("test_graph");
  nodeg->set_op_type("If");
  AttributeProto *attrg = nodeg->add_attribute();
  attrg->set_name("bias");
  GraphProto *nested = attrg->add_g();

  TensorProto *weights2 = nested->add_initializer();
  weights2->set_name("weights2");
  weights2->set_data_type(TensorProto::DataType::FLOAT);
  weights2->ref_dims().push_back(1);
  weights2->ref_dims().push_back(1);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);
  weights2->ref_raw_data().push_back(3);

  NodeProto *node2 = nested->add_node();
  node2->set_name("test_node");
  node2->set_op_type("Add");
  AttributeProto *attr2 = node2->add_attribute();
  attr2->set_name("bias");
  TensorProto &biasw2 = attr2->ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path() /
      ("onnx_light_file_stream_model_proto_write_read_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::remove_all(tmpdir);
  ASSERT_TRUE(std::filesystem::create_directories(tmpdir));
  std::string temp_filename = (tmpdir / "test_tensor_file_stream_read.0.tmp").string();
  std::string temp_weights = (tmpdir / "test_tensor_file_stream_read.0.weight.tmp").string();

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  ModelProto metadata;
  {
    utils::FileStream rstream(temp_filename);
    ParseOptions ropts;
    ParseProtoFromStream(metadata, rstream, ropts, /*clear_external_data=*/false);
  }
  ASSERT_EQ(metadata.ref_graph().ref_initializer().size(), 1u);
  const TensorProto &saved = metadata.ref_graph().ref_initializer()[0];
  EXPECT_FALSE(saved.has_raw_data());
  EXPECT_TRUE(saved.has_external_data());
  EXPECT_TRUE(saved.has_data_location());

  ModelProto model2;
  {
    utils::TwoFilesStream rstream(temp_filename, temp_weights);
    ParseOptions ropts;
    ParseProtoFromStream(model2, rstream, ropts);
  }

  EXPECT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data());
    EXPECT_EQ(std::string(model.ref_graph().ref_initializer()[i].ref_name()),
              model2.ref_graph().ref_initializer()[i].ref_name());
  }

  std::error_code remove_error;
  const auto removed = std::filesystem::remove_all(tmpdir, remove_error);
  EXPECT_FALSE(remove_error) << tmpdir << ": " << remove_error.message();
  EXPECT_GT(removed, 0u);
}

TEST(onnx_external_ressource, LoadWithExternalData) {
  // Build a model with 7 initializers (2 of them large and multi-dimensional),
  // write it out with external data, then read it back through TwoFilesStream and
  // verify the large initializers round-trip via the external weights file.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("graph");

  for (int i = 0; i < 7; ++i) {
    TensorProto *weights = graph->add_initializer();
    weights->set_name("weights" + std::to_string(i));
    weights->set_data_type(TensorProto::DataType::FLOAT);
    if (i < 2) {
      // Two large, multi-dimensional initializers with more than 1024 raw bytes.
      weights->ref_dims().push_back(16);
      weights->ref_dims().push_back(32);
      weights->ref_raw_data() =
          std::vector<uint8_t>(16 * 32 * sizeof(float), static_cast<uint8_t>(1 + i));
    } else {
      weights->ref_dims().push_back(1);
      weights->ref_raw_data() = std::vector<uint8_t>{1, 2, static_cast<uint8_t>(3 + i), 4};
    }
  }

  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path() /
      ("onnx_light_load_with_external_data_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::remove_all(tmpdir);
  ASSERT_TRUE(std::filesystem::create_directories(tmpdir));

  std::string onnx_file = (tmpdir / "test_writing_external_weights_read_from_onnx.onnx").string();
  std::string weights_file =
      (tmpdir / "test_writing_external_weights_read_from_onnx.data").string();
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }
  std::error_code model_size_error;
  auto size = std::filesystem::file_size(onnx_file, model_size_error);
  EXPECT_FALSE(model_size_error) << onnx_file << ": " << model_size_error.message();
  std::error_code weights_size_error;
  auto weights_size = std::filesystem::file_size(weights_file, weights_size_error);
  EXPECT_FALSE(weights_size_error) << weights_file << ": " << weights_size_error.message();
  if (!model_size_error) {
    EXPECT_GT(size, 10);
  }
  if (!weights_size_error) {
    EXPECT_GT(weights_size, 1024);
  }

  ModelProto metadata;
  {
    utils::FileStream metadata_stream(onnx_file);
    ParseOptions metadata_opts;
    ParseProtoFromStream(metadata, metadata_stream, metadata_opts, /*clear_external_data=*/false);
  }
  ASSERT_EQ(metadata.ref_graph().ref_initializer().size(), 7u);
  for (const TensorProto &saved : metadata.ref_graph().ref_initializer()) {
    EXPECT_FALSE(saved.has_raw_data());
    EXPECT_TRUE(saved.has_external_data());
    EXPECT_EQ(saved.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    bool has_location = false;
    bool has_offset = false;
    bool has_length = false;
    for (const StringStringEntryProto &entry : saved.ref_external_data()) {
      has_location = has_location || entry.ref_key() == "location";
      has_offset = has_offset || entry.ref_key() == "offset";
      has_length = has_length || entry.ref_key() == "length" || entry.ref_key() == "size";
    }
    EXPECT_TRUE(has_location);
    EXPECT_TRUE(has_offset);
    EXPECT_TRUE(has_length);
  }

  ModelProto model2;
  {
    utils::TwoFilesStream stream(onnx_file, weights_file);
    ONNX_LIGHT_NAMESPACE::ParseOptions opts;
    EXPECT_TRUE(model2.ParseFromStream(stream, opts));
  }
  EXPECT_EQ(model2.ref_graph().ref_initializer().size(), 7);
  IteratorTensorProto it(&model2.ref_graph());
  int big = 0;
  while (it.next()) {
    if (it->ref_dims().size() > 1) {
      EXPECT_TRUE(it->ref_raw_data().size() > 1024);
      ++big;
    }
  }
  EXPECT_EQ(big, 2);

  std::error_code remove_error;
  const auto removed = std::filesystem::remove_all(tmpdir, remove_error);
  EXPECT_FALSE(remove_error) << tmpdir << ": " << remove_error.message();
  EXPECT_GT(removed, 0u);
}

// -----------------------------------------------------------------------
// Alignment tests
// -----------------------------------------------------------------------

// Helper: returns true when ptr is aligned to align bytes.
static bool is_aligned(const void *ptr, size_t align) {
  return reinterpret_cast<uintptr_t>(ptr) % align == 0;
}

TEST(onnx_alignment, ByteSpanResizeAligned) {
  utils::ByteSpan span;
  // Plain resize — owned mode.
  span.resize(16);
  EXPECT_EQ(span.size(), 16u);
  EXPECT_FALSE(span.is_borrowed());
  EXPECT_FALSE(span.is_aligned_owned());

  // Aligned resize to 64-byte boundary.
  span.resize_aligned(16, 64);
  EXPECT_EQ(span.size(), 16u);
  EXPECT_FALSE(span.is_borrowed());
  EXPECT_TRUE(span.is_aligned_owned());
  EXPECT_TRUE(is_aligned(span.data(), 64));

  // Write/read via mutable data().
  for (size_t i = 0; i < 16; ++i)
    span.data()[i] = static_cast<uint8_t>(i);
  for (size_t i = 0; i < 16; ++i)
    EXPECT_EQ(span[i], static_cast<uint8_t>(i));
}

TEST(onnx_alignment, ByteSpanOwnedResizePreservesPrefix) {
  utils::ByteSpan span;
  span.resize(4);
  for (size_t i = 0; i < 4; ++i)
    span.data()[i] = static_cast<uint8_t>(i + 1);

  span.resize(6);
  EXPECT_EQ(span.size(), 6u);
  EXPECT_FALSE(span.is_borrowed());
  EXPECT_FALSE(span.is_aligned_owned());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(span[i], static_cast<uint8_t>(i + 1));

  span.data()[4] = 5;
  span.data()[5] = 6;
  span.resize(3);
  EXPECT_EQ(span.size(), 3u);
  EXPECT_EQ(span[0], 1);
  EXPECT_EQ(span[1], 2);
  EXPECT_EQ(span[2], 3);
}

TEST(onnx_alignment, ByteSpanCopyAligned) {
  utils::ByteSpan src;
  src.resize_aligned(8, 32);
  for (size_t i = 0; i < 8; ++i)
    src.data()[i] = static_cast<uint8_t>(i + 1);

  // Copy constructor.
  utils::ByteSpan dst(src);
  EXPECT_EQ(dst.size(), 8u);
  EXPECT_TRUE(dst.is_aligned_owned());
  EXPECT_TRUE(is_aligned(dst.data(), 32));
  for (size_t i = 0; i < 8; ++i)
    EXPECT_EQ(dst[i], static_cast<uint8_t>(i + 1));
  // dst should own its own buffer — modifying it must not affect src.
  dst.data()[0] = 99;
  EXPECT_EQ(src[0], 1);

  // Copy assignment.
  utils::ByteSpan dst2;
  dst2 = src;
  EXPECT_EQ(dst2.size(), 8u);
  EXPECT_TRUE(dst2.is_aligned_owned());
  EXPECT_TRUE(is_aligned(dst2.data(), 32));
  for (size_t i = 0; i < 8; ++i)
    EXPECT_EQ(dst2[i], static_cast<uint8_t>(i + 1));
}

TEST(onnx_alignment, ByteSpanMoveAligned) {
  utils::ByteSpan src;
  src.resize_aligned(8, 64);
  for (size_t i = 0; i < 8; ++i)
    src.data()[i] = static_cast<uint8_t>(i + 10);

  utils::ByteSpan dst(std::move(src));
  EXPECT_EQ(dst.size(), 8u);
  EXPECT_TRUE(dst.is_aligned_owned());
  EXPECT_TRUE(is_aligned(dst.data(), 64));
  for (size_t i = 0; i < 8; ++i)
    EXPECT_EQ(dst[i], static_cast<uint8_t>(i + 10));
  // src should be empty after move.
  EXPECT_TRUE(src.empty());
}

TEST(onnx_alignment, ByteSpanPushBackCopiesBorrowedAndAlignedData) {
  const std::vector<uint8_t> borrowed = {1, 2, 3};
  utils::ByteSpan span;
  span.assign_borrowed(borrowed.data(), borrowed.size());
  span.push_back(4);
  EXPECT_FALSE(span.is_borrowed());
  EXPECT_FALSE(span.is_aligned_owned());
  ASSERT_EQ(span.size(), 4u);
  EXPECT_EQ(span[0], 1);
  EXPECT_EQ(span[1], 2);
  EXPECT_EQ(span[2], 3);
  EXPECT_EQ(span[3], 4);

  span.resize_aligned(3, 32);
  for (size_t i = 0; i < 3; ++i)
    span.data()[i] = static_cast<uint8_t>(10 + i);
  span.push_back(13);
  EXPECT_FALSE(span.is_borrowed());
  EXPECT_FALSE(span.is_aligned_owned());
  ASSERT_EQ(span.size(), 4u);
  EXPECT_EQ(span[0], 10);
  EXPECT_EQ(span[1], 11);
  EXPECT_EQ(span[2], 12);
  EXPECT_EQ(span[3], 13);
}

TEST(onnx_alignment, ParseOptionsAlignmentInlineData) {
  // Build a TensorProto with inline raw_data.
  TensorProto tensor;
  tensor.set_name("t");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(4);
  const std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
  tensor.ref_raw_data().resize(vals.size() * sizeof(float));
  std::memcpy(tensor.ref_raw_data().data(), vals.data(), vals.size() * sizeof(float));

  // Serialize to bytes.
  std::string serialized;
  tensor.SerializeToString(serialized);

  // Parse back with alignment=64.
  TensorProto parsed;
  ParseOptions popts;
  popts.alignment = 64;
  parsed.ParseFromString(serialized, popts);

  EXPECT_EQ(parsed.ref_raw_data().size(), vals.size() * sizeof(float));
  EXPECT_TRUE(parsed.ref_raw_data().is_aligned_owned());
  EXPECT_TRUE(is_aligned(parsed.ref_raw_data().data(), 64));
  const float *fp = reinterpret_cast<const float *>(parsed.ref_raw_data().data());
  for (size_t i = 0; i < vals.size(); ++i)
    EXPECT_FLOAT_EQ(fp[i], vals[i]);
}

TEST(onnx_alignment, SerializeOptionsAlignmentExternalData) {
  // Build a model with two initializers.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  const std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  const std::vector<float> data1 = {4.0f, 5.0f};

  TensorProto *w0 = graph->add_initializer();
  w0->set_name("w0");
  w0->set_data_type(TensorProto::DataType::FLOAT);
  w0->ref_dims().push_back(3);
  w0->ref_raw_data().resize(data0.size() * sizeof(float));
  std::memcpy(w0->ref_raw_data().data(), data0.data(), data0.size() * sizeof(float));

  TensorProto *w1 = graph->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(TensorProto::DataType::FLOAT);
  w1->ref_dims().push_back(2);
  w1->ref_raw_data().resize(data1.size() * sizeof(float));
  std::memcpy(w1->ref_raw_data().data(), data1.data(), data1.size() * sizeof(float));

  // Serialize to in-memory buffers with alignment=16.
  // Use a large max_external_file_size so all tensors go into one file.
  SerializeOptions sopts;
  sopts.raw_data_threshold = 0;
  sopts.alignment = 16;
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  model.SerializeToString(serialized, external_files, 1000000, "weights", sopts);

  ASSERT_EQ(external_files.size(), 1u);
  const std::string &wbuf = external_files.begin()->second;

  // Parse the model (metadata only, not loading raw_data from external).
  ModelProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_EQ(parsed.ref_graph().ref_initializer().size(), 2u);

  const auto &p0 = parsed.ref_graph().ref_initializer()[0];
  const auto &p1 = parsed.ref_graph().ref_initializer()[1];

  // Both tensors must be marked EXTERNAL.
  EXPECT_EQ(p0.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(p1.ref_data_location(), TensorProto::DataLocation::EXTERNAL);

  // Extract the offsets stored in the metadata.
  auto get_offset = [](const TensorProto &t) -> int64_t {
    for (size_t i = 0; i < t.ref_external_data().size(); ++i) {
      if (t.ref_external_data()[i].ref_key() == "offset")
        return t.ref_external_data()[i].ref_value().toint64();
    }
    return -1;
  };

  const int64_t off0 = get_offset(p0);
  const int64_t off1 = get_offset(p1);

  // First offset must be 0 (no predecessor).
  EXPECT_EQ(off0, 0);
  // Second offset must be a multiple of the requested alignment.
  EXPECT_GE(off1, 0);
  EXPECT_EQ(off1 % 16, 0) << "offset of w1 is " << off1 << ", not aligned to 16";

  // The weight buffer must be at least as large as (off1 + size of w1).
  const int64_t size_w1 = static_cast<int64_t>(data1.size() * sizeof(float));
  EXPECT_GE(static_cast<int64_t>(wbuf.size()), off1 + size_w1);

  // Verify raw payload correctness.
  const float *fp0 = reinterpret_cast<const float *>(wbuf.data() + off0);
  for (size_t i = 0; i < data0.size(); ++i)
    EXPECT_FLOAT_EQ(fp0[i], data0[i]);
  const float *fp1 = reinterpret_cast<const float *>(wbuf.data() + off1);
  for (size_t i = 0; i < data1.size(); ++i)
    EXPECT_FLOAT_EQ(fp1[i], data1[i]);
}

TEST(onnx_alignment, SerializeOptionsAlignmentExternalDataManyRandomSizes) {
  constexpr int64_t align = 16;
  constexpr size_t n_tensors = 128;

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> size_dist(1, 257);
  std::vector<std::vector<uint8_t>> payloads;
  payloads.reserve(n_tensors);

  for (size_t i = 0; i < n_tensors; ++i) {
    const size_t sz = static_cast<size_t>(size_dist(gen));
    payloads.emplace_back(sz);
    for (size_t j = 0; j < sz; ++j)
      payloads.back()[j] = static_cast<uint8_t>((i * 31 + j * 17) % 251);

    TensorProto *t = graph->add_initializer();
    t->set_name("w" + std::to_string(i));
    t->set_data_type(TensorProto::DataType::UINT8);
    t->ref_dims().push_back(static_cast<int64_t>(sz));
    t->ref_raw_data().resize(sz);
    std::memcpy(t->ref_raw_data().data(), payloads.back().data(), sz);
  }

  SerializeOptions sopts;
  sopts.raw_data_threshold = 0;
  sopts.alignment = align;
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  model.SerializeToString(serialized, external_files, 100000000, "weights", sopts);

  ASSERT_EQ(external_files.size(), 1u);
  const std::string &wbuf = external_files.begin()->second;

  ModelProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_EQ(parsed.ref_graph().ref_initializer().size(), n_tensors);

  auto get_external_value = [](const TensorProto &t, const char *key) -> int64_t {
    for (const auto &ed : t.ref_external_data()) {
      if (ed.ref_key() == key)
        return ed.ref_value().toint64();
    }
    return -1;
  };

  for (size_t i = 0; i < n_tensors; ++i) {
    const TensorProto &t = parsed.ref_graph().ref_initializer()[i];
    EXPECT_EQ(t.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    const int64_t off = get_external_value(t, "offset");
    const int64_t len = get_external_value(t, "length");
    ASSERT_GE(off, 0) << "missing offset for tensor " << i;
    ASSERT_GE(len, 0) << "missing length for tensor " << i;
    EXPECT_EQ(off % align, 0) << "tensor " << i << " offset=" << off << " is not aligned to "
                              << align;
    ASSERT_EQ(len, static_cast<int64_t>(payloads[i].size()));
    ASSERT_GE(static_cast<int64_t>(wbuf.size()), off + len);
    EXPECT_TRUE(std::memcmp(wbuf.data() + off, payloads[i].data(), payloads[i].size()) == 0);
  }
}

TEST(onnx_alignment, SerializeToFileWithAlignment) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  // Create three tensors with sizes that are NOT multiples of the alignment
  // so the padding behaviour is exercised.
  const std::vector<float> vals0(3, 1.0f); // 12 bytes
  const std::vector<float> vals1(5, 2.0f); // 20 bytes
  const std::vector<float> vals2(7, 3.0f); // 28 bytes

  auto add_tensor = [&](const std::string &name, const std::vector<float> &v) {
    TensorProto *t = graph->add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(v.size()));
    t->ref_raw_data().resize(v.size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), v.data(), v.size() * sizeof(float));
  };
  add_tensor("t0", vals0);
  add_tensor("t1", vals1);
  add_tensor("t2", vals2);

  const std::string onnx_file = "test_alignment_file.onnx";
  const std::string weights_file = "test_alignment_file.data";
  const int64_t align = 64;

  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions sopts;
    sopts.raw_data_threshold = 0;
    sopts.alignment = align;
    SerializeProtoToStream(model, wstream, sopts);
  }

  // Load back and verify offsets are aligned.
  ModelProto loaded;
  {
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ParseProtoFromStream(loaded, rstream, ropts);
  }

  ASSERT_EQ(loaded.ref_graph().ref_initializer().size(), 3u);
  for (size_t i = 0; i < 3u; ++i) {
    const TensorProto &t = loaded.ref_graph().ref_initializer()[i];
    EXPECT_FALSE(t.ref_raw_data().empty()) << "tensor " << i << " has empty raw_data";
    for (size_t j = 0; j < t.ref_external_data().size(); ++j) {
      if (t.ref_external_data()[j].ref_key() == "offset") {
        const int64_t off = t.ref_external_data()[j].ref_value().toint64();
        EXPECT_EQ(off % align, 0) << "tensor " << i << " offset=" << off << " is not aligned to "
                                  << align;
      }
    }
  }

  // Verify data correctness.
  const float *fp0 = reinterpret_cast<const float *>(
      loaded.ref_graph().ref_initializer()[0].ref_raw_data().data());
  const float *fp1 = reinterpret_cast<const float *>(
      loaded.ref_graph().ref_initializer()[1].ref_raw_data().data());
  const float *fp2 = reinterpret_cast<const float *>(
      loaded.ref_graph().ref_initializer()[2].ref_raw_data().data());
  for (size_t i = 0; i < vals0.size(); ++i)
    EXPECT_FLOAT_EQ(fp0[i], vals0[i]);
  for (size_t i = 0; i < vals1.size(); ++i)
    EXPECT_FLOAT_EQ(fp1[i], vals1[i]);
  for (size_t i = 0; i < vals2.size(); ++i)
    EXPECT_FLOAT_EQ(fp2[i], vals2[i]);

  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
}

TEST(onnx_alignment, ParseOptionsAlignmentExternalData) {
  // Serialize a model with external data (no alignment).
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  const std::vector<float> vals(8, 42.0f); // 32 bytes
  TensorProto *t = graph->add_initializer();
  t->set_name("big");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_dims().push_back(8);
  t->ref_raw_data().resize(vals.size() * sizeof(float));
  std::memcpy(t->ref_raw_data().data(), vals.data(), vals.size() * sizeof(float));

  const std::string onnx_file = "test_parse_alignment.onnx";
  const std::string weights_file = "test_parse_alignment.data";
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions sopts;
    sopts.raw_data_threshold = 0;
    SerializeProtoToStream(model, wstream, sopts);
  }

  // Load back with alignment=64 so raw_data_ is allocated aligned.
  ModelProto loaded;
  {
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ropts.alignment = 64;
    ParseProtoFromStream(loaded, rstream, ropts);
  }

  ASSERT_EQ(loaded.ref_graph().ref_initializer().size(), 1u);
  const TensorProto &lt = loaded.ref_graph().ref_initializer()[0];
  EXPECT_EQ(lt.ref_raw_data().size(), vals.size() * sizeof(float));
  EXPECT_TRUE(lt.ref_raw_data().is_aligned_owned());
  EXPECT_TRUE(is_aligned(lt.ref_raw_data().data(), 64));
  const float *fp = reinterpret_cast<const float *>(lt.ref_raw_data().data());
  for (size_t i = 0; i < vals.size(); ++i)
    EXPECT_FLOAT_EQ(fp[i], vals[i]);

  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
}

// -----------------------------------------------------------------------
// Scenario:
//  1. Build a model with an Add node and two initializers, save it with
//     external data (two files: .onnx + .data).
//  2. Re-load the .onnx file alone (without the external weights file), change
//     Add into Sub and save it again to a *new* .onnx file. The external data
//     file must NOT be created nor updated by this save.
//  3. Re-load both files (with external data this time), negate one initializer
//     and save the model again. The modified initializer now lives inline in
//     the main .onnx file while the other initializer still references the
//     original external data file, which must remain unchanged.
// -----------------------------------------------------------------------
TEST(onnx_external_ressource, EditModelWithoutTouchingExternalData) {
  namespace fs = std::filesystem;

  // ---- step 1: build and save a model with external data -----------------
  ModelProto model;
  model.set_ir_version(9);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  const std::vector<float> w1_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> w2_vals = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};

  auto add_initializer = [&](const std::string &name, const std::vector<float> &v) {
    TensorProto *t = graph->add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(v.size()));
    t->ref_raw_data().resize(v.size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), v.data(), v.size() * sizeof(float));
  };
  add_initializer("W1", w1_vals);
  add_initializer("W2", w2_vals);

  NodeProto *node = graph->add_node();
  node->set_name("add_node");
  node->set_op_type("Add");
  *node->add_input() = "W1";
  *node->add_input() = "W2";
  *node->add_output() = "Y";

  const std::string onnx_file = "test_edit_external_step1.onnx";
  const std::string weights_file = "test_edit_external_step1.data";
  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());

  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0; // force both tensors to be external
    SerializeProtoToStream(model, wstream, wopts);
  }
  ASSERT_TRUE(fs::exists(onnx_file));
  ASSERT_TRUE(fs::exists(weights_file));

  // Capture the size and mtime of the external data file so we can later
  // assert that subsequent saves leave it untouched.
  const auto data_size_before = fs::file_size(weights_file);
  const auto data_mtime_before = fs::last_write_time(weights_file);
  EXPECT_EQ(data_size_before, (w1_vals.size() + w2_vals.size()) * sizeof(float));

  // ---- step 2: load only the .onnx file, change Add to Sub, save back ----
  // Loading via FileStream (single file) leaves initializers as EXTERNAL with
  // their external_data metadata intact but does not load the raw bytes.
  ModelProto edited;
  {
    utils::FileStream rstream(onnx_file);
    ParseOptions ropts;
    ParseProtoFromStream(edited, rstream, ropts);
  }
  ASSERT_EQ(edited.ref_graph().ref_initializer().size(), 2u);
  for (const auto &t : edited.ref_graph().ref_initializer()) {
    ASSERT_TRUE(t.has_data_location()) << "name=" << t.ref_name();
    EXPECT_EQ(t.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    EXPECT_FALSE(t.ref_external_data().empty());
    EXPECT_TRUE(t.ref_raw_data().empty()) << "raw bytes should not be loaded without weights file";
  }
  ASSERT_EQ(edited.ref_graph().ref_node().size(), 1u);
  EXPECT_EQ(edited.ref_graph().ref_node()[0].ref_op_type(), "Add");

  // Mutate the node, keep initializers untouched.
  edited.ref_graph().ref_node()[0].set_op_type("Sub");

  // Sleep a moment so that, if a write were to occur, the mtime would change.
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));

  const std::string onnx_file_v2 = "test_edit_external_step2.onnx";
  std::remove(onnx_file_v2.c_str());
  {
    // Saving with a single-file stream means the external metadata is
    // preserved in the proto but no weights file is created or updated.
    utils::FileWriteStream wstream(onnx_file_v2);
    SerializeOptions wopts;
    SerializeProtoToStream(edited, wstream, wopts);
  }
  ASSERT_TRUE(fs::exists(onnx_file_v2));
  EXPECT_FALSE(fs::exists("test_edit_external_step2.data"));
  // External data file untouched.
  EXPECT_EQ(fs::file_size(weights_file), data_size_before);
  EXPECT_EQ(fs::last_write_time(weights_file), data_mtime_before);

  // Sanity-check the v2 model: Sub op and tensors still reference the
  // original external data file.
  {
    ModelProto v2;
    utils::FileStream rstream(onnx_file_v2);
    ParseOptions ropts;
    ParseProtoFromStream(v2, rstream, ropts);
    ASSERT_EQ(v2.ref_graph().ref_node().size(), 1u);
    EXPECT_EQ(v2.ref_graph().ref_node()[0].ref_op_type(), "Sub");
    ASSERT_EQ(v2.ref_graph().ref_initializer().size(), 2u);
    for (const auto &t : v2.ref_graph().ref_initializer()) {
      ASSERT_TRUE(t.has_data_location());
      EXPECT_EQ(t.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
      ASSERT_FALSE(t.ref_external_data().empty());
      const std::string loc = t.ref_external_data()[0].ref_value();
      EXPECT_NE(loc.find(weights_file), std::string::npos)
          << "external location='" << loc << "' does not reference '" << weights_file << "'";
    }
  }

  // ---- step 3: negate one initializer, save it inline in the main file ---
  // Start from the same metadata-only view loaded in step 2 (the .onnx alone),
  // so initializers are still EXTERNAL with their original external_data refs.
  ModelProto full;
  {
    utils::FileStream rstream(onnx_file_v2);
    ParseOptions ropts;
    ParseProtoFromStream(full, rstream, ropts);
  }
  ASSERT_EQ(full.ref_graph().ref_initializer().size(), 2u);

  // Negate W1 and store it inline in the main file: drop the external metadata
  // and write the new raw bytes directly into the TensorProto.
  TensorProto &w1_loaded = full.ref_graph().ref_initializer()[0];
  ASSERT_EQ(w1_loaded.ref_name(), "W1");
  ASSERT_TRUE(w1_loaded.ref_raw_data().empty());
  w1_loaded.reset_data_location();
  w1_loaded.clr_external_data();
  w1_loaded.ref_raw_data().resize(w1_vals.size() * sizeof(float));
  {
    float *fp = reinterpret_cast<float *>(w1_loaded.ref_raw_data().data());
    for (size_t i = 0; i < w1_vals.size(); ++i)
      fp[i] = -w1_vals[i];
  }

  // W2 remains external, pointing at the original file.
  const TensorProto &w2_loaded = full.ref_graph().ref_initializer()[1];
  ASSERT_EQ(w2_loaded.ref_name(), "W2");
  ASSERT_TRUE(w2_loaded.has_data_location());
  ASSERT_EQ(w2_loaded.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  ASSERT_FALSE(w2_loaded.ref_external_data().empty());

  std::this_thread::sleep_for(std::chrono::milliseconds(1100));

  const std::string onnx_file_v3 = "test_edit_external_step3.onnx";
  std::remove(onnx_file_v3.c_str());
  {
    // Single-file save: inline tensors go into the main .onnx, external
    // tensors keep their existing metadata, no weights file is touched.
    utils::FileWriteStream wstream(onnx_file_v3);
    SerializeOptions wopts;
    SerializeProtoToStream(full, wstream, wopts);
  }
  ASSERT_TRUE(fs::exists(onnx_file_v3));
  EXPECT_FALSE(fs::exists("test_edit_external_step3.data"));
  // External data file still unchanged.
  EXPECT_EQ(fs::file_size(weights_file), data_size_before);
  EXPECT_EQ(fs::last_write_time(weights_file), data_mtime_before);

  // Reload v3 via FileStream alone to inspect the structure persisted in the
  // main .onnx file: W1 should be inline (raw_data bytes) while W2 should
  // still carry its external metadata pointing at the unchanged weights file.
  {
    ModelProto v3_struct;
    utils::FileStream rstream(onnx_file_v3);
    ParseOptions ropts;
    ParseProtoFromStream(v3_struct, rstream, ropts);
    ASSERT_EQ(v3_struct.ref_graph().ref_initializer().size(), 2u);
    const TensorProto &w1_struct = v3_struct.ref_graph().ref_initializer()[0];
    const TensorProto &w2_struct = v3_struct.ref_graph().ref_initializer()[1];
    EXPECT_FALSE(w1_struct.has_data_location());
    EXPECT_TRUE(w1_struct.ref_external_data().empty());
    ASSERT_EQ(w1_struct.ref_raw_data().size(), w1_vals.size() * sizeof(float));
    const float *fp1 = reinterpret_cast<const float *>(w1_struct.ref_raw_data().data());
    for (size_t i = 0; i < w1_vals.size(); ++i)
      EXPECT_FLOAT_EQ(fp1[i], -w1_vals[i]);
    ASSERT_TRUE(w2_struct.has_data_location());
    EXPECT_EQ(w2_struct.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    ASSERT_FALSE(w2_struct.ref_external_data().empty());
    EXPECT_TRUE(w2_struct.ref_raw_data().empty());
  }

  // Reload v3 alongside the (still unchanged) weights file: W2's bytes should
  // now be populated from the original external file and match the originals.
  ModelProto v3;
  {
    utils::TwoFilesStream rstream(onnx_file_v3, weights_file);
    ParseOptions ropts;
    ParseProtoFromStream(v3, rstream, ropts);
  }
  ASSERT_EQ(v3.ref_graph().ref_initializer().size(), 2u);
  const TensorProto &w1_v3 = v3.ref_graph().ref_initializer()[0];
  const TensorProto &w2_v3 = v3.ref_graph().ref_initializer()[1];
  ASSERT_EQ(w1_v3.ref_raw_data().size(), w1_vals.size() * sizeof(float));
  const float *fp1 = reinterpret_cast<const float *>(w1_v3.ref_raw_data().data());
  for (size_t i = 0; i < w1_vals.size(); ++i)
    EXPECT_FLOAT_EQ(fp1[i], -w1_vals[i]);
  ASSERT_EQ(w2_v3.ref_raw_data().size(), w2_vals.size() * sizeof(float));
  const float *fp2 = reinterpret_cast<const float *>(w2_v3.ref_raw_data().data());
  for (size_t i = 0; i < w2_vals.size(); ++i)
    EXPECT_FLOAT_EQ(fp2[i], w2_vals[i]);

  // Cleanup.
  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
  std::remove(onnx_file_v2.c_str());
  std::remove(onnx_file_v3.c_str());
}

TEST(onnx_helper, AddAttribute) {
  NodeProto node;
  AddAttribute<int64_t>(node, "i", 7);
  AddAttribute<float>(node, "f", 1.5f);
  AddAttribute<std::string>(node, "s", "abc");
  AddAttribute<std::vector<int64_t>>(node, "ints", {1, 2, 3});
  AddAttribute<std::vector<float>>(node, "floats", {0.5f, 1.5f});
  AddAttribute<std::vector<std::string>>(node, "strings", {"a", "b"});

  ASSERT_EQ(node.attribute().size(), 6);
  EXPECT_EQ(node.attribute()[0].type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(node.attribute()[0].i(), 7);
  EXPECT_EQ(node.attribute()[1].type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(node.attribute()[1].f(), 1.5f);
  EXPECT_EQ(node.attribute()[2].type(), AttributeProto::AttributeType::STRING);
  EXPECT_EQ(node.attribute()[2].s(), "abc");
  EXPECT_EQ(node.attribute()[3].type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(node.attribute()[3].ints().size(), 3);
  EXPECT_EQ(node.attribute()[3].ints()[2], 3);
  EXPECT_EQ(node.attribute()[4].type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(node.attribute()[4].floats().size(), 2);
  EXPECT_EQ(node.attribute()[5].type(), AttributeProto::AttributeType::STRINGS);
  EXPECT_EQ(node.attribute()[5].strings().size(), 2);
}

TEST(onnx_helper, AddInputsAndAddOutputs) {
  // initializer_list overload.
  NodeProto node;
  AddInputs(node, {"a", "b", "c"});
  AddOutputs(node, {"y1", "y2"});
  ASSERT_EQ(node.ref_input().size(), 3u);
  ASSERT_EQ(node.ref_output().size(), 2u);
  EXPECT_EQ(std::string(node.ref_input()[0].data(), node.ref_input()[0].size()), "a");
  EXPECT_EQ(std::string(node.ref_input()[1].data(), node.ref_input()[1].size()), "b");
  EXPECT_EQ(std::string(node.ref_input()[2].data(), node.ref_input()[2].size()), "c");
  EXPECT_EQ(std::string(node.ref_output()[0].data(), node.ref_output()[0].size()), "y1");
  EXPECT_EQ(std::string(node.ref_output()[1].data(), node.ref_output()[1].size()), "y2");

  // std::vector range overload, also works on FunctionProto which has the same
  // input/output FIELD_REPEATED_STR members.
  FunctionProto fn;
  std::vector<std::string> ins = {"x1", "x2"};
  std::vector<std::string> outs = {"o1"};
  AddInputs(fn, ins);
  AddOutputs(fn, outs);
  ASSERT_EQ(fn.ref_input().size(), 2u);
  ASSERT_EQ(fn.ref_output().size(), 1u);
  EXPECT_EQ(std::string(fn.ref_input()[0].data(), fn.ref_input()[0].size()), "x1");
  EXPECT_EQ(std::string(fn.ref_input()[1].data(), fn.ref_input()[1].size()), "x2");
  EXPECT_EQ(std::string(fn.ref_output()[0].data(), fn.ref_output()[0].size()), "o1");
}

TEST(onnx_helper, MakeNodeMinimal) {
  NodeProto node = MakeNode("Add", {"a", "b"}, {"c"});
  EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Add");
  ASSERT_EQ(node.ref_input().size(), 2u);
  EXPECT_EQ(std::string(node.ref_input()[0].data(), node.ref_input()[0].size()), "a");
  EXPECT_EQ(std::string(node.ref_input()[1].data(), node.ref_input()[1].size()), "b");
  ASSERT_EQ(node.ref_output().size(), 1u);
  EXPECT_EQ(std::string(node.ref_output()[0].data(), node.ref_output()[0].size()), "c");
  // domain and name are left empty when not provided.
  EXPECT_TRUE(node.ref_domain().empty());
  EXPECT_TRUE(node.ref_name().empty());
}

TEST(onnx_helper, MakeNodeWithDomainAndName) {
  NodeProto node = MakeNode("Conv", {"X", "W"}, {"Y"}, "ai.onnx", "conv1");
  EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Conv");
  EXPECT_EQ(std::string(node.ref_domain().data(), node.ref_domain().size()), "ai.onnx");
  EXPECT_EQ(std::string(node.ref_name().data(), node.ref_name().size()), "conv1");
  ASSERT_EQ(node.ref_input().size(), 2u);
  ASSERT_EQ(node.ref_output().size(), 1u);
}

TEST(onnx_helper, MakeNodeFromVectors) {
  std::vector<std::string> ins = {"a", "b", "c"};
  std::vector<std::string> outs = {"y"};
  NodeProto node = MakeNode("Sum", ins, outs);
  EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Sum");
  ASSERT_EQ(node.ref_input().size(), 3u);
  ASSERT_EQ(node.ref_output().size(), 1u);
  EXPECT_EQ(std::string(node.ref_input()[2].data(), node.ref_input()[2].size()), "c");
}

TEST(onnx_helper, MakeNodeEmptyOutputs) {
  // Some ops can have zero outputs (e.g. fictional sinks). MakeNode should
  // not require any.
  NodeProto node = MakeNode("Sink", {"x"}, {});
  ASSERT_EQ(node.ref_input().size(), 1u);
  EXPECT_EQ(node.ref_output().size(), 0u);
}

TEST(onnx_helper, AddFloatAttribute) {
  NodeProto node;
  AddFloatAttribute(node, "alpha", 0.25f);
  AddFloatAttribute(node, "beta", -1.5f);
  ASSERT_EQ(node.ref_attribute().size(), 2u);
  const AttributeProto &a0 = node.ref_attribute()[0];
  const AttributeProto &a1 = node.ref_attribute()[1];
  EXPECT_EQ(std::string(a0.ref_name().data(), a0.ref_name().size()), "alpha");
  EXPECT_EQ(a0.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(a0.ref_f(), 0.25f);
  EXPECT_EQ(std::string(a1.ref_name().data(), a1.ref_name().size()), "beta");
  EXPECT_EQ(a1.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(a1.ref_f(), -1.5f);
}

TEST(onnx_helper, FindAttributeReturnsMatchOrNull) {
  NodeProto node;
  node.set_op_type("Op");
  AddAttribute<int64_t>(node, "a", int64_t{42});
  AddAttribute<float>(node, "b", 1.5f);

  const AttributeProto *a = FindAttribute(node, "a");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->ref_i(), 42);
  EXPECT_NE(FindAttribute(node, "b"), nullptr);
  EXPECT_EQ(FindAttribute(node, "missing"), nullptr);
}

TEST(onnx_helper, GetAttributeOrReturnsValueOrDefault) {
  NodeProto node;
  node.set_op_type("Op");
  AddAttribute<int64_t>(node, "keepdims", int64_t{1});
  AddAttribute<float>(node, "alpha", 2.5f);
  AddAttribute<std::string>(node, "mode", std::string("linear"));

  EXPECT_EQ(GetAttributeOr<int64_t>(node, "keepdims", 0), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(node, "missing", 7), 7);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(node, "alpha", 0.0f), 2.5f);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(node, "missing", -1.0f), -1.0f);
  EXPECT_EQ(GetAttributeOr<std::string>(node, "mode", std::string("default")), "linear");
  EXPECT_EQ(GetAttributeOr<std::string>(node, "missing", std::string("default")), "default");
}

TEST(onnx_helper, GetAttributeIntsAppendsValues) {
  NodeProto node;
  node.set_op_type("Op");
  AddAttribute<std::vector<int64_t>>(node, "axes", std::vector<int64_t>{1, -2, 3});

  std::vector<int64_t> out{99};
  ASSERT_TRUE(GetAttributeInts(node, "axes", out));
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0], 99);
  EXPECT_EQ(out[1], 1);
  EXPECT_EQ(out[2], -2);
  EXPECT_EQ(out[3], 3);

  std::vector<int64_t> empty;
  EXPECT_FALSE(GetAttributeInts(node, "missing", empty));
  EXPECT_TRUE(empty.empty());
}

TEST(onnx_helper, MakeInitializerInt64ShapeLike) {
  TensorProto tensor = MakeInitializer<int64_t>("reshape_shape", {3}, {0, 0, -1});
  EXPECT_EQ(std::string(tensor.ref_name().data(), tensor.ref_name().size()), "reshape_shape");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::INT64);
  ASSERT_EQ(tensor.ref_dims().size(), 1u);
  EXPECT_EQ(tensor.ref_dims()[0], 3);
  ASSERT_EQ(tensor.ref_int64_data().size(), 3u);
  EXPECT_EQ(tensor.ref_int64_data()[0], 0);
  EXPECT_EQ(tensor.ref_int64_data()[1], 0);
  EXPECT_EQ(tensor.ref_int64_data()[2], -1);
}

TEST(onnx_helper, MakeInitializerFloat2D) {
  TensorProto tensor = MakeInitializer<float>("W", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(std::string(tensor.ref_name().data(), tensor.ref_name().size()), "W");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  ASSERT_EQ(tensor.ref_dims().size(), 2u);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 2);
  ASSERT_EQ(tensor.ref_float_data().size(), 4u);
  EXPECT_FLOAT_EQ(tensor.ref_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(tensor.ref_float_data()[3], 4.0f);
}

TEST(onnx_helper, MakeInitializerInt32Double) {
  TensorProto t32 = MakeInitializer<int32_t>("i32", {2}, {7, -8});
  EXPECT_EQ(t32.ref_data_type(), TensorProto::DataType::INT32);
  ASSERT_EQ(t32.ref_int32_data().size(), 2u);
  EXPECT_EQ(t32.ref_int32_data()[0], 7);
  EXPECT_EQ(t32.ref_int32_data()[1], -8);

  TensorProto td = MakeInitializer<double>("d", {1}, {2.5});
  EXPECT_EQ(td.ref_data_type(), TensorProto::DataType::DOUBLE);
  ASSERT_EQ(td.ref_double_data().size(), 1u);
  EXPECT_DOUBLE_EQ(td.ref_double_data()[0], 2.5);
}

TEST(onnx_helper, MakeInitializerUint64AndString) {
  TensorProto tu = MakeInitializer<uint64_t>("u", {3}, {1u, 2u, 3u});
  EXPECT_EQ(tu.ref_data_type(), TensorProto::DataType::UINT64);
  ASSERT_EQ(tu.ref_uint64_data().size(), 3u);
  EXPECT_EQ(tu.ref_uint64_data()[2], 3u);

  TensorProto ts = MakeInitializer<std::string>("s", {2}, {"hello", "world"});
  EXPECT_EQ(ts.ref_data_type(), TensorProto::DataType::STRING);
  ASSERT_EQ(ts.ref_string_data().size(), 2u);
  EXPECT_EQ(std::string(ts.ref_string_data()[0].data(), ts.ref_string_data()[0].size()), "hello");
  EXPECT_EQ(std::string(ts.ref_string_data()[1].data(), ts.ref_string_data()[1].size()), "world");
}

TEST(onnx_helper, AddInitializerAppendsToGraphAndReturnsReference) {
  ModelProto model;
  GraphProto *graph = model.add_graph();

  TensorProto &t1 = AddInitializer<int64_t>(*graph, "reshape_shape", {3}, {0, 0, -1});
  TensorProto &t2 = AddInitializer<float>(*graph, "W", {2}, {0.5f, -0.25f});

  ASSERT_EQ(graph->ref_initializer().size(), 2u);
  EXPECT_EQ(&graph->ref_initializer()[0], &t1);
  EXPECT_EQ(&graph->ref_initializer()[1], &t2);
  EXPECT_EQ(std::string(t1.ref_name().data(), t1.ref_name().size()), "reshape_shape");
  EXPECT_EQ(t1.ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(t2.ref_data_type(), TensorProto::DataType::FLOAT);
}

TEST(onnx_helper, MakeInitializerShape1DInt64) {
  TensorProto tensor = MakeInitializerShape("axes", {0, 2, -1});
  EXPECT_EQ(std::string(tensor.ref_name().data(), tensor.ref_name().size()), "axes");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::INT64);
  ASSERT_EQ(tensor.ref_dims().size(), 1u);
  EXPECT_EQ(tensor.ref_dims()[0], 3);
  ASSERT_EQ(tensor.ref_int64_data().size(), 3u);
  EXPECT_EQ(tensor.ref_int64_data()[0], 0);
  EXPECT_EQ(tensor.ref_int64_data()[1], 2);
  EXPECT_EQ(tensor.ref_int64_data()[2], -1);
}

TEST(onnx_helper, AddInitializerShapeAppendsInt64Initializer) {
  ModelProto model;
  GraphProto *graph = model.add_graph();

  TensorProto &t = AddInitializerShape(*graph, "reshape_shape", {0, 0, -1});

  ASSERT_EQ(graph->ref_initializer().size(), 1u);
  EXPECT_EQ(&graph->ref_initializer()[0], &t);
  EXPECT_EQ(std::string(t.ref_name().data(), t.ref_name().size()), "reshape_shape");
  EXPECT_EQ(t.ref_data_type(), TensorProto::DataType::INT64);
  ASSERT_EQ(t.ref_dims().size(), 1u);
  EXPECT_EQ(t.ref_dims()[0], 3);
  ASSERT_EQ(t.ref_int64_data().size(), 3u);
  EXPECT_EQ(t.ref_int64_data()[2], -1);
}

// -----------------------------------------------------------------------
// Streaming alignment of an existing two-file model. The function must:
//  - rewrite the external weights file so that every tensor's offset is a
//    multiple of the requested alignment,
//  - update the proto's external_data entries (location + offset + length),
//  - preserve raw bytes exactly,
//  - and never load the full set of weights in memory (we rely on the
//    chunk_size argument to enforce this: setting it to a tiny value still
//    produces a correct output).
// -----------------------------------------------------------------------
TEST(onnx_alignment, AlignExternalDataStreamingRewritesAlignedWeights) {
  namespace fs = std::filesystem;

  // Build a model with three initializers of different sizes so that the
  // streaming layout must insert padding between them when re-aligned.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  const std::vector<std::vector<float>> payloads = {
      std::vector<float>(7, 1.5f),  // 28 bytes, not a 64-multiple
      std::vector<float>(3, -2.0f), //  12 bytes
      std::vector<float>(17, 3.0f), //  68 bytes
  };
  for (size_t i = 0; i < payloads.size(); ++i) {
    TensorProto *t = graph->add_initializer();
    t->set_name(std::string("w") + std::to_string(i));
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(payloads[i].size()));
    t->ref_raw_data().resize(payloads[i].size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), payloads[i].data(), payloads[i].size() * sizeof(float));
  }

  const std::string src_onnx = "stream_align_src.onnx";
  const std::string src_weights = "stream_align_src.data";
  const std::string dst_onnx = "stream_align_dst.onnx";
  const std::string dst_weights = "stream_align_dst.data";
  for (const auto &p : {src_onnx, src_weights, dst_onnx, dst_weights}) {
    std::remove(p.c_str());
  }

  // 1) Save the model with unaligned (offset 0) external data — packed tight.
  {
    utils::TwoFilesWriteStream wstream(src_onnx, src_weights);
    SerializeOptions sopts;
    sopts.raw_data_threshold = 0;
    SerializeProtoToStream(model, wstream, sopts);
  }
  ASSERT_TRUE(fs::exists(src_onnx));
  ASSERT_TRUE(fs::exists(src_weights));

  // 2) Stream-align with a tiny chunk_size to exercise the chunked copy path.
  constexpr int64_t alignment = 64;
  constexpr int64_t chunk_size = 7; // < every payload, forces multiple I/O loops
  const offset_t total =
      AlignExternalDataStreaming(src_onnx, dst_onnx, dst_weights, alignment, chunk_size);
  ASSERT_GT(total, 0);
  EXPECT_EQ(static_cast<int64_t>(fs::file_size(dst_weights)), static_cast<int64_t>(total));

  // 3) Load the destination model and verify alignment + content.
  ModelProto loaded;
  {
    utils::TwoFilesStream rstream(dst_onnx, dst_weights);
    ParseOptions ropts;
    ParseProtoFromStream(loaded, rstream, ropts);
  }
  ASSERT_EQ(loaded.ref_graph().ref_initializer().size(), payloads.size());

  // 4) Inspect the proto-side metadata (offsets must be aligned, location must
  //    point at the new file). Use FileStream so we don't load weights twice.
  ModelProto metadata;
  {
    utils::FileStream meta_stream(dst_onnx);
    ParseOptions meta_opts;
    ParseProtoFromStream(metadata, meta_stream, meta_opts, /*clear_external_data=*/false);
  }
  ASSERT_EQ(metadata.ref_graph().ref_initializer().size(), payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    const TensorProto &meta = metadata.ref_graph().ref_initializer()[i];
    ASSERT_TRUE(meta.has_data_location());
    ASSERT_EQ(meta.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    std::string loc;
    int64_t off = -1;
    int64_t len = -1;
    for (int j = 0; j < static_cast<int>(meta.ref_external_data().size()); ++j) {
      const StringStringEntryProto &e = meta.ref_external_data()[j];
      if (e.ref_key() == "location")
        loc = e.ref_value();
      else if (e.ref_key() == "offset")
        off = std::stoll(e.ref_value());
      else if (e.ref_key() == "length" || e.ref_key() == "size")
        len = std::stoll(e.ref_value());
    }
    EXPECT_EQ(loc, dst_weights);
    ASSERT_GE(off, 0);
    EXPECT_EQ(off % alignment, 0);
    EXPECT_EQ(len, static_cast<int64_t>(payloads[i].size() * sizeof(float)));

    // Compare bytes.
    const TensorProto &lt = loaded.ref_graph().ref_initializer()[i];
    ASSERT_EQ(lt.ref_raw_data().size(), payloads[i].size() * sizeof(float));
    EXPECT_EQ(std::memcmp(lt.ref_raw_data().data(), payloads[i].data(),
                          payloads[i].size() * sizeof(float)),
              0);
  }

  for (const auto &p : {src_onnx, src_weights, dst_onnx, dst_weights}) {
    std::remove(p.c_str());
  }
}

// -----------------------------------------------------------------------
// SaveModelWithSharedExternalData mirrors the AlignExternalDataStreaming test
// for the "two models" scenario from the upstream issue: a first model is
// saved with external data, then loaded without it; a second model reuses
// those external initializers and adds new ones, and is saved through this
// function — the new external file must only contain the new weights, the
// reused ones must keep referencing the first model's file.
// -----------------------------------------------------------------------
TEST(onnx_alignment, SaveModelWithSharedExternalDataReusesFirstModelWeights) {
  namespace fs = std::filesystem;

  // ---- 1) Build the first model with two external initializers.
  ModelProto first;
  GraphProto *g1 = first.add_graph();
  g1->set_name("g1");
  const std::vector<std::vector<float>> first_payloads = {
      std::vector<float>(7, 1.5f),   // 28 bytes
      std::vector<float>(11, -2.0f), // 44 bytes
  };
  for (size_t i = 0; i < first_payloads.size(); ++i) {
    TensorProto *t = g1->add_initializer();
    t->set_name(std::string("a") + std::to_string(i));
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(first_payloads[i].size()));
    t->ref_raw_data().resize(first_payloads[i].size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), first_payloads[i].data(),
                first_payloads[i].size() * sizeof(float));
  }

  const std::string src_onnx = "share_weights_src.onnx";
  const std::string src_weights = "share_weights_src.data";
  const std::string dst_onnx = "share_weights_dst.onnx";
  const std::string dst_weights = dst_onnx + ".data";
  for (const auto &p : {src_onnx, src_weights, dst_onnx, dst_weights}) {
    std::remove(p.c_str());
  }
  {
    utils::TwoFilesWriteStream wstream(src_onnx, src_weights);
    SerializeOptions sopts;
    sopts.raw_data_threshold = 0;
    SerializeProtoToStream(first, wstream, sopts);
  }
  ASSERT_TRUE(fs::exists(src_onnx));
  ASSERT_TRUE(fs::exists(src_weights));

  // ---- 2) Re-parse the first model without loading external data (so its
  // initializers keep external_data entries with location='share_weights_src.data').
  ModelProto first_meta;
  {
    utils::FileStream rstream(src_onnx);
    ParseOptions ropts;
    ropts.skip_raw_data = true;
    ropts.raw_data_threshold = 0;
    ParseModelProtoFromStream(first_meta, rstream, ropts, /*clear_external_data=*/false);
  }

  // ---- 3) Build the second model: reuse first's two initializers and add a new one.
  //
  // Capture the location each reused initializer carries (it was recorded by the original
  // TwoFilesWriteStream; its exact spelling — bare filename vs ``./<file>`` vs ``.\<file>`` —
  // depends on the std::filesystem implementation, so we don't hard-code it).
  ModelProto second;
  GraphProto *g2 = second.add_graph();
  g2->set_name("g2");
  std::unordered_map<std::string, std::string> reused_expected_location;
  for (size_t i = 0; i < first_meta.ref_graph().ref_initializer().size(); ++i) {
    TensorProto *t = g2->add_initializer();
    *t = first_meta.ref_graph().ref_initializer()[i];
    for (int j = 0; j < static_cast<int>(t->ref_external_data().size()); ++j) {
      const StringStringEntryProto &e = t->ref_external_data()[j];
      if (e.ref_key() == "location") {
        reused_expected_location[t->ref_name()] = e.ref_value();
        break;
      }
    }
  }
  std::vector<float> new_payload(5, 7.0f); // 20 bytes
  {
    TensorProto *t = g2->add_initializer();
    t->set_name("n0");
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(new_payload.size()));
    t->ref_raw_data().resize(new_payload.size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), new_payload.data(), new_payload.size() * sizeof(float));
  }

  // ---- 4) Save with shared weights.
  constexpr int64_t alignment = 64;
  const auto src_size_before = fs::file_size(src_weights);
  SerializeOptions save_opts;
  save_opts.alignment = alignment;
  const offset_t total = SaveModelWithSharedExternalData(second, dst_onnx, save_opts);
  EXPECT_EQ(total, static_cast<offset_t>(new_payload.size() * sizeof(float))); // 20 bytes only
  ASSERT_TRUE(fs::exists(dst_weights));
  EXPECT_EQ(static_cast<offset_t>(fs::file_size(dst_weights)), total);
  // First weights file is untouched.
  EXPECT_EQ(fs::file_size(src_weights), src_size_before);

  // ---- 5) Inspect the rewritten metadata.
  ModelProto meta;
  {
    utils::FileStream meta_stream(dst_onnx);
    ParseOptions meta_opts;
    ParseProtoFromStream(meta, meta_stream, meta_opts, /*clear_external_data=*/false);
  }
  ASSERT_EQ(meta.ref_graph().ref_initializer().size(), 3);
  for (int i = 0; i < static_cast<int>(meta.ref_graph().ref_initializer().size()); ++i) {
    const TensorProto &t = meta.ref_graph().ref_initializer()[i];
    ASSERT_TRUE(t.has_data_location());
    ASSERT_EQ(t.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
    std::string loc;
    int64_t off = -1;
    int64_t len = -1;
    for (int j = 0; j < static_cast<int>(t.ref_external_data().size()); ++j) {
      const StringStringEntryProto &e = t.ref_external_data()[j];
      if (e.ref_key() == "location")
        loc = e.ref_value();
      else if (e.ref_key() == "offset")
        off = std::stoll(e.ref_value());
      else if (e.ref_key() == "length" || e.ref_key() == "size")
        len = std::stoll(e.ref_value());
    }
    if (t.ref_name()[0] == 'a') {
      // Reused — must still reference the first model's weights file using the exact
      // location string recorded when the first model was originally saved.
      auto it = reused_expected_location.find(t.ref_name());
      ASSERT_NE(it, reused_expected_location.end());
      EXPECT_EQ(loc, it->second);
      EXPECT_NE(loc.find(src_weights), std::string::npos);
    } else {
      // New — written into the auto-derived secondary file at an aligned offset.
      EXPECT_EQ(loc, dst_weights);
      ASSERT_GE(off, 0);
      EXPECT_EQ(off % alignment, 0);
      EXPECT_EQ(len, static_cast<int64_t>(new_payload.size() * sizeof(float)));
    }
  }

  for (const auto &p : {src_onnx, src_weights, dst_onnx, dst_weights}) {
    std::remove(p.c_str());
  }
}

TEST(onnx_helper, ReadScalarAsDouble_TypedFields) {
  double out = 0.0;

  TensorProto f;
  f.set_data_type(TensorProto::DataType::FLOAT);
  f.add_float_data(2.5f);
  ASSERT_TRUE(ReadScalarAsDouble(f, out));
  EXPECT_DOUBLE_EQ(out, 2.5);

  TensorProto d;
  d.set_data_type(TensorProto::DataType::DOUBLE);
  d.add_double_data(-3.25);
  ASSERT_TRUE(ReadScalarAsDouble(d, out));
  EXPECT_DOUBLE_EQ(out, -3.25);

  TensorProto i64;
  i64.set_data_type(TensorProto::DataType::INT64);
  i64.add_int64_data(-7);
  ASSERT_TRUE(ReadScalarAsDouble(i64, out));
  EXPECT_DOUBLE_EQ(out, -7.0);

  TensorProto i8;
  i8.set_data_type(TensorProto::DataType::INT8);
  i8.add_int32_data(-5);
  ASSERT_TRUE(ReadScalarAsDouble(i8, out));
  EXPECT_DOUBLE_EQ(out, -5.0);

  TensorProto u8;
  u8.set_data_type(TensorProto::DataType::UINT8);
  u8.add_int32_data(200);
  ASSERT_TRUE(ReadScalarAsDouble(u8, out));
  EXPECT_DOUBLE_EQ(out, 200.0);

  TensorProto u64;
  u64.set_data_type(TensorProto::DataType::UINT64);
  u64.add_uint64_data(42);
  ASSERT_TRUE(ReadScalarAsDouble(u64, out));
  EXPECT_DOUBLE_EQ(out, 42.0);
}

TEST(onnx_helper, ReadScalarAsDouble_StoragePrecedenceAndFallback) {
  const struct {
    TensorProto::DataType type;
    size_t width;
  } cases[] = {
      {TensorProto::DataType::FLOAT, 4},  {TensorProto::DataType::DOUBLE, 8},
      {TensorProto::DataType::INT64, 8},  {TensorProto::DataType::INT32, 4},
      {TensorProto::DataType::INT16, 2},  {TensorProto::DataType::INT8, 1},
      {TensorProto::DataType::UINT8, 1},  {TensorProto::DataType::BOOL, 1},
      {TensorProto::DataType::UINT16, 2}, {TensorProto::DataType::UINT32, 4},
      {TensorProto::DataType::UINT64, 8},
  };
  const uint8_t raw[8] = {};
  for (const auto &test : cases) {
    SCOPED_TRACE(static_cast<int>(test.type));
    for (size_t raw_size = 0; raw_size <= test.width; ++raw_size) {
      SCOPED_TRACE(raw_size);
      TensorProto tensor;
      tensor.set_data_type(test.type);
      if (raw_size > 0) {
        tensor.set_raw_data(raw, raw_size);
      }
      double out = 123.0;
      EXPECT_EQ(ReadScalarAsDouble(tensor, out), raw_size == test.width);
      EXPECT_DOUBLE_EQ(out, raw_size == test.width ? 0.0 : 123.0);

      switch (test.type) {
      case TensorProto::DataType::FLOAT:
        tensor.add_float_data(1.0f);
        tensor.add_float_data(2.0f);
        break;
      case TensorProto::DataType::DOUBLE:
        tensor.add_double_data(1.0);
        tensor.add_double_data(2.0);
        break;
      case TensorProto::DataType::INT64:
        tensor.add_int64_data(1);
        tensor.add_int64_data(2);
        break;
      case TensorProto::DataType::UINT32:
      case TensorProto::DataType::UINT64:
        tensor.add_uint64_data(1);
        tensor.add_uint64_data(2);
        break;
      default:
        tensor.add_int32_data(1);
        tensor.add_int32_data(2);
        break;
      }
      ASSERT_TRUE(ReadScalarAsDouble(tensor, out));
      EXPECT_DOUBLE_EQ(out, 1.0);
      if (test.type == TensorProto::DataType::FLOAT || test.type == TensorProto::DataType::DOUBLE) {
        std::vector<double> values;
        ASSERT_TRUE(ReadFloatingValues(tensor, values));
        ASSERT_EQ(values.size(), 2u);
        EXPECT_DOUBLE_EQ(out, values[0]);
      }
    }
  }
}

TEST(onnx_helper, ReadScalarAsDouble_RawData) {
  double out = 0.0;

  const float fval = 1.5f;
  TensorProto f;
  f.set_data_type(TensorProto::DataType::FLOAT);
  f.set_raw_data(&fval, sizeof(fval));
  ASSERT_TRUE(ReadScalarAsDouble(f, out));
  EXPECT_DOUBLE_EQ(out, 1.5);

  const int16_t ival = -258;
  TensorProto i16;
  i16.set_data_type(TensorProto::DataType::INT16);
  i16.set_raw_data(&ival, sizeof(ival));
  ASSERT_TRUE(ReadScalarAsDouble(i16, out));
  EXPECT_DOUBLE_EQ(out, -258.0);

  const int64_t lval = -9000000000LL;
  TensorProto i64;
  i64.set_data_type(TensorProto::DataType::INT64);
  i64.set_raw_data(&lval, sizeof(lval));
  ASSERT_TRUE(ReadScalarAsDouble(i64, out));
  EXPECT_DOUBLE_EQ(out, static_cast<double>(lval));
}

TEST(onnx_helper, ReadIntegerValuesNegativeInt64RawData) {
  const uint8_t raw[] = {
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // -1
      0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // -2
      0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // -3
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // INT64_MIN
  };
  TensorProto tensor;
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.set_raw_data(raw, sizeof(raw));

  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(tensor, values));
  EXPECT_EQ(values, (std::vector<int64_t>{-1, -2, -3, std::numeric_limits<int64_t>::min()}));
}

TEST(onnx_helper, ReadScalarAsDouble_EmptyOrUnsupported) {
  double out = 123.0;

  TensorProto empty_float;
  empty_float.set_data_type(TensorProto::DataType::FLOAT);
  EXPECT_FALSE(ReadScalarAsDouble(empty_float, out));
  EXPECT_DOUBLE_EQ(out, 123.0);

  // A raw_data buffer smaller than the element size must not be read.
  const uint8_t half[2] = {0, 0};
  TensorProto truncated;
  truncated.set_data_type(TensorProto::DataType::FLOAT);
  truncated.set_raw_data(half, sizeof(half));
  out = 123.0;
  EXPECT_FALSE(ReadScalarAsDouble(truncated, out));
  EXPECT_DOUBLE_EQ(out, 123.0);

  TensorProto str;
  str.set_data_type(TensorProto::DataType::STRING);
  out = 123.0;
  EXPECT_FALSE(ReadScalarAsDouble(str, out));
  EXPECT_DOUBLE_EQ(out, 123.0);
}

TEST(onnx_helper, EmptyNodeList) {
  const std::vector<const NodeProto *> &a = EmptyNodeList();
  const std::vector<const NodeProto *> &b = EmptyNodeList();
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(&a, &b);
}
