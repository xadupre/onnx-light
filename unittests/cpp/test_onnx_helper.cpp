#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

TEST(onnx_helper, IteratorTensorProto) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
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

TEST(onnx_helper, IteratorTensorProto_NestedGraph) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
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

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
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

TEST(onnx_helper, SerializeModelProtoToStream) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
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

  GraphProto &model_graph = model.add_graph();
  model_graph.set_name("g");

  for (int i = 0; i < 3; ++i) {
    TensorProto &weights = model_graph.add_initializer();
    const std::vector<uint8_t> tensor_raw_data{1, 2, static_cast<uint8_t>(3 + i), 4};
    weights.set_name("weights" + std::to_string(i));
    weights.set_data_type(TensorProto::DataType::FLOAT);
    weights.add_dims(1);
    weights.set_raw_data(tensor_raw_data);
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

  IteratorTensorProto tensor_it(&model_graph);
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
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / "data" / "test_writing_external_weights.original.onnx";
  if (!std::filesystem::exists(file_path)) {
    GTEST_SKIP() << "File not found: " << file_path.string();
  }

  ModelProto model;
  utils::FileStream stream(file_path.string());
  ONNX_LIGHT_NAMESPACE::ParseOptions opts;
  model.ParseFromStream(stream, opts);

  auto serialized = source_dir / "test_onnx_file_save_with_external_data.onnx";
  auto weights = source_dir / "test_onnx_file_save_with_external_data.data";
  {
    utils::TwoFilesWriteStream wstream(serialized.string(), weights.string());
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }
  auto size = std::filesystem::file_size(serialized);
  auto weights_size = std::filesystem::file_size(weights);
  EXPECT_GT(weights_size, 1000);
  EXPECT_GT(size, 10);

  std::remove(serialized.string().c_str());
  std::remove(weights.string().c_str());
}

TEST(onnx_external_ressource, SaveWithExternalDataMaxFileSize) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("graph");

  for (int i = 0; i < 3; ++i) {
    TensorProto &weights = graph.add_initializer();
    weights.set_name("weights" + std::to_string(i));
    weights.set_data_type(TensorProto::DataType::FLOAT);
    weights.ref_dims().push_back(1);
    weights.ref_raw_data() = std::vector<uint8_t>{1, 2, 3, 4};
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

TEST(onnx_external_ressource, SaveWithMultipleExternalDataFiles) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  TensorProto &w1 = graph.add_initializer();
  w1.set_name("w1");
  w1.set_data_type(TensorProto::DataType::FLOAT);
  w1.ref_dims().push_back(1);
  w1.ref_raw_data() = {1, 2, 3, 4};
  w1.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto &w1_loc = w1.add_external_data();
  w1_loc.set_key("location");
  w1_loc.set_value("weights_1.data");
  auto &w1_off = w1.add_external_data();
  w1_off.set_key("offset");
  w1_off.set_value("0");
  auto &w1_len = w1.add_external_data();
  w1_len.set_key("length");
  w1_len.set_value("4");

  TensorProto &w2 = graph.add_initializer();
  w2.set_name("w2");
  w2.set_data_type(TensorProto::DataType::FLOAT);
  w2.ref_dims().push_back(1);
  w2.ref_raw_data() = {5, 6, 7, 8};
  w2.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto &w2_loc = w2.add_external_data();
  w2_loc.set_key("location");
  w2_loc.set_value("weights_2.data");
  auto &w2_off = w2.add_external_data();
  w2_off.set_key("offset");
  w2_off.set_value("0");
  auto &w2_len = w2.add_external_data();
  w2_len.set_key("length");
  w2_len.set_value("4");

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
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  TensorProto &w1 = graph.add_initializer();
  w1.set_name("w1");
  w1.set_data_type(TensorProto::DataType::FLOAT);
  w1.ref_dims().push_back(1);
  w1.ref_raw_data() = {1, 2, 3, 4};
  w1.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  auto &w1_loc = w1.add_external_data();
  w1_loc.set_key("location");
  w1_loc.set_value("w1.data");
  auto &w1_off = w1.add_external_data();
  w1_off.set_key("offset");
  w1_off.set_value("0");
  auto &w1_len = w1.add_external_data();
  w1_len.set_key("length");
  w1_len.set_value("4");

  utils::TwoFilesWriteStream wstream("SerializeModelProtoToStreamOptionDisabled.onnx",
                                     "SerializeModelProtoToStreamOptionDisabled.data");
  SerializeOptions wopts;
  wopts.raw_data_threshold = 2;
  wopts.use_external_data_location = false;
  EXPECT_THROW(SerializeProtoToStream(model, wstream, wopts), std::exception);
}

TEST(onnx_external_ressource, SerializeToStringWithSplitExternalFiles) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  for (int i = 0; i < 3; ++i) {
    TensorProto &w = graph.add_initializer();
    w.set_name("w" + std::to_string(i));
    w.set_data_type(TensorProto::DataType::FLOAT);
    w.ref_dims().push_back(1);
    w.ref_raw_data() = {static_cast<uint8_t>(1 + i), static_cast<uint8_t>(2 + i),
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
  EXPECT_EQ(i0.ref_external_data()[0].ref_value().as_string(), "weights_part_0.data");
  EXPECT_EQ(i1.ref_external_data()[0].ref_value().as_string(), "weights_part_0.data");
  EXPECT_EQ(i2.ref_external_data()[0].ref_value().as_string(), "weights_part_1.data");
  EXPECT_EQ(i0.ref_external_data()[1].ref_value().as_string(), "0");
  EXPECT_EQ(i1.ref_external_data()[1].ref_value().as_string(), "4");
  EXPECT_EQ(i2.ref_external_data()[1].ref_value().as_string(), "0");
}

TEST(onnx_external_ressource, LoadWithExternalDataSplitFiles) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("graph");

  for (int i = 0; i < 3; ++i) {
    TensorProto &weights = graph.add_initializer();
    weights.set_name("weights" + std::to_string(i));
    weights.set_data_type(TensorProto::DataType::FLOAT);
    weights.ref_dims().push_back(1);
    weights.ref_raw_data() = std::vector<uint8_t>{1, 2, static_cast<uint8_t>(3 + i), 4};
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

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::string temp_filename = "test_tensor_file_stream.tmp";
  std::string temp_filename2 = "test_tensor_file_stream2.tmp";
  std::string temp_weights = "test_tensor_file_stream3.tmp";

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

  auto size = std::filesystem::file_size(temp_filename);
  auto size2 = std::filesystem::file_size(temp_filename2);
  EXPECT_EQ(size, size2);

  std::remove(temp_filename.c_str());
  std::remove(temp_filename2.c_str());
  std::remove(temp_weights.c_str());
}

TEST(onnx_file, FileStream_ModelProto_WriteRead) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::string temp_filename = "test_tensor_file_stream_read.0.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.0.weight.tmp";

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

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
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_name().as_string(),
              model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}

TEST(onnx_external_ressource, LoadWithExternalData) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / "data" / "test_writing_external_weights_read_from_onnx.onnx";
  fs::path weights_path = source_dir / "data" / "test_writing_external_weights_read_from_onnx.data";
  if (!std::filesystem::exists(file_path) || !std::filesystem::exists(weights_path)) {
    GTEST_SKIP() << "File not found: " << file_path.string() << " or " << weights_path.string();
  }

  ModelProto model;
  utils::TwoFilesStream stream(file_path.string(), weights_path.string());
  ONNX_LIGHT_NAMESPACE::ParseOptions opts;
  model.ParseFromStream(stream, opts);
  EXPECT_EQ(model.ref_graph().ref_initializer().size(), 7);
  IteratorTensorProto it(&model.ref_graph());
  int big = 0;
  while (it.next()) {
    if (it->ref_dims().size() > 1) {
      EXPECT_TRUE(it->ref_raw_data().size() > 1024);
      ++big;
    }
  }
  EXPECT_EQ(big, 2);
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
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  const std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  const std::vector<float> data1 = {4.0f, 5.0f};

  TensorProto &w0 = graph.add_initializer();
  w0.set_name("w0");
  w0.set_data_type(TensorProto::DataType::FLOAT);
  w0.ref_dims().push_back(3);
  w0.ref_raw_data().resize(data0.size() * sizeof(float));
  std::memcpy(w0.ref_raw_data().data(), data0.data(), data0.size() * sizeof(float));

  TensorProto &w1 = graph.add_initializer();
  w1.set_name("w1");
  w1.set_data_type(TensorProto::DataType::FLOAT);
  w1.ref_dims().push_back(2);
  w1.ref_raw_data().resize(data1.size() * sizeof(float));
  std::memcpy(w1.ref_raw_data().data(), data1.data(), data1.size() * sizeof(float));

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

TEST(onnx_alignment, SerializeToFileWithAlignment) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  // Create three tensors with sizes that are NOT multiples of the alignment
  // so the padding behaviour is exercised.
  const std::vector<float> vals0(3, 1.0f); // 12 bytes
  const std::vector<float> vals1(5, 2.0f); // 20 bytes
  const std::vector<float> vals2(7, 3.0f); // 28 bytes

  auto add_tensor = [&](const std::string &name, const std::vector<float> &v) {
    TensorProto &t = graph.add_initializer();
    t.set_name(name);
    t.set_data_type(TensorProto::DataType::FLOAT);
    t.ref_dims().push_back(static_cast<int64_t>(v.size()));
    t.ref_raw_data().resize(v.size() * sizeof(float));
    std::memcpy(t.ref_raw_data().data(), v.data(), v.size() * sizeof(float));
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
  GraphProto &graph = model.add_graph();
  graph.set_name("g");

  const std::vector<float> vals(8, 42.0f); // 32 bytes
  TensorProto &t = graph.add_initializer();
  t.set_name("big");
  t.set_data_type(TensorProto::DataType::FLOAT);
  t.ref_dims().push_back(8);
  t.ref_raw_data().resize(vals.size() * sizeof(float));
  std::memcpy(t.ref_raw_data().data(), vals.data(), vals.size() * sizeof(float));

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
