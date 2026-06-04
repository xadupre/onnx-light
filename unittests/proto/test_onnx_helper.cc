#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

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

  fs::remove_all(root);
}

TEST(onnx_external_ressource, SaveWithExternalDataLocationMustBeNextToModel) {
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
  fs::create_directories(root);
  fs::path model_path = root / "model.onnx";
  fs::path weights_path = root / "weights.data";

  {
    utils::TwoFilesWriteStream stream(model_path.string(), weights_path.string());
    SerializeOptions wopts;
    wopts.raw_data_threshold = 0;
    EXPECT_THROW(SerializeProtoToStream(model, stream, wopts), std::exception);
  }

  fs::remove_all(root);
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
  EXPECT_EQ(i0.ref_external_data()[0].ref_value().as_string(), "weights_part_0.data");
  EXPECT_EQ(i1.ref_external_data()[0].ref_value().as_string(), "weights_part_0.data");
  EXPECT_EQ(i2.ref_external_data()[0].ref_value().as_string(), "weights_part_1.data");
  EXPECT_EQ(i0.ref_external_data()[1].ref_value().as_string(), "0");
  EXPECT_EQ(i1.ref_external_data()[1].ref_value().as_string(), "4");
  EXPECT_EQ(i2.ref_external_data()[1].ref_value().as_string(), "0");
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
    ASSERT_TRUE(t.has_data_location()) << "name=" << t.ref_name().as_string();
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
      const std::string loc = t.ref_external_data()[0].ref_value().as_string();
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
  ASSERT_EQ(w1_loaded.ref_name().as_string(), "W1");
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
  ASSERT_EQ(w2_loaded.ref_name().as_string(), "W2");
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
  AddAttribute<std::string>(node, "s", std::string("abc"));
  AddAttribute<std::vector<int64_t>>(node, "ints", {1, 2, 3});
  AddAttribute<std::vector<float>>(node, "floats", {0.5f, 1.5f});
  AddAttribute<std::vector<std::string>>(node, "strings", {"a", "b"});

  ASSERT_EQ(node.attribute().size(), 6);
  EXPECT_EQ(node.attribute()[0].type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(node.attribute()[0].i(), 7);
  EXPECT_EQ(node.attribute()[1].type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(node.attribute()[1].f(), 1.5f);
  EXPECT_EQ(node.attribute()[2].type(), AttributeProto::AttributeType::STRING);
  EXPECT_EQ(node.attribute()[2].s(), utils::String("abc"));
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
    for (int j = 0; j < meta.ref_external_data().size(); ++j) {
      const StringStringEntryProto &e = meta.ref_external_data()[j];
      if (e.ref_key() == "location")
        loc = e.ref_value().as_string();
      else if (e.ref_key() == "offset")
        off = std::stoll(e.ref_value().as_string());
      else if (e.ref_key() == "length" || e.ref_key() == "size")
        len = std::stoll(e.ref_value().as_string());
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
