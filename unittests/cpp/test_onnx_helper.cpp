#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace onnx;
using namespace onnx::utils;

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
  onnx::ParseOptions opts;
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
  onnx::ParseOptions opts;
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
