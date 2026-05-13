#include "onnx_helper.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

bool is_aligned(const void *ptr, size_t align) {
  return align == 0 || (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
}

int64_t get_external_i64(const TensorProto &tensor, const char *key) {
  for (const auto &entry : tensor.ref_external_data()) {
    if (entry.ref_key() == key) {
      return entry.ref_value().toint64();
    }
  }
  return -1;
}

std::string get_external_location(const TensorProto &tensor) {
  for (const auto &entry : tensor.ref_external_data()) {
    if (entry.ref_key() == "location") {
      return entry.ref_value().as_string();
    }
  }
  return "";
}

} // namespace

TEST(onnx_alignment_options, AlignmentMustBePowerOfTwo) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  TensorProto *t = graph->add_initializer();
  t->set_name("w0");
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->ref_dims().push_back(2);
  t->ref_raw_data() = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8};

  SerializeOptions bad_sopts;
  bad_sopts.raw_data_threshold = 0;
  bad_sopts.alignment = 3;
  std::string serialized_model;
  std::unordered_map<std::string, std::string> external_files;
  EXPECT_THROW(
      model.SerializeToString(serialized_model, external_files, 1024, "weights", bad_sopts),
      std::runtime_error);

  TensorProto tensor;
  tensor.set_name("inline");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_raw_data() = std::vector<uint8_t>{9, 10, 11, 12, 13, 14, 15, 16};
  std::string serialized_tensor;
  tensor.SerializeToString(serialized_tensor);

  ParseOptions bad_popts;
  bad_popts.alignment = 6;
  TensorProto parsed;
  EXPECT_THROW(parsed.ParseFromString(serialized_tensor, bad_popts), std::runtime_error);
}

TEST(onnx_alignment_options, SerializeToStringAlignmentSplitExternalFiles) {
  constexpr int64_t align = 16;
  constexpr int64_t max_external_file_size = 48;

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  std::vector<std::vector<uint8_t>> payloads = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
      {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
      {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
      {36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
  };

  for (size_t i = 0; i < payloads.size(); ++i) {
    TensorProto *t = graph->add_initializer();
    t->set_name("w" + std::to_string(i));
    t->set_data_type(TensorProto::DataType::UINT8);
    t->ref_dims().push_back(static_cast<int64_t>(payloads[i].size()));
    t->ref_raw_data() = payloads[i];
  }

  SerializeOptions sopts;
  sopts.raw_data_threshold = 0;
  sopts.alignment = align;
  std::string serialized;
  std::unordered_map<std::string, std::string> external_files;
  model.SerializeToString(serialized, external_files, static_cast<size_t>(max_external_file_size),
                          "weights_part", sopts);

  EXPECT_GE(external_files.size(), 2u);

  ModelProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_EQ(parsed.ref_graph().ref_initializer().size(), payloads.size());

  for (size_t i = 0; i < payloads.size(); ++i) {
    const TensorProto &t = parsed.ref_graph().ref_initializer()[i];
    const std::string location = get_external_location(t);
    const int64_t off = get_external_i64(t, "offset");
    const int64_t len = get_external_i64(t, "length");
    ASSERT_FALSE(location.empty());
    ASSERT_GE(off, 0);
    ASSERT_EQ(off % align, 0) << "tensor " << i << " offset=" << off;
    ASSERT_EQ(len, static_cast<int64_t>(payloads[i].size()));
    ASSERT_EQ(external_files.count(location), 1u);
    const std::string &buffer = external_files.at(location);
    ASSERT_GE(static_cast<int64_t>(buffer.size()), off + len);
    EXPECT_EQ(std::memcmp(buffer.data() + off, payloads[i].data(), payloads[i].size()), 0);
  }
}

TEST(onnx_alignment_options, ParseAlignmentIncompatibleWithExternalDataWarnsOrErrorsNoCopy) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  const std::vector<float> vals0(4, 1.0f); // 16 bytes
  const std::vector<float> vals1(4, 2.0f); // 16 bytes
  auto add_tensor = [&](const std::string &name, const std::vector<float> &vals) {
    TensorProto *t = graph->add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->ref_dims().push_back(static_cast<int64_t>(vals.size()));
    t->ref_raw_data().resize(vals.size() * sizeof(float));
    std::memcpy(t->ref_raw_data().data(), vals.data(), vals.size() * sizeof(float));
  };
  add_tensor("w0", vals0);
  add_tensor("w1", vals1);

  const std::string onnx_file = "test_parse_alignment_incompatible.onnx";
  const std::string weights_file = "test_parse_alignment_incompatible.data";
  {
    utils::TwoFilesWriteStream wstream(onnx_file, weights_file);
    SerializeOptions sopts;
    sopts.raw_data_threshold = 0;
    sopts.alignment = 16;
    SerializeProtoToStream(model, wstream, sopts);
  }

  ModelProto loaded;
  std::string stderr_text;
  {
    testing::internal::CaptureStderr();
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ropts.alignment = 64;
    ParseProtoFromStream(loaded, rstream, ropts);
    stderr_text = testing::internal::GetCapturedStderr();
  }
  EXPECT_NE(stderr_text.find("Warning:"), std::string::npos);
  EXPECT_NE(stderr_text.find("incompatible with ParseOptions.alignment"), std::string::npos);
  ASSERT_EQ(loaded.ref_graph().ref_initializer().size(), 2u);
  EXPECT_TRUE(loaded.ref_graph().ref_initializer()[0].ref_raw_data().is_aligned_owned());
  EXPECT_TRUE(is_aligned(loaded.ref_graph().ref_initializer()[0].ref_raw_data().data(), 64));
  EXPECT_TRUE(loaded.ref_graph().ref_initializer()[1].ref_raw_data().is_aligned_owned());
  EXPECT_TRUE(is_aligned(loaded.ref_graph().ref_initializer()[1].ref_raw_data().data(), 64));

  ModelProto loaded_fail;
  {
    utils::TwoFilesStream rstream(onnx_file, weights_file);
    ParseOptions ropts;
    ropts.alignment = 64;
    ropts.no_copy = true;
    EXPECT_THROW(ParseProtoFromStream(loaded_fail, rstream, ropts), std::runtime_error);
  }

  std::remove(onnx_file.c_str());
  std::remove(weights_file.c_str());
}
