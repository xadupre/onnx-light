#include "onnx_helper.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

class DeferredWeights : public testing::Test {
protected:
  std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("onnx_light_deferred_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::string path;

  void SetUp() override {
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    path = (directory / "model.onnx").string();
  }
  void TearDown() override { std::filesystem::remove_all(directory); }

  ModelProto MakeModel() {
    ModelProto model;
    auto *a = model.ref_graph().add_initializer();
    a->set_name("A");
    a->set_data_type(TensorProto::FLOAT);
    a->add_dims(32768);
    a->ref_raw_data() = std::vector<uint8_t>(32768 * 4, 0xAB);
    auto *b = model.ref_graph().add_initializer();
    b->set_name("B");
    b->set_data_type(TensorProto::FLOAT);
    b->add_dims(1);
    b->ref_raw_data() = std::vector<uint8_t>(4, 0xCD);
    auto *empty = model.ref_graph().add_initializer();
    empty->set_name("empty");
    empty->set_data_type(TensorProto::FLOAT);
    empty->add_dims(0);
    return model;
  }

  ModelProto Load(ParseOptions options = {}) {
    options.defer_weights = true;
    options.num_threads = 1;
    ModelProto model;
    utils::FileStream stream(path);
    EXPECT_TRUE(model.ParseFromStream(stream, options));
    return model;
  }
};

class CountingFileStream : public utils::FileStream {
public:
  using utils::FileStream::FileStream;
  int64_t bytes_read = 0;
  int64_t bytes_skipped = 0;
  const uint8_t *read_bytes(utils::offset_t size, uint8_t *buffer = nullptr) override {
    bytes_read += size;
    return utils::FileStream::read_bytes(size, buffer);
  }
  void skip_bytes(utils::offset_t size) override {
    EXPECT_EQ(read_buf_end_, read_buf_pos_) << "Read-ahead must not fetch skipped raw payloads.";
    bytes_skipped += size;
    utils::FileStream::skip_bytes(size);
  }
};

TEST_F(DeferredWeights, InlineRangesAreIndependentAndReloadableWithoutPayloadReads) {
  auto original = MakeModel();
  ASSERT_TRUE(original.SerializeToFile(path));
  CountingFileStream stream(path);
  ParseOptions options;
  options.defer_weights = true;
  options.num_threads = 1;
  options.raw_data_threshold = 1 << 20;
  ModelProto model;
  ASSERT_TRUE(model.ParseFromStream(stream, options));
  EXPECT_LT(stream.bytes_read, 1000);
  EXPECT_GE(stream.bytes_skipped, 32768 * 4 + 4);
  EXPECT_FALSE(model.IsWeightLoaded("A"));
  EXPECT_FALSE(model.IsWeightLoaded("B"));
  EXPECT_FALSE(model.IsWeightLoaded("empty"));
  EXPECT_FALSE(model.UnloadWeight("A"));
  EXPECT_TRUE(model.LoadWeight("A"));
  EXPECT_FALSE(model.LoadWeight("A"));
  EXPECT_FALSE(model.IsWeightLoaded("B"));
  auto &raw = model.ref_graph().ref_initializer()[0].ref_raw_data();
  EXPECT_EQ(raw.size(), 32768 * 4);
  EXPECT_EQ(raw.data()[raw.size() - 1], 0xAB);
  const uint8_t *retained = raw.data();
  auto guard = raw.acquire_export_guard();
  EXPECT_TRUE(model.UnloadWeight("A"));
  EXPECT_EQ(raw.data(), nullptr);
  EXPECT_EQ(raw.size(), 0);
  EXPECT_EQ(retained[32768 * 4 - 1], 0xAB);
  guard.reset();
  EXPECT_TRUE(model.LoadWeight("empty"));
  EXPECT_TRUE(model.IsWeightLoaded("empty"));
  EXPECT_FALSE(model.LoadWeight("empty"));
  EXPECT_TRUE(model.UnloadWeight("empty"));
  EXPECT_TRUE(model.LoadWeight("A"));
  EXPECT_EQ(raw.data()[0], 0xAB);

  ModelProto copy;
  copy.CopyFrom(model);
  EXPECT_NE(copy.graph().initializer()[0].raw_data().data(),
            model.graph().initializer()[0].raw_data().data());
  ModelProto ordinary_copy = model;
  EXPECT_TRUE(ordinary_copy.LoadWeight("B"));
  auto shared = utils::CopyWithSharedTensorStorage(model);
  EXPECT_TRUE(model.UnloadWeight("A"));
  EXPECT_TRUE(copy.IsWeightLoaded("A"));
  EXPECT_TRUE(shared.IsWeightLoaded("A"));
  EXPECT_TRUE(copy.LoadWeight("B"));
  EXPECT_TRUE(shared.LoadWeight("B"));
  EXPECT_FALSE(model.IsWeightLoaded("B"));
  EXPECT_THROW(model.LoadWeight("missing"), std::runtime_error);
  EXPECT_THROW(original.UnloadWeight("A"), std::runtime_error);
}

TEST_F(DeferredWeights, SharedExternalRangesRetainMetadataAndSupportSingleFileExport) {
  auto original = MakeModel();
  const std::string weights = (directory / "weights.bin").string();
  {
    utils::TwoFilesWriteStream stream(path, weights);
    SerializeOptions options;
    options.raw_data_threshold = 0;
    options.num_threads = 1;
    ASSERT_TRUE(SerializeModelProtoToStream(original, stream, options));
  }
  auto model = Load();
  EXPECT_TRUE(model.LoadWeight("B"));
  EXPECT_FALSE(model.IsWeightLoaded("A"));
  EXPECT_EQ(model.graph().initializer()[1].raw_data().data()[0], 0xCD);
  const auto metadata = model.graph().initializer()[1].external_data();
  EXPECT_TRUE(model.UnloadWeight("B"));
  EXPECT_EQ(model.graph().initializer()[1].external_data().size(), metadata.size());
  EXPECT_THROW(model.SerializeToFile(path), std::runtime_error);
  const auto output = (directory / "export.onnx").string();
  EXPECT_THROW(model.SerializeToFile(output), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(output));
  EXPECT_EQ(model.LoadWeights(), 3);
  EXPECT_EQ(model.LoadWeights(), 0);
  EXPECT_THROW(model.SerializeToFile(path), std::runtime_error);
  EXPECT_THROW(model.SerializeToFile(weights), std::runtime_error);
  ASSERT_TRUE(model.SerializeToFile(output));
  ModelProto eager;
  utils::FileStream stream(output);
  ParseOptions options;
  ASSERT_TRUE(eager.ParseFromStream(stream, options));
  EXPECT_FALSE(eager.HasDeferredWeights());
  EXPECT_FALSE(eager.graph().initializer()[0].has_data_location());
  EXPECT_TRUE(eager.graph().initializer()[0].external_data().empty());
  EXPECT_EQ(eager.graph().initializer()[0].raw_data().size(), 32768 * 4);
  EXPECT_EQ(eager.graph().initializer()[1].raw_data().data()[0], 0xCD);
  EXPECT_EQ(model.UnloadWeights(), 3);
  EXPECT_EQ(model.UnloadWeights(), 0);
}

TEST_F(DeferredWeights, RejectsChangedMissingAndTruncatedSourcesIncludingEmptyTensors) {
  ASSERT_TRUE(MakeModel().SerializeToFile(path));
  auto model = Load();
  {
    std::ofstream file(path, std::ios::app | std::ios::binary);
    file.put('x');
  }
  EXPECT_THROW(model.LoadWeight("A"), std::runtime_error);
  EXPECT_THROW(model.LoadWeight("empty"), std::runtime_error);
  EXPECT_FALSE(model.IsWeightLoaded("A"));
  ASSERT_TRUE(MakeModel().SerializeToFile(path));
  model = Load();
  const auto modified = std::filesystem::last_write_time(path);
  std::filesystem::last_write_time(path, modified + std::chrono::seconds(2));
  EXPECT_THROW(model.LoadWeight("B"), std::runtime_error);
  ASSERT_TRUE(MakeModel().SerializeToFile(path));
  model = Load();
  std::filesystem::resize_file(path, 1);
  EXPECT_THROW(model.LoadWeight("B"), std::runtime_error);
  std::filesystem::remove(path);
  EXPECT_THROW(model.LoadWeight("empty"), std::runtime_error);
}

TEST_F(DeferredWeights, ResolvesNestedAttributeNamesAndRejectsAmbiguity) {
  auto original = MakeModel();
  auto *attribute = original.ref_graph().add_node()->add_attribute();
  attribute->ref_g().add_initializer()->CopyFrom(original.graph().initializer()[1]);
  attribute->ref_g().ref_initializer()[0].set_name("nested");
  auto *constant = original.ref_graph().add_node()->add_attribute()->add_t();
  constant->CopyFrom(original.graph().initializer()[1]);
  constant->set_name("constant");
  ASSERT_TRUE(original.SerializeToFile(path));
  auto model = Load();
  EXPECT_EQ(model.WeightNames().size(), 5);
  EXPECT_TRUE(model.LoadWeight("nested"));
  EXPECT_TRUE(model.LoadWeight("constant"));
  EXPECT_FALSE(model.IsWeightLoaded("B"));
  model.ref_graph().ref_node()[0].ref_attribute()[0].ref_g().ref_initializer()[0].set_name("B");
  EXPECT_THROW(model.IsWeightLoaded("B"), std::runtime_error);
  EXPECT_THROW(model.LoadWeight("B"), std::runtime_error);
  EXPECT_THROW(model.UnloadWeight("B"), std::runtime_error);
}

TEST_F(DeferredWeights, RejectsUnsupportedTypedPayloadsStreamsAndResourceLimits) {
  auto original = MakeModel();
  auto &tensor = original.ref_graph().ref_initializer()[0];
  tensor.ref_raw_data() = utils::ByteSpan{};
  tensor.ref_dims()[0] = 1;
  tensor.add_float_data(1.0f);
  ASSERT_TRUE(original.SerializeToFile(path));
  EXPECT_THROW(Load(), std::runtime_error);
  ASSERT_TRUE(MakeModel().SerializeToFile(path));
  ParseOptions options;
  options.max_tensor_size_bytes = 4;
  EXPECT_THROW(Load(options), std::runtime_error);
  options = ParseOptions{};
  options.no_copy = true;
  EXPECT_THROW(Load(options), std::runtime_error);
  std::string serialized;
  ASSERT_TRUE(MakeModel().SerializeToString(serialized));
  utils::StringStream stream(serialized.data(), static_cast<int64_t>(serialized.size()));
  options.no_copy = false;
  options.defer_weights = true;
  ModelProto model;
  EXPECT_THROW(model.ParseFromStream(stream, options), std::runtime_error);
}

TEST_F(DeferredWeights, RejectsExternalTraversalLinksAndInvalidZeroLengthRanges) {
  ModelProto original;
  auto *tensor = original.ref_graph().add_initializer();
  tensor->set_name("zero");
  tensor->set_data_type(TensorProto::FLOAT);
  tensor->add_dims(0);
  tensor->set_data_location(TensorProto::EXTERNAL);
  auto *location = tensor->add_external_data();
  location->set_key("location");
  location->set_value("../outside");
  auto *length = tensor->add_external_data();
  length->set_key("length");
  length->set_value("0");
  ASSERT_TRUE(original.SerializeToFile(path));
  EXPECT_THROW(Load(), std::runtime_error);
  location->set_value("weights.bin");
  const auto weights = directory / "weights.bin";
  {
    std::ofstream file(weights);
  }
  auto *offset = tensor->add_external_data();
  offset->set_key("offset");
  offset->set_value("9223372036854775807");
  ASSERT_TRUE(original.SerializeToFile(path));
  EXPECT_THROW(Load(), std::runtime_error);
  offset->set_value("0");
  ASSERT_TRUE(original.SerializeToFile(path));
  EXPECT_TRUE(Load().LoadWeight("zero"));
#ifndef _WIN32
  std::filesystem::create_hard_link(weights, directory / "hardlink.bin");
  EXPECT_THROW(Load(), std::runtime_error);
  std::filesystem::remove(directory / "hardlink.bin");
  location->set_value("symlink.bin");
  std::filesystem::create_symlink(weights, directory / "symlink.bin");
  ASSERT_TRUE(original.SerializeToFile(path));
  EXPECT_THROW(Load(), std::runtime_error);
#endif
}

} // namespace
