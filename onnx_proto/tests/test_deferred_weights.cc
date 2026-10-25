#include "external_data_internal.h"
#include "onnx_helper.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

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
  void CloseDeferredDescriptor() {
    PinDeferredSource();
    external_file_ = std::make_shared<int>(-1);
  }
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

TEST_F(DeferredWeights, PinnedMetadataReadsDoNotRequireDescriptorReads) {
  const std::string metadata = "metadata spanning multiple read operations";
  const std::vector<uint8_t> payload(32768 * 4, 0xAB);
  utils::StringWriteStream encoded;
  encoded.write_variant_uint64(300);
  encoded.write_variant_uint64(UINT64_MAX);
  encoded.write_raw_bytes(reinterpret_cast<const uint8_t *>(metadata.data()), metadata.size());
  for (uint64_t i = 0; i < 8192; ++i)
    encoded.write_variant_uint64(16384 + i);
  encoded.write_raw_bytes(payload.data(), payload.size());
  encoded.write_variant_uint64(16384);
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
    ASSERT_TRUE(output.good());
  }
  CountingFileStream stream(path);
  stream.CloseDeferredDescriptor();
  EXPECT_EQ(stream.next_uint64(), 300);
  EXPECT_EQ(stream.next_uint64(), UINT64_MAX);
  stream.LimitToNext(metadata.size());
  uint8_t prefix[3];
  EXPECT_EQ(stream.read_bytes(sizeof(prefix), prefix), prefix);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(prefix), sizeof(prefix)),
            metadata.substr(0, sizeof(prefix)));
  const auto remaining = metadata.size() - sizeof(prefix);
  const auto *suffix = stream.read_bytes(remaining);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(suffix), remaining),
            metadata.substr(sizeof(prefix)));
  EXPECT_FALSE(stream.NotEnd());
  EXPECT_THROW(stream.next_uint64(), std::runtime_error);
  EXPECT_THROW(stream.read_bytes(1), std::runtime_error);
  stream.Restore();
  for (uint64_t i = 0; i < 8192; ++i)
    ASSERT_EQ(stream.next_uint64(), 16384 + i);
  stream.skip_bytes(payload.size());
  EXPECT_EQ(stream.bytes_skipped, static_cast<int64_t>(payload.size()));
  EXPECT_EQ(stream.next_uint64(), 16384);
  EXPECT_FALSE(stream.NotEnd());
  EXPECT_EQ(stream.tell(), encoded.size());
  EXPECT_THROW(stream.read_bytes(INT64_MAX), std::runtime_error);
  EXPECT_THROW(stream.skip_bytes(INT64_MAX), std::runtime_error);
  EXPECT_THROW(stream.read_bytes(-1), std::runtime_error);
  EXPECT_THROW(stream.skip_bytes(-1), std::runtime_error);
}

TEST_F(DeferredWeights, PinnedMetadataRejectsTruncatedAndOverlongVarints) {
  for (const std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{10}}) {
    SCOPED_TRACE(size);
    {
      std::ofstream output(path, std::ios::binary);
      const std::string bytes(size, '\x80');
      output.write(bytes.data(), bytes.size());
      ASSERT_TRUE(output.good());
    }
    CountingFileStream stream(path);
    stream.CloseDeferredDescriptor();
    EXPECT_THROW(stream.next_uint64(), std::runtime_error);
    EXPECT_EQ(stream.tell(), static_cast<int64_t>(size));
  }
}

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

TEST_F(DeferredWeights, StandaloneTensorExportsRejectBeforeWritingLoadedOrUnloadedPayloads) {
  auto original = MakeModel();
  std::string before;
  ASSERT_TRUE(original.SerializeToString(before));
  ASSERT_TRUE(original.SerializeToFile(path));
  auto model = Load();
  const auto file = utils::detail::OpenExternalDataFile(directory.string(), "model.onnx", true);
  for (const auto &name : {"A", "empty"}) {
    auto &tensor = model.ref_graph().ref_initializer()[std::string(name) == "A" ? 0 : 2];
    for (const bool loaded : {false, true}) {
      SCOPED_TRACE(std::string(name) + (loaded ? " loaded" : " unloaded"));
      if (loaded) {
        ASSERT_TRUE(model.LoadWeight(name));
      }
      EXPECT_THROW(tensor.SerializeToFileDescriptor(*file), std::runtime_error);
      EXPECT_THROW(tensor.SaveToFileDescriptor(*file), std::runtime_error);
      std::fstream output(path, std::ios::in | std::ios::out | std::ios::binary);
      EXPECT_THROW(tensor.SerializeToOstream(&output), std::runtime_error);
      utils::FdWriteStream stream(*file);
      EXPECT_THROW(tensor.SerializeToZeroCopyStream(&stream), std::runtime_error);
      std::ifstream input(path, std::ios::binary);
      const std::string after((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
      EXPECT_EQ(after, before);
    }
  }
}

TEST_F(DeferredWeights, RejectsExtensionOwnedWeightsWithoutChangingTheirBackingFile) {
  const auto source = MakeModel();
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    ModelProto original;
    if (kind == 0) {
      auto *block = original.ref_graph().add_extended_initializer()->ref_paged_cache().add_blocks();
      block->ref_key().CopyFrom(source.graph().initializer()[0]);
      block->ref_value().CopyFrom(source.graph().initializer()[1]);
    } else if (kind == 1) {
      auto &affine =
          original.ref_graph().add_extended_initializer()->ref_encoded_value().ref_affine();
      affine.ref_scale().CopyFrom(source.graph().initializer()[0]);
      affine.ref_zero_point().CopyFrom(source.graph().initializer()[1]);
    } else {
      auto &structure = kind == 2 ? original.add_struct_types()->ref_structure()
                                  : original.ref_graph()
                                        .add_input()
                                        ->ref_type()
                                        .ref_sequence_type()
                                        .ref_elem_type()
                                        .ref_struct_type()
                                        .ref_structure();
      structure.add_field()->ref_constant().CopyFrom(source.graph().initializer()[0]);
    }
    std::string before;
    ASSERT_TRUE(original.SerializeToString(before));
    ASSERT_TRUE(original.SerializeToFile(path));
    EXPECT_THROW(Load(), std::runtime_error);
    std::ifstream input(path, std::ios::binary);
    const std::string after((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    EXPECT_EQ(after, before);
    ModelProto eager;
    utils::FileStream stream(path);
    ParseOptions options;
    EXPECT_TRUE(eager.ParseFromStream(stream, options));
  }
}

TEST_F(DeferredWeights, CopiedExtensionSourcesRejectBeforeDestinationIO) {
  ASSERT_TRUE(MakeModel().SerializeToFile(path));
  std::ifstream before_stream(path, std::ios::binary);
  const std::string before((std::istreambuf_iterator<char>(before_stream)),
                           std::istreambuf_iterator<char>());
  const auto file = utils::detail::OpenExternalDataFile(directory.string(), "model.onnx", true);
  for (const bool loaded : {false, true}) {
    auto source = Load();
    if (loaded)
      source.LoadWeights();
    for (int kind = 0; kind < 9; ++kind) {
      SCOPED_TRACE(loaded);
      SCOPED_TRACE(kind);
      ModelProto model;
      TensorProto *tensor;
      if (kind == 0) {
        tensor = &model.ref_graph()
                      .add_extended_initializer()
                      ->ref_paged_cache()
                      .add_blocks()
                      ->ref_key();
      } else if (kind == 1) {
        tensor = &model.ref_graph()
                      .add_extended_initializer()
                      ->ref_encoded_value()
                      .ref_affine()
                      .ref_scale();
      } else if (kind == 2) {
        tensor = &model.add_struct_types()->ref_structure().add_field()->ref_constant();
      } else if (kind == 3) {
        tensor = &model.ref_graph()
                      .add_input()
                      ->ref_type()
                      .ref_sequence_type()
                      .ref_elem_type()
                      .ref_struct_type()
                      .ref_structure()
                      .add_field()
                      ->ref_constant();
      } else if (kind == 4 || kind == 5) {
        auto &attribute = *model.ref_graph().add_node()->add_attribute();
        auto &type = kind == 4 ? attribute.ref_tp() : *attribute.add_type_protos();
        tensor = &type.ref_struct_type().ref_structure().add_field()->ref_constant();
      } else if (kind == 6 || kind == 7) {
        auto &function = kind == 6 ? *model.add_functions()
                                   : *model.ref_rewritings()["plan"].add_added_functions();
        tensor = &function.add_value_info()
                      ->ref_type()
                      .ref_struct_type()
                      .ref_structure()
                      .add_field()
                      ->ref_constant();
      } else {
        tensor = &model.ref_graph()
                      .add_node()
                      ->add_attribute()
                      ->ref_g()
                      .add_extended_initializer()
                      ->ref_paged_cache()
                      .add_blocks()
                      ->ref_value();
      }
      tensor->CopyFrom(source.graph().initializer()[0]);
      ASSERT_TRUE(tensor->HasDeferredWeight());
      for (const bool mixed : {false, true}) {
        SCOPED_TRACE(mixed);
        if (mixed)
          model.ref_graph().add_initializer()->CopyFrom(source.graph().initializer()[1]);
        EXPECT_THROW(model.ValidateWeightSavePath(path), std::runtime_error);
        EXPECT_THROW(model.SerializeToFile(path), std::runtime_error);
        const auto output = (directory / "rejected.onnx").string();
        EXPECT_THROW(model.SerializeToFile(output), std::runtime_error);
        EXPECT_FALSE(std::filesystem::exists(output));
        EXPECT_THROW(model.SerializeToFileDescriptor(*file), std::runtime_error);
        utils::StringWriteStream stream;
        SerializeOptions options;
        EXPECT_THROW(SerializeModelProtoToStream(model, stream, options), std::runtime_error);
        EXPECT_EQ(stream.size(), 0);
        std::ifstream input(path, std::ios::binary);
        EXPECT_EQ(
            std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()),
            before);
      }
    }
  }
}

TEST_F(DeferredWeights, PipelineExportsRejectSourcesInEveryStageBeforeWriting) {
  auto original = MakeModel();
  original.ref_graph().add_node();
  std::string before;
  ASSERT_TRUE(original.SerializeToString(before));
  ASSERT_TRUE(original.SerializeToFile(path));
  const auto file = utils::detail::OpenExternalDataFile(directory.string(), "model.onnx", true);
  for (const bool loaded : {false, true}) {
    auto model = Load();
    if (loaded)
      model.LoadWeights();
    for (int stage = 0; stage < 3; ++stage) {
      SCOPED_TRACE(stage);
      SCOPED_TRACE(loaded);
      PipelineProto pipeline;
      auto &part = stage == 0   ? pipeline.ref_embedding()
                   : stage == 1 ? pipeline.ref_prefill()
                                : pipeline.ref_decode();
      part.CopyFrom(model);
      std::string output;
      EXPECT_THROW(pipeline.SerializeToString(output), std::runtime_error);
      EXPECT_THROW(pipeline.SerializeToFileDescriptor(*file), std::runtime_error);
      int callbacks = 0;
      SerializeOptions options;
      options.node_callback = [&callbacks](NodeProto &, GraphProto &graph) {
        ++callbacks;
        graph.ref_initializer().clear();
      };
      EXPECT_THROW(pipeline.SerializeToString(output, options), std::runtime_error);
      EXPECT_THROW(pipeline.SerializeToFileDescriptor(*file, options), std::runtime_error);
      EXPECT_EQ(callbacks, 0);
      std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
      EXPECT_THROW(pipeline.SerializeToOstream(&stream), std::runtime_error);
      utils::FdWriteStream writer(*file);
      EXPECT_THROW(pipeline.SerializeToZeroCopyStream(&writer), std::runtime_error);
      std::ifstream input(path, std::ios::binary);
      EXPECT_EQ(
          std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()),
          before);
    }
  }
}

TEST_F(DeferredWeights, SubmessageExportsRejectNestedSourcesBeforeWriting) {
  auto original = MakeModel();
  auto &graph = original.ref_graph();
  auto &attribute = *graph.add_node()->add_attribute();
  attribute.ref_t().CopyFrom(graph.initializer()[1]);
  attribute.ref_t().set_name("attribute");
  auto &sparse = *graph.add_sparse_initializer();
  sparse.ref_values().CopyFrom(graph.initializer()[1]);
  sparse.ref_values().set_name("sparse_values");
  auto &indices = sparse.ref_indices();
  indices.set_name("sparse_indices");
  indices.set_data_type(TensorProto::INT64);
  indices.add_dims(1);
  indices.ref_raw_data() = std::vector<uint8_t>(8, 0);
  sparse.add_dims(1);
  ASSERT_TRUE(original.SerializeToFile(path));
  std::ifstream before_stream(path, std::ios::binary);
  const std::string before((std::istreambuf_iterator<char>(before_stream)),
                           std::istreambuf_iterator<char>());
  const auto file = utils::detail::OpenExternalDataFile(directory.string(), "model.onnx", true);
  for (const bool loaded : {false, true}) {
    SCOPED_TRACE(loaded);
    auto model = Load();
    if (loaded)
      model.LoadWeights();
    const auto check = [&](const auto &value) {
      EXPECT_THROW(value.SerializeToFileDescriptor(*file), std::runtime_error);
      EXPECT_THROW(value.SaveToFileDescriptor(*file), std::runtime_error);
      std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
      EXPECT_THROW(value.SerializeToOstream(&stream), std::runtime_error);
      utils::FdWriteStream writer(*file);
      EXPECT_THROW(value.SerializeToZeroCopyStream(&writer), std::runtime_error);
    };
    check(model.graph());
    check(model.graph().node()[0]);
    check(model.graph().node()[0].attribute()[0]);
    check(model.graph().sparse_initializer()[0]);
    GraphProto nested;
    nested.add_node()->add_attribute()->ref_g().add_initializer()->CopyFrom(
        model.graph().initializer()[1]);
    check(nested);
    SequenceProto sequence;
    sequence.add_tensor_values()->CopyFrom(model.graph().initializer()[1]);
    check(sequence);
    MapProto map;
    map.ref_values().add_tensor_values()->CopyFrom(model.graph().initializer()[1]);
    check(map);
    OptionalProto optional;
    optional.ref_tensor_value().CopyFrom(model.graph().initializer()[1]);
    check(optional);
    std::ifstream input(path, std::ios::binary);
    EXPECT_EQ(
        std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()),
        before);
    if (loaded) {
      std::string serialized;
      ASSERT_TRUE(model.graph().SerializeToString(serialized));
      GraphProto eager;
      ASSERT_TRUE(eager.ParseFromString(serialized));
      std::ostringstream output;
      ASSERT_TRUE(eager.SerializeToOstream(&output));
      EXPECT_EQ(output.str(), serialized);
    }
  }
}

TEST_F(DeferredWeights, NodeCallbacksPreserveDeferredAttributeAndSubgraphSources) {
  auto original = MakeModel();
  auto *node = original.ref_graph().add_node();
  auto *attribute = node->add_attribute();
  attribute->ref_t().CopyFrom(original.graph().initializer()[1]);
  attribute->ref_t().set_name("attribute");
  auto *nested = node->add_attribute()->ref_g().add_initializer();
  nested->CopyFrom(original.graph().initializer()[1]);
  nested->set_name("nested");
  ASSERT_TRUE(original.SerializeToFile(path));
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    auto model = Load();
    model.LoadWeights();
    SerializeOptions options;
    options.node_callback = [mode](NodeProto &value, GraphProto &) {
      if (mode) {
        value.set_doc_string("serialized only");
        value.ref_attribute()[0].ref_t().ref_raw_data().data()[0] = 0x42;
      }
      if (mode == 2)
        throw std::runtime_error("callback failure");
    };
    std::string serialized;
    if (mode == 2)
      EXPECT_THROW(model.SerializeToString(serialized, options), std::runtime_error);
    else
      ASSERT_TRUE(model.SerializeToString(serialized, options));
    const auto &restored = model.graph().node()[0];
    EXPECT_FALSE(restored.has_doc_string());
    EXPECT_EQ(restored.attribute()[0].t().raw_data().data()[0], 0xCD);
    for (const std::string name : {"attribute", "nested"}) {
      EXPECT_TRUE(model.UnloadWeight(name));
      EXPECT_TRUE(model.LoadWeight(name));
    }
    const auto output = (directory / "callback.onnx").string();
    if (mode == 2)
      EXPECT_THROW(model.SerializeToFile(output, options), std::runtime_error);
    else
      ASSERT_TRUE(model.SerializeToFile(output, options));
    for (const std::string name : {"attribute", "nested"}) {
      EXPECT_TRUE(model.UnloadWeight(name));
      EXPECT_TRUE(model.LoadWeight(name));
    }
  }
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
