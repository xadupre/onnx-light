// Unit tests for the protobuf-compatible message API surface that onnx-light
// exposes on generated proto classes: ParseFromString / SerializeToString now
// return bool (true on success) and ByteSizeLong() reports the serialized size
// without performing a real serialization (it delegates to SerializeSize()).
#include "external_data_internal.h"
#include "google_protobuf_compat.h"
#include "onnx.h"
#include "onnx_helper.h"
#include "windows_junction.h"
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Builds a small AttributeProto carrying an int payload.
AttributeProto MakeIntAttr() {
  AttributeProto attr;
  attr.set_name("i_attr");
  attr.ref_i() = 123456789;
  return attr;
}

int OpenTempFdForWrite(const std::string &path) {
#if defined(_WIN32)
  int fd = -1;
  _sopen_s(&fd, path.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _SH_DENYNO,
           _S_IREAD | _S_IWRITE);
  return fd;
#else
  return ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
}

int OpenTempFdForRead(const std::string &path) {
#if defined(_WIN32)
  int fd = -1;
  _sopen_s(&fd, path.c_str(), _O_RDONLY | _O_BINARY, _SH_DENYNO, _S_IREAD);
  return fd;
#else
  return ::open(path.c_str(), O_RDONLY);
#endif
}

void CloseFd(int fd) {
#if defined(_WIN32)
  _close(fd);
#else
  ::close(fd);
#endif
}

void WriteWholeFile(const std::string &path, const std::string &data) {
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

int64_t SeekFd(int fd, int64_t offset) {
#if defined(_WIN32)
  return ::_lseeki64(fd, offset, SEEK_SET);
#else
  return ::lseek(fd, static_cast<off_t>(offset), SEEK_SET);
#endif
}

int64_t TellFd(int fd) {
#if defined(_WIN32)
  return ::_lseeki64(fd, 0, SEEK_CUR);
#else
  return ::lseek(fd, 0, SEEK_CUR);
#endif
}

std::string EncodeVarint(uint64_t value) {
  std::string encoded;
  while (value >= 0x80) {
    encoded.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  encoded.push_back(static_cast<char>(value));
  return encoded;
}

std::string BuildNestedTypeProto(int levels) {
  std::string inner;
  for (int i = 0; i < levels; ++i) {
    std::string sequence = EncodeVarint((1 << 3) | 2);
    sequence += EncodeVarint(inner.size());
    sequence += inner;
    inner = EncodeVarint((4 << 3) | 2) + EncodeVarint(sequence.size()) + sequence;
  }
  return inner;
}

std::string ReadWholeFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

class ExternalDataDescriptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    root = std::string("test_external_data_anchored_streams_") +
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model" / "sub");
    std::filesystem::create_directories(root / "outside");
    WriteWholeFile((root / "model" / "model.onnx").string(), "model");
    WriteWholeFile((root / "model" / "sub" / "weights.bin").string(), "original");
    WriteWholeFile((root / "model" / "sub" / "extra.bin").string(), "extra!!!");
    WriteWholeFile((root / "outside" / "weights.bin").string(), "sentinel");
    WriteWholeFile((root / "outside" / "extra.bin").string(), "sentinel");
  }
  void TearDown() override { std::filesystem::remove_all(root); }
  std::filesystem::path root;
};

TEST_F(ExternalDataDescriptorTest, RejectsUnsafeObjectsBeforeTruncation) {
  using utils::detail::OpenExternalDataFile;
  const auto base = (root / "model").string();
  EXPECT_ANY_THROW(OpenExternalDataFile(base, "../outside/weights.bin", true));
  EXPECT_ANY_THROW(OpenExternalDataFile(base, "sub", true));
  EXPECT_ANY_THROW(OpenExternalDataFile(base, "sub"));
  std::filesystem::create_hard_link(root / "outside" / "weights.bin",
                                    root / "model" / "linked.bin");
  EXPECT_ANY_THROW(OpenExternalDataFile(base, "linked.bin", true));
  EXPECT_EQ(ReadWholeFile((root / "outside" / "weights.bin").string()), "sentinel");
  auto created = OpenExternalDataFile(base, "sub/new.bin", true);
  utils::FileWriteStream output("unused", created);
  EXPECT_EQ(utils::FileStream("unused", created).size(), 0);
  output.write_raw_bytes_at_offset(reinterpret_cast<const uint8_t *>("data"), 4, 3);
  utils::FileStream input("unused", created);
  EXPECT_EQ(input.size(), 7);
  input.skip_bytes(7);
  char byte = 'x';
  EXPECT_ANY_THROW(input.read_bytes(1, reinterpret_cast<uint8_t *>(&byte)));
  EXPECT_EQ(byte, 'x');
  EXPECT_ANY_THROW(output.write_raw_bytes_at_offset(reinterpret_cast<const uint8_t *>("x"), 1, -1));
  utils::FileWriteStream truncated("unused", created);
  EXPECT_EQ(utils::FileStream("unused", created).size(), 0);
}

TEST_F(ExternalDataDescriptorTest, SidecarWritesRejectLinksBeforeTruncation) {
  const auto base = root / "model";
  const auto victim = std::filesystem::absolute(root / "outside" / "weights.bin");
  const auto sidecar = base / "linked.bin";
  for (const bool shared : {false, true}) {
    for (const bool symbolic : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "shared=" << shared << " symbolic=" << symbolic);
      std::error_code error;
      if (symbolic)
        std::filesystem::create_symlink(victim, sidecar, error);
      else
        std::filesystem::create_hard_link(victim, sidecar, error);
#ifdef _WIN32
      if (symbolic && error)
        continue;
#endif
      ASSERT_FALSE(error) << error.message();
      if (shared) {
        EXPECT_ANY_THROW(
            utils::TwoFilesWriteStream((base / "model.onnx").string(), sidecar.string()));
      } else {
        utils::TwoFilesWriteStream stream((base / "model.onnx").string(),
                                          (base / "weights.bin").string());
        EXPECT_ANY_THROW(stream.write_raw_bytes_in_second_stream(
            reinterpret_cast<const uint8_t *>("data"), 4, "linked.bin"));
      }
      EXPECT_EQ(ReadWholeFile(victim.string()), "sentinel");
      EXPECT_TRUE(std::filesystem::remove(sidecar));
    }
  }
}

TEST_F(ExternalDataDescriptorTest, SidecarWritesTruncateRegularFilesAndPreservePayloads) {
  const auto base = root / "model";
  const auto weights = base / "weights.bin";
  const auto extra = base / "extra.bin";
  for (const bool shared : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "shared=" << shared);
    WriteWholeFile(weights.string(), std::string(500, 'x'));
    WriteWholeFile(extra.string(), std::string(500, 'x'));
    {
      utils::TwoFilesWriteStream stream((base / "model.onnx").string(), weights.string());
      EXPECT_EQ(stream.weights_size(), 0);
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("first"), 5);
      EXPECT_EQ(stream.weights_size(shared ? "weights.bin" : "extra.bin"), shared ? 5 : 0);
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("second"), 6,
                                              shared ? "weights.bin" : "extra.bin");
      EXPECT_EQ(stream.weights_size(shared ? "weights.bin" : "extra.bin"), shared ? 11 : 6);
    }
    EXPECT_EQ(ReadWholeFile(weights.string()), shared ? "firstsecond" : "first");
    EXPECT_EQ(ReadWholeFile(extra.string()), shared ? std::string(500, 'x') : "second");
  }
}

TEST_F(ExternalDataDescriptorTest, ReaderPreservesSupportForConfinedNestedPaths) {
  const auto base = root / "model";
  std::filesystem::create_directories(base / "sub" / "nested");
  WriteWholeFile((base / "sub" / "nested" / "weights.bin").string(), "nested");
  WriteWholeFile((base / "sub" / "nested" / "extra.bin").string(), "second");
  utils::TwoFilesStream stream((base / "model.onnx").string(),
                               (base / "sub" / "nested" / "weights.bin").string());
  uint8_t bytes[6]{};
  stream.read_bytes_from_weights_stream(6, bytes, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 6), "nested");
  stream.set_active_weights_location("sub/nested/extra.bin");
  stream.read_bytes_from_weights_stream(6, bytes, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 6), "second");
}

TEST_F(ExternalDataDescriptorTest, ModelExternalDataDefaultsOmittedOffsetToZero) {
  const auto base = root / "model";
  const auto model_path = (base / "model.onnx").string();
  const auto weights_path = (base / "weights.bin").string();
  const std::string payload(1024, 'w');
  WriteWholeFile(weights_path, payload);
  for (const std::string length_key : {"length", "size"}) {
    for (const bool location_first : {false, true}) {
      for (const bool extra_metadata : {false, true}) {
        ModelProto model;
        auto *tensor = model.add_graph()->add_initializer();
        tensor->set_name("W");
        tensor->set_data_type(TensorProto::DataType::FLOAT);
        tensor->add_dims(256);
        tensor->set_data_location(TensorProto::DataLocation::EXTERNAL);
        for (const bool location : {location_first, !location_first}) {
          auto *entry = tensor->add_external_data();
          entry->set_key(location ? "location" : length_key);
          entry->set_value(location ? "weights.bin" : "1024");
        }
        if (extra_metadata) {
          auto *entry = tensor->add_external_data();
          entry->set_key("checksum");
          entry->set_value("unused");
        }
        std::string serialized;
        ASSERT_TRUE(model.SerializeToString(serialized));
        WriteWholeFile(model_path, serialized);
        tensor->LoadExternalData(base.string());
        EXPECT_EQ(std::string(reinterpret_cast<const char *>(tensor->ref_raw_data().data()),
                              tensor->ref_raw_data().size()),
                  payload);
        for (int mode = 0; mode < 4; ++mode) {
          SCOPED_TRACE(::testing::Message()
                       << length_key << ", location_first=" << location_first
                       << ", extra_metadata=" << extra_metadata << ", mode=" << mode);
          utils::TwoFilesStream stream(model_path, weights_path, mode == 3);
          ParseOptions options;
          options.no_copy = mode == 2;
          options.num_threads = mode == 1 ? 2 : 1;
          options.min_parallel_block_size = 0;
          ModelProto loaded;
          ASSERT_NO_THROW(ParseProtoFromStream(loaded, stream, options));
          const auto &result = loaded.ref_graph().ref_initializer()[0];
          EXPECT_EQ(std::string(reinterpret_cast<const char *>(result.ref_raw_data().data()),
                                result.ref_raw_data().size()),
                    payload);
        }
      }
    }
  }
}

TEST_F(ExternalDataDescriptorTest, ModelExternalDataRetainsOffsetRangeAndPathChecks) {
  const auto base = root / "model";
  const auto model_path = (base / "model.onnx").string();
  const auto weights_path = (base / "weights.bin").string();
  WriteWholeFile(weights_path, std::string(1024, 'w'));
  const struct {
    const char *location;
    const char *offset;
    const char *length;
  } invalid_cases[] = {{"weights.bin", "-1", "1024"},
                       {"weights.bin", "1", "1024"},
                       {"weights.bin", nullptr, "1025"},
                       {"weights.bin", nullptr, "0"},
                       {"../outside/weights.bin", nullptr, "1024"}};
  for (const auto &invalid_case : invalid_cases) {
    SCOPED_TRACE(::testing::Message() << invalid_case.location << ", offset="
                                      << (invalid_case.offset ? invalid_case.offset : "omitted")
                                      << ", length=" << invalid_case.length);
    ModelProto model;
    auto *tensor = model.add_graph()->add_initializer();
    tensor->set_name("W");
    tensor->set_data_type(TensorProto::DataType::FLOAT);
    tensor->add_dims(256);
    tensor->set_data_location(TensorProto::DataLocation::EXTERNAL);
    auto *location = tensor->add_external_data();
    location->set_key("location");
    location->set_value(invalid_case.location);
    if (invalid_case.offset) {
      auto *offset = tensor->add_external_data();
      offset->set_key("offset");
      offset->set_value(invalid_case.offset);
    }
    auto *length = tensor->add_external_data();
    length->set_key("length");
    length->set_value(invalid_case.length);
    std::string serialized;
    ASSERT_TRUE(model.SerializeToString(serialized));
    WriteWholeFile(model_path, serialized);
    utils::TwoFilesStream stream(model_path, weights_path);
    ParseOptions options;
    ModelProto loaded;
    EXPECT_THROW(ParseProtoFromStream(loaded, stream, options), std::runtime_error);
  }
}

TEST_F(ExternalDataDescriptorTest, NonAsciiNativePathsWorkForExternalReadsAndWrites) {
  const auto base = root / "model";
  const auto filename = std::filesystem::path(u8"weights-\u00e9.bin");
  WriteWholeFile((base / filename).string(), "payload");

  TensorProto tensor;
  tensor.set_name("unicode");
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  StringStringEntryProto *location = tensor.add_external_data();
  location->set_key("location");
  location->set_value(filename.string());
  tensor.LoadExternalData(base.string());
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(tensor.ref_raw_data().data()),
                        tensor.ref_raw_data().size()),
            "payload");

  const auto extra_filename = std::filesystem::path(u8"weights-\u00e9-extra.bin");
  utils::TwoFilesWriteStream stream((base / "model.onnx").string(),
                                    (base / "sub" / "weights.bin").string());
  stream.set_active_weights_location(extra_filename.string());
  stream.pre_allocate_weights(extra_filename.string(), 7);
  stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("written"), 7,
                                          extra_filename.string());
  EXPECT_EQ(ReadWholeFile((base / extra_filename).string()), std::string(7, '\0') + "written");
}

TEST_F(ExternalDataDescriptorTest, DiscoveryPlaceholderValidatesBeforeReadingModelAsWeights) {
  const auto base = root / "model";
  std::filesystem::create_hard_link(base / "model.onnx", base / "model-link.onnx");
  utils::TwoFilesStream stream((base / "model.onnx").string(),
                               (base / "sub" / ".." / "model.onnx").string());
  const auto *model_bytes = stream.read_bytes(5);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(model_bytes), 5), "model");

  stream.set_active_weights_location("sub/weights.bin");
  uint8_t bytes[8]{};
  stream.read_bytes_from_weights_stream(8, bytes, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");

  stream.set_active_weights_location("model.onnx");
  EXPECT_ANY_THROW(stream.read_bytes_from_weights_stream(5, bytes, 0));
}

TEST_F(ExternalDataDescriptorTest, TrustedBaseAliasesPreservePrimaryAndAdditionalWeightsLocations) {
  const auto base = std::filesystem::absolute(root / "model");
  const auto alias = std::filesystem::absolute(root / "alias");
#ifdef _WIN32
  const auto error = onnx_light_test::CreateWindowsJunction(alias, base);
  ASSERT_FALSE(error) << error.message();
#else
  std::filesystem::create_directory_symlink(base, alias);
#endif
  for (const auto &model_base : {base, alias}) {
    for (const auto &weights_base : {base, alias}) {
      utils::TwoFilesStream stream((model_base / "model.onnx").string(),
                                   (weights_base / "sub" / "weights.bin").string(), true);
      uint8_t bytes[8]{};
      stream.read_bytes_from_weights_stream(8, bytes, 0);
      EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
      stream.set_active_weights_location((weights_base / "sub" / "extra.bin").string());
      stream.read_bytes_from_weights_stream(8, bytes, 0);
      EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "extra!!!");
      std::shared_ptr<void> owner;
      const auto *borrowed = stream.borrow_weights_bytes("sub/weights.bin", 0, 8, 1, owner);
      EXPECT_EQ(std::string(reinterpret_cast<const char *>(borrowed), 8), "original");
    }
  }
#ifdef _WIN32
  const auto child_error = onnx_light_test::CreateWindowsJunction(
      base / "linked", std::filesystem::absolute(root / "outside"));
  ASSERT_FALSE(child_error) << child_error.message();
#else
  std::filesystem::create_directory_symlink(std::filesystem::absolute(root / "outside"),
                                            base / "linked");
#endif
  EXPECT_ANY_THROW(utils::TwoFilesStream((alias / "model.onnx").string(),
                                         (base / "linked" / "weights.bin").string()));
}

#ifndef _WIN32
TEST_F(ExternalDataDescriptorTest, SidecarWritesKeepDescriptorAfterLeafReplacement) {
  const auto base = root / "model";
  const auto victim = std::filesystem::absolute(root / "outside" / "weights.bin");
  for (const bool shared : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "shared=" << shared);
    const auto sidecar = base / (shared ? "weights.bin" : "extra.bin");
    const auto pinned = base / "pinned.bin";
    {
      utils::TwoFilesWriteStream stream((base / "model.onnx").string(),
                                        (base / "weights.bin").string());
      const std::string location = sidecar.filename().string();
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("first"), 5,
                                              location);
      EXPECT_EQ(stream.weights_size(location), 5);
      std::filesystem::rename(sidecar, pinned);
      std::filesystem::create_symlink(victim, sidecar);
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("second"), 6,
                                              location);
      EXPECT_EQ(stream.weights_size(location), 11);
    }
    EXPECT_EQ(ReadWholeFile(pinned.string()), "firstsecond");
    EXPECT_EQ(ReadWholeFile(victim.string()), "sentinel");
    std::filesystem::remove(sidecar);
    std::filesystem::remove(pinned);
  }
}

TEST_F(ExternalDataDescriptorTest, RejectsSymlinkLeavesAndNonregularFilesWithoutDamage) {
  const auto base = root / "model";
  std::filesystem::create_symlink(std::filesystem::absolute(root / "outside" / "weights.bin"),
                                  base / "linked.bin");
  for (bool write : {false, true})
    EXPECT_ANY_THROW(utils::detail::OpenExternalDataFile(base.string(), "linked.bin", write));
  ASSERT_EQ(::mkfifo((base / "pipe").c_str(), 0600), 0);
  for (bool write : {false, true})
    EXPECT_ANY_THROW(utils::detail::OpenExternalDataFile(base.string(), "pipe", write));
  EXPECT_EQ(ReadWholeFile((root / "outside" / "weights.bin").string()), "sentinel");
}

TEST_F(ExternalDataDescriptorTest, RetainedReadsWorkersAndMappingIgnoreDirectorySwap) {
  const auto base = root / "model";
  utils::TwoFilesStream stream((base / "model.onnx").string(),
                               (base / "sub" / "weights.bin").string(), true);
  stream.set_active_weights_location("sub/extra.bin");
  stream.set_active_weights_location("sub/weights.bin");
  std::filesystem::rename(base / "sub", base / "pinned");
  std::filesystem::create_directory_symlink(std::filesystem::absolute(root / "outside"),
                                            base / "sub");
  uint8_t bytes[8]{};
  stream.read_bytes_from_weights_stream(8, bytes, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
  std::shared_ptr<void> owner;
  const auto *borrowed = stream.borrow_weights_bytes("sub/weights.bin", 0, 8, 1, owner);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(borrowed), 8), "original");
  stream.StartThreadPool(2);
  utils::DelayedBlock block{8, bytes, 0, 1};
  stream.ReadDelayedBlock(block);
  EXPECT_EQ(stream.TotalDelayedBlockBytes(), 8);
  stream.WaitForDelayedBlock();
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
  stream.set_active_weights_location("sub/extra.bin");
  stream.read_bytes_from_weights_stream(0, bytes, 0);
  stream.StartThreadPool(1);
  utils::DelayedBlock extra{8, bytes, 0, 1};
  stream.ReadDelayedBlock(extra);
  stream.WaitForDelayedBlock();
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "extra!!!");
  borrowed = stream.borrow_weights_bytes("sub/extra.bin", 0, 8, 1, owner);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(borrowed), 8), "extra!!!");
  EXPECT_ANY_THROW(utils::detail::OpenExternalDataFile(base.string(), "sub/new.bin", true));
  EXPECT_FALSE(std::filesystem::exists(root / "outside" / "new.bin"));
}

TEST_F(ExternalDataDescriptorTest, ParallelAndInlineWritesIgnoreDirectorySwap) {
  const auto base = root / "model";
  {
    utils::TwoFilesWriteStream stream((base / "model.onnx").string(),
                                      (base / "sub" / "weights.bin").string());
    stream.pre_allocate_weights(8);
    stream.pre_allocate_weights("sub/extra.bin", 8);
    std::filesystem::rename(base / "sub", base / "pinned");
    std::filesystem::create_directory_symlink(std::filesystem::absolute(root / "outside"),
                                              base / "sub");
    stream.StartWriteThreadPool(2);
    stream.SetMinParallelBlockSize(1024);
    const std::string payload(2 * 1024 * 1024, 'x');
    stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>(payload.data()),
                                            payload.size());
    stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("tail"), 4);
    stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>(payload.data()),
                                            payload.size(), "sub/extra.bin");
    stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("more"), 4,
                                            "sub/extra.bin");
    stream.WaitForWriteCompletion();
    EXPECT_EQ(ReadWholeFile((base / "pinned" / "weights.bin").string()), payload + "tail");
    EXPECT_EQ(ReadWholeFile((base / "pinned" / "extra.bin").string()), payload + "more");
  }
  EXPECT_EQ(ReadWholeFile((root / "outside" / "weights.bin").string()), "sentinel");
  EXPECT_EQ(ReadWholeFile((root / "outside" / "extra.bin").string()), "sentinel");
}

TEST_F(ExternalDataDescriptorTest, RetainedFileStreamsKeepDelayedOperationsAnchored) {
  const auto base = root / "model";
  auto reader = utils::detail::OpenExternalDataFile(base.string(), "sub/weights.bin");
  auto writer = utils::detail::OpenExternalDataFile(base.string(), "sub/new.bin", true);
  {
    utils::FileStream input("unused", reader);
    utils::FileWriteStream output("unused", writer);
    std::filesystem::rename(base / "sub", base / "pinned");
    std::filesystem::create_directory_symlink(std::filesystem::absolute(root / "outside"),
                                              base / "sub");
    uint8_t bytes[8]{};
    input.StartThreadPool(1);
    utils::DelayedBlock read{8, bytes, 0};
    input.ReadDelayedBlock(read);
    input.WaitForDelayedBlock();
    EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
    output.StartThreadPool(1);
    utils::DelayedWriteBlock write{8, bytes};
    output.WriteDelayedBlock(write);
    output.WaitForDelayedBlock();
  }
  EXPECT_EQ(ReadWholeFile((base / "pinned" / "new.bin").string()), "original");
  EXPECT_FALSE(std::filesystem::exists(root / "outside" / "new.bin"));
}
#else
TEST_F(ExternalDataDescriptorTest, RejectsActualJunctionParentsForReadingAndCreation) {
  const auto base = root / "model";
  const auto error = onnx_light_test::CreateWindowsJunction(base / "junction", root / "outside");
  ASSERT_FALSE(error) << error.message();
  for (bool write : {false, true})
    EXPECT_ANY_THROW(
        utils::detail::OpenExternalDataFile(base.string(), "junction/weights.bin", write));
  EXPECT_ANY_THROW(utils::detail::OpenExternalDataFile(base.string(), "junction/new.bin", true));
  EXPECT_FALSE(std::filesystem::exists(root / "outside" / "new.bin"));
  EXPECT_EQ(ReadWholeFile((root / "outside" / "weights.bin").string()), "sentinel");
}

TEST_F(ExternalDataDescriptorTest, RetainedStreamsIgnoreJunctionReplacement) {
  const auto base = root / "model";
  std::filesystem::rename(base / "sub", base / "pinned");
  auto error = onnx_light_test::CreateWindowsJunction(base / "sub", base / "pinned");
  ASSERT_FALSE(error) << error.message();
  utils::TwoFilesStream input((base / "model.onnx").string(),
                              (base / "sub" / "weights.bin").string(), true);
  auto output_file = utils::detail::OpenExternalDataFile(base.string(), "sub/new.bin", true);
  utils::FileWriteStream output("unused", output_file);
  error = onnx_light_test::CreateWindowsJunction(base / "sub", root / "outside", false);
  ASSERT_FALSE(error) << error.message();
  uint8_t bytes[8]{};
  input.read_bytes_from_weights_stream(8, bytes, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
  std::shared_ptr<void> owner;
  const auto *borrowed = input.borrow_weights_bytes("sub/weights.bin", 0, 8, 1, owner);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(borrowed), 8), "original");
  input.StartThreadPool(1);
  utils::DelayedBlock read{8, bytes, 0, 1};
  input.ReadDelayedBlock(read);
  input.WaitForDelayedBlock();
  EXPECT_EQ(std::string(reinterpret_cast<char *>(bytes), 8), "original");
  output.StartThreadPool(1);
  utils::DelayedWriteBlock write{8, bytes};
  output.WriteDelayedBlock(write);
  output.WaitForDelayedBlock();
  EXPECT_EQ(ReadWholeFile((base / "pinned" / "new.bin").string()), "original");
  EXPECT_FALSE(std::filesystem::exists(root / "outside" / "new.bin"));
}
#endif

TEST(proto_stream_api, SerializeToStringReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  EXPECT_TRUE(attr.SerializeToString(serialized));
  EXPECT_FALSE(serialized.empty());
}

TEST(proto_stream_api, ParseFromStringReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_TRUE(parsed.has_i());
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, ParseFromArrayUsesBoundedBorrowedInput) {
  TensorProto tensor;
  tensor.set_name("weight");
  tensor.set_raw_data(std::string(8192, '\x2a'));
  std::string serialized;
  ASSERT_TRUE(tensor.SerializeToString(serialized));

  TensorProto parsed;
  ASSERT_TRUE(parsed.ParseFromArray(serialized.data(), static_cast<int>(serialized.size())));
  std::fill(serialized.begin(), serialized.end(), '\0');
  ASSERT_EQ(parsed.ref_raw_data().size(), 8192u);
  EXPECT_EQ(parsed.ref_raw_data().data()[4096], 0x2a);
}

TEST(proto_stream_api, ParseFromArrayMalformedInputsReturnFalse) {
  const std::string invalid_varint(11, static_cast<char>(0x80));
  AttributeProto invalid;
  EXPECT_FALSE(
      invalid.ParseFromArray(invalid_varint.data(), static_cast<int>(invalid_varint.size())));

  AttributeProto attr = MakeIntAttr();
  std::string truncated;
  ASSERT_TRUE(attr.SerializeToString(truncated));
  truncated.pop_back();
  AttributeProto parsed;
  EXPECT_FALSE(parsed.ParseFromArray(truncated.data(), static_cast<int>(truncated.size())));
}

TEST(proto_stream_api, ParseFromFileDescriptorStartsAtCurrentOffsetAndReachesEof) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));
  const std::string prefix = "ignored prefix";
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_parse_proto_fd.bin";
  WriteWholeFile(path.string(), prefix + serialized);
  const int fd = OpenTempFdForRead(path.string());
  ASSERT_GE(fd, 0);
  ASSERT_EQ(SeekFd(fd, static_cast<int64_t>(prefix.size())), static_cast<int64_t>(prefix.size()));

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromFileDescriptor(fd));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
  EXPECT_EQ(TellFd(fd), static_cast<int64_t>(prefix.size() + serialized.size()));
  CloseFd(fd);
  std::filesystem::remove(path);
}

TEST(proto_stream_api, ParseFromFileDescriptorMalformedInputsReturnFalse) {
  AttributeProto attr = MakeIntAttr();
  std::string truncated;
  ASSERT_TRUE(attr.SerializeToString(truncated));
  truncated.pop_back();
  const std::string invalid_varint(11, static_cast<char>(0x80));

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_parse_malformed_fd.bin";
  const std::string *inputs[] = {&truncated, &invalid_varint};
  for (const std::string *input : inputs) {
    WriteWholeFile(path.string(), *input);
    const int fd = OpenTempFdForRead(path.string());
    ASSERT_GE(fd, 0);
    AttributeProto parsed;
    EXPECT_FALSE(parsed.ParseFromFileDescriptor(fd));
    CloseFd(fd);
  }
  std::filesystem::remove(path);
}

#if !defined(_WIN32)
TEST(proto_stream_api, ParseFromFileDescriptorSupportsPipes) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));
  int descriptors[2];
  ASSERT_EQ(::pipe(descriptors), 0);
  ASSERT_EQ(::write(descriptors[1], serialized.data(), serialized.size()),
            static_cast<ssize_t>(serialized.size()));
  CloseFd(descriptors[1]);

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromFileDescriptor(descriptors[0]));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
  CloseFd(descriptors[0]);
}
#endif

TEST(proto_stream_api, ParseFromFileDescriptorReadFailureReturnsFalse) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_parse_closed_fd.bin";
  WriteWholeFile(path.string(), "data");
  const int fd = OpenTempFdForRead(path.string());
  ASSERT_GE(fd, 0);
  CloseFd(fd);

  AttributeProto parsed;
  EXPECT_FALSE(parsed.ParseFromFileDescriptor(fd));
  std::filesystem::remove(path);
}

TEST(proto_stream_api, FdReadStreamPreservesTensorLimit) {
  TensorProto tensor;
  tensor.set_raw_data(std::string(32, '\x01'));
  std::string serialized;
  ASSERT_TRUE(tensor.SerializeToString(serialized));
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_parse_limited_fd.bin";
  WriteWholeFile(path.string(), serialized);
  const int fd = OpenTempFdForRead(path.string());
  ASSERT_GE(fd, 0);

  utils::FdReadStream stream(fd);
  ParseOptions options;
  options.max_tensor_size_bytes = 16;
  TensorProto parsed;
  EXPECT_THROW(parsed.ParseFromZeroCopyStream(&stream, options),
               onnx_light_helpers::ParseLimitExceeded);
  EXPECT_EQ(options._recursion_depth, 0);
  CloseFd(fd);
  std::filesystem::remove(path);
}

TEST(proto_stream_api, FdReadStreamPreservesRecursionLimit) {
  const std::string serialized = BuildNestedTypeProto(10);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_parse_recursion_fd.bin";
  WriteWholeFile(path.string(), serialized);
  const int fd = OpenTempFdForRead(path.string());
  ASSERT_GE(fd, 0);

  utils::FdReadStream stream(fd);
  ParseOptions options;
  options.max_recursion_depth = 5;
  TypeProto parsed;
  EXPECT_THROW(parsed.ParseFromZeroCopyStream(&stream, options),
               onnx_light_helpers::ParseLimitExceeded);
  EXPECT_EQ(options._recursion_depth, 0);
  CloseFd(fd);
  std::filesystem::remove(path);
}

class FdReadStreamTest : public ::testing::Test {
protected:
  void SetUp() override {
    path = std::filesystem::temp_directory_path() /
           (std::string("onnx_light_fd_skip_") +
            ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".bin");
  }
  void TearDown() override {
    if (fd >= 0)
      CloseFd(fd);
    std::filesystem::remove(path);
  }
  void Open(const std::string &payload, int64_t prefix = 0) {
    {
      std::ofstream out(path, std::ios::binary);
      out.seekp(prefix);
      out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
      ASSERT_TRUE(out.good());
    }
    fd = OpenTempFdForRead(path.string());
    ASSERT_GE(fd, 0);
  }
  std::filesystem::path path;
  int fd = -1;
};

TEST_F(FdReadStreamTest, BufferedAndUnbufferedSkips) {
  Open("0123456789abcdef");
  utils::FdReadStream stream(fd, 4);
  uint8_t byte;
  stream.read_bytes(1, &byte);
  EXPECT_EQ(byte, '0');
  EXPECT_EQ(TellFd(fd), 4);
  stream.skip_bytes(2);
  EXPECT_EQ(TellFd(fd), 4);
  EXPECT_EQ(stream.tell(), 3);
  EXPECT_EQ(stream.ByteCount(), 3);
  ASSERT_TRUE(stream.Skip(6));
  EXPECT_EQ(TellFd(fd), 9);
  EXPECT_EQ(stream.tell(), 9);
  EXPECT_EQ(stream.ByteCount(), 9);
  stream.read_bytes(1, &byte);
  EXPECT_EQ(byte, '9');
  stream.skip_bytes(0);
  EXPECT_EQ(stream.tell(), 10);
  EXPECT_EQ(stream.ByteCount(), 10);
  stream.skip_bytes(5);
  EXPECT_EQ(TellFd(fd), 15);
  stream.read_bytes(1, &byte);
  EXPECT_EQ(byte, 'f');
  EXPECT_FALSE(stream.NotEnd());
  EXPECT_FALSE(stream.failed());
}

TEST_F(FdReadStreamTest, NonzeroInitialOffsetAndLargeSkip) {
  constexpr int64_t prefix = (int64_t{1} << 31) + 123;
  Open("payload", prefix);
  ASSERT_EQ(SeekFd(fd, 17), 17);
  utils::FdReadStream stream(fd, 64 * 1024);
  stream.skip_bytes(prefix - 17);
  EXPECT_EQ(TellFd(fd), prefix);
  EXPECT_EQ(stream.tell(), prefix - 17);
  EXPECT_EQ(stream.ByteCount(), prefix - 17);
  uint8_t payload[7];
  stream.read_bytes(7, payload);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(payload), 7), "payload");
  EXPECT_EQ(stream.tell(), prefix - 17 + 7);
  EXPECT_EQ(stream.ByteCount(), stream.tell());
}

TEST_F(FdReadStreamTest, ZeroCopyAndNestedLimits) {
  Open("0123456789abcdef");
  utils::FdReadStream stream(fd, 4);
  const void *data;
  int size;
  ASSERT_TRUE(stream.Next(&data, &size));
  ASSERT_EQ(size, 4);
  stream.BackUp(2);
  EXPECT_EQ(stream.tell(), 2);
  EXPECT_EQ(stream.ByteCount(), 2);
  stream.LimitToNext(1);
  ASSERT_TRUE(stream.Next(&data, &size));
  EXPECT_EQ(std::string(static_cast<const char *>(data), size), "2");
  EXPECT_FALSE(stream.NotEnd());
  stream.BackUp(1);
  stream.Restore();
  EXPECT_EQ(stream.tell(), 2);
  EXPECT_EQ(stream.ByteCount(), 2);
  stream.LimitToNext(7);
  stream.LimitToNext(3);
  EXPECT_FALSE(stream.Skip(4));
  EXPECT_THROW(stream.skip_bytes(4), std::runtime_error);
  EXPECT_EQ(stream.tell(), 2);
  ASSERT_TRUE(stream.Skip(3));
  EXPECT_EQ(stream.tell(), 5);
  EXPECT_EQ(stream.ByteCount(), 5);
  EXPECT_FALSE(stream.Next(&data, &size));
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(size, 0);
  stream.Restore();
  ASSERT_TRUE(stream.Next(&data, &size));
  EXPECT_EQ(std::string(static_cast<const char *>(data), size), "5678");
  EXPECT_EQ(stream.tell(), 9);
  EXPECT_FALSE(stream.Skip(1));
  stream.Restore();
  ASSERT_TRUE(stream.Next(&data, &size));
  EXPECT_EQ(std::string(static_cast<const char *>(data), size), "9abc");
  stream.BackUp(3);
  stream.skip_bytes(2);
  EXPECT_THROW(stream.BackUp(1), std::runtime_error);
  ASSERT_TRUE(stream.Next(&data, &size));
  EXPECT_EQ(std::string(static_cast<const char *>(data), size), "c");
  EXPECT_EQ(stream.ByteCount(), stream.tell());
}

TEST_F(FdReadStreamTest, RejectsInvalidDistancesAndEof) {
  Open("data");
  utils::FdReadStream stream(fd, 2);
  EXPECT_FALSE(stream.Skip(-1));
  EXPECT_THROW(stream.skip_bytes(-1), std::runtime_error);
  EXPECT_FALSE(stream.Skip(5));
  EXPECT_THROW(stream.skip_bytes(std::numeric_limits<int64_t>::max()), std::runtime_error);
  EXPECT_EQ(stream.tell(), 0);
  EXPECT_EQ(TellFd(fd), 0);
  stream.skip_bytes(4);
  EXPECT_EQ(stream.tell(), 4);
  EXPECT_EQ(stream.ByteCount(), 4);
  EXPECT_FALSE(stream.Skip(1));
  EXPECT_THROW(stream.skip_bytes(1), std::runtime_error);
  EXPECT_FALSE(stream.NotEnd());
  EXPECT_FALSE(stream.failed());
}

TEST_F(FdReadStreamTest, TruncationDoesNotAllowSeekingPastEof) {
  Open("0123456789");
  utils::FdReadStream stream(fd, 2);
  std::filesystem::resize_file(path, 3);
  EXPECT_FALSE(stream.Skip(8));
  EXPECT_EQ(stream.tell(), 3);
  EXPECT_EQ(stream.ByteCount(), 3);
  EXPECT_EQ(TellFd(fd), 3);
  EXPECT_FALSE(stream.failed());
  EXPECT_THROW(stream.skip_bytes(1), std::runtime_error);
}

TEST_F(FdReadStreamTest, InvalidAndClosedDescriptorsReportFailure) {
  utils::FdReadStream invalid(-1, 2);
  EXPECT_FALSE(invalid.Skip(1));
  EXPECT_TRUE(invalid.failed());
  EXPECT_THROW(invalid.skip_bytes(1), std::runtime_error);
  Open("data");
  utils::FdReadStream closed(fd, 2);
  CloseFd(fd);
  fd = -1;
  EXPECT_FALSE(closed.Skip(1));
  EXPECT_TRUE(closed.failed());
  EXPECT_THROW(closed.skip_bytes(1), std::runtime_error);
}

TEST_F(FdReadStreamTest, WriteOnlyDescriptorReportsFailure) {
  fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  {
    utils::FdWriteStream writer(fd);
    writer.write_raw_bytes(reinterpret_cast<const uint8_t *>("data"), 4);
  }
  ASSERT_EQ(SeekFd(fd, 0), 0);
  utils::FdReadStream stream(fd, 2);
  EXPECT_FALSE(stream.Skip(1));
  EXPECT_TRUE(stream.failed());
  EXPECT_EQ(stream.tell(), 0);
  EXPECT_EQ(stream.ByteCount(), 0);
  EXPECT_EQ(TellFd(fd), 0);
  EXPECT_THROW(stream.skip_bytes(1), std::runtime_error);
}

TEST_F(FdReadStreamTest, EmptyWriteOnlyDescriptorReportsReadFailure) {
  fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);

  utils::FdReadStream stream(fd, 2);
  const void *data = nullptr;
  int size = -1;
  EXPECT_FALSE(stream.Next(&data, &size));
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(size, 0);
  EXPECT_TRUE(stream.failed());

  AttributeProto parsed;
  EXPECT_FALSE(parsed.ParseFromFileDescriptor(fd));
}

#if defined(__linux__)
TEST_F(FdReadStreamTest, CharacterDevicesRetainUnboundedInput) {
  fd = ::open("/dev/zero", O_RDONLY);
  ASSERT_GE(fd, 0);

  google::protobuf::io::FileInputStream stream(fd, 4);
  const void *data = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&data, &size));
  ASSERT_GT(size, 0);
  EXPECT_EQ(static_cast<const uint8_t *>(data)[0], 0);
  EXPECT_FALSE(stream.failed());
}

TEST_F(FdReadStreamTest, ProcFilesRetainReadableInputAndLimits) {
  const std::string source = "/proc/self/cmdline";
  fd = ::open(source.c_str(), O_RDONLY);
  if (fd < 0 && (errno == ENOENT || errno == EACCES))
    GTEST_SKIP() << source << " is unavailable.";
  ASSERT_GE(fd, 0);
  const std::string payload = ReadWholeFile(source);
  ASSERT_GT(payload.size(), 3u);
  struct stat info;
  ASSERT_EQ(::fstat(fd, &info), 0);
  ASSERT_TRUE(S_ISREG(info.st_mode));
  ASSERT_EQ(info.st_size, 0);

  google::protobuf::io::FileInputStream stream(fd, 4);
  stream.LimitToNext(3);
  const void *data = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&data, &size));
  EXPECT_EQ(std::string(static_cast<const char *>(data), size), payload.substr(0, 3));
  EXPECT_FALSE(stream.NotEnd());
  EXPECT_FALSE(stream.Skip(1));
  EXPECT_THROW(stream.skip_bytes(1), std::runtime_error);
  EXPECT_EQ(stream.ByteCount(), 3);
  stream.Restore();
  std::string remainder;
  while (stream.Next(&data, &size))
    remainder.append(static_cast<const char *>(data), size);
  EXPECT_EQ(remainder, payload.substr(3));
  EXPECT_EQ(stream.ByteCount(), static_cast<int64_t>(payload.size()));
  EXPECT_FALSE(stream.failed());
}

TEST_F(FdReadStreamTest, SyntheticFileSizeDoesNotAllowSkippingPastEof) {
  const std::string source = "/sys/devices/system/cpu/online";
  fd = ::open(source.c_str(), O_RDONLY);
  if (fd < 0 && (errno == ENOENT || errno == EACCES))
    GTEST_SKIP() << source << " is unavailable.";
  ASSERT_GE(fd, 0);
  const std::string payload = ReadWholeFile(source);
  ASSERT_GT(payload.size(), 1u);
  ASSERT_LT(payload.size(), 1000u);
  struct stat info;
  ASSERT_EQ(::fstat(fd, &info), 0);
  ASSERT_TRUE(S_ISREG(info.st_mode));
  ASSERT_GT(info.st_size, 1000);

  for (const bool buffered : {false, true}) {
    for (const bool binary_api : {false, true}) {
      SCOPED_TRACE(buffered);
      SCOPED_TRACE(binary_api);
      ASSERT_EQ(SeekFd(fd, 0), 0);
      utils::FdReadStream stream(fd, 4);
      if (buffered) {
        uint8_t byte;
        stream.read_bytes(1, &byte);
        EXPECT_EQ(byte, static_cast<uint8_t>(payload[0]));
      }
      if (binary_api)
        EXPECT_THROW(stream.skip_bytes(1000), std::runtime_error);
      else
        EXPECT_FALSE(stream.Skip(1000));
      EXPECT_EQ(stream.tell(), static_cast<int64_t>(payload.size()));
      EXPECT_EQ(stream.ByteCount(), stream.tell());
      EXPECT_FALSE(stream.NotEnd());
      EXPECT_FALSE(stream.failed());
    }
  }
}
#endif

TEST_F(FdReadStreamTest, PipesRetainBufferedFallback) {
  int descriptors[2];
#if defined(_WIN32)
  ASSERT_EQ(::_pipe(descriptors, 4096, _O_BINARY), 0);
  ASSERT_EQ(::_write(descriptors[1], "0123456789", 10), 10);
#else
  ASSERT_EQ(::pipe(descriptors), 0);
  ASSERT_EQ(::write(descriptors[1], "0123456789", 10), 10);
#endif
  fd = descriptors[0];
  CloseFd(descriptors[1]);
  utils::FdReadStream stream(fd, 4);
  ASSERT_TRUE(stream.Skip(2));
  stream.skip_bytes(5);
  EXPECT_EQ(stream.tell(), 7);
  EXPECT_EQ(stream.ByteCount(), 7);
  uint8_t byte;
  stream.read_bytes(1, &byte);
  EXPECT_EQ(byte, '7');
  EXPECT_FALSE(stream.Skip(3));
  EXPECT_EQ(stream.tell(), 10);
  EXPECT_EQ(stream.ByteCount(), 10);
  EXPECT_FALSE(stream.failed());
  EXPECT_THROW(stream.skip_bytes(1), std::runtime_error);
}

#if defined(__linux__)
TEST_F(FdReadStreamTest, LargePrefixDoesNotReadProportionalBytesOrCalls) {
  constexpr int64_t prefix = 32 * 1024 * 1024;
  Open("payload", prefix);
  utils::FdReadStream stream(fd, 4096);
  const auto read_io = []() {
    std::ifstream in("/proc/self/io");
    std::string key;
    uint64_t value, bytes = 0, calls = 0;
    while (in >> key >> value) {
      if (key == "rchar:")
        bytes = value;
      if (key == "syscr:")
        calls = value;
    }
    return std::make_pair(bytes, calls);
  };
  const auto before = read_io();
  ASSERT_GT(before.second, 0u);
  stream.skip_bytes(prefix / 2);
  ASSERT_TRUE(stream.Skip(static_cast<int>(prefix / 2)));
  uint8_t payload[7];
  stream.read_bytes(7, payload);
  const auto after = read_io();
  EXPECT_EQ(std::string(reinterpret_cast<char *>(payload), 7), "payload");
  // Includes the small /proc/self/io read itself; no prefix bytes should be read.
  EXPECT_LT(after.first - before.first, 4096u);
  EXPECT_LE(after.second - before.second, 4u);
  EXPECT_EQ(stream.ByteCount(), prefix + 7);
}
#endif

TEST(proto_stream_api, SerializeToStringWithOptionsReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  SerializeOptions sopts;
  std::string serialized;
  EXPECT_TRUE(attr.SerializeToString(serialized, sopts));

  AttributeProto parsed;
  ParseOptions popts;
  EXPECT_TRUE(parsed.ParseFromString(serialized, popts));
  EXPECT_EQ(parsed.ref_i(), 123456789);
}

TEST(proto_stream_api, ByteSizeLongMatchesSerializedSize) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));

  // ByteSizeLong() must equal the number of bytes an actual serialization
  // produces, and equal SerializeSize().size(), without serializing itself.
  EXPECT_EQ(attr.ByteSizeLong(), serialized.size());
  EXPECT_EQ(attr.ByteSizeLong(), static_cast<size_t>(attr.SerializeSize().size()));
}

TEST(proto_stream_api, ByteSizeLongOnModelProto) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");
  std::string serialized;
  ASSERT_TRUE(model.SerializeToString(serialized));
  EXPECT_EQ(model.ByteSizeLong(), serialized.size());
}

TEST(proto_stream_api, SerializesPackedFloatingPointPayloadsInBlocks) {
  class CountingWriteStream : public utils::StringWriteStream {
  public:
    void write_raw_bytes(const uint8_t *data, utils::offset_t size) override {
      ++calls;
      utils::StringWriteStream::write_raw_bytes(data, size);
    }
    std::size_t calls = 0;
  };
  for (const bool doubles : {false, true}) {
    SCOPED_TRACE(doubles);
    TensorProto tensor;
    tensor.set_name("weights");
    tensor.set_data_type(doubles ? TensorProto::DOUBLE : TensorProto::FLOAT);
    tensor.add_dims(16384);
    for (int i = 0; i < 16384; ++i) {
      if (doubles)
        tensor.add_double_data(i == 0 ? -0.0 : (i - 8192) * 0.125);
      else
        tensor.add_float_data(i == 0 ? -0.0f : (i - 8192) * 0.125f);
    }
    if (doubles)
      tensor.ref_double_data()[1] = std::numeric_limits<double>::quiet_NaN();
    else
      tensor.ref_float_data()[1] = std::numeric_limits<float>::quiet_NaN();
    CountingWriteStream stream;
    SerializeOptions options;
    tensor.SerializeSize(stream, options);
    tensor.SerializeToStream(stream, options);
    EXPECT_LT(stream.calls, 32u);
    TensorProto parsed;
    ASSERT_TRUE(parsed.ParseFromString(
        std::string(reinterpret_cast<const char *>(stream.data()), stream.size())));
    std::string difference;
    EXPECT_TRUE(tensor.Equals(parsed, &difference)) << difference;
  }
}

TEST(proto_stream_api, SerializeToFileDescriptorReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_proto_fd.bin";
  const int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(attr.SerializeToFileDescriptor(fd));
  CloseFd(fd);

  AttributeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(ReadWholeFile(path.string())));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
  std::filesystem::remove(path);
}

TEST(proto_stream_api, SerializeToFileDescriptorWithOptionsReturnsTrue) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");
  SerializeOptions sopts;
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_model_proto_fd.bin";
  const int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(model.SerializeToFileDescriptor(fd, sopts));
  CloseFd(fd);

  ModelProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(ReadWholeFile(path.string())));
  EXPECT_EQ(parsed.ref_ir_version(), 7);
  EXPECT_EQ(parsed.ref_producer_name(), "onnx-light");
  std::filesystem::remove(path);
}

TEST(proto_stream_api, SaveToFileDescriptorReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_save_proto_fd.bin";
  const int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(attr.SaveToFileDescriptor(fd));
  CloseFd(fd);

  AttributeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(ReadWholeFile(path.string())));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
  std::filesystem::remove(path);
}

TEST(proto_stream_api, SaveToFileDescriptorWithOptionsReturnsTrue) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");
  SerializeOptions sopts;
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_save_model_proto_fd.bin";
  const int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  ASSERT_TRUE(model.SaveToFileDescriptor(fd, sopts));
  CloseFd(fd);

  ModelProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(ReadWholeFile(path.string())));
  EXPECT_EQ(parsed.ref_ir_version(), 7);
  EXPECT_EQ(parsed.ref_producer_name(), "onnx-light");
  std::filesystem::remove(path);
}

TEST(proto_stream_api, SaveToFileDescriptorMatchesSerializeToFileDescriptor) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");

  const std::filesystem::path path1 =
      std::filesystem::temp_directory_path() / "onnx_light_serialize_fd.bin";
  const int fd1 = OpenTempFdForWrite(path1.string());
  ASSERT_GE(fd1, 0);
  ASSERT_TRUE(model.SerializeToFileDescriptor(fd1));
  CloseFd(fd1);

  const std::filesystem::path path2 =
      std::filesystem::temp_directory_path() / "onnx_light_save_fd.bin";
  const int fd2 = OpenTempFdForWrite(path2.string());
  ASSERT_GE(fd2, 0);
  ASSERT_TRUE(model.SaveToFileDescriptor(fd2));
  CloseFd(fd2);

  EXPECT_EQ(ReadWholeFile(path1.string()), ReadWholeFile(path2.string()));
  std::filesystem::remove(path1);
  std::filesystem::remove(path2);
}

TEST(proto_stream_api, SerializeAsString) {
  AttributeProto attr = MakeIntAttr();
  const std::string serialized = attr.SerializeAsString();
  EXPECT_FALSE(serialized.empty());

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, SerializeAsStringMatchesSerializeToString) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");

  std::string via_to_string;
  ASSERT_TRUE(model.SerializeToString(via_to_string));
  const std::string via_as_string = model.SerializeAsString();
  EXPECT_EQ(via_to_string, via_as_string);
}

TEST(proto_stream_api, SerializeToArrayReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  const size_t n = attr.ByteSizeLong();
  std::vector<uint8_t> buffer(n);
  ASSERT_TRUE(attr.SerializeToArray(buffer.data(), static_cast<int>(n)));

  AttributeProto parsed;
  ASSERT_TRUE(
      parsed.ParseFromString(std::string(reinterpret_cast<const char *>(buffer.data()), n)));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, SerializeToArrayMatchesSerializeToString) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");

  std::string via_to_string;
  ASSERT_TRUE(model.SerializeToString(via_to_string));
  const size_t n = via_to_string.size();
  std::vector<uint8_t> buffer(n);
  ASSERT_TRUE(model.SerializeToArray(buffer.data(), static_cast<int>(n)));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(buffer.data()), n), via_to_string);
}

TEST(proto_stream_api, SerializeToOstreamReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::ostringstream oss;
  ASSERT_TRUE(attr.SerializeToOstream(&oss));
  const std::string serialized = oss.str();
  EXPECT_FALSE(serialized.empty());

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, SerializeToOstreamMatchesSerializeToString) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");

  std::string via_to_string;
  ASSERT_TRUE(model.SerializeToString(via_to_string));
  std::ostringstream oss;
  ASSERT_TRUE(model.SerializeToOstream(&oss));
  EXPECT_EQ(oss.str(), via_to_string);
}

TEST(proto_stream_api, SerializeToOStreamReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::ostringstream oss;
  ASSERT_TRUE(attr.SerializeToOStream(&oss));
  const std::string serialized = oss.str();
  EXPECT_FALSE(serialized.empty());

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, SerializeToOStreamMatchesSerializeToOstream) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");

  std::ostringstream oss1;
  ASSERT_TRUE(model.SerializeToOstream(&oss1));
  std::ostringstream oss2;
  ASSERT_TRUE(model.SerializeToOStream(&oss2));
  EXPECT_EQ(oss1.str(), oss2.str());
}
