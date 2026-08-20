// Unit tests for the protobuf-compatible message API surface that onnx-light
// exposes on generated proto classes: ParseFromString / SerializeToString now
// return bool (true on success) and ByteSizeLong() reports the serialized size
// without performing a real serialization (it delegates to SerializeSize()).
#include "onnx.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <gtest/gtest.h>
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
