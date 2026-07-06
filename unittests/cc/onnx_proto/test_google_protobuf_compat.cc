#include "fields.h"
#include "google_protobuf_compat.h"
#include "stream.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using namespace ONNX_LIGHT_NAMESPACE::utils;

// Opens *path* for binary writing and returns a raw file descriptor (or -1).
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

// Closes a raw file descriptor opened with OpenTempFdForWrite.
void CloseFd(int fd) {
#if defined(_WIN32)
  _close(fd);
#else
  ::close(fd);
#endif
}

// Reads a whole file into a std::string.
std::string ReadWholeFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Verify that google::protobuf aliases resolve correctly.
TEST(GoogleProtobufCompat, RepeatedFieldAlias) {
  google::protobuf::RepeatedField<int64_t> field;
  field.push_back(10);
  field.push_back(20);
  field.push_back(30);
  ASSERT_EQ(field.size(), 3u);
  ASSERT_EQ(field[0], 10);
  ASSERT_EQ(field[1], 20);
  ASSERT_EQ(field[2], 30);
}

TEST(GoogleProtobufCompat, RepeatedFieldData) {
  google::protobuf::RepeatedField<int32_t> field;
  field.push_back(1);
  field.push_back(2);
  field.push_back(3);
  const int32_t *ptr = field.data();
  ASSERT_EQ(ptr[0], 1);
  ASSERT_EQ(ptr[1], 2);
  ASSERT_EQ(ptr[2], 3);
}

TEST(GoogleProtobufCompat, RepeatedFieldResize) {
  google::protobuf::RepeatedField<float> field;
  field.Resize(5, 1.0f);
  ASSERT_EQ(field.size(), 5u);
  for (size_t i = 0; i < 5; ++i) {
    ASSERT_EQ(field[i], 1.0f);
  }
}

TEST(GoogleProtobufCompat, RepeatedFieldCopyFrom) {
  google::protobuf::RepeatedField<int64_t> src;
  src.push_back(100);
  src.push_back(200);

  google::protobuf::RepeatedField<int64_t> dst;
  dst.CopyFrom(src);
  ASSERT_EQ(dst.size(), 2u);
  ASSERT_EQ(dst[0], 100);
  ASSERT_EQ(dst[1], 200);
}

TEST(GoogleProtobufCompat, RepeatedFieldAssign) {
  std::vector<int64_t> v = {5, 6, 7, 8};
  google::protobuf::RepeatedField<int64_t> field;
  field.Assign(v.begin(), v.end());
  ASSERT_EQ(field.size(), 4u);
  ASSERT_EQ(field[0], 5);
  ASSERT_EQ(field[3], 8);
}

TEST(GoogleProtobufCompat, BackInserter) {
  google::protobuf::RepeatedField<int32_t> field;
  auto inserter = google::protobuf::RepeatedFieldBackInserter(&field);
  *inserter = 42;
  ++inserter;
  *inserter = 99;

  ASSERT_EQ(field.size(), 2u);
  ASSERT_EQ(field[0], 42);
  ASSERT_EQ(field[1], 99);
}

TEST(GoogleProtobufCompat, BackInserterWithCopy) {
  google::protobuf::RepeatedField<int64_t> field;
  std::vector<int64_t> src = {1, 2, 3, 4, 5};
  std::copy(src.begin(), src.end(), google::protobuf::RepeatedFieldBackInserter(&field));
  ASSERT_EQ(field.size(), 5u);
  ASSERT_EQ(field[4], 5);
}

TEST(GoogleProtobufCompat, ShutdownIsNoOp) {
  // Should not crash or have side effects.
  google::protobuf::ShutdownProtobufLibrary();
}

TEST(GoogleProtobufCompat, ArrayInputStream) {
  const char data[] = "hello world";
  google::protobuf::io::ArrayInputStream stream(data, 11);
  const void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_EQ(size, 11);
  ASSERT_EQ(std::string(static_cast<const char *>(ptr), static_cast<size_t>(size)), "hello world");
  ASSERT_FALSE(stream.Next(&ptr, &size));
}

TEST(GoogleProtobufCompat, IstreamInputStream) {
  std::istringstream ss("test data here");
  google::protobuf::io::IstreamInputStream stream(&ss, 64);
  const void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_GT(size, 0);
  std::string result(static_cast<const char *>(ptr), static_cast<size_t>(size));
  ASSERT_EQ(result, "test data here");
}

TEST(GoogleProtobufCompat, StringOutputStream) {
  std::string target;
  google::protobuf::io::StringOutputStream stream(&target);
  void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_GE(size, 5);
  std::memcpy(ptr, "hello", 5);
  stream.BackUp(size - 5);
  ASSERT_EQ(target, "hello");
}

// Every class in google_protobuf_compat.h must be a pure `using` alias to a
// concrete onnx-light class (no thin wrapper layer). These static assertions
// fail to compile if any alias is replaced by a distinct wrapper type.
TEST(GoogleProtobufCompat, AliasesAreOnnxLightTypes) {
  static_assert(std::is_same<google::protobuf::RepeatedField<int>,
                             ONNX_LIGHT_NAMESPACE::utils::RepeatedField<int>>::value,
                "RepeatedField must alias onnx-light RepeatedField");
  static_assert(std::is_same<google::protobuf::io::ArrayInputStream,
                             ONNX_LIGHT_NAMESPACE::utils::StringStream>::value,
                "ArrayInputStream must alias onnx-light StringStream");
  static_assert(std::is_same<google::protobuf::io::CodedInputStream,
                             ONNX_LIGHT_NAMESPACE::utils::CodedInputStream>::value,
                "CodedInputStream must alias onnx-light CodedInputStream");
  static_assert(std::is_same<google::protobuf::io::StringOutputStream,
                             ONNX_LIGHT_NAMESPACE::utils::StdStringWriteStream>::value,
                "StringOutputStream must alias onnx-light StdStringWriteStream");
  static_assert(std::is_same<google::protobuf::io::FileOutputStream,
                             ONNX_LIGHT_NAMESPACE::utils::FdWriteStream>::value,
                "FileOutputStream must alias onnx-light FdWriteStream");
  static_assert(std::is_same<google::protobuf::io::IstreamInputStream,
                             ONNX_LIGHT_NAMESPACE::utils::IstreamStream>::value,
                "IstreamInputStream must alias onnx-light IstreamStream");
  static_assert(std::is_same<google::protobuf::io::OstreamOutputStream,
                             ONNX_LIGHT_NAMESPACE::utils::OstreamWriteStream>::value,
                "OstreamOutputStream must alias onnx-light OstreamWriteStream");
  SUCCEED();
}

TEST(GoogleProtobufCompat, StringOutputStreamByteCount) {
  std::string target = "abc";
  google::protobuf::io::StringOutputStream stream(&target);
  EXPECT_EQ(stream.ByteCount(), 3);
  void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  std::memcpy(ptr, "de", 2);
  stream.BackUp(size - 2);
  EXPECT_EQ(target, "abcde");
  EXPECT_EQ(stream.ByteCount(), 5);
}

TEST(GoogleProtobufCompat, OstreamOutputStream) {
  std::ostringstream oss;
  {
    google::protobuf::io::OstreamOutputStream stream(&oss);
    void *ptr = nullptr;
    int size = 0;
    ASSERT_TRUE(stream.Next(&ptr, &size));
    ASSERT_GE(size, 3);
    std::memcpy(ptr, "abc", 3);
    stream.BackUp(size - 3);
    EXPECT_EQ(stream.ByteCount(), 3);
    ASSERT_TRUE(stream.Flush());
  }
  EXPECT_EQ(oss.str(), "abc");
}

TEST(GoogleProtobufCompat, FileOutputStream) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_fdwritestream.bin";
  int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  {
    google::protobuf::io::FileOutputStream stream(fd);
    void *ptr = nullptr;
    int size = 0;
    ASSERT_TRUE(stream.Next(&ptr, &size));
    ASSERT_GE(size, 5);
    std::memcpy(ptr, "hello", 5);
    stream.BackUp(size - 5);
    ASSERT_TRUE(stream.Flush());
    EXPECT_EQ(stream.ByteCount(), 5);
  }
  CloseFd(fd);
  EXPECT_EQ(ReadWholeFile(path.string()), "hello");
  std::filesystem::remove(path);
}

TEST(GoogleProtobufCompat, FileOutputStreamRawBytes) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "onnx_light_fdwritestream_raw.bin";
  int fd = OpenTempFdForWrite(path.string());
  ASSERT_GE(fd, 0);
  // A payload larger than the internal buffer exercises the looped platform write.
  std::string payload(9000, 'x');
  {
    google::protobuf::io::FileOutputStream stream(fd);
    stream.write_raw_bytes(reinterpret_cast<const uint8_t *>(payload.data()),
                           static_cast<int64_t>(payload.size()));
    ASSERT_TRUE(stream.Flush());
    EXPECT_EQ(stream.ByteCount(), static_cast<int64_t>(payload.size()));
  }
  CloseFd(fd);
  EXPECT_EQ(ReadWholeFile(path.string()), payload);
  std::filesystem::remove(path);
}

TEST(GoogleProtobufCompat, CodedInputStream) {
  const char data[] = "hello world";
  google::protobuf::io::ArrayInputStream in(data, 11);
  google::protobuf::io::CodedInputStream cis(&in);
  EXPECT_EQ(cis.TotalBytesLimit(), 0x7FFFFFFF);
  cis.SetTotalBytesLimit(123);
  EXPECT_EQ(cis.TotalBytesLimit(), 123);
}

// ---------------------------------------------------------------------------
// Native onnx-light stream classes directly implement the protobuf ZeroCopy
// interfaces (Next / BackUp / ByteCount), which the compat aliases rely on.
// ---------------------------------------------------------------------------

TEST(OnnxLightStream, StringStreamZeroCopy) {
  const char data[] = "abcdefghij";
  StringStream stream(data, 10);
  const void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_EQ(size, 10);
  EXPECT_EQ(std::string(static_cast<const char *>(ptr), 10), "abcdefghij");
  EXPECT_EQ(stream.ByteCount(), 10);
  // Push the last 4 bytes back and re-read them.
  stream.BackUp(4);
  EXPECT_EQ(stream.ByteCount(), 6);
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_EQ(size, 4);
  EXPECT_EQ(std::string(static_cast<const char *>(ptr), 4), "ghij");
  EXPECT_EQ(stream.ByteCount(), 10);
  EXPECT_FALSE(stream.Next(&ptr, &size));
}

TEST(OnnxLightStream, StringWriteStreamZeroCopy) {
  StringWriteStream stream;
  void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_GE(size, 5);
  std::memcpy(ptr, "hello", 5);
  stream.BackUp(size - 5);
  EXPECT_EQ(stream.ByteCount(), 5);
  EXPECT_EQ(stream.size(), 5);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(stream.data()), 5), "hello");
}

TEST(OnnxLightStream, BorrowedStringWriteStreamZeroCopy) {
  uint8_t buffer[8] = {0};
  BorrowedStringWriteStream stream(buffer, 8);
  void *ptr = nullptr;
  int size = 0;
  ASSERT_TRUE(stream.Next(&ptr, &size));
  ASSERT_EQ(size, 8);
  std::memcpy(ptr, "12345678", 8);
  EXPECT_EQ(stream.ByteCount(), 8);
  // Fixed-capacity buffer is exhausted: Next must now report false.
  EXPECT_FALSE(stream.Next(&ptr, &size));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(buffer), 8), "12345678");
}

} // namespace
