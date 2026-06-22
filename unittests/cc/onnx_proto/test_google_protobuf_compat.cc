#include "fields.h"
#include "google_protobuf_compat.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ONNX_LIGHT_NAMESPACE::utils;

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

} // namespace
