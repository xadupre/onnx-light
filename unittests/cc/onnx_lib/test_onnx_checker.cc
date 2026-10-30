// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from https://github.com/onnx/onnx/tree/main/onnx/test/cpp/checker_test.cc

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#ifdef _WIN32
#include "../../../onnx_proto/tests/windows_junction.h"
#include <io.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/common/path.h"
#include "onnx_lib/common/scoped_resource.h"

namespace fs = std::filesystem;

using ONNX_LIGHT_NAMESPACE::ScopedFd;

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

TEST(CHECKER, RejectsUnknownTensorDataType) {
  checker::CheckerContext ctx;
  for (int dtype : {-1, 999}) {
    SCOPED_TRACE(dtype);
    TensorProto tensor;
    tensor.set_name("t");
    tensor.set_data_type(dtype);
    tensor.add_dims(1);
    tensor.set_raw_data(std::string(1, '\0'));
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);

    tensor.clear_raw_data();
    tensor.ref_dims()[0] = 0;
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);

    tensor.ref_dims()[0] = 1;
    tensor.set_data_location(TensorProto::EXTERNAL);
    auto *location = tensor.add_external_data();
    location->set_key("location");
    location->set_value("#in-memory");
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  }
}

TEST(CHECKER, ValidDataLocationTest) {
#ifndef ONNX_NO_EXCEPTIONS
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location("localfolder", "..",
                                                                             "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   "localfolder", "/usr/any", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   "localfolder", "./sub/example", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   "localfolder", "sub/example", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
#endif
}

TEST(CHECKER, ExternalTensorDataBounds) {
#ifndef ONNX_NO_EXCEPTIONS
  const fs::path dir = fs::temp_directory_path() / "onnx_light_external_bounds_checker_test";
  fs::create_directories(dir);
  {
    std::ofstream out(dir / "weights.bin", std::ios::binary);
    out << "0123456789";
  }
  checker::CheckerContext ctx;
  ctx.set_model_dir(ONNX_LIGHT_NAMESPACE::path_to_utf8(dir));
  TensorProto tensor;
  tensor.set_name("weights");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(2);
  tensor.set_data_location(TensorProto::EXTERNAL);
  auto *location = tensor.add_external_data();
  location->set_key("location");
  location->set_value("weights.bin");
  auto *offset = tensor.add_external_data();
  offset->set_key("offset");
  offset->set_value("2");
  auto *length = tensor.add_external_data();
  length->set_key("length");
  length->set_value("8");
  EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  length->set_value("9");
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  length->set_value("7");
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  length->set_value("8");
  offset->set_value("3");
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  offset->set_value("11");
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  offset->set_value("2");

  for (const auto &invalid : {"", "-1", "+2", " 2", "abc", "18446744073709551616"}) {
    offset->set_value(invalid);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError) << invalid;
    offset->set_value("2");
    length->set_value(invalid);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError) << invalid;
    length->set_value("8");
  }
  length->set_value("1");
  auto *last_length = tensor.add_external_data();
  last_length->set_key("length");
  last_length->set_value("8");
  EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  auto *last_offset = tensor.add_external_data();
  last_offset->set_key("offset");
  last_offset->set_value("2");
  offset->set_value("invalid");
  EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  offset->set_value("2");
  last_length->set_value("7");
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  last_length->set_value("8");

  last_length->set_value("8");
  tensor.set_data_type(TensorProto::INT4);
  tensor.ref_dims().clear();
  tensor.add_dims(16);
  EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  tensor.ref_dims().clear();
  tensor.add_dims(17);
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.ref_dims().clear();
  tensor.add_dims(2);
  TensorProto whole_file = tensor;
  whole_file.ref_external_data().clear();
  auto *whole_location = whole_file.add_external_data();
  whole_location->set_key("location");
  whole_location->set_value("weights.bin");
  EXPECT_NO_THROW(checker::check_tensor(whole_file, ctx));
  auto *end_offset = whole_file.add_external_data();
  end_offset->set_key("offset");
  end_offset->set_value("2");
  EXPECT_NO_THROW(checker::check_tensor(whole_file, ctx));
  end_offset->set_value("3");
  EXPECT_THROW(checker::check_tensor(whole_file, ctx), checker::ValidationError);
  end_offset->clear_value();
  EXPECT_THROW(checker::check_tensor(whole_file, ctx), checker::ValidationError);
  end_offset->set_value("2");
  whole_location->set_value("#in-memory");
  EXPECT_NO_THROW(checker::check_tensor(whole_file, ctx));
  whole_location->set_value("weights.bin");
  whole_file.set_data_type(TensorProto::INT2);
  whole_file.ref_dims().clear();
  whole_file.add_dims(std::numeric_limits<int64_t>::max());
  EXPECT_THROW(checker::check_tensor(whole_file, ctx), checker::ValidationError);
  tensor.add_dims(-1);
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  TensorProto raw;
  raw.set_name("raw");
  raw.set_data_type(TensorProto::FLOAT);
  raw.add_dims(2);
  raw.set_raw_data("1234567");
  EXPECT_THROW(checker::check_tensor(raw, ctx), checker::ValidationError);
  fs::remove(dir / "weights.bin");
  fs::remove(dir);
#endif
}

TEST(CHECKER, TensorSegmentRawData) {
  checker::CheckerContext ctx;
  for (const auto &[dtype, bytes] :
       {std::pair{TensorProto::BOOL, 4},  {TensorProto::UINT8, 4},
        {TensorProto::INT8, 4},           {TensorProto::FLOAT8E4M3FN, 4},
        {TensorProto::FLOAT8E4M3FNUZ, 4}, {TensorProto::FLOAT8E5M2, 4},
        {TensorProto::FLOAT8E5M2FNUZ, 4}, {TensorProto::FLOAT8E8M0, 4},
        {TensorProto::FLOAT16, 8},        {TensorProto::BFLOAT16, 8},
        {TensorProto::UINT16, 8},         {TensorProto::INT16, 8},
        {TensorProto::FLOAT, 16},         {TensorProto::INT32, 16},
        {TensorProto::UINT32, 16},        {TensorProto::DOUBLE, 32},
        {TensorProto::INT64, 32},         {TensorProto::UINT64, 32},
        {TensorProto::COMPLEX64, 32},     {TensorProto::COMPLEX128, 64},
        {TensorProto::INT4, 2},           {TensorProto::UINT4, 2},
        {TensorProto::FLOAT4E2M1, 2},     {TensorProto::INT2, 1},
        {TensorProto::UINT2, 1},          {TensorProto::FLOAT6E2M3, 3},
        {TensorProto::FLOAT6E3M2, 3}}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    TensorProto tensor;
    tensor.set_data_type(dtype);
    tensor.add_dims(1000);
    tensor.mutable_segment()->set_begin(5);
    tensor.mutable_segment()->set_end(9);
    tensor.set_raw_data(std::string(bytes, '\0'));
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    tensor.ref_raw_data().resize(bytes - 1);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    tensor.set_raw_data(std::string(bytes + 1, '\0'));
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  }
}

TEST(CHECKER, TensorSegmentTypedData) {
  checker::CheckerContext ctx;
  for (const auto dtype :
       {TensorProto::FLOAT, TensorProto::COMPLEX64, TensorProto::DOUBLE, TensorProto::COMPLEX128,
        TensorProto::INT32, TensorProto::INT64, TensorProto::UINT32, TensorProto::UINT64,
        TensorProto::STRING, TensorProto::BOOL, TensorProto::FLOAT16, TensorProto::FLOAT6E2M3,
        TensorProto::FLOAT6E3M2, TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1,
        TensorProto::INT2, TensorProto::UINT2}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    TensorProto tensor;
    tensor.set_data_type(dtype);
    tensor.add_dims(1000);
    tensor.mutable_segment()->set_begin(2);
    tensor.mutable_segment()->set_end(19);
    const int count = dtype == TensorProto::COMPLEX64 || dtype == TensorProto::COMPLEX128 ? 34
                      : dtype == TensorProto::INT4 || dtype == TensorProto::UINT4 ||
                              dtype == TensorProto::FLOAT4E2M1
                          ? 3
                      : dtype == TensorProto::INT2 || dtype == TensorProto::UINT2 ? 2
                                                                                  : 17;
    auto resize_data = [&](int size) {
      switch (dtype) {
      case TensorProto::FLOAT:
      case TensorProto::COMPLEX64:
        tensor.ref_float_data().resize(size);
        break;
      case TensorProto::DOUBLE:
      case TensorProto::COMPLEX128:
        tensor.ref_double_data().resize(size);
        break;
      case TensorProto::INT64:
        tensor.ref_int64_data().resize(size);
        break;
      case TensorProto::UINT32:
      case TensorProto::UINT64:
        tensor.ref_uint64_data().resize(size);
        break;
      case TensorProto::STRING:
        tensor.ref_string_data().resize(size);
        break;
      default:
        tensor.ref_int32_data().resize(size);
      }
    };
    resize_data(count);
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    resize_data(count - 1);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    resize_data(count + 1);
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    tensor.clear_segment();
    tensor.ref_dims()[0] = 17;
    resize_data(count);
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    resize_data(count - 1);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  }
}

TEST(CHECKER, TensorSegmentShapeAndSizeOverflow) {
  checker::CheckerContext ctx;
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(std::numeric_limits<int64_t>::max());
  tensor.mutable_segment()->set_begin(std::numeric_limits<int64_t>::max() - 1);
  tensor.mutable_segment()->set_end(std::numeric_limits<int64_t>::max());
  tensor.set_raw_data(std::string(4, '\0'));
  EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
  tensor.add_dims(2);
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  tensor.ref_dims()[1] = -1;
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  tensor.ref_dims().resize(1);
  tensor.mutable_segment()->set_begin(0);
  EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  tensor.clear_raw_data();
  for (auto dtype : {TensorProto::COMPLEX64, TensorProto::COMPLEX128}) {
    tensor.set_data_type(dtype);
    if (dtype == TensorProto::COMPLEX64) {
      tensor.add_float_data(0);
    } else {
      tensor.clear_float_data();
      tensor.add_double_data(0);
    }
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
  }
}

TEST(CHECKER, TensorInvalidSegmentAllPayloads) {
  for (int payload = 0; payload < 4; ++payload) {
    SCOPED_TRACE(payload);
    checker::CheckerContext ctx;
    TensorProto tensor;
    tensor.set_data_type(TensorProto::FLOAT);
    tensor.add_dims(1000);
    if (payload == 0) {
      tensor.set_raw_data(std::string(4000, '\0'));
    } else if (payload == 1) {
      tensor.ref_float_data().resize(1000);
    } else {
      tensor.set_data_location(TensorProto::EXTERNAL);
      auto *location = tensor.add_external_data();
      location->set_key("location");
      location->set_value("#in-memory");
      ctx.set_skip_external_data_location_check(payload == 3);
    }
    for (int invalid = 0; invalid < 6; ++invalid) {
      SCOPED_TRACE(invalid);
      auto *segment = tensor.mutable_segment();
      segment->set_begin(0);
      segment->set_end(4);
      switch (invalid) {
      case 0:
        segment->clear_begin();
        break;
      case 1:
        segment->clear_end();
        break;
      case 2:
        segment->set_begin(-1);
        break;
      case 3:
        segment->set_begin(5);
        break;
      case 4:
        segment->set_end(1001);
        break;
      case 5:
        segment->set_begin(std::numeric_limits<int64_t>::min());
        segment->set_end(std::numeric_limits<int64_t>::max());
        break;
      }
      EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    }
  }
}

TEST(CHECKER, TensorEmptySegment) {
  checker::CheckerContext ctx;
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(1000);
  for (int64_t position : {0, 500, 1000}) {
    tensor.mutable_segment()->set_begin(position);
    tensor.mutable_segment()->set_end(position);
    EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    tensor.add_float_data(0);
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    tensor.clear_float_data();
    tensor.set_raw_data("data");
    EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    tensor.clear_raw_data();
  }
}

TEST(CHECKER, TensorSegmentFloat6Padding) {
  checker::CheckerContext ctx;
  for (auto dtype : {TensorProto::FLOAT6E2M3, TensorProto::FLOAT6E3M2}) {
    for (const auto &[count, last_byte] : {std::pair{1, 0x3f}, {2, 0x0f}, {3, 0x03}}) {
      TensorProto tensor;
      tensor.set_data_type(dtype);
      tensor.add_dims(1000);
      tensor.mutable_segment()->set_begin(5);
      tensor.mutable_segment()->set_end(5 + count);
      tensor.set_raw_data(std::string((count * 6 + 7) / 8, '\xff'));
      tensor.ref_raw_data()[tensor.raw_data().size() - 1] = static_cast<char>(last_byte);
      EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
      tensor.ref_raw_data()[tensor.raw_data().size() - 1] = static_cast<char>(last_byte + 1);
      EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
    }
  }
}

TEST(CHECKER, TensorSegmentExternalData) {
  const fs::path dir = fs::temp_directory_path() / "onnx_light_segment_checker_test";
  fs::create_directories(dir);
  checker::CheckerContext ctx;
  ctx.set_model_dir(ONNX_LIGHT_NAMESPACE::path_to_utf8(dir));
  for (int offset : {0, 8}) {
    for (int bytes : {15, 16}) {
      SCOPED_TRACE(offset);
      SCOPED_TRACE(bytes);
      {
        std::ofstream out(dir / "weights.bin", std::ios::binary);
        out << std::string(offset + bytes, '\0');
      }
      TensorProto tensor;
      tensor.set_data_type(TensorProto::FLOAT);
      tensor.add_dims(1000);
      tensor.mutable_segment()->set_begin(1);
      tensor.mutable_segment()->set_end(5);
      tensor.set_data_location(TensorProto::EXTERNAL);
      auto *location = tensor.add_external_data();
      location->set_key("location");
      location->set_value("weights.bin");
      auto *start = tensor.add_external_data();
      start->set_key("offset");
      start->set_value(std::to_string(offset));
      if (bytes == 16) {
        EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
        auto *length = tensor.add_external_data();
        length->set_key("length");
        length->set_value("16");
        EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
        length->set_value("15");
        EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
      } else {
        EXPECT_THROW(checker::check_tensor(tensor, ctx), checker::ValidationError);
      }
      tensor.mutable_segment()->set_end(1);
      EXPECT_NO_THROW(checker::check_tensor(tensor, ctx));
    }
  }
  std::ofstream(dir / "weights.bin", std::ios::binary).close();
  TensorProto empty;
  empty.set_data_type(TensorProto::FLOAT);
  empty.add_dims(1000);
  empty.mutable_segment()->set_begin(1000);
  empty.mutable_segment()->set_end(1000);
  empty.set_data_location(TensorProto::EXTERNAL);
  auto *location = empty.add_external_data();
  location->set_key("location");
  location->set_value("weights.bin");
  EXPECT_NO_THROW(checker::check_tensor(empty, ctx));
  fs::remove(dir / "weights.bin");
  fs::remove(dir);
}

TEST(CHECKER, ValidDataLocationSymLinkTest) {
#if !defined(ONNX_NO_EXCEPTIONS) && !defined(_WIN32)
  // Use a temp directory as the base_dir (simulating the model directory).
  // We pass a relative filename as the location so that the absolute-path
  // rejection is NOT triggered, and the is_symlink() check is actually exercised.
  fs::path modelDir = fs::temp_directory_path() / "onnx_light_symlink_checker_test";
  fs::remove_all(modelDir);
  fs::create_directories(modelDir);

  // Create a regular target file so the symlink has a valid target.
  fs::path target = modelDir / "target.data";
  {
    std::ofstream ofs(target);
    ofs << "test data";
  }

  // Create a symlink pointing to the target file.
  fs::path link = modelDir / "link.data";
  fs::create_symlink(target, link);

  // Pass relative filename "link.data" — the checker resolves it to
  // modelDir/link.data and should reject it because it is a symlink.
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   modelDir.string(), "link.data", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);

  fs::remove_all(modelDir);
#endif
}

TEST(CHECKER, ValidDataLocationParentDirSymLinkTest) {
#if !defined(ONNX_NO_EXCEPTIONS) && !defined(_WIN32)
  // Test that symlinks in parent directory components are detected.
  // A location like "symlink_subdir/real_file.data" where symlink_subdir
  // is a symlink to an outside directory should be rejected by the
  // canonical path containment check in checker.cc.
  fs::path modelDir = fs::temp_directory_path() / "onnx_light_parent_symlink_test";
  fs::remove_all(modelDir);
  fs::create_directories(modelDir);

  // Create a target directory outside the model directory.
  fs::path outsideDir = fs::temp_directory_path() / "onnx_light_outside_target";
  fs::remove_all(outsideDir);
  fs::create_directories(outsideDir);

  // Create a real file in the outside directory.
  fs::path targetFile = outsideDir / "secret.data";
  {
    std::ofstream ofs(targetFile);
    ofs << "sensitive data";
  }

  // Create a directory symlink inside modelDir pointing outside.
  fs::path symlinkSubdir = modelDir / "subdir";
  fs::create_directory_symlink(outsideDir, symlinkSubdir);

  // "subdir/secret.data" is a relative path where "subdir" is a symlink.
  // The canonical path resolves outside modelDir, so this should be rejected.
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   modelDir.string(), "subdir/secret.data", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);

  fs::remove_all(modelDir);
  fs::remove_all(outsideDir);
#endif
}

TEST(CHECKER, HashPrefixedRelativeModelDirTest) {
#ifndef ONNX_NO_EXCEPTIONS
  const fs::path modelDir = "#onnx_light_external_checker_test";
  fs::remove_all(modelDir);
  fs::create_directories(modelDir / "sub");
  const std::string base_dir = ONNX_LIGHT_NAMESPACE::path_to_utf8(modelDir);
  {
    std::ofstream ofs(modelDir / "sub" / "weights.bin", std::ios::binary);
    ofs << "data";
  }
  EXPECT_EQ(checker::resolve_external_data_location(base_dir, "sub/weights.bin", "t"),
            modelDir / "sub" / "weights.bin");
  {
    ScopedFd fd(
        static_cast<int>(checker::open_external_data(base_dir, "sub/weights.bin", "t", true)));
    ASSERT_GE(fd.get(), 0);
    char buf[4];
#ifdef _WIN32
    EXPECT_EQ(_read(fd.get(), buf, 4), 4);
#else
    EXPECT_EQ(read(fd.get(), buf, 4), 4);
#endif
    EXPECT_EQ(std::string(buf, 4), "data");
  }
  EXPECT_THROW(checker::resolve_external_data_location(base_dir, "sub/missing.bin", "t"),
               checker::ValidationError);
  EXPECT_THROW(checker::resolve_external_data_location(base_dir, "sub", "t"),
               checker::ValidationError);
  for (const auto &dir : {base_dir, std::string("onnx_light_in_memory_checker_test")}) {
    EXPECT_NO_THROW({
      EXPECT_EQ(checker::resolve_external_data_location(dir, "#in-memory", "t"),
                ONNX_LIGHT_NAMESPACE::utf8_to_path(dir) / "#in-memory");
    });
  }

#ifndef _WIN32
  const fs::path outsideDir = fs::temp_directory_path() / "onnx_light_hash_checker_outside";
  fs::remove_all(outsideDir);
  fs::create_directories(outsideDir);
  {
    std::ofstream ofs(outsideDir / "secret.data");
    ofs << "secret";
  }
  fs::create_directory_symlink(outsideDir, modelDir / "link");
  EXPECT_THROW(checker::resolve_external_data_location(base_dir, "link/secret.data", "t"),
               checker::ValidationError);
  fs::create_hard_link(modelDir / "sub" / "weights.bin", modelDir / "sub" / "hardlink.data");
  EXPECT_THROW(checker::resolve_external_data_location(base_dir, "sub/hardlink.data", "t"),
               checker::ValidationError);
  fs::remove_all(outsideDir);
#endif
  fs::remove_all(modelDir);
#endif
}

TEST(CHECKER, OpenExternalDataTest) {
#ifndef ONNX_NO_EXCEPTIONS
  // Use a UTF-8 directory name (raw bytes for C++20 char8_t compat).
  fs::path dir = fs::temp_directory_path() / ONNX_LIGHT_NAMESPACE::utf8_to_path(
                                                 "\xe6\xa8\xa1\xe5\x9e\x8b_onnx_light_open_test");
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::string dir_utf8 = ONNX_LIGHT_NAMESPACE::path_to_utf8(dir);

  // Read existing file (UTF-8 filename)
  std::string utf8_file = "\xe3\x83\x86\xe3\x83\xb3\xe3\x82\xbd\xe3\x83\xab.bin";
  {
    std::ofstream ofs(dir / ONNX_LIGHT_NAMESPACE::utf8_to_path(utf8_file), std::ios::binary);
    ofs << "data";
  }
  auto raw = ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, utf8_file, "t", true);
  ASSERT_GE(raw, 0);
  {
    ScopedFd fd(static_cast<int>(raw));
    ASSERT_GE(fd.get(), 0);
    char buf[4];
#ifdef _WIN32
    EXPECT_EQ(_read(fd.get(), buf, 4), 4);
#else
    EXPECT_EQ(read(fd.get(), buf, 4), 4);
#endif
    EXPECT_EQ(std::string(buf, 4), "data");
  } // fd closed by ScopedFd

  // Write creates new file
  raw = ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "new.bin", "t", false);
  ASSERT_GE(raw, 0);
  {
    ScopedFd fd(static_cast<int>(raw));
    ASSERT_GE(fd.get(), 0);
  } // fd closed by ScopedFd
  EXPECT_TRUE(fs::exists(dir / "new.bin"));

  // Reject invalid locations
  EXPECT_THROW(
      ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "missing.bin", "t", true),
      ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "/etc/passwd", "t", true),
      ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "../escape", "t", true),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, ".", "t", false),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  fs::create_directories(dir / "sub");
  for (const bool read_only : {false, true}) {
    ScopedFd child(static_cast<int>(ONNX_LIGHT_NAMESPACE::checker::open_external_data(
        dir_utf8, "sub/weights.bin", "t", read_only)));
    EXPECT_GE(child.get(), 0);
    EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(
                     dir_utf8, "sub/nested/weights.bin", "t", read_only),
                 ONNX_LIGHT_NAMESPACE::checker::ValidationError);
    EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "sub/../weights.bin",
                                                                   "t", read_only),
                 ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  }
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(dir_utf8,
                                                                          "./sub/weights.bin", "t"),
            dir / "sub/weights.bin");
  // UTF-8 traversal
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(
                   dir_utf8, "../\xe3\x83\x86\xe3\x83\xb3\xe3\x82\xbd\xe3\x83\xab.bin", "t", true),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);

#ifndef _WIN32
  // Symlink to outside directory
  fs::path outside = fs::temp_directory_path() / "onnx_light_open_ext_outside";
  fs::remove_all(outside);
  fs::create_directories(outside);
  {
    std::ofstream ofs(outside / "secret.data", std::ios::binary);
    ofs << "secret";
  }
  fs::create_symlink(outside / "secret.data", dir / "link.data");
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "link.data", "t", true),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  fs::remove_all(outside);
#endif

  fs::remove_all(dir);
#endif
}

TEST(CHECKER, OpenExternalDataEmptyBaseDirTest) {
#ifndef ONNX_NO_EXCEPTIONS
  fs::path original_dir = fs::current_path();
  fs::path dir = fs::temp_directory_path() / "onnx_light_empty_base_checker_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    std::ofstream ofs(dir / "data.bin", std::ios::binary);
    ofs << "data";
  }

  fs::current_path(dir);

  EXPECT_NO_THROW({
    EXPECT_EQ(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location("", "data.bin", "t"),
              fs::path("data.bin"));
  });
  EXPECT_NO_THROW({
    ScopedFd fd(static_cast<int>(
        ONNX_LIGHT_NAMESPACE::checker::open_external_data("", "data.bin", "t", true)));
    EXPECT_GE(fd.get(), 0);
  });
  EXPECT_NO_THROW({
    ScopedFd fd(static_cast<int>(
        ONNX_LIGHT_NAMESPACE::checker::open_external_data("", "created.bin", "t", false)));
    EXPECT_GE(fd.get(), 0);
  });
  EXPECT_TRUE(fs::exists(dir / "created.bin"));

#ifndef _WIN32
  fs::path outside = fs::temp_directory_path() / "onnx_light_empty_base_outside";
  fs::remove_all(outside);
  fs::create_directories(outside);
  {
    std::ofstream ofs(outside / "secret.bin", std::ios::binary);
    ofs << "secret";
  }
  fs::create_symlink(outside / "secret.bin", dir / "link.bin");
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data("", "link.bin", "t", true),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  fs::create_directory_symlink(outside, dir / "subdir");
  EXPECT_THROW(
      ONNX_LIGHT_NAMESPACE::checker::open_external_data("", "subdir/secret.bin", "t", true),
      ONNX_LIGHT_NAMESPACE::checker::ValidationError);
#endif

  fs::current_path(original_dir);
  fs::remove_all(dir);
#ifndef _WIN32
  fs::remove_all(outside);
#endif
#endif
}

#ifndef _WIN32
TEST(CHECKER, OpenExternalDataRetainsVerifiedFileAndRejectsUnsafeCreation) {
  const fs::path root = "test_checker_anchored_external_data";
  fs::remove_all(root);
  fs::create_directories(root / "model" / "sub");
  fs::create_directories(root / "outside");
  {
    std::ofstream(root / "model" / "sub" / "data.bin") << "original";
    std::ofstream(root / "outside" / "data.bin") << "sentinel";
  }
  {
    ScopedFd descriptor(static_cast<int>(
        checker::open_external_data((root / "model").string(), "sub/data.bin", "tensor", true)));
    fs::rename(root / "model" / "sub", root / "model" / "pinned");
    fs::create_directory_symlink(fs::absolute(root / "outside"), root / "model" / "sub");
    char bytes[8]{};
    EXPECT_EQ(::read(descriptor.get(), bytes, 8), 8);
    EXPECT_EQ(std::string(bytes, 8), "original");
    EXPECT_THROW(
        checker::open_external_data((root / "model").string(), "sub/new.bin", "tensor", false),
        checker::ValidationError);
    EXPECT_FALSE(fs::exists(root / "outside" / "new.bin"));
    fs::create_hard_link(root / "outside" / "data.bin", root / "model" / "linked.bin");
    EXPECT_THROW(
        checker::open_external_data((root / "model").string(), "linked.bin", "tensor", false),
        checker::ValidationError);
    EXPECT_THROW(checker::open_external_data((root / "model").string(), "pinned", "tensor", false),
                 checker::ValidationError);
  }
  {
    std::ifstream outside(root / "outside" / "data.bin");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(outside), {}), "sentinel");
  }
  fs::remove_all(root);
}
#else
TEST(CHECKER, OpenExternalDataRejectsActualWindowsJunctionBeforeCreation) {
  const fs::path root = "test_checker_external_data_junction";
  fs::remove_all(root);
  fs::create_directories(root / "model");
  fs::create_directories(root / "outside");
  std::ofstream(root / "outside" / "data.bin") << "sentinel";
  const auto error =
      onnx_light_test::CreateWindowsJunction(root / "model" / "junction", root / "outside");
  ASSERT_FALSE(error) << error.message();
  for (bool read_only : {false, true})
    EXPECT_THROW(checker::open_external_data((root / "model").string(), "junction/data.bin",
                                             "tensor", read_only),
                 checker::ValidationError);
  EXPECT_THROW(
      checker::open_external_data((root / "model").string(), "junction/new.bin", "tensor", false),
      checker::ValidationError);
  EXPECT_FALSE(fs::exists(root / "outside" / "new.bin"));
  {
    std::ifstream outside(root / "outside" / "data.bin");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(outside), {}), "sentinel");
  }
  fs::remove_all(root);
}
#endif

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
