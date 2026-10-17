// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from https://github.com/onnx/onnx/tree/main/onnx/test/cpp/checker_test.cc

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#if defined(__linux__) && defined(__x86_64__)
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <thread>
#include <ucontext.h>
#endif

#ifdef _WIN32
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

TEST(CHECKER, ValidDataLocationTest) {
#ifndef ONNX_NO_EXCEPTIONS
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location("localfolder", "..",
                                                                             "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   "localfolder", "/usr/any", "tensor_name"),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(
                   "localfolder", "sub/../example", "tensor_name"),
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
  {
    std::ofstream weights(dir / "weights..bin", std::ios::binary);
    weights << "data";
  }
  EXPECT_EQ(
      ONNX_LIGHT_NAMESPACE::checker::resolve_external_data_location(dir_utf8, "weights..bin", "t"),
      dir / "weights..bin");

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
  fs::create_directories(dir / ".weights");
  raw = ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, ".weights/weights.bin", "t",
                                                          false);
  {
    ScopedFd fd(static_cast<int>(raw));
  }
  raw = ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "./.weights/weights.bin", "t",
                                                          true);
  {
    ScopedFd fd(static_cast<int>(raw));
  }
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(
                   dir_utf8, ".weights/../weights.bin", "t", false),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  for (const bool read_only : {false, true}) {
    EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(
                     dir_utf8, ".weights/nested/weights.bin", "t", read_only),
                 ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  }
  EXPECT_THROW(ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "", "t", false),
               ONNX_LIGHT_NAMESPACE::checker::ValidationError);
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
  fs::create_directory_symlink(outside, dir / "escape");
  EXPECT_THROW(
      ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "escape/secret.data", "t", true),
      ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_LIGHT_NAMESPACE::checker::open_external_data(dir_utf8, "escape/new.data", "t", false),
      ONNX_LIGHT_NAMESPACE::checker::ValidationError);
  fs::remove_all(outside);
#endif

  fs::remove_all(dir);
#endif
}

#if defined(__linux__) && defined(__x86_64__) && defined(SYS_openat2)
namespace {
const char *swap_directory = nullptr;
const char *saved_directory = nullptr;
const char *outside_directory = nullptr;
volatile sig_atomic_t swap_count = 0;

void SwapDirectoryBeforeFallback(int, siginfo_t *info, void *context) {
  if (info->si_syscall != SYS_openat2 || swap_count != 0)
    ::_exit(90);
  if (::rename(swap_directory, saved_directory) != 0 ||
      ::symlink(outside_directory, swap_directory) != 0)
    ::_exit(91);
  swap_count = 1;
  static_cast<ucontext_t *>(context)->uc_mcontext.gregs[REG_RAX] = -ENOSYS;
}
} // namespace

TEST(CHECKER, ExternalWriteFallbackRejectsSwappedDirectoryBeforeCreation) {
#ifndef ONNX_NO_EXCEPTIONS
  const fs::path root = fs::temp_directory_path() / "onnx_light_checker_fallback_swap";
  const fs::path outside = fs::temp_directory_path() / "onnx_light_checker_fallback_outside";
  fs::remove_all(root);
  fs::remove_all(outside);
  fs::create_directories(root / ".weights");
  fs::create_directories(outside);
  const std::string base = root.string();
  const std::string original = (root / ".weights").string();
  const std::string saved = (root / "saved").string();
  const std::string escaped = outside.string();
  EXPECT_EXIT(([&]() {
                // A fresh thread resets the production thread-local openat2 availability cache.
                std::thread worker([&]() {
                  swap_directory = original.c_str();
                  saved_directory = saved.c_str();
                  outside_directory = escaped.c_str();
                  struct sigaction action{};
                  action.sa_sigaction = SwapDirectoryBeforeFallback;
                  action.sa_flags = SA_SIGINFO;
                  ::sigemptyset(&action.sa_mask);
                  ASSERT_EQ(::sigaction(SIGSYS, &action, nullptr), 0);
                  struct sock_filter filter[] = {
                      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
                      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_openat2, 0, 1),
                      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
                      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)};
                  struct sock_fprog program{};
                  program.len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0]));
                  program.filter = filter;
                  ASSERT_EQ(::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
                  ASSERT_EQ(::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program), 0);
                  EXPECT_THROW(
                      checker::open_external_data(base, ".weights/escaped.bin", "t", false),
                      checker::ValidationError);
                  EXPECT_EQ(swap_count, 1);
                  EXPECT_FALSE(fs::exists(outside / "escaped.bin"));
                  fs::remove(root / ".weights");
                  fs::rename(root / "saved", root / ".weights");
                  for (const char *location : {"flat.bin", ".weights/safe.bin"}) {
                    ScopedFd output(
                        static_cast<int>(checker::open_external_data(base, location, "t", false)));
                    ASSERT_EQ(::write(output.get(), "safe", 4), 4);
                    ScopedFd input(
                        static_cast<int>(checker::open_external_data(base, location, "t", true)));
                    char bytes[4]{};
                    ASSERT_EQ(::read(input.get(), bytes, 4), 4);
                    EXPECT_EQ(std::string(bytes, 4), "safe");
                  }
                });
                worker.join();
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
              }()),
              ::testing::ExitedWithCode(0), "");
  EXPECT_FALSE(fs::exists(outside / "escaped.bin"));
  fs::remove_all(root);
  fs::remove_all(outside);
#endif
}
#endif

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

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
