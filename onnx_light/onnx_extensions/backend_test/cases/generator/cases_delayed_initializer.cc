// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr const char *kAiRtDomain = "ai.rt";

struct CleanupState {
  std::mutex mutex;
  std::vector<std::filesystem::path> paths;
};

CleanupState &GetCleanupState() {
  static CleanupState state;
  return state;
}

// Removes every temporary weights file registered for DelayedInitializer
// backend cases and logs cleanup failures at process exit.
void CleanupWeightsFiles() {
  auto &state = GetCleanupState();
  std::lock_guard<std::mutex> lock(state.mutex);
  for (const auto &path : state.paths) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
      std::fprintf(stderr, "DelayedInitializer backend-test cleanup failed for '%s': %s\n",
                   path.string().c_str(), ec.message().c_str());
    }
  }
}

// Registers one temporary weights-file path for process-exit cleanup, making
// sure the atexit handler is installed exactly once in a thread-safe way.
void RegisterCleanupPath(const std::filesystem::path &path) {
  static std::once_flag cleanup_once;
  std::call_once(cleanup_once, []() {
    // Construct the cleanup state before registering the atexit handler so it
    // remains alive when cleanup runs during process shutdown.
    (void)GetCleanupState();
    std::atexit(CleanupWeightsFiles);
  });
  auto &state = GetCleanupState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (std::find(state.paths.begin(), state.paths.end(), path) == state.paths.end()) {
    state.paths.push_back(path);
  }
}

// Writes one temporary weights file for a backend case, requiring a basename,
// registering cleanup, and returning the absolute path used by the test node.
std::string WriteWeightsFile(const std::string &filename, const std::vector<uint8_t> &bytes) {
  namespace fs = std::filesystem;
  const fs::path filename_path(filename);
  if (filename_path.has_parent_path()) {
    throw std::runtime_error("DelayedInitializer backend-test filename must be a basename, got '" +
                             filename_path.string() + "'.");
  }
  const fs::path path = fs::temp_directory_path() / filename_path.filename();
  std::error_code ec;
  fs::remove(path, ec);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("Unable to open DelayedInitializer backend-test weights file '" +
                             path.string() + "'.");
  }
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out.good()) {
    throw std::runtime_error(
        "Unable to write data to DelayedInitializer backend-test weights file '" + path.string() +
        "'.");
  }
  RegisterCleanupPath(path);
  return path.string();
}

// Creates one ``ai.rt::DelayedInitializer`` backend case, writes the expected
// bytes to a temporary weights file (optionally prefixed before ``offset``),
// and registers the single-node model with the provided expected output.
void RegisterDelayedInitializerCase(std::vector<TestCase> &registry, const std::string &case_name,
                                    const std::string &filename, const std::string &load_device,
                                    int64_t offset, const std::vector<uint8_t> &prefix,
                                    const Tensor &output) {
  std::vector<uint8_t> bytes = prefix;
  bytes.insert(bytes.end(), output.data.begin(), output.data.end());
  const std::string weights_path = WriteWeightsFile(filename, bytes);

  NodeProto node;
  node.set_op_type("DelayedInitializer");
  node.set_domain(kAiRtDomain);
  node.add_output("y");

  AttributeProto *shape = node.add_attribute();
  shape->set_name("shape");
  shape->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t dim : output.shape) {
    shape->add_ints(dim);
  }

  AttributeProto *dtype = node.add_attribute();
  dtype->set_name("dtype");
  dtype->set_type(AttributeProto::AttributeType::INT);
  dtype->set_i(output.data_type);

  AttributeProto *load = node.add_attribute();
  load->set_name("load_device");
  load->set_type(AttributeProto::AttributeType::STRING);
  load->set_s(load_device);

  AttributeProto *runtime = node.add_attribute();
  runtime->set_name("runtime_device");
  runtime->set_type(AttributeProto::AttributeType::STRING);
  runtime->set_s("cpu");

  AttributeProto *file = node.add_attribute();
  file->set_name("filename");
  file->set_type(AttributeProto::AttributeType::STRING);
  file->set_s(weights_path);

  AttributeProto *offset_attr = node.add_attribute();
  offset_attr->set_name("offset");
  offset_attr->set_type(AttributeProto::AttributeType::INT);
  offset_attr->set_i(offset);

  Expect(registry, std::move(node), case_name, {DefaultOpset(18), OpsetId(kAiRtDomain, 1)},
         [=]() -> IoData { return IoData{{}, {output}}; });
}

} // namespace

// Registers backend cases covering both supported load-device modes:
// eager CPU loading and deferred file loading.
void RegisterDelayedInitializerCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> big_shape = {512, 512};
    RegisterDelayedInitializerCase(registry, "test_cc_delayedinitializer_benchmark",
                                   "onnx_light_backend_delayedinitializer_benchmark.bin", "cpu", 0,
                                   {},
                                   Tensor::FromFloat("", big_shape, Randn<float>(big_shape, 4601)));
    return;
  }

  RegisterDelayedInitializerCase(
      registry, "test_cc_delayedinitializer_file", "onnx_light_backend_delayedinitializer_file.bin",
      "file", 8, std::vector<uint8_t>(8, 0), Tensor::FromFloat("", {2}, {1.5f, -2.0f}));

  RegisterDelayedInitializerCase(registry, "test_cc_delayedinitializer_cpu",
                                 "onnx_light_backend_delayedinitializer_cpu.bin", "cpu", 0, {},
                                 Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f}));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
