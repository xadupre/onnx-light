// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/test_case.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr const char *kAiRtDomain = "ai.rt";

std::vector<std::filesystem::path> &CleanupPaths() {
  static std::vector<std::filesystem::path> paths;
  return paths;
}

void CleanupWeightsFiles() {
  for (const auto &path : CleanupPaths()) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
}

void RegisterCleanupPath(const std::filesystem::path &path) {
  [[maybe_unused]] static const int cleanup_registered = []() {
    std::atexit(CleanupWeightsFiles);
    return 0;
  }();
  auto &paths = CleanupPaths();
  if (std::find(paths.begin(), paths.end(), path) == paths.end()) {
    paths.push_back(path);
  }
}

std::string WriteWeightsFile(const std::string &filename, const std::vector<uint8_t> &bytes) {
  namespace fs = std::filesystem;
  const fs::path path = fs::temp_directory_path() / filename;
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

  Expect(node, /*inputs=*/{}, {output}, case_name,
         {DefaultOpset(18), onnx_kernels::kernel::OpsetId(kAiRtDomain, 1)}, "backend-test",
         registry);
}

} // namespace

void RegisterDelayedInitializerCases(std::vector<TestCase> &registry) {
  RegisterDelayedInitializerCase(
      registry, "test_cc_delayedinitializer_file", "onnx_light_backend_delayedinitializer_file.bin",
      "file", 8, std::vector<uint8_t>(8, 0), Tensor::FromFloat("", {2}, {1.5f, -2.0f}));

  RegisterDelayedInitializerCase(registry, "test_cc_delayedinitializer_cpu",
                                 "onnx_light_backend_delayedinitializer_cpu.bin", "cpu", 0, {},
                                 Tensor::FromFloat("", {3}, {3.0f, 4.0f, 5.0f}));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
