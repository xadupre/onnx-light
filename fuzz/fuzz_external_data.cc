// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Filesystem-aware external-data fuzzing, ported from onnx/onnx#8659.

#include "onnx_lib/checker.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::ParseOptions;
using ONNX_LIGHT_NAMESPACE::SerializeOptions;
using ONNX_LIGHT_NAMESPACE::TensorProto;
namespace checker = ONNX_LIGHT_NAMESPACE::checker;
namespace utils = ONNX_LIGHT_NAMESPACE::utils;
namespace fs = std::filesystem;

namespace {

const std::string kSentinelContent(64, '\xa5');
constexpr std::array<const char *, 11> kIntegers = {nullptr,
                                                    "0",
                                                    "1",
                                                    "-1",
                                                    "+1",
                                                    " 1 ",
                                                    "not-an-integer",
                                                    "2147483648",
                                                    "9223372036854775807",
                                                    "9223372036854775808",
                                                    "18446744073709551616"};

[[noreturn]] void finding(const char *message) {
  std::fprintf(stderr, "external-data fuzz finding: %s\n", message);
  std::abort();
}

void write_file(const fs::path &path, const std::string &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

class Workspace {
public:
  Workspace() {
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
      root = fs::temp_directory_path() / ("onnx-light-external-fuzz-" + std::to_string(random()) +
                                          "-" + std::to_string(random()));
      if (fs::create_directory(root)) {
        fs::permissions(root, fs::perms::owner_all);
        return;
      }
    }
    throw std::runtime_error("could not create external-data fuzz workspace");
  }

  ~Workspace() {
    std::error_code error;
    fs::remove_all(root, error);
  }

  void reset(size_t payload_size) const {
    fs::remove_all(base());
    fs::create_directories(base() / "nested");
    fs::create_directory(base() / "sub");
    // Read fixtures cannot contain the sentinel byte, independently of fuzz input.
    const std::string file_data(21 + std::min<size_t>(payload_size, 1024), 'D');
    write_file(base() / "data.bin", file_data);
    write_file(base() / "nested/data.bin", file_data);
    write_file(sentinel(), kSentinelContent);
    fs::create_hard_link(sentinel(), base() / "hardlink.bin");
#ifdef _WIN32
    // Windows may require additional privileges to create symlinks.
    std::error_code error;
    fs::create_symlink(sentinel(), base() / "symlink.bin", error);
#else
    fs::create_symlink(sentinel(), base() / "symlink.bin");
#endif
  }

  fs::path base() const { return root / "model"; }
  fs::path sentinel() const { return root / "sentinel.bin"; }

  void check_sentinel() const {
    std::ifstream file(sentinel(), std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    if (!file || content != kSentinelContent) {
      finding("operation modified or removed the sentinel");
    }
  }

private:
  fs::path root;
};

std::array<std::string, 16> locations(const Workspace &workspace) {
  return {"data.bin",
          "nested/data.bin",
          "missing.bin",
          "../sentinel.bin",
          "sub/../../sentinel.bin",
          workspace.sentinel().string(),
          "symlink.bin",
          "hardlink.bin",
          "..\\sentinel.bin",
          "\\\\server\\share\\data.bin",
          "C:data.bin",
          "C:\\data.bin",
          "%2e%2e/sentinel.bin",
          "./data.bin",
          "",
          "data.bin/child"};
}

bool forbidden_location(size_t index) { return (index >= 3 && index <= 11) || index == 14; }

void add_metadata(TensorProto &tensor, const char *key, const std::string &value) {
  auto *entry = tensor.add_external_data();
  entry->set_key(key);
  entry->set_value(value);
}

TensorProto make_tensor(const uint8_t *data, const std::string &location,
                        const std::string &alternate, const std::string &name, size_t tensor_size) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(TensorProto::DataType::UINT8);
  tensor.add_dims(static_cast<int64_t>(tensor_size));
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  add_metadata(tensor, "location", location);
  if (data[2] & 1) {
    add_metadata(tensor, "location", alternate);
  }
  for (size_t field = 3; field <= 4; ++field) {
    const char *key = field == 3 ? "offset" : "length";
    const size_t index = data[field] % kIntegers.size();
    if (kIntegers[index]) {
      add_metadata(tensor, key, kIntegers[index]);
    }
    if ((data[2] & (field == 3 ? 2 : 4)) && kIntegers[(index + 1) % kIntegers.size()]) {
      add_metadata(tensor, key, kIntegers[(index + 1) % kIntegers.size()]);
    }
  }
  if (data[2] & 8) {
    add_metadata(tensor, "unknown", "value");
    add_metadata(tensor, "another_unknown", "value");
  }
  if (data[2] & 16) {
    add_metadata(tensor, "checksum", "invalid-checksum");
  }
  if (data[2] & 32) {
    add_metadata(tensor, "basepath", alternate);
  }
  return tensor;
}

ModelProto make_model(const TensorProto &tensor) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(21);
  auto *graph = model.add_graph();
  graph->set_name("external-data-fuzz");
  *graph->add_initializer() = tensor;
  auto *output = graph->add_output();
  output->set_name(tensor.name());
  auto *type = output->add_type()->add_tensor_type();
  type->set_elem_type(static_cast<int32_t>(TensorProto::DataType::UINT8));
  type->add_shape()->add_dim()->set_dim_value(tensor.dims()[0]);
  return model;
}

void check_read(const TensorProto &tensor) {
  for (uint8_t byte : tensor.raw_data()) {
    if (byte == 0xa5) {
      finding("operation read sentinel bytes");
    }
  }
}

// The standalone loader has no allocation-budget option. Oversized lengths
// still reach the bounded streaming parser rather than allocating gigabytes.
bool needs_bounded_parser(const TensorProto &tensor) {
  for (const auto &entry : tensor.external_data()) {
    if ((entry.key() == "length" || entry.key() == "size") &&
        (entry.value() == "2147483648" || entry.value() == "9223372036854775807")) {
      return true;
    }
  }
  return false;
}

void exercise(const uint8_t *data, const Workspace &workspace, const std::string &location,
              const std::string &alternate, const std::string &payload, bool forbidden) {
  const unsigned operation = data[0] % 8;
  const std::string base = workspace.base().string();
  const std::string model_path = (workspace.base() / "model.onnx").string();
  const std::string weights_path = (workspace.base() / "default.bin").string();
  TensorProto tensor = make_tensor(data, location, alternate, "weight", payload.size());
  ModelProto model;
  ParseOptions parse_options;
  parse_options.max_tensor_size_bytes = 8192;
  parse_options.no_copy = (data[2] & 64) != 0;
  parse_options.raw_data_callback = [](TensorProto &loaded, auto *) {
    check_read(loaded);
    return std::function<void()>{};
  };
  bool read_succeeded = false;
  try {
    if (operation == 0) {
      checker::CheckerContext context;
      context.set_ir_version(10);
      context.set_model_dir(base);
      checker::check_tensor(tensor, context);
    } else if (operation == 1 && !needs_bounded_parser(tensor)) {
      tensor.LoadExternalData(base);
      read_succeeded = true;
    } else if (operation == 5 && !needs_bounded_parser(tensor)) {
      model = make_model(tensor);
      ONNX_LIGHT_NAMESPACE::LoadExternalDataForModel(model, base);
      read_succeeded = true;
    } else if (operation == 1 || operation == 3 || operation == 5) {
      SerializeOptions metadata_options;
      metadata_options.use_external_data_location = false;
      model = make_model(tensor);
      if (!model.SerializeToFile(model_path, metadata_options)) {
        throw std::runtime_error("failed to write model metadata");
      }
      utils::TwoFilesStream stream(model_path, (workspace.base() / "data.bin").string());
      model.ParseFromStream(stream, parse_options);
      read_succeeded = true;
    } else {
      tensor.set_raw_data(payload);
      SerializeOptions options;
      options.max_serialized_size_bytes = 128 * 1024;
      options.raw_data_threshold = 0;
      if (operation == 2) {
        utils::TwoFilesWriteStream stream(model_path, weights_path);
        tensor.WriteExternalData(stream, options);
      } else {
        model = make_model(tensor);
        if (operation == 4) {
          std::array<uint8_t, 6> second_header;
          std::copy_n(data, second_header.size(), second_header.begin());
          ++second_header[3];
          auto second = make_tensor(second_header.data(), location, alternate, "second_weight",
                                    payload.size());
          second.set_raw_data(std::string(payload.rbegin(), payload.rend()));
          *model.ref_graph().add_initializer() = second;
        } else {
          auto &embedded = model.ref_graph().ref_initializer()[0];
          embedded.clr_external_data();
          embedded.set_data_location(TensorProto::DataLocation::DEFAULT);
          ONNX_LIGHT_NAMESPACE::ConvertModelToExternalData(
              model, operation == 6 || (data[2] & 1), operation == 6 ? location : "roundtrip.bin",
              0);
        }
        {
          utils::TwoFilesWriteStream stream(model_path, weights_path);
          if (operation == 4) {
            // Preserve the supplied offsets, including overlaps, rather than
            // regenerating metadata through PopulateExternalData.
            const auto size = model.SerializeSize(stream, options);
            if (ONNX_LIGHT_NAMESPACE::EnforceMaxSerializedSize(size, options, "fuzzer")) {
              model.SerializeToStream(stream, options);
              stream.FlushMainToFile();
            }
          } else if (!ONNX_LIGHT_NAMESPACE::SerializeModelProtoToStream(model, stream, options)) {
            throw std::runtime_error("failed to serialize model");
          }
        }
        if (operation == 6 || operation == 7) {
          utils::TwoFilesStream stream(model_path, weights_path);
          ModelProto loaded;
          ParseOptions roundtrip_options;
          roundtrip_options.max_tensor_size_bytes = 8192;
          loaded.ParseFromStream(stream, roundtrip_options);
          const auto &raw = loaded.graph().initializer()[0].raw_data();
          if (raw.size() != payload.size() ||
              std::memcmp(raw.data(), payload.data(), payload.size()) != 0) {
            finding("model roundtrip changed tensor data");
          }
        }
      }
    }
  } catch (const std::runtime_error &) {
    // Invalid metadata, paths, and ranges are expected target errors.
  } catch (const std::invalid_argument &) {
  } catch (const std::out_of_range &) {
  }
  // The callback checks completed tensor reads even if model parsing later fails.
  // Do not inspect failed read buffers, which may contain uninitialized bytes.
  if (read_succeeded) {
    check_read(tensor);
    if (model.has_graph()) {
      for (const auto &loaded : model.graph().initializer()) {
        check_read(loaded);
      }
    }
    if (forbidden) {
      finding("read accepted an unsafe location");
    }
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 6) {
    return 0;
  }
  static Workspace workspace;
  const auto paths = locations(workspace);
  const size_t location = data[1] % paths.size();
  const size_t alternate = (location + data[5] + 1) % paths.size();
  const std::string payload = size > 6 ? std::string(reinterpret_cast<const char *>(data + 6),
                                                     std::min<size_t>(size - 6, 4096))
                                       : std::string(1, '\0');
  // Fixture failures must not be suppressed as target validation errors.
  workspace.reset(payload.size());
  exercise(data, workspace, paths[location], paths[alternate], payload,
           forbidden_location(location) && (!(data[2] & 1) || forbidden_location(alternate)));
  workspace.check_sentinel();
  return 0;
}
