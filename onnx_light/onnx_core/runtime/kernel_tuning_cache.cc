// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning_cache.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr uint32_t kCacheSchemaVersion = 1;

struct CacheProfile {
  KernelTuningParameters parameters;
  CpuExecutionDescriptor execution;
};

struct ParsedCache {
  std::vector<CacheProfile> profiles;
  std::vector<std::string> diagnostics;
  bool malformed = false;
};

template <typename T> bool Contains(const std::vector<T> &values, const T &value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

std::string KeyDescription(const KernelTuningKey &key) {
  return key.library + "/" + key.kernel + "/" + key.implementation;
}

template <typename T> bool ParseInteger(std::string_view text, T &value) {
  const char *begin = text.data();
  const char *end = begin + text.size();
  auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc{} && result.ptr == end;
}

template <typename T> bool ParseOptionalUnsigned(std::string_view text, std::optional<T> &value) {
  if (text == "-") {
    value.reset();
    return true;
  }
  T parsed = 0;
  if (!ParseInteger(text, parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

bool ReadQuoted(std::istringstream &stream, std::string &value) {
  stream >> std::quoted(value);
  return static_cast<bool>(stream);
}

bool AtEnd(std::istringstream &stream) {
  stream >> std::ws;
  return stream.eof();
}

void AddMalformed(ParsedCache &cache, size_t line_number, std::string message) {
  cache.malformed = true;
  cache.diagnostics.push_back("Line " + std::to_string(line_number) + ": " + std::move(message));
}

bool ParseProfileField(CacheProfile &profile, std::string_view field, std::istringstream &stream) {
  std::string text;
  if (field == "library") {
    return ReadQuoted(stream, profile.parameters.key.library) && AtEnd(stream);
  }
  if (field == "kernel") {
    return ReadQuoted(stream, profile.parameters.key.kernel) && AtEnd(stream);
  }
  if (field == "implementation") {
    return ReadQuoted(stream, profile.parameters.key.implementation) && AtEnd(stream);
  }
  if (field == "architecture") {
    return ReadQuoted(stream, profile.execution.processor.architecture) && AtEnd(stream);
  }
  if (field == "vendor") {
    return ReadQuoted(stream, profile.execution.processor.vendor) && AtEnd(stream);
  }
  if (field == "microarchitecture") {
    return ReadQuoted(stream, profile.execution.processor.microarchitecture) && AtEnd(stream);
  }
  if (!(stream >> text)) {
    return false;
  }
  if (field == "element_type") {
    return ParseInteger(text, profile.parameters.key.element_type) && AtEnd(stream);
  }
  if (field == "device") {
    int32_t device = 0;
    if (!ParseInteger(text, device) || !AtEnd(stream)) {
      return false;
    }
    profile.parameters.key.device = static_cast<Device>(device);
    return true;
  }
  if (field == "tuning_abi") {
    return ParseInteger(text, profile.parameters.key.tuning_abi) && AtEnd(stream);
  }
  if (field == "family") {
    return ParseOptionalUnsigned(text, profile.execution.processor.family) && AtEnd(stream);
  }
  if (field == "model") {
    return ParseOptionalUnsigned(text, profile.execution.processor.model) && AtEnd(stream);
  }
  if (field == "stepping") {
    return ParseOptionalUnsigned(text, profile.execution.processor.stepping) && AtEnd(stream);
  }
  if (field == "features") {
    uint64_t bits = 0;
    if (!ParseInteger(text, bits) || !AtEnd(stream)) {
      return false;
    }
    profile.execution.processor.features = platform::CpuFeatureSet(bits);
    return true;
  }
  if (field == "cache_line_bytes") {
    return ParseOptionalUnsigned(text, profile.execution.processor.cache_line_bytes) &&
           AtEnd(stream);
  }
  if (field == "l1_data_bytes") {
    return ParseOptionalUnsigned(text, profile.execution.processor.l1_data_bytes) && AtEnd(stream);
  }
  if (field == "l2_bytes") {
    return ParseOptionalUnsigned(text, profile.execution.processor.l2_bytes) && AtEnd(stream);
  }
  if (field == "l3_bytes") {
    return ParseOptionalUnsigned(text, profile.execution.processor.l3_bytes) && AtEnd(stream);
  }
  if (field == "physical_cores") {
    return ParseOptionalUnsigned(text, profile.execution.processor.physical_cores) && AtEnd(stream);
  }
  if (field == "logical_cores") {
    return ParseOptionalUnsigned(text, profile.execution.processor.logical_cores) && AtEnd(stream);
  }
  if (field == "effective_threads") {
    return ParseInteger(text, profile.execution.effective_threads) && AtEnd(stream);
  }
  return false;
}

bool ParseValue(CacheProfile &profile, std::istringstream &stream) {
  std::string name;
  std::string type;
  if (!ReadQuoted(stream, name) || !(stream >> type) || profile.parameters.values.contains(name)) {
    return false;
  }
  if (type == "int64") {
    std::string text;
    int64_t value = 0;
    if (!(stream >> text) || !ParseInteger(text, value) || !AtEnd(stream)) {
      return false;
    }
    profile.parameters.values.emplace(std::move(name), value);
    return true;
  }
  if (type == "double") {
    double value = 0;
    if (!(stream >> value) || !AtEnd(stream) || !std::isfinite(value)) {
      return false;
    }
    profile.parameters.values.emplace(std::move(name), value);
    return true;
  }
  if (type == "bool") {
    std::string text;
    if (!(stream >> text) || !AtEnd(stream) || (text != "true" && text != "false")) {
      return false;
    }
    profile.parameters.values.emplace(std::move(name), text == "true");
    return true;
  }
  if (type == "string") {
    std::string value;
    if (!ReadQuoted(stream, value) || !AtEnd(stream)) {
      return false;
    }
    profile.parameters.values.emplace(std::move(name), std::move(value));
    return true;
  }
  return false;
}

ParsedCache ParseCache(std::istream &input) {
  ParsedCache cache;
  std::string line;
  size_t line_number = 0;
  if (!std::getline(input, line)) {
    AddMalformed(cache, 1, "missing cache header.");
    return cache;
  }
  ++line_number;
  std::istringstream header(line);
  std::string magic;
  uint32_t version = 0;
  if (!(header >> magic >> version) || !AtEnd(header) ||
      magic != "onnx_light_kernel_tuning_cache" || version != kCacheSchemaVersion) {
    AddMalformed(cache, line_number, "unsupported or malformed cache header.");
    return cache;
  }

  std::optional<CacheProfile> current;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.starts_with("#")) {
      continue;
    }
    std::istringstream stream(line);
    std::string field;
    stream >> field;
    if (field == "profile" && !current.has_value() && AtEnd(stream)) {
      current.emplace();
      continue;
    }
    if (field == "end" && current.has_value() && AtEnd(stream)) {
      if (current->execution.effective_threads == 0 || current->parameters.values.empty() ||
          current->parameters.key.library.empty() || current->parameters.key.kernel.empty() ||
          current->parameters.key.implementation.empty() ||
          current->execution.processor.architecture.empty()) {
        AddMalformed(cache, line_number, "incomplete profile.");
      } else if (std::any_of(cache.profiles.begin(), cache.profiles.end(),
                             [&](const CacheProfile &profile) {
                               return profile.parameters.key == current->parameters.key &&
                                      profile.execution == current->execution;
                             })) {
        AddMalformed(cache, line_number,
                     "duplicate profile key and execution descriptor for '" +
                         KeyDescription(current->parameters.key) + "'.");
      } else {
        cache.profiles.push_back(std::move(*current));
      }
      current.reset();
      continue;
    }
    if (!current.has_value()) {
      AddMalformed(cache, line_number, "expected 'profile'.");
      continue;
    }
    bool valid = field == "value" ? ParseValue(*current, stream)
                                  : ParseProfileField(*current, field, stream);
    if (!valid) {
      AddMalformed(cache, line_number, "invalid or unknown field '" + field + "'.");
    }
  }
  if (current.has_value()) {
    AddMalformed(cache, line_number + 1, "unterminated profile.");
  }
  return cache;
}

CpuExecutionDescriptor CurrentExecutionDescriptor() {
  CpuExecutionDescriptor execution;
  execution.processor = platform::GetCpuDescriptor();
  execution.effective_threads = execution.processor.logical_cores.value_or(1);
  return execution;
}

bool SameIdentityIgnoringAbi(const KernelTuningKey &left, const KernelTuningKey &right) {
  return left.library == right.library && left.kernel == right.kernel &&
         left.implementation == right.implementation && left.element_type == right.element_type &&
         left.device == right.device;
}

} // namespace

bool KernelCalibrationSelection::Matches(const KernelTuningKey &key) const {
  return (!library.has_value() || key.library == *library) && Contains(kernels, key.kernel) &&
         Contains(implementations, key.implementation) &&
         Contains(element_types, key.element_type) &&
         (!device.has_value() || key.device == *device);
}

std::filesystem::path DefaultKernelTuningCachePath() {
#if defined(_WIN32)
  if (const char *directory = std::getenv("LOCALAPPDATA")) {
    return std::filesystem::path(directory) / "onnx-light" / "kernel_tuning.cache";
  }
#else
  if (const char *directory = std::getenv("XDG_CACHE_HOME")) {
    return std::filesystem::path(directory) / "onnx-light" / "kernel_tuning.cache";
  }
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / ".cache" / "onnx-light" / "kernel_tuning.cache";
  }
#endif
  return std::filesystem::path("onnx-light-kernel-tuning.cache");
}

KernelTuningCacheLoadReport LoadKernelTuningCache(const KernelCalibrationSelection &selection,
                                                  const KernelTuningCacheOptions &options) {
  KernelTuningCacheLoadReport report;
  const std::filesystem::path path =
      options.path.empty() ? DefaultKernelTuningCachePath() : options.path;
  std::error_code error;
  bool exists = std::filesystem::exists(path, error);
  if (!exists) {
    report.status =
        error ? KernelTuningCacheLoadStatus::kUnreadable : KernelTuningCacheLoadStatus::kNotFound;
    if (error) {
      report.diagnostics.push_back("Unable to inspect cache '" + path.string() +
                                   "': " + error.message());
    }
    return report;
  }

  std::ifstream input(path);
  if (!input) {
    report.status = KernelTuningCacheLoadStatus::kUnreadable;
    report.diagnostics.push_back("Unable to read cache '" + path.string() + "'.");
    return report;
  }
  ParsedCache cache = ParseCache(input);
  report.diagnostics = std::move(cache.diagnostics);
  if (cache.malformed) {
    report.status = KernelTuningCacheLoadStatus::kMalformed;
    return report;
  }

  KernelTuningRegistry &registry = GetKernelTuningRegistry();
  std::vector<KernelTuningKey> registered = registry.RegisteredKeys();
  std::vector<KernelTuningKey> selected;
  for (const KernelTuningKey &key : registered) {
    if (selection.Matches(key)) {
      selected.push_back(key);
    }
  }

  CpuExecutionDescriptor execution = options.execution.value_or(CurrentExecutionDescriptor());
  std::vector<KernelTuningParameters> valid;
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> loaded;
  for (const CacheProfile &profile : cache.profiles) {
    if (!selection.Matches(profile.parameters.key)) {
      continue;
    }
    auto schema = registry.FindSchema(profile.parameters.key);
    if (schema == nullptr) {
      bool stale = std::any_of(selected.begin(), selected.end(), [&](const KernelTuningKey &key) {
        return SameIdentityIgnoringAbi(key, profile.parameters.key);
      });
      (stale ? report.stale : report.incompatible).push_back(profile.parameters.key);
      continue;
    }
    if (profile.execution != execution) {
      report.incompatible.push_back(profile.parameters.key);
      continue;
    }
    try {
      schema->Validate(profile.parameters);
    } catch (const std::invalid_argument &exception) {
      report.invalid.push_back(profile.parameters.key);
      report.diagnostics.push_back(KeyDescription(profile.parameters.key) + ": " +
                                   exception.what());
      continue;
    }
    loaded.emplace(profile.parameters.key);
    report.loaded.push_back(profile.parameters.key);
    valid.push_back(profile.parameters);
  }
  for (const KernelTuningKey &key : selected) {
    if (!loaded.contains(key)) {
      report.missing.push_back(key);
    }
  }

  registry.PublishProfiles(valid, selected);
  report.status = KernelTuningCacheLoadStatus::kLoaded;
  report.published_generation = registry.Snapshot().generation();
  return report;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
