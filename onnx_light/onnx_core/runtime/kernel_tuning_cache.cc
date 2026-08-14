// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning_cache.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <unordered_set>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr uint32_t kCacheSchemaVersion = 1;

using CacheProfile = CalibratedKernelProfile;

struct ParsedCache {
  std::vector<CacheProfile> profiles;
  std::vector<std::string> diagnostics;
  bool malformed = false;
};

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

std::filesystem::path CachePath(const KernelTuningCacheOptions &options) {
  return options.path.empty() ? DefaultKernelTuningCachePath() : options.path;
}

void WriteOptionalUnsigned(std::ostream &output, const auto &value) {
  if (value.has_value()) {
    output << *value;
  } else {
    output << '-';
  }
}

void WriteExecution(std::ostream &output, const CpuExecutionDescriptor &execution) {
  const platform::CpuDescriptor &processor = execution.processor;
  output << "architecture " << std::quoted(processor.architecture) << '\n'
         << "vendor " << std::quoted(processor.vendor) << '\n'
         << "family ";
  WriteOptionalUnsigned(output, processor.family);
  output << "\nmodel ";
  WriteOptionalUnsigned(output, processor.model);
  output << "\nstepping ";
  WriteOptionalUnsigned(output, processor.stepping);
  output << "\nmicroarchitecture " << std::quoted(processor.microarchitecture) << '\n'
         << "features " << processor.features.bits() << "\ncache_line_bytes ";
  WriteOptionalUnsigned(output, processor.cache_line_bytes);
  output << "\nl1_data_bytes ";
  WriteOptionalUnsigned(output, processor.l1_data_bytes);
  output << "\nl2_bytes ";
  WriteOptionalUnsigned(output, processor.l2_bytes);
  output << "\nl3_bytes ";
  WriteOptionalUnsigned(output, processor.l3_bytes);
  output << "\nphysical_cores ";
  WriteOptionalUnsigned(output, processor.physical_cores);
  output << "\nlogical_cores ";
  WriteOptionalUnsigned(output, processor.logical_cores);
  output << "\neffective_threads " << execution.effective_threads << '\n';
}

void WriteProfile(std::ostream &output, const CacheProfile &profile) {
  const KernelTuningKey &key = profile.parameters.key;
  output << "profile\n"
         << "library " << std::quoted(key.library) << '\n'
         << "kernel " << std::quoted(key.kernel) << '\n'
         << "implementation " << std::quoted(key.implementation) << '\n'
         << "element_type " << key.element_type << "\ndevice " << static_cast<int32_t>(key.device)
         << "\ntuning_abi " << key.tuning_abi << '\n';
  WriteExecution(output, profile.execution);
  std::vector<std::string> names;
  names.reserve(profile.parameters.values.size());
  for (const auto &[name, value] : profile.parameters.values) {
    names.push_back(name);
    (void)value;
  }
  std::sort(names.begin(), names.end());
  for (const std::string &name : names) {
    const TuningValue &value = profile.parameters.values.at(name);
    output << "value " << std::quoted(name) << ' ' << TuningValueTypeName(value) << ' ';
    std::visit(
        [&output](const auto &typed_value) {
          using T = std::decay_t<decltype(typed_value)>;
          if constexpr (std::is_same_v<T, bool>) {
            output << (typed_value ? "true" : "false");
          } else if constexpr (std::is_same_v<T, std::string>) {
            output << std::quoted(typed_value);
          } else {
            output << typed_value;
          }
        },
        value);
    output << '\n';
  }
  output << "end\n";
}

class InterprocessCacheLock {
public:
  explicit InterprocessCacheLock(const std::filesystem::path &path) {
#if defined(_WIN32)
    handle_ = ::CreateFileW(path.native().c_str(), GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                              "Unable to open kernel tuning cache lock");
    }
    if (!::LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped_)) {
      const int error = static_cast<int>(::GetLastError());
      ::CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
      throw std::system_error(error, std::system_category(),
                              "Unable to acquire kernel tuning cache lock");
    }
#else
    descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (descriptor_ < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "Unable to open kernel tuning cache lock");
    }
    if (::flock(descriptor_, LOCK_EX) != 0) {
      const int error = errno;
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::system_error(error, std::generic_category(),
                              "Unable to acquire kernel tuning cache lock");
    }
#endif
  }

  ~InterprocessCacheLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) {
      ::UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &overlapped_);
      ::CloseHandle(handle_);
    }
#else
    if (descriptor_ >= 0) {
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
#endif
  }

  InterprocessCacheLock(const InterprocessCacheLock &) = delete;
  InterprocessCacheLock &operator=(const InterprocessCacheLock &) = delete;

private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  OVERLAPPED overlapped_{};
#else
  int descriptor_ = -1;
#endif
};

bool AtomicWriteCache(const std::filesystem::path &path, std::span<const CacheProfile> profiles,
                      std::vector<std::string> &diagnostics) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
      diagnostics.push_back("Unable to write temporary cache '" + temporary.string() + "'.");
      return false;
    }
    output << "onnx_light_kernel_tuning_cache " << kCacheSchemaVersion << '\n';
    for (const CacheProfile &profile : profiles) {
      WriteProfile(output, profile);
    }
    output.flush();
    if (!output) {
      diagnostics.push_back("Unable to flush temporary cache '" + temporary.string() + "'.");
      return false;
    }
  }

#if defined(_WIN32)
  HANDLE temporary_handle =
      ::CreateFileW(temporary.native().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (temporary_handle == INVALID_HANDLE_VALUE || !::FlushFileBuffers(temporary_handle)) {
    const int error = static_cast<int>(::GetLastError());
    if (temporary_handle != INVALID_HANDLE_VALUE) {
      ::CloseHandle(temporary_handle);
    }
    diagnostics.push_back("Unable to synchronize temporary cache: " +
                          std::system_category().message(error));
    return false;
  }
  ::CloseHandle(temporary_handle);
  if (!::MoveFileExW(temporary.native().c_str(), path.native().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    diagnostics.push_back("Unable to replace cache '" + path.string() + "': " +
                          std::system_category().message(static_cast<int>(::GetLastError())));
    return false;
  }
#else
  const int descriptor = ::open(temporary.c_str(), O_RDONLY);
  if (descriptor < 0 || ::fsync(descriptor) != 0) {
    const int error = errno;
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    diagnostics.push_back("Unable to synchronize temporary cache: " +
                          std::generic_category().message(error));
    return false;
  }
  ::close(descriptor);
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    diagnostics.push_back("Unable to replace cache '" + path.string() + "': " + error.message());
    return false;
  }
  const std::filesystem::path directory = path.parent_path().empty() ? "." : path.parent_path();
  const int directory_descriptor = ::open(directory.c_str(), O_RDONLY);
  if (directory_descriptor >= 0) {
    (void)::fsync(directory_descriptor);
    ::close(directory_descriptor);
  }
#endif
  return true;
}

} // namespace

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

KernelTuningCacheUpdateReport
UpdateKernelTuningCache(std::span<const CalibratedKernelProfile> profiles,
                        const KernelTuningCacheOptions &options) {
  KernelTuningCacheUpdateReport report;
  KernelTuningRegistry &registry = GetKernelTuningRegistry();
  for (size_t index = 0; index < profiles.size(); ++index) {
    const CalibratedKernelProfile &profile = profiles[index];
    if (profile.execution.effective_threads == 0) {
      throw std::invalid_argument("Cached calibration effective_threads must be positive.");
    }
    std::shared_ptr<const KernelTuningSchema> schema = registry.FindSchema(profile.parameters.key);
    if (schema == nullptr) {
      throw std::invalid_argument("Cannot cache unregistered kernel tuning key '" +
                                  KeyDescription(profile.parameters.key) + "'.");
    }
    schema->Validate(profile.parameters);
    for (size_t previous = 0; previous < index; ++previous) {
      if (profiles[previous].parameters.key == profile.parameters.key &&
          profiles[previous].execution == profile.execution) {
        throw std::invalid_argument("Duplicate kernel tuning cache update profile for '" +
                                    KeyDescription(profile.parameters.key) + "'.");
      }
    }
  }
  if (options.read_only) {
    report.status = KernelTuningCacheUpdateStatus::kReadOnly;
    report.diagnostics.push_back("Kernel tuning cache is read-only; no profiles were written.");
    return report;
  }

  const std::filesystem::path path = CachePath(options);
  const std::filesystem::path directory = path.parent_path();
  std::error_code error;
  if (!directory.empty()) {
    std::filesystem::create_directories(directory, error);
    if (error) {
      report.status = KernelTuningCacheUpdateStatus::kWriteFailed;
      report.diagnostics.push_back("Unable to create cache directory '" + directory.string() +
                                   "': " + error.message());
      return report;
    }
  }

  try {
    std::filesystem::path lock_path = path;
    lock_path += ".lock";
    InterprocessCacheLock lock(lock_path);
    ParsedCache cache;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
      report.status = KernelTuningCacheUpdateStatus::kUnreadable;
      report.diagnostics.push_back("Unable to inspect cache '" + path.string() +
                                   "': " + error.message());
      return report;
    }
    if (exists) {
      std::ifstream input(path);
      if (!input) {
        report.status = KernelTuningCacheUpdateStatus::kUnreadable;
        report.diagnostics.push_back("Unable to read cache '" + path.string() + "'.");
        return report;
      }
      cache = ParseCache(input);
      report.diagnostics = std::move(cache.diagnostics);
      if (cache.malformed) {
        report.status = KernelTuningCacheUpdateStatus::kMalformed;
        return report;
      }
    }

    const std::vector<KernelTuningKey> registered = registry.RegisteredKeys();
    if (options.prune_stale_abis) {
      std::erase_if(cache.profiles, [&](const CacheProfile &profile) {
        const bool stale =
            std::any_of(registered.begin(), registered.end(), [&](const KernelTuningKey &key) {
              return SameIdentityIgnoringAbi(key, profile.parameters.key) &&
                     key.tuning_abi != profile.parameters.key.tuning_abi;
            });
        if (stale) {
          report.pruned.push_back(profile.parameters.key);
        }
        return stale;
      });
    }

    for (const CalibratedKernelProfile &profile : profiles) {
      auto existing = std::find_if(cache.profiles.begin(), cache.profiles.end(),
                                   [&](const CacheProfile &cached) {
                                     return cached.parameters.key == profile.parameters.key &&
                                            cached.execution == profile.execution;
                                   });
      if (existing != cache.profiles.end() && !options.replace_existing) {
        report.preserved.push_back(profile.parameters.key);
        continue;
      }
      if (existing == cache.profiles.end()) {
        cache.profiles.push_back(profile);
      } else {
        *existing = profile;
      }
      report.updated.push_back(profile.parameters.key);
    }

    if (!AtomicWriteCache(path, cache.profiles, report.diagnostics)) {
      report.status = KernelTuningCacheUpdateStatus::kWriteFailed;
      std::filesystem::path temporary = path;
      temporary += ".tmp";
      std::filesystem::remove(temporary, error);
      return report;
    }
    report.status = KernelTuningCacheUpdateStatus::kUpdated;
    return report;
  } catch (const std::system_error &exception) {
    report.status = KernelTuningCacheUpdateStatus::kUnreadable;
    report.diagnostics.push_back(exception.what());
    return report;
  }
}

KernelTuningCacheUpdateReport
UpdateKernelTuningCache(std::span<const KernelTuningParameters> profiles,
                        const KernelTuningCacheOptions &options) {
  const CpuExecutionDescriptor execution = options.execution.value_or(CurrentExecutionDescriptor());
  std::vector<CalibratedKernelProfile> calibrated;
  calibrated.reserve(profiles.size());
  for (const KernelTuningParameters &profile : profiles) {
    calibrated.push_back({profile, execution});
  }
  return UpdateKernelTuningCache(calibrated, options);
}

KernelTuningCacheLoadReport LoadKernelTuningCache(const KernelCalibrationSelection &selection,
                                                  const KernelTuningCacheOptions &options) {
  KernelTuningCacheLoadReport report;
  const std::filesystem::path path = CachePath(options);
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
    if (selection.Matches(key) && (selection.device.has_value() || key.device == Device::kCPU)) {
      selected.push_back(key);
    }
  }

  CpuExecutionDescriptor execution = options.execution.value_or(CurrentExecutionDescriptor());
  std::vector<KernelTuningParameters> valid;
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> loaded;
  for (const CacheProfile &profile : cache.profiles) {
    if (!selection.Matches(profile.parameters.key) ||
        (!selection.device.has_value() && profile.parameters.key.device != Device::kCPU)) {
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

  registry.PublishCalibratedProfiles(valid, execution, selected);
  report.status = KernelTuningCacheLoadStatus::kLoaded;
  report.published_generation = registry.Snapshot().generation();
  return report;
}

KernelTuningDeploymentImportReport
ImportKernelTuningDeploymentProfiles(const KernelCalibrationSelection &selection,
                                     const KernelTuningDeploymentImportOptions &options) {
  KernelTuningDeploymentImportReport report;
  const std::filesystem::path path =
      options.path.empty() ? DefaultKernelTuningCachePath() : options.path;
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (!exists) {
    report.status =
        error ? KernelTuningCacheLoadStatus::kUnreadable : KernelTuningCacheLoadStatus::kNotFound;
    if (error) {
      report.diagnostics.push_back("Unable to inspect deployment profile '" + path.string() +
                                   "': " + error.message());
    }
    return report;
  }
  std::ifstream input(path);
  if (!input) {
    report.status = KernelTuningCacheLoadStatus::kUnreadable;
    report.diagnostics.push_back("Unable to read deployment profile '" + path.string() + "'.");
    return report;
  }
  ParsedCache cache = ParseCache(input);
  report.diagnostics = std::move(cache.diagnostics);
  if (cache.malformed) {
    report.status = KernelTuningCacheLoadStatus::kMalformed;
    return report;
  }

  KernelTuningRegistry &registry = GetKernelTuningRegistry();
  const std::vector<KernelTuningKey> registered = registry.RegisteredKeys();
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> seen;
  std::vector<ProcessorKernelTuningProfile> imported;
  for (const CacheProfile &profile : cache.profiles) {
    const KernelTuningKey &key = profile.parameters.key;
    if (!selection.Matches(key) || (!selection.device.has_value() && key.device != Device::kCPU)) {
      continue;
    }
    std::shared_ptr<const KernelTuningSchema> schema = registry.FindSchema(key);
    if (schema == nullptr) {
      const bool stale =
          std::any_of(registered.begin(), registered.end(), [&](const KernelTuningKey &candidate) {
            return SameIdentityIgnoringAbi(candidate, key);
          });
      (stale ? report.stale : report.incompatible).push_back(key);
      continue;
    }
    if (!seen.emplace(key).second) {
      report.invalid.push_back(key);
      report.diagnostics.push_back(KeyDescription(key) +
                                   ": deployment import contains multiple execution profiles.");
      continue;
    }
    try {
      schema->Validate(profile.parameters);
    } catch (const std::invalid_argument &exception) {
      report.invalid.push_back(key);
      report.diagnostics.push_back(KeyDescription(key) + ": " + exception.what());
      continue;
    }
    imported.push_back({profile.parameters, options.processors, options.priority});
  }
  if (!report.invalid.empty()) {
    report.status = KernelTuningCacheLoadStatus::kLoaded;
    report.published_generation = registry.Snapshot().generation();
    return report;
  }

  registry.RegisterProfiles(imported);
  report.imported.reserve(imported.size());
  for (const ProcessorKernelTuningProfile &profile : imported) {
    report.imported.push_back(profile.parameters.key);
  }
  report.status = KernelTuningCacheLoadStatus::kLoaded;
  report.published_generation = registry.Snapshot().generation();
  return report;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
