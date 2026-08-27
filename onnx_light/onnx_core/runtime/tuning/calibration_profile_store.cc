// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/calibration_profile_store.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr uint32_t kSchemaVersion = 1;
constexpr std::string_view kMagic = "onnx_light_calibration_profiles";

struct ParsedProfiles {
  CalibrationProfileStoreStatus status = CalibrationProfileStoreStatus::kOk;
  std::vector<CalibrationProfile> profiles;
  std::vector<std::string> diagnostics;
};

bool AtEnd(std::istringstream &stream) {
  stream >> std::ws;
  return stream.eof();
}

bool ReadQuoted(std::istringstream &stream, std::string &value) {
  stream >> std::quoted(value);
  return static_cast<bool>(stream);
}

std::string_view TrimLeadingWhitespace(std::string_view line) {
  size_t start = line.find_first_not_of(" \t\r\n");
  return start == std::string_view::npos ? std::string_view() : line.substr(start);
}

void Malformed(ParsedProfiles &parsed, size_t line, std::string message) {
  parsed.status = CalibrationProfileStoreStatus::kMalformed;
  parsed.diagnostics.push_back("Line " + std::to_string(line) + ": " + std::move(message));
}

bool CompleteKey(const CalibrationProfileKey &key) {
  if (key.backend.empty() || key.operator_name.empty() || key.implementation_version.empty()) {
    return false;
  }
  if (key.kind == CalibrationProfileKind::kExact) {
    return !key.model_digest.empty() && !key.processor.empty() && key.thread_count != 0 &&
           key.structural_properties.empty();
  }
  return key.model_digest.empty() && key.processor.empty() && key.thread_count == 0 &&
         !key.structural_properties.empty();
}

bool ParseProfileField(CalibrationProfile &profile, std::string_view field,
                       std::istringstream &stream) {
  if (field == "backend") {
    return ReadQuoted(stream, profile.key.backend) && AtEnd(stream);
  }
  if (field == "operator") {
    return ReadQuoted(stream, profile.key.operator_name) && AtEnd(stream);
  }
  if (field == "implementation") {
    return ReadQuoted(stream, profile.key.implementation_version) && AtEnd(stream);
  }
  if (field == "model_digest") {
    return ReadQuoted(stream, profile.key.model_digest) && AtEnd(stream);
  }
  if (field == "processor") {
    return ReadQuoted(stream, profile.key.processor) && AtEnd(stream);
  }
  if (field == "thread_count") {
    return (stream >> profile.key.thread_count) && AtEnd(stream);
  }
  if (field == "structure") {
    std::string name;
    std::string value;
    return ReadQuoted(stream, name) && ReadQuoted(stream, value) && AtEnd(stream) &&
           !name.empty() &&
           profile.key.structural_properties.emplace(std::move(name), std::move(value)).second;
  }
  if (field == "measurement") {
    CalibrationMeasurement measurement;
    if (!ReadQuoted(stream, measurement.name) || !(stream >> measurement.value) ||
        !ReadQuoted(stream, measurement.unit) || !AtEnd(stream) || measurement.name.empty() ||
        !std::isfinite(measurement.value)) {
      return false;
    }
    profile.measurements.push_back(std::move(measurement));
    return true;
  }
  if (field == "policy") {
    return ReadQuoted(stream, profile.policy) && AtEnd(stream);
  }
  return false;
}

ParsedProfiles Parse(std::istream &input) {
  ParsedProfiles parsed;
  std::string line;
  if (!std::getline(input, line)) {
    Malformed(parsed, 1, "missing profile-store header.");
    return parsed;
  }
  std::istringstream header(line);
  std::string magic;
  uint32_t version = 0;
  if (!(header >> magic >> version) || !AtEnd(header) || magic != kMagic) {
    Malformed(parsed, 1, "malformed profile-store header.");
    return parsed;
  }
  if (version != kSchemaVersion) {
    parsed.status = CalibrationProfileStoreStatus::kUnsupportedVersion;
    parsed.diagnostics.push_back("Unsupported calibration profile schema version " +
                                 std::to_string(version) + ".");
    return parsed;
  }

  std::optional<CalibrationProfile> current;
  size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    std::string_view trimmed = TrimLeadingWhitespace(line);
    if (trimmed.empty() || trimmed.starts_with("#")) {
      continue;
    }
    std::istringstream stream(line);
    std::string field;
    stream >> field;
    if (field == "profile" && !current.has_value()) {
      std::string kind;
      std::string source;
      if (!(stream >> kind >> source) || !AtEnd(stream) ||
          (kind != "exact" && kind != "portable") ||
          (source != "calibrated" && source != "override")) {
        Malformed(parsed, line_number, "invalid profile declaration.");
        continue;
      }
      current.emplace();
      current->key.kind =
          kind == "exact" ? CalibrationProfileKind::kExact : CalibrationProfileKind::kPortable;
      current->user_override = source == "override";
      continue;
    }
    if (field == "end" && current.has_value() && AtEnd(stream)) {
      if (!CompleteKey(current->key) || current->policy.empty()) {
        Malformed(parsed, line_number, "incomplete profile.");
      } else if (std::any_of(parsed.profiles.begin(), parsed.profiles.end(),
                             [&](const CalibrationProfile &profile) {
                               return profile.key == current->key &&
                                      profile.user_override == current->user_override;
                             })) {
        Malformed(parsed, line_number, "duplicate profile identity.");
      } else {
        parsed.profiles.push_back(std::move(*current));
      }
      current.reset();
      continue;
    }
    if (!current.has_value()) {
      Malformed(parsed, line_number, "expected 'profile'.");
    } else if (!ParseProfileField(*current, field, stream)) {
      Malformed(parsed, line_number, "invalid or unknown field '" + field + "'.");
    }
  }
  if (current.has_value()) {
    Malformed(parsed, line_number + 1, "unterminated profile.");
  }
  return parsed;
}

void WriteProfile(std::ostream &output, const CalibrationProfile &profile) {
  output << "profile "
         << (profile.key.kind == CalibrationProfileKind::kExact ? "exact" : "portable") << ' '
         << (profile.user_override ? "override" : "calibrated") << '\n'
         << "backend " << std::quoted(profile.key.backend) << '\n'
         << "operator " << std::quoted(profile.key.operator_name) << '\n'
         << "implementation " << std::quoted(profile.key.implementation_version) << '\n';
  if (profile.key.kind == CalibrationProfileKind::kExact) {
    output << "model_digest " << std::quoted(profile.key.model_digest) << '\n'
           << "processor " << std::quoted(profile.key.processor) << '\n'
           << "thread_count " << profile.key.thread_count << '\n';
  } else {
    for (const auto &[name, value] : profile.key.structural_properties) {
      output << "structure " << std::quoted(name) << ' ' << std::quoted(value) << '\n';
    }
  }
  for (const CalibrationMeasurement &measurement : profile.measurements) {
    output << "measurement " << std::quoted(measurement.name) << ' ' << std::setprecision(17)
           << measurement.value << ' ' << std::quoted(measurement.unit) << '\n';
  }
  output << "policy " << std::quoted(profile.policy) << "\nend\n";
}

class InterprocessLock {
public:
  InterprocessLock() = default;

  bool Acquire(const std::filesystem::path &path, std::string &message) {
#if defined(_WIN32)
    handle_ = ::CreateFileW(path.native().c_str(), GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      message = "Unable to open calibration profile lock: " +
                std::system_category().message(static_cast<int>(::GetLastError()));
      return false;
    }
    if (!::LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped_)) {
      const int error = static_cast<int>(::GetLastError());
      ::CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
      message =
          "Unable to acquire calibration profile lock: " + std::system_category().message(error);
      return false;
    }
#else
    descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (descriptor_ < 0) {
      message =
          "Unable to open calibration profile lock: " + std::generic_category().message(errno);
      return false;
    }
    if (::flock(descriptor_, LOCK_EX) != 0) {
      const int error = errno;
      ::close(descriptor_);
      descriptor_ = -1;
      message =
          "Unable to acquire calibration profile lock: " + std::generic_category().message(error);
      return false;
    }
#endif
    return true;
  }

  ~InterprocessLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) {
      ::UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &overlapped_);
      ::CloseHandle(handle_);
    }
#else
    if (descriptor_ >= 0) {
      (void)::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
#endif
  }

  InterprocessLock(const InterprocessLock &) = delete;
  InterprocessLock &operator=(const InterprocessLock &) = delete;

private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  OVERLAPPED overlapped_{};
#else
  int descriptor_ = -1;
#endif
};

bool AtomicWrite(const std::filesystem::path &path, std::span<const CalibrationProfile> profiles,
                 std::vector<std::string> &diagnostics) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
      diagnostics.push_back("Unable to write temporary profile store '" + temporary.string() +
                            "'.");
      return false;
    }
    output << kMagic << ' ' << kSchemaVersion << '\n';
    for (const CalibrationProfile &profile : profiles) {
      WriteProfile(output, profile);
    }
    output.flush();
    if (!output) {
      diagnostics.push_back("Unable to flush temporary profile store '" + temporary.string() +
                            "'.");
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
    diagnostics.push_back("Unable to synchronize temporary profile store: " +
                          std::system_category().message(error));
    return false;
  }
  ::CloseHandle(temporary_handle);
  if (!::MoveFileExW(temporary.native().c_str(), path.native().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    diagnostics.push_back("Unable to replace profile store '" + path.string() + "': " +
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
    diagnostics.push_back("Unable to synchronize temporary profile store: " +
                          std::generic_category().message(error));
    return false;
  }
  ::close(descriptor);
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    diagnostics.push_back("Unable to replace profile store '" + path.string() +
                          "': " + error.message());
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

bool BaseIdentityMatches(const CalibrationProfileKey &left, const CalibrationProfileKey &right) {
  return left.backend == right.backend && left.operator_name == right.operator_name &&
         left.implementation_version == right.implementation_version;
}

bool MatchesExact(const CalibrationProfileKey &candidate, const CalibrationProfileKey &requested) {
  return candidate.kind == CalibrationProfileKind::kExact &&
         BaseIdentityMatches(candidate, requested) &&
         candidate.model_digest == requested.model_digest &&
         candidate.processor == requested.processor &&
         candidate.thread_count == requested.thread_count;
}

bool MatchesPortable(const CalibrationProfileKey &candidate,
                     const CalibrationProfileLookupOptions &requested) {
  return candidate.kind == CalibrationProfileKind::kPortable &&
         BaseIdentityMatches(candidate, requested.exact_key) &&
         candidate.structural_properties == requested.structural_properties;
}

CalibrationProfileRejection ClassifyRejection(const CalibrationProfile &profile,
                                              const CalibrationProfileLookupOptions &requested) {
  CalibrationProfileRejection rejection{
      profile.key, CalibrationProfileRejectionReason::kDifferentStructure, {}};
  if (profile.key.backend != requested.exact_key.backend) {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentBackend;
    rejection.message = "backend does not match";
  } else if (profile.key.operator_name != requested.exact_key.operator_name) {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentOperator;
    rejection.message = "operator does not match";
  } else if (profile.key.implementation_version != requested.exact_key.implementation_version) {
    rejection.reason = CalibrationProfileRejectionReason::kOutdatedImplementation;
    rejection.message = "implementation version is incompatible or outdated";
  } else if (profile.key.kind == CalibrationProfileKind::kPortable) {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentStructure;
    rejection.message = "structural model properties do not match";
  } else if (profile.key.model_digest != requested.exact_key.model_digest) {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentModel;
    rejection.message = "model digest does not match";
  } else if (profile.key.processor != requested.exact_key.processor) {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentProcessor;
    rejection.message = "processor does not match";
  } else {
    rejection.reason = CalibrationProfileRejectionReason::kDifferentThreadCount;
    rejection.message = "thread count does not match";
  }
  return rejection;
}

void ValidateKey(const CalibrationProfileKey &key) {
  if (!CompleteKey(key)) {
    throw std::invalid_argument(
        "Calibration profile key is incomplete or mixes exact and portable fields.");
  }
}

} // namespace

CalibrationProfileStore::CalibrationProfileStore(CalibrationProfileStoreOptions options)
    : options_(std::move(options)), path_(options_.path.empty() ? DefaultPath() : options_.path) {
  if (options_.persistence_enabled) {
    (void)Reload();
  }
}

std::filesystem::path CalibrationProfileStore::DefaultPath() {
#if defined(_WIN32)
  if (const char *directory = std::getenv("LOCALAPPDATA")) {
    return std::filesystem::path(directory) / "onnx-light" / "calibration_profiles.cache";
  }
#else
  if (const char *directory = std::getenv("XDG_CACHE_HOME")) {
    return std::filesystem::path(directory) / "onnx-light" / "calibration_profiles.cache";
  }
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / ".cache" / "onnx-light" / "calibration_profiles.cache";
  }
#endif
  return "onnx-light-calibration-profiles.cache";
}

CalibrationProfileStoreReport CalibrationProfileStore::Reload() {
  std::lock_guard guard(mutex_);
  CalibrationProfileStoreReport report;
  if (!options_.persistence_enabled) {
    report.status = CalibrationProfileStoreStatus::kDisabled;
    return report;
  }
  std::error_code error;
  if (!std::filesystem::exists(path_, error)) {
    report.status = error ? CalibrationProfileStoreStatus::kUnreadable
                          : CalibrationProfileStoreStatus::kNotFound;
    if (error) {
      report.diagnostics.push_back("Unable to inspect profile store '" + path_.string() +
                                   "': " + error.message());
    }
    return report;
  }
  std::ifstream input(path_);
  if (!input) {
    report.status = CalibrationProfileStoreStatus::kUnreadable;
    report.diagnostics.push_back("Unable to read profile store '" + path_.string() + "'.");
    return report;
  }
  ParsedProfiles parsed = Parse(input);
  report.status = parsed.status;
  report.diagnostics = std::move(parsed.diagnostics);
  if (parsed.status == CalibrationProfileStoreStatus::kOk) {
    profiles_ = std::move(parsed.profiles);
    report.affected_profiles = profiles_.size();
  }
  return report;
}

CalibrationProfileStoreReport CalibrationProfileStore::Store(
    const CalibrationProfileKey &key, std::vector<CalibrationMeasurement> measurements,
    const CalibrationPolicySerializer &serialize, const CalibrationPolicyValidator &validate) {
  ValidateKey(key);
  if (!serialize) {
    throw std::invalid_argument("Calibration policy serializer is required.");
  }
  for (const CalibrationMeasurement &measurement : measurements) {
    if (measurement.name.empty() || !std::isfinite(measurement.value)) {
      throw std::invalid_argument("Calibration measurements require a name and finite value.");
    }
  }
  std::string policy = serialize();
  std::string validation_error;
  if (policy.empty() || (validate && !validate(policy, validation_error))) {
    CalibrationProfileStoreReport report;
    report.status = CalibrationProfileStoreStatus::kPolicyRejected;
    report.diagnostics.push_back(policy.empty()
                                     ? "Backend serialized an empty calibration policy."
                                     : "Backend rejected calibration policy: " + validation_error);
    return report;
  }
  CalibrationProfile replacement{key, std::move(policy), std::move(measurements), false};
  return Mutate([&](std::vector<CalibrationProfile> &profiles) {
    auto found =
        std::find_if(profiles.begin(), profiles.end(), [&](const CalibrationProfile &profile) {
          return !profile.user_override && profile.key == key;
        });
    if (found == profiles.end()) {
      profiles.push_back(std::move(replacement));
    } else {
      *found = std::move(replacement);
    }
    return size_t{1};
  });
}

CalibrationProfileLookupReport
CalibrationProfileStore::Lookup(const CalibrationProfileLookupOptions &options,
                                const CalibrationPolicyValidator &validate) const {
  ValidateKey(options.exact_key);
  if (options.exact_key.kind != CalibrationProfileKind::kExact) {
    throw std::invalid_argument("Calibration lookup exact_key must be exact.");
  }
  std::lock_guard guard(mutex_);
  CalibrationProfileLookupReport report;
  bool found = false;
  auto try_source = [&](bool overrides, CalibrationProfileKind kind) {
    for (const CalibrationProfile &profile : profiles_) {
      if (profile.user_override != overrides || profile.key.kind != kind) {
        continue;
      }
      const bool matches = kind == CalibrationProfileKind::kExact
                               ? MatchesExact(profile.key, options.exact_key)
                               : MatchesPortable(profile.key, options);
      if (!matches) {
        report.rejections.push_back(ClassifyRejection(profile, options));
        continue;
      }
      if (found) {
        continue;
      }
      std::string validation_error;
      if (validate && !validate(profile.policy, validation_error)) {
        report.rejections.push_back({profile.key,
                                     CalibrationProfileRejectionReason::kPolicyRejected,
                                     "Backend rejected stored policy: " + validation_error});
        continue;
      }
      report.profile = profile;
      report.status = CalibrationProfileStoreStatus::kOk;
      found = true;
    }
  };
  if (!options.force_portable) {
    try_source(true, CalibrationProfileKind::kExact);
  }
  try_source(true, CalibrationProfileKind::kPortable);
  if (!options.force_portable) {
    try_source(false, CalibrationProfileKind::kExact);
  }
  try_source(false, CalibrationProfileKind::kPortable);
  if (!found) {
    report.status = CalibrationProfileStoreStatus::kNotFound;
  }
  return report;
}

CalibrationProfileStoreReport
CalibrationProfileStore::InstallOverride(const CalibrationProfileKey &key,
                                         const CalibrationPolicySerializer &serialize,
                                         const CalibrationPolicyValidator &validate) {
  ValidateKey(key);
  if (!serialize) {
    throw std::invalid_argument("Calibration policy serializer is required.");
  }
  std::string policy = serialize();
  std::string validation_error;
  if (policy.empty() || (validate && !validate(policy, validation_error))) {
    CalibrationProfileStoreReport report;
    report.status = CalibrationProfileStoreStatus::kPolicyRejected;
    report.diagnostics.push_back(
        policy.empty() ? "Backend serialized an empty calibration override."
                       : "Backend rejected calibration override: " + validation_error);
    return report;
  }
  CalibrationProfile replacement{key, std::move(policy), {}, true};
  return Mutate([&](std::vector<CalibrationProfile> &profiles) {
    auto found =
        std::find_if(profiles.begin(), profiles.end(), [&](const CalibrationProfile &profile) {
          return profile.user_override && profile.key == key;
        });
    if (found == profiles.end()) {
      profiles.push_back(std::move(replacement));
    } else {
      *found = std::move(replacement);
    }
    return size_t{1};
  });
}

CalibrationProfileStoreReport
CalibrationProfileStore::ClearOverride(const CalibrationProfileKey &key) {
  ValidateKey(key);
  return Mutate([&](std::vector<CalibrationProfile> &profiles) {
    const size_t before = profiles.size();
    std::erase_if(profiles, [&](const CalibrationProfile &profile) {
      return profile.user_override && profile.key == key;
    });
    return before - profiles.size();
  });
}

CalibrationProfileStoreReport
CalibrationProfileStore::Invalidate(std::string_view backend,
                                    std::optional<std::string_view> implementation_version) {
  if (backend.empty()) {
    throw std::invalid_argument("Calibration profile backend must not be empty.");
  }
  return Mutate([&](std::vector<CalibrationProfile> &profiles) {
    const size_t before = profiles.size();
    std::erase_if(profiles, [&](const CalibrationProfile &profile) {
      return profile.key.backend == backend &&
             (!implementation_version.has_value() ||
              profile.key.implementation_version == *implementation_version);
    });
    return before - profiles.size();
  });
}

std::vector<CalibrationProfile> CalibrationProfileStore::Inspect() const {
  std::lock_guard guard(mutex_);
  return profiles_;
}

CalibrationProfileStoreReport CalibrationProfileStore::Mutate(
    const std::function<size_t(std::vector<CalibrationProfile> &)> &mutation) {
  std::lock_guard guard(mutex_);
  CalibrationProfileStoreReport report;
  if (!options_.persistence_enabled) {
    report.affected_profiles = mutation(profiles_);
    report.status = CalibrationProfileStoreStatus::kDisabled;
    return report;
  }

  const std::filesystem::path directory = path_.parent_path();
  std::error_code error;
  if (!directory.empty()) {
    std::filesystem::create_directories(directory, error);
    if (error) {
      report.status = CalibrationProfileStoreStatus::kWriteFailed;
      report.diagnostics.push_back("Unable to create profile-store directory '" +
                                   directory.string() + "': " + error.message());
      return report;
    }
  }
  std::filesystem::path lock_path = path_;
  lock_path += ".lock";
  InterprocessLock lock;
  std::string lock_error;
  if (!lock.Acquire(lock_path, lock_error)) {
    report.status = CalibrationProfileStoreStatus::kUnreadable;
    report.diagnostics.push_back(std::move(lock_error));
    return report;
  }

  std::vector<CalibrationProfile> latest;
  const bool exists = std::filesystem::exists(path_, error);
  if (error) {
    report.status = CalibrationProfileStoreStatus::kUnreadable;
    report.diagnostics.push_back("Unable to inspect profile store '" + path_.string() +
                                 "': " + error.message());
    return report;
  }
  if (exists) {
    std::ifstream input(path_);
    if (!input) {
      report.status = CalibrationProfileStoreStatus::kUnreadable;
      report.diagnostics.push_back("Unable to read profile store '" + path_.string() + "'.");
      return report;
    }
    ParsedProfiles parsed = Parse(input);
    if (parsed.status != CalibrationProfileStoreStatus::kOk) {
      report.status = parsed.status;
      report.diagnostics = std::move(parsed.diagnostics);
      return report;
    }
    latest = std::move(parsed.profiles);
  }
  report.affected_profiles = mutation(latest);
  if (!AtomicWrite(path_, latest, report.diagnostics)) {
    report.status = CalibrationProfileStoreStatus::kWriteFailed;
    std::filesystem::remove(path_.string() + ".tmp", error);
    return report;
  }
  profiles_ = std::move(latest);
  report.status = CalibrationProfileStoreStatus::kOk;
  return report;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
