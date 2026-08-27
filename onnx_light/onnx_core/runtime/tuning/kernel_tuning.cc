// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/kernel_tuning.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/kernels/random.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2 : values[middle];
}

double CriterionValue(const KernelTuningLatencyMetrics &metrics, KernelTuningCriterion criterion) {
  switch (criterion) {
  case KernelTuningCriterion::kAverage:
    return metrics.average;
  case KernelTuningCriterion::kSum:
    return metrics.sum;
  case KernelTuningCriterion::kMedian:
    return metrics.median;
  case KernelTuningCriterion::kAverageSpeedup:
    return *metrics.average_speedup;
  case KernelTuningCriterion::kMedianSpeedup:
    return *metrics.median_speedup;
  case KernelTuningCriterion::kMaxSpeedup:
    return *metrics.max_speedup;
  case KernelTuningCriterion::kMaxLatency:
    return metrics.max_latency;
  }
  throw std::invalid_argument("Unknown kernel tuning criterion.");
}

bool Maximizes(KernelTuningCriterion criterion) {
  return criterion == KernelTuningCriterion::kAverageSpeedup ||
         criterion == KernelTuningCriterion::kMedianSpeedup ||
         criterion == KernelTuningCriterion::kMaxSpeedup;
}

bool HasCriterion(const KernelTuningLatencyMetrics &metrics, KernelTuningCriterion criterion) {
  switch (criterion) {
  case KernelTuningCriterion::kAverageSpeedup:
    return metrics.average_speedup.has_value();
  case KernelTuningCriterion::kMedianSpeedup:
    return metrics.median_speedup.has_value();
  case KernelTuningCriterion::kMaxSpeedup:
    return metrics.max_speedup.has_value();
  default:
    return true;
  }
}

template <typename T> bool Contains(const std::vector<T> &values, const T &value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

void HashCombine(size_t &seed, size_t value) noexcept {
  constexpr size_t kHashCombine = sizeof(size_t) == 8 ? static_cast<size_t>(0x9e3779b97f4a7c15ULL)
                                                      : static_cast<size_t>(0x9e3779b9UL);
  seed ^= value + kHashCombine + (seed << 6) + (seed >> 2);
}

bool IsValidParameterName(std::string_view name) {
  bool segment_start = true;
  for (unsigned char character : name) {
    if (character == '.') {
      if (segment_start) {
        return false;
      }
      segment_start = true;
      continue;
    }
    if (segment_start) {
      if (std::isalpha(character) == 0 && character != '_') {
        return false;
      }
      segment_start = false;
    } else if (std::isalnum(character) == 0 && character != '_') {
      return false;
    }
  }
  return !name.empty() && !segment_start;
}

void ValidateKey(const KernelTuningKey &key) {
  if (key.library.empty()) {
    throw std::invalid_argument("Kernel tuning key library must not be empty.");
  }
  if (key.kernel.empty()) {
    throw std::invalid_argument("Kernel tuning key kernel must not be empty.");
  }
  if (key.implementation.empty()) {
    throw std::invalid_argument("Kernel tuning key implementation must not be empty.");
  }
  if (key.element_type <= 0) {
    throw std::invalid_argument("Kernel tuning key element_type must be a defined ONNX type.");
  }
  if (key.device == Device::kUndefined) {
    throw std::invalid_argument("Kernel tuning key device must not be undefined.");
  }
  if (key.tuning_abi == 0) {
    throw std::invalid_argument("Kernel tuning key tuning_abi must be positive.");
  }
}

std::string KeyDescription(const KernelTuningKey &key) {
  return key.library + "/" + key.kernel + "/" + key.implementation;
}

std::string NormalizeSelectorValue(std::string_view value) {
  std::string canonical;
  for (unsigned char character : value) {
    if (std::isalnum(character) != 0) {
      canonical.push_back(static_cast<char>(std::tolower(character)));
    }
  }
  return canonical;
}

constexpr std::string_view CanonicalSelectorValue(std::string_view canonical) noexcept {
  if (canonical == "amd64" || canonical == "x64") {
    return "x8664";
  }
  if (canonical == "arm64") {
    return "aarch64";
  }
  if (canonical == "genuineintel") {
    return "intel";
  }
  if (canonical == "authenticamd") {
    return "amd";
  }
  return canonical;
}

bool OptionalStringsConflict(const std::optional<std::string> &left,
                             const std::optional<std::string> &right) {
  if (!left.has_value() || !right.has_value()) {
    return false;
  }
  const std::string normalized_left = NormalizeSelectorValue(*left);
  const std::string normalized_right = NormalizeSelectorValue(*right);
  return CanonicalSelectorValue(normalized_left) != CanonicalSelectorValue(normalized_right);
}

bool OptionalIntegersConflict(const std::optional<uint32_t> &left,
                              const std::optional<uint32_t> &right) {
  return left.has_value() && right.has_value() && left != right;
}

bool ModelsOverlap(const std::vector<uint32_t> &left, const std::vector<uint32_t> &right) {
  if (left.empty() || right.empty()) {
    return true;
  }
  return std::any_of(left.begin(), left.end(), [&](uint32_t model) {
    return std::find(right.begin(), right.end(), model) != right.end();
  });
}

bool SelectorsCanOverlap(const platform::CpuSelector &left, const platform::CpuSelector &right) {
  if (OptionalStringsConflict(left.architecture, right.architecture) ||
      OptionalStringsConflict(left.vendor, right.vendor) ||
      OptionalIntegersConflict(left.family, right.family) ||
      OptionalStringsConflict(left.microarchitecture, right.microarchitecture) ||
      !ModelsOverlap(left.models, right.models) ||
      left.required_features.Intersects(right.excluded_features) ||
      right.required_features.Intersects(left.excluded_features)) {
    return false;
  }
  const uint32_t minimum =
      std::max(left.minimum_threads.value_or(0), right.minimum_threads.value_or(0));
  const uint32_t maximum =
      std::min(left.maximum_threads.value_or(std::numeric_limits<uint32_t>::max()),
               right.maximum_threads.value_or(std::numeric_limits<uint32_t>::max()));
  return minimum <= maximum;
}

uint32_t SelectorSpecificity(const platform::CpuSelector &selector) {
  const bool exact =
      selector.vendor.has_value() && selector.family.has_value() && selector.models.size() == 1;
  const bool processor = selector.vendor.has_value() || selector.family.has_value() ||
                         !selector.models.empty() || selector.microarchitecture.has_value();
  uint32_t specificity = exact ? 3000 : (processor ? 2000 : 1000);
  specificity += selector.architecture.has_value();
  specificity += selector.vendor.has_value();
  specificity += selector.family.has_value();
  specificity += selector.microarchitecture.has_value();
  specificity += selector.required_features.bits() != 0;
  specificity += selector.excluded_features.bits() != 0;
  specificity += selector.minimum_threads.has_value();
  specificity += selector.maximum_threads.has_value();
  if (!selector.models.empty()) {
    specificity += 100 / static_cast<uint32_t>(selector.models.size());
  }
  return specificity;
}

void ValidateSelector(const platform::CpuSelector &selector) {
  const bool empty = !selector.architecture.has_value() && !selector.vendor.has_value() &&
                     !selector.family.has_value() && selector.models.empty() &&
                     !selector.microarchitecture.has_value() &&
                     selector.required_features.bits() == 0 &&
                     selector.excluded_features.bits() == 0 &&
                     !selector.minimum_threads.has_value() && !selector.maximum_threads.has_value();
  if (empty) {
    throw std::invalid_argument("Kernel tuning processor selector must not be empty.");
  }
  if (selector.required_features.Intersects(selector.excluded_features)) {
    throw std::invalid_argument(
        "Kernel tuning processor selector cannot require and exclude the same feature.");
  }
  if (selector.minimum_threads.has_value() && *selector.minimum_threads == 0) {
    throw std::invalid_argument("Kernel tuning minimum_threads must be positive.");
  }
  if (selector.maximum_threads.has_value() && *selector.maximum_threads == 0) {
    throw std::invalid_argument("Kernel tuning maximum_threads must be positive.");
  }
  if (selector.minimum_threads.has_value() && selector.maximum_threads.has_value() &&
      *selector.minimum_threads > *selector.maximum_threads) {
    throw std::invalid_argument("Kernel tuning minimum_threads must not exceed maximum_threads.");
  }
  std::unordered_set<uint32_t> models;
  for (uint32_t model : selector.models) {
    if (!models.emplace(model).second) {
      throw std::invalid_argument("Kernel tuning processor selector contains duplicate models.");
    }
  }
}

KernelTuningQueryValidationHook WrapValidationHook(KernelTuningValidationHook validation_hook) {
  if (!validation_hook) {
    return {};
  }
  return [validation_hook = std::move(validation_hook)](
             const KernelTuningParameters &parameters) -> std::optional<std::string> {
    try {
      validation_hook(parameters);
      return std::nullopt;
    } catch (const std::invalid_argument &exception) {
      return exception.what();
    }
  };
}

} // namespace

size_t KernelTuningKeyHash::operator()(const KernelTuningKey &key) const noexcept {
  size_t hash = std::hash<std::string>{}(key.library);
  HashCombine(hash, std::hash<std::string>{}(key.kernel));
  HashCombine(hash, std::hash<std::string>{}(key.implementation));
  HashCombine(hash, std::hash<int32_t>{}(key.element_type));
  HashCombine(hash, std::hash<int32_t>{}(static_cast<int32_t>(key.device)));
  HashCombine(hash, std::hash<uint32_t>{}(key.tuning_abi));
  return hash;
}

KernelTuningCriterion ParseKernelTuningCriterion(std::string_view criterion) {
  if (criterion == "average")
    return KernelTuningCriterion::kAverage;
  if (criterion == "sum")
    return KernelTuningCriterion::kSum;
  if (criterion == "median")
    return KernelTuningCriterion::kMedian;
  if (criterion == "average-speedup")
    return KernelTuningCriterion::kAverageSpeedup;
  if (criterion == "median-speedup")
    return KernelTuningCriterion::kMedianSpeedup;
  if (criterion == "max-speedup")
    return KernelTuningCriterion::kMaxSpeedup;
  if (criterion == "max-latency")
    return KernelTuningCriterion::kMaxLatency;
  throw std::invalid_argument("Unknown kernel tuning criterion '" + std::string(criterion) + "'.");
}

std::string_view KernelTuningCriterionName(KernelTuningCriterion criterion) {
  switch (criterion) {
  case KernelTuningCriterion::kAverage:
    return "average";
  case KernelTuningCriterion::kSum:
    return "sum";
  case KernelTuningCriterion::kMedian:
    return "median";
  case KernelTuningCriterion::kAverageSpeedup:
    return "average-speedup";
  case KernelTuningCriterion::kMedianSpeedup:
    return "median-speedup";
  case KernelTuningCriterion::kMaxSpeedup:
    return "max-speedup";
  case KernelTuningCriterion::kMaxLatency:
    return "max-latency";
  }
  throw std::invalid_argument("Unknown kernel tuning criterion.");
}

KernelTuningLatencyReport
AnalyzeKernelTuningLatencies(const std::vector<std::vector<std::optional<double>>> &latencies,
                             KernelTuningCriterion criterion) {
  if (latencies.empty() || latencies.front().empty()) {
    throw std::invalid_argument("Kernel tuning latency measurements must not be empty.");
  }
  const size_t case_count = latencies.front().size();
  for (const auto &row : latencies) {
    if (row.size() != case_count) {
      throw std::invalid_argument("Kernel tuning latency rows must have the same number of cases.");
    }
    for (const std::optional<double> &latency : row) {
      if (latency.has_value() && (!std::isfinite(*latency) || *latency <= 0)) {
        throw std::invalid_argument("Kernel tuning latencies must be positive and finite.");
      }
    }
  }

  KernelTuningLatencyReport report;
  report.criterion = criterion;
  report.values.reserve(latencies.size());
  const bool has_baseline =
      std::all_of(latencies.front().begin(), latencies.front().end(),
                  [](const std::optional<double> &value) { return value.has_value(); });
  for (const auto &row : latencies) {
    if (std::any_of(row.begin(), row.end(),
                    [](const std::optional<double> &value) { return !value.has_value(); })) {
      report.values.emplace_back(std::nullopt);
      continue;
    }
    std::vector<double> values;
    std::vector<double> speedups;
    values.reserve(case_count);
    speedups.reserve(case_count);
    double sum = 0;
    for (size_t index = 0; index < case_count; ++index) {
      const double value = *row[index];
      values.push_back(value);
      if (has_baseline) {
        speedups.push_back(*latencies.front()[index] / value);
      }
      sum += value;
    }
    KernelTuningLatencyMetrics metrics;
    metrics.average = sum / static_cast<double>(case_count);
    metrics.sum = sum;
    metrics.median = Median(values);
    metrics.max_latency = *std::max_element(values.begin(), values.end());
    if (has_baseline) {
      metrics.average_speedup =
          std::accumulate(speedups.begin(), speedups.end(), 0.0) / static_cast<double>(case_count);
      metrics.median_speedup = Median(speedups);
      metrics.max_speedup = *std::max_element(speedups.begin(), speedups.end());
    }
    report.values.push_back(std::move(metrics));
  }

  std::optional<size_t> selected;
  for (size_t index = 0; index < report.values.size(); ++index) {
    if (!report.values[index].has_value() || !HasCriterion(*report.values[index], criterion)) {
      continue;
    }
    if (!selected.has_value() ||
        (Maximizes(criterion) ? CriterionValue(*report.values[index], criterion) >
                                    CriterionValue(*report.values[*selected], criterion)
                              : CriterionValue(*report.values[index], criterion) <
                                    CriterionValue(*report.values[*selected], criterion))) {
      selected = index;
    }
  }
  report.selected_index = selected;
  return report;
}

std::string_view TuningValueTypeName(const TuningValue &value) noexcept {
  switch (value.index()) {
  case 0:
    return "int64";
  case 1:
    return "double";
  case 2:
    return "bool";
  case 3:
    return "string";
  default:
    return "unknown";
  }
}

bool KernelCalibrationSelection::Matches(const KernelTuningKey &key) const {
  return (!library.has_value() || key.library == *library) && Contains(kernels, key.kernel) &&
         Contains(implementations, key.implementation) &&
         Contains(element_types, key.element_type) &&
         (!device.has_value() || key.device == *device);
}

void CalibrationReporter::AddDiagnostic(std::string message) {
  diagnostics_.push_back(std::move(message));
}

CalibrationReporter::CalibrationReporter(const CalibrationOptions &options) {
  if (options.profiling_capacity != 0) {
    parallel_region_collector_ = std::make_unique<ParallelRegionCollector>(
        options.profiling_capacity, options.profiling_hardware_counters);
  }
}

void CalibrationReporter::RecordBenchmark(uint64_t memory_bytes, uint64_t duration_ns) {
  ++benchmark_cases_;
  peak_memory_bytes_ = std::max(peak_memory_bytes_, memory_bytes);
  measured_duration_ns_ += duration_ns;
}

void CalibrationReporter::ProfileCandidate(const std::function<void()> &run) {
  if (parallel_region_collector_ == nullptr) {
    return;
  }
  ParallelRegionCollectorScope collector_scope(parallel_region_collector_.get());
  run();
}

void CalibrationReporter::FinalizeCandidateDiagnostics() {
  if (parallel_region_collector_ != nullptr && !parallel_region_report_.has_value()) {
    parallel_region_report_ = parallel_region_collector_->Report();
    parallel_region_collector_.reset();
  }
}

void CalibrationReporter::SetComparison(KernelTuningComparison comparison) {
  comparison_ = std::move(comparison);
}

bool KernelTuningParameters::Contains(std::string_view name) const {
  return values.find(std::string(name)) != values.end();
}

void KernelTuningParameters::ThrowMissingValue(std::string_view name) {
  throw std::invalid_argument("Missing kernel tuning parameter '" + std::string(name) + "'.");
}

void KernelTuningParameters::ThrowWrongType(std::string_view name, std::string_view expected,
                                            std::string_view actual) {
  throw std::invalid_argument("Kernel tuning parameter '" + std::string(name) + "' must be " +
                              std::string(expected) + ", not " + std::string(actual) + ".");
}

KernelTuningSchema
KernelTuningSchema::WithQueryValidation(KernelTuningParameters portable_defaults,
                                        KernelTuningQueryValidationHook validation_hook) {
  return KernelTuningSchema(std::move(portable_defaults), QueryValidationTag{},
                            std::move(validation_hook));
}

KernelTuningSchema::KernelTuningSchema(KernelTuningParameters portable_defaults,
                                       KernelTuningValidationHook validation_hook)
    : KernelTuningSchema(std::move(portable_defaults), QueryValidationTag{},
                         WrapValidationHook(std::move(validation_hook))) {}

KernelTuningSchema::KernelTuningSchema(KernelTuningParameters portable_defaults,
                                       QueryValidationTag /*tag*/,
                                       KernelTuningQueryValidationHook validation_hook)
    : portable_defaults_(std::move(portable_defaults)),
      validation_hook_(std::move(validation_hook)) {
  ValidateKey(portable_defaults_.key);
  if (portable_defaults_.values.empty()) {
    throw std::invalid_argument("Kernel tuning portable defaults must not be empty.");
  }
  for (const auto &[name, value] : portable_defaults_.values) {
    if (!IsValidParameterName(name)) {
      throw std::invalid_argument("Invalid kernel tuning parameter name '" + name + "'.");
    }
    (void)value;
  }
  if (std::optional<std::string> error = ValidationError(portable_defaults_)) {
    throw std::invalid_argument(*error);
  }
}

void KernelTuningSchema::Validate(const KernelTuningParameters &parameters) const {
  if (std::optional<std::string> error = ValidationError(parameters)) {
    throw std::invalid_argument(*error);
  }
}

bool KernelTuningSchema::TryValidate(const KernelTuningParameters &parameters,
                                     std::string *error_message) const {
  if (std::optional<std::string> error = ValidationError(parameters)) {
    if (error_message != nullptr) {
      *error_message = *error;
    }
    return false;
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

std::optional<std::string>
KernelTuningSchema::ValidationError(const KernelTuningParameters &parameters) const {
  if (parameters.key != portable_defaults_.key) {
    return "Kernel tuning parameters for '" + KeyDescription(parameters.key) +
           "' do not match schema '" + KeyDescription(portable_defaults_.key) + "'.";
  }
  for (const auto &[name, value] : parameters.values) {
    auto expected = portable_defaults_.values.find(name);
    if (expected == portable_defaults_.values.end()) {
      return "Unknown kernel tuning parameter '" + name + "'.";
    }
    if (value.index() != expected->second.index()) {
      return "Kernel tuning parameter '" + name + "' must be " +
             std::string(TuningValueTypeName(expected->second)) + ", not " +
             std::string(TuningValueTypeName(value)) + ".";
    }
  }
  for (const auto &[name, value] : portable_defaults_.values) {
    if (!parameters.Contains(name)) {
      return "Missing kernel tuning parameter '" + name + "'.";
    }
    (void)value;
  }
  if (validation_hook_) {
    return validation_hook_(parameters);
  }
  return std::nullopt;
}

struct KernelTuningRegistrySnapshot::State {
  struct AccessCounters {
    std::atomic<uint64_t> snapshots{0};
    std::atomic<uint64_t> lookups{0};
    std::atomic<uint64_t> resolutions{0};
  };

  struct RegisteredProfile {
    KernelTuningKey key;
    platform::CpuSelector processors;
    KernelTuningParameters parameters;
    int priority = 0;
    uint32_t specificity = 0;
  };

  struct PublishedExecutionProfile {
    KernelTuningParameters parameters;
    CpuExecutionDescriptor execution;
  };

  uint64_t generation = 0;
  std::unordered_map<KernelTuningKey, KernelTuningParameters, KernelTuningKeyHash> profiles;
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> published_keys;
  std::vector<PublishedExecutionProfile> published_execution_profiles;
  std::vector<RegisteredProfile> registered_profiles;
  std::shared_ptr<AccessCounters> access_counters;
};

struct KernelTuningRegistry::Impl {
  mutable std::mutex mutex;
  std::unordered_map<KernelTuningKey, std::shared_ptr<const KernelTuningSchema>,
                     KernelTuningKeyHash>
      schemas;
  std::unordered_map<KernelTuningKey, KernelCalibrationFunction, KernelTuningKeyHash>
      calibration_functions;
  std::shared_ptr<KernelTuningRegistrySnapshot::State::AccessCounters> access_counters;
  std::shared_ptr<const KernelTuningRegistrySnapshot::State> state;

  Impl() {
    auto initial = std::make_shared<KernelTuningRegistrySnapshot::State>();
    access_counters = std::make_shared<KernelTuningRegistrySnapshot::State::AccessCounters>();
    initial->access_counters = access_counters;
    state = std::move(initial);
  }
};

uint64_t KernelTuningRegistrySnapshot::generation() const noexcept { return state_->generation; }

const KernelTuningParameters *
KernelTuningRegistrySnapshot::Find(const KernelTuningKey &key) const noexcept {
  state_->access_counters->lookups.fetch_add(1, std::memory_order_relaxed);
  auto found = state_->profiles.find(key);
  return found == state_->profiles.end() ? nullptr : &found->second;
}

bool KernelTuningRegistrySnapshot::HasPublishedProfile(
    const KernelTuningKey &key, const CpuExecutionDescriptor &execution) const noexcept {
  state_->access_counters->lookups.fetch_add(1, std::memory_order_relaxed);
  return state_->published_keys.contains(key) ||
         std::any_of(state_->published_execution_profiles.begin(),
                     state_->published_execution_profiles.end(),
                     [&](const State::PublishedExecutionProfile &profile) {
                       return profile.parameters.key == key && profile.execution == execution;
                     });
}

const KernelTuningParameters *
KernelTuningRegistrySnapshot::Resolve(const KernelTuningKey &key,
                                      const CpuExecutionDescriptor &execution) const noexcept {
  state_->access_counters->lookups.fetch_add(1, std::memory_order_relaxed);
  state_->access_counters->resolutions.fetch_add(1, std::memory_order_relaxed);
  auto portable = state_->profiles.find(key);
  if (portable == state_->profiles.end() || state_->published_keys.contains(key)) {
    return portable == state_->profiles.end() ? nullptr : &portable->second;
  }

  auto calibrated = std::find_if(
      state_->published_execution_profiles.rbegin(), state_->published_execution_profiles.rend(),
      [&](const State::PublishedExecutionProfile &profile) {
        return profile.parameters.key == key && profile.execution == execution;
      });
  if (calibrated != state_->published_execution_profiles.rend()) {
    return &calibrated->parameters;
  }

  const State::RegisteredProfile *selected = nullptr;
  for (const State::RegisteredProfile &profile : state_->registered_profiles) {
    if (profile.key != key ||
        !profile.processors.Matches(execution.processor, execution.effective_threads)) {
      continue;
    }
    if (selected == nullptr || profile.specificity > selected->specificity ||
        (profile.specificity == selected->specificity && profile.priority > selected->priority)) {
      selected = &profile;
    }
  }
  return selected == nullptr ? &portable->second : &selected->parameters;
}

KernelTuningRegistry::KernelTuningRegistry() : impl_(std::make_unique<Impl>()) {}

KernelTuningRegistry::~KernelTuningRegistry() = default;

void KernelTuningRegistry::RegisterSchema(KernelTuningSchema schema) {
  auto shared_schema = std::make_shared<const KernelTuningSchema>(std::move(schema));
  std::lock_guard lock(impl_->mutex);
  if (impl_->schemas.contains(shared_schema->key())) {
    throw std::invalid_argument("Kernel tuning schema is already registered for '" +
                                KeyDescription(shared_schema->key()) + "'.");
  }

  auto current = impl_->state;
  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*current);
  ++next->generation;
  next->profiles.emplace(shared_schema->key(), shared_schema->portable_defaults());
  impl_->schemas.emplace(shared_schema->key(), std::move(shared_schema));
  impl_->state = std::move(next);
}

void KernelTuningRegistry::RegisterProfile(const KernelTuningKey &key,
                                           platform::CpuSelector processors,
                                           KernelTuningParameters parameters, int priority) {
  if (parameters.key != key) {
    throw std::invalid_argument("Kernel tuning profile parameters have a mismatched key.");
  }
  const ProcessorKernelTuningProfile profile{std::move(parameters), std::move(processors),
                                             priority};
  RegisterProfiles(std::span<const ProcessorKernelTuningProfile>(&profile, 1));
}

void KernelTuningRegistry::RegisterProfiles(
    std::span<const ProcessorKernelTuningProfile> profiles) {
  if (profiles.empty()) {
    return;
  }

  std::lock_guard lock(impl_->mutex);
  struct ValidatedProfile {
    const ProcessorKernelTuningProfile *profile;
    uint32_t specificity;
  };
  std::vector<ValidatedProfile> validated;
  validated.reserve(profiles.size());
  for (const ProcessorKernelTuningProfile &profile : profiles) {
    ValidateSelector(profile.processors);
    auto schema = impl_->schemas.find(profile.parameters.key);
    if (schema == impl_->schemas.end()) {
      throw std::invalid_argument("Cannot register profile for unregistered kernel tuning key '" +
                                  KeyDescription(profile.parameters.key) + "'.");
    }
    schema->second->Validate(profile.parameters);
    const uint32_t specificity = SelectorSpecificity(profile.processors);
    for (const auto &registered : impl_->state->registered_profiles) {
      if (registered.key == profile.parameters.key && registered.specificity == specificity &&
          registered.priority == profile.priority &&
          SelectorsCanOverlap(registered.processors, profile.processors)) {
        throw std::invalid_argument("Ambiguous kernel tuning profile for '" +
                                    KeyDescription(profile.parameters.key) +
                                    "': matching selectors have equal specificity and priority.");
      }
    }
    for (const ValidatedProfile &other : validated) {
      if (other.profile->parameters.key == profile.parameters.key &&
          other.specificity == specificity && other.profile->priority == profile.priority &&
          SelectorsCanOverlap(other.profile->processors, profile.processors)) {
        throw std::invalid_argument("Ambiguous kernel tuning profile batch for '" +
                                    KeyDescription(profile.parameters.key) +
                                    "': matching selectors have equal specificity and priority.");
      }
    }
    validated.push_back({&profile, specificity});
  }

  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*impl_->state);
  for (const ValidatedProfile &entry : validated) {
    next->registered_profiles.push_back({entry.profile->parameters.key, entry.profile->processors,
                                         entry.profile->parameters, entry.profile->priority,
                                         entry.specificity});
  }
  ++next->generation;
  impl_->state = std::move(next);
}

void KernelTuningRegistry::RegisterCalibrationFunction(const KernelTuningKey &key,
                                                       KernelCalibrationFunction function) {
  if (!function) {
    throw std::invalid_argument("Kernel calibration function must not be empty.");
  }
  std::lock_guard lock(impl_->mutex);
  if (!impl_->schemas.contains(key)) {
    throw std::invalid_argument("Cannot register calibration for unregistered kernel tuning key '" +
                                KeyDescription(key) + "'.");
  }
  if (!impl_->calibration_functions.emplace(key, std::move(function)).second) {
    throw std::invalid_argument("Kernel calibration function is already registered for '" +
                                KeyDescription(key) + "'.");
  }
}

KernelTuningRegistrySnapshot KernelTuningRegistry::Snapshot() const noexcept {
  std::lock_guard lock(impl_->mutex);
  impl_->access_counters->snapshots.fetch_add(1, std::memory_order_relaxed);
  return KernelTuningRegistrySnapshot(impl_->state);
}

KernelTuningRegistryAccessCounts KernelTuningRegistry::AccessCounts() const noexcept {
  const auto &counters = impl_->access_counters;
  return {counters->snapshots.load(std::memory_order_relaxed),
          counters->lookups.load(std::memory_order_relaxed),
          counters->resolutions.load(std::memory_order_relaxed)};
}

void KernelTuningRegistry::PublishProfiles(std::span<const KernelTuningParameters> profiles,
                                           std::span<const KernelTuningKey> reset_keys) {
  std::lock_guard lock(impl_->mutex);
  for (const KernelTuningKey &key : reset_keys) {
    if (!impl_->schemas.contains(key)) {
      throw std::invalid_argument("Cannot reset unregistered kernel tuning key '" +
                                  KeyDescription(key) + "'.");
    }
  }
  for (const KernelTuningParameters &profile : profiles) {
    auto schema = impl_->schemas.find(profile.key);
    if (schema == impl_->schemas.end()) {
      throw std::invalid_argument("Cannot publish unregistered kernel tuning key '" +
                                  KeyDescription(profile.key) + "'.");
    }
    schema->second->Validate(profile);
  }

  auto current = impl_->state;
  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*current);
  for (const KernelTuningKey &key : reset_keys) {
    next->profiles[key] = impl_->schemas.at(key)->portable_defaults();
    next->published_keys.erase(key);
  }
  for (const KernelTuningParameters &profile : profiles) {
    next->profiles[profile.key] = profile;
    next->published_keys.emplace(profile.key);
  }
  ++next->generation;
  impl_->state = std::move(next);
}

void KernelTuningRegistry::PublishCalibratedProfiles(
    std::span<const KernelTuningParameters> profiles, const CpuExecutionDescriptor &execution,
    std::span<const KernelTuningKey> reset_keys) {
  if (execution.effective_threads == 0) {
    throw std::invalid_argument("Published calibration effective_threads must be positive.");
  }
  std::lock_guard lock(impl_->mutex);
  for (const KernelTuningKey &key : reset_keys) {
    if (!impl_->schemas.contains(key)) {
      throw std::invalid_argument("Cannot reset unregistered kernel tuning key '" +
                                  KeyDescription(key) + "'.");
    }
  }
  for (const KernelTuningParameters &profile : profiles) {
    auto schema = impl_->schemas.find(profile.key);
    if (schema == impl_->schemas.end()) {
      throw std::invalid_argument("Cannot publish unregistered kernel tuning key '" +
                                  KeyDescription(profile.key) + "'.");
    }
    schema->second->Validate(profile);
  }

  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> replaced(reset_keys.begin(),
                                                                    reset_keys.end());
  for (const KernelTuningParameters &profile : profiles) {
    replaced.emplace(profile.key);
  }
  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*impl_->state);
  std::erase_if(next->published_execution_profiles,
                [&](const KernelTuningRegistrySnapshot::State::PublishedExecutionProfile &profile) {
                  return profile.execution == execution &&
                         replaced.contains(profile.parameters.key);
                });
  for (const KernelTuningParameters &profile : profiles) {
    next->published_execution_profiles.push_back({profile, execution});
  }
  ++next->generation;
  impl_->state = std::move(next);
}

std::vector<KernelTuningKey> KernelTuningRegistry::RegisteredKeys() const {
  std::lock_guard lock(impl_->mutex);
  std::vector<KernelTuningKey> keys;
  keys.reserve(impl_->schemas.size());
  for (const auto &[key, schema] : impl_->schemas) {
    keys.push_back(key);
    (void)schema;
  }
  return keys;
}

std::shared_ptr<const KernelTuningSchema>
KernelTuningRegistry::FindSchema(const KernelTuningKey &key) const {
  std::lock_guard lock(impl_->mutex);
  auto found = impl_->schemas.find(key);
  return found == impl_->schemas.end() ? nullptr : found->second;
}

KernelCalibrationFunction
KernelTuningRegistry::FindCalibrationFunction(const KernelTuningKey &key) const {
  std::lock_guard lock(impl_->mutex);
  auto found = impl_->calibration_functions.find(key);
  return found == impl_->calibration_functions.end() ? KernelCalibrationFunction{} : found->second;
}

KernelTuningRegistry &GetKernelTuningRegistry() {
  static KernelTuningRegistry registry;
  return registry;
}

void RegisterKernelTuningSchema(KernelTuningSchema schema) {
  GetKernelTuningRegistry().RegisterSchema(std::move(schema));
}

void RegisterKernelTuningProfile(const KernelTuningKey &key, platform::CpuSelector processors,
                                 KernelTuningParameters parameters, int priority) {
  GetKernelTuningRegistry().RegisterProfile(key, std::move(processors), std::move(parameters),
                                            priority);
}

void RegisterKernelCalibrationFunction(const KernelTuningKey &key,
                                       KernelCalibrationFunction function) {
  GetKernelTuningRegistry().RegisterCalibrationFunction(key, std::move(function));
}

std::vector<KernelCalibrationCase>
MakeElementwiseCalibrationCases(int32_t element_type, size_t input_count, int64_t first_elements,
                                int64_t maximum_elements, bool include_broadcasting) {
  if (input_count == 0 || input_count > 2) {
    throw std::invalid_argument("Elementwise calibration supports one or two inputs.");
  }
  if (first_elements <= 0 || maximum_elements < first_elements) {
    throw std::invalid_argument("Elementwise calibration element bounds are invalid.");
  }
  if (include_broadcasting && input_count != 2) {
    throw std::invalid_argument("Elementwise calibration broadcasting requires two inputs.");
  }

  std::vector<KernelCalibrationCase> cases;
  for (int64_t elements = first_elements; elements <= maximum_elements;) {
    KernelCalibrationCase equal;
    equal.name = input_count == 1 ? "unary" : "equal_shape";
    equal.problem_size = static_cast<uint64_t>(elements);
    equal.output_element_type = element_type;
    equal.output_shape = {elements};
    for (size_t input = 0; input < input_count; ++input) {
      equal.inputs.push_back(
          {element_type, {elements}, uint64_t{5} + static_cast<uint64_t>(input)});
    }
    cases.push_back(std::move(equal));

    if (include_broadcasting) {
      cases.push_back({"scalar_broadcast",
                       static_cast<uint64_t>(elements),
                       {{element_type, {elements}, 5}, {element_type, {}, 6}},
                       element_type,
                       {elements}});
      if (elements >= 4 && elements % 4 == 0) {
        cases.push_back({"multidirectional_broadcast",
                         static_cast<uint64_t>(elements),
                         {{element_type, {elements / 4, 4}, 5}, {element_type, {1, 4}, 6}},
                         element_type,
                         {elements / 4, 4}});
      }
    }
    if (elements > maximum_elements / 2) {
      break;
    }
    elements *= 2;
  }
  return cases;
}

namespace {

Tensor GenerateCalibrationInput(const CalibrationInputSpec &spec) {
  switch (static_cast<DataType>(spec.element_type)) {
  case DataType::UINT8:
    return Tensor::From("", spec.shape, RandUint<uint8_t>(16, spec.shape, spec.seed));
  case DataType::UINT16:
    return Tensor::From("", spec.shape, RandUint<uint16_t>(16, spec.shape, spec.seed));
  case DataType::UINT32:
    return Tensor::From("", spec.shape, RandUint<uint32_t>(16, spec.shape, spec.seed));
  case DataType::UINT64:
    return Tensor::From("", spec.shape, RandUint<uint64_t>(16, spec.shape, spec.seed));
  default:
    return RandnTensor(spec.element_type, spec.shape, spec.seed);
  }
}

uint64_t TensorStorageBytes(int32_t element_type, const Shape &shape) {
  const int64_t elements = shape.product(0, shape.size(), "calibration tensor");
  return static_cast<uint64_t>(PackedByteSize(element_type, elements));
}

uint64_t CaseMemoryBytes(const KernelCalibrationCase &benchmark_case) {
  const uint64_t output_bytes =
      TensorStorageBytes(benchmark_case.output_element_type, benchmark_case.output_shape);
  if (output_bytes > std::numeric_limits<uint64_t>::max() / 2) {
    throw std::invalid_argument("Calibration case output memory size overflows uint64.");
  }
  uint64_t bytes = 2 * output_bytes;
  for (const CalibrationInputSpec &input : benchmark_case.inputs) {
    const uint64_t input_bytes = TensorStorageBytes(input.element_type, input.shape);
    if (input_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::invalid_argument("Calibration case memory size overflows uint64.");
    }
    bytes += input_bytes;
  }
  return bytes;
}

bool ExactCalibrationOutput(const Tensor &reference, const Tensor &candidate) {
  return reference.data_type == candidate.data_type && reference.shape == candidate.shape &&
         reference.size_bytes() == candidate.size_bytes() &&
         (reference.size_bytes() == 0 ||
          std::memcmp(reference.bytes(), candidate.bytes(), reference.size_bytes()) == 0);
}

int64_t Median(std::vector<int64_t> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int64_t Measure(const std::function<void()> &run) {
  const auto begin = std::chrono::steady_clock::now();
  run();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              begin)
      .count();
}

} // namespace

KernelTuningParameters CompareKernelBenchmarkValues(const KernelTuningKey &key,
                                                    const CalibrationOptions &options,
                                                    CalibrationReporter &reporter,
                                                    const KernelCalibrationBenchmark &benchmark) {
  if (!options.parameter_name.has_value() || options.parameter_values.empty()) {
    throw std::invalid_argument("Explicit tuning comparison requires a parameter and values.");
  }
  if (options.parameter_values.size() > 64) {
    throw std::invalid_argument("Explicit tuning comparison supports at most 64 values.");
  }
  if (*options.parameter_name != benchmark.parameter_name) {
    throw std::invalid_argument("Calibration callback benchmarks parameter '" +
                                benchmark.parameter_name + "', not '" + *options.parameter_name +
                                "'.");
  }

  const uint64_t memory_budget = options.maximum_memory_bytes == 0
                                     ? benchmark.default_maximum_memory_bytes
                                     : options.maximum_memory_bytes;
  const uint64_t duration_ms = options.maximum_duration_ms == 0
                                   ? benchmark.default_maximum_duration_ms
                                   : options.maximum_duration_ms;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
  const auto validate =
      benchmark.validate_output ? benchmark.validate_output : ExactCalibrationOutput;
  benchmark.reference.configure(benchmark.serial_parameter_value);

  std::vector<uint64_t> durations(options.parameter_values.size(), 0);
  uint64_t measured_cases = 0;
  for (const KernelCalibrationCase &benchmark_case : benchmark.cases) {
    if (benchmark_case.name.empty() || benchmark_case.inputs.empty()) {
      throw std::invalid_argument("Calibration case name and inputs must not be empty.");
    }
    const uint64_t memory_bytes = CaseMemoryBytes(benchmark_case);
    if (memory_bytes > memory_budget) {
      continue;
    }

    std::vector<Tensor> inputs;
    inputs.reserve(benchmark_case.inputs.size());
    for (const CalibrationInputSpec &input : benchmark_case.inputs) {
      inputs.push_back(GenerateCalibrationInput(input));
    }
    const uint64_t output_bytes =
        TensorStorageBytes(benchmark_case.output_element_type, benchmark_case.output_shape);
    Tensor reference_output =
        MakeOutputTensor(benchmark_case.output_element_type, benchmark_case.output_shape,
                         static_cast<size_t>(output_bytes), nullptr);
    const std::span<const Tensor> input_span(inputs);
    benchmark.reference.run(input_span, reference_output);

    std::vector<Tensor> candidate_outputs;
    candidate_outputs.reserve(options.parameter_values.size());
    for (size_t value_index = 0; value_index < options.parameter_values.size(); ++value_index) {
      const int64_t value = options.parameter_values[value_index];
      benchmark.candidate.configure(value);
      candidate_outputs.push_back(MakeOutputTensor(benchmark_case.output_element_type,
                                                   benchmark_case.output_shape,
                                                   static_cast<size_t>(output_bytes), nullptr));
      benchmark.candidate.run(input_span, candidate_outputs.back());
      if (!validate(reference_output, candidate_outputs.back())) {
        throw std::runtime_error(key.kernel + " comparison case '" + benchmark_case.name +
                                 "' output differs for " + benchmark.parameter_name + "=" +
                                 std::to_string(value) + ".");
      }
    }

    std::vector<std::vector<int64_t>> samples(options.parameter_values.size());
    std::vector<uint64_t> measured_durations(options.parameter_values.size(), 0);
    for (std::vector<int64_t> &value_samples : samples) {
      value_samples.reserve(static_cast<size_t>(benchmark.repetitions));
    }
    for (int repetition = 0; repetition < benchmark.repetitions; ++repetition) {
      for (size_t value_index = 0; value_index < options.parameter_values.size(); ++value_index) {
        benchmark.candidate.configure(options.parameter_values[value_index]);
        const auto run_candidate = [&]() {
          benchmark.candidate.run(input_span, candidate_outputs[value_index]);
        };
        const int64_t sample = Measure(run_candidate);
        samples[value_index].push_back(sample);
        measured_durations[value_index] += static_cast<uint64_t>(sample);
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
    }
    for (size_t value_index = 0; value_index < options.parameter_values.size(); ++value_index) {
      reporter.RecordBenchmark(memory_bytes, measured_durations[value_index]);
      durations[value_index] += static_cast<uint64_t>(Median(std::move(samples[value_index])));
    }
    ++measured_cases;
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
  }

  KernelTuningParameters selected = benchmark.portable_parameters;
  selected.values[benchmark.parameter_name] = options.parameter_values.front();
  if (measured_cases == 0) {
    reporter.AddDiagnostic(key.kernel +
                           " comparison memory budget is too small; kept the baseline value.");
    reporter.FinalizeCandidateDiagnostics();
    return selected;
  }

  const auto fastest = std::min_element(durations.begin(), durations.end());
  const size_t fastest_index = static_cast<size_t>(std::distance(durations.begin(), fastest));
  const int64_t selected_value = options.parameter_values[fastest_index];
  selected.values[benchmark.parameter_name] = selected_value;

  KernelTuningComparison comparison;
  comparison.parameter_name = benchmark.parameter_name;
  comparison.baseline_value = options.parameter_values.front();
  comparison.selected_value = selected_value;
  comparison.values.reserve(options.parameter_values.size());
  for (size_t index = 0; index < options.parameter_values.size(); ++index) {
    comparison.values.push_back(
        {options.parameter_values[index], durations[index], measured_cases});
  }
  reporter.SetComparison(std::move(comparison));
  reporter.AddDiagnostic(key.kernel + " selected " + benchmark.parameter_name + "=" +
                         std::to_string(selected_value) + " from explicit values.");
  reporter.FinalizeCandidateDiagnostics();
  return selected;
}

KernelTuningParameters CalibrateKernelBenchmark(const KernelTuningKey &key,
                                                const CpuExecutionDescriptor &execution,
                                                const CalibrationOptions &options,
                                                CalibrationReporter &reporter,
                                                const KernelCalibrationBenchmark &benchmark) {
  if (benchmark.portable_parameters.key != key ||
      !benchmark.portable_parameters.Contains(benchmark.parameter_name)) {
    throw std::invalid_argument("Calibration benchmark portable parameters are incomplete.");
  }
  if (!benchmark.reference.configure || !benchmark.reference.run ||
      !benchmark.candidate.configure || !benchmark.candidate.run || benchmark.cases.empty()) {
    throw std::invalid_argument("Calibration benchmark runners and cases must not be empty.");
  }
  if (benchmark.repetitions <= 0 || benchmark.required_consecutive_wins <= 0 ||
      benchmark.minimum_speedup < 0.0 || benchmark.minimum_speedup >= 1.0) {
    throw std::invalid_argument("Calibration benchmark search parameters are invalid.");
  }
  const uint32_t actual_threads = static_cast<uint32_t>(ParallelForThreadCount());
  if (execution.effective_threads != actual_threads) {
    throw std::invalid_argument(
        "Calibration requested " + std::to_string(execution.effective_threads) +
        " threads, but ParallelFor uses " + std::to_string(actual_threads) + ".");
  }

  if (options.parameter_name.has_value() || !options.parameter_values.empty()) {
    return CompareKernelBenchmarkValues(key, options, reporter, benchmark);
  }

  KernelTuningParameters selected = benchmark.portable_parameters;
  if (actual_threads == 1) {
    reporter.AddDiagnostic(key.kernel +
                           " kept the portable threshold because parallel execution is "
                           "unavailable.");
    reporter.FinalizeCandidateDiagnostics();
    return selected;
  }

  const uint64_t memory_budget = options.maximum_memory_bytes == 0
                                     ? benchmark.default_maximum_memory_bytes
                                     : options.maximum_memory_bytes;
  const uint64_t duration_ms = options.maximum_duration_ms == 0
                                   ? benchmark.default_maximum_duration_ms
                                   : options.maximum_duration_ms;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
  const auto validate =
      benchmark.validate_output ? benchmark.validate_output : ExactCalibrationOutput;
  benchmark.reference.configure(benchmark.serial_parameter_value);

  int consecutive_wins = 0;
  uint64_t first_winning_size = 0;
  bool measured_any = false;
  bool tuned = false;
  uint64_t previous_problem_size = 0;
  for (size_t case_index = 0; case_index < benchmark.cases.size();) {
    const uint64_t problem_size = benchmark.cases[case_index].problem_size;
    if (problem_size == 0 || problem_size < previous_problem_size ||
        problem_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument(
          "Calibration cases must have positive, nondecreasing int64 problem sizes.");
    }
    previous_problem_size = problem_size;
    const int64_t candidate_value = static_cast<int64_t>((problem_size + 1) / 2);
    benchmark.candidate.configure(candidate_value);
    bool group_won = true;
    bool group_measured = false;

    while (case_index < benchmark.cases.size() &&
           benchmark.cases[case_index].problem_size == problem_size) {
      const KernelCalibrationCase &benchmark_case = benchmark.cases[case_index++];
      if (benchmark_case.name.empty() || benchmark_case.inputs.empty()) {
        throw std::invalid_argument("Calibration case name and inputs must not be empty.");
      }
      const uint64_t memory_bytes = CaseMemoryBytes(benchmark_case);
      if (memory_bytes > memory_budget) {
        group_won = false;
        continue;
      }

      std::vector<Tensor> inputs;
      inputs.reserve(benchmark_case.inputs.size());
      for (const CalibrationInputSpec &input : benchmark_case.inputs) {
        inputs.push_back(GenerateCalibrationInput(input));
      }
      const uint64_t output_bytes =
          TensorStorageBytes(benchmark_case.output_element_type, benchmark_case.output_shape);
      Tensor reference_output =
          MakeOutputTensor(benchmark_case.output_element_type, benchmark_case.output_shape,
                           static_cast<size_t>(output_bytes), nullptr);
      Tensor candidate_output =
          MakeOutputTensor(benchmark_case.output_element_type, benchmark_case.output_shape,
                           static_cast<size_t>(output_bytes), nullptr);
      const std::span<const Tensor> input_span(inputs);
      const auto run_reference = [&]() { benchmark.reference.run(input_span, reference_output); };
      const auto run_candidate = [&]() { benchmark.candidate.run(input_span, candidate_output); };
      run_reference();
      run_candidate();
      if (!validate(reference_output, candidate_output)) {
        throw std::runtime_error(key.kernel + " calibration case '" + benchmark_case.name +
                                 "' candidate output differs from the reference output.");
      }

      std::vector<int64_t> reference_samples;
      std::vector<int64_t> candidate_samples;
      reference_samples.reserve(static_cast<size_t>(benchmark.repetitions));
      candidate_samples.reserve(static_cast<size_t>(benchmark.repetitions));
      uint64_t measured_duration_ns = 0;
      for (int repetition = 0; repetition < benchmark.repetitions; ++repetition) {
        const int64_t reference_ns = Measure(run_reference);
        const int64_t candidate_ns = Measure(run_candidate);
        reference_samples.push_back(reference_ns);
        candidate_samples.push_back(candidate_ns);
        measured_duration_ns +=
            static_cast<uint64_t>(reference_ns) + static_cast<uint64_t>(candidate_ns);
      }
      reporter.RecordBenchmark(memory_bytes, measured_duration_ns);
      measured_any = true;
      group_measured = true;
      const int64_t reference_ns = Median(std::move(reference_samples));
      const int64_t candidate_ns = Median(std::move(candidate_samples));
      if (static_cast<double>(candidate_ns) >
          static_cast<double>(reference_ns) * (1.0 - benchmark.minimum_speedup)) {
        group_won = false;
      }
      const auto profiling_begin = std::chrono::steady_clock::now();
      reporter.ProfileCandidate(run_candidate);
      if (!validate(reference_output, candidate_output)) {
        throw std::runtime_error(key.kernel + " calibration case '" + benchmark_case.name +
                                 "' profiled candidate output differs from the reference output.");
      }
      deadline += std::chrono::steady_clock::now() - profiling_begin;
    }

    if (group_measured && group_won) {
      if (consecutive_wins == 0) {
        first_winning_size = problem_size;
      }
      ++consecutive_wins;
      if (consecutive_wins == benchmark.required_consecutive_wins) {
        const int64_t minimum_elements = static_cast<int64_t>((first_winning_size + 1) / 2);
        selected.values[benchmark.parameter_name] = minimum_elements;
        reporter.AddDiagnostic(key.kernel + " selected " + benchmark.parameter_name + "=" +
                               std::to_string(minimum_elements) + ".");
        tuned = true;
        break;
      }
    } else {
      consecutive_wins = 0;
      first_winning_size = 0;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
  }
  if (!measured_any) {
    reporter.AddDiagnostic(key.kernel +
                           " calibration memory budget is too small; kept the portable "
                           "threshold.");
  } else if (!tuned) {
    reporter.AddDiagnostic(key.kernel +
                           " calibration found no stable parallel crossover; kept the portable "
                           "threshold.");
  }
  reporter.FinalizeCandidateDiagnostics();
  return selected;
}

CalibrationBatchReport CalibrateRegisteredKernels(const KernelCalibrationSelection &selection,
                                                  const CalibrationOptions &options) {
  CalibrationBatchReport report;
  KernelTuningRegistry &registry = GetKernelTuningRegistry();
  const KernelTuningRegistrySnapshot before = registry.Snapshot();
  const platform::CpuDescriptor &processor = platform::GetCpuDescriptor();
  CpuExecutionDescriptor execution = options.execution.value_or(
      CpuExecutionDescriptor{processor, static_cast<uint32_t>(ParallelForThreadCount())});
  const uint32_t active_threads = static_cast<uint32_t>(ParallelForThreadCount());
  if (options.maximum_threads.has_value()) {
    if (*options.maximum_threads == 0) {
      throw std::invalid_argument("Calibration maximum_threads must be positive.");
    }
    if (*options.maximum_threads < active_threads) {
      throw std::invalid_argument(
          "Calibration maximum_threads cannot be lower than the active executor participant "
          "count; select a smaller CpuExecutor before calibration.");
    }
    execution.effective_threads = std::min(execution.effective_threads, *options.maximum_threads);
  }
  if (execution.effective_threads == 0) {
    throw std::invalid_argument("Calibration effective thread count must be positive.");
  }
  if (execution.effective_threads != active_threads) {
    throw std::invalid_argument(
        "Calibration execution descriptor requests " + std::to_string(execution.effective_threads) +
        " threads, but the active executor uses " + std::to_string(active_threads) + ".");
  }
  if (selection.device.has_value() && *selection.device != Device::kCPU) {
    throw std::invalid_argument("Kernel calibration currently supports only the CPU device.");
  }

  std::vector<KernelTuningKey> keys = registry.RegisteredKeys();
  std::sort(keys.begin(), keys.end(),
            [](const KernelTuningKey &left, const KernelTuningKey &right) {
              return std::tie(left.library, left.kernel, left.implementation, left.element_type,
                              left.device, left.tuning_abi) <
                     std::tie(right.library, right.kernel, right.implementation, right.element_type,
                              right.device, right.tuning_abi);
            });
  for (const KernelTuningKey &key : keys) {
    if (!selection.Matches(key) || (!selection.device.has_value() && key.device != Device::kCPU)) {
      continue;
    }
    if (selection.only_missing && before.HasPublishedProfile(key, execution)) {
      report.skipped.push_back(key);
      continue;
    }
    KernelCalibrationFunction function = registry.FindCalibrationFunction(key);
    if (!function) {
      report.unsupported.push_back(key);
      continue;
    }

    CalibrationReporter reporter(options);
    KernelTuningParameters parameters = function(key, execution, options, reporter);
    std::shared_ptr<const KernelTuningSchema> schema = registry.FindSchema(key);
    if (schema == nullptr) {
      throw std::invalid_argument("Kernel tuning schema disappeared during calibration.");
    }
    schema->Validate(parameters);
    report.calibrated.push_back(std::move(parameters));
    for (const std::string &diagnostic : reporter.diagnostics()) {
      report.diagnostics.push_back({key, diagnostic});
    }
    if (reporter.benchmark_cases() != 0) {
      report.resources.push_back({key, reporter.benchmark_cases(), reporter.peak_memory_bytes(),
                                  reporter.measured_duration_ns()});
    }
    if (const std::optional<ParallelRegionReport> &parallel_regions =
            reporter.parallel_region_report()) {
      report.candidate_diagnostics.push_back({key, *parallel_regions});
    }
    if (const std::optional<KernelTuningComparison> &comparison = reporter.comparison()) {
      report.comparisons.push_back({key, *comparison});
    }
  }

  if (!report.calibrated.empty()) {
    registry.PublishCalibratedProfiles(report.calibrated, execution);
  }
  report.published_generation = registry.Snapshot().generation();
  return report;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
