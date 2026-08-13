// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

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

KernelTuningSchema::KernelTuningSchema(KernelTuningParameters portable_defaults,
                                       KernelTuningValidationHook validation_hook)
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
  if (validation_hook_) {
    validation_hook_(portable_defaults_);
  }
}

void KernelTuningSchema::Validate(const KernelTuningParameters &parameters) const {
  if (parameters.key != portable_defaults_.key) {
    throw std::invalid_argument("Kernel tuning parameters for '" + KeyDescription(parameters.key) +
                                "' do not match schema '" + KeyDescription(portable_defaults_.key) +
                                "'.");
  }
  ValidateValues(parameters);
  if (validation_hook_) {
    validation_hook_(parameters);
  }
}

void KernelTuningSchema::ValidateValues(const KernelTuningParameters &parameters) const {
  for (const auto &[name, value] : parameters.values) {
    auto expected = portable_defaults_.values.find(name);
    if (expected == portable_defaults_.values.end()) {
      throw std::invalid_argument("Unknown kernel tuning parameter '" + name + "'.");
    }
    if (value.index() != expected->second.index()) {
      throw std::invalid_argument("Kernel tuning parameter '" + name + "' must be " +
                                  std::string(TuningValueTypeName(expected->second)) + ", not " +
                                  std::string(TuningValueTypeName(value)) + ".");
    }
  }
  for (const auto &[name, value] : portable_defaults_.values) {
    if (!parameters.Contains(name)) {
      throw std::invalid_argument("Missing kernel tuning parameter '" + name + "'.");
    }
    (void)value;
  }
}

struct KernelTuningRegistrySnapshot::State {
  struct RegisteredProfile {
    KernelTuningKey key;
    platform::CpuSelector processors;
    KernelTuningParameters parameters;
    int priority = 0;
    uint32_t specificity = 0;
  };

  uint64_t generation = 0;
  std::unordered_map<KernelTuningKey, KernelTuningParameters, KernelTuningKeyHash> profiles;
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> published_keys;
  std::vector<RegisteredProfile> registered_profiles;
};

struct KernelTuningRegistry::Impl {
  mutable std::mutex mutex;
  std::unordered_map<KernelTuningKey, std::shared_ptr<const KernelTuningSchema>,
                     KernelTuningKeyHash>
      schemas;
  std::shared_ptr<const KernelTuningRegistrySnapshot::State> state;

  Impl() : state(std::make_shared<const KernelTuningRegistrySnapshot::State>()) {}
};

uint64_t KernelTuningRegistrySnapshot::generation() const noexcept { return state_->generation; }

const KernelTuningParameters *
KernelTuningRegistrySnapshot::Find(const KernelTuningKey &key) const noexcept {
  auto found = state_->profiles.find(key);
  return found == state_->profiles.end() ? nullptr : &found->second;
}

const KernelTuningParameters *
KernelTuningRegistrySnapshot::Resolve(const KernelTuningKey &key,
                                      const CpuExecutionDescriptor &execution) const noexcept {
  auto portable = state_->profiles.find(key);
  if (portable == state_->profiles.end() || state_->published_keys.contains(key)) {
    return portable == state_->profiles.end() ? nullptr : &portable->second;
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
  ValidateSelector(processors);
  if (parameters.key != key) {
    throw std::invalid_argument("Kernel tuning profile parameters have a mismatched key.");
  }
  const uint32_t specificity = SelectorSpecificity(processors);

  std::lock_guard lock(impl_->mutex);
  auto schema = impl_->schemas.find(key);
  if (schema == impl_->schemas.end()) {
    throw std::invalid_argument("Cannot register profile for unregistered kernel tuning key '" +
                                KeyDescription(key) + "'.");
  }
  schema->second->Validate(parameters);
  for (const auto &registered : impl_->state->registered_profiles) {
    if (registered.key == key && registered.specificity == specificity &&
        registered.priority == priority && SelectorsCanOverlap(registered.processors, processors)) {
      throw std::invalid_argument("Ambiguous kernel tuning profile for '" + KeyDescription(key) +
                                  "': matching selectors have equal specificity and priority.");
    }
  }

  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*impl_->state);
  next->registered_profiles.push_back(
      {key, std::move(processors), std::move(parameters), priority, specificity});
  ++next->generation;
  impl_->state = std::move(next);
}

KernelTuningRegistrySnapshot KernelTuningRegistry::Snapshot() const noexcept {
  std::lock_guard lock(impl_->mutex);
  return KernelTuningRegistrySnapshot(impl_->state);
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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
