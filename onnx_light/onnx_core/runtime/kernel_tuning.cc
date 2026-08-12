// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning.h"

#include <atomic>
#include <cctype>
#include <mutex>
#include <stdexcept>
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
  uint64_t generation = 0;
  std::unordered_map<KernelTuningKey, KernelTuningParameters, KernelTuningKeyHash> profiles;
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

KernelTuningRegistry::KernelTuningRegistry() : impl_(std::make_unique<Impl>()) {}

KernelTuningRegistry::~KernelTuningRegistry() = default;

void KernelTuningRegistry::RegisterSchema(KernelTuningSchema schema) {
  auto shared_schema = std::make_shared<const KernelTuningSchema>(std::move(schema));
  std::lock_guard lock(impl_->mutex);
  if (impl_->schemas.contains(shared_schema->key())) {
    throw std::invalid_argument("Kernel tuning schema is already registered for '" +
                                KeyDescription(shared_schema->key()) + "'.");
  }

  auto current = std::atomic_load(&impl_->state);
  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*current);
  ++next->generation;
  next->profiles.emplace(shared_schema->key(), shared_schema->portable_defaults());
  impl_->schemas.emplace(shared_schema->key(), std::move(shared_schema));
  std::shared_ptr<const KernelTuningRegistrySnapshot::State> published = std::move(next);
  std::atomic_store(&impl_->state, std::move(published));
}

KernelTuningRegistrySnapshot KernelTuningRegistry::Snapshot() const noexcept {
  return KernelTuningRegistrySnapshot(std::atomic_load(&impl_->state));
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

  auto current = std::atomic_load(&impl_->state);
  auto next = std::make_shared<KernelTuningRegistrySnapshot::State>(*current);
  for (const KernelTuningKey &key : reset_keys) {
    next->profiles[key] = impl_->schemas.at(key)->portable_defaults();
  }
  for (const KernelTuningParameters &profile : profiles) {
    next->profiles[profile.key] = profile;
  }
  ++next->generation;
  std::shared_ptr<const KernelTuningRegistrySnapshot::State> published = std::move(next);
  std::atomic_store(&impl_->state, std::move(published));
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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
