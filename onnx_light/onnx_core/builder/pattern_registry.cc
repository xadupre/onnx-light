// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_registry.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

struct PatternRegistration {
  std::string name;
  PatternFactory factory;
  bool enabled_by_default;
};

struct PatternRegistry {
  std::mutex mutex;
  std::vector<PatternRegistration> registrations;
  std::unordered_map<std::string, std::size_t> indices;
  std::map<std::string, std::vector<std::string>> pattern_sets;
  std::vector<std::string> all_pattern_names;
};

PatternRegistry &MutablePatternRegistry() {
  static PatternRegistry registry;
  return registry;
}

} // namespace

void PatternOptimization::SetRegisteredName(const std::string &name) {
  if (name.empty()) {
    throw PatternRegistrationError("PatternOptimization: registered name cannot be empty.");
  }
  if (!Name().empty() && Name() != name) {
    throw PatternRegistrationError("PatternOptimization: intrinsic name '" + Name() +
                                   "' does not match registered name '" + name + "'.");
  }
  name_ = name;
}

void RegisterPattern(const std::string &name, PatternFactory factory, bool enabled_by_default) {
  if (name.empty()) {
    throw PatternRegistrationError("RegisterPattern: pattern name cannot be empty.");
  }
  if (!factory) {
    throw PatternRegistrationError("RegisterPattern: factory for '" + name + "' is empty.");
  }

  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (registry.indices.contains(name)) {
    throw PatternRegistrationError("RegisterPattern: pattern '" + name +
                                   "' is already registered.");
  }
  const std::size_t index = registry.registrations.size();
  registry.registrations.push_back(
      PatternRegistration{name, std::move(factory), enabled_by_default});
  registry.indices.insert({name, index});
}

std::vector<std::string> RegisteredPatternNames(bool include_opt_in) {
  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<std::string> names;
  names.reserve(registry.registrations.size());
  for (const PatternRegistration &registration : registry.registrations) {
    if (include_opt_in || registration.enabled_by_default)
      names.push_back(registration.name);
  }
  return names;
}

void RegisterPatternSet(const std::string &name, const std::vector<std::string> &pattern_names) {
  if (name.empty() || name == "all") {
    throw PatternRegistrationError("RegisterPatternSet: invalid set name '" + name + "'.");
  }
  for (const std::string &pattern_name : pattern_names) {
    if (pattern_name.empty()) {
      throw PatternRegistrationError("RegisterPatternSet: pattern names cannot be empty.");
    }
  }

  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (registry.pattern_sets.contains(name)) {
    throw PatternRegistrationError("RegisterPatternSet: set '" + name + "' is already registered.");
  }
  std::vector<std::string> unique_names;
  for (const std::string &pattern_name : pattern_names) {
    if (std::find(unique_names.begin(), unique_names.end(), pattern_name) == unique_names.end()) {
      unique_names.push_back(pattern_name);
    }
  }
  registry.pattern_sets.emplace(name, std::move(unique_names));
}

void ExtendPatternSet(const std::string &name, const std::vector<std::string> &pattern_names) {
  if (name.empty()) {
    throw PatternRegistrationError("ExtendPatternSet: set name cannot be empty.");
  }
  for (const std::string &pattern_name : pattern_names) {
    if (pattern_name.empty()) {
      throw PatternRegistrationError("ExtendPatternSet: pattern names cannot be empty.");
    }
  }

  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<std::string> *names = nullptr;
  if (name == "all") {
    names = &registry.all_pattern_names;
  } else {
    const auto set = registry.pattern_sets.find(name);
    if (set == registry.pattern_sets.end()) {
      throw PatternRegistrationError("ExtendPatternSet: set '" + name + "' is not registered.");
    }
    names = &set->second;
  }
  for (const std::string &pattern_name : pattern_names) {
    if (std::find(names->begin(), names->end(), pattern_name) == names->end()) {
      names->push_back(pattern_name);
    }
  }
}

bool UnregisterPatternSet(const std::string &name) {
  if (name.empty()) {
    throw PatternRegistrationError("UnregisterPatternSet: set name cannot be empty.");
  }
  if (name == "all" || name == "L1" || name == "L2" || name == "Layer") {
    return false;
  }
  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.pattern_sets.erase(name) != 0;
}

std::map<std::string, std::vector<std::string>> RegisteredPatternSets() {
  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::map<std::string, std::vector<std::string>> sets = registry.pattern_sets;
  std::vector<std::string> all_names;
  all_names.reserve(registry.registrations.size() + registry.all_pattern_names.size());
  for (const PatternRegistration &registration : registry.registrations) {
    all_names.push_back(registration.name);
  }
  const auto append_unique = [&all_names](const std::string &name) {
    if (std::find(all_names.begin(), all_names.end(), name) == all_names.end()) {
      all_names.push_back(name);
    }
  };
  for (const std::string &name : registry.all_pattern_names) {
    append_unique(name);
  }
  for (const auto &[set_name, names] : registry.pattern_sets) {
    (void)set_name;
    for (const std::string &name : names) {
      append_unique(name);
    }
  }
  sets.emplace("all", std::move(all_names));
  return sets;
}

std::vector<std::unique_ptr<PatternOptimization>>
CreateRegisteredPatterns(std::optional<symbolic::Device> device) {
  PatternRegistry &registry = MutablePatternRegistry();
  std::vector<PatternRegistration> registrations;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registrations = registry.registrations;
  }

  std::vector<std::unique_ptr<PatternOptimization>> patterns;
  patterns.reserve(registrations.size());
  for (const PatternRegistration &registration : registrations) {
    if (!registration.enabled_by_default)
      continue;
    std::unique_ptr<PatternOptimization> pattern = registration.factory();
    if (pattern == nullptr) {
      throw PatternRegistrationError("CreateRegisteredPatterns: factory for '" + registration.name +
                                     "' returned null.");
    }
    pattern->SetRegisteredName(registration.name);
    if (!device.has_value() || pattern->device == symbolic::Device::kUndefined ||
        pattern->device == *device)
      patterns.push_back(std::move(pattern));
  }
  return patterns;
}

std::unique_ptr<PatternOptimization> CreateRegisteredPattern(const std::string &name,
                                                             std::optional<int> priority) {
  if (name.empty()) {
    throw PatternRegistrationError("CreateRegisteredPattern: pattern name cannot be empty.");
  }

  PatternFactory factory;
  {
    PatternRegistry &registry = MutablePatternRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto index = registry.indices.find(name);
    if (index == registry.indices.end()) {
      throw PatternRegistrationError("CreateRegisteredPattern: pattern '" + name +
                                     "' is not registered.");
    }
    factory = registry.registrations[index->second].factory;
  }

  std::unique_ptr<PatternOptimization> pattern = factory();
  if (pattern == nullptr) {
    throw PatternRegistrationError("CreateRegisteredPattern: factory for '" + name +
                                   "' returned null.");
  }
  pattern->SetRegisteredName(name);
  if (priority.has_value()) {
    pattern->priority = *priority;
  }
  return pattern;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
