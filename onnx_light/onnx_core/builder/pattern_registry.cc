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
};

struct PatternRegistry {
  std::mutex mutex;
  std::vector<PatternRegistration> registrations;
  std::unordered_map<std::string, std::size_t> indices;
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

void RegisterPattern(const std::string &name, PatternFactory factory) {
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
  registry.registrations.push_back(PatternRegistration{name, std::move(factory)});
  registry.indices.insert({name, index});
}

std::vector<std::string> RegisteredPatternNames() {
  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<std::string> names;
  names.reserve(registry.registrations.size());
  for (const PatternRegistration &registration : registry.registrations) {
    names.push_back(registration.name);
  }
  return names;
}

std::vector<std::unique_ptr<PatternOptimization>> CreateRegisteredPatterns() {
  PatternRegistry &registry = MutablePatternRegistry();
  std::vector<PatternRegistration> registrations;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registrations = registry.registrations;
  }

  std::vector<std::unique_ptr<PatternOptimization>> patterns;
  patterns.reserve(registrations.size());
  for (const PatternRegistration &registration : registrations) {
    std::unique_ptr<PatternOptimization> pattern = registration.factory();
    if (pattern == nullptr) {
      throw PatternRegistrationError("CreateRegisteredPatterns: factory for '" + registration.name +
                                     "' returned null.");
    }
    pattern->SetRegisteredName(registration.name);
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
