// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_registry.h"

#include <algorithm>
#include <mutex>
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
};

PatternRegistry &MutablePatternRegistry() {
  static PatternRegistry registry;
  return registry;
}

} // namespace

void RegisterPattern(const std::string &name, PatternFactory factory) {
  if (name.empty()) {
    throw PatternRegistrationError("RegisterPattern: pattern name cannot be empty.");
  }
  if (!factory) {
    throw PatternRegistrationError("RegisterPattern: factory for '" + name + "' is empty.");
  }

  PatternRegistry &registry = MutablePatternRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto existing = std::find_if(
      registry.registrations.begin(), registry.registrations.end(),
      [&name](const PatternRegistration &registration) { return registration.name == name; });
  if (existing != registry.registrations.end()) {
    throw PatternRegistrationError("RegisterPattern: pattern '" + name +
                                   "' is already registered.");
  }
  registry.registrations.push_back(PatternRegistration{name, std::move(factory)});
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
    patterns.push_back(std::move(pattern));
  }
  return patterns;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
