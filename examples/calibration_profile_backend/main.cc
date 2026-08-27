// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/calibration_profile_store.h"

#include <iostream>
#include <string>
#include <string_view>

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;

struct ExampleBackendPolicy {
  std::string algorithm;
  int tile_size = 0;

  std::string Serialize() const {
    return "algorithm=" + algorithm + ";tile_size=" + std::to_string(tile_size);
  }

  static bool Validate(std::string_view payload, std::string &error) {
    if (payload.starts_with("algorithm=") && payload.find(";tile_size=") != payload.npos) {
      return true;
    }
    error = "expected algorithm and tile_size";
    return false;
  }
};

int main() {
  runtime::CalibrationProfileStore store({{}, false});
  runtime::CalibrationProfileKey key;
  key.backend = "example_backend";
  key.operator_name = "MatMul";
  key.implementation_version = "1";
  key.model_digest = "example-model-digest";
  key.processor = "example-processor:l2=1048576";
  key.thread_count = 4;

  const ExampleBackendPolicy calibrated{"blocked", 32};
  store.Store(
      key, {{"median_latency", 2.4, "us"}}, [&calibrated] { return calibrated.Serialize(); },
      ExampleBackendPolicy::Validate);

  runtime::CalibrationProfileLookupOptions lookup;
  lookup.exact_key = key;
  const runtime::CalibrationProfileLookupReport selected =
      store.Lookup(lookup, ExampleBackendPolicy::Validate);
  if (!selected.profile.has_value()) {
    return 1;
  }
  std::cout << selected.profile->policy << '\n';
  return 0;
}
