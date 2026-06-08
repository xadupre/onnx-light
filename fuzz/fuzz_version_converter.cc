// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``onnx_light::version_converter::ConvertVersion``.
//
// Parses random bytes into a ``ModelProto`` and, on success, attempts
// to convert it to a handful of neighbouring opset versions. Mirrors
// the former ``onnx_light/fuzz/fuzz_version_converter.py``.

#include "onnx_lib/defs/schema.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::OpSchemaRegistry;
namespace version_conversion = ONNX_LIGHT_NAMESPACE::version_conversion;

namespace {

int default_opset_version(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    const auto &domain = opset.ref_domain();
    if (domain == "" || domain == "ai.onnx") {
      return opset.has_version() ? static_cast<int>(opset.ref_version()) : -1;
    }
  }
  return -1;
}

int latest_onnx_opset_version() {
  const auto &map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  auto it = map.find("");
  if (it == map.end()) {
    it = map.find("ai.onnx");
  }
  return it == map.end() ? 1 : it->second.second;
}

std::vector<int> candidate_target_versions(const ModelProto &model) {
  const int latest = latest_onnx_opset_version();
  const int current = default_opset_version(model);
  if (current <= 0) {
    return {latest};
  }
  std::vector<int> targets;
  if (current > 1) {
    targets.push_back(current - 1);
  }
  if (current < latest) {
    targets.push_back(current + 1);
  }
  if (current != latest) {
    bool already = false;
    for (int v : targets) {
      if (v == latest) {
        already = true;
        break;
      }
    }
    if (!already) {
      targets.push_back(latest);
    }
  }
  return targets;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  ModelProto model;
  try {
    model.ParseFromString(std::string(reinterpret_cast<const char *>(data), size));
  } catch (...) {
    return 0;
  }

  for (int target_version : candidate_target_versions(model)) {
    try {
      version_conversion::ConvertVersion(model, target_version);
    } catch (...) {
      // ConvertError and friends are expected on malformed inputs.
    }
  }
  return 0;
}
