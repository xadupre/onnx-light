// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Version converter interface for ONNX models between different opset versions.

#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "onnx/common/ir.h"
#include "onnx/defs/schema.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace version_conversion {

// TODO(ONNX): Consider creating interface for this class.
/// Base class that stores adapters and converts models between opset versions.
class BaseVersionConverter {
  /// Registered adapters keyed as {op_name: {from_opset: {to_opset: adapter}}}.
protected:
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Adapter>>>>
      adapters;

  /// Operator schemas keyed as {op_name: {domain: {version: schema}}}.
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::map<int64_t, const OpSchema *>>>
      all_schemas;

public:
  BaseVersionConverter() = default;

  virtual ~BaseVersionConverter() = default;

  /// Returns the adapter for a node and a specific source/target opset pair.
  /// This method is intended to be called from convert_version once the caller
  /// has determined that an adapter must exist for this conversion step.
  ///
  /// \param op Node whose operator name selects the adapter family.
  /// \param initial_version Source opset identifier for the conversion step.
  /// \param target_version Target opset identifier for the conversion step.
  /// \return Registered adapter matching op, initial_version and target_version.
  const Adapter &adapter_lookup(const Node *op, const OpSetID &initial_version,
                                const OpSetID &target_version) const {
    const std::string op_name = op->kind().toString();
    const std::string initial = initial_version.toString();
    const std::string target = target_version.toString();
    // Find appropriate adapter in adapters map for provided initial and target versions
    // TODO(ONNX): Consider abstracting elements of this that are specific to
    // DefaultConverter to separate methods here and maintain the procedure in Base Converter
    const auto op_adapters = adapters.find(op_name);
    if (op_adapters != adapters.end()) {
      // If we're adapting downwards, we just want to find the one downwards
      // adapter implemented for initial_version. If we're adapting upwards, we
      // want to actually use the SinceVersion value for the given op.
      const auto target_map = op_adapters->second.find(initial);
      if (target_map != op_adapters->second.end()) {
        // Either adapt from SinceVersion or Incompatible Breaking Change
        const auto adapter_ptr = target_map->second.find(target);
        if (adapter_ptr != target_map->second.end()) {
          return *(adapter_ptr->second);
        } else {
          ONNX_ASSERTM(false, "No Adapter To Version %s for %s", target.c_str(), op_name.c_str())
        }
      } else {
        ONNX_ASSERTM(false, "No Adapter From Version %s for %s", initial.c_str(), op_name.c_str())
      }
    } else {
      // No adapters exist for the given op
      ONNX_ASSERTM(false, "No Adapter For %s", op_name.c_str())
    }
  }

  /// Converts a model from one opset version to another.
  ///
  /// \param mp_in Input model.
  /// \param initial_version Source opset identifier.
  /// \param target_version Target opset identifier.
  /// \return Converted model.
  virtual ModelProto convert_version(const ModelProto &mp_in, const OpSetID &initial_version,
                                     const OpSetID &target_version) const = 0;

  /// Registers an adapter instance.
  void registerAdapter(std::unique_ptr<Adapter> a_ptr) {
    const OpSetID &iv = a_ptr->initial_version();
    const OpSetID &tv = a_ptr->target_version();
    adapters[a_ptr->name()][iv.toString()][tv.toString()] = std::move(a_ptr);
  }

  /// Registers a generic adapter from a transformation callback.
  void registerAdapter(const char *op, int64_t from, int64_t to,
                       const NodeTransformerFunction &transformer) {
    registerAdapter(std::make_unique<GenericAdapter>(op, from, to, transformer));
  }
};

} // namespace version_conversion
} // namespace ONNX_LIGHT_NAMESPACE
