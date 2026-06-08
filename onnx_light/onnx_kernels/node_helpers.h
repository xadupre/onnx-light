// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_proto/onnx.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @file node_helpers.h
 * @brief Small inline helpers shared by the dispatcher
 *        (``run_nodes.cc``) and the kernel dispatch table
 *        (``kernel_dispatch_table.cc``).
 *
 * The helpers normalise the default ONNX domain, validate input/output
 * arity declared on a ``NodeProto``, look up tensors by name in a
 * ``RuntimeContext`` and read the most common attribute types
 * (``INT``, ``INTS``, ``FLOAT``, ``STRING``). They are kept ``inline``
 * (and namespace-private) so that each translation unit gets its own
 * copy and there is no ODR risk.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace detail {

// Canonical name of the default ONNX domain. The empty string used by
// ``NodeProto::domain()`` for the default ONNX domain is normalised to
// this value before dispatch-table lookups.
inline constexpr const char *kDefaultOnnxDomain = "ai.onnx";

inline std::string NormaliseDispatchDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  return domain.empty() ? std::string(kDefaultOnnxDomain) : domain;
}

inline const Tensor &GetInput(const NodeProto &node, int index, const TensorMap &tensors) {
  const std::string name = node.input(index).as_string();
  if (name.empty()) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' input #" +
                                std::to_string(index) + " is unset (empty name).");
  }
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::invalid_argument("RunNode: input '" + name + "' of op '" +
                                node.op_type().as_string() + "' is missing from the tensor map.");
  }
  return it->second;
}

// Same as :func:`GetInput` but returns ``nullptr`` when the input slot
// is either absent (``index >= node.input_size()``) or declared with
// an empty name (the ONNX convention for an unconnected optional
// input). Throws if the slot has a non-empty name but the tensor is
// missing from ``tensors``, since that indicates a graph-wiring bug
// rather than an "absent" optional input.
inline const Tensor *GetOptionalInput(const NodeProto &node, int index, const TensorMap &tensors) {
  if (index >= node.input_size()) {
    return nullptr;
  }
  const std::string name = node.input(index).as_string();
  if (name.empty()) {
    return nullptr;
  }
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::invalid_argument("RunNode: input '" + name + "' of op '" +
                                node.op_type().as_string() + "' is missing from the tensor map.");
  }
  return &it->second;
}

inline void SetOutput(const NodeProto &node, int index, Tensor result, TensorMap &tensors) {
  const std::string name = node.output(index).as_string();
  if (name.empty()) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' output #" +
                                std::to_string(index) + " is unset (empty name).");
  }
  result.name = name;
  tensors[name] = std::move(result);
}

inline void RequireInputCount(const NodeProto &node, int expected) {
  if (static_cast<int>(node.input_size()) != expected) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(expected) + " input(s), got " +
                                std::to_string(node.input_size()) + ".");
  }
}

inline void RequireMinInputCount(const NodeProto &node, int min_expected) {
  if (static_cast<int>(node.input_size()) < min_expected) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                "' expects at least " + std::to_string(min_expected) +
                                " input(s), got " + std::to_string(node.input_size()) + ".");
  }
}

inline void RequireOutputCount(const NodeProto &node, int expected) {
  if (static_cast<int>(node.output_size()) != expected) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(expected) + " output(s), got " +
                                std::to_string(node.output_size()) + ".");
  }
}

inline const AttributeProto *FindAttribute(const NodeProto &node, const std::string &name) {
  for (size_t i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    if (attr.name().as_string() == name) {
      return &attr;
    }
  }
  return nullptr;
}

inline const GraphProto &GetRequiredGraphAttribute(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' is missing '" +
                                name + "' graph attribute.");
  }
  if (attr->type() != AttributeProto::AttributeType::GRAPH) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a GRAPH.");
  }
  return attr->ref_g();
}

inline int64_t GetAttributeIntOrDefault(const NodeProto &node, const std::string &name,
                                        int64_t fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::INT) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be an INT.");
  }
  return attr->i();
}

inline std::vector<int64_t> GetAttributeIntsOrDefault(const NodeProto &node,
                                                      const std::string &name,
                                                      const std::vector<int64_t> &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::INTS) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be INTS.");
  }
  std::vector<int64_t> values;
  values.reserve(attr->ints().size());
  for (size_t i = 0; i < attr->ints().size(); ++i) {
    values.push_back(attr->ints()[i]);
  }
  return values;
}

inline float GetAttributeFloatOrDefault(const NodeProto &node, const std::string &name,
                                        float fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::FLOAT) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a FLOAT.");
  }
  return attr->f();
}

inline std::string GetAttributeStringOrDefault(const NodeProto &node, const std::string &name,
                                               const std::string &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::STRING) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a STRING.");
  }
  return attr->s().as_string();
}

inline std::string GetRequiredAttributeString(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' is missing '" +
                                name + "' STRING attribute.");
  }
  if (attr->type() != AttributeProto::AttributeType::STRING) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a STRING.");
  }
  return attr->s().as_string();
}

} // namespace detail
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
