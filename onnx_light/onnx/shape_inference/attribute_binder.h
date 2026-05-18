// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "onnx/common/visitor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace internal { // internal/private API

/// Maps formal attribute names to concrete attribute values from a call-site node.
using AttributeMap = std::unordered_map<std::string, const AttributeProto *>;

/// Binds attribute references in a function body to call-site attribute values.
///
/// A bound attribute keeps the original attribute name from the callee function and
/// copies only the value from the corresponding call-site attribute. If the call-site
/// does not supply a value for a referenced attribute, the referenced attribute is removed.
class AttributeBinder : public MutableVisitor {
public:
  /// Initializes the binder with a map of call-site attributes indexed by name.
  explicit AttributeBinder(const AttributeMap &attr_map) : attr_map_(attr_map) {}

  /// Updates attributes in a node and recursively updates attributes in subgraphs.
  ///
  /// Binding a formal attribute parameter may remove an attribute from a node
  /// when the call-site omits that parameter. This requires processing at node scope
  /// (not at attribute scope) because attributes may be erased from the node list.
  void VisitNode(NodeProto *node) override {
    auto &attributes = node->attribute();
    for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
      auto &attr = *attr_iter;
      if (!attr.ref_attr_name().empty()) {
        // Attribute-references must be replaced by the corresponding attribute-value in the
        // call-node if the call-node contains the attribute. Otherwise, this attribute must be
        // removed.
        auto it = attr_map_.find(attr.ref_attr_name().as_string());
        if (it != attr_map_.end()) {
          const AttributeProto *replacement = it->second;
          // Copy value of attribute, but retain original name:
          std::string name = attr.name().as_string();
          std::string serialized;
          replacement->SerializeToString(serialized);
          AttributeProto rebound;
          rebound.ParseFromString(serialized);
          rebound.set_name(name);
          attr = std::move(rebound);
          ++attr_iter;
        } else {
          attr_iter = attributes.erase(attr_iter);
        }
      } else {
        // For regular attributes, we process subgraphs, if present, recursively.
        VisitAttribute(&attr);
        ++attr_iter;
      }
    }
  }

  /// Binds all attribute references in a callee function using a call-site node.
  ///
  /// @param callnode Node that provides concrete attribute values.
  /// @param callee Function body to update in place.
  static void BindAttributes(const NodeProto &callnode, FunctionProto &callee) {
    AttributeMap map;
    for (const auto &attr : callnode.attribute()) {
      map[attr.name().as_string()] = &attr;
    }
    AttributeBinder attr_binder(map);
    attr_binder.VisitFunction(&callee);
  }

private:
  const AttributeMap &attr_map_;
};

} // namespace internal
} // namespace ONNX_LIGHT_NAMESPACE
