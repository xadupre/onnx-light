// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Visitor classes for readonly and mutable traversal of ONNX proto objects.
// Adapted from the ONNX reference implementation for use in onnx_light.

#pragma once

#include "onnx/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace internal {

// Visitor: A readonly visitor class for ONNX Proto objects.
// VisitX methods invoke ProcessX; if ProcessX returns true the visitor
// recurses into children.
struct Visitor {
  virtual void VisitGraph(const GraphProto &graph) {
    if (ProcessGraph(graph))
      for (const auto &node : graph.ref_node())
        VisitNode(node);
  }

  virtual void VisitFunction(const FunctionProto &function) {
    if (ProcessFunction(function))
      for (const auto &node : function.ref_node())
        VisitNode(node);
  }

  virtual void VisitNode(const NodeProto &node) {
    if (ProcessNode(node)) {
      for (const auto &attr : node.ref_attribute()) {
        VisitAttribute(attr);
      }
    }
  }

  virtual void VisitAttribute(const AttributeProto &attr) {
    if (ProcessAttribute(attr)) {
      if (attr.has_g()) {
        VisitGraph(attr.ref_g());
      }
      for (const auto &graph : attr.ref_graphs())
        VisitGraph(graph);
    }
  }

  virtual bool ProcessGraph(const GraphProto & /*graph*/) { return true; }

  virtual bool ProcessFunction(const FunctionProto & /*function*/) { return true; }

  virtual bool ProcessNode(const NodeProto & /*node*/) { return true; }

  virtual bool ProcessAttribute(const AttributeProto & /*attr*/) { return true; }

  virtual ~Visitor() = default;
};

// MutableVisitor: A version of Visitor that allows mutation of the visited objects.
struct MutableVisitor {
  virtual void VisitGraph(GraphProto *graph) {
    if (ProcessGraph(graph))
      for (auto &node : graph->ref_node())
        VisitNode(&node);
  }

  virtual void VisitFunction(FunctionProto *function) {
    if (ProcessFunction(function))
      for (auto &node : function->ref_node())
        VisitNode(&node);
  }

  virtual void VisitNode(NodeProto *node) {
    if (ProcessNode(node)) {
      for (auto &attr : node->ref_attribute()) {
        VisitAttribute(&attr);
      }
    }
  }

  virtual void VisitAttribute(AttributeProto *attr) {
    if (ProcessAttribute(attr)) {
      if (attr->has_g()) {
        VisitGraph(attr->mutable_g());
      }
      for (auto &graph : attr->ref_graphs())
        VisitGraph(&graph);
    }
  }

  virtual bool ProcessGraph(GraphProto * /*graph*/) { return true; }

  virtual bool ProcessFunction(FunctionProto * /*function*/) { return true; }

  virtual bool ProcessNode(NodeProto * /*node*/) { return true; }

  virtual bool ProcessAttribute(AttributeProto * /*attr*/) { return true; }

  virtual ~MutableVisitor() = default;
};

} // namespace internal
} // namespace ONNX_LIGHT_NAMESPACE
