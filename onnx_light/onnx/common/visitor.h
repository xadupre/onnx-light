// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file visitor.h
 * @brief Read-only and mutable visitor base classes for ONNX proto objects.
 *
 * Provides two CRTP-style base structs — @c Visitor and @c MutableVisitor —
 * for traversing the ONNX proto object graph (graphs, functions, nodes, and
 * attributes).  Override the @c ProcessX virtual methods to react to each
 * element; override @c VisitX to change the traversal itself.
 *
 * This header is adapted from @c onnx/common/visitor.h for the onnx-light
 * proto API (@c ref_* accessors instead of the standard protobuf @c mutable_*
 * pattern).
 */

#pragma once
#include "onnx/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace internal {

/**
 * @brief Read-only visitor for ONNX proto objects.
 *
 * Covers Nodes, Graphs, Attributes, and Functions.  Each @c VisitX method
 * calls the corresponding @c ProcessX virtual method; if @c ProcessX returns
 * @c true the traversal continues into the children of the visited element.
 *
 * Subclasses should override one or more @c ProcessX methods to observe
 * elements during traversal, and may override @c VisitX methods to change
 * how children are enumerated.
 */
struct Visitor {
  /**
   * @brief Visits all nodes of @p graph if @c ProcessGraph returns @c true.
   * @param graph Graph to traverse.
   */
  virtual void VisitGraph(const GraphProto &graph) {
    if (ProcessGraph(graph))
      for (const auto &node : graph.ref_node())
        VisitNode(node);
  }

  /**
   * @brief Visits all nodes of @p function if @c ProcessFunction returns @c true.
   * @param function Function to traverse.
   */
  virtual void VisitFunction(const FunctionProto &function) {
    if (ProcessFunction(function))
      for (const auto &node : function.ref_node())
        VisitNode(node);
  }

  /**
   * @brief Visits all attributes of @p node if @c ProcessNode returns @c true.
   * @param node Node to traverse.
   */
  virtual void VisitNode(const NodeProto &node) {
    if (ProcessNode(node)) {
      for (const auto &attr : node.ref_attribute()) {
        VisitAttribute(attr);
      }
    }
  }

  /**
   * @brief Visits graph-valued attribute sub-graphs if @c ProcessAttribute returns @c true.
   * @param attr Attribute to traverse.
   */
  virtual void VisitAttribute(const AttributeProto &attr) {
    if (ProcessAttribute(attr)) {
      if (attr.has_g()) {
        VisitGraph(attr.ref_g());
      }
      for (const auto &graph : attr.ref_graphs())
        VisitGraph(graph);
    }
  }

  /**
   * @brief Called for each graph; returns @c true to visit child nodes.
   * @param graph The current graph.
   * @returns @c true to descend into child nodes; @c false to skip them.
   */
  virtual bool ProcessGraph(const GraphProto & /*graph*/) { return true; }

  /**
   * @brief Called for each function; returns @c true to visit child nodes.
   * @param function The current function.
   * @returns @c true to descend into child nodes; @c false to skip them.
   */
  virtual bool ProcessFunction(const FunctionProto & /*function*/) { return true; }

  /**
   * @brief Called for each node; returns @c true to visit child attributes.
   * @param node The current node.
   * @returns @c true to descend into child attributes; @c false to skip them.
   */
  virtual bool ProcessNode(const NodeProto & /*node*/) { return true; }

  /**
   * @brief Called for each attribute; returns @c true to visit sub-graphs.
   * @param attr The current attribute.
   * @returns @c true to descend into sub-graphs; @c false to skip them.
   */
  virtual bool ProcessAttribute(const AttributeProto & /*attr*/) { return true; }

  virtual ~Visitor() = default;
};

/**
 * @brief Mutable visitor for ONNX proto objects.
 *
 * Identical in structure to @c Visitor but receives non-const pointers,
 * allowing in-place modification of the visited elements.  Subclasses may
 * override @c ProcessX methods to mutate elements or @c VisitX methods to
 * change traversal order.
 */
struct MutableVisitor {
  /**
   * @brief Visits all nodes of @p graph if @c ProcessGraph returns @c true.
   * @param graph Mutable pointer to the graph to traverse.
   */
  virtual void VisitGraph(GraphProto *graph) {
    if (ProcessGraph(graph))
      for (auto &node : graph->ref_node())
        VisitNode(&node);
  }

  /**
   * @brief Visits all nodes of @p function if @c ProcessFunction returns @c true.
   * @param function Mutable pointer to the function to traverse.
   */
  virtual void VisitFunction(FunctionProto *function) {
    if (ProcessFunction(function))
      for (auto &node : function->ref_node())
        VisitNode(&node);
  }

  /**
   * @brief Visits all attributes of @p node if @c ProcessNode returns @c true.
   * @param node Mutable pointer to the node to traverse.
   */
  virtual void VisitNode(NodeProto *node) {
    if (ProcessNode(node)) {
      for (auto &attr : node->ref_attribute()) {
        VisitAttribute(&attr);
      }
    }
  }

  /**
   * @brief Visits graph-valued attribute sub-graphs if @c ProcessAttribute returns @c true.
   * @param attr Mutable pointer to the attribute to traverse.
   */
  virtual void VisitAttribute(AttributeProto *attr) {
    if (ProcessAttribute(attr)) {
      if (attr->has_g()) {
        VisitGraph(&attr->ref_g());
      }
      for (auto &graph : attr->ref_graphs())
        VisitGraph(&graph);
    }
  }

  /**
   * @brief Called for each graph; returns @c true to visit child nodes.
   * @param graph Mutable pointer to the current graph.
   * @returns @c true to descend into child nodes; @c false to skip them.
   */
  virtual bool ProcessGraph(GraphProto * /*graph*/) { return true; }

  /**
   * @brief Called for each function; returns @c true to visit child nodes.
   * @param function Mutable pointer to the current function.
   * @returns @c true to descend into child nodes; @c false to skip them.
   */
  virtual bool ProcessFunction(FunctionProto * /*function*/) { return true; }

  /**
   * @brief Called for each node; returns @c true to visit child attributes.
   * @param node Mutable pointer to the current node.
   * @returns @c true to descend into child attributes; @c false to skip them.
   */
  virtual bool ProcessNode(NodeProto * /*node*/) { return true; }

  /**
   * @brief Called for each attribute; returns @c true to visit sub-graphs.
   * @param attr Mutable pointer to the current attribute.
   * @returns @c true to descend into sub-graphs; @c false to skip them.
   */
  virtual bool ProcessAttribute(AttributeProto * /*attr*/) { return true; }

  virtual ~MutableVisitor() = default;
};

} // namespace internal
} // namespace ONNX_LIGHT_NAMESPACE
