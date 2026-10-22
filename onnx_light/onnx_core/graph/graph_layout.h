// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "onnx_proto/onnx.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

using LayoutNodeId = uint64_t;

struct LayoutPoint {
  double x = 0, y = 0;
};

struct LayoutNode {
  LayoutNodeId id = 0;
  double x = 0, y = 0, width = 160, height = 64;
  size_t layer = 0;
  bool folded = false;
  std::vector<std::string> inputs, outputs;
};

struct LayoutEdge {
  LayoutNodeId source = 0, target = 0;
  size_t source_port = 0, target_port = 0;
  std::string value;
  bool folded = false, capture = false;
  std::vector<LayoutPoint> points;
};

struct LayoutStats {
  size_t nodes = 0, edges = 0, scopes = 0;
};

/// Maintains an incremental layout of a graph. Mutations must use this API while bound.
class ONNX_LIGHT_CORE_API GraphLayout {
public:
  static constexpr const char *VersionKey = "onnx_light.layout.version";
  static constexpr const char *IdKey = "onnx_light.layout.id";
  static constexpr const char *PositionKey = "onnx_light.layout.position";
  static constexpr const char *AutomaticKey = "onnx_light.layout.automatic";
  static constexpr const char *ScopeKey = "onnx_light.layout.scope";
  static constexpr const char *FixedSizeKey = "onnx_light.layout.fixed_size";
  static constexpr const char *RankKey = "onnx_light.layout.rank";
  static constexpr const char *NextIdKey = "onnx_light.layout.next_id";
  static constexpr const char *SignatureKey = "onnx_light.layout.signature";
  explicit GraphLayout(GraphProto &graph);
  /// Binds and loads the graph with the requested child-scope policy.
  GraphLayout(GraphProto &graph, bool recurse);
  ~GraphLayout();
  GraphLayout(const GraphLayout &) = delete;
  GraphLayout &operator=(const GraphLayout &) = delete;

  /// Loads nodes, optionally reusing persisted child summaries instead of binding child layouts.
  void Load(bool recurse = true);
  /// Writes versioned layout metadata, preserving unrelated metadata.
  void Persist(bool recurse = true);
  LayoutNodeId Append(const NodeProto &node);
  LayoutNodeId Insert(size_t index, const NodeProto &node);
  bool Erase(LayoutNodeId id);
  bool Replace(LayoutNodeId id, const NodeProto &node);
  bool RewireInput(LayoutNodeId id, size_t port, const std::string &value);
  bool SetPosition(LayoutNodeId id, double x, double y);
  bool SetSize(LayoutNodeId id, double width, double height);
  const LayoutNode *Get(LayoutNodeId id) const;
  const std::vector<LayoutNodeId> &Order() const;
  const std::vector<LayoutEdge> &Edges(LayoutNodeId consumer) const;
  std::vector<GraphLayout *> Children(LayoutNodeId container) const;
  const std::vector<std::string> &Diagnostics() const;
  LayoutStats Stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
