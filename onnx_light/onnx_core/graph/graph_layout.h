// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "onnx_proto/onnx.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/// Identifies a node within one scope; persisted IDs are positive and at most 2^53 - 1.
using LayoutNodeId = uint64_t;

/// Stores a route point in scope-local visual units, with y increasing downward.
struct LayoutPoint {
  /// Holds the horizontal coordinate; negative coordinates are valid.
  double x = 0;
  /// Holds the vertical coordinate, increasing downward.
  double y = 0;
};

/// Describes the geometry and ordered value ports of a semantic ONNX node.
struct LayoutNode {
  /// Holds the stable scope-local ID, independently of the node name or list position.
  LayoutNodeId id = 0;
  /// Holds the rectangle's left coordinate.
  double x = 0;
  /// Holds the rectangle's top coordinate.
  double y = 0;
  /// Holds the positive rectangle width in visual units.
  double width = 160;
  /// Holds the positive rectangle height in visual units.
  double height = 64;
  /// Holds the dependency rank, not an index into the semantic node list.
  size_t layer = 0;
  /// Folded constants are virtual port content; their boxes may overlap and do not occupy space.
  bool folded = false;
  /// Preserves declared input order, including empty optional input names.
  std::vector<std::string> inputs;
  /// Preserves declared output order.
  std::vector<std::string> outputs;
};

/// Describes one ordered input dependency, including folded values and outer captures.
struct LayoutEdge {
  /// Holds the producer ID, or zero for an input, initializer or external value.
  LayoutNodeId source = 0;
  /// Holds the consumer's scope-local ID.
  LayoutNodeId target = 0;
  /// Holds the producer's zero-based output port.
  size_t source_port = 0;
  /// Holds the target port; implicit captures follow the declared input ports.
  size_t target_port = 0;
  /// Holds the original ONNX value name without renaming or normalization.
  std::string value;
  /// Marks initializer/Constant content intended for inline rendering at consumer ports.
  bool folded = false;
  /// Marks an implicit or outer-scope dependency.
  bool capture = false;
  /// Stores the orthogonal route from source to target, in the current scope's coordinates.
  std::vector<LayoutPoint> points;
};

/// Reports work performed in this layout scope, without aggregating nested layouts.
struct LayoutStats {
  /// Counts nodes processed by the latest load or local update.
  size_t nodes = 0;
  /// Counts rebuilt input routes.
  size_t edges = 0;
  /// Counts solved scopes, excluding child-layout work.
  size_t scopes = 0;
  /// Counts spatial-index entries visited during collision checks.
  size_t collision_index_visits = 0;
};

/// Bounds occupied, non-folded rectangles; an empty scope has all-zero bounds.
struct LayoutBounds {
  /// Holds the minimum occupied x coordinate, excluding collision padding.
  double left = 0;
  /// Holds the minimum occupied y coordinate.
  double top = 0;
  /// Holds the maximum occupied x+width coordinate.
  double right = 0;
  /// Holds the maximum occupied y+height coordinate.
  double bottom = 0;
};

/**
 * Maintains incremental positions, routes and child layouts for a live graph or function.
 *
 * The caller owns the proto and must keep it alive for the layout's lifetime. Single-argument
 * constructors only bind it; Load initializes indexes and reuses validated persisted geometry.
 * Subsequent semantic edits must use this API. Direct proto edits require Load before further use.
 * Mutations update affected consumers and owning containers, without rebuilding unrelated scopes.
 *
 * Coordinates are scope-local, left-to-right and y-down. Node order, value names and operator
 * attributes are not changed for placement. Persist writes namespaced metadata before
 * serialization. Borrowed node/edge/order views should be copied before mutations. Child pointers
 * must be reacquired after Load, Replace or Erase can destroy the corresponding child layout.
 */
class ONNX_LIGHT_CORE_API GraphLayout {
public:
  /// Names the graph/function metadata entry identifying layout schema version "1".
  static constexpr const char *VersionKey = "onnx_light.layout.version";
  /// Names the stable node ID metadata entry.
  static constexpr const char *IdKey = "onnx_light.layout.id";
  /// Names the locale-independent node "x y width height" metadata entry.
  static constexpr const char *PositionKey = "onnx_light.layout.position";
  /// Names the positioning flag: "1" for automatic positions, "0" for manual pins.
  static constexpr const char *AutomaticKey = "onnx_light.layout.automatic";
  /// Names the bounded scope geometry/capture summary used by non-recursive loads.
  static constexpr const char *ScopeKey = "onnx_light.layout.scope";
  /// Names the sizing flag: "1" for explicit sizes, "0" or absent for inferred sizes.
  static constexpr const char *FixedSizeKey = "onnx_light.layout.fixed_size";
  /// Names the persisted node dependency rank.
  static constexpr const char *RankKey = "onnx_light.layout.rank";
  /// Names the scope's monotonic next-ID allocation point.
  static constexpr const char *NextIdKey = "onnx_light.layout.next_id";
  /// Names the non-cryptographic semantic signature used to detect stale node geometry.
  static constexpr const char *SignatureKey = "onnx_light.layout.signature";
  /// Binds an unloaded graph without taking ownership; requires a subsequent Load.
  explicit GraphLayout(GraphProto &graph);
  /// Binds and loads the graph with the requested child-scope policy.
  GraphLayout(GraphProto &graph, bool recurse);
  /// Binds an unloaded function body without copying it; requires a subsequent Load.
  explicit GraphLayout(FunctionProto &function);
  /// Binds and loads the function body with the requested child-scope policy.
  GraphLayout(FunctionProto &function, bool recurse);
  /// Releases layout indexes and children without destroying the bound proto.
  ~GraphLayout();
  /// Disables copying of layout state bound to a live proto.
  GraphLayout(const GraphLayout &) = delete;
  /// Disables assignment between layouts bound to different live protos.
  GraphLayout &operator=(const GraphLayout &) = delete;

  /**
   * Rebuilds indexes, assigns missing IDs and validates or computes node geometry.
   *
   * Clears diagnostics and invalidates child pointers. With recurse=false, reuses valid persisted
   * child summaries; children without usable summaries are loaded directly. Recoverable metadata
   * problems are reported through Diagnostics. Exhausting the ID range throws std::overflow_error.
   */
  void Load(bool recurse = true);
  /// Writes layout metadata, preserving unrelated keys; optionally persists bound children too.
  void Persist(bool recurse = true);
  /// Appends a node copy, updates affected consumers and returns its scope-local stable ID.
  LayoutNodeId Append(const NodeProto &node);
  /// Inserts a node copy at a semantic list index; throws std::out_of_range if index > size.
  LayoutNodeId Insert(size_t index, const NodeProto &node);
  /// Removes the node and its children, updating consumers; returns false for an unknown ID.
  bool Erase(LayoutNodeId id);
  /// Replaces the node and child scopes while retaining its ID; returns false for an unknown ID.
  bool Replace(LayoutNodeId id, const NodeProto &node);
  /// Rewires a zero-based declared input port; returns false for an unknown ID or invalid port.
  bool RewireInput(LayoutNodeId id, size_t port, const std::string &value);
  /**
   * Pins a node at finite coordinates with magnitude <= 1e9, repairing affected consumers.
   * Returns false for an unknown ID, invalid coordinates, or a rejected dependency/collision
   * constraint; constraint rejection also appends a diagnostic. Folded boxes may overlap.
   */
  bool SetPosition(LayoutNodeId id, double x, double y);
  /**
   * Sets explicit finite dimensions in [1, 1e6], respecting child bounds and updating consumers.
   * Returns false for an unknown ID or invalid dimensions.
   */
  bool SetSize(LayoutNodeId id, double width, double height);
  /// Returns a borrowed geometry view, or nullptr for an unknown ID.
  const LayoutNode *Get(LayoutNodeId id) const;
  /// Returns borrowed stable IDs in the semantic node-list order, never dependency-sorted.
  const std::vector<LayoutNodeId> &Order() const;
  /// Returns borrowed input routes in port order; an unknown ID returns an empty vector.
  const std::vector<LayoutEdge> &Edges(LayoutNodeId consumer) const;
  /// Returns borrowed child layouts in attribute order; unknown/summary-only nodes yield none.
  std::vector<GraphLayout *> Children(LayoutNodeId container) const;
  /// Returns recovery and rejected-placement messages accumulated since the most recent Load.
  const std::vector<std::string> &Diagnostics() const;
  /// Returns work counters for the latest load or local update in this scope.
  LayoutStats Stats() const;
  /// Returns the occupied rectangle extent, excluding folded content and collision padding.
  LayoutBounds Bounds() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
