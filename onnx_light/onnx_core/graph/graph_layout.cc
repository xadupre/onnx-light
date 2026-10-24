// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
#include "onnx_core/graph/graph_layout.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {
constexpr double CoordinateLimit = 1e9;
constexpr size_t DepthLimit = 64;
constexpr LayoutNodeId IdLimit = 9007199254740991ULL;
constexpr size_t SummaryLimit = 1024 * 1024;
constexpr double NodePadding = 24;
constexpr double LayerGap = 80;
constexpr double PortPitch = 18;
constexpr double HeaderHeight = 48;
constexpr double ContainerPadding = 48;
constexpr double LabelBaseWidth = 80;
constexpr double CharacterWidth = 8;
constexpr double MinNodeWidth = 160;
constexpr double MaxNodeWidth = 480;
constexpr double FoldedNodeWidth = 64;
constexpr double FoldedNodeHeight = 24;
constexpr double ExternalEdgeLength = 32;
constexpr double DimensionLimit = 1e6;

template <class Proto> std::string_view Metadata(const Proto &proto, const char *key) {
  for (const auto &entry : proto.metadata_props())
    if (entry.key() == key)
      return std::string_view(entry.value().data(), entry.value().size());
  return {};
}

template <class Proto> void SetMetadata(Proto &proto, const char *key, const std::string &value) {
  auto &entries = *proto.mutable_metadata_props();
  bool found = false;
  for (auto it = entries.begin(); it != entries.end();) {
    if (it->key() != key) {
      ++it;
    } else if (!found) {
      it->set_value(value);
      found = true;
      ++it;
    } else {
      it = entries.erase(it);
    }
  }
  if (!found) {
    auto *entry = proto.add_metadata_props();
    entry->set_key(key);
    entry->set_value(value);
  }
}

void ClearNodeLayoutMetadata(NodeProto &node) {
  auto &metadata = *node.mutable_metadata_props();
  for (auto it = metadata.begin(); it != metadata.end();) {
    const auto &key = it->key();
    if (key == GraphLayout::IdKey || key == GraphLayout::PositionKey ||
        key == GraphLayout::AutomaticKey || key == GraphLayout::FixedSizeKey ||
        key == GraphLayout::RankKey || key == GraphLayout::SignatureKey)
      it = metadata.erase(it);
    else
      ++it;
  }
}

bool ValidCoordinate(double value) {
  return std::isfinite(value) && std::abs(value) <= CoordinateLimit;
}

std::string Signature(const NodeProto &node) {
  uint64_t hash = 14695981039346656037ULL;
  auto append = [&](std::string_view value) {
    for (char character : value) {
      hash ^= static_cast<unsigned char>(character);
      hash *= 1099511628211ULL;
    }
    hash ^= 255;
    hash *= 1099511628211ULL;
  };
  auto append_summary = [&](const GraphProto &graph) {
    bool found = false;
    for (const auto &entry : graph.metadata_props()) {
      if (entry.key() == GraphLayout::ScopeKey) {
        const std::string_view summary(entry.value().data(), entry.value().size());
        append(summary.size() <= SummaryLimit ? summary : std::string_view{});
        found = true;
      }
    }
    if (!found)
      append({});
  };
  append(node.op_type());
  append(node.domain());
  append(std::to_string(node.input().size()));
  for (const auto &input : node.input())
    append(input);
  append(std::to_string(node.output().size()));
  for (const auto &output : node.output())
    append(output);
  for (const auto &attribute : node.attribute()) {
    if (attribute.type() == AttributeProto::GRAPH && attribute.has_g()) {
      append(attribute.name());
      append_summary(attribute.g());
      append(std::to_string(attribute.g().node().size()));
    } else if (attribute.type() == AttributeProto::GRAPHS) {
      append(attribute.name());
      append(std::to_string(attribute.graphs().size()));
      for (const auto &graph : attribute.graphs()) {
        append_summary(graph);
        append(std::to_string(graph.node().size()));
      }
    }
  }
  return std::to_string(hash);
}
} // namespace

struct GraphLayout::Impl {
  struct ChildSummary {
    double left = 0, top = 0, right = 0, bottom = 0;
    std::set<std::string> external;
  };
  struct State {
    LayoutNode box;
    NodeProto *proto = nullptr;
    bool pinned = false;
    bool persisted_automatic = false;
    bool fixed_size = false;
    bool placed = false;
    std::vector<std::string> dependencies;
    std::vector<std::unique_ptr<GraphLayout>> children;
    std::vector<ChildSummary> summaries;
    std::vector<LayoutEdge> edges;
  };
  std::variant<GraphProto *, FunctionProto *> scope;
  size_t depth = 0;
  LayoutNodeId next_id = 1;
  std::unordered_map<LayoutNodeId, State> states;
  std::vector<LayoutNodeId> order;
  struct Producer {
    LayoutNodeId id;
    size_t port;
  };
  std::unordered_map<std::string, Producer> producers;
  std::unordered_map<std::string, std::set<LayoutNodeId>> consumers;
  std::unordered_set<std::string> local_values;
  std::unordered_set<std::string> initializers;
  std::unordered_set<std::string> graph_outputs;
  std::unordered_set<LayoutNodeId> reserved_ids;
  struct HorizontalNode {
    double left, right, top, bottom, max_right, min_top, max_bottom;
    LayoutNodeId id;
    int height = 1;
    std::unique_ptr<HorizontalNode> lower, upper;
  };
  std::unique_ptr<HorizontalNode> horizontal;
  std::map<double, std::set<LayoutNodeId>> left_edges, top_edges, right_edges, bottom_edges;
  std::set<std::string> external;
  std::set<std::string> external_changes;
  std::vector<std::string> diagnostics;
  LayoutStats stats;
  std::function<void()> changed;
  bool loading = false;
  bool trusted_positions = false;
  bool recursive = true;

  explicit Impl(GraphProto &graph) : scope(&graph) {}
  explicit Impl(FunctionProto &function) : scope(&function) {}

  auto &Nodes() {
    return std::visit([](auto *proto) -> auto & { return *proto->mutable_node(); }, scope);
  }

  std::string_view ScopeMetadata(const char *key) const {
    return std::visit([&](const auto *proto) { return Metadata(*proto, key); }, scope);
  }

  void SetScopeMetadata(const char *key, const std::string &value) {
    std::visit([&](auto *proto) { SetMetadata(*proto, key, value); }, scope);
  }

  void IndexScopeValues(const GraphProto &graph) {
    for (const auto &input : graph.input())
      local_values.insert(input.name());
    for (const auto &output : graph.output())
      graph_outputs.insert(output.name());
    for (const auto &tensor : graph.initializer()) {
      local_values.insert(tensor.name());
      initializers.insert(tensor.name());
    }
    for (const auto &tensor : graph.sparse_initializer()) {
      local_values.insert(tensor.values().name());
      initializers.insert(tensor.values().name());
    }
    for (const auto &tensor : graph.extended_initializer()) {
      local_values.insert(tensor.name());
      initializers.insert(tensor.name());
    }
  }

  void IndexScopeValues(const FunctionProto &function) {
    for (const auto &input : function.input())
      local_values.insert(input);
    for (const auto &output : function.output())
      graph_outputs.insert(output);
  }

  void Diagnose(const std::string &message) {
    if (diagnostics.size() < 256)
      diagnostics.push_back(message);
  }

  void UpdateExternal(const std::string &value) {
    if (value.empty())
      return;
    const bool used = consumers.count(value) || graph_outputs.count(value);
    const bool changed = used && !producers.count(value) && !local_values.count(value)
                             ? external.insert(value).second
                             : external.erase(value) != 0;
    if (changed && !external_changes.insert(value).second)
      external_changes.erase(value);
  }

  std::pair<double, double> Span() const {
    if (left_edges.empty())
      return {0, 0};
    return {right_edges.rbegin()->first - left_edges.begin()->first,
            bottom_edges.rbegin()->first - top_edges.begin()->first};
  }

  static int HorizontalHeight(const std::unique_ptr<HorizontalNode> &node) {
    return node ? node->height : 0;
  }

  static bool KeyLess(double left, LayoutNodeId id, const HorizontalNode &node) {
    return left < node.left || (left == node.left && id < node.id);
  }

  static void UpdateHorizontal(HorizontalNode &node) {
    node.height = 1 + std::max(HorizontalHeight(node.lower), HorizontalHeight(node.upper));
    node.max_right = node.right;
    node.min_top = node.top;
    node.max_bottom = node.bottom;
    if (node.lower) {
      node.max_right = std::max(node.max_right, node.lower->max_right);
      node.min_top = std::min(node.min_top, node.lower->min_top);
      node.max_bottom = std::max(node.max_bottom, node.lower->max_bottom);
    }
    if (node.upper) {
      node.max_right = std::max(node.max_right, node.upper->max_right);
      node.min_top = std::min(node.min_top, node.upper->min_top);
      node.max_bottom = std::max(node.max_bottom, node.upper->max_bottom);
    }
  }

  static void RotateHorizontalRight(std::unique_ptr<HorizontalNode> &root) {
    auto pivot = std::move(root->lower);
    root->lower = std::move(pivot->upper);
    UpdateHorizontal(*root);
    pivot->upper = std::move(root);
    UpdateHorizontal(*pivot);
    root = std::move(pivot);
  }

  static void RotateHorizontalLeft(std::unique_ptr<HorizontalNode> &root) {
    auto pivot = std::move(root->upper);
    root->upper = std::move(pivot->lower);
    UpdateHorizontal(*root);
    pivot->lower = std::move(root);
    UpdateHorizontal(*pivot);
    root = std::move(pivot);
  }

  static void BalanceHorizontal(std::unique_ptr<HorizontalNode> &root) {
    if (!root)
      return;
    UpdateHorizontal(*root);
    const int balance = HorizontalHeight(root->lower) - HorizontalHeight(root->upper);
    if (balance > 1) {
      if (HorizontalHeight(root->lower->lower) < HorizontalHeight(root->lower->upper))
        RotateHorizontalLeft(root->lower);
      RotateHorizontalRight(root);
    } else if (balance < -1) {
      if (HorizontalHeight(root->upper->upper) < HorizontalHeight(root->upper->lower))
        RotateHorizontalRight(root->upper);
      RotateHorizontalLeft(root);
    }
  }

  void InsertHorizontal(std::unique_ptr<HorizontalNode> &root, double left, double right,
                        double top, double bottom, LayoutNodeId id) {
    if (!root) {
      root = std::make_unique<HorizontalNode>(
          HorizontalNode{left, right, top, bottom, right, top, bottom, id, 1, nullptr, nullptr});
      return;
    }
    if (KeyLess(left, id, *root)) {
      InsertHorizontal(root->lower, left, right, top, bottom, id);
    } else {
      InsertHorizontal(root->upper, left, right, top, bottom, id);
    }
    BalanceHorizontal(root);
  }

  void EraseHorizontal(std::unique_ptr<HorizontalNode> &root, double left, LayoutNodeId id) {
    if (!root)
      return;
    if (root->left == left && root->id == id) {
      if (!root->lower) {
        auto upper = std::move(root->upper);
        root = std::move(upper);
      } else if (!root->upper) {
        auto lower = std::move(root->lower);
        root = std::move(lower);
      } else {
        const HorizontalNode *successor = root->upper.get();
        while (successor->lower)
          successor = successor->lower.get();
        root->left = successor->left;
        root->right = successor->right;
        root->top = successor->top;
        root->bottom = successor->bottom;
        root->id = successor->id;
        EraseHorizontal(root->upper, root->left, root->id);
      }
    } else if (KeyLess(left, id, *root)) {
      EraseHorizontal(root->lower, left, id);
    } else {
      EraseHorizontal(root->upper, left, id);
    }
    BalanceHorizontal(root);
  }

  template <typename Callback>
  bool QueryHorizontal(const HorizontalNode *node, double left, double right, double top,
                       double bottom, Callback &&callback) {
    if (!node || node->max_right <= left || node->min_top >= bottom || node->max_bottom <= top)
      return false;
    if (node->lower && QueryHorizontal(node->lower.get(), left, right, top, bottom, callback))
      return true;
    ++stats.collision_index_visits;
    if (node->left < right && node->right > left && node->top < bottom && node->bottom > top &&
        callback(node->id))
      return true;
    return node->left < right &&
           QueryHorizontal(node->upper.get(), left, right, top, bottom, callback);
  }

  void RemoveOccupied(const State &state) {
    auto remove_extent = [&](auto &index, double value) {
      auto it = index.find(value);
      if (it != index.end()) {
        it->second.erase(state.box.id);
        if (it->second.empty())
          index.erase(it);
      }
    };
    remove_extent(right_edges, state.box.x + state.box.width);
    remove_extent(bottom_edges, state.box.y + state.box.height);
    remove_extent(left_edges, state.box.x);
    remove_extent(top_edges, state.box.y);
    EraseHorizontal(horizontal, state.box.x - NodePadding, state.box.id);
  }

  void IndexOccupied(const State &state) {
    const auto &box = state.box;
    right_edges[box.x + box.width].insert(box.id);
    bottom_edges[box.y + box.height].insert(box.id);
    left_edges[box.x].insert(box.id);
    top_edges[box.y].insert(box.id);
    InsertHorizontal(horizontal, box.x - NodePadding, box.x + box.width + NodePadding,
                     box.y - NodePadding, box.y + box.height + NodePadding, box.id);
  }

  void RequireAvailableId() {
    while (next_id <= IdLimit && (states.count(next_id) || reserved_ids.count(next_id)))
      ++next_id;
    if (next_id > IdLimit)
      throw std::overflow_error("Layout node id space exhausted.");
  }

  LayoutNodeId Identify(NodeProto &node) {
    const auto text = Metadata(node, GraphLayout::IdKey);
    LayoutNodeId id = 0;
    if (!text.empty()) {
      if (text.size() <= 16) {
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), id);
        if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
          id = 0;
      }
      if (!id || id > IdLimit || states.count(id)) {
        Diagnose("Invalid or duplicate layout node id; assigned a fresh id.");
        id = 0;
      }
    }
    if (!id) {
      RequireAvailableId();
      id = next_id++;
    }
    // Choosing small fresh ids also handles stale, maximal persisted ids.
    SetMetadata(node, GraphLayout::IdKey, std::to_string(id));
    return id;
  }

  void LoadPosition(State &state) {
    const auto text = Metadata(*state.proto, GraphLayout::PositionKey);
    if (text.empty() || !trusted_positions)
      return;
    const auto signature = Metadata(*state.proto, GraphLayout::SignatureKey);
    if (!signature.empty() && signature != Signature(*state.proto)) {
      Diagnose("Stale layout signature; recomputed node geometry.");
      return;
    }
    double x = 0, y = 0, width = 0, height = 0;
    std::istringstream stream(text.size() <= 256 ? std::string(text) : "");
    stream.imbue(std::locale::classic());
    if (!(stream >> x >> y >> width >> height) || !(stream >> std::ws).eof() ||
        !ValidCoordinate(x) || !ValidCoordinate(y) || !std::isfinite(width) ||
        !std::isfinite(height) || width < 1 || height < 1 || width > DimensionLimit ||
        height > DimensionLimit) {
      Diagnose("Invalid layout position; computed a replacement.");
      return;
    }
    state.box.x = x;
    state.box.y = y;
    state.box.width = width;
    state.box.height = height;
    state.pinned = true;
    const auto automatic = Metadata(*state.proto, GraphLayout::AutomaticKey);
    state.persisted_automatic = automatic == "1";
    if (!automatic.empty() && automatic != "0" && automatic != "1")
      Diagnose("Invalid layout positioning mode; preserved the position as pinned.");
    const auto fixed = Metadata(*state.proto, GraphLayout::FixedSizeKey);
    state.fixed_size = fixed == "1";
    if (!fixed.empty() && fixed != "0" && fixed != "1")
      Diagnose("Invalid layout sizing mode; used automatic dimensions.");
    const auto rank = Metadata(*state.proto, GraphLayout::RankKey);
    if (!rank.empty()) {
      size_t layer = 0;
      const auto bounded = rank.substr(0, 17);
      const auto result = std::from_chars(bounded.data(), bounded.data() + bounded.size(), layer);
      if (rank.size() > 16 || result.ec != std::errc() ||
          result.ptr != bounded.data() + bounded.size() || layer >= Nodes().size()) {
        Diagnose("Invalid persisted layout rank; recomputed the rank.");
      } else {
        state.box.layer = layer;
      }
    }
  }

  void MakeChildren(LayoutNodeId id) {
    State &state = states.at(id);
    state.children.clear();
    state.summaries.clear();
    for (auto &attribute : *state.proto->mutable_attribute()) {
      auto add = [&](GraphProto &child_graph) {
        if (!recursive && Metadata(child_graph, GraphLayout::VersionKey) == "1") {
          for (const auto &entry : child_graph.metadata_props()) {
            if (entry.key() != GraphLayout::ScopeKey)
              continue;
            ChildSummary summary;
            size_t count = 0;
            const std::string text =
                entry.value().size() <= SummaryLimit ? std::string(entry.value()) : "";
            std::istringstream stream(text);
            stream.imbue(std::locale::classic());
            std::string format;
            bool valid = static_cast<bool>(stream >> format >> summary.left >> summary.top >>
                                           summary.right >> summary.bottom >> count) &&
                         format == "2";
            valid = valid && std::isfinite(summary.right) && std::isfinite(summary.bottom) &&
                    ValidCoordinate(summary.left) && ValidCoordinate(summary.top) &&
                    std::abs(summary.right) <= CoordinateLimit + DimensionLimit &&
                    std::abs(summary.bottom) <= CoordinateLimit + DimensionLimit &&
                    summary.right >= summary.left && summary.bottom >= summary.top &&
                    count <= 65536;
            for (size_t i = 0; valid && i < count; ++i) {
              std::string value;
              valid = static_cast<bool>(stream >> std::quoted(value)) && !value.empty();
              if (valid)
                summary.external.insert(std::move(value));
            }
            if (valid && (stream >> std::ws).eof()) {
              state.summaries.push_back(std::move(summary));
              return;
            }
            Diagnose("Invalid child layout summary; ignored metadata entry.");
          }
        }
        if (depth >= DepthLimit) {
          Diagnose("Layout subgraph depth limit exceeded.");
          return;
        }
        auto child = std::make_unique<GraphLayout>(child_graph);
        child->impl_->depth = depth + 1;
        child->Load();
        child->impl_->changed = [this, id]() {
          const auto previous_span = Span();
          external_changes.clear();
          stats = {};
          Refresh(id);
          Solve({id});
          if (Span() != previous_span || !external_changes.empty())
            Notify();
        };
        state.children.push_back(std::move(child));
      };
      if (attribute.type() == AttributeProto::GRAPH && attribute.has_g())
        add(*attribute.mutable_g());
      if (attribute.type() == AttributeProto::GRAPHS)
        for (auto &child : *attribute.mutable_graphs())
          add(child);
    }
  }

  void Unindex(LayoutNodeId id, std::set<LayoutNodeId> &affected) {
    State &state = states.at(id);
    for (const auto &input : state.dependencies) {
      auto it = consumers.find(input);
      if (it != consumers.end()) {
        it->second.erase(id);
        if (it->second.empty())
          consumers.erase(it);
      }
      UpdateExternal(input);
    }
    for (const auto &output : state.box.outputs) {
      auto producer = producers.find(output);
      if (producer != producers.end() && producer->second.id == id)
        producers.erase(producer);
      auto users = consumers.find(output);
      if (users != consumers.end())
        affected.insert(users->second.begin(), users->second.end());
      UpdateExternal(output);
    }
  }

  void Refresh(LayoutNodeId id) {
    State &state = states.at(id);
    RemoveOccupied(state);
    std::set<LayoutNodeId> ignored;
    Unindex(id, ignored);
    state.box.inputs.clear();
    state.box.outputs.clear();
    state.dependencies.clear();
    for (const auto &input : state.proto->input())
      state.box.inputs.emplace_back(input);
    for (const auto &output : state.proto->output())
      state.box.outputs.emplace_back(output);
    state.dependencies = state.box.inputs;
    std::unordered_set<std::string> dependencies(state.dependencies.begin(),
                                                 state.dependencies.end());
    for (const auto &child : state.children)
      for (const auto &capture : child->impl_->external)
        if (dependencies.insert(capture).second)
          state.dependencies.push_back(capture);
    for (const auto &summary : state.summaries)
      for (const auto &capture : summary.external)
        if (dependencies.insert(capture).second)
          state.dependencies.push_back(capture);
    for (const auto &input : state.dependencies)
      if (!input.empty()) {
        consumers[input].insert(id);
        UpdateExternal(input);
      }
    for (size_t port = 0; port < state.box.outputs.size(); ++port) {
      const auto &output = state.box.outputs[port];
      if (output.empty())
        continue;
      auto [it, inserted] = producers.emplace(output, Producer{id, port});
      if (!inserted && it->second.id != id) {
        std::string name = output.substr(0, 128);
        for (char &character : name)
          if (static_cast<unsigned char>(character) < 32 ||
              static_cast<unsigned char>(character) >= 127)
            character = '?';
        Diagnose("Multiple producers for value: " + name + (output.size() > 128 ? "..." : ""));
      }
      UpdateExternal(output);
    }
    state.box.folded = state.proto->op_type() == "Constant" &&
                       (state.proto->domain().empty() || state.proto->domain() == "ai.onnx");
    if (!state.fixed_size) {
      state.box.width = std::clamp(LabelBaseWidth + CharacterWidth * state.proto->op_type().size(),
                                   MinNodeWidth, MaxNodeWidth);
      state.box.height =
          HeaderHeight + PortPitch * std::max(state.dependencies.size(), state.box.outputs.size());
      if (state.box.folded) {
        state.box.width = FoldedNodeWidth;
        state.box.height = FoldedNodeHeight;
      }
    }
    FitChildBounds(state);
  }

  void FitChildBounds(State &state) {
    double children_height = ContainerPadding;
    for (const auto &child : state.children) {
      const auto &nested = *child->impl_;
      const double left = nested.left_edges.empty() ? 0 : nested.left_edges.begin()->first;
      const double top = nested.top_edges.empty() ? 0 : nested.top_edges.begin()->first;
      const double right = nested.right_edges.empty() ? 0 : nested.right_edges.rbegin()->first;
      const double bottom = nested.bottom_edges.empty() ? 0 : nested.bottom_edges.rbegin()->first;
      state.box.width = std::max(state.box.width, right - left + ContainerPadding);
      children_height += bottom - top + ContainerPadding;
    }
    for (const auto &summary : state.summaries) {
      state.box.width = std::max(state.box.width, summary.right - summary.left + ContainerPadding);
      children_height += summary.bottom - summary.top + ContainerPadding;
    }
    if (!state.children.empty() || !state.summaries.empty())
      state.box.height = std::max(state.box.height, children_height);
    if (state.box.width > DimensionLimit || state.box.height > DimensionLimit) {
      Diagnose("Layout dimensions exceeded the supported bounds; clipped the node box.");
      state.box.width = std::min(state.box.width, DimensionLimit);
      state.box.height = std::min(state.box.height, DimensionLimit);
    }
  }

  void Route(LayoutNodeId id) {
    auto &state = states.at(id);
    state.edges.clear();
    for (size_t port = 0; port < state.dependencies.size(); ++port) {
      const auto &value = state.dependencies[port];
      if (value.empty())
        continue;
      LayoutEdge edge;
      edge.target = id;
      edge.target_port = port;
      edge.value = value;
      edge.capture = port >= state.box.inputs.size() || external.count(value);
      edge.folded = initializers.count(value);
      LayoutPoint end{state.box.x, state.box.y + NodePadding + PortPitch * port};
      auto producer = producers.find(value);
      if (producer != producers.end()) {
        edge.source = producer->second.id;
        edge.source_port = producer->second.port;
        const auto &source = states.at(edge.source).box;
        edge.folded = source.folded;
        LayoutPoint start{source.x + source.width,
                          source.y + NodePadding + PortPitch * edge.source_port};
        const double middle = (start.x + end.x) / 2;
        edge.points = {start, {middle, start.y}, {middle, end.y}, end};
      } else {
        edge.points = {{end.x - ExternalEdgeLength, end.y}, end};
      }
      state.edges.push_back(std::move(edge));
      ++stats.edges;
    }
  }

  bool Collides(const LayoutNode &box) {
    return QueryHorizontal(horizontal.get(), box.x, box.x + box.width, box.y, box.y + box.height,
                           [&](LayoutNodeId other) { return other != box.id; });
  }

  std::pair<bool, double> RequiredX(const State &state) const {
    bool required = false;
    double x = 0;
    for (const auto &input : state.dependencies) {
      auto producer = producers.find(input);
      if (producer == producers.end() || producer->second.id == state.box.id)
        continue;
      const auto &source = states.at(producer->second.id).box;
      if (!source.folded) {
        const double next = source.x + source.width + LayerGap;
        x = required ? std::max(x, next) : next;
        required = true;
      }
    }
    return {required, x};
  }

  void Solve(const std::set<LayoutNodeId> &seeds) {
    std::deque<LayoutNodeId> queue(seeds.begin(), seeds.end());
    std::set<LayoutNodeId> affected;
    while (!queue.empty()) {
      const auto id = queue.front();
      queue.pop_front();
      if (!states.count(id) || !affected.insert(id).second)
        continue;
      for (const auto &output : states.at(id).box.outputs) {
        auto users = consumers.find(output);
        if (users != consumers.end())
          for (auto user : users->second)
            if (!affected.count(user))
              queue.push_back(user);
      }
    }
    std::unordered_map<LayoutNodeId, size_t> pending;
    std::unordered_map<LayoutNodeId, std::set<LayoutNodeId>> dependents;
    for (auto id : affected) {
      RemoveOccupied(states.at(id));
      std::set<LayoutNodeId> sources;
      for (const auto &input : states.at(id).dependencies) {
        auto producer = producers.find(input);
        if (producer != producers.end() && affected.count(producer->second.id))
          sources.insert(producer->second.id);
      }
      pending[id] = sources.size();
      for (auto source : sources)
        dependents[source].insert(id);
      if (sources.empty())
        queue.push_back(id);
    }
    std::unordered_set<LayoutNodeId> reserved;
    if (loading) {
      for (auto id : affected) {
        const State &state = states.at(id);
        const auto [required, minimum_x] = RequiredX(state);
        if (state.pinned && !state.box.folded && (!required || state.box.x >= minimum_x) &&
            !Collides(state.box)) {
          IndexOccupied(state);
          reserved.insert(id);
        }
      }
    }
    auto place = [&](LayoutNodeId id) {
      State &state = states.at(id);
      if (reserved.erase(id))
        RemoveOccupied(state);
      ++stats.nodes;
      size_t layer = 0;
      double x = 0;
      bool has_predecessor = false;
      double barycenter = 0;
      size_t barycenter_count = 0;
      for (size_t port = 0; port < state.dependencies.size(); ++port) {
        const auto &input = state.dependencies[port];
        auto producer = producers.find(input);
        if (producer != producers.end() && producer->second.id != id) {
          const auto &source = states.at(producer->second.id).box;
          if (!source.folded) {
            layer = std::max(layer, source.layer + 1);
            const double next = source.x + source.width + LayerGap;
            x = has_predecessor ? std::max(x, next) : next;
            has_predecessor = true;
            barycenter += source.y + PortPitch * producer->second.port - PortPitch * port;
            ++barycenter_count;
          }
        } else if (producer != producers.end())
          Diagnose("Self-referencing layout dependency.");
      }
      layer = std::min(layer, states.size() - 1);
      if (!has_predecessor && state.placed && state.box.layer == layer)
        x = state.box.x;
      bool repaired_pin = false;
      if (state.pinned && !state.box.folded &&
          ((has_predecessor && state.box.x < x) || Collides(state.box))) {
        Diagnose("Pinned position violated layout constraints; repaired the affected node.");
        if (!has_predecessor || state.box.x >= x)
          x = state.box.x;
        state.pinned = false;
        repaired_pin = true;
      }
      const bool preserve_y = state.placed && state.box.layer == layer;
      state.box.layer = layer;
      if (state.box.folded) {
        if (!state.pinned) {
          state.box.x = std::min(x, CoordinateLimit);
          state.box.y = barycenter_count ? barycenter / barycenter_count : 0;
        }
        state.placed = true;
        return;
      }
      if (!state.pinned) {
        state.box.x = std::min(x, CoordinateLimit);
        double y = preserve_y || repaired_pin
                       ? state.box.y
                       : (barycenter_count ? barycenter / barycenter_count : 0);
        y = std::clamp(y, -CoordinateLimit, CoordinateLimit);
        bool collision;
        do {
          collision = false;
          QueryHorizontal(horizontal.get(), state.box.x, state.box.x + state.box.width, y,
                          y + state.box.height, [&](LayoutNodeId other) {
                            const auto &box = states.at(other).box;
                            y = box.y + box.height + NodePadding;
                            collision = true;
                            return true;
                          });
        } while (collision && y <= CoordinateLimit);
        if (y > CoordinateLimit)
          Diagnose("Layout coordinate space exhausted; clipped the node position.");
        state.box.y = std::min(y, CoordinateLimit);
      }
      IndexOccupied(state);
      state.placed = true;
    };
    std::set<LayoutNodeId> placed;
    while (!queue.empty()) {
      const auto id = queue.front();
      queue.pop_front();
      place(id);
      placed.insert(id);
      if (auto it = dependents.find(id); it != dependents.end())
        for (auto user : it->second)
          if (--pending[user] == 0)
            queue.push_back(user);
    }
    if (placed.size() != affected.size()) {
      Diagnose("Cyclic layout dependencies; laid out the remaining nodes once.");
      for (auto id : affected)
        if (!placed.count(id))
          place(id);
    }
    for (auto id : affected) {
      Route(id);
    }
    stats.scopes = 1;
  }

  void Notify() {
    if (!loading && changed)
      changed();
  }

  LayoutNodeId Add(NodeProto &node) {
    const auto id = Identify(node);
    State &state = states[id];
    state.proto = &node;
    state.box.id = id;
    LoadPosition(state);
    MakeChildren(id);
    Refresh(id);
    return id;
  }
};

GraphLayout::GraphLayout(GraphProto &graph) : impl_(std::make_unique<Impl>(graph)) {}
GraphLayout::GraphLayout(GraphProto &graph, bool recurse) : GraphLayout(graph) { Load(recurse); }
GraphLayout::GraphLayout(FunctionProto &function) : impl_(std::make_unique<Impl>(function)) {}
GraphLayout::GraphLayout(FunctionProto &function, bool recurse) : GraphLayout(function) {
  Load(recurse);
}
GraphLayout::~GraphLayout() = default;

void GraphLayout::Load(bool recurse) {
  auto &impl = *impl_;
  const auto previous_span = impl.Span();
  const auto previous_external = std::move(impl.external);

  const auto reset_for_load = [&](bool recursive) {
    impl.recursive = recursive;
    impl.loading = true;
    impl.states.clear();
    impl.order.clear();
    impl.producers.clear();
    impl.consumers.clear();
    impl.horizontal.reset();
    impl.local_values.clear();
    impl.initializers.clear();
    impl.graph_outputs.clear();
    impl.reserved_ids.clear();
    impl.right_edges.clear();
    impl.bottom_edges.clear();
    impl.left_edges.clear();
    impl.top_edges.clear();
    impl.external.clear();
    impl.external_changes.clear();
    impl.diagnostics.clear();
    impl.stats = {};
    impl.next_id = 1;
  };

  reset_for_load(recurse);
  const auto next = impl.ScopeMetadata(NextIdKey);
  if (!next.empty()) {
    LayoutNodeId next_id = 0;
    const auto bounded = next.substr(0, 17);
    const auto parsed = std::from_chars(bounded.data(), bounded.data() + bounded.size(), next_id);
    if (next.size() <= 16 && parsed.ec == std::errc() &&
        parsed.ptr == bounded.data() + bounded.size() && next_id && next_id <= IdLimit + 1)
      impl.next_id = next_id;
    else
      impl.Diagnose("Invalid persisted next layout id; chose a fresh allocation point.");
  }
  const auto version = impl.ScopeMetadata(VersionKey);
  impl.trusted_positions = version == "1";
  if (!version.empty() && version != "1")
    impl.Diagnose("Unsupported layout metadata version; recomputed geometry.");
  std::visit([&](const auto *proto) { impl.IndexScopeValues(*proto); }, impl.scope);
  for (const auto &node : impl.Nodes()) {
    const auto text = Metadata(node, IdKey);
    LayoutNodeId id = 0;
    if (text.size() <= 16) {
      auto parsed = std::from_chars(text.data(), text.data() + text.size(), id);
      if (parsed.ec == std::errc() && parsed.ptr == text.data() + text.size() && id &&
          id <= IdLimit)
        impl.reserved_ids.insert(id);
    }
  }
  std::set<LayoutNodeId> seeds;
  for (auto &node : impl.Nodes()) {
    auto id = impl.Add(node);
    impl.order.push_back(id);
    seeds.insert(id);
  }
  for (const auto &output : impl.graph_outputs)
    impl.UpdateExternal(output);
  impl.Solve(seeds);
  // Restores the exact saved frame first; subsequent edits can move automatic nodes.
  for (auto &[id, state] : impl.states) {
    (void)id;
    if (state.persisted_automatic)
      state.pinned = false;
  }
  impl.loading = false;
  if (impl.Span() != previous_span || impl.external != previous_external)
    impl.Notify();
}

void GraphLayout::Persist(bool recurse) {
  impl_->SetScopeMetadata(VersionKey, "1");
  impl_->SetScopeMetadata(NextIdKey, std::to_string(impl_->next_id));
  std::ostringstream summary;
  summary.imbue(std::locale::classic());
  summary << std::setprecision(17) << "2 "
          << (impl_->left_edges.empty() ? 0 : impl_->left_edges.begin()->first) << ' '
          << (impl_->top_edges.empty() ? 0 : impl_->top_edges.begin()->first) << ' '
          << (impl_->right_edges.empty() ? 0 : impl_->right_edges.rbegin()->first) << ' '
          << (impl_->bottom_edges.empty() ? 0 : impl_->bottom_edges.rbegin()->first) << ' '
          << impl_->external.size();
  size_t summary_size = 128;
  bool bounded = impl_->external.size() <= 65536;
  for (const auto &value : impl_->external) {
    if (value.size() > (SummaryLimit - std::min(summary_size, SummaryLimit)) / 2) {
      bounded = false;
      break;
    }
    summary_size += value.size() * 2 + 3;
    summary << ' ' << std::quoted(value);
  }
  const std::string summary_text = summary.str();
  if (!bounded || summary_text.size() > SummaryLimit) {
    impl_->Diagnose("Layout capture summary exceeded the metadata size limit.");
    std::visit(
        [](auto *proto) {
          auto &entries = *proto->mutable_metadata_props();
          for (auto it = entries.begin(); it != entries.end();)
            if (it->key() == ScopeKey)
              it = entries.erase(it);
            else
              ++it;
        },
        impl_->scope);
  } else {
    impl_->SetScopeMetadata(ScopeKey, summary_text);
  }
  for (auto &[id, state] : impl_->states) {
    SetMetadata(*state.proto, IdKey, std::to_string(id));
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17) << state.box.x << ' ' << state.box.y << ' ' << state.box.width
           << ' ' << state.box.height;
    SetMetadata(*state.proto, PositionKey, stream.str());
    SetMetadata(*state.proto, AutomaticKey, state.pinned ? "0" : "1");
    SetMetadata(*state.proto, FixedSizeKey, state.fixed_size ? "1" : "0");
    SetMetadata(*state.proto, RankKey, std::to_string(state.box.layer));
    if (recurse)
      for (auto &child : state.children)
        child->Persist();
    SetMetadata(*state.proto, SignatureKey, Signature(*state.proto));
  }
}

LayoutNodeId GraphLayout::Append(const NodeProto &node) {
  return Insert(impl_->order.size(), node);
}

LayoutNodeId GraphLayout::Insert(size_t index, const NodeProto &node) {
  auto &impl = *impl_;
  if (index > impl.order.size())
    throw std::out_of_range("Layout insertion index is out of range.");
  impl.RequireAvailableId();
  impl.stats = {};
  auto &nodes = impl.Nodes();
  nodes.push_back(node);
  // Swaps owning handles, preserving existing node payloads and child-layout proto pointers.
  for (int i = static_cast<int>(nodes.size()) - 1; i > static_cast<int>(index); --i)
    nodes.SwapElements(i, i - 1);
  ClearNodeLayoutMetadata(nodes[index]);
  auto id = impl.Add(nodes[index]);
  impl.order.insert(impl.order.begin() + static_cast<ptrdiff_t>(index), id);
  impl.Solve({id});
  if (impl.states.at(id).persisted_automatic)
    impl.states.at(id).pinned = false;
  impl.Notify();
  return id;
}

bool GraphLayout::Erase(LayoutNodeId id) {
  auto &impl = *impl_;
  if (!impl.states.count(id))
    return false;
  const auto it = std::find(impl.order.begin(), impl.order.end(), id);
  if (it == impl.order.end() || static_cast<size_t>(it - impl.order.begin()) >= impl.Nodes().size())
    throw std::logic_error(
        "Layout node order is inconsistent; call Load() after direct graph edits.");
  const auto index = it - impl.order.begin();
  impl.stats = {};
  std::set<LayoutNodeId> affected;
  impl.Unindex(id, affected);
  impl.RemoveOccupied(impl.states.at(id));
  impl.states.erase(id);
  impl.Nodes().erase(impl.Nodes().begin() + index);
  impl.order.erase(it);
  impl.Solve(affected);
  impl.Notify();
  return true;
}

bool GraphLayout::Replace(LayoutNodeId id, const NodeProto &node) {
  auto &impl = *impl_;
  if (!impl.states.count(id))
    return false;
  impl.stats = {};
  std::set<LayoutNodeId> affected{id};
  impl.Unindex(id, affected);
  auto &state = impl.states.at(id);
  state.children.clear();
  *state.proto = node;
  ClearNodeLayoutMetadata(*state.proto);
  SetMetadata(*state.proto, IdKey, std::to_string(id));
  impl.MakeChildren(id);
  impl.Refresh(id);
  impl.Solve(affected);
  impl.Notify();
  return true;
}

bool GraphLayout::RewireInput(LayoutNodeId id, size_t port, const std::string &value) {
  auto &impl = *impl_;
  if (!impl.states.count(id) || port >= impl.states.at(id).box.inputs.size())
    return false;
  impl.stats = {};
  impl.states.at(id).proto->mutable_input()->Set(port, value);
  impl.Refresh(id);
  impl.Solve({id});
  impl.Notify();
  return true;
}

bool GraphLayout::SetPosition(LayoutNodeId id, double x, double y) {
  auto &impl = *impl_;
  if (!impl.states.count(id) || !ValidCoordinate(x) || !ValidCoordinate(y))
    return false;
  const auto &current = impl.states.at(id).box;
  LayoutNode proposed;
  proposed.id = id;
  proposed.width = current.width;
  proposed.height = current.height;
  proposed.folded = current.folded;
  proposed.x = x;
  proposed.y = y;
  const auto [required, minimum_x] = impl.RequiredX(impl.states.at(id));
  if (!proposed.folded && ((required && x < minimum_x) || impl.Collides(proposed))) {
    impl.Diagnose("Rejected a pinned position that violated layout constraints.");
    return false;
  }
  impl.stats = {};
  auto &state = impl.states.at(id);
  impl.RemoveOccupied(state);
  state.box.x = x;
  state.box.y = y;
  state.pinned = true;
  impl.Solve({id});
  impl.Notify();
  return true;
}

bool GraphLayout::SetSize(LayoutNodeId id, double width, double height) {
  auto &impl = *impl_;
  if (!impl.states.count(id) || !std::isfinite(width) || !std::isfinite(height) || width < 1 ||
      height < 1 || width > DimensionLimit || height > DimensionLimit)
    return false;
  impl.stats = {};
  auto &state = impl.states.at(id);
  impl.RemoveOccupied(state);
  state.box.width = width;
  state.box.height = height;
  state.fixed_size = true;
  impl.FitChildBounds(state);
  impl.Solve({id});
  impl.Notify();
  return true;
}

const LayoutNode *GraphLayout::Get(LayoutNodeId id) const {
  auto it = impl_->states.find(id);
  return it == impl_->states.end() ? nullptr : &it->second.box;
}
const std::vector<LayoutNodeId> &GraphLayout::Order() const { return impl_->order; }
const std::vector<LayoutEdge> &GraphLayout::Edges(LayoutNodeId consumer) const {
  static const std::vector<LayoutEdge> empty;
  auto it = impl_->states.find(consumer);
  return it == impl_->states.end() ? empty : it->second.edges;
}
std::vector<GraphLayout *> GraphLayout::Children(LayoutNodeId container) const {
  std::vector<GraphLayout *> children;
  auto it = impl_->states.find(container);
  if (it != impl_->states.end())
    for (const auto &child : it->second.children)
      children.push_back(child.get());
  return children;
}
const std::vector<std::string> &GraphLayout::Diagnostics() const { return impl_->diagnostics; }
LayoutStats GraphLayout::Stats() const { return impl_->stats; }
LayoutBounds GraphLayout::Bounds() const {
  LayoutBounds bounds;
  if (!impl_->left_edges.empty()) {
    bounds.left = impl_->left_edges.begin()->first;
    bounds.top = impl_->top_edges.begin()->first;
    bounds.right = impl_->right_edges.rbegin()->first;
    bounds.bottom = impl_->bottom_edges.rbegin()->first;
  }
  return bounds;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
