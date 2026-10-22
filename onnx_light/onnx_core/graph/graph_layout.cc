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

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {
constexpr double CoordinateLimit = 1e9;
constexpr size_t DepthLimit = 64;
constexpr LayoutNodeId IdLimit = 9007199254740991ULL;
constexpr size_t SummaryLimit = 1024 * 1024;

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
      append(Metadata(attribute.g(), GraphLayout::ScopeKey));
      append(std::to_string(attribute.g().node().size()));
    } else if (attribute.type() == AttributeProto::GRAPHS) {
      append(attribute.name());
      append(std::to_string(attribute.graphs().size()));
      for (const auto &graph : attribute.graphs()) {
        append(Metadata(graph, GraphLayout::ScopeKey));
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
  GraphProto &graph;
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
  struct OccupiedLayer {
    std::map<double, std::set<LayoutNodeId>> positions;
    std::multiset<double> lefts, rights, heights;
  };
  std::map<size_t, OccupiedLayer> occupied;
  std::map<double, std::set<LayoutNodeId>> left_edges, top_edges, right_edges, bottom_edges;
  std::set<std::string> external;
  std::set<std::string> external_changes;
  std::vector<std::string> diagnostics;
  LayoutStats stats;
  std::function<void()> changed;
  bool loading = false;
  bool trusted_positions = false;
  bool recursive = true;

  explicit Impl(GraphProto &g) : graph(g) {}

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
    auto layer = occupied.find(state.box.layer);
    if (layer == occupied.end())
      return;
    auto &index = layer->second;
    auto position = index.positions.find(state.box.y);
    if (position != index.positions.end() && position->second.erase(state.box.id)) {
      auto erase_one = [](auto &values, double value) {
        auto it = values.find(value);
        if (it != values.end())
          values.erase(it);
      };
      erase_one(index.lefts, state.box.x);
      erase_one(index.rights, state.box.x + state.box.width);
      erase_one(index.heights, state.box.height);
      if (position->second.empty())
        index.positions.erase(position);
    }
    if (index.positions.empty())
      occupied.erase(layer);
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
      while (states.count(next_id) || reserved_ids.count(next_id))
        ++next_id;
      if (next_id > IdLimit)
        throw std::overflow_error("Layout node id space exhausted.");
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
        !std::isfinite(height) || width < 1 || height < 1 || width > 1e6 || height > 1e6) {
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
    state.fixed_size = fixed.empty() || fixed == "1";
    if (!fixed.empty() && fixed != "0" && fixed != "1")
      Diagnose("Invalid layout sizing mode; used automatic dimensions.");
    const auto rank = Metadata(*state.proto, GraphLayout::RankKey);
    if (!rank.empty()) {
      size_t layer = 0;
      const auto bounded = rank.substr(0, 17);
      const auto result = std::from_chars(bounded.data(), bounded.data() + bounded.size(), layer);
      if (rank.size() > 16 || result.ec != std::errc() ||
          result.ptr != bounded.data() + bounded.size() || layer > graph.node().size()) {
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
                    std::abs(summary.right) <= CoordinateLimit + 1e6 &&
                    std::abs(summary.bottom) <= CoordinateLimit + 1e6 &&
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
            Diagnose("Invalid child layout summary; loaded the child scope directly.");
            break;
          }
        }
        if (depth >= DepthLimit) {
          Diagnose("Layout subgraph depth limit exceeded.");
          return;
        }
        auto child = std::make_unique<GraphLayout>(child_graph);
        child->impl_->depth = depth + 1;
        child->impl_->changed = [this, id]() {
          const auto previous_span = Span();
          external_changes.clear();
          stats = {};
          Refresh(id);
          Solve({id});
          if (Span() != previous_span || !external_changes.empty())
            Notify();
        };
        child->Load();
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
    for (const auto &child : state.children)
      for (const auto &capture : child->impl_->external)
        if (std::find(state.dependencies.begin(), state.dependencies.end(), capture) ==
            state.dependencies.end())
          state.dependencies.push_back(capture);
    for (const auto &summary : state.summaries)
      for (const auto &capture : summary.external)
        if (std::find(state.dependencies.begin(), state.dependencies.end(), capture) ==
            state.dependencies.end())
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
      state.box.width = std::clamp(80.0 + 8.0 * state.proto->op_type().size(), 160.0, 480.0);
      state.box.height =
          48.0 + 18.0 * std::max(state.dependencies.size(), state.box.outputs.size());
      if (state.box.folded) {
        state.box.width = 64;
        state.box.height = 24;
      }
    }
    double children_height = 48;
    for (const auto &child : state.children) {
      const auto &nested = *child->impl_;
      const double left = nested.left_edges.empty() ? 0 : nested.left_edges.begin()->first;
      const double top = nested.top_edges.empty() ? 0 : nested.top_edges.begin()->first;
      const double right = nested.right_edges.empty() ? 0 : nested.right_edges.rbegin()->first;
      const double bottom = nested.bottom_edges.empty() ? 0 : nested.bottom_edges.rbegin()->first;
      state.box.width = std::max(state.box.width, right - left + 48);
      children_height += bottom - top + 48;
    }
    for (const auto &summary : state.summaries) {
      state.box.width = std::max(state.box.width, summary.right - summary.left + 48);
      children_height += summary.bottom - summary.top + 48;
    }
    if (!state.children.empty() || !state.summaries.empty())
      state.box.height = std::max(state.box.height, children_height);
    if (state.box.width > 1e6 || state.box.height > 1e6) {
      Diagnose("Layout dimensions exceeded the supported bounds; clipped the node box.");
      state.box.width = std::min(state.box.width, 1e6);
      state.box.height = std::min(state.box.height, 1e6);
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
      LayoutPoint end{state.box.x, state.box.y + 24 + 18.0 * port};
      auto producer = producers.find(value);
      if (producer != producers.end()) {
        edge.source = producer->second.id;
        edge.source_port = producer->second.port;
        const auto &source = states.at(edge.source).box;
        edge.folded = source.folded;
        LayoutPoint start{source.x + source.width, source.y + 24 + 18.0 * edge.source_port};
        const double middle = (start.x + end.x) / 2;
        edge.points = {start, {middle, start.y}, {middle, end.y}, end};
      } else {
        edge.points = {{end.x - 32, end.y}, end};
      }
      state.edges.push_back(std::move(edge));
      ++stats.edges;
    }
  }

  bool Collides(const LayoutNode &box) const {
    for (const auto &[layer, index] : occupied) {
      (void)layer;
      if (*index.rights.rbegin() + 24 <= box.x || *index.lefts.begin() >= box.x + box.width + 24)
        continue;
      auto position = index.positions.lower_bound(box.y - *index.heights.rbegin() - 24);
      for (; position != index.positions.end() && position->first < box.y + box.height + 24;
           ++position)
        for (auto other : position->second) {
          const auto &occupied_box = states.at(other).box;
          if (other != box.id && box.x < occupied_box.x + occupied_box.width + 24 &&
              box.x + box.width + 24 > occupied_box.x && box.y + box.height + 24 > occupied_box.y &&
              box.y < occupied_box.y + occupied_box.height + 24)
            return true;
        }
    }
    return false;
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
        const double next = source.x + source.width + 80;
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
    auto place = [&](LayoutNodeId id) {
      State &state = states.at(id);
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
            const double next = source.x + source.width + 80;
            x = has_predecessor ? std::max(x, next) : next;
            has_predecessor = true;
            barycenter += source.y + 18.0 * producer->second.port - 18.0 * port;
            ++barycenter_count;
          }
        } else if (producer != producers.end())
          Diagnose("Self-referencing layout dependency.");
      }
      layer = std::min(layer, states.size());
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
      const bool moved =
          state.box.layer != layer || (!state.pinned && state.box.x != x) || seeds.count(id) ||
          repaired_pin;
      if (!moved)
        return;
      RemoveOccupied(state);
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
          for (const auto &[other_layer, index] : occupied) {
            (void)other_layer;
            if (*index.rights.rbegin() + 24 <= state.box.x ||
                *index.lefts.begin() >= state.box.x + state.box.width + 24)
              continue;
            auto position = index.positions.lower_bound(y - *index.heights.rbegin() - 24);
            for (; position != index.positions.end() && position->first < y + state.box.height + 24;
                 ++position) {
              for (auto other : position->second) {
                const auto &box = states.at(other).box;
                if (state.box.x < box.x + box.width + 24 &&
                    state.box.x + state.box.width + 24 > box.x &&
                    y + state.box.height + 24 > box.y && y < box.y + box.height + 24) {
                  y = box.y + box.height + 24;
                  collision = true;
                }
              }
            }
          }
        } while (collision && y <= CoordinateLimit);
        if (y > CoordinateLimit)
          Diagnose("Layout coordinate space exhausted; clipped the node position.");
        state.box.y = std::min(y, CoordinateLimit);
      }
      auto &index = occupied[layer];
      index.positions[state.box.y].insert(id);
      index.lefts.insert(state.box.x);
      index.rights.insert(state.box.x + state.box.width);
      index.heights.insert(state.box.height);
      right_edges[state.box.x + state.box.width].insert(id);
      bottom_edges[state.box.y + state.box.height].insert(id);
      left_edges[state.box.x].insert(id);
      top_edges[state.box.y].insert(id);
      state.placed = true;
    };
    std::set<LayoutNodeId> placed;
    while (!queue.empty()) {
      const auto id = queue.front();
      queue.pop_front();
      place(id);
      placed.insert(id);
      for (auto user : dependents[id])
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
GraphLayout::~GraphLayout() = default;

void GraphLayout::Load(bool recurse) {
  auto &impl = *impl_;
  impl.recursive = recurse;
  impl.loading = true;
  impl.states.clear();
  impl.order.clear();
  impl.producers.clear();
  impl.consumers.clear();
  impl.occupied.clear();
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
  const auto next = Metadata(impl.graph, NextIdKey);
  if (!next.empty()) {
    LayoutNodeId next_id = 0;
    const auto bounded = next.substr(0, 17);
    const auto parsed = std::from_chars(bounded.data(), bounded.data() + bounded.size(), next_id);
    if (next.size() <= 16 && parsed.ec == std::errc() &&
        parsed.ptr == bounded.data() + bounded.size() && next_id && next_id <= IdLimit)
      impl.next_id = next_id;
    else
      impl.Diagnose("Invalid persisted next layout id; chose a fresh allocation point.");
  }
  const auto version = Metadata(impl.graph, VersionKey);
  impl.trusted_positions = version == "1";
  if (!version.empty() && version != "1")
    impl.Diagnose("Unsupported layout metadata version; recomputed geometry.");
  for (const auto &input : impl.graph.input())
    impl.local_values.insert(input.name());
  for (const auto &output : impl.graph.output())
    impl.graph_outputs.insert(output.name());
  for (const auto &tensor : impl.graph.initializer()) {
    impl.local_values.insert(tensor.name());
    impl.initializers.insert(tensor.name());
  }
  for (const auto &tensor : impl.graph.sparse_initializer()) {
    impl.local_values.insert(tensor.values().name());
    impl.initializers.insert(tensor.values().name());
  }
  for (const auto &tensor : impl.graph.extended_initializer()) {
    impl.local_values.insert(tensor.name());
    impl.initializers.insert(tensor.name());
  }
  for (const auto &node : impl.graph.node()) {
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
  for (auto &node : *impl.graph.mutable_node()) {
    auto id = impl.Add(node);
    impl.order.push_back(id);
    seeds.insert(id);
  }
  for (const auto &output : impl.graph.output())
    impl.UpdateExternal(output.name());
  impl.Solve(seeds);
  // Restores the exact saved frame first; subsequent edits can move automatic nodes.
  for (auto &[id, state] : impl.states) {
    (void)id;
    if (state.persisted_automatic)
      state.pinned = false;
  }
  impl.loading = false;
}

void GraphLayout::Persist(bool recurse) {
  SetMetadata(impl_->graph, VersionKey, "1");
  SetMetadata(impl_->graph, NextIdKey, std::to_string(impl_->next_id));
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
  if (!bounded || summary.str().size() > SummaryLimit) {
    impl_->Diagnose("Layout capture summary exceeded the metadata size limit.");
    SetMetadata(impl_->graph, ScopeKey, "");
  } else {
    SetMetadata(impl_->graph, ScopeKey, summary.str());
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
  impl.stats = {};
  auto &nodes = *impl.graph.mutable_node();
  nodes.push_back(node);
  for (size_t i = nodes.size() - 1; i > index; --i)
    std::swap(nodes.get(i), nodes.get(i - 1));
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
  impl.stats = {};
  std::set<LayoutNodeId> affected;
  impl.Unindex(id, affected);
  impl.RemoveOccupied(impl.states.at(id));
  impl.states.erase(id);
  const auto it = std::find(impl.order.begin(), impl.order.end(), id);
  const auto index = it - impl.order.begin();
  impl.graph.mutable_node()->erase(impl.graph.mutable_node()->begin() + index);
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
      height < 1 || width > 1e6 || height > 1e6)
    return false;
  impl.stats = {};
  auto &state = impl.states.at(id);
  impl.RemoveOccupied(state);
  state.box.width = width;
  state.box.height = height;
  state.fixed_size = true;
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
