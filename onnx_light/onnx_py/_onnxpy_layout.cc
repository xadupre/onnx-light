#include "_onnxpyprotoop.h"
#include "onnx_core/graph/graph_layout.h"
#include <memory>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::graph;

namespace {
struct PyGraphLayout {
  nb::object graph;
  std::shared_ptr<GraphLayout> root;
  std::shared_ptr<size_t> revision;
  size_t expected_revision = 0;
  std::vector<std::pair<LayoutNodeId, size_t>> path;

  explicit PyGraphLayout(nb::object owner)
      : graph(std::move(owner)), root(Bind(graph)), revision(std::make_shared<size_t>(0)) {}

  static std::shared_ptr<GraphLayout> Bind(nb::handle proto) {
    if (nb::isinstance<GraphProto>(proto))
      return std::make_shared<GraphLayout>(nb::cast<GraphProto &>(proto));
    if (nb::isinstance<FunctionProto>(proto))
      return std::make_shared<GraphLayout>(nb::cast<FunctionProto &>(proto));
    throw nb::type_error("GraphLayout expects a GraphProto or FunctionProto.");
  }

  GraphLayout &Get() const {
    if (!path.empty() && expected_revision != *revision)
      throw std::runtime_error("Child layout handle was invalidated; reacquire children().");
    auto *layout = root.get();
    for (const auto &[container, index] : path) {
      const auto children = layout->Children(container);
      if (index >= children.size())
        throw std::runtime_error("Child layout no longer exists.");
      layout = children[index];
    }
    return *layout;
  }

  void InvalidateChildren() {
    ++*revision;
    expected_revision = *revision;
  }
};
} // namespace

void AddOnnxPyLayout(nb::module_ &m) {
  nb::class_<LayoutPoint>(m, "LayoutPoint", "Stores a scope-local route point in visual units.")
      .def_ro("x", &LayoutPoint::x, "Returns the horizontal coordinate; negative values are valid.")
      .def_ro("y", &LayoutPoint::y, "Returns the vertical coordinate, increasing downward.");
  nb::class_<LayoutNode>(m, "LayoutNode", "Stores a read-only snapshot of node geometry and ports.")
      .def_ro("id", &LayoutNode::id, "Returns the stable scope-local ID, not a node-list index.")
      .def_ro("x", &LayoutNode::x, "Returns the rectangle's left coordinate.")
      .def_ro("y", &LayoutNode::y, "Returns the rectangle's top coordinate.")
      .def_ro("width", &LayoutNode::width, "Returns the positive width in visual units.")
      .def_ro("height", &LayoutNode::height, "Returns the positive height in visual units.")
      .def_ro("layer", &LayoutNode::layer, "Returns the dependency rank.")
      .def_ro("folded", &LayoutNode::folded,
              "Reports virtual Constant content rendered at consumer ports; boxes may overlap.")
      .def_ro("inputs", &LayoutNode::inputs,
              "Returns declared inputs in port order, including empty names.")
      .def_ro("outputs", &LayoutNode::outputs, "Returns declared outputs in port order.");
  nb::class_<LayoutEdge>(m, "LayoutEdge", "Stores a read-only snapshot of an ordered input route.")
      .def_ro("source", &LayoutEdge::source,
              "Returns the producer ID, or zero for inputs, initializers and external values.")
      .def_ro("target", &LayoutEdge::target, "Returns the consumer's scope-local ID.")
      .def_ro("source_port", &LayoutEdge::source_port,
              "Returns the zero-based producer output port.")
      .def_ro("target_port", &LayoutEdge::target_port,
              "Returns the target port; implicit captures follow declared inputs.")
      .def_ro("value", &LayoutEdge::value, "Returns the original ONNX value name.")
      .def_ro("folded", &LayoutEdge::folded,
              "Reports initializer or Constant content rendered inline.")
      .def_ro("capture", &LayoutEdge::capture, "Reports an implicit or outer-scope dependency.")
      .def_ro("points", &LayoutEdge::points,
              "Returns orthogonal route points from source to target.");
  nb::class_<LayoutStats>(m, "LayoutStats",
                          "Stores work counters for one scope's latest load or local update.")
      .def_ro("nodes", &LayoutStats::nodes, "Returns the number of processed nodes.")
      .def_ro("edges", &LayoutStats::edges, "Returns the number of rebuilt input routes.")
      .def_ro("scopes", &LayoutStats::scopes,
              "Returns the number of solved scopes, excluding children.")
      .def_ro("collision_index_visits", &LayoutStats::collision_index_visits,
              "Returns the number of spatial-index entries visited during collision checks.");
  nb::class_<PyGraphLayout>(
      m, "GraphLayout",
      "Maintains incremental positions for a live GraphProto or FunctionProto.\n\n"
      "Retains the proto owner. Call load() once, edit through this object, then "
      "persist() before saving. Coordinates are scope-local, x-right and y-down. "
      "Edits update affected consumers without rebuilding unrelated scopes; "
      "direct proto edits require load() to rebuild indexes. Query results are "
      "snapshots, except children(), which returns live handles.")
      .def(nb::init<nb::object>(), nb::arg("graph"),
           "Binds a GraphProto or FunctionProto without loading; raises TypeError for other types.")
      .def(
          "load",
          [](PyGraphLayout &self, bool recursive) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            layout.Load(recursive);
          },
          nb::arg("recursive") = true,
          "Loads indexes and validates or computes geometry, clearing diagnostics and invalidating "
          "child handles. Reuses persisted child summaries when recursive=False; missing or "
          "invalid "
          "summaries cause direct child loading. Assigns missing IDs but does not persist "
          "geometry.")
      .def(
          "persist", [](PyGraphLayout &self, bool recursive) { self.Get().Persist(recursive); },
          nb::arg("recursive") = true,
          "Writes namespaced metadata without changing unrelated keys. Persists bound children "
          "when recursive=True, without invalidating child handles; call on the root after child "
          "edits before serializing.")
      .def(
          "append",
          [](PyGraphLayout &self, const NodeProto &node) { return self.Get().Append(node); },
          nb::arg("node"),
          "Appends a node copy with fresh layout state, updates affected consumers, and returns "
          "its new stable scope-local ID.")
      .def(
          "insert",
          [](PyGraphLayout &self, size_t index, const NodeProto &node) {
            return self.Get().Insert(index, node);
          },
          nb::arg("index"), nb::arg("node"),
          "Inserts a node copy with fresh layout state at a semantic list index and returns its "
          "new stable ID. Raises "
          "IndexError "
          "when index exceeds the node count; never reorders existing semantic nodes.")
      .def(
          "erase",
          [](PyGraphLayout &self, LayoutNodeId id) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            return layout.Erase(id);
          },
          nb::arg("id"),
          "Erases the node and its child scopes, updating affected consumers. Returns False for an "
          "unknown ID. Invalidates existing child handles; reacquire them with children().")
      .def(
          "replace",
          [](PyGraphLayout &self, LayoutNodeId id, const NodeProto &node) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            return layout.Replace(id, node);
          },
          nb::arg("id"), nb::arg("node"),
          "Replaces the node with a copy while retaining its ID and updating affected consumers. "
          "Returns False for an unknown ID. Invalidates child handles and reconstructs child "
          "scopes.")
      .def(
          "rewire_input",
          [](PyGraphLayout &self, LayoutNodeId id, size_t port, const std::string &value) {
            return self.Get().RewireInput(id, port, value);
          },
          nb::arg("id"), nb::arg("port"), nb::arg("value"),
          "Rewires a zero-based declared input port and updates affected consumers. Allows an "
          "empty "
          "value for an optional input. Returns False for an unknown ID or invalid port.")
      .def(
          "set_position",
          [](PyGraphLayout &self, LayoutNodeId id, double x, double y) {
            return self.Get().SetPosition(id, x, y);
          },
          nb::arg("id"), nb::arg("x"), nb::arg("y"),
          "Pins the node at finite x/y coordinates with magnitude at most 1e9. Returns False for "
          "an "
          "unknown ID, invalid coordinates, or a dependency/collision constraint violation; "
          "rejected "
          "constraints append a diagnostic. Folded boxes may overlap.")
      .def(
          "set_size",
          [](PyGraphLayout &self, LayoutNodeId id, double width, double height) {
            return self.Get().SetSize(id, width, height);
          },
          nb::arg("id"), nb::arg("width"), nb::arg("height"),
          "Sets explicit finite dimensions in [1, 1e6], respecting child bounds and updating "
          "consumers. Returns False for an unknown ID or invalid dimensions.")
      .def(
          "get",
          [](const PyGraphLayout &self, LayoutNodeId id) -> LayoutNode {
            const auto *node = self.Get().Get(id);
            if (!node)
              throw nb::key_error("Unknown layout node id.");
            return *node;
          },
          nb::arg("id"),
          "Returns a read-only LayoutNode snapshot that survives subsequent edits. Raises KeyError "
          "for an unknown ID.")
      .def_prop_ro(
          "order", [](const PyGraphLayout &self) { return self.Get().Order(); },
          "Returns stable IDs in semantic node-list order, not dependency order.")
      .def(
          "edges", [](const PyGraphLayout &self, LayoutNodeId id) { return self.Get().Edges(id); },
          nb::arg("consumer"),
          "Returns input-route snapshots in target-port order, or an empty list for an unknown ID.")
      .def(
          "children",
          [](const PyGraphLayout &self, LayoutNodeId id) {
            std::vector<PyGraphLayout> result;
            const auto children = self.Get().Children(id);
            for (size_t index = 0; index < children.size(); ++index) {
              result.push_back(self);
              result.back().path.emplace_back(id, index);
              result.back().expected_revision = *self.revision;
            }
            return result;
          },
          nb::arg("container"),
          "Returns live child layouts in graph-attribute order, retaining the root proto owner. "
          "Unknown or summary-only containers yield an empty list. Any load(), replace() or "
          "erase() "
          "on a related handle invalidates existing child handles; stale access raises "
          "RuntimeError.")
      .def_prop_ro(
          "diagnostics", [](const PyGraphLayout &self) { return self.Get().Diagnostics(); },
          "Returns recovery and rejected-placement messages accumulated since load().")
      .def_prop_ro(
          "stats", [](const PyGraphLayout &self) { return self.Get().Stats(); },
          "Returns work counters for this scope's latest load or local update.");
  m.def(
      "enable_graph_layout",
      [](ParseOptions &options) {
        auto enable = [](auto &callback) {
          const auto previous = callback;
          callback = [previous](auto &proto) {
            if (previous)
              previous(proto);
            GraphLayout layout(proto);
            layout.Load(false);
            layout.Persist(false);
            const auto &diagnostics = layout.Diagnostics();
            if (!diagnostics.empty()) {
              nb::gil_scoped_acquire gil;
              for (const auto &diagnostic : diagnostics) {
                if (PyErr_WarnEx(PyExc_RuntimeWarning, diagnostic.c_str(), 1) < 0)
                  throw nb::python_error();
              }
            }
          };
        };
        enable(options.graph_callback);
        enable(options.function_callback);
      },
      nb::arg("options"),
      "Enables native graph and function layout on ParseOptions.\n\n"
      "Chains graph_callback and function_callback, invoking existing callbacks before placement. "
      "Runs after each scope finishes parsing, children first, and persists layout metadata "
      "without "
      "a separate model traversal. Emits RuntimeWarning for recoverable metadata diagnostics. "
      "Retains the callbacks for the lifetime of the options; leaves node and tensor callbacks "
      "unchanged.");
}
