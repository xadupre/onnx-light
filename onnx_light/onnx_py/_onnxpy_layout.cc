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
      : graph(std::move(owner)), root(std::make_shared<GraphLayout>(nb::cast<GraphProto &>(graph))),
        revision(std::make_shared<size_t>(0)) {}

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
  nb::class_<LayoutPoint>(m, "LayoutPoint")
      .def_ro("x", &LayoutPoint::x)
      .def_ro("y", &LayoutPoint::y);
  nb::class_<LayoutNode>(m, "LayoutNode")
      .def_ro("id", &LayoutNode::id)
      .def_ro("x", &LayoutNode::x)
      .def_ro("y", &LayoutNode::y)
      .def_ro("width", &LayoutNode::width)
      .def_ro("height", &LayoutNode::height)
      .def_ro("layer", &LayoutNode::layer)
      .def_ro("folded", &LayoutNode::folded)
      .def_ro("inputs", &LayoutNode::inputs)
      .def_ro("outputs", &LayoutNode::outputs);
  nb::class_<LayoutEdge>(m, "LayoutEdge")
      .def_ro("source", &LayoutEdge::source)
      .def_ro("target", &LayoutEdge::target)
      .def_ro("source_port", &LayoutEdge::source_port)
      .def_ro("target_port", &LayoutEdge::target_port)
      .def_ro("value", &LayoutEdge::value)
      .def_ro("folded", &LayoutEdge::folded)
      .def_ro("capture", &LayoutEdge::capture)
      .def_ro("points", &LayoutEdge::points);
  nb::class_<LayoutStats>(m, "LayoutStats")
      .def_ro("nodes", &LayoutStats::nodes)
      .def_ro("edges", &LayoutStats::edges)
      .def_ro("scopes", &LayoutStats::scopes);
  nb::class_<PyGraphLayout>(m, "GraphLayout",
                            "Maintains graph-scoped incremental positions; edits use this object.")
      .def(nb::init<nb::object>(), nb::arg("graph"))
      .def(
          "load",
          [](PyGraphLayout &self, bool recursive) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            layout.Load(recursive);
          },
          nb::arg("recursive") = true)
      .def(
          "persist", [](PyGraphLayout &self, bool recursive) { self.Get().Persist(recursive); },
          nb::arg("recursive") = true)
      .def(
          "append",
          [](PyGraphLayout &self, const NodeProto &node) { return self.Get().Append(node); },
          nb::arg("node"))
      .def(
          "insert",
          [](PyGraphLayout &self, size_t index, const NodeProto &node) {
            return self.Get().Insert(index, node);
          },
          nb::arg("index"), nb::arg("node"))
      .def(
          "erase",
          [](PyGraphLayout &self, LayoutNodeId id) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            return layout.Erase(id);
          },
          nb::arg("id"))
      .def(
          "replace",
          [](PyGraphLayout &self, LayoutNodeId id, const NodeProto &node) {
            auto &layout = self.Get();
            self.InvalidateChildren();
            return layout.Replace(id, node);
          },
          nb::arg("id"), nb::arg("node"))
      .def(
          "rewire_input",
          [](PyGraphLayout &self, LayoutNodeId id, size_t port, const std::string &value) {
            return self.Get().RewireInput(id, port, value);
          },
          nb::arg("id"), nb::arg("port"), nb::arg("value"))
      .def(
          "set_position",
          [](PyGraphLayout &self, LayoutNodeId id, double x, double y) {
            return self.Get().SetPosition(id, x, y);
          },
          nb::arg("id"), nb::arg("x"), nb::arg("y"))
      .def(
          "set_size",
          [](PyGraphLayout &self, LayoutNodeId id, double width, double height) {
            return self.Get().SetSize(id, width, height);
          },
          nb::arg("id"), nb::arg("width"), nb::arg("height"))
      .def(
          "get",
          [](const PyGraphLayout &self, LayoutNodeId id) -> LayoutNode {
            const auto *node = self.Get().Get(id);
            if (!node)
              throw nb::key_error("Unknown layout node id.");
            return *node;
          },
          nb::arg("id"))
      .def_prop_ro("order", [](const PyGraphLayout &self) { return self.Get().Order(); })
      .def(
          "edges", [](const PyGraphLayout &self, LayoutNodeId id) { return self.Get().Edges(id); },
          nb::arg("consumer"))
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
          nb::arg("container"))
      .def_prop_ro("diagnostics",
                   [](const PyGraphLayout &self) { return self.Get().Diagnostics(); })
      .def_prop_ro("stats", [](const PyGraphLayout &self) { return self.Get().Stats(); });
  m.def(
      "enable_graph_layout",
      [](ParseOptions &options) {
        const auto previous = options.graph_callback;
        options.graph_callback = [previous](GraphProto &graph) {
          if (previous)
            previous(graph);
          GraphLayout layout(graph);
          layout.Load(false);
          layout.Persist(false);
          for (const auto &diagnostic : layout.Diagnostics()) {
            nb::gil_scoped_acquire gil;
            if (PyErr_WarnEx(PyExc_RuntimeWarning, diagnostic.c_str(), 1) < 0)
              throw nb::python_error();
          }
        };
      },
      nb::arg("options"),
      "Enables native graph-local layout during parsing, chaining any existing graph callback.");
}
