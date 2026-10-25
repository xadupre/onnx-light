// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"

#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::builder::GraphGraph;
using core::builder::MatchResult;
using core::builder::PatternOptimization;

std::string PyValueName(nb::handle value) { return nb::cast<std::string>(nb::str(value)); }

class PyPatternOptimization : public PatternOptimization {
public:
  NB_TRAMPOLINE(PatternOptimization);

  std::set<std::string> FastOpType() const override {
    NB_OVERRIDE_NAME("fast_op_type", FastOpType);
  }

  MatchResult Match(GraphGraph &graph, const NodeProto &candidate) const override {
    nb::detail::ticket ticket(nb_trampoline, "match", nb::detail::str_hash("match"), true);
    nb::object result = nb_trampoline.base().attr(ticket.key)(
        nb::cast(&graph, nb::rv_policy::reference), nb::cast(&candidate, nb::rv_policy::reference));
    return nb::cast<MatchResult>(result);
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const override {
    nb::detail::ticket ticket(nb_trampoline, "apply", nb::detail::str_hash("apply"), true);
    nb::list borrowed_nodes;
    for (const NodeProto *node : nodes) {
      borrowed_nodes.append(nb::cast(node, nb::rv_policy::reference));
    }
    nb::object result = nb_trampoline.base().attr(ticket.key)(
        nb::cast(&graph, nb::rv_policy::reference), borrowed_nodes);
    utils::RepeatedProtoField<NodeProto> copied;
    for (nb::handle item : nb::borrow<nb::iterable>(result)) {
      copied.push_back(nb::cast<const NodeProto &>(item));
    }
    return copied;
  }
};

} // namespace

void AddOnnxPyPatternCore(nb::module_ &m) {
  using core::builder::OptimizationReport;
  using core::builder::PatternNoMatchStatistics;
  using core::builder::PatternOptimizationStatistics;
  using core::builder::RegisteredPatternNames;
  using core::builder::SubgraphOptimizationStatistics;

  nb::module_ builder_mod = m.attr("builder");

  nb::class_<PatternNoMatchStatistics>(builder_mod, "PatternNoMatchStatistics")
      .def_ro("source_file", &PatternNoMatchStatistics::source_file)
      .def_ro("source_line", &PatternNoMatchStatistics::source_line)
      .def_ro("reason", &PatternNoMatchStatistics::reason)
      .def_ro("occurrences", &PatternNoMatchStatistics::occurrences)
      .def("__str__", &PatternNoMatchStatistics::ToString)
      .def("__repr__", &PatternNoMatchStatistics::ToString);

  nb::class_<PatternOptimizationStatistics>(builder_mod, "PatternOptimizationStatistics")
      .def_ro("pattern_name", &PatternOptimizationStatistics::pattern_name)
      .def_ro("attempts", &PatternOptimizationStatistics::attempts)
      .def_ro("matches", &PatternOptimizationStatistics::matches)
      .def_ro("match_time_ns", &PatternOptimizationStatistics::match_time_ns)
      .def_ro("apply_time_ns", &PatternOptimizationStatistics::apply_time_ns)
      .def_ro("no_matches", &PatternOptimizationStatistics::no_matches)
      .def("__str__", &PatternOptimizationStatistics::ToString)
      .def("__repr__", &PatternOptimizationStatistics::ToString);

  nb::class_<SubgraphOptimizationStatistics>(builder_mod, "SubgraphOptimizationStatistics")
      .def_ro("graph_path", &SubgraphOptimizationStatistics::graph_path)
      .def_ro("iterations", &SubgraphOptimizationStatistics::iterations)
      .def_ro("rewrites", &SubgraphOptimizationStatistics::rewrites)
      .def_ro("elapsed_time_ns", &SubgraphOptimizationStatistics::elapsed_time_ns)
      .def("__str__", &SubgraphOptimizationStatistics::ToString)
      .def("__repr__", &SubgraphOptimizationStatistics::ToString);

  nb::class_<OptimizationReport>(builder_mod, "OptimizationReport")
      .def(nb::init<>())
      .def_ro("iterations", &OptimizationReport::iterations)
      .def_ro("rewrites", &OptimizationReport::rewrites)
      .def_ro("matching_time_ns", &OptimizationReport::matching_time_ns)
      .def_ro("rewriting_time_ns", &OptimizationReport::rewriting_time_ns)
      .def_ro("cleanup_time_ns", &OptimizationReport::cleanup_time_ns)
      .def_ro("constant_folding_time_ns", &OptimizationReport::constant_folding_time_ns)
      .def_ro("subgraph_optimization_time_ns", &OptimizationReport::subgraph_optimization_time_ns)
      .def_ro("patterns", &OptimizationReport::patterns)
      .def_ro("subgraphs", &OptimizationReport::subgraphs)
      .def_prop_ro("total_time_ns", &OptimizationReport::TotalTimeNs)
      .def("__str__", &OptimizationReport::ToString)
      .def("__repr__", &OptimizationReport::ToString);

  nb::class_<PatternOptimization, PyPatternOptimization>(builder_mod, "PatternOptimization")
      .def(nb::init<int, std::string, core::symbolic::Device>(), nb::arg("priority") = 1,
           nb::arg("name") = "", nb::arg("device") = core::symbolic::Device::kUndefined)
      .def_prop_ro("name", &PatternOptimization::Name)
      .def_rw("priority", &PatternOptimization::priority)
      .def_rw("device", &PatternOptimization::device,
              "Exact target device, or Device.kUndefined for a device-independent pattern.")
      .def("fast_op_type", &PatternOptimization::FastOpType)
      .def(
          "result",
          [](const PatternOptimization &pattern, const std::vector<const NodeProto *> &nodes,
             const NodeProto *insert_at) {
            return MatchResult{&pattern, nodes, insert_at, std::nullopt};
          },
          nb::arg("nodes"), nb::arg("insert_at").none() = nullptr, nb::keep_alive<0, 1>(),
          "Creates a successful match referring to this pattern.")
      .def(
          "no_match",
          [](const PatternOptimization &, const NodeProto &candidate, const std::string &reason) {
            MatchResult result;
            result.no_match = core::builder::PatternNoMatch{&candidate, "<python>", 0, reason};
            return result;
          },
          nb::arg("candidate"), nb::arg("reason"),
          "Creates a rejected match with a Python diagnostic.")
      .def("__str__", &PatternOptimization::ToString)
      .def("__repr__", &PatternOptimization::ToString);

  nb::class_<MatchResult>(builder_mod, "MatchResult")
      .def(nb::init<>())
      .def_prop_ro(
          "pattern",
          [](const MatchResult &result) -> const PatternOptimization * { return result.pattern; },
          nb::rv_policy::reference)
      .def_prop_ro("nodes",
                   [](const MatchResult &result) {
                     nb::list nodes;
                     for (const NodeProto *node : result.nodes) {
                       nodes.append(nb::cast(node, nb::rv_policy::reference));
                     }
                     return nodes;
                   })
      .def_prop_ro(
          "insert_at",
          [](const MatchResult &result) -> const NodeProto * { return result.insert_at; },
          nb::rv_policy::reference)
      .def_prop_ro("matched", [](const MatchResult &result) { return result.pattern != nullptr; })
      .def("__str__", &MatchResult::ToString)
      .def("__repr__", &MatchResult::ToString);

  nb::class_<GraphGraph>(builder_mod, "GraphGraph")
      .def(
          "__init__",
          [](GraphGraph *self, core::builder::GraphBuilder &builder) {
            new (self) GraphGraph(builder);
          },
          nb::arg("builder"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](GraphGraph *self, core::builder::GraphBuilder &builder, nb::iterable patterns) {
            std::vector<std::shared_ptr<PatternOptimization>> owned;
            for (nb::handle pattern : patterns) {
              owned.push_back(nb::cast<std::shared_ptr<PatternOptimization>>(pattern));
            }
            new (self) GraphGraph(builder, std::move(owned));
          },
          nb::arg("builder"), nb::arg("patterns"), nb::keep_alive<1, 2>())
      .def_prop_ro("builder", nb::overload_cast<>(&GraphGraph::Builder),
                   nb::rv_policy::reference_internal)
      .def_prop_ro("name", [](const GraphGraph &graph) { return graph.Builder().name(); })
      .def_prop_ro("patterns", &GraphGraph::Patterns)
      .def("set_target_device", &GraphGraph::SetTargetDevice, nb::arg("device"),
           "Sets a target on the builder and its subgraphs, rejecting conflicting devices.")
      .def(
          "optimize",
          [](GraphGraph &graph, int max_iter, bool report,
             const core::builder::ConstantFoldingOptions &folding_options, bool remove_unused,
             nb::object callback) -> nb::object {
            GraphGraph::OptimizationCallback on_match;
            if (!callback.is_none()) {
              if (!PyCallable_Check(callback.ptr()))
                throw nb::type_error("callback must be callable or None.");
              on_match = [&](const GraphGraph &scope, const MatchResult &match) {
                nb::object decision = callback(nb::cast(&scope, nb::rv_policy::reference),
                                               nb::cast(match, nb::rv_policy::copy));
                return decision.is_none() || nb::cast<bool>(decision);
              };
            }
            if (!report) {
              return nb::cast(
                  graph.Optimize(max_iter, nullptr, folding_options, remove_unused, on_match));
            }
            OptimizationReport statistics;
            RewritingProto rewrites =
                graph.Optimize(max_iter, &statistics, folding_options, remove_unused, on_match);
            return nb::make_tuple(std::move(rewrites), std::move(statistics));
          },
          nb::arg("max_iter") = -1, nb::kw_only(), nb::arg("report") = false,
          nb::arg("folding_options") = core::builder::ConstantFoldingOptions{},
          nb::arg("remove_unused") = true, nb::arg("callback") = nb::none())
      .def(
          "node_before",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.NodeBefore(PyValueName(name));
          },
          nb::arg("name"), nb::rv_policy::reference_internal)
      .def(
          "next_nodes",
          [](const GraphGraph &graph, nb::handle name) -> const std::vector<const NodeProto *> & {
            return graph.NextNodes(PyValueName(name));
          },
          nb::arg("name"), nb::rv_policy::reference_internal)
      .def("predecessors", &GraphGraph::Predecessors, nb::arg("node"))
      .def("successors", &GraphGraph::Successors, nb::arg("node"))
      .def(
          "is_output",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.IsOutput(PyValueName(name));
          },
          nb::arg("name"))
      .def(
          "is_used",
          [](const GraphGraph &graph, nb::handle name) { return graph.IsUsed(PyValueName(name)); },
          nb::arg("name"))
      .def(
          "is_used_more_than_once",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.IsUsedMoreThanOnce(PyValueName(name));
          },
          nb::arg("name"))
      .def(
          "is_used_by_subgraph",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.IsUsedBySubgraph(PyValueName(name));
          },
          nb::arg("name"))
      .def("position", &GraphGraph::Position, nb::arg("node"))
      .def(
          "has_shape",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.HasShape(PyValueName(name));
          },
          nb::arg("name"))
      .def(
          "get_shape",
          [](const GraphGraph &graph, nb::handle name) -> const core::symbolic::SymTensor & {
            return graph.GetShape(PyValueName(name));
          },
          nb::arg("name"), nb::rv_policy::reference_internal)
      .def(
          "has_type",
          [](const GraphGraph &graph, nb::handle name) { return graph.HasType(PyValueName(name)); },
          nb::arg("name"))
      .def(
          "get_type",
          [](const GraphGraph &graph, nb::handle name) { return graph.GetType(PyValueName(name)); },
          nb::arg("name"))
      .def(
          "is_constant",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.IsConstant(PyValueName(name));
          },
          nb::arg("name"))
      .def(
          "is_constant_scalar",
          [](const GraphGraph &graph, nb::handle name, nb::object value, bool broadcast) {
            const std::string resolved_name = PyValueName(name);
            return value.is_none()
                       ? graph.IsConstantScalar(resolved_name, broadcast)
                       : graph.IsConstantScalar(resolved_name, nb::cast<double>(value), broadcast);
          },
          nb::arg("name"), nb::arg("value") = nb::none(), nb::arg("broadcast") = false)
      .def(
          "get_computed_constant",
          [](const GraphGraph &graph, nb::handle name) {
            return graph.GetComputedConstant(PyValueName(name));
          },
          nb::arg("name"), nb::rv_policy::reference_internal);

  builder_mod.def("registered_pattern_names", &RegisteredPatternNames,
                  nb::arg("include_opt_in") = false,
                  "Returns registered C++ pattern names in evaluation order, excluding opt-in "
                  "patterns unless requested. Listing patterns does not enable them.");
  builder_mod.def(
      "apply_rewritings",
      [](const ModelProto &model, const RewritingProto &rewrites, nb::object schema_lookup) {
        core::builder::GraphBuilder builder(
            model, schema_lookup.is_none()
                       ? core::builder::GraphBuilder::SchemaLookupFn{}
                       : nb::cast<core::builder::GraphBuilder::SchemaLookupFn>(schema_lookup));
        return builder.ApplyRewritings(rewrites).BuildGraph();
      },
      nb::arg("model"), nb::arg("rewrites"), nb::arg("schema_lookup") = nb::none(),
      "Applies captured rewrites and returns a graph; use a model export to retain functions.");
  builder_mod.def(
      "apply_rewritings_to_model",
      [](const ModelProto &model, const RewritingProto &rewrites, nb::object schema_lookup) {
        core::builder::GraphBuilder builder(
            model, schema_lookup.is_none()
                       ? core::builder::GraphBuilder::SchemaLookupFn{}
                       : nb::cast<core::builder::GraphBuilder::SchemaLookupFn>(schema_lookup));
        return builder.ApplyRewritings(rewrites).ToModel();
      },
      nb::arg("model"), nb::arg("rewrites"), nb::arg("schema_lookup") = nb::none(),
      "Applies captured rewrites and exports a model with its functions and imports.");
}
