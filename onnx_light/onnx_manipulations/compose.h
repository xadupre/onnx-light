// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file compose.h
 * @brief Declares helpers to combine and manipulate ONNX graphs and models.
 *
 * These functions mirror ``onnx.compose`` from the upstream ONNX Python package
 * and are used to add name prefixes, check for overlapping names, merge graphs
 * and models, and expand output dimensions.
 */

#pragma once

#include "onnx.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Checks whether two graphs have overlapping names.
 *
 * Returns a list of ``(category, names)`` pairs for each category where
 * names appear in both *g1* and *g2*.  Recognised categories are
 * ``"edge"``, ``"value_info"``, ``"initializer"`` and
 * ``"sparse_initializer"``.
 *
 * The optional *io_map* lists output/input pairs that are intentionally
 * shared between the two graphs; those overlaps are excluded from the
 * result.
 */
std::vector<std::pair<std::string, std::vector<std::string>>>
CheckOverlappingNames(const GraphProto &g1, const GraphProto &g2,
                      const std::vector<std::pair<std::string, std::string>> &io_map = {});

/**
 * Adds a prefix to names of elements in a graph.
 *
 * Applies *prefix* to nodes, edges, inputs, outputs, initializers, sparse
 * initializers and value-infos as requested.  Empty names are never prefixed.
 *
 * @param graph             Graph to prefix.
 * @param prefix            Prefix string to prepend.
 * @param rename_nodes      Whether to prefix node names.
 * @param rename_edges      Whether to prefix node edge (input/output) names.
 * @param rename_inputs     Whether to prefix graph-input names.
 * @param rename_outputs    Whether to prefix graph-output names.
 * @param rename_initializers Whether to prefix initializer names.
 * @param rename_value_infos Whether to prefix value-info names.
 * @param inplace           If true, mutates *graph* in place; otherwise
 *                          operates on a copy.
 * @param name_map          Optional shared name map that accumulates all
 *                          renames across recursive subgraph calls.
 * @return The (possibly new) GraphProto with prefixed names.
 */
GraphProto AddPrefixGraph(const GraphProto &graph, const std::string &prefix,
                          bool rename_nodes = true, bool rename_edges = true,
                          bool rename_inputs = true, bool rename_outputs = true,
                          bool rename_initializers = true, bool rename_value_infos = true,
                          bool inplace = false,
                          std::unordered_map<std::string, std::string> *name_map = nullptr);

/**
 * Adds a prefix to names of elements in a model.
 *
 * Applies *prefix* to graph nodes, edges, inputs, outputs, initializers,
 * sparse initializers, value-infos and local functions as requested.
 * Empty names are never prefixed.
 *
 * @param model              Model to prefix.
 * @param prefix             Prefix string to prepend.
 * @param rename_nodes       Whether to prefix node names.
 * @param rename_edges       Whether to prefix node edge names.
 * @param rename_inputs      Whether to prefix input names.
 * @param rename_outputs     Whether to prefix output names.
 * @param rename_initializers Whether to prefix initializer names.
 * @param rename_value_infos Whether to prefix value-info names.
 * @param rename_functions   Whether to prefix local function names.
 * @param inplace            If true, mutates *model* in place; otherwise
 *                           operates on a copy.
 * @return The (possibly new) ModelProto with prefixed names.
 */
ModelProto AddPrefix(const ModelProto &model, const std::string &prefix,
                     bool rename_nodes = true, bool rename_edges = true,
                     bool rename_inputs = true, bool rename_outputs = true,
                     bool rename_initializers = true, bool rename_value_infos = true,
                     bool rename_functions = true, bool inplace = false);

/**
 * Combines two ONNX graphs into a single one.
 *
 * The combined graph is defined by connecting the specified set of
 * outputs/inputs from *g1* to inputs of *g2* as listed in *io_map*.
 * Inputs/outputs not present in *io_map* remain as inputs/outputs of the
 * combined graph.
 *
 * @param g1         First graph.
 * @param g2         Second graph.
 * @param io_map     Pairs ``[(out_name, in_name), …]`` mapping outputs of
 *                   *g1* to inputs of *g2*.
 * @param inputs     Optional list of inputs to include.  When absent all
 *                   inputs not in *io_map* are included.
 * @param outputs    Optional list of outputs to include.  When absent all
 *                   outputs not in *io_map* are included.
 * @param prefix1    Optional prefix for all names in *g1*.
 * @param prefix2    Optional prefix for all names in *g2*.
 * @param name       Optional name for the combined graph.
 * @param doc_string Optional doc-string for the combined graph.
 * @return Combined GraphProto.
 */
GraphProto MergeGraphs(const GraphProto &g1, const GraphProto &g2,
                       const std::vector<std::pair<std::string, std::string>> &io_map,
                       const std::vector<std::string> &inputs = {},
                       const std::vector<std::string> &outputs = {},
                       const std::string &prefix1 = "", const std::string &prefix2 = "",
                       const std::string &name = "", const std::string &doc_string = "");

/**
 * Combines two ONNX models into a single one.
 *
 * Both models must share the same IR version and operator sets.
 *
 * @param m1              First model.
 * @param m2              Second model.
 * @param io_map          Output/input pairs mapping outputs of *m1* to
 *                        inputs of *m2*.
 * @param inputs          Optional list of inputs for the combined model.
 * @param outputs         Optional list of outputs for the combined model.
 * @param prefix1         Optional prefix for all names in *m1*.
 * @param prefix2         Optional prefix for all names in *m2*.
 * @param name            Optional name for the combined graph.
 * @param doc_string      Optional doc-string for the combined graph.
 * @param producer_name   Producer name for the combined model.
 * @param producer_version Producer version string.
 * @param domain          Domain of the combined model.
 * @param model_version   Version of the combined model.
 * @return Combined ModelProto.
 */
ModelProto MergeModels(const ModelProto &m1, const ModelProto &m2,
                       const std::vector<std::pair<std::string, std::string>> &io_map,
                       const std::vector<std::string> &inputs = {},
                       const std::vector<std::string> &outputs = {},
                       const std::string &prefix1 = "", const std::string &prefix2 = "",
                       const std::string &name = "", const std::string &doc_string = "",
                       const std::string &producer_name = "onnx_light.onnx.compose.merge_models",
                       const std::string &producer_version = "1.0",
                       const std::string &domain = "", int64_t model_version = 1);

/**
 * Inserts an extra dimension with extent 1 to each output in the graph.
 *
 * Appends an ``Unsqueeze`` node for each output.  Useful before merging
 * graphs when the second graph expects a batch dimension.
 *
 * @param graph    Graph to modify.
 * @param dim_idx  Index at which the new dimension is inserted.  Negative
 *                 values count from the back.
 * @param inplace  If true, mutates *graph* in place; otherwise operates on
 *                 a copy.
 * @return The (possibly new) GraphProto with expanded output dimensions.
 */
GraphProto ExpandOutDimGraph(const GraphProto &graph, int64_t dim_idx, bool inplace = false);

/**
 * Inserts an extra dimension with extent 1 to each output in the model.
 *
 * @param model    Model to modify.
 * @param dim_idx  Index at which the new dimension is inserted.
 * @param inplace  If true, mutates *model* in place; otherwise operates on
 *                 a copy.
 * @return The (possibly new) ModelProto with expanded output dimensions.
 */
ModelProto ExpandOutDim(const ModelProto &model, int64_t dim_idx, bool inplace = false);

} // namespace ONNX_LIGHT_NAMESPACE
