// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/**
 * Returns the list of input names referenced by ``nodes`` that are not produced
 * as outputs by any node in the same list.
 *
 * Recursively inspects subgraph attributes (``GRAPH`` / ``GRAPHS``) and
 * appends names read by subgraph nodes when neither the subgraph nor the outer
 * ``nodes`` produce those names.
 *
 * The returned list preserves first-seen order and contains no duplicates.
 * Empty input names are skipped.
 */
std::vector<std::string> CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes);

/// Node-pointer overload of :cpp:func:`CollectExternalInputs`, used to collect
/// the external inputs of an :cpp:class:`ExecutionPlan`'s node list.
std::vector<std::string> CollectExternalInputs(const std::vector<const NodeProto *> &nodes);

/**
 * Returns, for every node in ``nodes``, the list of input names that must
 * already be available before that node runs in order to eventually produce
 * the requested ``outputs``.
 *
 * The algorithm performs a backward reachability analysis: starting from the
 * requested ``outputs``, it determines the ancestors (the nodes and tensors the
 * outputs transitively depend on). For each index ``i`` it considers the suffix
 * ``nodes[i..]``, keeps only the nodes of that suffix that are ancestors of
 * ``outputs`` (branches that do not contribute to ``outputs`` are pruned), and
 * reports the names those relevant nodes read — including names captured by
 * subgraph attributes ``GRAPH`` / ``GRAPHS`` — that are not produced within the
 * suffix. Those are the tensors that need to be kept alive before running node
 * ``i``.
 *
 * ``nodes`` is expected to be in topological order. The returned vector has
 * exactly ``nodes.size()`` entries. Each inner list preserves first-seen order
 * and contains no duplicates; empty input names are skipped.
 */
std::vector<std::vector<std::string>>
CollectRemainingInputs(const utils::RepeatedProtoField<NodeProto> &nodes,
                       const std::vector<std::string> &outputs);

/**
 * Returns the full list of tensor / sequence names a single ``node`` depends on
 * at runtime.
 *
 * Includes names referenced by ``node.input()`` and external inputs captured by
 * subgraph attributes (``GRAPH`` / ``GRAPHS``), preserves first-seen order
 * without duplicates, and skips empty input names.
 */
std::vector<std::string> CollectNodeInputs(const NodeProto &node);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
