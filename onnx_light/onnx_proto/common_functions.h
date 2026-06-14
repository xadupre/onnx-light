#pragma once

#include "onnx.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

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

/// ``std::vector`` overload of :cpp:func:`CollectExternalInputs`.
std::vector<std::string> CollectExternalInputs(const std::vector<NodeProto> &nodes);

/**
 * Returns, for every node in ``nodes``, the list of input names that are
 * needed by that node and all the nodes that follow it in the list.
 *
 * For each index ``i`` the result is the set of names referenced by the
 * sub-list ``nodes[i..]`` (including names captured by subgraph attributes,
 * ``GRAPH`` / ``GRAPHS``) that are not produced as outputs by any node in
 * that same sub-list — i.e. the external inputs of the remaining nodes, as
 * computed by :cpp:func:`CollectExternalInputs` on the suffix.
 *
 * The returned vector has exactly ``nodes.size()`` entries. Each inner list
 * preserves first-seen order and contains no duplicates; empty input names
 * are skipped. Because every name produced earlier in the list has been
 * consumed or is no longer required once it disappears from these lists, the
 * result can be used to decide which tensors still need to be kept alive
 * before running node ``i``.
 */
std::vector<std::vector<std::string>>
CollectRemainingInputs(const utils::RepeatedProtoField<NodeProto> &nodes);

/// ``std::vector`` overload of :cpp:func:`CollectRemainingInputs`.
std::vector<std::vector<std::string>> CollectRemainingInputs(const std::vector<NodeProto> &nodes);

/**
 * Returns the full list of tensor / sequence names a single ``node`` depends on
 * at runtime.
 *
 * Includes names referenced by ``node.input()`` and external inputs captured by
 * subgraph attributes (``GRAPH`` / ``GRAPHS``), preserves first-seen order
 * without duplicates, and skips empty input names.
 */
std::vector<std::string> CollectNodeInputs(const NodeProto &node);

} // namespace ONNX_LIGHT_NAMESPACE
