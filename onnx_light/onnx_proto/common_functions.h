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

/// ``std::vector``-overload of :cpp:func:`CollectExternalInputs`.
std::vector<std::string> CollectExternalInputs(const std::vector<NodeProto> &nodes);

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
