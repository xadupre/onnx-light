#pragma once

#include "onnx.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Returns the list of input names referenced by ``nodes`` that are not produced
 * as outputs by any node in the same list.
 *
 * Subgraph attributes (``GRAPH`` / ``GRAPHS``) are inspected recursively:
 * names read by subgraph nodes that are neither produced inside the subgraph
 * nor produced by the outer ``nodes`` are appended.
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
 * The result includes names referenced by ``node.input()`` and external inputs
 * captured by subgraph attributes (``GRAPH`` / ``GRAPHS``), preserving
 * first-seen order without duplicates. Empty input names are skipped.
 */
std::vector<std::string> CollectNodeInputs(const NodeProto &node);

} // namespace ONNX_LIGHT_NAMESPACE
