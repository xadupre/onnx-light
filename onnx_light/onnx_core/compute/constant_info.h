// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "onnx_proto/onnx.h"

/**
 * @file constant_info.h
 * @brief Constant-value / constant-node analysis.
 *
 * A *constant* value is one whose content is known before any inference
 * starts: graph initializers, the output of a ``Constant`` node, and — more
 * generally — the output of any deterministic node all of whose inputs are
 * themselves constant. A *constant node* is a node whose outputs are all
 * constant. Graph inputs are never constant.
 *
 * The analysis mirrors the value / node tag machinery
 * (:file:`value_tags.h`): the result is recorded per value on the
 * ``metadata_props`` of every ``ValueInfoProto`` (graph inputs, outputs and
 * shape-inference ``value_info``) and initializer ``TensorProto``, and per
 * node on the ``NodeProto`` ``metadata_props``, under
 * :cpp:var:`kConstantMetadataKey`. Nested subgraphs are supported: the
 * analysis recurses into ``GRAPH`` / ``GRAPHS`` attributes, seeding each
 * subgraph with the constant values captured from the enclosing scope.
 */

namespace ONNX_LIGHT_NAMESPACE::core::compute {

/// Metadata key under which the constant analysis records that a value or node
/// is constant (known before inference). The associated value is ``"1"``.
constexpr const char *kConstantMetadataKey = "onnx_light.constant";

/// Classifies whether a node produces constant outputs.
enum class ConstantInfo : uint8_t {
  kNotConstant = 0,
  kConstant = 1,
};

/// Returns whether ``op_type`` (in the default ONNX domain) is
/// non-deterministic and therefore never produces a constant output, even when
/// all of its inputs are constant (e.g. the random generators).
bool IsNonDeterministicOp(const std::string &domain, const std::string &op_type);

/// Returns whether ``node`` produces constant outputs given the set of already
/// known ``constants`` value names. A node is constant when it is
/// deterministic, every value it references (direct inputs and subgraph
/// captures) is constant, and none of its subgraphs contain a
/// non-deterministic op. Nodes with no reachable inputs (e.g. ``Constant``)
/// are constant unless the op is non-deterministic.
bool IsNodeConstant(const NodeProto &node, const std::unordered_set<std::string> &constants);

/// Infers the set of constant value names and the per-node constant flags for
/// ``graph``. ``outer_constants`` seeds the analysis with values captured from
/// an enclosing scope (used when recursing into subgraphs). The returned
/// ``std::vector<ConstantInfo>`` has one entry per node, aligned with
/// ``graph.node()``.
std::pair<std::unordered_set<std::string>, std::vector<ConstantInfo>>
InferConstants(const GraphProto &graph,
               const std::unordered_set<std::string> &outer_constants = {});

/// Same as :cpp:func:`InferConstants(const GraphProto&, ...)` but for a
/// ``FunctionProto`` (which carries no initializers; its formal inputs are
/// never constant).
std::pair<std::unordered_set<std::string>, std::vector<ConstantInfo>>
InferConstants(const FunctionProto &function,
               const std::unordered_set<std::string> &outer_constants = {});

/// Runs :cpp:func:`InferConstants` on ``graph`` and records the result in the
/// ``metadata_props`` of every constant node and every constant value
/// (graph inputs, outputs, ``value_info`` and initializers) under
/// :cpp:var:`kConstantMetadataKey`. Recurses into subgraphs, seeding each with
/// the constant values captured from the enclosing scope.
void WriteConstantInfoToMetadata(GraphProto &graph);

/// Same as :cpp:func:`WriteConstantInfoToMetadata(GraphProto&)` but for a
/// ``FunctionProto``.
void WriteConstantInfoToMetadata(FunctionProto &function);

/// Same as :cpp:func:`WriteConstantInfoToMetadata(GraphProto&)` but for a
/// ``ModelProto`` (operates on its main graph).
void WriteConstantInfoToMetadata(ModelProto &model);

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
