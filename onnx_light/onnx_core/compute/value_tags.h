// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

class ComputeContext;

constexpr const char *kValueTagMetadataKey = "onnx_light.value_tag";
// Historic aggregate key. Value tags are stored per-value under
// ``kValueTagMetadataKey`` on each ValueInfoProto/TensorProto; this aggregate is
// no longer written to the graph or function ``metadata_props``.
constexpr const char *kValueTagsMetadataKey = "onnx_light.value_tags";
constexpr const char *kNodeTagMetadataKey = "onnx_light.node_tag";

// Normalizes a raw value-tag string to one of the known tags ("shape", "axes",
// "weight", "ambiguous"), returning an empty string for unknown tags.
std::string NormalizeValueTag(std::string_view tag);

// Sets ``name`` to ``tag`` (after normalization) and returns whether the map
// content changed (new key or updated tag value).
bool TrySetValueTag(std::unordered_map<std::string, std::string> &value_tags,
                    const std::string &name, const std::string &tag);

// Seeds ``value_tags`` from a graph's inputs, initializers, value_info and
// outputs (reading their ``onnx_light.value_tag`` metadata).
void CollectGraphSeedTags(const GraphProto &graph,
                          std::unordered_map<std::string, std::string> &value_tags);

// Applies the tag-inference rules for a single node ``node`` at index ``n``,
// updating ``value_tags`` and ``node_tags`` in place. ``has_custom_node_tag_override``
// (sized to the node count) persists whether a custom callback already set a
// node-tag override for a node across passes. ``ctx`` (may be null) provides
// custom per-op value-tag callbacks. When ``changed_values`` is non-null, the
// names of values whose tag changed are appended to it so incremental callers
// can re-queue dependent nodes. Returns whether anything changed for this node.
bool ProcessNodeTags(const NodeProto &node, std::size_t n,
                     std::unordered_map<std::string, std::string> &value_tags,
                     std::vector<std::string> &node_tags,
                     std::vector<char> &has_custom_node_tag_override, ComputeContext *ctx,
                     std::vector<std::string> *changed_values);

// Runs the fixed-point value/node tag inference over ``nodes``, updating
// ``value_tags`` and ``node_tags`` in place. ``ctx`` (may be null) provides
// custom per-op value-tag callbacks.
void InferNodesTags(const std::vector<const NodeProto *> &nodes,
                    std::unordered_map<std::string, std::string> &value_tags,
                    std::vector<std::string> &node_tags, ComputeContext *ctx);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const GraphProto &graph);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const FunctionProto &function);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes);

void WriteValueAndNodeTagsToMetadata(GraphProto &graph);
void WriteValueAndNodeTagsToMetadata(FunctionProto &function);
void WriteValueAndNodeTagsToMetadata(ModelProto &model);

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
