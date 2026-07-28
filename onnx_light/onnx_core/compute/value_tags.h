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

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace compute {

class ComputeContext;

constexpr const char *kValueTagMetadataKey = "onnx_light.value_tag";
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

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const std::vector<NodeProto> &nodes);

void WriteValueAndNodeTagsToMetadata(GraphProto &graph);
void WriteValueAndNodeTagsToMetadata(FunctionProto &function);
void WriteValueAndNodeTagsToMetadata(ModelProto &model);

} // namespace compute
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
