// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace annotations {

constexpr const char *kValueTagMetadataKey = "onnx_light.value_tag";
constexpr const char *kValueTagsMetadataKey = "onnx_light.value_tags";
constexpr const char *kNodeTagMetadataKey = "onnx_light.node_tag";

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const GraphProto &graph);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const FunctionProto &function);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const std::vector<NodeProto> &nodes);

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const std::vector<const NodeProto *> &nodes);

void WriteValueAndNodeTagsToMetadata(GraphProto &graph);
void WriteValueAndNodeTagsToMetadata(FunctionProto &function);
void WriteValueAndNodeTagsToMetadata(ModelProto &model);

} // namespace annotations
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
