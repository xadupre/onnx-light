// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file attr_proto_util.h
 * @brief Declares helpers to construct ONNX AttributeProto instances.
 *
 * This header provides overloads for building typed attributes and helper
 * utilities to create reference attributes used in function-body nodes.
 */

#pragma once

#include <string>
#include <vector>

#include "onnx/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {

/// Creates a FLOAT attribute with the provided value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, float value);
/// Creates an INT attribute with a 64-bit integer value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, int64_t value);
/// Creates an INT attribute from a 32-bit integer value (stored as int64).
ONNX_API AttributeProto MakeAttribute(std::string attr_name, int value);
/// Creates a STRING attribute with the provided UTF-8 text value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::string value);
/// Creates a TENSOR attribute with an inline tensor value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, TensorProto value);
/// Creates a GRAPH attribute with a subgraph value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, GraphProto value);
/// Creates a TYPE_PROTO attribute with a type description value.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, TypeProto value);
/// Creates a FLOATS attribute from a list of floating-point values.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<float> values);
/// Creates an INTS attribute from a list of 64-bit integers.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<int64_t> values);
/// Creates a STRINGS attribute from a list of string values.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<std::string> values);
/// Creates a TENSORS attribute from a list of tensor values.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<TensorProto> values);
/// Creates a GRAPHS attribute from a list of graph values.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<GraphProto> values);
/// Creates a TYPE_PROTOS attribute from a list of type description values.
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<TypeProto> values);

/// Creates a reference attribute with identical source and target names.
AttributeProto MakeRefAttribute(const std::string &attr_name, AttributeProto::AttributeType type);

/// Creates a reference attribute with distinct function-body and source names.
AttributeProto MakeRefAttribute(const std::string &attr_name, const std::string &referred_attr_name,
                                AttributeProto::AttributeType type);

} // namespace ONNX_LIGHT_NAMESPACE
