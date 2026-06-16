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

#include "onnx_lib/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Creates a FLOAT attribute with the provided value.
 *
 * @param attr_name The name of the attribute.
 * @param value The floating-point value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, float value);
/**
 * Creates an INT attribute with a 64-bit integer value.
 *
 * @param attr_name The name of the attribute.
 * @param value The 64-bit integer value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, int64_t value);
/**
 * Creates an INT attribute with a 32-bit integer value (converted to int64_t).
 *
 * @param attr_name The name of the attribute.
 * @param value The 32-bit integer value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, int value);
/**
 * Creates a STRING attribute with the provided UTF-8 text value.
 *
 * @param attr_name The name of the attribute.
 * @param value The string value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::string value);
/**
 * Creates a TENSOR attribute with an inline tensor value.
 *
 * @param attr_name The name of the attribute.
 * @param value The tensor value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, TensorProto value);
/**
 * Creates a GRAPH attribute with a subgraph value.
 *
 * @param attr_name The name of the attribute.
 * @param value The graph value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, GraphProto value);
/**
 * Creates a TYPE_PROTO attribute with a type description value.
 *
 * @param attr_name The name of the attribute.
 * @param value The type-proto value assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, TypeProto value);
/**
 * Creates a FLOATS attribute from a list of floating-point values.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of floating-point values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<float> values);
/**
 * Creates an INTS attribute from a list of 64-bit integers.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of 64-bit integer values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<int64_t> values);
/**
 * Creates a STRINGS attribute from a list of string values.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of string values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<std::string> values);
/**
 * Creates a TENSORS attribute from a list of tensor values.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of tensor values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<TensorProto> values);
/**
 * Creates a GRAPHS attribute from a list of graph values.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of graph values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<GraphProto> values);
/**
 * Creates a TYPE_PROTOS attribute from a list of type description values.
 *
 * @param attr_name The name of the attribute.
 * @param values The list of type-proto values assigned to the attribute.
 * @return The constructed AttributeProto instance.
 */
ONNX_API AttributeProto MakeAttribute(std::string attr_name, std::vector<TypeProto> values);

/**
 * Creates a reference attribute for a node in a function body.
 *
 * @param attr_name The attribute name used by both function and function-body nodes.
 * @param type The declared ONNX attribute kind for the reference.
 * @return The constructed AttributeProto instance.
 */
AttributeProto MakeRefAttribute(const std::string &attr_name, AttributeProto::AttributeType type);

/**
 * Creates a reference attribute for a node in a function body.
 *
 * @param attr_name The attribute name used by the function-body node.
 * @param referred_attr_name The attribute name on the surrounding function node to reference.
 * @param type The declared ONNX attribute kind for the reference.
 * @return The constructed AttributeProto instance.
 */
AttributeProto MakeRefAttribute(const std::string &attr_name, const std::string &referred_attr_name,
                                AttributeProto::AttributeType type);

} // namespace ONNX_LIGHT_NAMESPACE
