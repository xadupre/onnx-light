// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file printer.h
 * @brief Declares stream-formatting helpers for ONNX protobuf structures.
 *
 * This header exposes ostream operator overloads that render ONNX protobuf
 * objects in a compact textual form for diagnostics and debugging, plus a
 * helper template to convert supported protobuf objects to std::string.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "onnx/defs/parser.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Streams a TensorShapeProto_Dimension to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const TensorShapeProto_Dimension &dim);

/**
 * Streams a TensorShapeProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const TensorShapeProto &shape);

/**
 * Streams a TypeProto_Tensor to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const TypeProto_Tensor &tensortype);

/**
 * Streams a TypeProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const TypeProto &type);

/**
 * Streams a TensorProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const TensorProto &tensor);

/**
 * Streams a ValueInfoProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const ValueInfoProto &value_info);

/**
 * Streams a ValueInfoList to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const ValueInfoList &vilist);

/**
 * Streams an AttributeProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const AttributeProto &attr);

/**
 * Streams an AttrList to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const AttrList &attrlist);

/**
 * Streams a NodeProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const NodeProto &node);

/**
 * Streams a NodeList to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const NodeList &nodelist);

/**
 * Streams a GraphProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const GraphProto &graph);

/**
 * Streams a FunctionProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const FunctionProto &fn);

/**
 * Streams a ModelProto to an output stream.
 */
std::ostream &operator<<(std::ostream &os, const ModelProto &model);

/**
 * Converts a streamable ONNX protobuf object to its textual representation.
 *
 * @tparam ProtoType Type supporting the corresponding stream operator.
 * @param proto Object to format.
 * @return Formatted textual representation of proto.
 */
template <typename ProtoType> std::string ProtoToString(const ProtoType &proto) {
  std::stringstream ss;
  ss << proto;
  return ss.str();
}

} // namespace ONNX_LIGHT_NAMESPACE
