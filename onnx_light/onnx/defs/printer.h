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
 * Formats and writes a TensorShapeProto_Dimension to an output stream.
 *
 * @param os The destination stream.
 * @param dim Dimension to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const TensorShapeProto_Dimension &dim);

/**
 * Formats and writes a TensorShapeProto to an output stream.
 *
 * @param os The destination stream.
 * @param shape Shape to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const TensorShapeProto &shape);

/**
 * Formats and writes a TypeProto_Tensor to an output stream.
 *
 * @param os The destination stream.
 * @param tensortype Tensor type to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const TypeProto_Tensor &tensortype);

/**
 * Formats and writes a TypeProto to an output stream.
 *
 * @param os The destination stream.
 * @param type Type to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const TypeProto &type);

/**
 * Formats and writes a TensorProto to an output stream.
 *
 * @param os The destination stream.
 * @param tensor Tensor to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const TensorProto &tensor);

/**
 * Formats and writes a ValueInfoProto to an output stream.
 *
 * @param os The destination stream.
 * @param value_info Value info to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const ValueInfoProto &value_info);

/**
 * Formats and writes a ValueInfoList to an output stream.
 *
 * @param os The destination stream.
 * @param vilist Value info list to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const ValueInfoList &vilist);

/**
 * Formats and writes an AttributeProto to an output stream.
 *
 * @param os The destination stream.
 * @param attr Attribute to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const AttributeProto &attr);

/**
 * Formats and writes an AttrList to an output stream.
 *
 * @param os The destination stream.
 * @param attrlist Attribute list to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const AttrList &attrlist);

/**
 * Formats and writes a NodeProto to an output stream.
 *
 * @param os The destination stream.
 * @param node Node to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const NodeProto &node);

/**
 * Formats and writes a NodeList to an output stream.
 *
 * @param os The destination stream.
 * @param nodelist Node list to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const NodeList &nodelist);

/**
 * Formats and writes a GraphProto to an output stream.
 *
 * @param os The destination stream.
 * @param graph Graph to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const GraphProto &graph);

/**
 * Formats and writes a FunctionProto to an output stream.
 *
 * @param os The destination stream.
 * @param fn Function to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const FunctionProto &fn);

/**
 * Formats and writes a ModelProto to an output stream.
 *
 * @param os The destination stream.
 * @param model Model to format.
 * Returns: A reference to os.
 */
std::ostream &operator<<(std::ostream &os, const ModelProto &model);

/**
 * Converts a streamable ONNX protobuf object to its textual representation.
 *
 * @tparam ProtoType Type supporting the corresponding stream operator.
 * @param proto Object to format.
 * Returns: The formatted textual representation of proto.
 */
template <typename ProtoType> std::string ProtoToString(const ProtoType &proto) {
  std::stringstream ss;
  ss << proto;
  return ss.str();
}

} // namespace ONNX_LIGHT_NAMESPACE
