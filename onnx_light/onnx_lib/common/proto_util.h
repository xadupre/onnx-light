// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/// @file proto_util.h
/// @brief Utilities for building canonical string identifiers for ONNX
///        functions and callee nodes.

#pragma once

#include "constants.h"
#include "onnx_pb.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE {

/// Opaque string identifier for a function specification (domain + op-type +
/// optional overload).
using FunctionSpecId = std::string;

/// Opaque string identifier for a concrete function implementation (domain +
/// op-type + optional overload), used as a lookup key in function registries.
using FunctionImplId = std::string;

/**
 * @brief Builds a canonical function-implementation identifier.
 *
 * Concatenates the normalized domain, op name, and (when non-empty) overload
 * into a "::" separated string.  The resulting value can be used as a stable
 * key in function-implementation maps.
 *
 * @param domain ONNX operator domain (e.g. @c "" or @c "ai.onnx.ml").
 * @param op     Operator type name.
 * @param overload Optional overload qualifier; omitted from the key when empty.
 * @returns A string of the form @c "domain::op" or @c "domain::op::overload".
 */
inline FunctionImplId GetFunctionImplId(const std::string &domain, const std::string &op,
                                        const std::string &overload) {
  if (overload.empty()) {
    return NormalizeDomain(domain) + "::" + op;
  }
  return NormalizeDomain(domain) + "::" + op + "::" + overload;
}

/**
 * @brief Builds a canonical function-implementation identifier from a
 *        FunctionProto.
 *
 * Convenience overload that extracts the domain, name, and overload fields
 * from @p function and delegates to the primary overload.
 *
 * @param function FunctionProto whose @c domain, @c name, and @c overload
 *                 fields are used to construct the identifier.
 * @returns A string of the form @c "domain::name" or
 *          @c "domain::name::overload".
 */
inline FunctionImplId GetFunctionImplId(const FunctionProto &function) {
  return GetFunctionImplId(function.ref_domain().as_string(), function.ref_name().as_string(),
                           function.ref_overload().as_string());
}

/**
 * @brief Builds the callee identifier for a NodeProto.
 *
 * Constructs the identifier that matches the @c FunctionImplId of the function
 * called by @p node, using the node's domain, op-type, and overload fields.
 *
 * @param node NodeProto whose @c domain, @c op_type, and @c overload fields
 *             identify the called function.
 * @returns A string of the form @c "domain::op_type" or
 *          @c "domain::op_type::overload".
 */
inline FunctionImplId GetCalleeId(const NodeProto &node) {
  return GetFunctionImplId(node.ref_domain().as_string(), node.ref_op_type().as_string(),
                           node.ref_overload().as_string());
}

} // namespace ONNX_LIGHT_NAMESPACE
