// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "constants.h"
#include "onnx_pb.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE {

using FunctionSpecId = std::string;
using FunctionImplId = std::string;

inline FunctionImplId GetFunctionImplId(const std::string &domain, const std::string &op,
                                        const std::string &overload) {
  if (overload.empty()) {
    return NormalizeDomain(domain) + "::" + op;
  }
  return NormalizeDomain(domain) + "::" + op + "::" + overload;
}

inline FunctionImplId GetFunctionImplId(const FunctionProto &function) {
  return GetFunctionImplId(function.ref_domain().as_string(), function.ref_name().as_string(),
                           function.ref_overload().as_string());
}

inline FunctionImplId GetCalleeId(const NodeProto &node) {
  return GetFunctionImplId(node.ref_domain().as_string(), node.ref_op_type().as_string(),
                           node.ref_overload().as_string());
}

} // namespace ONNX_LIGHT_NAMESPACE
