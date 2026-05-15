// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace shape_inference {

// Infers the output TypeProto for each output of a FunctionProto.
//
// Walks the function body and runs per-node type-and-shape inference using the
// OpSchemaRegistry.  Attribute references of the form @ref_attr_name are
// resolved against the supplied formal_attrs map before inference is called.
//
// Adapted from onnx/cpp2py_export.cc infer_function_output_types.
//
// Returns a vector of TypeProto objects, one per function output (in the same
// order as FunctionProto.output).  An output whose type cannot be inferred is
// returned as an empty TypeProto (has_type() == false).
std::vector<TypeProto> InferFunctionOutputTypes(const FunctionProto &function,
                                                const std::vector<TypeProto> &input_types,
                                                const std::vector<AttributeProto> &attributes);

} // namespace shape_inference
} // namespace ONNX_LIGHT_NAMESPACE
