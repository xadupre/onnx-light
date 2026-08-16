// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_sequence.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/**
 * @file node_helpers.h
 * @brief Small helpers shared by the dispatcher (``run_nodes.cc``) and
 *        the kernel dispatch table (``kernel_dispatch_table.cc``).
 *
 * The helpers normalise the default ONNX domain, validate input/output
 * arity declared on a ``NodeProto``, look up tensors by name in a
 * ``RuntimeContext`` and read the most common attribute types
 * (``INT``, ``INTS``, ``FLOAT``, ``STRING``). Their definitions live in
 * ``node_helpers.cc`` (part of ``lib_onnx_core``).
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/// Vector-like wrapper for the value returned by
/// :func:`GetAttributeIntsOrDefault`.
class ParamInts : public std::vector<int64_t> {
public:
  using std::vector<int64_t>::vector;
  ParamInts() = default;
  ParamInts(const std::vector<int64_t> &values) : std::vector<int64_t>(values) {}
  ParamInts(std::vector<int64_t> &&values) : std::vector<int64_t>(std::move(values)) {}
};

/// Vector-like wrapper for the value returned by
/// :func:`GetAttributeFloatsOrDefault`.
class ParamFloats : public std::vector<float> {
public:
  using std::vector<float>::vector;
  ParamFloats() = default;
  ParamFloats(const std::vector<float> &values) : std::vector<float>(values) {}
  ParamFloats(std::vector<float> &&values) : std::vector<float>(std::move(values)) {}
};

/// Vector-like wrapper for the value returned by
/// :func:`GetAttributeStringsOrDefault`.
class ParamStrings : public std::vector<std::string> {
public:
  using std::vector<std::string>::vector;
  ParamStrings() = default;
  ParamStrings(const std::vector<std::string> &values) : std::vector<std::string>(values) {}
  ParamStrings(std::vector<std::string> &&values) : std::vector<std::string>(std::move(values)) {}
};

const Tensor &GetInput(const NodeProto &node, int index, const TensorMap &tensors);

// Same as :func:`GetInput` but returns ``nullptr`` when the input slot
// is either absent (``index >= node.input_size()``) or declared with
// an empty name (the ONNX convention for an unconnected optional
// input). Throws if the slot has a non-empty name but the tensor is
// missing from ``tensors``, since that indicates a graph-wiring bug
// rather than an "absent" optional input.
const Tensor *GetOptionalInput(const NodeProto &node, int index, const TensorMap &tensors);

void SetOutput(const NodeProto &node, int index, Tensor result, TensorMap &tensors);

// Overload that routes the assignment through :cpp:func:`RuntimeContext::Put`
// so the tensor map mutation is recorded in the context's event log.
void SetOutput(const NodeProto &node, int index, Tensor result, RuntimeContext &rt);

// Looks up the sequence-typed input at slot ``index`` in
// ``rt.sequences()``. Throws when the input name is empty (unset
// optional sequence input) or when no sequence with that name has been
// produced by an earlier node / supplied by the caller.
const Sequence &GetInputSequence(const NodeProto &node, int index, const RuntimeContext &rt);

// Routes a freshly-produced sequence to the output slot ``index`` via
// :cpp:func:`RuntimeContext::PutSequence`. Throws when the slot's name
// is empty.
void SetOutputSequence(const NodeProto &node, int index, Sequence result, RuntimeContext &rt);

void RequireInputCount(const NodeProto &node, int expected);

void RequireMinInputCount(const NodeProto &node, int min_expected);

void RequireOutputCount(const NodeProto &node, int expected);

const AttributeProto *FindAttribute(const NodeProto &node, std::string_view name);

const GraphProto &GetRequiredGraphAttribute(const NodeProto &node, std::string_view name);

int64_t GetAttributeIntOrDefault(const NodeProto &node, const std::string &name, int64_t fallback);

ParamInts GetAttributeIntsOrDefault(const NodeProto &node, const std::string &name,
                                    const std::vector<int64_t> &fallback);

/// Reads the repeated INTS attribute ``name`` of ``node`` and returns it as a
/// ``Shape``. When the attribute is absent, ``fallback`` is returned unchanged.
/// Throws when the attribute exists but has a type other than INTS.
Shape GetAttributeShapeOrDefault(const NodeProto &node, const std::string &name,
                                 const Shape &fallback);

ParamFloats GetAttributeFloatsOrDefault(const NodeProto &node, const std::string &name,
                                        const std::vector<float> &fallback);

ParamStrings GetAttributeStringsOrDefault(const NodeProto &node, const std::string &name,
                                          const std::vector<std::string> &fallback);

float GetAttributeFloatOrDefault(const NodeProto &node, const std::string &name, float fallback);

std::string GetAttributeStringOrDefault(const NodeProto &node, const std::string &name,
                                        const std::string &fallback);

std::string GetRequiredAttributeString(const NodeProto &node, const std::string &name);

int64_t GetRequiredAttributeInt(const NodeProto &node, const std::string &name);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
