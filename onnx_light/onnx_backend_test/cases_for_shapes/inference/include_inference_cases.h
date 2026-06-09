// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Multi-node backend test cases dedicated to exercising the shape-inference
// path. Each model assembles several operators in a row so the
// :cpp:func:`shape_inference::InferShapes` pass has to propagate (and, when
// possible, resolve) tensor shapes through every intermediate value.
//
// The models mirror the examples documented in
// ``yet-another-onnx-builder``'s
// ``auto_examples_core/plot_computed_shapes.html`` gallery page:
//
//   * ``Add → Concat → Reshape`` — symbolic-shape propagation through
//     element-wise and shape-changing ops (the ``[0, 0, -1]`` reshape pattern
//     preserves the leading dimensions).
//   * ``Abs → Relu → Add → Mul → NonZero → Transpose → Cast`` — a chain that
//     ends in a data-dependent operator (``NonZero``) which introduces a
//     fresh symbolic dimension that propagates through ``Transpose`` and
//     ``Cast``. Two flavours are registered: one with anonymous output
//     value-info shapes (``[None, None]``), one with named output dimensions
//     (e.g. ``["rank", "nnz"]``).
//
// The cases use small concrete shapes so the models remain executable; the
// generic ``BackendTestCaseShapeInference`` tests
// (see ``unittests/cc_onnx_backend_test/test_backend_shape_inference.cc``)
// also exercise the symbolic-dim propagation path on top of the same cases.
// ---------------------------------------------------------------------------

/// Registers a multi-node ``Add → Concat → Reshape`` case.
void RegisterAddConcatReshapeShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a multi-node ``Abs → Relu → Add → Mul → NonZero → Transpose →
/// Cast`` case with named output value-info dimensions.
void RegisterNonZeroChainNamedShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a multi-node ``Shape → Identity → Unsqueeze`` case that
/// exercises shape-data propagation through ``Shape``/``Identity`` and the
/// INT64 ``axes`` initializer path of ``Unsqueeze``. Mirrors the upstream
/// onnxruntime regression model from
/// https://github.com/microsoft/onnxruntime/pull/28778.
void RegisterShapeIdentityUnsqueezeShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a single-node case whose op is a call to a **model-local
/// function** (declared in ``ModelProto::functions``). The function body
/// is a one-node ``Add`` of two same-shape inputs. Exercises the
/// FunctionProto-expansion path of ``onnx_optim`` shape inference.
void RegisterLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a single-node case whose op is a call to a **model-local
/// function whose body itself calls another model-local function**. The
/// outer function is ``local:func_outer_add(a, b) -> c`` whose body is a
/// single call into ``local:func_inner_add(a, b) -> c { c = Add(a, b) }``.
/// Exercises the recursive FunctionProto-expansion path of ``onnx_optim``
/// shape inference, including the forwarding of the local-function map
/// into nested sub-contexts so nested calls are dispatched too.
void RegisterNestedLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a model including a NonZero followed by an expression.
/// Expressions must be simplified.
void RegisterDimensionExpressionShapeInferenceCase(std::vector<TestCase> &registry);

/// Collects all shape-inference oriented backend test cases by invoking
/// every ``Register*ShapeInferenceCases`` helper declared in this header.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry,
                                    const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
