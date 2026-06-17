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

/// Registers a ``Loop`` case that computes the pairwise Euclidean distance
/// matrix of an input ``X`` of shape ``[N, D]``. The Loop iterates ``N``
/// times: each iteration gathers one row of the outer-scope ``X`` and emits
/// the row of distances to every other row as a FLOAT ``[N]`` scan output.
/// Stacking the per-iteration scan outputs across the ``N`` iterations
/// produces the ``[N, N]`` distance matrix. Exercises shape inference
/// through a non-trivial ``Loop`` body, including outer-scope reference
/// from inside the body subgraph.
void RegisterLoopPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a ``Scan`` case that computes the running (cumulative) row sum
/// of an input ``X`` of shape ``[T, D]``. Each Scan iteration accumulates
/// one row into a running state (initially zeros) and emits the accumulated
/// sum as a per-iteration scan output. Stacking the ``T`` outputs produces
/// the cumulative-sum matrix ``Y_pre_abs`` of shape ``[T, D]``; the final
/// output ``Y = Abs(Y_pre_abs)`` exercises shape propagation through one
/// node after the ``Scan``. Exercises :cpp:func:`ComputeShapeScan` state
/// propagation and scan-output stacking.
void RegisterScanRunningSumShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers the ``Shape → Shape → Concat → Add → Sub → Expand → 3 × Add →
/// Add → Add`` value-as-shape case translated from
/// https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_value_as_shape.py. Exercises value-as-shape
/// propagation through ``Shape``/``Concat``/``Add``/``Sub`` so the
/// downstream ``Expand`` can recover the precise symbolic output shape.
void RegisterValueAsShapeShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a single-node ``If`` model whose ``then_branch`` and
/// ``else_branch`` each produce **two** outputs of the same rank but with
/// *different* symbolic shapes (the leading axis differs, every trailing
/// axis matches). Exercises the branch-merging path of
/// :cpp:func:`onnx_optim::shapes::controlflow::ComputeShapeIf`, which must
/// keep matching axes and synthesize a fresh ``If_<out>_d<i>`` symbolic
/// dim for the differing one.
void RegisterIfSymbolicShapesShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers an ``Unsqueeze → Unsqueeze → Reshape → Reshape → Cast →
/// MatMul → Reshape`` case translated from the ``test_check_shape`` example
/// in https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_shape_builder.py. Exercises shape inference through
/// rank-changing ``Unsqueeze`` / ``Reshape`` and through ``MatMul`` of two
/// 3-D tensors.
void RegisterCheckShapeShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a ``Reshape → Reshape → Add`` case translated from the
/// ``test_reshape_reshape`` example in https://github.com/xadupre/
/// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py.
/// Exercises shape inference through chained reshapes with the ``[0, 0, …]``
/// "carry-over" pattern.
void RegisterReshapeReshapeShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a ``Shape → Concat → 3 × MatMul → 3 × Reshape → 3 × Transpose``
/// case translated from the ``test_value_as_shape`` example in
/// https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_shape_builder.py. Exercises value-as-shape
/// propagation: ``new_shape`` is built at graph-runtime from a ``Shape``
/// node + a constant ``[32, 8]`` initializer and is then consumed by
/// ``Reshape``.
void RegisterValueAsShapeBuilderShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a ``Concat → Split → Concat → Relu`` case translated from the
/// ``test_concat_split`` example in https://github.com/xadupre/
/// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py.
/// Exercises Concat / Split shape propagation when the concat axis dims
/// are symbolic. When ``even`` is ``true``, the split sizes are equal;
/// when ``false``, the split sizes differ.
void RegisterConcatSplitShapeInferenceCases(std::vector<TestCase> &registry, bool even);

/// Registers a ``Resize(scales=[0.5, 0.5]) → Tile(repeats=[2, 2])`` case
/// whose input ``X`` carries symbolic dimensions ``H`` (odd concrete value)
/// and ``W`` (even concrete value). Exercises shape inference through
/// ``Resize`` with a FLOAT ``scales`` initializer (output dims become fresh
/// ``Resize_dim{i}`` symbols) followed by ``Tile`` with an INT64
/// ``repeats`` initializer (output dims become ``Tile_dim{i}`` because the
/// input dims are still symbolic).
void RegisterResizeTileShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a ``Pad(reflect) → Conv(canny) → Sub(ReduceMean)`` case whose
/// single input ``X`` is a grayscale image batch ``float[N, 1, H, W]`` with
/// symbolic spatial dims. The image is reflect-padded by one pixel, filtered
/// with a 3×3 Laplacian (Canny-style edge) ``Conv`` and finally has its
/// global average removed via ``Sub`` with a ``ReduceMean`` over every axis.
/// Exercises symbolic-dim propagation through ``Pad`` (symbolic ``H+2`` /
/// ``W+2`` expressions), ``Conv`` (which collapses them back to ``H`` / ``W``)
/// and broadcasting ``Sub`` against a reduced ``[1, 1, 1, 1]`` mean.
void RegisterPadCannyAverageShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers a single decoder layer of a tiny Llama-style causal language
/// model (mirroring ``arnir0/Tiny-LLM``) translated to ONNX. The model takes
/// the four inputs of a cached-generation step — ``input_ids``,
/// ``attention_mask``, ``past_key`` and ``past_value`` — with fully dynamic
/// (symbolic) shapes and random weight initializers, and produces the
/// next-token ``logits`` plus the updated ``present_key`` / ``present_value``
/// cache. Exercises shape inference through ``Gather`` (token embedding),
/// ``RMSNormalization``, the QKV / output / MLP ``MatMul`` projections, the
/// additive ``attention_mask`` path (``Cast`` / ``Unsqueeze`` / ``Sub`` /
/// ``Mul``), the SwiGLU activation and the ``Attention`` operator with a KV
/// cache.
void RegisterTinyLlmShapeInferenceCases(std::vector<TestCase> &registry);

/// Registers the same single Llama-style decoder layer as
/// :cpp:func:`RegisterTinyLlmShapeInferenceCases` but with the fused
/// ``RMSNormalization`` and ``Attention`` operators **inlined** into their
/// primitive subgraphs (``Mul`` / ``ReduceMean`` / ``Add`` / ``Sqrt`` / ``Div``
/// for RMSNorm; ``Reshape`` / ``Transpose`` / ``Concat`` / ``MatMul`` /
/// ``Softmax`` for scaled dot-product attention with a KV cache). Exercises
/// shape inference through the longer chains an exporter emits when those
/// operators are decomposed, while keeping the same four dynamic inputs and
/// three outputs as the fused companion.
void RegisterTinyLlmInlinedShapeInferenceCases(std::vector<TestCase> &registry);

/// Collects all shape-inference oriented backend test cases by invoking
/// every ``Register*ShapeInferenceCases`` helper declared in this header.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry,
                                    const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
