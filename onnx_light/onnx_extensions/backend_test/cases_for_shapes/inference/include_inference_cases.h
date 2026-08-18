// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

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
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterAddConcatReshapeShapeInferenceCases(std::vector<TestCase> &registry,
                                            TestMode mode = TestMode::TEST);

/// Registers a multi-node ``Abs → Relu → Add → Mul → NonZero → Transpose →
/// Cast`` case with named output value-info dimensions.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterNonZeroChainNamedShapeInferenceCases(std::vector<TestCase> &registry,
                                             TestMode mode = TestMode::TEST);

/// Registers a multi-node ``Shape → Identity → Unsqueeze`` case that
/// exercises shape-data propagation through ``Shape``/``Identity`` and the
/// INT64 ``axes`` initializer path of ``Unsqueeze``. Mirrors the upstream
/// onnxruntime regression model from
/// https://github.com/microsoft/onnxruntime/pull/28778.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeIdentityUnsqueezeShapeInferenceCases(std::vector<TestCase> &registry,
                                                  TestMode mode = TestMode::TEST);

/// Registers a single-node case whose op is a call to a **model-local
/// function** (declared in ``ModelProto::functions``). The function body
/// is a one-node ``Add`` of two same-shape inputs. Exercises the
/// FunctionProto-expansion path of ``onnx_shapes`` shape inference.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry,
                                            TestMode mode = TestMode::TEST);

/// Registers a single-node case exercising **shape-as-value propagation
/// through a local-function call boundary** when the function body contains
/// a ``Range`` node whose ``limit`` input is the function's own input
/// parameter. The caller passes the graph initializer
/// ``limit_val : int64[] = 5`` which is seeded with ``ValueAsShape = [5]``;
/// ``ExpandLocalFunctionCall`` must copy that annotation into the
/// sub-context so ``ComputeShapeRange`` can resolve the output length to 5.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterLocalFunctionRangeShapeInferenceCases(std::vector<TestCase> &registry,
                                              TestMode mode = TestMode::TEST);

/// Registers a single-node case whose op is a call to a **model-local
/// function whose body itself calls another model-local function**. The
/// outer function is ``local:func_outer_add(a, b) -> c`` whose body is a
/// single call into ``local:func_inner_add(a, b) -> c { c = Add(a, b) }``.
/// Exercises the recursive FunctionProto-expansion path of ``onnx_shapes``
/// shape inference, including the forwarding of the local-function map
/// into nested sub-contexts so nested calls are dispatched too.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterNestedLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry,
                                                  TestMode mode = TestMode::TEST);

/// Registers a model including a NonZero followed by an expression.
/// Expressions must be simplified.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterDimensionExpressionShapeInferenceCase(std::vector<TestCase> &registry);

/// Registers a MaxPool case whose output spatial dim simplifies from
/// ``(seq + 10) // 5`` to ``seq//5+2``.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterFloorDivOffsetShapeInferenceCase(std::vector<TestCase> &registry);

/// Registers a ``Slice(axis=2, starts=0, ends=-1) → Abs`` case on symbolic
/// input ``X[a,b,c]`` to exercise symbolic Slice-length inference
/// (``c-1``) without creating fresh dimension names.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterSliceSymbolicEndShapeInferenceCases(std::vector<TestCase> &registry,
                                            TestMode mode = TestMode::TEST);

/// Registers a ``Loop`` case that computes the pairwise Euclidean distance
/// matrix of an input ``X`` of shape ``[N, D]``. The Loop iterates ``N``
/// times: each iteration gathers one row of the outer-scope ``X`` and emits
/// the row of distances to every other row as a FLOAT ``[N]`` scan output.
/// Stacking the per-iteration scan outputs across the ``N`` iterations
/// produces the ``[N, N]`` distance matrix. Exercises shape inference
/// through a non-trivial ``Loop`` body, including outer-scope reference
/// from inside the body subgraph.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterLoopPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry,
                                                TestMode mode = TestMode::TEST);

/// Registers an ``Unsqueeze → Unsqueeze → Sub → Mul → ReduceSum → Sqrt →
/// TopK → ReduceMean`` case that computes the pairwise Euclidean distance
/// matrix of an input ``X`` of symbolic shape ``[N, D]``, keeps the ``k``
/// largest distances of each row and averages them. The TopK ``k`` is a
/// **model input** (INT64 ``[1]``) so its value is unknown at shape-inference
/// time: ``TopK`` must emit a fresh symbolic dim for its output axis, which
/// ``ReduceMean`` then reduces away to recover the concrete-rank ``[N]``
/// output. Exercises symbolic-dim propagation through broadcasting and a
/// data-dependent ``TopK`` axis.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry,
                                                TestMode mode = TestMode::TEST);

/// Registers a ``Shape → Gather → Loop → TopK → ReduceMean`` case that
/// computes the pairwise Euclidean distance matrix of an input ``X`` of
/// symbolic shape ``[N, D]`` via a ``Loop`` (one row of the ``[N, N]`` matrix
/// per iteration), keeps the ``k`` largest distances of each row and averages
/// them. The Loop trip count comes from ``Shape(X)[0]`` (runtime), so the
/// stacked matrix has a symbolic leading axis, and the TopK ``k`` is a
/// **model input** (INT64 ``[1]``) so ``TopK`` must emit a fresh symbolic dim
/// for its output axis, which ``ReduceMean`` then reduces away to recover the
/// rank-1 output. Exercises symbolic-dim propagation through a non-trivial
/// ``Loop`` body and a data-dependent ``TopK`` axis.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterLoopTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers a ``Scan → Sqrt → TopK → ReduceMean`` case that computes the
/// pairwise Euclidean distance matrix of an input ``X`` of symbolic shape
/// ``[N, D]`` via a ``Scan`` (``X`` is both scan input and carried state so
/// each row broadcasts against the full matrix), keeps the ``k`` largest
/// distances of each row and averages them. The Scan trip count comes from
/// ``X``'s scan axis (``N``), and the TopK ``k`` is a **model input**
/// (INT64 ``[1]``) so ``TopK`` must emit a fresh symbolic dim for its output
/// axis, which ``ReduceMean`` then reduces away to recover the rank-1 output.
/// Exercises symbolic-dim propagation through a non-trivial ``Scan`` body and
/// a data-dependent ``TopK`` axis.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterScanTopKPairwiseDistanceShapeInferenceCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers a ``Scan`` case that computes the running (cumulative) row sum
/// of an input ``X`` of shape ``[T, D]``. Each Scan iteration accumulates
/// one row into a running state (initially zeros) and emits the accumulated
/// sum as a per-iteration scan output. Stacking the ``T`` outputs produces
/// the cumulative-sum matrix ``Y_pre_abs`` of shape ``[T, D]``; the final
/// output ``Y = Abs(Y_pre_abs)`` exercises shape propagation through one
/// node after the ``Scan``. Exercises :cpp:func:`ComputeShapeScan` state
/// propagation and scan-output stacking.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterScanRunningSumShapeInferenceCases(std::vector<TestCase> &registry,
                                          TestMode mode = TestMode::TEST);

/// Registers the ``Shape → Shape → Concat → Add → Sub → Expand → 3 × Add →
/// Add → Add`` value-as-shape case translated from
/// https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_value_as_shape.py. Exercises value-as-shape
/// propagation through ``Shape``/``Concat``/``Add``/``Sub`` so the
/// downstream ``Expand`` can recover the precise symbolic output shape.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterValueAsShapeShapeInferenceCases(std::vector<TestCase> &registry,
                                        TestMode mode = TestMode::TEST);

/// Registers a ``Shape → Gather → Expand → Abs`` case whose input ``x``
/// carries symbolic dims ``[N, D]``. ``Shape(x)`` lifts the symbolic dims into
/// an INT64 tensor with a *value-as-shape* annotation ``[N, D]``. ``Gather``
/// with a constant index ``[0]`` then slices the annotation to produce an
/// INT64 ``[1]`` tensor with VAS ``[N]``. ``Expand`` consumes it as the
/// target shape so shape inference must recover the precise output
/// shape ``float[N]``. Directly exercises the VAS-propagation logic in
/// :cpp:func:`onnx_shapes::shapes::tensor::ComputeShapeGather`.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterGatherValueAsShapeShapeInferenceCases(std::vector<TestCase> &registry,
                                              TestMode mode = TestMode::TEST);

/// Registers a single-node ``If`` model whose ``then_branch`` and
/// ``else_branch`` each produce **two** outputs of the same rank but with
/// *different* symbolic shapes (the leading axis differs, every trailing
/// axis matches). Exercises the branch-merging path of
/// :cpp:func:`onnx_shapes::shapes::controlflow::ComputeShapeIf`, which must
/// keep matching axes and synthesize a fresh ``If_<out>_d<i>`` symbolic
/// dim for the differing one.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterIfSymbolicShapesShapeInferenceCases(std::vector<TestCase> &registry,
                                            TestMode mode = TestMode::TEST);

/// Registers an ``Unsqueeze → Unsqueeze → Reshape → Reshape → Cast →
/// MatMul → Reshape`` case translated from the ``test_check_shape`` example
/// in https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_shape_builder.py. Exercises shape inference through
/// rank-changing ``Unsqueeze`` / ``Reshape`` and through ``MatMul`` of two
/// 3-D tensors.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterCheckShapeShapeInferenceCases(std::vector<TestCase> &registry,
                                      TestMode mode = TestMode::TEST);

/// Registers a ``Reshape → Reshape → Add`` case translated from the
/// ``test_reshape_reshape`` example in https://github.com/xadupre/
/// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py.
/// Exercises shape inference through chained reshapes with the ``[0, 0, …]``
/// "carry-over" pattern.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterReshapeReshapeShapeInferenceCases(std::vector<TestCase> &registry,
                                          TestMode mode = TestMode::TEST);

/// Registers a ``Shape → Concat → 3 × MatMul → 3 × Reshape → 3 × Transpose``
/// case translated from the ``test_value_as_shape`` example in
/// https://github.com/xadupre/yet-another-onnx-builder/blob/main/
/// unittests/xshape/test_shape_builder.py. Exercises value-as-shape
/// propagation: ``new_shape`` is built at graph-runtime from a ``Shape``
/// node + a constant ``[32, 8]`` initializer and is then consumed by
/// ``Reshape``.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterValueAsShapeBuilderShapeInferenceCases(std::vector<TestCase> &registry,
                                               TestMode mode = TestMode::TEST);

/// Registers a ``Concat → Split → Concat → Relu`` case translated from the
/// ``test_concat_split`` example in https://github.com/xadupre/
/// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py.
/// Exercises Concat / Split shape propagation when the concat axis dims
/// are symbolic. When ``even`` is ``true``, the split sizes are equal;
/// when ``false``, the split sizes differ.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterConcatSplitShapeInferenceCases(std::vector<TestCase> &registry, bool even);

/// Registers a ``Resize(scales=[0.5, 0.5]) → Tile(repeats=[2, 2])`` case
/// whose input ``X`` carries symbolic dimensions ``H`` (odd concrete value)
/// and ``W`` (even concrete value). Exercises shape inference through
/// ``Resize`` with a FLOAT ``scales`` initializer (output dims become fresh
/// ``Resize_dim{i}`` symbols) followed by ``Tile`` with an INT64
/// ``repeats`` initializer (output dims become ``Tile_dim{i}`` because the
/// input dims are still symbolic).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterResizeTileShapeInferenceCases(std::vector<TestCase> &registry,
                                      TestMode mode = TestMode::TEST);

/// Registers a ``Pad(reflect) → Conv(canny) → Sub(ReduceMean)`` case whose
/// single input ``X`` is a grayscale image batch ``float[N, 1, H, W]`` with
/// symbolic spatial dims. The image is reflect-padded by one pixel, filtered
/// with a 3×3 Laplacian (Canny-style edge) ``Conv`` and finally has its
/// global average removed via ``Sub`` with a ``ReduceMean`` over every axis.
/// Exercises symbolic-dim propagation through ``Pad`` (symbolic ``H+2`` /
/// ``W+2`` expressions), ``Conv`` (which collapses them back to ``H`` / ``W``)
/// and broadcasting ``Sub`` against a reduced ``[1, 1, 1, 1]`` mean.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterPadCannyAverageShapeInferenceCases(std::vector<TestCase> &registry,
                                           TestMode mode = TestMode::TEST);

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
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTinyLlmShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the same single Llama-style decoder layer as
/// :cpp:func:`RegisterTinyLlmShapeInferenceCases` but with the fused
/// ``RMSNormalization`` and ``Attention`` operators **inlined** into their
/// primitive subgraphs (``Mul`` / ``ReduceMean`` / ``Add`` / ``Sqrt`` / ``Div``
/// for RMSNorm; ``Reshape`` / ``Transpose`` / ``Concat`` / ``MatMul`` /
/// ``Softmax`` for scaled dot-product attention with a KV cache). Exercises
/// shape inference through the longer chains an exporter emits when those
/// operators are decomposed, while keeping the same four dynamic inputs and
/// three outputs as the fused companion.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTinyLlmInlinedShapeInferenceCases(std::vector<TestCase> &registry,
                                          TestMode mode = TestMode::TEST);

/// Registers a ``TopK(K, axis=-1) → TopK(K, axis=-1) → ReduceMean`` case
/// where both TopK nodes share the **same** runtime K input (INT64 ``[1]``).
/// Because K is unknown at shape-inference time, each TopK emits a fresh
/// symbolic dim (``TopK_k`` and ``TopK_k`` respectively);
/// ``ReduceMean`` then collapses the second symbolic axis to recover the
/// rank-1 output ``Y [N]``. Exercises shape inference through two chained
/// TopK nodes that share the same K but produce distinct symbolic axes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTwoTopKSameKShapeInferenceCases(std::vector<TestCase> &registry,
                                        TestMode mode = TestMode::TEST);

/// Registers a ``TopK(K1, axis=-1) → TopK(K2, axis=-1) → ReduceMean`` case
/// where the two TopK nodes use **different** runtime K inputs (K1 > K2).
/// Because both K values are unknown at shape-inference time, each TopK emits
/// a distinct symbolic dim (``TopK_k`` and ``TopK_k``);
/// ``ReduceMean`` then collapses the second symbolic axis to recover the
/// rank-1 output ``Y [N]``. Exercises shape inference through two chained
/// TopK nodes with independent symbolic K axes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTwoTopKDifferentKShapeInferenceCases(std::vector<TestCase> &registry,
                                             TestMode mode = TestMode::TEST);

/// Registers a ``Shape → Gather → Unsqueeze → Concat → Reshape`` case on
/// inputs ``y: float[M, D1]`` and ``z: float[K, D2]`` whose dim-1 symbolic
/// values are extracted via ``Gather``, wrapped into 1-D tensors via
/// ``Unsqueeze``, concatenated into a 2-element ``new_shape`` tensor, and
/// consumed by ``Reshape`` applied to ``x: float[D1, D2]``. Directly
/// exercises the VAS-forwarding fix in
/// :cpp:func:`onnx_shapes::shapes::tensor::ComputeShapeUnsqueeze`: without
/// VAS propagation through ``Unsqueeze``, ``Concat`` never sees the
/// per-element symbolic values and ``Reshape`` falls back to inventing
/// undefined placeholder names instead of the real dims ``D1``/``D2``.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterUnsqueezeVasReshapeShapeInferenceCases(std::vector<TestCase> &registry,
                                               TestMode mode = TestMode::TEST);

/// Registers a 4-layer Qwen3-style causal language model (IR 10) reproduced
/// from a PyTorch-exported graph, in two variants selected by an internal
/// switch: an unfused variant
/// (``test_cc_shape_inference_big_qwen3_4_layers_like``, opset 21) that spells
/// out RMSNorm and the grouped-query attention core as explicit subgraphs, and
/// a fused variant (``..._fused``, opset 23) that expresses every RMSNorm with
/// a single ``RMSNormalization`` node and the attention core with a single
/// ``Attention`` node. Both share the same signature and architecture (GQA
/// with 16 Q / 8 KV heads, head-dim 128, RoPE, causal masking and a SwiGLU
/// MLP per layer); RoPE and the causal-mask construction remain explicit in
/// both as they have no fused-operator equivalent. External weight
/// initializers carry shape/dtype metadata only; doc_strings are omitted.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterQwen3_4LayersLikeShapeInferenceCases(std::vector<TestCase> &registry,
                                             TestMode mode = TestMode::TEST);

/// Collects all shape-inference oriented backend test cases by invoking
/// every ``Register*ShapeInferenceCases`` helper declared in this header.
/// @param include_big When ``false`` (the default), test cases whose name
///                    contains ``"_big_"`` are excluded from the output.
///                    Pass ``true`` to also include those large cases.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry,
                                    const std::string &op_type = "", bool include_big = false,
                                    TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
