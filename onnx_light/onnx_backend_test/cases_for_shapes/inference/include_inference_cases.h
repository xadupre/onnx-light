// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

// The per-operator ``Register*`` helpers used to build these cases live in the
// library-private companion header below and are only needed while compiling
// ``lib_onnx_backend_test`` itself; external consumers only see ``Collect*``.
#ifdef ONNX_LIGHT_BACKEND_TEST_INTERNAL
#include "onnx_backend_test/cases_for_shapes/inference/register_inference_cases.h"
#endif
namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/// Registers a single-node case exercising **shape-as-value propagation
/// through a local-function call boundary** when the function body contains
/// a ``Range`` node whose ``limit`` input is the function's own input
/// parameter. The caller passes the graph initializer
/// ``limit_val : int64[] = 5`` which is seeded with ``ValueAsShape = [5]``;

/// Registers a ``Concat → Split → Concat → Relu`` case translated from the
/// ``test_concat_split`` example in https://github.com/xadupre/
/// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py.
/// Exercises Concat / Split shape propagation when the concat axis dims
/// are symbolic. When ``even`` is ``true``, the split sizes are equal;

/// Registers a ``TopK(K, axis=-1) → TopK(K, axis=-1) → ReduceMean`` case
/// where both TopK nodes share the **same** runtime K input (INT64 ``[1]``).
/// Because K is unknown at shape-inference time, each TopK emits a fresh
/// symbolic dim (``TopK_k`` and ``TopK_k`` respectively);

/// Registers a ``TopK(K1, axis=-1) → TopK(K2, axis=-1) → ReduceMean`` case
/// where the two TopK nodes use **different** runtime K inputs (K1 > K2).
/// Because both K values are unknown at shape-inference time, each TopK emits
/// a distinct symbolic dim (``TopK_k`` and ``TopK_k``);

/// Registers a 4-layer Qwen3-style causal language model (opset 21, IR 10)
/// reproduced from a PyTorch-exported graph. The model uses GQA-style
/// attention (16 Q / 8 KV heads, head-dim 128), manual RMSNorm, RoPE
/// embeddings, causal masking and a SwiGLU MLP per layer.  External weight
/// initializers are replaced with deterministic random FP16 values;

/// Collects all shape-inference oriented backend test cases by invoking
/// every ``Register*ShapeInferenceCases`` helper declared in this header.
/// @param include_big When ``false`` (the default), test cases whose name
///                    contains ``"_big_"`` are excluded from the output.
///                    Pass ``true`` to also include those large cases.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry,
                                    const std::string &op_type = "", bool include_big = false,
                                    TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
