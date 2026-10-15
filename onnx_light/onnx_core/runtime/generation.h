// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_session.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Configures autoregressive token generation. */
struct GenerationOptions {
  int64_t max_new_tokens = 20;
  /// Zero selects greedy decoding; positive values sample softmax(logits / temperature).
  double temperature = 0.;
  std::optional<uint64_t> seed;
  std::optional<int64_t> eos_token_id;
  /// Defaults to eos_token_id for already-finished batch rows.
  std::optional<int64_t> pad_token_id;
  std::string input_ids_name = "input_ids";
  std::string logits_name = "logits";
  /// These inputs are maintained only when declared by the graph.
  std::string attention_mask_name = "attention_mask";
  std::string position_ids_name = "position_ids";
};

/**
 * Generates tokens and returns the prompt followed by at most max_new_tokens tokens.
 *
 * Inputs are INT64 [batch, sequence] token IDs, optional INT64 attention masks
 * (left padding only) and position IDs, plus any other fixed model feeds.
 * Logits are FLOAT, DOUBLE, FLOAT16 or BFLOAT16 [batch, sequence, vocabulary]
 * or [batch, vocabulary]. The final position predicts the next token.
 *
 * Without graph persistent_bindings, each iteration evaluates the full prefix
 * using an ordinary RuntimeSession. With bindings, initial cache values come
 * from feeds (or paged-cache initializers), and PersistentValueState carries
 * them across decoding iterations: the prompt is evaluated once, then one token
 * per iteration. Kernels without append support keep their ordinary allocation
 * behavior. Initial caches must represent an empty prefix. State is local to
 * this Generate call; neither the model nor the feeds are modified. The supplied
 * execution context is reused and its invocation values may change. Model
 * functions remain scoped to each model session and override caller-registered
 * functions with the same identity. Reachable caller functions follow the same
 * private-name collision policy. Function and custom-kernel registrations are
 * not changed, including on failure.
 * This function is not thread-safe when calls share a RuntimeContext. The caller
 * must serialize generation and other execution using that context; concurrent
 * requests require independent contexts. Parallel kernels do not change this
 * requirement, and overlapping context use is not checked.
 *
 * Finished batch rows receive pad_token_id until every row has emitted EOS.
 * All context/model/allocator owners must outlive this synchronous call.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const ModelProto &model, RuntimeContext &context,
                                    const RuntimeValueMap &feeds,
                                    const GenerationOptions &options = {},
                                    RuntimeSessionOptions session_options = {});

/**
 * Generates tokens using the models and generation settings stored in a pipeline.
 *
 * Prefill is a prompt-processing phase, not a separate model: pipeline.prefill
 * also runs for subsequent tokens. The embedding model consumes input_ids before
 * each language-model invocation. Graphs connect through declared input/output
 * names, without additional mappings. Both models use the supplied context,
 * sharing its custom kernels, allocators and installed CPU executor.
 * Top-level graph inputs and outputs may share names, including an
 * initializer-backed embedding output connected to a language-model input.
 * Private initializer and intermediate names must be unique across the two
 * models; conflicts are rejected before either model executes.
 * Each feed is supplied once. It is routed to every model that declares the
 * corresponding graph input and model-local parameter references are resolved
 * against that model's catalogue.
 * A separate decode model and top-k/top-p sampling are not yet supported.
 * The same lifetime requirements apply as for a single model.
 * Calls sharing a context must be serialized, including calls to the single-model
 * overload or other runtime execution on that context.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const PipelineProto &pipeline, RuntimeContext &context,
                                    const RuntimeValueMap &feeds,
                                    RuntimeSessionOptions session_options = {});

/** Generates tokens using explicit options instead of the settings stored in the pipeline. */
ONNX_LIGHT_CORE_API Tensor Generate(const PipelineProto &pipeline, RuntimeContext &context,
                                    const RuntimeValueMap &feeds, const GenerationOptions &options,
                                    RuntimeSessionOptions session_options = {});

/**
 * Generates tokens from borrowed prefill and optional embedding models.
 *
 * This overload avoids copying model-owned weights when callers already retain
 * the individual models. Both models must outlive the synchronous call.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const ModelProto &prefill, const ModelProto *embedding,
                                    RuntimeContext &context, const RuntimeValueMap &feeds,
                                    const GenerationOptions &options = {},
                                    RuntimeSessionOptions session_options = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
