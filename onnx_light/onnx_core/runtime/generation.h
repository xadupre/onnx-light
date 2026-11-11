// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_session.h"

#include <chrono>
#include <functional>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Prepares an explicitly generation-only model in place, without copying its weights.
 *
 * Changes the named graph output contract from [B,S,V] to [B,1,V] by gathering
 * the last hidden position before a standard floating-point MatMul LM head.
 * Supports a trailing standard Cast/Identity chain only. Rejects unsupported or
 * shared heads before mutation, including persistent aliases and nested captures.
 * Requires rank-three output metadata and rank-two weight metadata/initializer;
 * hidden rank may be inferred from these. Invalidates saved rewritings and
 * execution/memory annotations. Initializer payload ownership is preserved.
 * A second preparation is rejected. Ordinary models and Generate calls are
 * never prepared implicitly. Callers must prepare before constructing sessions.
 *
 * The inserted Gather rejects an empty sequence. Generate already rejects empty
 * prompts and all-padding rows; left-padded rows must end in a real token.
 */
ONNX_LIGHT_CORE_API void PrepareGenerationModel(ModelProto &model,
                                                const std::string &logits_name = "logits");

/** Identifies the logical generation phase, including full-prefix decoding without caches. */
enum class GenerationPhase { kPrefill, kDecode };

/** Describes one completed generation iteration, with one emitted token per batch row. */
struct GenerationStep {
  int64_t index = 0;
  GenerationPhase phase = GenerationPhase::kPrefill;
  std::chrono::steady_clock::time_point started_at{};
  std::chrono::steady_clock::time_point completed_at{};
  /// Includes padding IDs for batch rows that finished on a previous iteration.
  std::vector<int64_t> token_ids;
  /// Owns unprocessed [batch, vocabulary] last-position logits in the original output dtype.
  /// Empty when capture is disabled or the complete snapshot exceeds the remaining byte budget.
  std::optional<Tensor> raw_logits;
};

/**
 * Receives each completed step synchronously on the calling thread.
 *
 * Returning false stops before the next iteration and keeps the token just emitted.
 * Callback exceptions propagate unchanged. The callback must not reenter generation
 * or mutate its model, feeds, options or RuntimeContext. Its step reference is valid
 * only during the callback; a copied step owns its token IDs and any captured logits
 * independently of the model, context and later iterations.
 *
 * started_at precedes per-iteration input preparation, including embedding execution;
 * completed_at follows model evaluation, sampling and appending the selected tokens.
 * Session construction and initial request validation, raw-logit snapshot copying,
 * callback execution and final output-tensor construction are excluded.
 * The first step is prefill; all later steps are decode, even without persistent caches.
 * Both timestamps use steady_clock; their difference is that batch iteration's latency.
 * No callback is made for zero requested tokens or an iteration that fails.
 */
using GenerationStepCallback = std::function<bool(const GenerationStep &step)>;

/** Configures autoregressive token generation. */
struct GenerationOptions {
  int64_t max_new_tokens = 20;
  /// Zero selects greedy decoding; positive values sample softmax(logits / temperature).
  double temperature = 0.;
  std::optional<uint64_t> seed;
  /// Stops each row on any listed token; an empty list disables EOS stopping.
  std::vector<int64_t> eos_token_ids;
  /// Defaults to the first eos_token_ids entry for already-finished rows.
  std::optional<int64_t> pad_token_id;
  std::string input_ids_name = "input_ids";
  std::string logits_name = "logits";
  /// These inputs are maintained only when declared by the graph.
  std::string attention_mask_name = "attention_mask";
  std::string position_ids_name = "position_ids";
  /// Zero infers the layout from the declared position input. Positive values select
  /// [axes, batch, sequence] positions (also for a graph with unspecified dimensions).
  int64_t position_axes = 0;
  /// Defaults to no observation, avoiding clock reads and observation allocations.
  GenerationStepCallback on_step;
  /// Bounds the total raw-logit payload delivered to on_step during this Generate call.
  /// Zero disables capture. Only complete [batch, vocabulary] snapshots are copied;
  /// token/timing notifications continue when the remaining budget cannot hold a snapshot.
  /// Requires on_step. Copies subsequently made by the callback are outside this budget.
  size_t max_logit_bytes = 0;
  /// Embedding-only tensor feeds replaced after prefill by empty tensors with the
  /// same dtype and trailing dimensions (leading dimension becomes zero).
  /// Requires decoder persistent bindings and explicit, nonscalar tensor feeds.
  /// Token, mask and position inputs cannot be listed. Other feeds remain fixed.
  std::vector<std::string> prefill_only_inputs;
  /// Must be nonnegative. Zero disables top-k; otherwise retains min(top_k, vocabulary) candidates.
  int64_t top_k = 0;
  /// Unset or one disables nucleus filtering; otherwise must be finite in (0, 1].
  std::optional<double> top_p;
};

/**
 * Generates tokens and returns the prompt followed by at most max_new_tokens tokens.
 *
 * Inputs are INT64 [batch, sequence] token IDs, optional INT64 attention masks
 * (left padding only) and position IDs, plus any other fixed model feeds.
 * Position IDs may be [batch, sequence] or [axes, batch, sequence]. For
 * multi-axis positions, generated text uses one coordinate per axis, equal
 * to the maximum unmasked prefill coordinate plus the decode step.
 * Logits are FLOAT, DOUBLE, FLOAT16 or BFLOAT16 [batch, sequence, vocabulary]
 * or [batch, vocabulary]. The final position predicts the next token.
 * Positive temperature scales logits before sampling. Top-k then retains the
 * highest-scoring candidates, and top-p keeps the smallest descending-probability
 * prefix reaching p of the remaining, renormalized probability mass (at least one
 * token). Score ties favor lower token IDs, including at either cutoff. The final
 * retained distribution is renormalized for sampling. Zero temperature ignores
 * filtering and preserves greedy decoding; all option ranges are still validated.
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
 * Optional step observation and cancellation use GenerationOptions::on_step.
 * All context/model/allocator owners must outlive this synchronous call.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const ModelProto &model, RuntimeContext &context,
                                    const RuntimeValueMap &feeds,
                                    const GenerationOptions &options = {},
                                    RuntimeSessionOptions session_options = {});

/**
 * Generates tokens using the models and generation settings stored in a pipeline.
 *
 * Runs pipeline.prefill on the prompt. When pipeline.decode is present, it runs
 * all subsequent iterations with one token per row; otherwise prefill also runs
 * subsequent iterations, preserving the single-model behavior. The embedding
 * model consumes input_ids before each language-model invocation. Graphs connect
 * through declared input/output names, without additional mappings. Models use the supplied
 * context, sharing its custom kernels, allocators and installed CPU executor. Top-level graph
 * inputs and outputs may share names, including an initializer-backed embedding output connected to
 * a language-model input. Private initializer and intermediate names must be unique between
 * embedding and each language model; conflicts are rejected before execution. Prefill and decode
 * may share all names, including private names and weights. Each feed is supplied once. It is
 * routed to every model that declares the corresponding graph input and model-local parameter
 * references are resolved against that model's catalogue. A separate decode model requires
 * persistent bindings with identical input/output name pairs in both models; both sets may be
 * empty. Without bindings, explicit decode still runs on one token, rather than recomputing the
 * prompt. Initial state comes only from prefill feeds or paged-cache initializers. Retained prefill
 * values and append reservations are transferred to the decode session without copying cache
 * payloads or losing reserved capacity; ordinary append/growth allocation rules still apply.
 * Fixed feeds are routed independently, including
 * decode-only inputs. Embedding outputs must connect compatibly to both language models.
 *
 * Both language-model contracts are validated before execution, even when zero
 * tokens are requested. Tensor dtypes and declared ranks must agree. Missing
 * shapes and symbolic or unspecified dimensions are wildcards; symbol spelling
 * does not establish cross-model equality. Conflicting concrete dimensions are
 * rejected except for sequence specialization: token/position inputs accept the
 * prompt length in prefill and length one in decode; masks cover the full prefix.
 * Embedding-connected language inputs of rank three or greater use
 * [batch, sequence, ...] and the same prompt/one sequence specialization.
 * The embedding output declarations must be compatible with both stages.
 * Logits may specialize their sequence dimension to prompt length or one in
 * prefill and one in decode; batch and vocabulary dimensions must agree.
 * Concrete control dimensions must fit the supplied request. Persistent types,
 * including tensor leaves in sequences and structures, do not receive sequence
 * specialization. Wildcards are checked against actual values by execution.
 *
 * All invocation sessions reuse session_options.value_store when supplied.
 * Model-owned initializer payloads are borrowed. Distinct model preparations
 * may still own separate transformed weights; equal names alone do not deduplicate
 * preparations or payloads. Top-k/top-p sampling is supported through
 * GenerationOptions or the settings stored in the pipeline.
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

class LoadedRuntimePipeline;

/**
 * Generates tokens from an owning pipeline using its stored generation settings.
 *
 * Reuses the loader's session settings and static preparation store. Invocation
 * sessions and persistent state are fresh for each call. The owner must outlive
 * the call and any initializer-backed views. Callers must not run invocations
 * sharing a context or the static store concurrently.
 * Temporarily installs the selected CPU executor, inheriting a caller's executor
 * unless the loader's session options explicitly select another CPU policy.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const LoadedRuntimePipeline &pipeline, RuntimeContext &context,
                                    const RuntimeValueMap &feeds);

/** Generates tokens from an owning pipeline using explicit generation options. */
ONNX_LIGHT_CORE_API Tensor Generate(const LoadedRuntimePipeline &pipeline, RuntimeContext &context,
                                    const RuntimeValueMap &feeds, const GenerationOptions &options);

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
