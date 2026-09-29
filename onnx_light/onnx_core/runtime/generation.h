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
 * this Generate call; neither the model, feeds nor the caller's context values
 * are modified.
 *
 * Finished batch rows receive pad_token_id until every row has emitted EOS.
 * All context/model/allocator owners must outlive this synchronous call.
 */
ONNX_LIGHT_CORE_API Tensor Generate(const ModelProto &model, RuntimeContext &context,
                                    const RuntimeValueMap &feeds,
                                    const GenerationOptions &options = {},
                                    RuntimeSessionOptions session_options = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
