// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_session.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Identifies a synchronous loading phase. */
enum class RuntimeLoadPhase {
  kDeclared,
  kReadStarted,
  kReadFinished,
  kPreparationStarted,
  kPreparationFinished,
  kComplete
};

/** Reports a loading phase using a steady-clock timestamp. */
struct RuntimeLoadEvent {
  RuntimeLoadPhase phase;
  std::string tensor_name;
  uint64_t timestamp_ns;
  /// Positive live static allocation growth on a finished preparation phase.
  /// Zero does not establish that packing occurred (including zero-sized work).
  size_t prepared_bytes_added = 0;
};

/** Configures synchronous owning model or pipeline loading. */
struct RuntimeLoadOptions {
  bool overlap_preparation = true;
  /// Bounds source payload, conversion reserve, bound function copies, shared quantization
  /// catalogues (including construction workspaces), and preparation work.
  /// Excludes proto metadata, parser allocations, executor pools and inference.
  size_t memory_limit = std::numeric_limits<size_t>::max();
  /// Receives events on the caller or I/O thread; must be thread safe.
  /// Callbacks must not mutate the model/context or wait for future loader work.
  std::function<void(const RuntimeLoadEvent &)> trace;
  /// Optional trusted reader, invoked instead of TensorProto::LoadExternalData.
  /// May change only this tensor's raw_data and must honor its declared length.
  /// It must return or throw: destruction joins, rather than interrupts, I/O.
  std::function<void(TensorProto &, const std::string &)> external_reader;
};

/** Reports conservative admission counters, not process resident memory. */
struct RuntimeLoadStatistics {
  size_t source_bytes = 0;
  size_t conversion_reserve_bytes = 0;
  size_t function_binding_bytes = 0;
  size_t quantization_catalogue_bytes = 0;
  size_t quantization_temporary_bytes = 0;
  size_t peak_admitted_bytes = 0;
  size_t external_tensors = 0;
  size_t peak_queued_tensors = 0;
};

/**
 * Owns a source model and its prepared session.
 *
 * Loading completes synchronously. A single bounded I/O worker may read the
 * next payload while the caller seeds the previous one and executes declared
 * preparations on the session's CPU executor. No worker mutates RuntimeContext
 * or ValueStore. The model is immutable after loading and outlives the session.
 */
class LoadedRuntimeSession {
public:
  LoadedRuntimeSession(const std::string &path, RuntimeContext &rt,
                       RuntimeSessionOptions session_options = {},
                       RuntimeLoadOptions load_options = {});
  ~LoadedRuntimeSession();
  LoadedRuntimeSession(const LoadedRuntimeSession &) = delete;
  LoadedRuntimeSession &operator=(const LoadedRuntimeSession &) = delete;
  LoadedRuntimeSession(LoadedRuntimeSession &&) = delete;
  LoadedRuntimeSession &operator=(LoadedRuntimeSession &&) = delete;

  RuntimeSession &session() noexcept { return *session_; }
  const RuntimeSession &session() const noexcept { return *session_; }
  const ModelProto &model() const noexcept { return model_; }
  const RuntimeLoadStatistics &statistics() const noexcept { return statistics_; }

private:
  ModelProto model_;
  std::unique_ptr<RuntimeSession> session_;
  RuntimeLoadStatistics statistics_;
};

/**
 * Owns a pipeline, its prepared stage sessions and their shared static store.
 *
 * Supports the prefill and optional embedding stages accepted by Generate.
 * The loading budget and statistics cover all stages together. Stages load
 * sequentially, with at most one bounded I/O worker overlapping preparation.
 * Generate overloads retain these session settings and reuse the static store,
 * while creating fresh invocation sessions for request-local persistent state.
 * Serialize generation, stage execution and other access to the shared store.
 * Borrowed initializer views must not outlive this owner.
 */
class LoadedRuntimePipeline {
public:
  LoadedRuntimePipeline(const std::string &path, RuntimeContext &rt,
                        RuntimeSessionOptions session_options = {},
                        RuntimeLoadOptions load_options = {});
  ~LoadedRuntimePipeline();
  LoadedRuntimePipeline(const LoadedRuntimePipeline &) = delete;
  LoadedRuntimePipeline &operator=(const LoadedRuntimePipeline &) = delete;
  LoadedRuntimePipeline(LoadedRuntimePipeline &&) = delete;
  LoadedRuntimePipeline &operator=(LoadedRuntimePipeline &&) = delete;

  const PipelineProto &pipeline() const noexcept { return pipeline_; }
  RuntimeSession &prefill_session() noexcept { return *prefill_session_; }
  const RuntimeSession &prefill_session() const noexcept { return *prefill_session_; }
  RuntimeSession *embedding_session() noexcept { return embedding_session_.get(); }
  const RuntimeSession *embedding_session() const noexcept { return embedding_session_.get(); }
  const RuntimeSessionOptions &session_options() const noexcept { return session_options_; }
  const RuntimeLoadStatistics &statistics() const noexcept { return statistics_; }

private:
  PipelineProto pipeline_;
  RuntimeSessionOptions session_options_;
  std::unique_ptr<RuntimeSession> embedding_session_;
  std::unique_ptr<RuntimeSession> prefill_session_;
  RuntimeLoadStatistics statistics_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
