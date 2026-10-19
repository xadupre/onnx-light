// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_threads/config.h"

namespace onnx_threads {

/**
 * Executes two non-null, non-throwing operations and waits for both to finish.
 *
 * When parallel is true, admits one asynchronous worker within the shared
 * hardware-concurrency budget. A saturated budget or a failed worker launch
 * executes both operations inline. Contexts remain borrowed until return.
 */
ONNX_THREADS_API void ForkJoin(void *left_context, void (*left)(void *) noexcept,
                               void *right_context, void (*right)(void *) noexcept,
                               bool parallel) noexcept;

} // namespace onnx_threads
