// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/tuning/kernel_tuning.h"

#include <cstdint>
#include <span>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {

inline constexpr const char *kGemmParallelMinimumTasks = "parallel.minimum_tasks";
inline constexpr const char *kGemmAlgorithmConfiguration = "algorithm.configuration";
inline constexpr uint32_t kGemmTuningAbi = 2;
/** Counts stable configuration IDs: defaults, M32, N64, K64, always/never pack, M32/N64/K128. */
inline constexpr int64_t kGemmAlgorithmConfigurationCount = 7;

/** Stores the portable Gemm configuration copied into one kernel instance. */
struct GemmTuning {
  int64_t tile_m = 64;
  int64_t tile_n = 256;
  int64_t tile_k = 256;
  int64_t pack_b_minimum_elements = 16384;
  int64_t skinny_m_limit = 8;
  int64_t parallel_fmas_per_work_unit = 256;
  int64_t parallel_minimum_tasks = 2;
  int64_t conversion_parallel_minimum_elements = 1048576;
};

/** Registers one portable Gemm tuning schema for every supported element type. */
void RegisterGemmTuningSchemas(std::span<const int32_t> supported_element_types,
                               uint32_t tuning_abi = kGemmTuningAbi);

/** Validates resolved Gemm parameters and resets the typed configuration before applying them. */
void ConfigureGemmTuning(const core::runtime::KernelTuningParameters &parameters,
                         GemmTuning &tuning, uint32_t tuning_abi = kGemmTuningAbi);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
