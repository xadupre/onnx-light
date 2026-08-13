// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_tuning.h"

#include <cstdint>
#include <span>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {

inline constexpr const char *kGemmTileM = "algorithm.tile_m";
inline constexpr const char *kGemmTileN = "algorithm.tile_n";
inline constexpr const char *kGemmTileK = "algorithm.tile_k";
inline constexpr const char *kGemmPackBMinimumElements = "algorithm.pack_b_minimum_elements";
inline constexpr const char *kGemmSkinnyMLimit = "algorithm.skinny_m_limit";
inline constexpr const char *kGemmParallelFmasPerWorkUnit = "parallel.fmas_per_work_unit";
inline constexpr const char *kGemmParallelMinimumTasks = "parallel.minimum_tasks";
inline constexpr const char *kGemmConversionParallelMinimumElements =
    "conversion.parallel_minimum_elements";

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
                               uint32_t tuning_abi = 1);

/** Validates and copies resolved Gemm parameters into a typed configuration. */
void ConfigureGemmTuning(const core::runtime::KernelTuningParameters &parameters,
                         GemmTuning &tuning, uint32_t tuning_abi = 1);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
