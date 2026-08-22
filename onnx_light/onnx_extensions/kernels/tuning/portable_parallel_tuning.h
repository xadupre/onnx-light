// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {

inline constexpr std::string_view kParallelMinimumElements = "parallel.minimum_elements";

/** Stores the portable parallel configuration copied into one kernel instance. */
struct ParallelTuning {
  explicit ParallelTuning(int64_t minimum_elements) : parallel_minimum_elements(minimum_elements) {}

  int64_t parallel_minimum_elements;
};

/** Implements the shared runtime contract for one parallel-tunable kernel. */
class ParallelTunableKernel : public core::runtime::KernelBase {
public:
  ParallelTunableKernel(const core::runtime::KernelContext &ctx, std::string_view kernel,
                        std::span<const int32_t> supported_element_types,
                        int64_t portable_minimum_elements, uint32_t tuning_abi = 1);

  core::runtime::KernelTuningKey TuningKey(int32_t element_type) const override;
  void Configure(const core::runtime::KernelTuningParameters &parameters) override;

  /** Returns the immutable configuration used by the execution path. */
  const ParallelTuning &tuning() const noexcept { return tuning_; }

private:
  std::string_view kernel_;
  std::span<const int32_t> supported_element_types_;
  ParallelTuning tuning_;
  uint32_t tuning_abi_;
};

/**
 * Returns the tuning key for one portable onnx-light kernel implementation.
 *
 * Returns:
 *   The exact key for ``kernel`` and ``element_type``.
 */
core::runtime::KernelTuningKey MakePortableTuningKey(std::string_view kernel, int32_t element_type,
                                                     uint32_t tuning_abi = 1);

/**
 * Returns whether an element type is supported by a kernel.
 *
 * Returns:
 *   ``true`` when ``element_type`` appears in ``supported_element_types``.
 */
bool IsSupportedElementType(int32_t element_type, std::span<const int32_t> supported_element_types);

/** Registers one parallel tuning schema for every supported element type. */
void RegisterParallelTuningSchemas(std::string_view kernel,
                                   std::span<const int32_t> supported_element_types,
                                   int64_t portable_minimum_elements, uint32_t tuning_abi = 1);

/** Validates and copies resolved parallel parameters into a typed configuration. */
void ConfigureParallelTuning(std::string_view kernel,
                             const core::runtime::KernelTuningParameters &parameters,
                             ParallelTuning &tuning, uint32_t tuning_abi = 1);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning

/**
 * Defines ``ClassName::RegisterTuningSchemas()`` for one ``ParallelTunableKernel`` subclass.
 *
 * Expects ``kSupportedElementTypes``, ``kPortableParallelMinimum``, and ``kTuningAbi`` to be
 * in scope at the point of invocation (typically declared in the kernel's anonymous
 * namespace, right above the macro invocation), and registers ``ClassName`` (stringified) as
 * the schema name so it always matches the kernel's class name.
 */
#define ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(ClassName)                                      \
  void ClassName::RegisterTuningSchemas() {                                                        \
    tuning::RegisterParallelTuningSchemas(#ClassName, kSupportedElementTypes,                      \
                                          kPortableParallelMinimum, kTuningAbi);                   \
  }
