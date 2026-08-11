// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::platform {

/**
 * Identifies an instruction-set capability usable by the current process.
 *
 * The numeric values are stable because persisted tuning profiles use the
 * corresponding names returned by CpuFeatureName.
 */
enum class CpuFeature : uint8_t {
  kSse2 = 0,
  kSse41,
  kSse42,
  kAvx,
  kFma,
  kAvx2,
  kAvx512F,
  kAvx512Bw,
  kAvx512Vl,
  kNeon,
  kDotProduct,
  kSve,
  kSve2,
};

/** Stores a compact set of CPU instruction-set capabilities. */
class CpuFeatureSet {
public:
  constexpr CpuFeatureSet() = default;
  explicit constexpr CpuFeatureSet(uint64_t bits) : bits_(bits) {}

  /** Adds a feature to the set. */
  constexpr void Add(CpuFeature feature) { bits_ |= FeatureBit(feature); }

  /**
   * Returns whether the set contains a feature.
   *
   * Returns:
   *   ``true`` when the feature is present.
   */
  constexpr bool Has(CpuFeature feature) const { return (bits_ & FeatureBit(feature)) != 0; }

  /**
   * Returns whether every feature in another set is present.
   *
   * Returns:
   *   ``true`` when all requested features are present.
   */
  constexpr bool ContainsAll(CpuFeatureSet other) const {
    return (bits_ & other.bits_) == other.bits_;
  }

  /**
   * Returns whether this set and another set share a feature.
   *
   * Returns:
   *   ``true`` when at least one feature is shared.
   */
  constexpr bool Intersects(CpuFeatureSet other) const { return (bits_ & other.bits_) != 0; }

  /** Returns the underlying stable bit mask. */
  constexpr uint64_t bits() const { return bits_; }

  constexpr bool operator==(const CpuFeatureSet &) const = default;

private:
  static_assert(static_cast<uint8_t>(CpuFeature::kSve2) < 64,
                "CpuFeatureSet bitmask supports up to 64 CpuFeature values.");
  static constexpr uint64_t FeatureBit(CpuFeature feature) {
    return uint64_t{1} << static_cast<uint8_t>(feature);
  }

  uint64_t bits_ = 0;
};

/**
 * Returns the stable serialized name of a CPU feature.
 *
 * Returns:
 *   The lowercase feature name.
 */
std::string_view CpuFeatureName(CpuFeature feature) noexcept;

/**
 * Parses a stable CPU feature name.
 *
 * Returns:
 *   The feature, or ``std::nullopt`` for an unknown name.
 */
std::optional<CpuFeature> CpuFeatureFromName(std::string_view name) noexcept;

/**
 * Describes immutable properties of the processor visible to the process.
 *
 * Optional values remain absent when the operating system or architecture
 * cannot report them reliably.
 */
struct CpuDescriptor {
  std::string architecture;
  std::string vendor;
  std::optional<uint32_t> family;
  std::optional<uint32_t> model;
  std::optional<uint32_t> stepping;
  std::string microarchitecture;
  CpuFeatureSet features;
  std::optional<size_t> cache_line_bytes;
  std::optional<size_t> l1_data_bytes;
  std::optional<size_t> l2_bytes;
  std::optional<size_t> l3_bytes;
  std::optional<uint32_t> physical_cores;
  std::optional<uint32_t> logical_cores;

  bool operator==(const CpuDescriptor &) const = default;
};

/**
 * Detects processor properties visible to the current process.
 *
 * Returns:
 *   A descriptor whose unavailable fields remain unknown.
 */
CpuDescriptor DetectCpuDescriptor();

/**
 * Returns the processor descriptor detected once for the current process.
 *
 * Returns:
 *   The immutable process-wide descriptor.
 */
const CpuDescriptor &GetCpuDescriptor();

/**
 * Selects processors using stable identifiers rather than display names.
 *
 * Empty optional fields and an empty model list are wildcards. Thread limits
 * require an explicit effective thread count when matching.
 */
struct CpuSelector {
  std::optional<std::string> architecture;
  std::optional<std::string> vendor;
  std::optional<uint32_t> family;
  std::vector<uint32_t> models;
  std::optional<std::string> microarchitecture;
  CpuFeatureSet required_features;
  CpuFeatureSet excluded_features;
  std::optional<uint32_t> minimum_threads;
  std::optional<uint32_t> maximum_threads;

  /**
   * Returns whether a processor and execution thread count satisfy the selector.
   *
   * Returns:
   *   ``true`` when every specified criterion matches.
   */
  bool Matches(const CpuDescriptor &processor,
               std::optional<uint32_t> effective_threads = std::nullopt) const;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::platform
