// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Default per-call budget for option-driven numeric output payloads (64 MiB). */
inline constexpr size_t kTensorStatisticsMaxOutputBytes = 64 * 1024 * 1024;

/**
 * Validates options before allocating outputs or decoding input.
 *
 * Charges bins*sizeof(int64_t) + (bins+1)*sizeof(double) for the histogram (50 bins
 * when bins=0), rectangle_height*rectangle_width*sizeof(double), one double per
 * requested quantile, and one double for the median when requested. Charges all
 * requested outputs even when empty, small or nonfinite inputs would omit them.
 * Rejects a combined payload exceeding max_output_bytes; callers may explicitly
 * raise or lower the default 64 MiB budget. This is not a total-memory limit:
 * input storage, decoding, input-dependent workspaces, vector capacity/metadata
 * and binding copies are excluded.
 */
void ValidateTensorStatisticsOptions(int64_t bins, const std::vector<double> &quantiles,
                                     bool median, const std::vector<int64_t> &rectangle_shape,
                                     size_t max_output_bytes = kTensorStatisticsMaxOutputBytes);

/** Holds numeric tensor statistics and original storage metadata. */
struct TensorStatistics {
  int32_t dtype = TensorProto::UNDEFINED;
  std::vector<int64_t> shape;
  size_t size_bytes = 0;
  int64_t count = 0;
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
  double mean = std::numeric_limits<double>::quiet_NaN();
  std::optional<int64_t> distinct_count;
  std::vector<int64_t> distribution;
  std::vector<double> bin_edges;
  std::vector<double> quantiles;
  std::optional<double> median;
  std::vector<double> rectangle_map;
  std::vector<int64_t> rectangle_shape;
};

/** Identifies the current standalone tensor statistics operation. */
enum class TensorStatisticsPhase {
  kDecode,
  kSummary,
  kMean,
  kDistinct,
  kHistogram,
  kRectangle,
  kQuantiles,
  kComplete
};

/**
 * Receives (phase, completed, total) synchronously on the calling thread.
 *
 * Returning false cancels with std::runtime_error; callback exceptions propagate unchanged.
 * Reductions and decoding report phase boundaries (0/1 and 1/1). Quantiles report distribution
 * refinement passes for large FLOAT tensors, otherwise selected unique ranks, or 0/1 and 1/1
 * for constant FLOAT data. These are per-phase
 * work units, not an overall percentage or a wall-clock estimate. Completion reports 1/1.
 * Unneeded phases are omitted. Cancellation is cooperative between phases and selections:
 * an individual reduction, refinement pass, decode or nth_element is not interrupted.
 * The input is never mutated,
 * and cancellation returns no partial statistics. Callbacks must not mutate the input.
 */
using TensorStatisticsProgressCallback =
    std::function<bool(TensorStatisticsPhase phase, int64_t completed, int64_t total)>;

/**
 * Computes standalone numeric statistics; rejects strings, complex types and malformed storage.
 *
 * Computes distinct_count for integer and boolean tensors only, using exact integer keys.
 * A NaN propagates to min/max/mean and every requested quantile;
 * otherwise infinities follow numeric ordering and IEEE arithmetic. Quantiles interpolate at
 * q*(count-1), with exact endpoints preserved (including infinities).
 *
 * Produces a histogram only for count >= 10, using finite values exclusively. Zero bins selects
 * 50 bins for count > 500 and 10 otherwise. Constant finite data uses [value-0.5,value+0.5]
 * where representable; extreme constants can have repeated edges. Bins are left-closed and
 * right-open except the final bin, which includes the maximum.
 * Uses arithmetic bin estimates when they agree with the exported edges, otherwise searches
 * those edges to preserve exact boundary handling.
 *
 * Resizes an optional rectangle using fractional-overlap area averages. Rank >= 2 becomes
 * [first_dimension, product_of_remaining_dimensions]; scalars and vectors become [1,count].
 * Output is row-major with the requested positive [height,width], with no axis transposition.
 * Nonfinite values propagate through the overlapping cells using IEEE arithmetic.
 *
 * Validates all options even for empty tensors. Empty tensors return NaN min/max/mean,
 * distinct_count=0 for integer/boolean tensors, no histogram or rectangle, one NaN per requested
 * quantile, and NaN median
 * only when requested. Rectangle metadata is empty when no map is produced.
 * Applies ValidateTensorStatisticsOptions and its output budget before reading input.
 * Large FLOAT tensors (count >= 32768, at most 128 unique requested ranks) reuse the histogram
 * and refine selected bins exactly, with at most 8 MiB of private histogram counters and no
 * tensor-sized selection copy. Other calls use a private FLOAT/DOUBLE/native 64-bit integer
 * workspace (DOUBLE for other numeric types), selecting only requested ranks instead of sorting.
 * Rectangle-only calls read the tensor directly without a full-size numeric copy.
 */
TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {},
                                         size_t max_output_bytes = kTensorStatisticsMaxOutputBytes);

/** Computes tensor statistics with optional progress and cooperative cancellation. */
TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress);

/**
 * Computes tensor or encoded RuntimeValue statistics, rejecting structures and sequences.
 *
 * Encoded values use DequantizeTensor (including shared parameter catalogues); numeric statistics
 * describe reconstructed values, including distinct_count (signed zeros compare equal and NaNs
 * count as one distinct value). Their dtype/shape describe the original logical tensor type,
 * since encoded heterogeneous struct storage has no single tensor dtype/shape. size_bytes is the
 * original encoded raw payload size, excluding schemas and separately shared parameters.
 * Applies the same output budget before dequantizing input.
 */
TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {},
                                         size_t max_output_bytes = kTensorStatisticsMaxOutputBytes);

/** Computes RuntimeValue statistics with optional progress and cooperative cancellation. */
TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress);

/**
 * Computes encoded statistics, resolving named storage types through catalogue.
 * Shared parameter references must first be materialized or use the RuntimeValue overload.
 * Applies the same output budget before dequantizing input.
 */
TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {},
                                         const StructTypeCatalogue &catalogue = {},
                                         size_t max_output_bytes = kTensorStatisticsMaxOutputBytes);

/** Computes encoded statistics with optional progress and cooperative cancellation. */
TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins,
                                         const std::vector<double> &quantiles, bool median,
                                         const std::vector<int64_t> &rectangle_shape,
                                         const StructTypeCatalogue &catalogue,
                                         size_t max_output_bytes,
                                         const TensorStatisticsProgressCallback &progress);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
