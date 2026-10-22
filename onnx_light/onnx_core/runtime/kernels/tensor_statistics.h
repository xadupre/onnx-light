// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <limits>
#include <optional>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

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
 */
TensorStatistics ComputeTensorStatistics(const Tensor &tensor, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {});

/**
 * Computes tensor or encoded RuntimeValue statistics, rejecting structures and sequences.
 *
 * Encoded values use DequantizeTensor (including shared parameter catalogues); numeric statistics
 * describe reconstructed values, including distinct_count (signed zeros compare equal and NaNs
 * count as one distinct value). Their dtype/shape describe the original logical tensor type,
 * since encoded heterogeneous struct storage has no single tensor dtype/shape. size_bytes is the
 * original encoded raw payload size, excluding schemas and separately shared parameters.
 */
TensorStatistics ComputeTensorStatistics(const RuntimeValue &value, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {});

/**
 * Computes encoded statistics, resolving named storage types through catalogue.
 * Shared parameter references must first be materialized or use the RuntimeValue overload.
 */
TensorStatistics ComputeTensorStatistics(const EncodedValueProto &value, int64_t bins = 0,
                                         const std::vector<double> &quantiles = {},
                                         bool median = false,
                                         const std::vector<int64_t> &rectangle_shape = {},
                                         const StructTypeCatalogue &catalogue = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
