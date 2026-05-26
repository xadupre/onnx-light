// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "onnx_backend_test/simple_tensor.h"
#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/// Default seed used when no explicit seed is provided.
inline constexpr uint64_t kDefaultSeed = 0;

/**
 * Computes the next SplitMix64 state and the corresponding random 64-bit
 * value. The implementation matches the canonical SplitMix64 generator and
 * is bit-identical to the Python reference in ``onnx_light.backend.random``.
 *
 * @param state Current 64-bit state.
 * @return ``std::pair{next_state, random_value}``.
 */
std::pair<uint64_t, uint64_t> NextUint64(uint64_t state);

/**
 * Generates deterministic uniform random values in ``[0, 1)`` shaped as
 * ``shape``. An empty ``shape`` produces a single value (count 1).
 *
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed
 *             (0).
 * @return Flat row-major ``std::vector<double>`` of length ``prod(shape)``.
 * @throws std::invalid_argument when any dimension is negative.
 */
std::vector<double> Rand(const std::vector<int64_t> &shape,
                         std::optional<uint64_t> seed = std::nullopt);

/**
 * Generates deterministic pseudo-random integers in the half-open interval
 * ``[low, high)`` using rejection sampling on the SplitMix64 output to obtain
 * an unbiased distribution.
 *
 * @param low Inclusive lower bound.
 * @param high Exclusive upper bound. Must be strictly greater than ``low``.
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed.
 * @return Flat row-major ``std::vector<int64_t>`` of length ``prod(shape)``.
 * @throws std::invalid_argument when ``high <= low`` or any dimension is
 *         negative.
 */
std::vector<int64_t> RandInt(int64_t low, int64_t high, const std::vector<int64_t> &shape,
                             std::optional<uint64_t> seed = std::nullopt);

/**
 * Generates deterministic pseudo-random values with an approximate normal
 * distribution using the Irwin-Hall approximation (sum of 12 uniform values
 * minus 6).
 *
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed.
 * @return Flat row-major ``std::vector<double>`` of length ``prod(shape)``.
 * @throws std::invalid_argument when any dimension is negative.
 */
std::vector<double> Randn(const std::vector<int64_t> &shape,
                          std::optional<uint64_t> seed = std::nullopt);

/**
 * Generates a deterministic ``BOOL`` ``Tensor`` of the requested shape by
 * drawing approximately-normal values from :cpp:func:`Randn` and thresholding
 * at 0. Mirrors the upstream ONNX test pattern
 * ``(np.random.randn(...) > 0).astype(bool)`` used by
 * ``onnx.backend.test.case.node`` cases (e.g. ``And``/``Or``/``Xor``).
 *
 * @param shape Output tensor shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed.
 * @return A ``Tensor`` with ``data_type == BOOL`` whose ``data`` is a flat
 *         row-major buffer of one byte per element (``0`` or ``1``).
 * @throws std::invalid_argument when any dimension is negative.
 */
Tensor RandBool(const std::vector<int64_t> &shape, std::optional<uint64_t> seed = std::nullopt);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
