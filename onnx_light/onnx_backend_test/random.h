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
 * The result element type ``T`` is selected by the template parameter
 * (defaults to ``double``). Samples are computed in ``double`` precision and
 * then ``static_cast`` to ``T``, mirroring the convention used by
 * :cpp:func:`Randn`. Explicit instantiations are provided for ``double`` and
 * ``float``.
 *
 * @tparam T Floating-point output element type (``double`` or ``float``).
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed
 *             (0).
 * @return Flat row-major ``std::vector<T>`` of length ``prod(shape)``.
 * @throws std::invalid_argument when any dimension is negative.
 */
template <typename T = double>
std::vector<T> Rand(const std::vector<int64_t> &shape, std::optional<uint64_t> seed = std::nullopt);

/// @cond DOXYGEN_SKIP_EXTERN_TEMPLATE
extern template std::vector<double> Rand<double>(const std::vector<int64_t> &shape,
                                                 std::optional<uint64_t> seed);
extern template std::vector<float> Rand<float>(const std::vector<int64_t> &shape,
                                               std::optional<uint64_t> seed);
/// @endcond

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
 * The result element type ``T`` is selected by the template parameter
 * (defaults to ``double``). Samples are computed in ``double`` precision and
 * then ``static_cast`` to ``T``, which avoids the repeated cast-to-float
 * loops previously written at each call site. Explicit instantiations are
 * provided for ``double`` and ``float``.
 *
 * @tparam T Floating-point output element type (``double`` or ``float``).
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed Optional 64-bit seed. ``std::nullopt`` selects the default seed.
 * @return Flat row-major ``std::vector<T>`` of length ``prod(shape)``.
 * @throws std::invalid_argument when any dimension is negative.
 */
template <typename T = double>
std::vector<T> Randn(const std::vector<int64_t> &shape,
                     std::optional<uint64_t> seed = std::nullopt);

/// @cond DOXYGEN_SKIP_EXTERN_TEMPLATE
extern template std::vector<double> Randn<double>(const std::vector<int64_t> &shape,
                                                  std::optional<uint64_t> seed);
extern template std::vector<float> Randn<float>(const std::vector<int64_t> &shape,
                                                std::optional<uint64_t> seed);
/// @endcond

/**
 * Builds a signed-integer vector whose elements are drawn from the same
 * Irwin-Hall-approximated :cpp:func:`Randn` distribution as the upstream
 * ``np.random.randn(...).astype(np.intN)`` pattern. Values are truncated via
 * ``static_cast<TInt>`` to the destination dtype (matching NumPy's
 * float-to-int cast semantics for in-range values).
 *
 * @tparam TInt Signed integer output element type.
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed 64-bit seed forwarded to :cpp:func:`Randn`.
 * @return Flat row-major ``std::vector<TInt>`` of length ``prod(shape)``.
 */
template <typename TInt>
std::vector<TInt> RandnInt(const std::vector<int64_t> &shape, uint64_t seed);

/// @cond DOXYGEN_SKIP_EXTERN_TEMPLATE
extern template std::vector<int8_t> RandnInt<int8_t>(const std::vector<int64_t> &shape,
                                                     uint64_t seed);
extern template std::vector<int16_t> RandnInt<int16_t>(const std::vector<int64_t> &shape,
                                                       uint64_t seed);
extern template std::vector<int32_t> RandnInt<int32_t>(const std::vector<int64_t> &shape,
                                                       uint64_t seed);
extern template std::vector<int64_t> RandnInt<int64_t>(const std::vector<int64_t> &shape,
                                                       uint64_t seed);
/// @endcond

/**
 * Builds an unsigned-integer vector whose elements are drawn uniformly from
 * ``[0, high)`` via :cpp:func:`RandInt`, mirroring the upstream
 * ``np.random.randint(high, ...)`` pattern used by the upstream
 * ``Greater``/``Less`` uint variants.
 *
 * @tparam TUInt Unsigned integer output element type.
 * @param high Exclusive upper bound forwarded to :cpp:func:`RandInt`.
 * @param shape Output shape; dimensions must be non-negative.
 * @param seed 64-bit seed forwarded to :cpp:func:`RandInt`.
 * @return Flat row-major ``std::vector<TUInt>`` of length ``prod(shape)``.
 */
template <typename TUInt>
std::vector<TUInt> RandUint(int64_t high, const std::vector<int64_t> &shape, uint64_t seed);

/// @cond DOXYGEN_SKIP_EXTERN_TEMPLATE
extern template std::vector<uint8_t>
RandUint<uint8_t>(int64_t high, const std::vector<int64_t> &shape, uint64_t seed);
extern template std::vector<uint16_t>
RandUint<uint16_t>(int64_t high, const std::vector<int64_t> &shape, uint64_t seed);
extern template std::vector<uint32_t>
RandUint<uint32_t>(int64_t high, const std::vector<int64_t> &shape, uint64_t seed);
extern template std::vector<uint64_t>
RandUint<uint64_t>(int64_t high, const std::vector<int64_t> &shape, uint64_t seed);
/// @endcond

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
