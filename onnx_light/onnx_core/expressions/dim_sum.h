// SPDX-License-Identifier: Apache-2.0
//
// Efficient accumulation of symbolic dimension expressions.

/**
 * @file dim_sum.h
 * @brief Helpers to sum many symbolic dimension expressions cheaply.
 *
 * ::onnx_light::core::expressions::simplify_expression is super-linear in the
 * number of additive terms it receives, so repeatedly simplifying an
 * ever-growing sum (as happens when profiling large graphs) dominates the
 * cost. :cpp:class:`DimSum` collapses byte-identical terms into
 * ``coeff*(term)`` before simplifying, feeding the simplifier only as many
 * terms as there are *distinct* expressions while producing a byte-identical
 * result.
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>

#include "onnx_core/expressions/expressions.h"

namespace onnx_light::core::expressions {

/// Memoization cache mapping a symbolic expression string to its simplified
/// form, shared across simplifications to avoid re-running the symbolic
/// simplifier on the same expression.
using SimplifiedExpressionCache = std::unordered_map<std::string, DimType>;

/// Returns ``value`` unchanged when it is already a concrete integer,
/// otherwise the canonical simplified form of the symbolic expression it
/// holds (memoized in ``cache`` when provided).
DimType simplify_dim_type(const DimType &value, SimplifiedExpressionCache *cache = nullptr);

/**
 * @brief Accumulates symbolic dimension expressions, grouping identical terms.
 *
 * Successive :cpp:func:`Add` calls bucket concrete integers into a running
 * constant and identical symbolic expressions into integer coefficients.
 * :cpp:func:`Build` emits the packed `constant + c1*(expr1) + c2*(expr2) ...`
 * expression and simplifies it once. The result is byte-identical to summing
 * and simplifying each term individually, but the simplifier only ever sees
 * the distinct terms.
 */
class DimSum {
public:
  /// Adds a single term to the running sum.
  void Add(const DimType &term);

  /// Returns the simplified sum of every added term (`0` when empty).
  DimType Build(SimplifiedExpressionCache *cache = nullptr) const;

private:
  int64_t constant_ = 0;
  // Maps each distinct symbolic expression string to its integer coefficient
  // (the number of times it appears in the sum).
  std::map<std::string, int64_t> coefficients_;
};

} // namespace onnx_light::core::expressions
