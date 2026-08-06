// SPDX-License-Identifier: Apache-2.0
//
// Efficient accumulation of symbolic dimension expressions.

#include "onnx_core/expressions/dim_sum.h"

#include <string>
#include <variant>

namespace onnx_light::core::expressions {

DimType simplify_dim_type(const DimType &value, SimplifiedExpressionCache *cache) {
  if (std::holds_alternative<int64_t>(value)) {
    return value;
  }
  const std::string &expr = std::get<std::string>(value);
  if (cache != nullptr) {
    auto it = cache->find(expr);
    if (it != cache->end()) {
      return it->second;
    }
  }
  const SimplifyResult simplified = simplify_expression(expr);
  if (std::holds_alternative<int64_t>(simplified)) {
    const DimType simplified_value = std::get<int64_t>(simplified);
    if (cache != nullptr) {
      cache->emplace(expr, simplified_value);
    }
    return simplified_value;
  }
  const DimType simplified_value = std::get<std::string>(simplified);
  if (cache != nullptr) {
    cache->emplace(expr, simplified_value);
  }
  return simplified_value;
}

void DimSum::Add(const DimType &term) {
  if (std::holds_alternative<int64_t>(term)) {
    constant_ += std::get<int64_t>(term);
  } else {
    coefficients_[std::get<std::string>(term)] += 1;
  }
}

DimType DimSum::Build(SimplifiedExpressionCache *cache) const {
  if (coefficients_.empty()) {
    return DimType{constant_};
  }
  std::string expr;
  if (constant_ != 0) {
    expr = std::to_string(constant_);
  }
  for (const auto &kv : coefficients_) {
    if (!expr.empty()) {
      expr += "+";
    }
    if (kv.second == 1) {
      expr += "(" + kv.first + ")";
    } else {
      expr += std::to_string(kv.second) + "*(" + kv.first + ")";
    }
  }
  return simplify_dim_type(DimType{expr}, cache);
}

} // namespace onnx_light::core::expressions
