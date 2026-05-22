// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/optim_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

OptimShape::OptimShape(std::initializer_list<OptimDim> dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.reserve(dims.size());
  for (const auto &d : dims) {
    dims_.push_back(d);
  }
}

OptimShape::OptimShape(const std::vector<OptimDim> &dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_ = dims;
}

void OptimShape::PushBack(OptimDim dim) {
  if (dims_.size() >= kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.push_back(std::move(dim));
}

bool OptimShape::IsFullyKnown() const noexcept {
  for (const auto &d : dims_) {
    if (!d.IsInt()) {
      return false;
    }
  }
  return true;
}

int64_t OptimShape::NumElements() const {
  int64_t total = 1;
  for (const auto &d : dims_) {
    if (!d.IsInt()) {
      throw std::runtime_error("OptimShape::NumElements requires a fully-known shape");
    }
    total *= d.AsInt();
  }
  return total;
}

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
