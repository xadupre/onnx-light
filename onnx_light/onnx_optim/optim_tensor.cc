// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file optim_tensor.cc
 * @brief Out-of-line implementation of :cpp:class:`OptimShape`.
 *
 * Most of the ``onnx_optim`` public API is small enough to be defined
 * inline in ``optim_tensor.h``. Only the few :cpp:class:`OptimShape`
 * members that perform bounds checking, iteration over the stored
 * dimensions, or arithmetic on integer dimensions live here so that
 * the header stays free of ``<stdexcept>``.
 *
 * @see optim_tensor.h
 */

#include "onnx_optim/optim_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

/**
 * Constructs an :cpp:class:`OptimShape` from a brace-enclosed list of
 * dimensions.
 *
 * @param dims Dimensions to copy into the new shape, in order.
 * @throws std::length_error if ``dims.size() > kMaxOptimRank``.
 */
OptimShape::OptimShape(std::initializer_list<OptimDim> dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.reserve(dims.size());
  for (const auto &d : dims) {
    dims_.push_back(d);
  }
}

/**
 * Constructs an :cpp:class:`OptimShape` by copying an existing
 * ``std::vector`` of :cpp:class:`OptimDim`.
 *
 * @param dims Source dimensions to copy.
 * @throws std::length_error if ``dims.size() > kMaxOptimRank``.
 */
OptimShape::OptimShape(const std::vector<OptimDim> &dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_ = dims;
}

/**
 * Appends a dimension to the shape.
 *
 * @param dim Dimension to append; may be integer or symbolic.
 * @throws std::length_error if the shape already contains
 *         ``kMaxOptimRank`` dimensions.
 */
void OptimShape::PushBack(OptimDim dim) {
  if (dims_.size() >= kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.push_back(std::move(dim));
}

/**
 * Returns ``true`` when every dimension in the shape is a concrete
 * integer. A rank-0 (empty) shape is considered fully known.
 */
bool OptimShape::IsFullyKnown() const noexcept {
  for (const auto &d : dims_) {
    if (!d.IsInt()) {
      return false;
    }
  }
  return true;
}

/**
 * Computes the product of every integer dimension.
 *
 * Returns ``1`` for a rank-0 (empty) shape, matching the standard
 * scalar element-count semantic.
 *
 * @throws std::runtime_error if any dimension is symbolic; check
 *         :cpp:func:`IsFullyKnown` first if the shape may be
 *         partially symbolic.
 */
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
