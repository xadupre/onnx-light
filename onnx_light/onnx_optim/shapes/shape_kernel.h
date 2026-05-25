// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_kernel.h
 * @brief Base interface for ``onnx_optim`` shape kernels.
 *
 * A *shape kernel* performs the shape (and dtype) inference for a single
 * ``NodeProto``. It consumes one or more :cpp:class:`OptimTensor` views
 * describing the operator inputs and produces one or more
 * :cpp:class:`OptimTensor` views describing the operator outputs.
 *
 * Concrete kernels live under ``onnx_optim/shapes/<domain>/<op>.h``
 * (for example ``onnx_optim/shapes/math/abs.h``) and are instantiated
 * from a :cpp:class:`NodeProto` via :cpp:func:`MakeShapeKernel`.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Abstract base class for all shape kernels.
 *
 * A shape kernel encapsulates the shape (and dtype) inference logic for
 * a single ONNX operator. The kernel is constructed once from a
 * :cpp:class:`NodeProto` (so any required attributes can be parsed
 * up-front) and may then be invoked any number of times against
 * different concrete input tensors.
 */
class ShapeKernel {
public:
  virtual ~ShapeKernel() = default;

  /// The ``op_type`` of the node this kernel was built from.
  const std::string &OpType() const noexcept { return op_type_; }

  /// The ``domain`` of the node this kernel was built from.
  const std::string &Domain() const noexcept { return domain_; }

  /**
   * Runs shape inference for a unary operator.
   *
   * Returns an :cpp:class:`OptimTensor` describing the single output of
   * the node given a single input. Kernels that take more than one
   * input or produce more than one output must provide their own
   * dedicated entry points; the default implementation throws
   * ``std::logic_error``.
   */
  virtual OptimTensor Run(const OptimTensor &input) const;

protected:
  /// Constructs the kernel from a ``NodeProto``, caching its identity.
  explicit ShapeKernel(const NodeProto &node)
      : op_type_(node.op_type().as_string()), domain_(node.domain().as_string()) {}

private:
  std::string op_type_;
  std::string domain_;
};

/**
 * Builds the shape kernel corresponding to ``node``.
 *
 * The factory inspects ``node.op_type()`` (and ``node.domain()``) and
 * returns an instance of the matching concrete :cpp:class:`ShapeKernel`
 * subclass. Throws ``std::runtime_error`` when no kernel is registered
 * for the given operator.
 */
std::unique_ptr<ShapeKernel> MakeShapeKernel(const NodeProto &node);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
