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

/// Sentinel value used by :cpp:func:`MakeShapeKernel` to indicate that
/// the caller does not know (or does not care about) the opset version
/// of the node being processed. Concrete kernels treat it as
/// "use the latest supported opset".
inline constexpr int kUnknownOpsetVersion = -1;

/**
 * Abstract base class for all shape kernels.
 *
 * A shape kernel encapsulates the shape (and dtype) inference logic for
 * a single ONNX operator. The kernel is constructed once from a
 * :cpp:class:`NodeProto` together with the *opset version* of the
 * operator's domain (so the right schema revision can be selected) and
 * may then be invoked any number of times against different concrete
 * input tensors.
 *
 * The opset version is not stored on :cpp:class:`NodeProto` itself: it
 * comes from the surrounding ``ModelProto``'s ``opset_import`` list and
 * must therefore be supplied explicitly by the caller. Pass
 * :cpp:var:`kUnknownOpsetVersion` to defer the choice to the kernel,
 * which will fall back to the latest opset it supports.
 */
class ShapeKernel {
public:
  virtual ~ShapeKernel() = default;

  /// The ``op_type`` of the node this kernel was built from.
  const std::string &OpType() const noexcept { return op_type_; }

  /// The ``domain`` of the node this kernel was built from.
  const std::string &Domain() const noexcept { return domain_; }

  /// The opset version supplied at construction time, or
  /// :cpp:var:`kUnknownOpsetVersion` when the caller did not provide
  /// one.
  int OpsetVersion() const noexcept { return opset_version_; }

  /**
   * The effective *since_version* of the operator schema selected for
   * shape inference. This is always a concrete, positive integer (i.e.
   * never :cpp:var:`kUnknownOpsetVersion`) and corresponds to the
   * highest schema revision in the operator's history whose
   * ``since_version`` is ``<= OpsetVersion()``. When
   * :cpp:func:`OpsetVersion` is :cpp:var:`kUnknownOpsetVersion`, this
   * returns the latest supported revision.
   */
  int SinceVersion() const noexcept { return since_version_; }

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
  /// Constructs the kernel from a ``NodeProto``, caching its identity
  /// and the opset version under which it should perform shape
  /// inference. Subclasses are responsible for resolving the matching
  /// schema revision and passing it as ``since_version``.
  ShapeKernel(const NodeProto &node, int opset_version, int since_version)
      : op_type_(node.op_type().as_string()), domain_(node.domain().as_string()),
        opset_version_(opset_version), since_version_(since_version) {}

private:
  std::string op_type_;
  std::string domain_;
  int opset_version_;
  int since_version_;
};

/**
 * Builds the shape kernel corresponding to ``node``.
 *
 * @param node           The ``NodeProto`` to build the kernel from.
 * @param opset_version  The opset version of ``node.domain()`` as
 *                       declared in the surrounding model's
 *                       ``opset_import``. Pass
 *                       :cpp:var:`kUnknownOpsetVersion` to let the
 *                       kernel fall back to its latest supported
 *                       revision.
 *
 * The factory inspects ``node.op_type()`` (and ``node.domain()``) and
 * returns an instance of the matching concrete :cpp:class:`ShapeKernel`
 * subclass. Throws ``std::runtime_error`` when no kernel is registered
 * for the given operator or when ``opset_version`` is older than the
 * earliest revision supported by the kernel.
 */
std::unique_ptr<ShapeKernel> MakeShapeKernel(const NodeProto &node,
                                             int opset_version = kUnknownOpsetVersion);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
