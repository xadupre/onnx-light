// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file wrapper_kernels.h
 * @brief Reusable wrapper kernel classes that carry the per-node "trampoline"
 *        logic previously duplicated in the free ``Make*Trampoline`` helpers.
 *
 * A concrete operator kernel (``kernel::Abs``, ``kernel::Add``, ...) only knows
 * how to compute an output ``Tensor`` from already-materialised input
 * ``Tensor``s. Turning such a kernel into a
 * :cpp:type:`core::runtime::KernelInvokeFn` — the closure the runtime invokes
 * per node — requires reading the node's inputs from the
 * :cpp:class:`RuntimeContext`, calling the concrete kernel and storing the
 * produced output back. :cpp:class:`UnaryKernel` and :cpp:class:`BinaryKernel`
 * bundle that boilerplate together with the concrete kernel, so the dispatch
 * table entries become ``UnaryKernel<kernel::Abs>::Factory()`` /
 * ``BinaryKernel<kernel::Add>::Factory()``.
 */

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

using ::onnx_light::core::runtime::GetInput;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::KernelInvokeFn;
using ::onnx_light::core::runtime::NodeKernelFn;
using ::onnx_light::core::runtime::RequireInputCount;
using ::onnx_light::core::runtime::RequireOutputCount;
using ::onnx_light::core::runtime::RuntimeContext;
using ::onnx_light::core::runtime::SetOutput;
using ::onnx_light::core::runtime::Tensor;

/**
 * Wraps a concrete element-wise unary kernel ``KernelT`` (whose call operator
 * has the shape ``Tensor operator()(const Tensor &x, RuntimeContext *rt)``)
 * into a runtime-invocable kernel. The wrapper owns one ``KernelT`` instance
 * constructed from the node's :cpp:class:`KernelContext`; ``operator()`` reads
 * the single input from ``rt`` and writes the produced output back.
 */
template <class KernelT> class UnaryKernel {
public:
  explicit UnaryKernel(const KernelContext &ctx) : kernel_(ctx) {}

  /// Reads input 0 from ``rt``, invokes the concrete kernel and stores output 0.
  void operator()(const NodeProto &node, RuntimeContext &rt) {
    const Tensor &x = GetInput(node, 0, rt.tensors());
    SetOutput(node, 0, kernel_(x, &rt), rt);
  }

  /// Returns the dispatch-table factory: it validates the node's I/O counts,
  /// constructs the wrapper from ``rt.kernel_ctx()`` and returns it as the
  /// ready-to-invoke closure.
  static NodeKernelFn Factory() {
    return [](const NodeProto &node, RuntimeContext &rt) -> KernelInvokeFn {
      RequireInputCount(node, 1);
      RequireOutputCount(node, 1);
      return UnaryKernel<KernelT>(rt.kernel_ctx());
    };
  }

private:
  KernelT kernel_;
};

/**
 * Wraps a concrete element-wise binary kernel ``KernelT`` (whose call operator
 * has the shape ``Tensor operator()(const Tensor &x, const Tensor &y,
 * RuntimeContext *rt)``) into a runtime-invocable kernel, analogously to
 * :cpp:class:`UnaryKernel`.
 */
template <class KernelT> class BinaryKernel {
public:
  explicit BinaryKernel(const KernelContext &ctx) : kernel_(ctx) {}

  /// Reads inputs 0 and 1 from ``rt``, invokes the concrete kernel and stores
  /// output 0.
  void operator()(const NodeProto &node, RuntimeContext &rt) {
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const Tensor &y = GetInput(node, 1, rt.tensors());
    SetOutput(node, 0, kernel_(x, y, &rt), rt);
  }

  /// Returns the dispatch-table factory (see :cpp:func:`UnaryKernel::Factory`).
  static NodeKernelFn Factory() {
    return [](const NodeProto &node, RuntimeContext &rt) -> KernelInvokeFn {
      RequireInputCount(node, 2);
      RequireOutputCount(node, 1);
      return BinaryKernel<KernelT>(rt.kernel_ctx());
    };
  }

private:
  KernelT kernel_;
};

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
