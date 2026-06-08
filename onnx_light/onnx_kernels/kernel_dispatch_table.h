// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_proto/onnx.h"

#include <functional>
#include <string>
#include <unordered_map>

/**
 * @file kernel_dispatch_table.h
 * @brief Static, name-keyed table that maps an ONNX
 *        ``(domain, op_type)`` pair to a per-operator trampoline able
 *        to execute the matching backend-test kernel from a
 *        :cpp:class:`NodeProto` + :cpp:class:`RuntimeContext`.
 *
 * The table powers :cpp:func:`onnx_kernels::RunNode` (defined in
 * ``run_nodes.cc``). It is kept in its own translation unit so that
 * adding a new operator requires touching only one file: the kernel
 * implementation under ``onnx_kernels/kernels`` and one new entry
 * in :cpp:func:`KernelDispatchTable` in ``kernel_dispatch_table.cc``.
 *
 * The default ONNX domain (empty string in ``NodeProto::domain()``)
 * is normalised to ``"ai.onnx"`` by the caller before lookup, so all
 * keys in the table use the explicit ``"ai.onnx:<op_type>"`` form.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

/**
 * Signature of every per-operator trampoline registered in
 * :cpp:func:`KernelDispatchTable`. Implementations read their inputs
 * from ``rt.tensors()`` by name, call the matching kernel
 * (constructed with ``rt.kernel_ctx()``), and insert the produced
 * outputs back into ``rt.tensors()`` under the names declared by
 * ``node.output(i)``.
 */
using NodeKernelFn = std::function<void(const NodeProto &node, RuntimeContext &rt)>;

/**
 * Returns the ``"<domain>:<op_type>"`` dispatch table used by
 * :cpp:func:`onnx_kernels::RunNode`. The table is constructed on first
 * use and shared across calls (the same reference is returned every
 * time). The default ONNX domain (empty string in
 * ``NodeProto::domain()``) is normalised to ``"ai.onnx"`` before
 * lookup; adding a new operator only requires inserting one new entry
 * in this table.
 */
const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable();

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
