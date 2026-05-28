// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include "onnx_backend_test/simple_tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/**
 * Lightweight opset identifier used by the backend test library.
 *
 * Mirrors the (domain, version) pair carried by ``OperatorSetIdProto`` but
 * keeps the public API of this library independent from the proto type so
 * test cases can be declared without touching the proto wire format.
 */
struct OpsetId {
  std::string domain;
  int64_t version = 0;

  OpsetId() = default;
  OpsetId(std::string domain_, int64_t version_) : domain(std::move(domain_)), version(version_) {}
};

/// Builds an :ref:`OpsetId` for the default ai.onnx domain (empty string).
inline OpsetId DefaultOpset(int64_t version) { return OpsetId(std::string(), version); }

/// A single (inputs, expected outputs) data set associated with a TestCase.
struct DataSet {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
};

/**
 * A backend test case mirroring ``onnx_light.backend.test.case.base.TestCase``.
 * It bundles a single-node ``ModelProto`` together with the expected input/
 * output data sets a runtime must reproduce.
 */
struct TestCase {
  std::string name;
  std::string model_name;
  std::string kind = "node";
  double rtol = 1e-3;
  double atol = 1e-7;
  ModelProto model;
  std::vector<DataSet> data_sets;
};

/**
 * Builds a single-node ``ModelProto`` from ``node`` and the provided typed
 * inputs/outputs, then appends a ``TestCase`` to ``registry``.
 *
 * Mirrors ``onnx_light.backend.test.case.base.expect()``. Only the inputs and
 * outputs whose name is non-empty in the node are wired into the graph.
 *
 * @param node Single-node template; its ``op_type``, ``domain`` and
 *             ``attribute``s are kept.
 * @param inputs Concrete input tensors corresponding to the non-empty entries
 *               of ``node.input``.
 * @param outputs Concrete expected output tensors corresponding to the
 *                non-empty entries of ``node.output``.
 * @param name Unique test name (used both for ``TestCase.name`` and the
 *             graph name).
 * @param opset_imports Opset imports for the generated model. If empty the
 *                      caller is responsible for ensuring a default has been
 *                      applied — typically pass at least ``DefaultOpset(since_version)``.
 * @param producer_name Producer name written into the model.
 * @param registry Output registry (appended to).
 * @throws std::invalid_argument if ``inputs.size()`` does not equal the number
 *         of non-empty entries in ``node.input`` or if ``outputs.size()`` does
 *         not equal the number of non-empty entries in ``node.output``.
 */
void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OpsetId> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry);

/// Function pointer registering one or more :ref:`TestCase` entries into the
/// caller-supplied ``registry``. Used by ``Collect*TestCases`` dispatch tables.
using RegisterCasesFn = void (*)(std::vector<TestCase> &);

/// One ``(op_type, register_fn)`` entry of a per-category dispatch table.
struct OpRegisterEntry {
  std::string_view op_type;
  RegisterCasesFn register_fn;
};

/// Invokes the ``Register*Cases`` functions declared in ``entries``.
/// When ``op_type`` is empty, every entry is invoked in declaration order.
/// Otherwise, only the entry whose ``op_type`` matches (case-sensitive) is
/// invoked; if no entry matches, no registration occurs. Used by per-category
/// ``Collect*TestCases`` helpers to dispatch via a static map instead of an
/// explicit ``if`` chain.
void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterEntry *entries, std::size_t count);

/// Convenience overload taking a fixed-size array (typically a
/// ``static constexpr`` table declared at the call site, so the dispatch
/// table is built once at program start rather than on every call).
template <std::size_t N>
inline void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                                     const OpRegisterEntry (&entries)[N]) {
  DispatchRegisterByOpType(registry, op_type, entries, N);
}

/**
 * Collects all C++-implemented backend test node cases. Each call is
 * deterministic and independent: the result owns its ``ModelProto``s and
 * ``Tensor`` data.
 *
 * @param op_type Optional operator type filter. When non-empty, only test
 *                cases whose top-level graph contains a node with this
 *                ``op_type`` are returned.
 *
 * @return A fresh registry of test cases (Abs, Add equal-shape, Add scalar
 *         broadcast).
 */
std::vector<TestCase> CollectTestCases(const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
