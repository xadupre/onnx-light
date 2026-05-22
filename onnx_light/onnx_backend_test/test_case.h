// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include "onnx_backend_test/tensor.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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
 *                      applied — typically pass at least ``{"", since_version}``.
 * @param producer_name Producer name written into the model.
 * @param registry Output registry (appended to).
 */
void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OperatorSetIdProto> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry);

/**
 * Returns all C++-implemented backend test node cases. Each call is
 * deterministic and independent: the result owns its ``ModelProto``s and
 * ``Tensor`` data.
 */
std::vector<TestCase> CollectTestCases();

// ---------------------------------------------------------------------------
// Per-operator registration helpers — exposed so individual cases live in
// separate translation units yet can be invoked from CollectTestCases().
// ---------------------------------------------------------------------------

/// Registers the ``Abs`` backend test node case(s).
void RegisterAbsCases(std::vector<TestCase> &registry);

/// Registers the ``Add`` backend test node case(s).
void RegisterAddCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
