// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

using namespace onnx_kernels;
using OpsetId = onnx_kernels::kernel::OpsetId;
using onnx_kernels::kernel::DefaultOpset;

/// A single (inputs, expected outputs) data set associated with a TestCase.
struct DataSet {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
};

/**
 * A backend test case mirroring ``onnx_light.backend.test.case.base.TestCase``.
 * It bundles a single-node ``ModelProto`` together with the expected input/
 * output data sets a runtime must reproduce.
 *
 * The string-typed fields (``name``, ``model_name``, ``kind``, ``tag``) are
 * declared ``const`` and must therefore be supplied at construction time.
 * ``tag`` is an optional, free-form label used to group families of cases
 * (e.g. ``"empty_shape"``, ``"nan_inf"``, ``"inference"``); it defaults to
 * the empty string for the ordinary node cases in the default ``ai.onnx``
 * domain. For test cases whose underlying node belongs to a non-default
 * operator domain (e.g. ``"ai.onnx.ml"``, ``"ai.onnx.preview.training"``),
 * :func:`Expect` defaults the tag to the node's domain string when the
 * caller does not provide an explicit one.
 */
struct TestCase {
  const std::string name;
  const std::string model_name;
  const std::string kind;
  const std::string tag;
  double rtol = 1e-3;
  double atol = 1e-7;
  ModelProto model;
  std::vector<DataSet> data_sets;

  TestCase() : kind("node"), tag() {}
  explicit TestCase(std::string name_, std::string model_name_ = "", std::string kind_ = "node",
                    std::string tag_ = "", double rtol_ = 1e-3, double atol_ = 1e-7)
      : name(std::move(name_)), model_name(std::move(model_name_)), kind(std::move(kind_)),
        tag(std::move(tag_)), rtol(rtol_), atol(atol_) {}

  // Explicit move constructor. Required because the ``const std::string``
  // members would otherwise cause the implicit move constructor to fall
  // back to ``std::string`` copy construction (and therefore not be
  // ``noexcept``), which in turn forces ``std::vector<TestCase>`` to
  // copy-construct existing elements on reallocation — impossible because
  // ``ModelProto`` is move-only. The ``const_cast`` is well-defined here:
  // every ``TestCase`` is originally allocated as a non-const object, so
  // casting away the member-level ``const`` to invoke ``std::string``'s
  // move constructor on the moved-from source does not modify an
  // actually-const object.
  TestCase(TestCase &&other) noexcept
      : name(std::move(const_cast<std::string &>(other.name))),
        model_name(std::move(const_cast<std::string &>(other.model_name))),
        kind(std::move(const_cast<std::string &>(other.kind))),
        tag(std::move(const_cast<std::string &>(other.tag))), rtol(other.rtol), atol(other.atol),
        model(std::move(other.model)), data_sets(std::move(other.data_sets)) {}

  TestCase(const TestCase &) = delete;
  TestCase &operator=(const TestCase &) = delete;
  TestCase &operator=(TestCase &&) = delete;
};

/**
 * Initializes ``model`` with ``ir_version``, ``producer_name`` and the given
 * ``opset_imports`` (default ai.onnx domain when an entry's ``domain`` is
 * empty). Mirrors the boilerplate that opens every manually-built backend
 * test case model so callers don't have to repeat it.
 */
void InitModel(ModelProto &model, int64_t ir_version, const std::vector<OpsetId> &opset_imports,
               const std::string &producer_name = "backend-test");

/**
 * Describes one tensor dimension entry used to build a ValueInfoProto.
 * - ``DimSpec(int64_t v)`` (``v >= 0``): concrete ``dim_value``.
 * - ``DimSpec("name")`` / ``DimSpec(std::string)``: symbolic ``dim_param``.
 * - ``DimSpec()``: unannotated dim (neither ``dim_value`` nor ``dim_param``).
 */
struct DimSpec {
  int64_t value = -1;
  std::string param;

  DimSpec() = default;
  DimSpec(int v) : value(v) {}
  DimSpec(int64_t v) : value(v) {}
  DimSpec(const char *p) : param(p) {}
  DimSpec(std::string p) : param(std::move(p)) {}
};

/**
 * Fills ``vi`` with a tensor-typed ValueInfo (``name``, ``elem_type`` and the
 * concrete dimension values from ``shape``). Mirrors the boilerplate every
 * manually-built graph repeats when declaring graph inputs / ``value_info``
 * / outputs for which a literal shape is already known (e.g. the gallery
 * shapes used by the shape-inference cases). For Tensor-backed metadata see
 * the ``FillValueInfo(const Tensor&, ValueInfoProto&)`` overload in
 * ``simple_tensor.h``.
 */
void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &shape);

/**
 * Overload of :ref:`AppendValueInfo` accepting a mix of concrete
 * (``DimSpec(int64_t)``), symbolic (``DimSpec("name")``) and unannotated
 * (``DimSpec()``) dimensions. Used by the shape-inference cases to declare
 * symbolic ``batch``/``seq``/``d_model``/``nnz`` dims without repeating the
 * ``TypeProto::Tensor::add_shape()`` + ``add_dim()`` boilerplate.
 */
void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<DimSpec> &dims);

/**
 * Appends a new ``DataSet`` to ``tc.data_sets`` populated with the given
 * ``inputs`` and ``outputs``. Saves the
 * ``DataSet ds; ds.inputs.push_back(...); ds.outputs.push_back(...);
 * tc.data_sets.emplace_back(std::move(ds));`` boilerplate that every
 * manually-built TestCase otherwise repeats.
 */
void AppendDataSet(TestCase &tc, std::vector<Tensor> inputs, std::vector<Tensor> outputs);

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
            std::vector<TestCase> &registry, const std::string &tag = "");

/// Function pointer registering one or more :ref:`TestCase` entries into the
/// caller-supplied ``registry``. Used by ``Collect*TestCases`` dispatch tables.
using RegisterCasesFn = void (*)(std::vector<TestCase> &);

/// Per-category dispatch table: maps an ``op_type`` to the function that
/// registers its test cases. Built once per ``Collect*TestCases`` as a
/// function-local ``static const`` so lookup is amortised O(1).
using OpRegisterMap = std::unordered_map<std::string_view, RegisterCasesFn>;

/// Invokes the ``Register*Cases`` functions declared in ``entries``.
/// When ``op_type`` is empty, every entry is invoked (order is unspecified).
/// Otherwise, only the entry whose key matches ``op_type`` (case-sensitive)
/// is invoked; if no entry matches, no registration occurs. Used by
/// per-category ``Collect*TestCases`` helpers to dispatch via a hash map
/// instead of an explicit ``if`` chain or linear scan.
void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries);

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

/**
 * Collects C++-implemented backend test node cases whose
 * :attr:`TestCase::name` matches a regular expression. Uses
 * ``std::regex_search`` semantics (substring match by default; anchor with
 * ``^...$`` to require a full match).
 *
 * @param name_regex ECMAScript regular expression matched against each
 *                   test case name. An empty string matches every case
 *                   (equivalent to :func:`CollectTestCases`).
 *
 * @return The subset of cases whose ``name`` matches ``name_regex``,
 *         in the same registration order as :func:`CollectTestCases`.
 *
 * @throws std::regex_error if ``name_regex`` is not a valid ECMAScript
 *         regular expression.
 */
std::vector<TestCase> CollectTestCasesByName(const std::string &name_regex);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
