// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/simple_map.h"
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

/**
 * Selects how a ``Register*Cases`` / ``Collect*`` helper generates its cases.
 *
 * - ``TEST`` (the default) produces the standard correctness cases with small,
 *   fixed inputs. The generated cases are byte-for-byte unchanged from before
 *   this mode existed.
 * - ``BENCHMARK`` produces cases whose inputs are enlarged so a single kernel
 *   evaluation processes enough elements to run long enough (~0.1 s) to be
 *   timed reliably. The exact sizes are hand-tuned per operator.
 */
enum class TestMode { TEST, BENCHMARK };

/// A single (inputs, expected outputs) data set associated with a TestCase.
struct DataSet {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  /// Map-typed inputs keyed by the graph input name.
  std::vector<Map> maps;
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
                    std::string tag_ = "", double atol_ = 1e-7, double rtol_ = 1e-3)
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
 * Overload of :ref:`AppendValueInfo` accepting a mix of concrete
 * (``DimSpec(int64_t)``), symbolic (``DimSpec("name")``) and unannotated
 * (``DimSpec()``) dimensions. Used by the shape-inference cases to declare
 * symbolic ``batch``/``seq``/``d_model``/``nnz`` dims without repeating the
 * ``TypeProto::Tensor::add_shape()`` + ``add_dim()`` boilerplate.
 */
void AppendValueInfo(ValueInfoProto &vi, const std::string &name, TensorProto::DataType elem_type,
                     const std::vector<DimSpec> &dims);

/**
 * Describes an ONNX value type for a graph value-info, supporting the
 * container kinds the backend test cases need: a plain ``Tensor``, a
 * ``Sequence`` of an element type, or a ``Map`` from a key type to a value
 * type. Built via the factory helpers :func:`TensorTypeSpec`,
 * :func:`SequenceTypeSpec` and :func:`MapTypeSpec` and consumed by
 * :func:`AppendValueInfo` / :func:`Expect` to emit value-infos whose declared
 * schema type differs from the materialized ``Tensor`` representation (e.g.
 * sequence- or map-valued outputs).
 */
struct TypeSpec {
  enum class Kind { kTensor, kSequence, kMap };

  Kind kind = Kind::kTensor;
  /// For ``kTensor``: the tensor element type. For ``kMap``: the key type.
  int32_t elem_type = 0;
  /// For ``kTensor`` only: whether a (possibly empty) shape is declared.
  bool has_shape = false;
  /// For ``kTensor`` only: the concrete dimension values of the shape.
  std::vector<int64_t> shape;
  /// Nested element type. For ``kSequence`` the single sequence element type,
  /// for ``kMap`` the single map value type; empty for ``kTensor``.
  std::vector<TypeSpec> children;
};

/// Returns a ``TypeSpec`` describing a ``Tensor`` of ``elem_type`` with no
/// declared shape (used e.g. for map value types).
TypeSpec TensorTypeSpec(int32_t elem_type);

/// Returns a ``TypeSpec`` describing a ``Tensor`` of ``elem_type`` whose
/// declared shape has the given concrete dimension values (an empty ``shape``
/// declares a rank-0 / scalar shape).
TypeSpec TensorTypeSpec(int32_t elem_type, std::vector<int64_t> shape);

/// Returns a ``TypeSpec`` describing a ``Sequence`` whose elements have type
/// ``elem``.
TypeSpec SequenceTypeSpec(TypeSpec elem);

/// Returns a ``TypeSpec`` describing a ``Map`` from ``key_type`` keys to
/// ``value`` values.
TypeSpec MapTypeSpec(int32_t key_type, TypeSpec value);

/// Fills ``vi`` with ``name`` and the type described by ``spec``.
void AppendValueInfo(ValueInfoProto &vi, const std::string &name, const TypeSpec &spec);

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
 * @param tag Optional grouping tag (defaults to the node domain for
 *            non-default operator domains).
 * @param output_types Optional per-output declared type specs. When non-empty
 *                     it must contain one entry per output tensor; each output
 *                     value-info is then declared from its ``TypeSpec`` instead
 *                     of the materialized tensor type. Used to declare
 *                     ``Sequence`` / ``Map`` valued outputs whose runtime
 *                     representation is a plain ``Tensor``.
 * @throws std::invalid_argument if ``inputs.size()`` does not equal the number
 *         of non-empty entries in ``node.input`` or if ``outputs.size()`` does
 *         not equal the number of non-empty entries in ``node.output``, or if
 *         ``output_types`` is non-empty and its size does not equal
 *         ``outputs.size()``.
 */
void Expect(const NodeProto &node, const std::vector<Tensor> &inputs,
            const std::vector<Tensor> &outputs, const std::string &name,
            const std::vector<OpsetId> &opset_imports, const std::string &producer_name,
            std::vector<TestCase> &registry, const std::string &tag = "",
            const std::vector<TypeSpec> &output_types = {});

/// Function pointer registering one or more :ref:`TestCase` entries into the
/// caller-supplied ``registry``. Used by ``Collect*TestCases`` dispatch tables.
using RegisterCasesFn = void (*)(std::vector<TestCase> &);

/// Per-category dispatch table: maps an ``op_type`` to the function that
/// registers its test cases. Built once per ``Collect*TestCases`` as a
/// function-local ``static const`` so lookup is amortised O(1).
using OpRegisterMap = std::unordered_map<std::string_view, RegisterCasesFn>;

/// Mode-aware variant of :ref:`RegisterCasesFn`. In addition to the output
/// ``registry`` it receives a :ref:`TestMode` selecting standard (``TEST``) or
/// benchmark-sized (``BENCHMARK``) case generation.
using RegisterCasesModeFn = void (*)(std::vector<TestCase> &, TestMode);

/// Mode-aware variant of :ref:`OpRegisterMap` whose values are
/// :ref:`RegisterCasesModeFn`.
using OpRegisterModeMap = std::unordered_map<std::string_view, RegisterCasesModeFn>;

/// Invokes the ``Register*Cases`` functions declared in ``entries``.
/// When ``op_type`` is empty, every entry is invoked (order is unspecified).
/// Otherwise, only the entry whose key matches ``op_type`` (case-sensitive)
/// is invoked; if no entry matches, no registration occurs. Used by
/// per-category ``Collect*TestCases`` helpers to dispatch via a hash map
/// instead of an explicit ``if`` chain or linear scan.
void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries);

/// Mode-aware overload of :ref:`DispatchRegisterByOpType`. Forwards ``mode`` to
/// each invoked :ref:`RegisterCasesModeFn` so a category can generate either the
/// standard (``TestMode::TEST``) or benchmark-sized (``TestMode::BENCHMARK``)
/// cases.
void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterModeMap &entries, TestMode mode);

/// Default element count for a 1-D float benchmark input of a cheap
/// element-wise operator. Sized (4M floats = 16 MiB) so a single kernel
/// evaluation processes enough data to be timed reliably (~0.1 s). Operators
/// with heavier per-element cost (transcendental, matmul, ...) pass a smaller
/// explicit size to the benchmark helpers below.
inline constexpr int64_t kBenchmarkElementwiseSize = 1 << 22;

/**
 * Appends a single benchmark :ref:`TestCase` for a unary element-wise float
 * operator. ``kernel`` is any callable mapping the input ``Tensor`` to the
 * output ``Tensor`` (typically the operator's kernel functor); the expected
 * output is computed by invoking it. The generated node carries no attributes,
 * so operators whose behaviour depends on attributes should build their own
 * benchmark case instead.
 */
template <typename Kernel>
void ExpectBenchmarkUnaryFloat(const std::string &op_type, const Kernel &kernel,
                               const std::string &name, const OpsetId &opset,
                               std::vector<TestCase> &registry,
                               int64_t size = kBenchmarkElementwiseSize,
                               uint64_t seed = 987654321ULL, const std::string &input_name = "x",
                               const std::string &output_name = "y") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(input_name);
  node.add_output(output_name);
  Tensor x = Tensor::FromFloat("", {size}, Randn<float>({size}, seed));
  Tensor y = kernel(x);
  Expect(node, {x}, {y}, name, {opset}, "backend-test", registry);
}

/**
 * Appends a single benchmark :ref:`TestCase` for a binary element-wise float
 * operator with two equally-shaped 1-D inputs. ``kernel`` is any callable
 * mapping the two input ``Tensor``s to the output ``Tensor``; the expected
 * output is computed by invoking it. The generated node carries no attributes.
 */
template <typename Kernel>
void ExpectBenchmarkBinaryFloat(const std::string &op_type, const Kernel &kernel,
                                const std::string &name, const OpsetId &opset,
                                std::vector<TestCase> &registry,
                                int64_t size = kBenchmarkElementwiseSize,
                                uint64_t seed = 987654321ULL) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  Tensor x = Tensor::FromFloat("", {size}, Randn<float>({size}, seed));
  Tensor y = Tensor::FromFloat("", {size}, Randn<float>({size}, seed + 1));
  Tensor z = kernel(x, y);
  Expect(node, {x, y}, {z}, name, {opset}, "backend-test", registry);
}

/**
 * Collects all C++-implemented backend test node cases. Each call is
 * deterministic and independent: the result owns its ``ModelProto``s and
 * ``Tensor`` data.
 *
 * @param op_type    Optional operator type filter. When non-empty, only test
 *                   cases whose top-level graph contains a node with this
 *                   ``op_type`` are returned.
 * @param include_big When ``false`` (the default), test cases whose name
 *                   contains the substring ``"_big_"`` are excluded from the
 *                   result. Pass ``true`` to include them; the big models are
 *                   intentionally opt-in because they carry large weight
 *                   tensors that make exhaustive test loops slow.
 * @param mode       When :cpp:enumerator:`TestMode::BENCHMARK`, categories that
 *                   support it emit benchmark-sized cases (large inputs) instead
 *                   of the standard correctness cases. Defaults to
 *                   :cpp:enumerator:`TestMode::TEST`.
 *
 * @return A fresh registry of test cases (Abs, Add equal-shape, Add scalar
 *         broadcast).
 */
std::vector<TestCase> CollectTestCases(const std::string &op_type = "", bool include_big = false,
                                       TestMode mode = TestMode::TEST);

/**
 * Collects C++-implemented backend test node cases whose
 * :attr:`TestCase::name` matches a regular expression. Uses
 * ``std::regex_search`` semantics (substring match by default; anchor with
 * ``^...$`` to require a full match).
 *
 * @param name_regex  ECMAScript regular expression matched against each
 *                    test case name. An empty string matches every case
 *                    (equivalent to :func:`CollectTestCases`).
 * @param include_big When ``false`` (the default), test cases whose name
 *                    contains ``"_big_"`` are excluded before the regex
 *                    filter is applied. Pass ``true`` to include them.
 * @param mode        Forwarded to :func:`CollectTestCases`; selects standard
 *                    or benchmark-sized case generation.
 *
 * @return The subset of cases whose ``name`` matches ``name_regex``,
 *         in the same registration order as :func:`CollectTestCases`.
 *
 * @throws std::regex_error if ``name_regex`` is not a valid ECMAScript
 *         regular expression.
 */
std::vector<TestCase> CollectTestCasesByName(const std::string &name_regex,
                                             bool include_big = false,
                                             TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
