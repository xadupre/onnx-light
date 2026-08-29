// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_map.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Marks the per-operator ``Register*Cases`` helpers as internal to
// ``lib_onnx_backend_test``. They are only ever invoked from the ``Collect*``
// aggregators inside the library, so on ELF/Mach-O we give them hidden
// visibility to keep them out of the shared library's dynamic symbol table.
// The public ``Collect*`` entry points keep the compiler's default (visible)
// visibility. On MSVC the macro is empty (symbols are exported via
// WINDOWS_EXPORT_ALL_SYMBOLS).
#if defined(__GNUC__) || defined(__clang__)
#define ONNX_LIGHT_BACKEND_TEST_LOCAL __attribute__((visibility("hidden")))
#else
#define ONNX_LIGHT_BACKEND_TEST_LOCAL
#endif

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

using namespace ::onnx_light::core::runtime;

/**
 * Selects how a ``Register*Cases`` / ``Collect*`` helper generates its cases.
 *
 * - ``TEST`` (the default) produces the standard correctness cases with small,
 *   fixed inputs. The generated cases are byte-for-byte unchanged from before
 *   this mode existed.
 * - ``BENCHMARK`` produces cases whose inputs are enlarged so a single kernel
 *   evaluation processes enough elements to run long enough (~0.1 s) to be
 *   timed reliably. By default these are input-only performance cases; callers
 *   may explicitly request reference outputs when validating correctness. The
 *   exact sizes are hand-tuned per operator.
 */
enum class TestMode { TEST, BENCHMARK };

/// A single (inputs, expected outputs) data set associated with a TestCase.
struct DataSet {
  Tensors inputs;
  Tensors outputs;
  /// Whether ``outputs`` contains generated reference values. False explicitly
  /// marks an input-only performance data set.
  bool expected_outputs_generated = true;
  /// Map-typed inputs keyed by the graph input name.
  std::vector<Map> maps;
};

struct TestCase;

/**
 * Product of a lazily-built :ref:`TestCase`: the single-node ``ModelProto``
 * together with its input/output data sets. A ``TestCase`` stores a builder
 * returning this so that constructing the (potentially large) model and
 * running the kernel that computes the expected outputs is deferred until a
 * consumer actually needs them. Collecting a large family of cases (in
 * particular the ``BENCHMARK`` cases whose inputs contain millions of
 * elements) therefore stays cheap.
 */
struct BuiltCase {
  ModelProto model;
  std::vector<DataSet> data_sets;
};

/**
 * A backend test case mirroring ``onnx_light.backend.test.case.base.TestCase``.
 * It bundles a single-node ``ModelProto`` together with the expected input/
 * output data sets a runtime must reproduce.
 *
 * The model is not stored directly. Every case built through :func:`Expect`
 * (both the correctness ``TEST`` cases and the ``BENCHMARK`` cases) is *lazy*:
 * it carries a ``build`` closure that produces the :ref:`BuiltCase` — the
 * ``ModelProto`` and its ``data_sets`` — on first access via :func:`model` /
 * :func:`data_sets` / :func:`Materialize`. A handful of manually-assembled
 * cases (control-flow, sequence, ...) are instead *eager*: they populate the
 * model cache with :func:`emplace_model` / :func:`set_model` and append their
 * data sets directly, so ``build`` is left unset and :func:`Materialize` is a
 * no-op. Every case records ``declared_input_element_counts`` /
 * ``declared_output_element_counts`` so its sizing can be checked without
 * running the (potentially expensive) builder.
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

  /// Optional builder producing the model + data sets on demand. When set the
  /// case is *lazy*: ``data_sets`` starts empty and the model is unbuilt until
  /// :func:`Materialize` / :func:`model` / :func:`data_sets` runs the builder
  /// once.
  std::function<BuiltCase(bool)> build;

  /// Whether this case retains generated reference outputs. Benchmark
  /// collection may disable these to build input-only performance cases.
  bool expected_outputs_generated = true;

  /// Declared element count of each input/output, recorded without
  /// materializing tensor data. Used to validate the sizing of a case (in
  /// particular the large benchmark cases) without running ``build``.
  std::vector<int64_t> declared_input_element_counts;
  std::vector<int64_t> declared_output_element_counts;

  TestCase() : kind("node"), tag() {}
  explicit TestCase(std::string name_, std::string model_name_ = "", std::string kind_ = "node",
                    std::string tag_ = "", double atol_ = 1e-7, double rtol_ = 1e-3)
      : name(std::move(name_)), model_name(std::move(model_name_)), kind(std::move(kind_)),
        tag(std::move(tag_)), rtol(rtol_), atol(atol_) {}

  // Explicit move constructor. Required because the ``const std::string``
  // members would otherwise cause the implicit move constructor to fall
  // back to ``std::string`` copy construction (and therefore not be
  // ``noexcept``), which in turn forces ``std::vector<TestCase>`` to
  // copy-construct existing elements on reallocation — impossible because the
  // ``std::unique_ptr`` model cache is move-only. The ``const_cast`` is
  // well-defined here: every ``TestCase`` is originally allocated as a
  // non-const object, so casting away the member-level ``const`` to invoke
  // ``std::string``'s move constructor on the moved-from source does not
  // modify an actually-const object.
  TestCase(TestCase &&other) noexcept
      : name(std::move(const_cast<std::string &>(other.name))),
        model_name(std::move(const_cast<std::string &>(other.model_name))),
        kind(std::move(const_cast<std::string &>(other.kind))),
        tag(std::move(const_cast<std::string &>(other.tag))), rtol(other.rtol), atol(other.atol),
        build(std::move(other.build)), expected_outputs_generated(other.expected_outputs_generated),
        declared_input_element_counts(std::move(other.declared_input_element_counts)),
        declared_output_element_counts(std::move(other.declared_output_element_counts)),
        data_sets_(std::move(other.data_sets_)), model_(std::move(other.model_)) {}

  TestCase(const TestCase &) = delete;
  TestCase &operator=(const TestCase &) = delete;
  TestCase &operator=(TestCase &&) = delete;

  /// Creates (if needed) and returns the mutable model cache. Used by eager
  /// case builders that populate the ``ModelProto`` in place. Clears any
  /// previously-built cache.
  ModelProto &emplace_model() {
    model_ = std::make_unique<ModelProto>();
    return *model_;
  }

  /// Stores an already-built model into the cache.
  void set_model(ModelProto model) { model_ = std::make_unique<ModelProto>(std::move(model)); }

  /// Returns whether the case has already been materialized (its model cache
  /// exists). Introspection helper that does *not* trigger materialization.
  bool materialized() const { return model_ != nullptr; }

  /// Returns whether the case is lazy (carries a ``build`` closure). Does not
  /// trigger materialization.
  bool is_lazy() const { return static_cast<bool>(build); }

  /// Returns whether the case has generated expected outputs.
  bool has_expected_outputs() const { return expected_outputs_generated; }

  /// Configures whether the case retains expected outputs.
  void set_expected_outputs_generated(bool value) {
    expected_outputs_generated = value;
    if (!value && materialized()) {
      for (DataSet &data_set : data_sets_) {
        data_set.expected_outputs_generated = false;
        data_set.outputs.clear();
      }
    }
  }

  /// Runs the ``build`` closure once (if the case is lazy and not yet built),
  /// materializing the model cache and ``data_sets``. No-op for eager cases and
  /// for already-materialized cases.
  void Materialize() {
    if (model_ || !build) {
      return;
    }
    BuiltCase built = build(expected_outputs_generated);
    model_ = std::make_unique<ModelProto>(std::move(built.model));
    if (data_sets_.empty()) {
      data_sets_ = std::move(built.data_sets);
    }
  }

  /// Lazily builds (once) and returns the model.
  ModelProto &model() {
    EnsureMaterialized();
    if (!model_) {
      model_ = std::make_unique<ModelProto>();
    }
    return *model_;
  }

  /// Const overload. Materializes the case (model *and* data sets) on first
  /// access via the same builder, so ``data_sets`` is consistent in const
  /// contexts as well.
  const ModelProto &model() const {
    EnsureMaterialized();
    if (!model_) {
      model_ = std::make_unique<ModelProto>();
    }
    return *model_;
  }

  /// Lazily builds (once) and returns the mutable data sets. Eager producers
  /// also use this to append their data sets (``build`` is unset, so
  /// materialization is a no-op).
  std::vector<DataSet> &data_sets() {
    EnsureMaterialized();
    return data_sets_;
  }

  /// Const overload. Materializes the case on first access.
  const std::vector<DataSet> &data_sets() const {
    EnsureMaterialized();
    return data_sets_;
  }

private:
  /// Data sets. Empty until the ``build`` closure has run (for lazy cases) or
  /// until an eager producer appends them directly via :func:`data_sets`.
  mutable std::vector<DataSet> data_sets_;
  mutable std::unique_ptr<ModelProto> model_;

  // Materializes through a const accessor. The object is never truly const
  // (every ``TestCase`` is allocated non-const), so casting away ``const`` to
  // run the builder and populate the caches is well-defined.
  void EnsureMaterialized() const { const_cast<TestCase *>(this)->Materialize(); }
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
 * Appends a new ``DataSet`` to ``tc.data_sets()`` populated with the given
 * ``inputs`` and ``outputs``. Saves the
 * ``DataSet ds; ds.inputs.push_back(...); ds.outputs.push_back(...);
 * tc.data_sets().emplace_back(std::move(ds));`` boilerplate that every
 * manually-built TestCase otherwise repeats.
 */
void AppendDataSet(TestCase &tc, Tensors inputs, Tensors outputs);

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

/**
 * Collects all C++-implemented backend test node cases. Each call is
 * deterministic and independent: the result owns its ``ModelProto``s and
 * ``Tensor`` data.
 *
 * Iterates the collector functions registered via
 * :func:`RegisterTestCasesCollector` (typically every ``Collect*TestCases``
 * category in ``lib_onnx_backend_test``).
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
 *                   of the standard correctness cases. Benchmark cases are
 *                   input-only unless ``generate_benchmark_expected_outputs``
 *                   is true. Defaults to :cpp:enumerator:`TestMode::TEST`.
 * @param generate_benchmark_expected_outputs Generate reference outputs for
 *                   benchmark cases. Ignored for ``TEST``, whose outputs are
 *                   always generated.
 *
 * @return A fresh registry of test cases (Abs, Add equal-shape, Add scalar
 *         broadcast).
 */
std::vector<TestCase> CollectTestCases(const std::string &op_type = "", bool include_big = false,
                                       TestMode mode = TestMode::TEST,
                                       bool generate_benchmark_expected_outputs = false);

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
                                             TestMode mode = TestMode::TEST,
                                             bool generate_benchmark_expected_outputs = false);

/**
 * Returns the single C++-implemented backend test case whose
 * :attr:`TestCase::name` equals ``name`` exactly, or an empty optional if no
 * such case exists.
 *
 * Unlike :func:`CollectTestCasesByName` this performs a plain string
 * comparison instead of compiling a ``std::regex`` and avoids copying all
 * non-matching cases. It is therefore the most efficient way to retrieve a
 * single known case by name.
 *
 * @param name        The exact test case name to look up (e.g.
 *                    ``"test_cc_abs"``).
 * @param include_big When ``false`` (the default), a case whose name contains
 *                    ``"_big_"`` is excluded even if it matches ``name``.
 * @param mode        Forwarded to :func:`CollectTestCases`; selects standard
 *                    or benchmark-sized case generation.
 *
 * @return A vector containing at most one :class:`TestCase`. An empty vector
 *         signals that no case with the requested name was found.
 */
std::vector<TestCase> GetTestCaseByName(const std::string &name, bool include_big = false,
                                        TestMode mode = TestMode::TEST,
                                        bool generate_benchmark_expected_outputs = false);

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
