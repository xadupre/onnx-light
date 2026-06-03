// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``text`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// Most kernels provide two flavors of ``operator()``:
//
//   * The returning overload (``Tensor operator()(...) const``) allocates a
//     fresh ``Tensor`` whose ``string_data`` buffer is owned by the returned
//     value.
//   * The in-place overload (``void operator()(..., Tensor &output) const``)
//     writes results into a caller-supplied output tensor whose
//     ``string_data`` vector has already been sized. The caller is
//     responsible for setting ``output.data_type`` to
//     ``DataType::STRING``, ``output.shape`` to the broadcasted
//     shape and ``output.string_data`` to the broadcasted element count; the
//     kernel validates these attributes and throws ``std::invalid_argument``
//     on mismatch.
//
// Multi-output kernels instead expose a returning overload that bundles every
// output tensor in a ``std::pair`` or ``std::vector``.
//
// ``StringConcat`` mirrors the ONNX ``StringConcat`` operator (since opset
// 20 in the ai.onnx domain): it concatenates two ``tensor(string)`` inputs
// element-wise with NumPy-style bidirectional broadcasting. ``output[i] =
// x[i] + y[i]`` after broadcasting both inputs to a common shape.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's ``string_data`` buffer may
// alias either input's buffer. ``StringConcat`` writes new strings whose
// bytes depend on both inputs, so aliasing an input is not permitted.
// ---------------------------------------------------------------------------

/// Element-wise concatenation of two ``tensor(string)`` inputs with
/// NumPy-style bidirectional broadcasting.
class StringConcat : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Output bytes depend on both inputs, so the output buffer cannot
  /// safely alias either input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``StringSplit`` operator
/// (ai.onnx, since opset 20).
///
/// Splits each element of the input ``tensor(string)`` according to the
/// ``delimiter`` attribute and returns:
///
/// * ``Y`` — a padded string tensor whose shape is ``input.shape + [M]``,
///   where ``M`` is the maximum number of substrings produced by any input
///   element;
/// * ``Z`` — an ``INT64`` tensor of shape ``input.shape`` storing the number
///   of substrings produced for each input element.
///
/// When ``delimiter`` is empty, the operator follows the ONNX reference
/// semantics and splits on consecutive whitespace.
class StringSplit : public KernelBase {
public:
  using KernelBase::KernelBase;

  std::pair<Tensor, Tensor> operator()(const Tensor &x, const std::string &delimiter = "",
                                       int64_t maxsplit = -1) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``StringNormalizer`` operator
/// (ai.onnx, since opset 10). Removes elements found in ``stopwords``
/// from the input string tensor (case-sensitively or not) and applies
/// the requested ``case_change_action`` (``"LOWER"``, ``"UPPER"`` or
/// ``"NONE"``) to the surviving elements.
///
/// Accepts only ``[C]``-shaped or ``[1, C]``-shaped ``tensor(string)``
/// inputs. When every element is dropped, the output is a single
/// empty string with shape ``[1]`` (for ``[C]`` input) or ``[1, 1]``
/// (for ``[1, C]`` input).
///
/// Case folding and comparison use the ``"C"`` locale (ASCII): the
/// ``locale`` attribute, when present, is ignored. This matches the
/// behavior of the upstream onnx reference implementation, which only
/// supports the ``"en_US"``-equivalent ASCII semantics.
class StringNormalizer : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// ``case_change_action`` attribute values.
  enum class CaseChangeAction { kNone, kLower, kUpper };

  /// Converts the string form of the ``case_change_action`` attribute
  /// (``"NONE"``, ``"LOWER"`` or ``"UPPER"``) into the matching
  /// enumerator. Throws ``std::invalid_argument`` for any other value.
  static CaseChangeAction ParseCaseChangeAction(const std::string &value);

  /// Allocating overload. ``stopwords`` may be empty.
  Tensor operator()(const Tensor &x, CaseChangeAction case_change_action = CaseChangeAction::kNone,
                    bool is_case_sensitive = false,
                    const std::vector<std::string> &stopwords = {}) const;

  /// In-place overload. ``output`` must be a ``tensor(string)`` with
  /// the shape returned by :cpp:func:`ComputeOutputShape` and with a
  /// matching number of pre-allocated entries in ``string_data``.
  void operator()(const Tensor &x, CaseChangeAction case_change_action, bool is_case_sensitive,
                  const std::vector<std::string> &stopwords, Tensor &output) const;

  /// Computes the output shape given the input shape and the number
  /// of surviving (non-stopword) elements. Encapsulates the
  /// ``[C] → [max(1, kept)]`` / ``[1, C] → [1, max(1, kept)]`` rule.
  static std::vector<int64_t> ComputeOutputShape(const std::vector<int64_t> &input_shape,
                                                 int64_t kept);

  /// Output bytes depend on the input contents and on the
  /// ``stopwords`` set; the output buffer cannot safely alias the
  /// input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``RegexFullMatch`` operator
/// (ai.onnx, since opset 20).
///
/// Performs an element-wise full-match regex test on a
/// ``tensor(string)`` input and produces a ``tensor(bool)`` output
/// with the same shape. The regex pattern is provided either as the
/// ``pattern`` attribute on the underlying node or directly to
/// ``operator()`` as a ``std::string``.
///
/// The ONNX specification references the
/// `RE2 <https://github.com/google/re2/wiki/Syntax>`_ regex syntax.
/// This reference kernel does not link against RE2; it uses the
/// C++ standard library's ECMAScript regex grammar
/// (``std::regex_match``), which is sufficient for the common
/// constructs exercised by the upstream backend tests (anchors,
/// character classes, alternation, quantifiers, and groups).
/// Patterns that rely on RE2-specific syntax not supported by
/// ``std::regex`` will throw ``std::invalid_argument``.
class RegexFullMatch : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Allocating overload. Returns a freshly allocated ``tensor(bool)``
  /// with the same shape as ``x``. Each output byte is ``1`` when the
  /// corresponding input string is fully matched by ``pattern`` and
  /// ``0`` otherwise.
  Tensor operator()(const Tensor &x, const std::string &pattern) const;

  /// In-place overload. ``output`` must already be a ``tensor(bool)``
  /// with the same shape as ``x`` and a pre-sized ``data`` buffer of
  /// ``x.element_count()`` bytes.
  void operator()(const Tensor &x, const std::string &pattern, Tensor &output) const;

  /// Output bytes depend on the input contents and on the regex
  /// pattern; the output buffer cannot safely alias the input
  /// ``string_data`` buffer (different layout / element size).
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``TfIdfVectorizer`` operator
/// (ai.onnx, since opset 9).
///
/// Extracts n-grams from a ``[C]``- or ``[N, C]``-shaped integer
/// (``tensor(int32)`` or ``tensor(int64)``) or string
/// (``tensor(string)``) input and produces a ``tensor(float)`` count
/// (``"TF"``) / weight (``"IDF"`` / ``"TFIDF"``) vector with last
/// dimension ``max(ngram_indexes) + 1``.
///
/// The kernel parameters mirror the operator attributes:
///
/// * ``mode`` &mdash; one of ``"TF"``, ``"IDF"`` or ``"TFIDF"``.
/// * ``min_gram_length`` / ``max_gram_length`` &mdash; inclusive
///   range of n-gram sizes to extract.
/// * ``max_skip_count`` &mdash; maximum number of tokens to skip
///   between consecutive elements of an n-gram (``skip_distance``
///   ranges from 1 to ``max_skip_count + 1``).
/// * ``ngram_counts`` &mdash; starting offsets of 1-grams, 2-grams,
///   ... in the pool (CSR-style).
/// * ``ngram_indexes`` &mdash; output coordinate of each pool entry
///   (parallel to ``pool_*``).
/// * ``pool_int64s`` or ``pool_strings`` &mdash; exactly one must be
///   set; the type of the active pool must match the input dtype.
/// * ``weights`` &mdash; optional per-output weight; defaults to all
///   ones when ``IDF``/``TFIDF`` is selected and ``weights`` is
///   empty.
class TfIdfVectorizer : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Weighting criteria. Matches the ``mode`` attribute of the
  /// ``TfIdfVectorizer`` operator.
  enum class Mode { kTF, kIDF, kTFIDF };

  /// Parses the string form of the ``mode`` attribute (``"TF"``,
  /// ``"IDF"`` or ``"TFIDF"``). Throws ``std::invalid_argument`` for
  /// any other value.
  static Mode ParseMode(const std::string &value);

  /// Computes the output shape of a TfIdfVectorizer node given the
  /// input shape and the size of the ``ngram_indexes`` attribute.
  /// Input rank must be 1 or 2.
  static std::vector<int64_t> ComputeOutputShape(const std::vector<int64_t> &input_shape,
                                                 int64_t output_size);

  /// Allocating overload. ``pool_int64s`` must be non-empty when ``x``
  /// is an integer tensor; ``pool_strings`` must be non-empty when
  /// ``x`` is a string tensor. ``weights`` may be empty (treated as
  /// all-ones for ``IDF`` / ``TFIDF``).
  Tensor operator()(const Tensor &x, Mode mode, int64_t min_gram_length, int64_t max_gram_length,
                    int64_t max_skip_count, const std::vector<int64_t> &ngram_counts,
                    const std::vector<int64_t> &ngram_indexes,
                    const std::vector<int64_t> &pool_int64s,
                    const std::vector<std::string> &pool_strings,
                    const std::vector<float> &weights) const;

  /// In-place overload. ``output`` must already be a ``tensor(float)``
  /// with the shape returned by :cpp:func:`ComputeOutputShape` and a
  /// pre-sized ``data`` buffer.
  void operator()(const Tensor &x, Mode mode, int64_t min_gram_length, int64_t max_gram_length,
                  int64_t max_skip_count, const std::vector<int64_t> &ngram_counts,
                  const std::vector<int64_t> &ngram_indexes,
                  const std::vector<int64_t> &pool_int64s,
                  const std::vector<std::string> &pool_strings, const std::vector<float> &weights,
                  Tensor &output) const;

  /// Output values depend on every input element and on the entire
  /// pool; the output buffer cannot safely alias the input buffer
  /// (different dtype / shape).
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
