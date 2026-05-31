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

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
