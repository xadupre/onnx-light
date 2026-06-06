// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kEinsumName = "kernel::Einsum";
// Synthetic labels used to expand ``...`` live outside the printable ASCII
// range so they cannot collide with user-supplied (letter) labels.
constexpr int kEllipsisLabelBase = 1;

// Removes ASCII space characters from ``equation`` in place.
std::string StripSpaces(const std::string &equation) {
  std::string out;
  out.reserve(equation.size());
  for (char c : equation) {
    if (c != ' ') {
      out.push_back(c);
    }
  }
  return out;
}

// Splits ``equation`` into a list of input terms and an optional output term.
// Each term is the raw substring as it appears in the equation; the ellipsis
// ``...`` is preserved verbatim. ``has_explicit_output`` reports whether the
// equation carried a ``->``.
void SplitEquation(const std::string &equation, std::vector<std::string> &input_terms,
                   std::string &output_term, bool &has_explicit_output) {
  std::string lhs;
  std::string rhs;
  has_explicit_output = false;
  const std::size_t arrow = equation.find("->");
  if (arrow == std::string::npos) {
    lhs = equation;
  } else {
    has_explicit_output = true;
    lhs = equation.substr(0, arrow);
    rhs = equation.substr(arrow + 2);
  }
  input_terms.clear();
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = lhs.find(',', start);
    if (comma == std::string::npos) {
      input_terms.push_back(lhs.substr(start));
      break;
    }
    input_terms.push_back(lhs.substr(start, comma - start));
    start = comma + 1;
  }
  output_term = std::move(rhs);
}

// Replaces the ``...`` ellipsis (if any) in ``term`` with a sequence of
// internal label characters drawn from the broadcast block of size
// ``ellipsis_rank``. Internal labels live outside the ASCII letter range so
// they cannot collide with user-supplied labels (lower- and upper-case
// letters per the ONNX spec).
std::string ExpandEllipsis(const std::string &term, std::size_t ellipsis_rank,
                           const std::vector<char> &ellipsis_labels) {
  const std::size_t dots = term.find("...");
  if (dots == std::string::npos) {
    return term;
  }
  std::string out;
  out.reserve(term.size() - 3 + ellipsis_rank);
  out.append(term, 0, dots);
  out.append(ellipsis_labels.begin(), ellipsis_labels.end());
  out.append(term, dots + 3, std::string::npos);
  return out;
}

struct EinsumPlan {
  // Expanded per-input label strings (with ellipsis replaced by labels).
  std::vector<std::string> input_labels;
  // Expanded output label string.
  std::string output_labels;
  // All distinct labels in iteration order: output labels first, then summed
  // labels in the order they are first encountered.
  std::vector<char> all_labels;
  // Size of each label in ``all_labels``. Same length as ``all_labels``.
  std::vector<int64_t> label_size;
  // For each input: precomputed strides so that the contribution of label
  // ``all_labels[k]`` (with current index ``ix[k]``) to the input's flat
  // offset is ``ix[k] * stride[k]`` (zero when the input does not use the
  // label).
  std::vector<std::vector<int64_t>> input_strides;
  // Output shape (in label order, i.e. ``output_labels``).
  std::vector<int64_t> output_shape;
  // Output strides indexed by ``all_labels``. Zero for summed labels.
  std::vector<int64_t> output_stride_per_label;
};

EinsumPlan BuildPlan(const std::vector<Tensor> &inputs, const std::string &raw_equation) {
  std::string equation = StripSpaces(raw_equation);
  if (equation.empty()) {
    throw std::invalid_argument(std::string(kEinsumName) + ": equation must not be empty.");
  }
  std::vector<std::string> input_terms;
  std::string output_term;
  bool has_explicit_output = false;
  SplitEquation(equation, input_terms, output_term, has_explicit_output);
  if (input_terms.size() != inputs.size()) {
    throw std::invalid_argument(
        std::string(kEinsumName) + ": number of input terms in the equation (" +
        std::to_string(input_terms.size()) + ") does not match number of inputs (" +
        std::to_string(inputs.size()) + ").");
  }

  // Determine the ellipsis rank (common across all inputs that use ``...``).
  std::size_t ellipsis_rank = 0;
  bool ellipsis_seen = false;
  for (std::size_t i = 0; i < input_terms.size(); ++i) {
    const std::string &term = input_terms[i];
    const std::size_t dots = term.find("...");
    if (dots == std::string::npos) {
      // No ellipsis in this term: the term length must match the input rank.
      if (term.size() != inputs[i].shape.size()) {
        throw std::invalid_argument(std::string(kEinsumName) + ": term '" + term + "' has " +
                                    std::to_string(term.size()) + " labels but input " +
                                    std::to_string(i) + " has rank " +
                                    std::to_string(inputs[i].shape.size()) + ".");
      }
      continue;
    }
    // The term has an ellipsis: it accounts for ``rank - (term.size() - 3)``
    // dimensions.
    if (term.size() - 3 > inputs[i].shape.size()) {
      throw std::invalid_argument(std::string(kEinsumName) + ": term '" + term +
                                  "' has more named labels than input " + std::to_string(i) +
                                  " has dimensions.");
    }
    const std::size_t this_rank = inputs[i].shape.size() - (term.size() - 3);
    if (!ellipsis_seen) {
      ellipsis_seen = true;
      ellipsis_rank = this_rank;
    } else if (this_rank != ellipsis_rank) {
      throw std::invalid_argument(std::string(kEinsumName) +
                                  ": ellipsis dimensions must be consistent across inputs, got " +
                                  std::to_string(this_rank) + " and " +
                                  std::to_string(ellipsis_rank) + ".");
    }
  }

  // Build the synthetic labels used to expand ``...``. They live outside the
  // ASCII letter range so they cannot collide with user-supplied labels.
  std::vector<char> ellipsis_labels;
  ellipsis_labels.reserve(ellipsis_rank);
  for (std::size_t i = 0; i < ellipsis_rank; ++i) {
    ellipsis_labels.push_back(static_cast<char>(kEllipsisLabelBase + static_cast<int>(i)));
  }

  EinsumPlan plan;
  plan.input_labels.reserve(input_terms.size());
  for (const std::string &term : input_terms) {
    plan.input_labels.push_back(ExpandEllipsis(term, ellipsis_rank, ellipsis_labels));
  }

  // Determine the size of every label and validate consistency across inputs.
  std::unordered_map<char, int64_t> label_size_map;
  for (std::size_t i = 0; i < plan.input_labels.size(); ++i) {
    const std::string &labels = plan.input_labels[i];
    const std::vector<int64_t> &shape = inputs[i].shape;
    if (labels.size() != shape.size()) {
      throw std::invalid_argument(std::string(kEinsumName) + ": expanded term '" + labels +
                                  "' has " + std::to_string(labels.size()) + " labels but input " +
                                  std::to_string(i) + " has rank " + std::to_string(shape.size()) +
                                  ".");
    }
    for (std::size_t d = 0; d < labels.size(); ++d) {
      const char lbl = labels[d];
      const int64_t dim = shape[d];
      auto it = label_size_map.find(lbl);
      if (it == label_size_map.end()) {
        label_size_map[lbl] = dim;
      } else if (it->second != dim) {
        // Allow broadcasting between 1 and N for the synthetic ellipsis
        // labels only; named labels must match exactly.
        const bool is_ellipsis =
            lbl >= kEllipsisLabelBase &&
            lbl <= static_cast<char>(kEllipsisLabelBase + static_cast<int>(ellipsis_rank) - 1);
        if (is_ellipsis && (it->second == 1 || dim == 1)) {
          it->second = std::max(it->second, dim);
        } else {
          throw std::invalid_argument(std::string(kEinsumName) + ": label '" + std::string(1, lbl) +
                                      "' has inconsistent sizes (" + std::to_string(it->second) +
                                      " and " + std::to_string(dim) + ") across inputs.");
        }
      }
    }
  }

  // Determine the output labels.
  std::string expanded_output;
  if (has_explicit_output) {
    expanded_output = ExpandEllipsis(output_term, ellipsis_rank, ellipsis_labels);
  } else {
    // Implicit mode: ellipsis dimensions first, then labels that appear
    // exactly once across all input terms, in ascending ASCII order.
    expanded_output.append(ellipsis_labels.begin(), ellipsis_labels.end());
    std::unordered_map<char, int> counts;
    for (const std::string &lbls : plan.input_labels) {
      for (char c : lbls) {
        // Skip ellipsis labels: they are handled above.
        if (c >= kEllipsisLabelBase &&
            c <= static_cast<char>(kEllipsisLabelBase + static_cast<int>(ellipsis_rank) - 1)) {
          continue;
        }
        counts[c] += 1;
      }
    }
    std::vector<char> singletons;
    for (const auto &[c, n] : counts) {
      if (n == 1) {
        singletons.push_back(c);
      }
    }
    std::sort(singletons.begin(), singletons.end());
    expanded_output.append(singletons.begin(), singletons.end());
  }
  plan.output_labels = expanded_output;

  // Validate: every label in the output must appear in some input.
  for (char c : plan.output_labels) {
    if (label_size_map.find(c) == label_size_map.end()) {
      throw std::invalid_argument(std::string(kEinsumName) + ": output label '" +
                                  std::string(1, c) + "' does not appear in any input term.");
    }
  }

  // Build the canonical iteration order: output labels first, then summed
  // labels (any input label not in the output) in first-occurrence order.
  std::set<char> seen;
  for (char c : plan.output_labels) {
    if (seen.insert(c).second) {
      plan.all_labels.push_back(c);
    }
  }
  for (const std::string &lbls : plan.input_labels) {
    for (char c : lbls) {
      if (seen.insert(c).second) {
        plan.all_labels.push_back(c);
      }
    }
  }
  plan.label_size.reserve(plan.all_labels.size());
  for (char c : plan.all_labels) {
    plan.label_size.push_back(label_size_map[c]);
  }

  // Build per-input strides keyed by label position.
  plan.input_strides.assign(plan.input_labels.size(),
                            std::vector<int64_t>(plan.all_labels.size(), 0));
  for (std::size_t i = 0; i < plan.input_labels.size(); ++i) {
    const std::string &labels = plan.input_labels[i];
    const std::vector<int64_t> &shape = inputs[i].shape;
    // Row-major strides for the input itself.
    std::vector<int64_t> in_stride(shape.size(), 1);
    for (std::size_t d = shape.size(); d-- > 0;) {
      if (d + 1 < shape.size()) {
        in_stride[d] = in_stride[d + 1] * shape[d + 1];
      }
    }
    for (std::size_t k = 0; k < plan.all_labels.size(); ++k) {
      const char lbl = plan.all_labels[k];
      int64_t accum = 0;
      bool any = false;
      for (std::size_t d = 0; d < labels.size(); ++d) {
        if (labels[d] == lbl) {
          // Repeated labels in the same input add together; a label with
          // size 1 contributes a zero stride (broadcast).
          if (shape[d] == 1 && plan.label_size[k] > 1) {
            // broadcast: zero contribution
            any = true;
          } else {
            accum += in_stride[d];
            any = true;
          }
        }
      }
      if (any) {
        plan.input_strides[i][k] = accum;
      }
    }
  }

  // Output shape & per-label strides.
  plan.output_shape.reserve(plan.output_labels.size());
  for (char c : plan.output_labels) {
    plan.output_shape.push_back(label_size_map[c]);
  }
  std::vector<int64_t> out_stride(plan.output_shape.size(), 1);
  for (std::size_t d = plan.output_shape.size(); d-- > 0;) {
    if (d + 1 < plan.output_shape.size()) {
      out_stride[d] = out_stride[d + 1] * plan.output_shape[d + 1];
    }
  }
  plan.output_stride_per_label.assign(plan.all_labels.size(), 0);
  for (std::size_t d = 0; d < plan.output_labels.size(); ++d) {
    const char lbl = plan.output_labels[d];
    for (std::size_t k = 0; k < plan.all_labels.size(); ++k) {
      if (plan.all_labels[k] == lbl) {
        plan.output_stride_per_label[k] = out_stride[d];
        break;
      }
    }
  }

  return plan;
}

template <typename T>
void RunEinsum(const std::vector<Tensor> &inputs, const EinsumPlan &plan, T *out_data) {
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  for (int64_t i = 0; i < out_count; ++i) {
    out_data[i] = T{0};
  }

  // Iterate over the Cartesian product of ``label_size``. ``ix[k]`` is the
  // current value of label ``all_labels[k]``.
  std::vector<int64_t> ix(plan.all_labels.size(), 0);
  int64_t total = 1;
  for (int64_t s : plan.label_size) {
    total *= s;
  }
  if (total == 0) {
    return; // Output is zero-sized along some dimension.
  }

  std::vector<const T *> in_ptrs;
  in_ptrs.reserve(inputs.size());
  for (const Tensor &t : inputs) {
    in_ptrs.push_back(t.As<T>());
  }

  for (int64_t step = 0; step < total; ++step) {
    // Compute the product of input values at the current label indices.
    T prod{1};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      int64_t off = 0;
      for (std::size_t k = 0; k < plan.all_labels.size(); ++k) {
        off += ix[k] * plan.input_strides[i][k];
      }
      prod *= in_ptrs[i][off];
    }
    // Accumulate into the output position.
    int64_t out_off = 0;
    for (std::size_t k = 0; k < plan.all_labels.size(); ++k) {
      out_off += ix[k] * plan.output_stride_per_label[k];
    }
    out_data[out_off] += prod;

    // Advance ``ix`` (row-major).
    for (std::size_t k = plan.all_labels.size(); k-- > 0;) {
      ix[k] += 1;
      if (ix[k] < plan.label_size[k]) {
        break;
      }
      ix[k] = 0;
    }
  }
}

template <typename T>
Tensor EinsumAlloc(const std::vector<Tensor> &inputs, const std::string &equation, int32_t dtype) {
  const EinsumPlan plan = BuildPlan(inputs, equation);
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  Tensor z("", dtype, plan.output_shape,
           std::vector<uint8_t>(static_cast<std::size_t>(out_count) * sizeof(T)));
  RunEinsum<T>(inputs, plan, z.As<T>());
  return z;
}

template <typename T>
void EinsumInPlace(const std::vector<Tensor> &inputs, const std::string &equation, int32_t dtype,
                   Tensor &output) {
  const EinsumPlan plan = BuildPlan(inputs, equation);
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  if (output.data_type != dtype) {
    throw std::invalid_argument(std::string(kEinsumName) + ": output dtype mismatch.");
  }
  if (output.shape != plan.output_shape) {
    throw std::invalid_argument(std::string(kEinsumName) + ": output shape mismatch.");
  }
  if (output.data.size() != static_cast<std::size_t>(out_count) * sizeof(T)) {
    throw std::invalid_argument(std::string(kEinsumName) + ": output buffer size mismatch.");
  }
  RunEinsum<T>(inputs, plan, output.As<T>());
}

void RequireHomogeneous(const std::vector<Tensor> &inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument(std::string(kEinsumName) + " requires at least one input.");
  }
  const int32_t dtype = inputs[0].data_type;
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].data_type != dtype) {
      throw std::invalid_argument(std::string(kEinsumName) +
                                  ": all inputs must share the same dtype.");
    }
  }
}

} // namespace

Tensor Einsum::operator()(const std::vector<Tensor> &inputs, const std::string &equation) const {
  RequireHomogeneous(inputs);
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return EinsumAlloc<float>(inputs, equation, DataType::FLOAT);
  case DataType::DOUBLE:
    return EinsumAlloc<double>(inputs, equation, DataType::DOUBLE);
  default:
    throw std::invalid_argument(std::string(kEinsumName) +
                                " only supports FLOAT and DOUBLE inputs.");
  }
}

void Einsum::operator()(const std::vector<Tensor> &inputs, const std::string &equation,
                        Tensor &output) const {
  RequireHomogeneous(inputs);
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return EinsumInPlace<float>(inputs, equation, DataType::FLOAT, output);
  case DataType::DOUBLE:
    return EinsumInPlace<double>(inputs, equation, DataType::DOUBLE, output);
  default:
    throw std::invalid_argument(std::string(kEinsumName) +
                                " only supports FLOAT and DOUBLE inputs.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
