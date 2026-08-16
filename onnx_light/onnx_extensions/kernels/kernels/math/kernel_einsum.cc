// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_light_helpers.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

// Precomputed contraction plan (forward-declared in the kernel header so
// :cpp:class:`Einsum` can cache it between calls).
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
  Shape output_shape;
  // Output strides indexed by ``all_labels``. Zero for summed labels.
  std::vector<int64_t> output_stride_per_label;
};

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

EinsumPlan BuildPlan(const Tensors &inputs, const std::string &raw_equation) {
  std::string equation = StripSpaces(raw_equation);
  EXT_ENFORCE_INVALID(!equation.empty(), kEinsumName, ": equation must not be empty.");
  std::vector<std::string> input_terms;
  std::string output_term;
  bool has_explicit_output = false;
  SplitEquation(equation, input_terms, output_term, has_explicit_output);
  EXT_ENFORCE_INVALID(input_terms.size() == inputs.size(), kEinsumName,
                      ": number of input terms in the equation (", input_terms.size(),
                      ") does not match number of inputs (", inputs.size(), ").");

  // Determine the ellipsis rank (common across all inputs that use ``...``).
  std::size_t ellipsis_rank = 0;
  bool ellipsis_seen = false;
  for (std::size_t i = 0; i < input_terms.size(); ++i) {
    const std::string &term = input_terms[i];
    const std::size_t dots = term.find("...");
    if (dots == std::string::npos) {
      // No ellipsis in this term: the term length must match the input rank.
      EXT_ENFORCE_INVALID(term.size() == inputs[i].shape.size(), kEinsumName, ": term '", term,
                          "' has ", term.size(), " labels but input ", i, " has rank ",
                          inputs[i].shape.size(), ".");
      continue;
    }
    // The term has an ellipsis: it accounts for ``rank - (term.size() - 3)``
    // dimensions.
    EXT_ENFORCE_INVALID(term.size() - 3 <= inputs[i].shape.size(), kEinsumName, ": term '", term,
                        "' has more named labels than input ", i, " has dimensions.");
    const std::size_t this_rank = inputs[i].shape.size() - (term.size() - 3);
    if (!ellipsis_seen) {
      ellipsis_seen = true;
      ellipsis_rank = this_rank;
    } else if (this_rank != ellipsis_rank) {
      EXT_THROW_INVALID(kEinsumName, ": ellipsis dimensions must be consistent across inputs, got ",
                        this_rank, " and ", ellipsis_rank, ".");
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
    const Shape &shape = inputs[i].shape;
    EXT_ENFORCE_INVALID(labels.size() == shape.size(), kEinsumName, ": expanded term '", labels,
                        "' has ", labels.size(), " labels but input ", i, " has rank ",
                        shape.size(), ".");
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
          EXT_THROW_INVALID(kEinsumName, ": label '", lbl, "' has inconsistent sizes (", it->second,
                            " and ", dim, ") across inputs.");
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
    EXT_ENFORCE_INVALID(label_size_map.find(c) != label_size_map.end(), kEinsumName,
                        ": output label '", c, "' does not appear in any input term.");
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
    const Shape &shape = inputs[i].shape;
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
  Shape out_stride;
  out_stride.assign(plan.output_shape.size(), 1);
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
void RunEinsum(const Tensors &inputs, const EinsumPlan &plan, T *out_data,
               RawBufferAllocator *allocator) {
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  for (int64_t i = 0; i < out_count; ++i) {
    out_data[i] = T{0};
  }

  // Iterate over the Cartesian product of ``label_size``. ``ix[k]`` is the
  // current value of label ``all_labels[k]``. The scratch buffer is drawn from
  // the runtime allocator when one is attached, falling back to inline
  // ``std::vector`` storage otherwise. Allocator-backed buffers are not
  // guaranteed zeroed, so the indices are explicitly zeroed below. A minimum
  // size of one element is requested when ``n_labels`` is 0 (scalar equations
  // such as ``->``) because the allocator rejects a zero-byte request; the
  // buffer is never dereferenced in that case.
  const std::size_t n_labels = plan.all_labels.size();
  detail::TemporaryTypedBuffer<int64_t> ix_buf(n_labels > 0 ? n_labels : 1, allocator,
                                               "kernel::Einsum ix");
  int64_t *ix = ix_buf.data();
  std::fill(ix, ix + n_labels, int64_t{0});
  int64_t total = 1;
  for (int64_t s : plan.label_size) {
    total *= s;
  }
  if (total == 0) {
    return; // Output is zero-sized along some dimension.
  }

  detail::TemporaryTypedBuffer<const T *> in_ptrs_buf(inputs.size(), allocator,
                                                      "kernel::Einsum in_ptrs");
  const T **in_ptrs = in_ptrs_buf.data();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    in_ptrs[i] = inputs[i].As<T>();
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
Tensor EinsumAlloc(const Tensors &inputs, const EinsumPlan &plan, int32_t dtype,
                   RuntimeContext *rt) {
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  const size_t z_n_bytes = static_cast<std::size_t>(out_count) * sizeof(T);
  Tensor z = MakeOutputTensor(dtype, plan.output_shape, z_n_bytes, rt ? rt->allocator() : nullptr);
  RunEinsum<T>(inputs, plan, z.As<T>(), rt ? rt->allocator() : nullptr);
  return z;
}

template <typename T>
void EinsumInPlace(const Tensors &inputs, const EinsumPlan &plan, int32_t dtype, Tensor &output) {
  int64_t out_count = 1;
  for (int64_t d : plan.output_shape) {
    out_count *= d;
  }
  EXT_ENFORCE_INVALID(output.data_type == dtype, kEinsumName, ": output dtype mismatch.");
  EXT_ENFORCE_INVALID(output.shape == plan.output_shape, kEinsumName, ": output shape mismatch.");
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<std::size_t>(out_count) * sizeof(T),
                      kEinsumName, ": output buffer size mismatch.");
  RunEinsum<T>(inputs, plan, output.As<T>(), nullptr);
}

void RequireHomogeneous(const Tensors &inputs) {
  EXT_ENFORCE_INVALID(!inputs.empty(), kEinsumName, " requires at least one input.");
  const int32_t dtype = inputs[0].data_type;
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == dtype, kEinsumName,
                        ": all inputs must share the same dtype.");
  }
}

} // namespace

const EinsumPlan &Einsum::EnsurePlan(const Tensors &inputs, const std::string &equation) const {
  std::vector<Shape> shapes;
  shapes.reserve(inputs.size());
  for (const Tensor &t : inputs) {
    shapes.push_back(t.shape);
  }
  if (plan_ == nullptr || plan_equation_ != equation || plan_shapes_ != shapes) {
    plan_ = std::make_shared<const EinsumPlan>(BuildPlan(inputs, equation));
    plan_equation_ = equation;
    plan_shapes_ = std::move(shapes);
  }
  return *plan_;
}

Tensor Einsum::operator()(const Tensors &inputs, const std::string &equation,
                          RuntimeContext *rt) const {
  RequireHomogeneous(inputs);
  const EinsumPlan &plan = EnsurePlan(inputs, equation);
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return EinsumAlloc<float>(inputs, plan, DataType::FLOAT, rt);
  case DataType::DOUBLE:
    return EinsumAlloc<double>(inputs, plan, DataType::DOUBLE, rt);
  default:
    EXT_THROW_INVALID(kEinsumName, ": unsupported data type ", inputs[0].data_type,
                      ", only supports FLOAT and DOUBLE inputs.");
  }
}

void Einsum::operator()(const Tensors &inputs, const std::string &equation, Tensor &output) const {
  RequireHomogeneous(inputs);
  const EinsumPlan &plan = EnsurePlan(inputs, equation);
  switch (inputs[0].data_type) {
  case DataType::FLOAT:
    return EinsumInPlace<float>(inputs, plan, DataType::FLOAT, output);
  case DataType::DOUBLE:
    return EinsumInPlace<double>(inputs, plan, DataType::DOUBLE, output);
  default:
    EXT_THROW_INVALID(kEinsumName, ": unsupported data type ", inputs[0].data_type,
                      ", only supports FLOAT and DOUBLE inputs.");
  }
}

void Einsum::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  RequireOutputCount(node, 1);
  Tensors inputs;
  inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    inputs.push_back(GetInput(node, i, rt.tensors()));
  }
  const std::string equation = GetRequiredAttributeString(node, "equation");
  onnx_kernels::kernel::Einsum k(rt.kernel_ctx());
  SetOutput(node, 0, k(inputs, equation, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
