// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

namespace {

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

// Merges two ``OptimDim`` instances representing the same label seen across
// inputs. Two concrete values must agree, except that broadcasting allows a
// dimension of 1 to be promoted to the other size. A concrete value wins
// over a symbolic one. Two symbolic values are merged only when they share
// the same expression; otherwise the existing entry is preserved.
OptimDim MergeLabelDim(const OptimDim &out, const OptimDim &in, char label) {
  if (out.IsInt() && in.IsInt()) {
    if (out.AsInt() == in.AsInt()) {
      return out;
    }
    if (out.AsInt() == 1) {
      return in;
    }
    if (in.AsInt() == 1) {
      return out;
    }
    throw std::invalid_argument("ComputeShapeEinsum: label '" + std::string(1, label) +
                                "' has inconsistent sizes (" + std::to_string(out.AsInt()) +
                                " and " + std::to_string(in.AsInt()) + ").");
  }
  if (in.IsInt()) {
    return in;
  }
  return out;
}

} // namespace

void ComputeShapeEinsum(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Einsum", "ComputeShapeEinsum");

  const int n_inputs = node.input_size();
  if (n_inputs < 1) {
    throw std::invalid_argument("ComputeShapeEinsum: Einsum requires at least one input.");
  }

  const std::string raw_equation = GetAttributeOr<std::string>(node, "equation", std::string());
  const std::string equation = StripSpaces(raw_equation);
  if (equation.empty()) {
    throw std::invalid_argument("ComputeShapeEinsum: 'equation' attribute is required.");
  }

  std::vector<std::string> input_terms;
  std::string output_term;
  bool has_explicit_output = false;
  SplitEquation(equation, input_terms, output_term, has_explicit_output);
  if (static_cast<int>(input_terms.size()) != n_inputs) {
    throw std::invalid_argument("ComputeShapeEinsum: number of input terms in the equation (" +
                                std::to_string(input_terms.size()) +
                                ") does not match number of inputs (" + std::to_string(n_inputs) +
                                ").");
  }

  // Collect input shapes and the dtype (propagated from the first input).
  std::vector<OptimShape> input_shapes;
  input_shapes.reserve(n_inputs);
  const OptimTensor &first = ctx.Get(node.input(0).as_string());
  const TensorType out_dtype = first.Dtype();
  input_shapes.push_back(first.Shape());
  for (int i = 1; i < n_inputs; ++i) {
    input_shapes.push_back(ctx.Get(node.input(i).as_string()).Shape());
  }

  // Determine ellipsis rank.
  std::size_t ellipsis_rank = 0;
  bool ellipsis_seen = false;
  for (int i = 0; i < n_inputs; ++i) {
    const std::string &term = input_terms[i];
    const std::size_t rank = input_shapes[i].Rank();
    const std::size_t dots = term.find("...");
    if (dots == std::string::npos) {
      if (term.size() != rank) {
        throw std::invalid_argument("ComputeShapeEinsum: term '" + term + "' has " +
                                    std::to_string(term.size()) + " labels but input " +
                                    std::to_string(i) + " has rank " + std::to_string(rank) + ".");
      }
      continue;
    }
    if (term.size() - 3 > rank) {
      throw std::invalid_argument("ComputeShapeEinsum: term '" + term +
                                  "' has more named labels than input " + std::to_string(i) +
                                  " has dimensions.");
    }
    const std::size_t this_rank = rank - (term.size() - 3);
    if (!ellipsis_seen) {
      ellipsis_seen = true;
      ellipsis_rank = this_rank;
    } else if (this_rank != ellipsis_rank) {
      throw std::invalid_argument(
          "ComputeShapeEinsum: ellipsis dimensions must be consistent across inputs, got " +
          std::to_string(this_rank) + " and " + std::to_string(ellipsis_rank) + ".");
    }
  }

  std::vector<char> ellipsis_labels;
  ellipsis_labels.reserve(ellipsis_rank);
  for (std::size_t i = 0; i < ellipsis_rank; ++i) {
    ellipsis_labels.push_back(static_cast<char>(1 + static_cast<int>(i)));
  }

  std::vector<std::string> input_labels;
  input_labels.reserve(n_inputs);
  for (const std::string &term : input_terms) {
    input_labels.push_back(ExpandEllipsis(term, ellipsis_rank, ellipsis_labels));
  }

  // Determine the size of every label and validate consistency.
  std::unordered_map<char, OptimDim> label_dim;
  for (int i = 0; i < n_inputs; ++i) {
    const std::string &labels = input_labels[i];
    const OptimShape &shape = input_shapes[i];
    if (labels.size() != shape.Rank()) {
      throw std::invalid_argument("ComputeShapeEinsum: expanded term '" + labels + "' has " +
                                  std::to_string(labels.size()) + " labels but input " +
                                  std::to_string(i) + " has rank " + std::to_string(shape.Rank()) +
                                  ".");
    }
    for (std::size_t d = 0; d < labels.size(); ++d) {
      const char lbl = labels[d];
      const OptimDim &dim = shape[d];
      auto it = label_dim.find(lbl);
      if (it == label_dim.end()) {
        label_dim.emplace(lbl, dim);
      } else {
        it->second = MergeLabelDim(it->second, dim, lbl);
      }
    }
  }

  // Determine the output labels.
  std::string expanded_output;
  if (has_explicit_output) {
    expanded_output = ExpandEllipsis(output_term, ellipsis_rank, ellipsis_labels);
  } else {
    expanded_output.append(ellipsis_labels.begin(), ellipsis_labels.end());
    std::unordered_map<char, int> counts;
    for (const std::string &lbls : input_labels) {
      for (char c : lbls) {
        if (c >= 1 && c <= static_cast<char>(ellipsis_rank)) {
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

  std::vector<OptimDim> out_dims;
  out_dims.reserve(expanded_output.size());
  for (char c : expanded_output) {
    auto it = label_dim.find(c);
    if (it == label_dim.end()) {
      throw std::invalid_argument("ComputeShapeEinsum: output label '" + std::string(1, c) +
                                  "' does not appear in any input term.");
    }
    out_dims.push_back(it->second);
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, OptimShape(out_dims)));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
