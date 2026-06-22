// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/inplace_reuse.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Returns whether two descriptors are byte-for-byte interchangeable: same
// element type and identical shape (rank and every dimension, concrete or
// symbolic). Equal descriptors imply equal element counts and therefore
// equal buffer sizes, which is the precondition for an in-place reuse.
bool SameStorage(const OptimTensor &a, const OptimTensor &b) {
  if (a.Dtype() != b.Dtype()) {
    return false;
  }
  const OptimShape &sa = a.Shape();
  const OptimShape &sb = b.Shape();
  if (sa.Rank() != sb.Rank()) {
    return false;
  }
  for (std::size_t i = 0; i < sa.Rank(); ++i) {
    if (sa[i] != sb[i]) {
      return false;
    }
  }
  return true;
}

// Number of bits used by a single element of ``t``. Returns ``0`` for element
// types whose byte size is not a fixed scalar (strings, sequences, maps,
// optionals and the undefined type), for which no buffer-size comparison is
// attempted.
int ElementBitWidth(TensorType t) {
  switch (t) {
  case TensorType::kBool:
  case TensorType::kUint8:
  case TensorType::kInt8:
  case TensorType::kFloat8e4m3fn:
  case TensorType::kFloat8e4m3fnuz:
  case TensorType::kFloat8e5m2:
  case TensorType::kFloat8e5m2fnuz:
  case TensorType::kFloat8e8m0:
    return 8;
  case TensorType::kUint16:
  case TensorType::kInt16:
  case TensorType::kFloat16:
  case TensorType::kBfloat16:
    return 16;
  case TensorType::kUint32:
  case TensorType::kInt32:
  case TensorType::kFloat:
    return 32;
  case TensorType::kUint64:
  case TensorType::kInt64:
  case TensorType::kDouble:
  case TensorType::kComplex64:
    return 64;
  case TensorType::kComplex128:
    return 128;
  case TensorType::kFloat4e2m1:
  case TensorType::kUint4:
  case TensorType::kInt4:
    return 4;
  case TensorType::kUint2:
  case TensorType::kInt2:
    return 2;
  default:
    return 0;
  }
}

// Packed byte size of ``t`` when its shape is fully known and its element type
// has a fixed bit width, otherwise ``std::nullopt``. Sub-byte element types are
// packed, matching the ONNX storage convention.
std::optional<int64_t> ByteSize(const OptimTensor &t) {
  const int bits = ElementBitWidth(t.Dtype());
  if (bits == 0) {
    return std::nullopt;
  }
  const OptimShape &shape = t.Shape();
  if (!shape.IsFullyKnown()) {
    return std::nullopt;
  }
  const int64_t num_elements = shape.NumElements();
  return (num_elements * bits + 7) / 8;
}

// Classifies a candidate reuse of input ``in`` by output ``out`` by comparing
// their buffer sizes. ``kEqual`` requires identical descriptors (same element
// type and shape), so an element-wise overwrite keeps the layout valid even
// for symbolic shapes. ``kGreater`` reports an input buffer that is strictly
// larger in bytes than the output, so the output still fits. Any other case
// (input smaller, or equal byte size but a different shape such as a
// transpose) yields no opportunity.
std::optional<InPlaceReuseKind> ClassifyReuse(const OptimTensor &out, const OptimTensor &in) {
  if (SameStorage(out, in)) {
    return InPlaceReuseKind::kEqual;
  }
  const std::optional<int64_t> out_bytes = ByteSize(out);
  const std::optional<int64_t> in_bytes = ByteSize(in);
  if (!out_bytes.has_value() || !in_bytes.has_value()) {
    return std::nullopt;
  }
  if (*in_bytes > *out_bytes) {
    return InPlaceReuseKind::kGreater;
  }
  return std::nullopt;
}

// Recursively collects every input name referenced by ``node`` — its direct
// inputs plus any name read inside the bodies of its subgraph attributes
// (``If``, ``Loop``, ``Scan``, ...). Subgraph-local names are included too;
// this only over-approximates the set of live names, which keeps the reuse
// guess conservative.
void CollectReferencedNames(const NodeProto &node, std::unordered_set<std::string> &out) {
  for (int i = 0; i < node.input_size(); ++i) {
    const std::string name = node.input(i).as_string();
    if (!name.empty()) {
      out.insert(name);
    }
  }
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.has_g()) {
      for (int j = 0; j < attr.g().node().size(); ++j) {
        CollectReferencedNames(attr.g().node()[j], out);
      }
    }
    for (int g = 0; g < attr.graphs().size(); ++g) {
      const GraphProto &subgraph = attr.graphs()[g];
      for (int j = 0; j < subgraph.node().size(); ++j) {
        CollectReferencedNames(subgraph.node()[j], out);
      }
    }
  }
}

} // namespace

void InplaceContext::ComputeInPlaceReuseGraph(const GraphProto &graph, const ShapesContext &ctx,
                                              bool allow_input_overwrite) {
  const int num_nodes = graph.node().size();
  std::vector<std::vector<InPlaceReuse>> result(static_cast<std::size_t>(num_nodes));

  // Names whose buffers must never be overwritten in place: declared graph
  // inputs, initializers and declared graph outputs are owned by the caller
  // or must outlive the run. Declared graph inputs are protected unless
  // ``allow_input_overwrite`` explicitly opts into reusing them, in which case
  // only initializers and outputs stay protected.
  std::unordered_set<std::string> keep;
  std::unordered_set<std::string> graph_inputs;
  for (int i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name().as_string();
    graph_inputs.insert(name);
    if (!allow_input_overwrite) {
      keep.insert(name);
    }
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    keep.insert(graph.initializer()[i].name().as_string());
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    keep.insert(graph.output()[i].name().as_string());
  }

  // Producer node index for every top-level intermediate, and the index of
  // the last node that references each name (directly or via a subgraph).
  // When input overwrite is allowed, declared graph inputs are treated as
  // available before the first node (producer index ``-1``) so they can be
  // reused once they reach their last use.
  std::unordered_map<std::string, int> producer;
  std::unordered_map<std::string, int> last_use;
  if (allow_input_overwrite) {
    for (const std::string &name : graph_inputs) {
      if (!name.empty()) {
        producer[name] = -1;
      }
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    const NodeProto &node = graph.node()[i];
    std::unordered_set<std::string> referenced;
    CollectReferencedNames(node, referenced);
    for (const std::string &name : referenced) {
      last_use[name] = i;
    }
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string name = node.output(o).as_string();
      if (!name.empty() && producer.find(name) == producer.end()) {
        producer[name] = i;
      }
    }
  }

  for (int i = 0; i < num_nodes; ++i) {
    const NodeProto &node = graph.node()[i];

    // Count direct-input occurrences so a value read twice is never aliased.
    std::unordered_map<std::string, int> input_occurrences;
    for (int k = 0; k < node.input_size(); ++k) {
      const std::string name = node.input(k).as_string();
      if (!name.empty()) {
        ++input_occurrences[name];
      }
    }

    std::unordered_set<int> used_inputs;
    std::unordered_set<int> matched_outputs;
    // Two passes so that same-sized (kEqual) reuse is always preferred over a
    // strictly larger (kGreater) input before either buffer is claimed.
    for (const InPlaceReuseKind kind : {InPlaceReuseKind::kEqual, InPlaceReuseKind::kGreater}) {
      for (int o = 0; o < node.output_size(); ++o) {
        if (matched_outputs.count(o)) {
          continue;
        }
        const std::string out_name = node.output(o).as_string();
        if (out_name.empty() || !ctx.Has(out_name)) {
          continue;
        }
        const OptimTensor &out_tensor = ctx.Get(out_name);

        for (int k = 0; k < node.input_size(); ++k) {
          if (used_inputs.count(k)) {
            continue;
          }
          const std::string in_name = node.input(k).as_string();
          if (in_name.empty() || in_name == out_name) {
            continue;
          }
          if (input_occurrences[in_name] != 1) {
            continue;
          }
          if (keep.count(in_name)) {
            continue;
          }
          auto prod_it = producer.find(in_name);
          if (prod_it == producer.end() || prod_it->second >= i) {
            continue;
          }
          auto use_it = last_use.find(in_name);
          if (use_it == last_use.end() || use_it->second != i) {
            continue;
          }
          if (!ctx.Has(in_name)) {
            continue;
          }
          const std::optional<InPlaceReuseKind> match = ClassifyReuse(out_tensor, ctx.Get(in_name));
          if (!match.has_value() || *match != kind) {
            continue;
          }
          InPlaceReuse reuse;
          reuse.output_index = o;
          reuse.input_index = k;
          reuse.kind = kind;
          result[static_cast<std::size_t>(i)].push_back(reuse);
          used_inputs.insert(k);
          matched_outputs.insert(o);
          break;
        }
      }
    }
    // Keep each node's opportunities ordered by output index regardless of the
    // pass in which they were discovered.
    std::sort(result[static_cast<std::size_t>(i)].begin(),
              result[static_cast<std::size_t>(i)].end(),
              [](const InPlaceReuse &a, const InPlaceReuse &b) {
                return a.output_index < b.output_index;
              });
  }

  reuse_ = std::move(result);
}

void InplaceContext::WriteToMetadata(GraphProto &graph) const {
  if (static_cast<std::size_t>(graph.node().size()) != reuse_.size()) {
    throw std::invalid_argument(
        "InplaceContext::WriteToMetadata: graph node count does not match the "
        "computed reuse result; call ComputeInPlaceReuseGraph on the same graph first.");
  }
  for (std::size_t i = 0; i < reuse_.size(); ++i) {
    if (reuse_[i].empty()) {
      continue;
    }
    std::ostringstream value;
    for (std::size_t j = 0; j < reuse_[i].size(); ++j) {
      const InPlaceReuse &r = reuse_[i][j];
      if (j != 0) {
        value << ";";
      }
      value << r.output_index << ":" << r.input_index << ":"
            << (r.kind == InPlaceReuseKind::kEqual ? "equal" : "greater");
    }
    NodeProto &node = (*graph.mutable_node())[i];
    node.add_metadata(kInPlaceReuseMetadataKey, value.str());
  }
}

std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuse(const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite) {
  InplaceContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite);
  return inplace.Reuse();
}

void WriteInPlaceReuseToMetadata(GraphProto &graph, const ShapesContext &ctx) {
  InplaceContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx);
  inplace.WriteToMetadata(graph);
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
