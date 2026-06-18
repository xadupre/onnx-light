// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/inplace_reuse.h"

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

std::vector<std::vector<InPlaceReuse>> ComputeInPlaceReuse(const GraphProto &graph,
                                                           const ShapesContext &ctx) {
  const int num_nodes = graph.node().size();
  std::vector<std::vector<InPlaceReuse>> result(static_cast<std::size_t>(num_nodes));

  // Names whose buffers must never be overwritten in place: declared graph
  // inputs, initializers and declared graph outputs are owned by the caller
  // or must outlive the run.
  std::unordered_set<std::string> keep;
  for (int i = 0; i < graph.input().size(); ++i) {
    keep.insert(graph.input()[i].name().as_string());
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    keep.insert(graph.initializer()[i].name().as_string());
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    keep.insert(graph.output()[i].name().as_string());
  }

  // Producer node index for every top-level intermediate, and the index of
  // the last node that references each name (directly or via a subgraph).
  std::unordered_map<std::string, int> producer;
  std::unordered_map<std::string, int> last_use;
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
    for (int o = 0; o < node.output_size(); ++o) {
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
        if (!ctx.Has(in_name) || !SameStorage(out_tensor, ctx.Get(in_name))) {
          continue;
        }
        InPlaceReuse reuse;
        reuse.output_index = o;
        reuse.input_index = k;
        result[static_cast<std::size_t>(i)].push_back(reuse);
        used_inputs.insert(k);
        break;
      }
    }
  }

  return result;
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
