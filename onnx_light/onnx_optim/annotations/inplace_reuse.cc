// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/annotations/inplace_reuse.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "onnx_optim/shapes/_helpers/shape_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace annotations {

namespace {

enum class MemoryValueSource : uint8_t {
  kInput,
  kInitializer,
  kIntermediate,
};

struct LiveAllocation {
  expressions::DimType bytes = int64_t{0};
  MemoryValueSource source = MemoryValueSource::kIntermediate;
  ShapeTag tag;
};

ShapeTag ValueTag(const std::unordered_map<std::string, std::string> &value_tags,
                  const std::string &name) {
  auto it = value_tags.find(name);
  if (it == value_tags.end()) {
    return {};
  }
  return it->second;
}

bool IsZeroDim(const expressions::DimType &d) {
  return std::holds_alternative<int64_t>(d) && std::get<int64_t>(d) == 0;
}

expressions::DimType SimplifyDimType(const expressions::DimType &value) {
  if (std::holds_alternative<int64_t>(value)) {
    return value;
  }
  const expressions::SimplifyResult simplified =
      expressions::simplify_expression(std::get<std::string>(value));
  if (std::holds_alternative<int64_t>(simplified)) {
    return std::get<int64_t>(simplified);
  }
  return std::get<std::string>(simplified);
}

void SimplifyTaggedMemory(TaggedMemory &bucket) {
  for (auto &entry : bucket) {
    entry.second = SimplifyDimType(entry.second);
  }
}

void SimplifyNodeMemoryProfile(NodeMemoryProfile &profile) {
  NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) =
      SimplifyDimType(NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey));
  NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) =
      SimplifyDimType(NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey));
  NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) =
      SimplifyDimType(NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey));
  SimplifyTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey));
  SimplifyTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey));
  SimplifyTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey));
  SimplifyTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey));
}

void AddTaggedBytes(TaggedMemory &dst, const ShapeTag &tag, const expressions::DimType &bytes) {
  const expressions::DimType simplified_bytes = SimplifyDimType(bytes);
  if (IsZeroDim(simplified_bytes)) {
    return;
  }
  auto it = dst.find(tag);
  if (it == dst.end()) {
    dst.emplace(tag, simplified_bytes);
    return;
  }
  it->second = SimplifyDimType(expressions::dim_add(it->second, simplified_bytes));
}

void AddLiveAllocation(NodeMemoryProfile &profile, const LiveAllocation &alloc) {
  const expressions::DimType simplified_bytes = SimplifyDimType(alloc.bytes);
  expressions::DimType &already_allocated =
      NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey);
  already_allocated = SimplifyDimType(expressions::dim_add(already_allocated, simplified_bytes));
  switch (alloc.source) {
  case MemoryValueSource::kInput:
    AddTaggedBytes(NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey), alloc.tag,
                   simplified_bytes);
    break;
  case MemoryValueSource::kInitializer:
    AddTaggedBytes(NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey), alloc.tag,
                   simplified_bytes);
    break;
  case MemoryValueSource::kIntermediate:
    AddTaggedBytes(NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey), alloc.tag,
                   simplified_bytes);
    break;
  }
}

NodeMemoryProfile MakeEmptyNodeMemoryProfile() {
  NodeMemoryProfile profile;
  NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) = int64_t{0};
  NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = int64_t{0};
  NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) = int64_t{0};
  NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey);
  return profile;
}

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
std::optional<int64_t> ConcreteByteSize(const OptimTensor &t) {
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

std::optional<expressions::DimType> ByteSizeExpr(const OptimTensor &t) {
  const int bits = ElementBitWidth(t.Dtype());
  if (bits == 0) {
    return std::nullopt;
  }
  expressions::DimType num_elements = int64_t{1};
  for (std::size_t i = 0; i < t.Shape().Rank(); ++i) {
    num_elements = expressions::dim_mul(num_elements, shapes::ToDimType(t.Shape()[i]));
  }
  if (bits % 8 == 0) {
    return SimplifyDimType(
        expressions::dim_mul(num_elements, expressions::DimType{int64_t{bits / 8}}));
  }
  // Sub-byte element types pack multiple values per byte, so the buffer size is
  // ceil(num_elements * bits / 8). ``dim_div`` is floor division, hence the
  // ``+7`` round-up term before dividing by 8.
  return SimplifyDimType(expressions::dim_div(
      expressions::dim_add(expressions::dim_mul(num_elements, expressions::DimType{int64_t{bits}}),
                           expressions::DimType{int64_t{7}}),
      expressions::DimType{int64_t{8}}));
}

// Classifies a candidate reuse of input ``in`` by output ``out`` by comparing
// their buffer sizes. ``kEqual`` requires identical descriptors (same element
// type and shape), so an element-wise overwrite keeps the layout valid even
// for symbolic shapes. ``kGreater`` reports an input buffer that is strictly
// larger in bytes than the output, so the output still fits. When descriptors
// differ but byte sizes are equal (for example a transpose), the opportunity
// is also reported as ``kGreater``: this keeps ``kEqual`` reserved for true
// same-storage matches while still allowing shape-changing rewrites that fit.
// Any other case (input smaller) yields no opportunity.
std::optional<InPlaceReuseKind> ClassifyReuse(const OptimTensor &out, const OptimTensor &in) {
  if (SameStorage(out, in)) {
    return InPlaceReuseKind::kEqual;
  }
  const std::optional<int64_t> out_bytes = ConcreteByteSize(out);
  const std::optional<int64_t> in_bytes = ConcreteByteSize(in);
  if (!out_bytes.has_value() || !in_bytes.has_value()) {
    return std::nullopt;
  }
  if (*in_bytes >= *out_bytes) {
    return InPlaceReuseKind::kGreater;
  }
  return std::nullopt;
}

// Recursively collects the values that ``graph`` captures from outside its own
// scope. Inputs/initializers/intermediates local to ``graph`` are excluded, as
// are names already produced by ancestor subgraphs tracked in
// ``ancestor_locals``.
void CollectGraphExternalInputs(const GraphProto &graph, std::vector<std::string> &out,
                                std::unordered_set<std::string> &seen,
                                const std::unordered_set<std::string> &ancestor_locals) {
  std::unordered_set<std::string> local;
  for (int i = 0; i < graph.input().size(); ++i) {
    local.insert(graph.input()[i].name().as_string());
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    local.insert(graph.initializer()[i].name().as_string());
  }
  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (int j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
      if (!name.empty()) {
        local.insert(name);
      }
    }
  }

  std::unordered_set<std::string> visible_locals = ancestor_locals;
  visible_locals.insert(local.begin(), local.end());

  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (int j = 0; j < nd.input().size(); ++j) {
      const std::string name = nd.input()[j].as_string();
      // Keep only true captures from an outer scope: skip empty names,
      // values local to this subgraph, and names produced by ancestor
      // subgraphs. Such names are already available within the enclosing
      // control-flow body and are not captures of the top-level node itself.
      if (name.empty() || local.count(name) || ancestor_locals.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (int a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, visible_locals);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (int k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, visible_locals);
        }
      }
    }
  }
}

// Collects every unique value a node depends on at runtime: its direct inputs
// plus any external values captured by nested GRAPH / GRAPHS attributes.
std::vector<std::string> CollectNodeInputs(const NodeProto &node) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (int i = 0; i < node.input().size(); ++i) {
    const std::string name = node.input()[i].as_string();
    if (name.empty()) {
      continue;
    }
    if (seen.insert(name).second) {
      out.push_back(name);
    }
  }

  std::unordered_set<std::string> empty_outer;
  for (int a = 0; a < node.attribute().size(); ++a) {
    const AttributeProto &attr = node.attribute()[a];
    if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
      CollectGraphExternalInputs(attr.g(), out, seen, empty_outer);
    } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
      for (int k = 0; k < attr.graphs().size(); ++k) {
        CollectGraphExternalInputs(attr.graphs()[k], out, seen, empty_outer);
      }
    }
  }
  return out;
}

} // namespace

void ComputeContext::ComputeInPlaceReuseGraph(
    const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite,
    const std::unordered_map<std::string, std::string> &value_tags) {
  const int num_nodes = graph.node().size();
  std::vector<std::vector<InPlaceReuse>> result(static_cast<std::size_t>(num_nodes));
  std::vector<std::vector<std::string>> release_after(static_cast<std::size_t>(num_nodes));
  std::vector<NodeMemoryProfile> memory(static_cast<std::size_t>(num_nodes),
                                        MakeEmptyNodeMemoryProfile());

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
  std::vector<std::vector<std::string>> referenced_per_node(static_cast<std::size_t>(num_nodes));
  std::unordered_set<std::string> graph_outputs;
  for (int i = 0; i < graph.output().size(); ++i) {
    graph_outputs.insert(graph.output()[i].name().as_string());
  }
  if (allow_input_overwrite) {
    for (const std::string &name : graph_inputs) {
      if (!name.empty()) {
        producer[name] = -1;
      }
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    const NodeProto &node = graph.node()[i];
    std::vector<std::string> &referenced = referenced_per_node[static_cast<std::size_t>(i)];
    referenced = CollectNodeInputs(node);
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
    const std::vector<std::string> &referenced = referenced_per_node[static_cast<std::size_t>(i)];

    for (const std::string &name : referenced) {
      auto use_it = last_use.find(name);
      if (use_it == last_use.end()) {
        continue;
      }
      if (use_it->second != i) {
        continue;
      }
      if (keep.count(name)) {
        continue;
      }
      auto prod_it = producer.find(name);
      // Releasable values here are top-level intermediates produced by an
      // earlier node. ``-1`` marks declared graph inputs when input overwrite
      // is enabled; they are not considered "results to release" metadata.
      // Values produced at this same node are not eligible.
      if (prod_it == producer.end() || prod_it->second < 0 || prod_it->second >= i) {
        continue;
      }
      release_after[static_cast<std::size_t>(i)].push_back(name);
    }

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

  std::unordered_map<std::string, LiveAllocation> alive;
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name().as_string();
    if (name.empty() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> bytes = ByteSizeExpr(ctx.Get(name));
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{SimplifyDimType(*bytes), MemoryValueSource::kInitializer,
                                 ValueTag(value_tags, name)};
  }
  for (int i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name().as_string();
    if (name.empty() || alive.find(name) != alive.end() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> bytes = ByteSizeExpr(ctx.Get(name));
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{SimplifyDimType(*bytes), MemoryValueSource::kInput,
                                 ValueTag(value_tags, name)};
  }

  for (int i = 0; i < num_nodes; ++i) {
    NodeMemoryProfile &profile = memory[static_cast<std::size_t>(i)];
    for (const auto &kv : alive) {
      AddLiveAllocation(profile, kv.second);
    }

    const NodeProto &node = graph.node()[i];
    std::unordered_map<int, int> reuse_by_output;
    for (const InPlaceReuse &reuse : result[static_cast<std::size_t>(i)]) {
      reuse_by_output[static_cast<int>(reuse.output_index)] = static_cast<int>(reuse.input_index);
    }

    for (int o = 0; o < node.output_size(); ++o) {
      if (reuse_by_output.find(o) != reuse_by_output.end()) {
        continue;
      }
      const std::string out_name = node.output(o).as_string();
      if (out_name.empty() || !ctx.Has(out_name)) {
        continue;
      }
      const std::optional<expressions::DimType> out_bytes = ByteSizeExpr(ctx.Get(out_name));
      if (!out_bytes.has_value()) {
        continue;
      }
      expressions::DimType &output_allocation =
          NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey);
      output_allocation =
          SimplifyDimType(expressions::dim_add(output_allocation, SimplifyDimType(*out_bytes)));
      AddTaggedBytes(NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey),
                     ValueTag(value_tags, out_name), *out_bytes);
    }
    NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) =
        SimplifyDimType(expressions::dim_add(
            NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey),
            NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey)));
    SimplifyNodeMemoryProfile(profile);

    std::unordered_map<std::string, LiveAllocation> outputs_to_add;
    std::unordered_set<std::string> inputs_reused_from_graph_input;
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string out_name = node.output(o).as_string();
      if (out_name.empty() || !ctx.Has(out_name)) {
        continue;
      }
      expressions::DimType alloc_bytes = int64_t{0};
      auto reuse_it = reuse_by_output.find(o);
      if (reuse_it != reuse_by_output.end()) {
        const std::string in_name = node.input(reuse_it->second).as_string();
        auto alive_it = alive.find(in_name);
        if (alive_it == alive.end()) {
          continue;
        }
        alloc_bytes = alive_it->second.bytes;
        if (graph_inputs.find(in_name) != graph_inputs.end()) {
          inputs_reused_from_graph_input.insert(in_name);
        }
      } else {
        const std::optional<expressions::DimType> out_bytes = ByteSizeExpr(ctx.Get(out_name));
        if (!out_bytes.has_value()) {
          continue;
        }
        alloc_bytes = SimplifyDimType(*out_bytes);
      }
      const auto use_it = last_use.find(out_name);
      const bool survives_after_node = graph_outputs.find(out_name) != graph_outputs.end() ||
                                       (use_it != last_use.end() && use_it->second > i);
      if (!survives_after_node) {
        continue;
      }
      outputs_to_add.emplace(out_name, LiveAllocation{SimplifyDimType(alloc_bytes),
                                                      MemoryValueSource::kIntermediate,
                                                      ValueTag(value_tags, out_name)});
    }

    for (const std::string &name : release_after[static_cast<std::size_t>(i)]) {
      alive.erase(name);
    }
    for (const std::string &name : inputs_reused_from_graph_input) {
      alive.erase(name);
    }
    for (auto &entry : outputs_to_add) {
      alive[entry.first] = std::move(entry.second);
    }
  }

  reuse_ = std::move(result);
  release_after_ = std::move(release_after);
  memory_ = std::move(memory);

  // Populate the shape-tagged subset from value_tags (when provided).
  // Only allocate the per-node sub-vectors when value_tags is actually non-empty
  // so that WriteToMetadata can use release_after_shape_tagged_.size() ==
  // reuse_.size() to detect whether shape-tag info was supplied.
  release_after_shape_tagged_.clear();
  if (!value_tags.empty()) {
    const std::size_t n = release_after_.size();
    release_after_shape_tagged_.assign(n, {});
    for (std::size_t i = 0; i < n; ++i) {
      for (const std::string &name : release_after_[i]) {
        auto it = value_tags.find(name);
        if (it != value_tags.end() && it->second == "shape") {
          release_after_shape_tagged_[i].push_back(name);
        }
      }
    }
  }
}

void ComputeContext::WriteToMetadata(GraphProto &graph) const {
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(graph.node().size()) == reuse_.size(),
                      "ComputeContext::WriteToMetadata: graph has ", graph.node().size(),
                      " node(s) but the computed reuse result has ", reuse_.size(),
                      " entry(ies); call WriteToMetadata on the same graph passed to ",
                      "ComputeInPlaceReuseGraph.");
  const bool has_shape_tag_info = release_after_shape_tagged_.size() == reuse_.size();
  for (std::size_t i = 0; i < reuse_.size(); ++i) {
    const bool has_shape_tagged = has_shape_tag_info && !release_after_shape_tagged_[i].empty();
    if (reuse_[i].empty() && release_after_[i].empty() && !has_shape_tagged) {
      continue;
    }
    NodeProto &node = (*graph.mutable_node())[i];
    if (!reuse_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < reuse_[i].size(); ++j) {
        const InPlaceReuse &r = reuse_[i][j];
        if (j != 0) {
          value << ";";
        }
        value << r.output_index << ":" << r.input_index << ":"
              << (r.kind == InPlaceReuseKind::kEqual ? "equal" : "greater");
      }
      node.add_metadata(kInPlaceReuseMetadataKey, value.str());
    }
    if (!release_after_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < release_after_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << release_after_[i][j];
      }
      node.add_metadata(kReleaseAfterMetadataKey, value.str());
    }
    if (has_shape_tagged) {
      std::ostringstream value;
      for (std::size_t j = 0; j < release_after_shape_tagged_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << release_after_shape_tagged_[i][j];
      }
      node.add_metadata(kReleaseAfterShapeTagMetadataKey, value.str());
    }
  }
}

std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuse(const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite);
  return inplace.Reuse();
}

void WriteInPlaceReuseToMetadata(GraphProto &graph, const ShapesContext &ctx,
                                 const std::unordered_map<std::string, std::string> &value_tags) {
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, value_tags);
  inplace.WriteToMetadata(graph);
}

} // namespace annotations
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
