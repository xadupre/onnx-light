#include "onnx_core/graph/graph_partition.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_proto/onnx_helper.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {

uint64_t CheckedAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::invalid_argument("Operator grouping transfer cost overflows uint64.");
  return a + b;
}

std::optional<uint64_t> TensorBytes(const ValueInfoProto &value) {
  if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape())
    return std::nullopt;
  const auto &tensor = value.type().tensor_type();
  const uint64_t bits = FixedBitWidth(static_cast<TensorProto::DataType>(tensor.elem_type()));
  if (!bits)
    return std::nullopt;
  bool zero = false;
  bool unknown = false;
  for (const auto &dim : tensor.shape().dim()) {
    unknown = unknown || !dim.has_dim_value() || dim.dim_value() < 0;
    zero = zero || (dim.has_dim_value() && dim.dim_value() == 0);
  }
  if (zero)
    return 0;
  if (unknown)
    return std::nullopt;
  uint64_t bytes = bits / 8;
  uint64_t remainder = bits % 8;
  for (const auto &dim : tensor.shape().dim()) {
    const uint64_t size = static_cast<uint64_t>(dim.dim_value());
    if (bytes > std::numeric_limits<uint64_t>::max() / size)
      return std::nullopt;
    bytes *= size;
    // Carries fractional bytes separately, even when the element count exceeds uint64.
    // remainder <= 7 and size <= INT64_MAX keep both tail products representable.
    const uint64_t tail_bits = remainder * (size % 8);
    const uint64_t tail_bytes = remainder * (size / 8) + tail_bits / 8;
    if (tail_bytes > std::numeric_limits<uint64_t>::max() - bytes)
      return std::nullopt;
    bytes += tail_bytes;
    remainder = tail_bits % 8;
  }
  if (remainder && bytes == std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return bytes + (remainder != 0);
}

struct Cost {
  uint64_t unknown = 0;
  uint64_t bytes = 0;
  // Retains the high word so losing alternatives cannot overflow the optimum.
  uint64_t overflow = 0;
};

Cost AddCost(const Cost &a, const Cost &b) {
  const uint64_t bytes = a.bytes + b.bytes;
  return {CheckedAdd(a.unknown, b.unknown), bytes,
          CheckedAdd(CheckedAdd(a.overflow, b.overflow), bytes < a.bytes)};
}

struct Score {
  Cost cost;
  size_t grouped = 0;
};

bool Better(const Score &a, const Score &b) {
  if (std::tie(a.cost.unknown, a.cost.overflow, a.cost.bytes) !=
      std::tie(b.cost.unknown, b.cost.overflow, b.cost.bytes))
    return std::tie(a.cost.unknown, a.cost.overflow, a.cost.bytes) <
           std::tie(b.cost.unknown, b.cost.overflow, b.cost.bytes);
  return a.grouped > b.grouped;
}

struct Step {
  size_t start = 0;
  std::optional<OperatorGroup> group;
};

} // namespace

OperatorGroupProposal
ProposeOperatorGroups(const GraphProto &graph, const OperatorGroupOptions &options,
                      const std::unordered_map<std::string, uint64_t> &value_sizes,
                      const std::set<size_t> &excluded_nodes) {
  if (options.min_group_nodes < 2 || options.max_group_nodes < options.min_group_nodes)
    throw std::invalid_argument(
        "Operator grouping requires 2 <= min_group_nodes <= max_group_nodes.");
  if (!excluded_nodes.empty() && *excluded_nodes.rbegin() >= graph.node().size())
    throw std::invalid_argument("Excluded operator grouping node index is out of range.");

  std::unordered_set<std::string> graph_inputs, defined, constants;
  for (const auto &value : graph.input()) {
    graph_inputs.insert(value.name());
    defined.insert(value.name());
  }
  auto initializer = [&](const std::string &name) {
    defined.insert(name);
    if (!graph_inputs.count(name))
      constants.insert(name);
  };
  for (const auto &value : graph.initializer())
    initializer(value.name());
  for (const auto &value : graph.sparse_initializer())
    initializer(value.values().name());
  for (const auto &value : graph.extended_initializer())
    initializer(value.name());

  std::unordered_map<std::string, size_t> producers, last_use;
  std::vector<size_t> indices;
  std::vector<std::vector<std::string>> node_inputs, node_outputs;
  for (size_t index = 0; index < graph.node().size(); ++index) {
    const auto &node = graph.node()[index];
    auto inputs = CollectNodeInputs(node);
    for (const auto &name : inputs)
      if (!defined.count(name))
        throw std::invalid_argument("Node " + std::to_string(index) +
                                    " has undefined or non-topological inputs.");
    const bool constant =
        node.op_type() == "Constant" && node.overload().empty() &&
        (node.domain().empty() || node.domain() == "ai.onnx" || node.domain() == "onnx.ai");
    std::vector<std::string> outputs;
    for (const auto &name : node.output()) {
      if (name.empty())
        continue;
      if (!defined.insert(name).second)
        throw std::invalid_argument("Duplicate value definition '" + name + "'.");
      outputs.push_back(name);
      if (constant)
        constants.insert(name);
      else
        producers[name] = indices.size();
    }
    if (!constant) {
      indices.push_back(index);
      node_inputs.push_back(std::move(inputs));
      node_outputs.push_back(std::move(outputs));
    }
  }
  std::unordered_map<std::string, std::optional<uint64_t>> sizes;
  for (const auto &[name, size] : value_sizes)
    sizes.emplace(name, size);
  auto descriptors = [&](const auto &values) {
    for (const auto &value : values)
      if (!sizes.count(value.name()))
        sizes.emplace(value.name(), TensorBytes(value));
  };
  descriptors(graph.value_info());
  descriptors(graph.input());
  descriptors(graph.output());
  const size_t count = indices.size();
  for (size_t position = 0; position < count; ++position)
    for (const auto &name : node_inputs[position])
      last_use[name] = position;
  for (const auto &value : graph.output()) {
    if (!defined.count(value.name()))
      throw std::invalid_argument("Undefined graph output '" + value.name() + "'.");
    last_use[value.name()] = count;
  }
  auto cost = [&](const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
    Cost result;
    for (const auto *names : {&inputs, &outputs}) {
      for (const auto &name : *names) {
        if (constants.count(name))
          continue;
        const auto size = sizes.find(name);
        if (size == sizes.end() || !size->second)
          result.unknown = CheckedAdd(result.unknown, 1);
        else
          result = AddCost(result, {0, *size->second});
      }
    }
    return result;
  };
  auto live_after = [&](const std::string &name, size_t position) {
    const auto use = last_use.find(name);
    return use != last_use.end() && use->second > position;
  };
  std::vector<std::optional<Score>> best(count + 1);
  std::vector<Step> previous(count + 1);
  best[0] = Score{};
  auto update = [&](size_t position, const Score &score, size_t start,
                    std::optional<OperatorGroup> group) {
    if (!best[position] || Better(score, *best[position])) {
      best[position] = score;
      previous[position] = {start, std::move(group)};
    }
  };
  for (size_t start = 0; start < count; ++start) {
    const Score state = *best[start];
    std::vector<std::string> single_outputs;
    for (const auto &name : node_outputs[start])
      if (live_after(name, start))
        single_outputs.push_back(name);
    const Cost single = cost(node_inputs[start], single_outputs);
    update(start + 1, {AddCost(state.cost, single), state.grouped}, start, std::nullopt);

    std::vector<std::string> inputs, outputs;
    std::unordered_set<std::string> seen_inputs;
    size_t constant_inputs = 0;
    const size_t length_limit =
        static_cast<size_t>(std::min<uint64_t>(count - start, options.max_group_nodes));
    const auto excluded = excluded_nodes.lower_bound(indices[start]);
    for (size_t end = start; end < start + length_limit; ++end) {
      if (excluded != excluded_nodes.end() && indices[end] >= *excluded)
        break;
      for (const auto &name : node_inputs[end]) {
        const auto producer = producers.find(name);
        if ((producer == producers.end() || producer->second < start) &&
            seen_inputs.insert(name).second) {
          inputs.push_back(name);
          constant_inputs += constants.count(name);
        }
        const auto use = last_use.find(name);
        if (use != last_use.end() && use->second == end)
          std::erase(outputs, name);
      }
      for (const auto &name : node_outputs[end])
        if (live_after(name, end))
          outputs.push_back(name);
      if (options.max_constant_inputs && constant_inputs > *options.max_constant_inputs)
        break;
      if (options.max_boundary_bytes) {
        const Cost input_cost = cost(inputs, {});
        if (input_cost.unknown || input_cost.overflow ||
            input_cost.bytes > *options.max_boundary_bytes)
          break;
      }
      const size_t length = end - start + 1;
      if (length < options.min_group_nodes || outputs.empty())
        continue;
      const Cost boundary = cost(inputs, outputs);
      if (options.max_boundary_bytes &&
          (boundary.unknown || boundary.overflow || boundary.bytes > *options.max_boundary_bytes))
        continue;
      const Score score{AddCost(state.cost, boundary), state.grouped + length};
      if (!best[end + 1] || Better(score, *best[end + 1])) {
        OperatorGroup group;
        group.node_indices.assign(indices.begin() + start, indices.begin() + end + 1);
        group.inputs = inputs;
        group.outputs = outputs;
        if (!boundary.unknown && !boundary.overflow)
          group.transfer_bytes = boundary.bytes;
        update(end + 1, score, start, std::move(group));
      }
    }
  }
  if (best[count]->cost.overflow)
    throw std::invalid_argument("Operator grouping transfer cost overflows uint64.");
  std::vector<OperatorGroup> result;
  for (size_t position = count; position;) {
    auto &step = previous[position];
    position = step.start;
    if (step.group)
      result.push_back(std::move(*step.group));
  }
  std::reverse(result.begin(), result.end());
  return MakeIntervalGroupProposal(std::move(result), graph.node().size());
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
