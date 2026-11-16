#include "onnx_core/graph/graph_partition.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

size_t BoundedInteger(const char *text, size_t maximum) {
  const std::string value(text);
  size_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < 2 ||
      result > maximum)
    throw std::invalid_argument("Benchmark argument is outside its supported range.");
  return result;
}

GraphProto MakeGraph(size_t count, bool chain) {
  GraphProto graph;
  graph.add_input()->set_name("x");
  for (size_t i = 0; i < count; ++i) {
    const auto output = "v" + std::to_string(i);
    const auto weight = "w" + std::to_string(i);
    graph.add_initializer()->set_name(weight);
    auto *node = graph.add_node();
    node->set_op_type("Add");
    node->add_input(chain && i ? "v" + std::to_string(i - 1) : "x");
    node->add_input(weight);
    node->add_output(output);
    if (!chain || i + 1 == count)
      graph.add_output()->set_name(output);
  }
  return graph;
}

} // namespace

// Usage: bench_graph_partition [max_nodes=768 (2..4096)] [iterations=7 (2..31)]
// Measures proposals only, at max_nodes/2 and max_nodes with L=128 and L=N.
int main(int argc, char **argv) {
  if (argc > 3)
    throw std::invalid_argument("Expected at most max_nodes and iterations.");
  const size_t maximum = argc > 1 ? BoundedInteger(argv[1], 4096) : 768;
  const size_t iterations = argc > 2 ? BoundedInteger(argv[2], 31) : 7;
  for (const bool chain : {false, true})
    for (const size_t count : {std::max(size_t{2}, maximum / 2), maximum}) {
      const auto graph = MakeGraph(count, chain);
      std::unordered_map<std::string, uint64_t> sizes{{"x", 4}};
      for (size_t i = 0; i < count; ++i)
        sizes.emplace("v" + std::to_string(i), 4);
      for (const size_t limit : {size_t{128}, count}) {
        core::graph::OperatorGroupOptions options;
        options.max_group_nodes = limit;
        const auto warmup = core::graph::ProposeOperatorGroups(graph, options, sizes);
        std::vector<double> times;
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
          const auto start = std::chrono::steady_clock::now();
          const auto proposal = core::graph::ProposeOperatorGroups(graph, options, sizes);
          const auto end = std::chrono::steady_clock::now();
          if (proposal.groups.size() != warmup.groups.size())
            throw std::runtime_error("Proposal changed between iterations.");
          times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        std::sort(times.begin(), times.end());
        std::cout << "topology=" << (chain ? "weighted_chain" : "independent") << " nodes=" << count
                  << " max_group_nodes=" << limit << " groups=" << warmup.groups.size()
                  << " median_proposal_ms=" << times[times.size() / 2] << "\n";
      }
    }
}
