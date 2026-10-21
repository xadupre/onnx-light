# Pattern Optimization: onnx-light vs. onnxscript

onnx-light pattern application algorithm:

* [class implementing the whole algorithm](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/builder/graph_graph.h)
* [patterns](https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_extensions/patterns)

onnx-light uses a pattern class (`match` / `apply`) and an algorithm that applies
patterns and manages graph updates. Each application records the matched nodes
and the produced nodes. The complete edit sequence can then be serialized and
replayed from the initial graph, without running pattern matching again.

| Aspect | onnx-light | this proposal |
|---|---|---|
| Pattern definition | A class with `match` and `apply`, in Python or C++. | Textual source and replacement patterns, with C++ callbacks for additional checks and bindings. |
| Custom-rule simplicity | More explicit code. A declarative shorthand would be straightforward to add, but its practical value may be limited now that Copilot can generate the class boilerplate. | Simpler custom rules for straightforward structural replacements, thanks to the declarative syntax. |
| Application algorithm | Selects and applies patterns, manages graph consistency, and records edits. | Matches and applies rules, manages graph updates, and collects statistics. |
| Each application | Records a sequence of matched nodes and a sequence of produced nodes, plus associated edits. | Binds the source pattern and instantiates the replacement in the live graph. No equivalent serialized edit record is exposed. |
| Replay | Replays the complete recorded sequence on the initial graph without matching again. | Reusable rule definitions still require matching. The proposed API has no equivalent concrete-diff replay. |
| Constant folding | Moves the folded computation into functions, recording calls that regenerate constants during replay rather than storing copies of computed initializers. | Supports constants in patterns and replacements, but does not expose an equivalent function-based constant-folding replay mechanism. |
| Symbolic expressions | Provides an expression engine to simplify and compare symbolic dimensions, e.g. recognize `2*n` and `n+n` as equal, supporting shape reasoning in patterns. | Matches existing shape metadata and repeated dimension bindings exactly; does not simplify expressions or solve equations. Additional reasoning would need C++ callback logic. |
| Multiple inputs and outputs | Supports patterns with multiple inputs and multiple outputs. | Also supports multiple inputs and explicitly returned outputs, but requires a connected pattern anchored at a unique final sink. Independent sinks are unsupported. |
| Optional nodes | One `match` can handle optional nodes and pass the actual configuration to `apply`, including `None` placeholders. | Supports absent input slots, not optional nodes. Requires separate rules for structural variants; if several match the same candidate, the first eligible rule in registration order wins. |

**The key difference is the replayable result, not just the matching syntax.**
onnx-light retains the transformations as data, including constant computations,
instead of retaining only the optimized graph.

For microsoft/onnxruntime PR 33404,
this suggests recording accepted runtime fusions as replayable plans. Such plans
would need to match the post-partition graph and preserve host validation;
the proposal currently excludes cached-graph replay from its V1 extension path.

## Custom Pattern Example

Both examples replace `Neg(Neg(x))` with `Identity(x)`. The intermediate value
must not be a graph output, captured by a subgraph, or shared with another node.
`model` denotes the original model containing this chain.

### Python

```python
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.optimization import (
    GraphBuilder, GraphGraph, PatternOptimization,
)


class NegNegPattern(PatternOptimization):
    def __init__(self):
        super().__init__(priority=1, name="NegNeg")

    def fast_op_type(self):
        return {"Neg"}

    def match(self, graph, node):
        previous = graph.node_before(node.input[0])
        if (
            node.domain not in ("", "ai.onnx")
            or previous is None
            or previous.op_type != "Neg"
            or previous.domain not in ("", "ai.onnx")
        ):
            return self.no_match(node, "expected two ONNX Neg nodes")
        name = previous.output[0]
        if (
            graph.is_output(name)
            or graph.is_used_by_subgraph(name)
            or graph.is_used_more_than_once(name)
        ):
            return self.no_match(node, "the intermediate must remain available")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        previous, node = nodes
        return [oh.make_node("Identity", [previous.input[0]], list(node.output))]


builder = GraphBuilder(model)
rewrites = GraphGraph(builder, patterns=[NegNegPattern()]).optimize()
replayed = GraphBuilder(model).apply_rewritings(rewrites)
```

### C++

```cpp
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_optimization.h"

using namespace onnx_light::core::builder;
using onnx_light::NodeProto;
using onnx_light::utils::RepeatedProtoField;

class NegNegPattern final : public PatternOptimization {
public:
  NegNegPattern() : PatternOptimization(1, "NegNeg") {}

  std::set<std::string> FastOpType() const override { return {"Neg"}; }

  MatchResult Match(GraphGraph &graph, const NodeProto &node) const override {
    const NodeProto *previous = graph.NodeBefore(node.input()[0].value());
    if ((node.domain().value() != "" && node.domain().value() != "ai.onnx") ||
        previous == nullptr || previous->op_type().value() != "Neg" ||
        (previous->domain().value() != "" && previous->domain().value() != "ai.onnx")) {
      return NoMatch(node, "expected two ONNX Neg nodes");
    }
    const auto &name = previous->output()[0].value();
    if (graph.IsOutput(name) || graph.IsUsedBySubgraph(name) ||
        graph.IsUsedMoreThanOnce(name)) {
      return NoMatch(node, "the intermediate must remain available");
    }
    return MatchResult{this, {previous, &node}, &node};
  }

  RepeatedProtoField<NodeProto>
  Apply(GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(onnx_light::MakeNode(
        "Identity", {nodes[0]->input()[0].value()}, {nodes[1]->output()[0].value()}));
    return replacements;
  }
};

void OptimizeExample(const onnx_light::ModelProto &model) {
  GraphBuilder builder(model);
  std::vector<std::unique_ptr<PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<NegNegPattern>());
  GraphGraph optimizer(builder, std::move(patterns));
  auto rewrites = optimizer.Optimize();
  auto replayed = GraphBuilder(model).ApplyRewritings(rewrites);
}
```

In both cases, `match` identifies the two nodes and `apply` returns one new node.
The optimizer handles graph updates and edit capture; replay does not call the
pattern again.

### The Same Rule with onnxscript's Design

The source and replacement are textual graphs embedded in C++. Here the rule
is restricted to float32 inputs and ONNX opsets 14 through 21.

```cpp
#include "core/optimizer/pattern_rewriter.h"

using namespace onnxruntime;
using namespace onnxruntime::rewrite;

Status OptimizeNegNeg(Graph &graph) {
  Rule rule{
      "NegNeg",
      R"(
        (x:float[]) => {
          intermediate = Neg(x)
          return Neg(intermediate)
        }
      )",
      "(x) => Identity(x)",
      {},
      {{kOnnxDomain, 14, 21}}};

  PatternRewriter optimizer(Rules{std::move(rule)});
  Statistics statistics;
  ORT_RETURN_IF_ERROR(optimizer.Run(graph, statistics));
  return graph.Resolve();
}
```

The matcher automatically rejects deletion of the intermediate if it is a graph
output, has an external consumer, or is captured by a subgraph. No custom callback
is needed for these checks.

**For this simple rule, the declarative version is shorter:** the engine provides
the structural matching that the onnx-light class writes explicitly. However,
`Run` modifies the graph without returning a serialized edit sequence to replay.
The proposed interface is C++; it does not expose an equivalent Python API.
