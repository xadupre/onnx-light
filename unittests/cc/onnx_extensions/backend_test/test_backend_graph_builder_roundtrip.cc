// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Feeds every backend test case (excluding the benchmark-sized and ``_big_``
// cases) into a :cpp:class:`core::builder::GraphBuilder`, rebuilds the ONNX
// model and checks that a further import / rebuild round-trip is stable. The
// comparison is order-insensitive for ``opset_import`` and ``metadata_props``:
// both are populated from ``std::unordered_map``s whose iteration order is not
// guaranteed, so the same logical model may serialise them in a different
// order. The Python counterpart lives in
// ``unittests/python/onnx_core/test_graph_builder_backend_roundtrip.py``.

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/builder/graph_builder.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::TestCase;
using core::builder::GraphBuilder;

namespace Test {

namespace {

// Schema provider backed by the built-in ONNX operator schemas, mirroring the
// wiring used by ``onnx_core/graph_builder.py``.
GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  };
}

// Operators the GraphBuilder cannot yet re-import faithfully: their real
// signature carries optional / variadic outputs (or control-flow / function
// bodies) that the built-in LightOpSchema output-count validation or the
// incremental shape inference does not model, so importing such a model raises.
// Cases exercising these operators are skipped; the vast majority of cases
// (~1900) still exercise the round-trip.
const std::unordered_set<std::string> &UnsupportedOps() {
  static const std::unordered_set<std::string> ops = {"Attention",
                                                      "SoftmaxCrossEntropyLoss",
                                                      "MaxPool",
                                                      "Scan",
                                                      "Dropout",
                                                      "LSTM",
                                                      "Adam",
                                                      "Momentum",
                                                      "Loop",
                                                      "BatchNormalization",
                                                      "SequenceMap",
                                                      "Adagrad",
                                                      "If",
                                                      "Optional",
                                                      "OptionalGetElement"};
  return ops;
}

// Returns true when ``graph`` (or any nested subgraph) uses an operator the
// GraphBuilder cannot currently round-trip.
bool UsesUnsupportedOp(const GraphProto &graph) {
  for (const NodeProto &node : graph.node()) {
    if (UnsupportedOps().count(node.op_type().value()) != 0) {
      return true;
    }
    for (const AttributeProto &attr : node.attribute()) {
      if (attr.has_g() && UsesUnsupportedOp(attr.g())) {
        return true;
      }
      for (const GraphProto &sub : attr.graphs()) {
        if (UsesUnsupportedOp(sub)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Cases whose model carries local functions or an unsupported operator are
// skipped (see UnsupportedOps).
bool ShouldSkip(const ModelProto &model) {
  return !model.functions().empty() || UsesUnsupportedOp(model.graph());
}

// Sorts the ``metadata_props`` entries of a proto in place so their order does
// not affect the serialized comparison.
void SortMetadata(utils::RepeatedProtoField<StringStringEntryProto> &props) {
  std::vector<std::pair<std::string, std::string>> items;
  items.reserve(props.size());
  for (const StringStringEntryProto &entry : props) {
    items.emplace_back(entry.key().value(), entry.value().value());
  }
  std::sort(items.begin(), items.end());
  props.clear();
  for (const auto &[key, value] : items) {
    StringStringEntryProto &entry = props.add();
    entry.set_key(key);
    entry.set_value(value);
  }
}

// Recursively canonicalises the ``metadata_props`` ordering of a graph and of
// every value it declares.
void NormalizeGraph(GraphProto &graph) {
  SortMetadata(graph.metadata_props());
  for (NodeProto &node : graph.node()) {
    SortMetadata(node.metadata_props());
    for (AttributeProto &attr : node.attribute()) {
      if (attr.has_g()) {
        NormalizeGraph(*attr.mutable_g());
      }
      for (GraphProto &sub : attr.graphs()) {
        NormalizeGraph(sub);
      }
    }
  }
  for (ValueInfoProto &value : graph.input()) {
    SortMetadata(value.metadata_props());
  }
  for (ValueInfoProto &value : graph.output()) {
    SortMetadata(value.metadata_props());
  }
  for (ValueInfoProto &value : graph.value_info()) {
    SortMetadata(value.metadata_props());
  }
  for (TensorProto &initializer : graph.initializer()) {
    SortMetadata(initializer.metadata_props());
  }
}

// Canonicalises a model so two logically-equal models serialise identically:
// sorts the ``opset_import`` entries and every ``metadata_props`` list.
void Normalize(ModelProto &model) {
  std::vector<std::pair<std::string, int64_t>> opsets;
  opsets.reserve(model.opset_import().size());
  for (const OperatorSetIdProto &opset : model.opset_import()) {
    opsets.emplace_back(opset.domain().value(), opset.version());
  }
  std::sort(opsets.begin(), opsets.end());
  model.clear_opset_import();
  for (const auto &[domain, version] : opsets) {
    model.add_opset(domain, version);
  }
  SortMetadata(model.metadata_props());
  NormalizeGraph(*model.mutable_graph());
}

std::string NormalizedSerialization(ModelProto model) {
  Normalize(model);
  std::string serialized;
  model.SerializeToString(serialized);
  return serialized;
}

} // namespace

// Every non-skipped backend test case must survive a GraphBuilder round-trip:
// importing the model and rebuilding it, then importing that result and
// rebuilding it again, must produce the same model (modulo opset_import and
// metadata_props ordering).
TEST(BackendGraphBuilderRoundTrip, AllCollectedCasesAreStable) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  std::size_t tested = 0;
  for (TestCase &tc : cases) {
    if (ShouldSkip(tc.model())) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    GraphBuilder builder(tc.model(), SchemaLookup());
    const ModelProto rebuilt = builder.ToModel();

    GraphBuilder rebuilder(rebuilt, SchemaLookup());
    const ModelProto rebuilt_again = rebuilder.ToModel();

    EXPECT_EQ(NormalizedSerialization(rebuilt), NormalizedSerialization(rebuilt_again))
        << "case: " << tc.name;
    ++tested;
  }
  // Guards against the collector silently returning nothing testable.
  EXPECT_GT(tested, 0u);
}

} // namespace Test
