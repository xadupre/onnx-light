// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
#include "onnx_core/graph/graph_layout.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::graph;

namespace {
NodeProto Node(const std::string &output, const std::vector<std::string> &inputs = {},
               const std::string &op = "Identity") {
  NodeProto node;
  node.set_op_type(op);
  for (const auto &input : inputs)
    node.add_input(input);
  node.add_output(output);
  return node;
}
template <class Proto> void Meta(Proto &proto, const char *key, const std::string &value) {
  auto *entry = proto.add_metadata_props();
  entry->set_key(key);
  entry->set_value(value);
}
std::string MetaValue(const NodeProto &node, const char *key) {
  for (const auto &entry : node.metadata_props())
    if (entry.key() == key)
      return entry.value();
  return {};
}
} // namespace

TEST(GraphLayout, FunctionBodyMutationsAndRoundTrip) {
  FunctionProto function;
  function.set_name("Activation");
  function.add_input("X");
  function.add_output("Y");
  *function.add_node() = Node("A", {"X"});
  *function.add_node() = Node("Y", {"A"}, "Relu");
  *function.add_node() = Node("unused", {"X"});
  GraphLayout layout(function, true);
  const auto first = layout.Order()[0], second = layout.Order()[1];
  const auto independent = *layout.Get(layout.Order()[2]);
  EXPECT_FALSE(layout.Edges(first)[0].capture);
  ASSERT_TRUE(layout.SetSize(first, 400, 100));
  EXPECT_GT(layout.Get(second)->x, 400);
  EXPECT_EQ(layout.Get(independent.id)->x, independent.x);
  EXPECT_EQ(layout.Get(independent.id)->y, independent.y);
  EXPECT_EQ(layout.Stats().nodes, 2u);
  auto inserted = layout.Insert(0, Node("prefix", {"X"}));
  ASSERT_TRUE(layout.Replace(first, Node("A", {"prefix"}, "Neg")));
  EXPECT_EQ(layout.Edges(first)[0].source, inserted);
  ASSERT_TRUE(layout.RewireInput(first, 0, "X"));
  ASSERT_TRUE(layout.Erase(inserted));
  const auto appended = layout.Append(Node("inspection", {"Y"}));
  EXPECT_EQ(layout.Edges(appended)[0].source, second);
  EXPECT_EQ(function.node(0).op_type(), "Neg");
  EXPECT_EQ(function.node(3).output(0), "inspection");
  ASSERT_TRUE(layout.SetPosition(first, -500, -100));
  layout.Persist();
  FunctionProto restored;
  ASSERT_TRUE(restored.ParseFromString(function.SerializeAsString()));
  GraphLayout resumed(restored, true);
  EXPECT_EQ(resumed.Order(), layout.Order());
  for (auto id : layout.Order()) {
    EXPECT_EQ(resumed.Get(id)->x, layout.Get(id)->x);
    EXPECT_EQ(resumed.Get(id)->y, layout.Get(id)->y);
    EXPECT_EQ(resumed.Get(id)->width, layout.Get(id)->width);
    EXPECT_EQ(resumed.Get(id)->height, layout.Get(id)->height);
  }
}

TEST(GraphLayout, FunctionSubgraphsCaptureFormalsAndPropagateLocalEdits) {
  FunctionProto function;
  function.add_input("X");
  function.add_output("Y");
  *function.add_node() = Node("capture", {"X"});
  auto container = Node("Y", {}, "Container");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  auto *graph = attribute->mutable_g();
  graph->add_output()->set_name("inner");
  *graph->add_node() = Node("inner", {"capture", "X"}, "Add");
  *function.add_node() = container;
  GraphLayout layout(function, true);
  const auto producer = layout.Order()[0], parent = layout.Order()[1];
  ASSERT_EQ(layout.Edges(parent).size(), 2u);
  EXPECT_EQ(layout.Edges(parent)[1].source, producer);
  auto *child = layout.Children(parent)[0];
  EXPECT_TRUE(child->Edges(child->Order()[0])[0].capture);
  ASSERT_TRUE(child->SetSize(child->Order()[0], 500, 100));
  EXPECT_GE(layout.Get(parent)->width, 500);
  layout.Persist();
  GraphLayout shallow(function, false);
  EXPECT_TRUE(shallow.Children(parent).empty());
  EXPECT_EQ(shallow.Get(parent)->width, layout.Get(parent)->width);
  EXPECT_EQ(shallow.Get(parent)->height, layout.Get(parent)->height);
}

TEST(GraphLayout, MissingFixedSizeMetadataRecomputesDimensions) {
  GraphProto graph;
  *graph.add_node() = Node("Y", {"X"});
  GraphLayout initial(graph, true);
  const auto id = initial.Order()[0];
  const auto automatic = *initial.Get(id);
  ASSERT_TRUE(initial.SetSize(id, 1000, 900));
  initial.Persist();
  auto &metadata = *graph.mutable_node(0)->mutable_metadata_props();
  for (auto it = metadata.begin(); it != metadata.end();)
    if (it->key() == GraphLayout::FixedSizeKey)
      it = metadata.erase(it);
    else
      ++it;
  GraphLayout resumed(graph, true);
  EXPECT_EQ(resumed.Get(id)->width, automatic.width);
  EXPECT_EQ(resumed.Get(id)->height, automatic.height);
}

TEST(GraphLayout, StreamingLateProducerAndRoutes) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  auto consumer = layout.Append(Node("z", {"late"}));
  EXPECT_EQ(layout.Edges(consumer)[0].source, 0u);
  EXPECT_TRUE(layout.Edges(consumer)[0].capture);
  auto producer = layout.Append(Node("late"));
  EXPECT_EQ(layout.Get(consumer)->layer, 1u);
  EXPECT_EQ(layout.Edges(consumer)[0].source, producer);
  EXPECT_FALSE(layout.Edges(consumer)[0].capture);
  EXPECT_GT(layout.Get(consumer)->x, layout.Get(producer)->x);
  ASSERT_EQ(layout.Edges(consumer)[0].points.size(), 4u);
  EXPECT_EQ(graph.node()[0].output()[0], "z");
  EXPECT_EQ(graph.node()[1].output()[0], "late");
}

TEST(GraphLayout, LocalMutationDoesNotVisitIndependentBranches) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  auto a = layout.Append(Node("a"));
  auto b = layout.Append(Node("b", {"a"}));
  auto c = layout.Append(Node("c", {"b"}));
  for (size_t i = 0; i < 200; ++i)
    layout.Append(Node("independent_" + std::to_string(i)));
  const auto untouched = *layout.Get(layout.Order().back());
  EXPECT_TRUE(layout.RewireInput(b, 0, "external"));
  EXPECT_LE(layout.Stats().nodes, 2u);
  EXPECT_LE(layout.Stats().edges, 2u);
  EXPECT_EQ(layout.Get(c)->layer, 1u);
  EXPECT_EQ(layout.Get(a)->layer, 0u);
  EXPECT_EQ(layout.Get(untouched.id)->x, untouched.x);
  EXPECT_EQ(layout.Get(untouched.id)->y, untouched.y);
}

TEST(GraphLayout, ReindexesUnchangedAffectedDescendants) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("a");
  *attribute->mutable_g()->add_node() = Node("b", {"a"});
  *attribute->mutable_g()->add_node() = Node("c");
  *graph.add_node() = container;

  GraphLayout layout(graph, true);
  const auto parent = layout.Order()[0];
  auto *child = layout.Children(parent)[0];
  const auto a = child->Order()[0];
  const auto b = child->Order()[1];
  const auto c = child->Order()[2];
  const auto before = *child->Get(b);
  const auto parent_width = layout.Get(parent)->width;

  ASSERT_TRUE(child->SetSize(a, child->Get(a)->width, 200));
  EXPECT_EQ(child->Get(b)->layer, before.layer);
  EXPECT_EQ(child->Get(b)->x, before.x);
  EXPECT_EQ(child->Bounds().right, before.x + before.width);
  EXPECT_EQ(layout.Get(parent)->width, parent_width);
  EXPECT_FALSE(child->SetPosition(c, before.x, before.y));

  layout.Persist();
  GraphLayout shallow(graph, false);
  EXPECT_EQ(shallow.Get(parent)->width, parent_width);
}

TEST(GraphLayout, CollisionQueriesStayLocalForLongChains) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  std::string input;
  for (size_t i = 0; i < 512; ++i) {
    const auto output = "chain_" + std::to_string(i);
    layout.Append(
        Node(output, input.empty() ? std::vector<std::string>{} : std::vector<std::string>{input}));
    input = output;
  }
  EXPECT_LT(layout.Stats().collision_index_visits, 40u);
}

TEST(GraphLayout, InsertDeleteReplacePreserveIdsAndOrder) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  auto a = layout.Append(Node("a"));
  auto c = layout.Append(Node("c", {"b"}));
  auto b = layout.Insert(1, Node("b", {"a"}));
  EXPECT_EQ(layout.Order(), (std::vector<LayoutNodeId>{a, b, c}));
  EXPECT_EQ(layout.Get(c)->layer, 2u);
  auto replacement = Node("other", {"external"}, "LongOperatorName");
  replacement.set_name("renamed");
  EXPECT_TRUE(layout.Replace(b, replacement));
  EXPECT_EQ(layout.Order()[1], b);
  EXPECT_EQ(layout.Edges(c)[0].source, 0u);
  EXPECT_TRUE(layout.Erase(b));
  EXPECT_EQ(layout.Order(), (std::vector<LayoutNodeId>{a, c}));
  EXPECT_EQ(graph.node()[1].output()[0], "c");
  EXPECT_EQ(layout.Get(b), nullptr);
  EXPECT_FALSE(layout.Erase(b));
  EXPECT_FALSE(layout.Replace(b, replacement));
  EXPECT_FALSE(layout.RewireInput(a, 9, "bad"));
  EXPECT_THROW(layout.Insert(20, replacement), std::out_of_range);
}

TEST(GraphLayout, StableIdsDoNotDependOnNodeNames) {
  GraphProto graph;
  *graph.add_node() = Node("a");
  *graph.add_node() = Node("b");
  graph.mutable_node(0)->set_name("duplicate");
  graph.mutable_node(1)->set_name("duplicate");
  GraphLayout layout(graph);
  layout.Load();
  const auto ids = layout.Order();
  EXPECT_NE(ids[0], ids[1]);
  layout.Persist();
  graph.mutable_node()->SwapElements(0, 1);
  graph.mutable_node(0)->set_name("");
  GraphLayout reloaded(graph);
  reloaded.Load();
  EXPECT_EQ(reloaded.Order(), (std::vector<LayoutNodeId>{ids[1], ids[0]}));
}

TEST(GraphLayout, ReloadResetsAllocatorWhenNextIdIsMissingOrInvalid) {
  for (const std::string next : {"", "invalid", "0", "9007199254740992"}) {
    SCOPED_TRACE(next);
    GraphProto graph;
    Meta(graph, GraphLayout::NextIdKey, "100");
    *graph.add_node() = Node("old");
    GraphLayout layout(graph, true);
    ASSERT_EQ(layout.Order()[0], 100u);

    graph = GraphProto();
    if (!next.empty())
      Meta(graph, GraphLayout::NextIdKey, next);
    *graph.add_node() = Node("new");
    layout.Load();

    EXPECT_EQ(layout.Order(), (std::vector<LayoutNodeId>{1}));
    EXPECT_EQ(layout.Append(Node("later")), 2u);
  }
}

TEST(GraphLayout, EraseRejectsMissingProtoNodeBeforeUpdatingIndexes) {
  GraphProto graph;
  *graph.add_node() = Node("a");
  GraphLayout layout(graph, true);
  const auto id = layout.Order()[0];
  graph.clear_node();

  EXPECT_THROW(layout.Erase(id), std::logic_error);
  EXPECT_EQ(layout.Order(), (std::vector<LayoutNodeId>{id}));
  layout.Load();
  EXPECT_FALSE(layout.Erase(id));
}

TEST(GraphLayout, PartialAndDuplicateIdsAreRepairedWithoutStealingValidIds) {
  GraphProto graph;
  Meta(graph, GraphLayout::VersionKey, "1");
  *graph.add_node() = Node("new");
  auto *valid = graph.add_node();
  *valid = Node("valid");
  Meta(*valid, GraphLayout::IdKey, "1");
  Meta(*valid, GraphLayout::PositionKey, "-75 -80 160 64");
  auto *duplicate = graph.add_node();
  *duplicate = Node("duplicate");
  Meta(*duplicate, GraphLayout::IdKey, "1");
  auto *bad = graph.add_node();
  *bad = Node("bad");
  Meta(*bad, GraphLayout::IdKey, "18446744073709551616");
  GraphLayout layout(graph);
  layout.Load();
  EXPECT_EQ(layout.Order()[1], 1u);
  EXPECT_NE(layout.Order()[0], 1u);
  EXPECT_NE(layout.Order()[2], 1u);
  EXPECT_EQ(layout.Get(1)->x, -75);
  EXPECT_EQ(layout.Get(1)->y, -80);
  EXPECT_FALSE(layout.Diagnostics().empty());
}

TEST(GraphLayout, RoundTripPreservesNegativeCoordinatesAndSemanticFields) {
  GraphProto graph;
  graph.set_doc_string("unchanged");
  Meta(graph, "unrelated", "keep");
  *graph.add_node() = Node("a", {"x", ""}, "Add");
  graph.mutable_node(0)->set_doc_string("node docs");
  Meta(*graph.mutable_node(0), "user", "data");
  GraphLayout layout(graph);
  layout.Load();
  auto id = layout.Order()[0];
  EXPECT_TRUE(layout.SetPosition(id, -123.5, -67.25));
  layout.Persist();
  layout.Persist();
  GraphLayout reload(graph);
  reload.Load();
  EXPECT_EQ(reload.Get(id)->x, -123.5);
  EXPECT_EQ(reload.Get(id)->y, -67.25);
  EXPECT_EQ(graph.doc_string(), "unchanged");
  EXPECT_EQ(graph.node()[0].input()[1], "");
  EXPECT_EQ(graph.node()[0].doc_string(), "node docs");
  EXPECT_EQ(MetaValue(graph.node()[0], "user"), "data");
  EXPECT_FALSE(reload.SetPosition(id, std::numeric_limits<double>::infinity(), 0));
  EXPECT_FALSE(reload.SetPosition(id, 0, std::nan("")));
  EXPECT_FALSE(reload.SetPosition(id, 1e12, 0));
}

TEST(GraphLayout, MetadataValidationRejectsMalformedAndUnsupportedGeometry) {
  for (const auto &position : {"nan 0 160 64", "0 0 -1 64", "0 0 1e99 64", "0 0 160 64 trailing",
                               "1e20 0 1 1", "0 0 0 1", "0 0 1 1 1", "0 0"}) {
    GraphProto graph;
    Meta(graph, GraphLayout::VersionKey, "1");
    *graph.add_node() = Node("a");
    Meta(*graph.mutable_node(0), GraphLayout::PositionKey, position);
    GraphLayout layout(graph);
    layout.Load();
    EXPECT_FALSE(layout.Diagnostics().empty()) << position;
    EXPECT_GE(layout.Get(layout.Order()[0])->width, 1);
  }
  GraphProto graph;
  Meta(graph, GraphLayout::VersionKey, "999");
  *graph.add_node() = Node("a");
  Meta(*graph.mutable_node(0), GraphLayout::PositionKey, "-200 -300 160 64");
  GraphLayout layout(graph);
  layout.Load();
  EXPECT_EQ(layout.Get(layout.Order()[0])->x, 0);
  EXPECT_FALSE(layout.Diagnostics().empty());
}

TEST(GraphLayout, PortsAreOrderedOptionalInputsStayOptionalAndConstantsFold) {
  GraphProto graph;
  graph.add_input()->set_name("x");
  graph.add_initializer()->set_name("weight");
  GraphLayout layout(graph);
  layout.Load();
  auto source_node = Node("first", {}, "Constant");
  source_node.add_output("second");
  auto source = layout.Append(source_node);
  auto consumer = layout.Append(Node("y", {"second", "", "first", "weight", "x"}, "Add"));
  const auto &edges = layout.Edges(consumer);
  ASSERT_EQ(edges.size(), 4u);
  EXPECT_EQ(edges[0].source, source);
  EXPECT_EQ(edges[0].source_port, 1u);
  EXPECT_EQ(edges[0].target_port, 0u);
  EXPECT_EQ(edges[1].source_port, 0u);
  EXPECT_EQ(edges[1].target_port, 2u);
  EXPECT_TRUE(edges[0].folded);
  EXPECT_TRUE(edges[2].folded);
  EXPECT_FALSE(edges[3].folded);
  EXPECT_TRUE(layout.Get(source)->folded);
  EXPECT_GT(layout.Get(consumer)->height, layout.Get(source)->height);
}

TEST(GraphLayout, OccupiedLayersAvoidOverlapAfterDeleteAndRewire) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  auto a = layout.Append(Node("a"));
  auto b = layout.Append(Node("b"));
  auto c = layout.Append(Node("c"));
  EXPECT_GE(layout.Get(b)->y, layout.Get(a)->y + layout.Get(a)->height);
  EXPECT_TRUE(layout.Erase(b));
  auto d = layout.Append(Node("d"));
  EXPECT_LT(layout.Get(d)->y, layout.Get(c)->y);
  EXPECT_GE(layout.Get(d)->y, layout.Get(a)->y + layout.Get(a)->height);
}

TEST(GraphLayout, NestedScopesCapturesAndTargetedChildPropagation) {
  GraphProto graph;
  auto container_node = Node("result", {}, "If");
  auto *attribute = container_node.add_attribute();
  attribute->set_name("then_branch");
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("inside", {"captured"});
  auto *list_attribute = container_node.add_attribute();
  list_attribute->set_name("bodies");
  list_attribute->set_type(AttributeProto::GRAPHS);
  *list_attribute->add_graphs()->add_node() = Node("inside", {"other_capture"});
  GraphLayout layout(graph);
  layout.Load();
  auto container = layout.Append(container_node);
  auto unrelated = layout.Append(Node("unrelated"));
  const auto unrelated_before = *layout.Get(unrelated);
  auto captured = layout.Append(Node("captured"));
  auto children = layout.Children(container);
  ASSERT_EQ(children.size(), 2u);
  EXPECT_EQ(children[0]->Order()[0], children[1]->Order()[0]);
  bool found_capture = false;
  for (const auto &edge : layout.Edges(container))
    if (edge.value == "captured") {
      EXPECT_TRUE(edge.capture);
      EXPECT_EQ(edge.source, captured);
      found_capture = true;
    }
  EXPECT_TRUE(found_capture);
  const auto sibling = *children[1]->Get(children[1]->Order()[0]);
  auto added = children[0]->Append(Node("more", {"inside"}));
  EXPECT_GT(children[0]->Get(added)->x, 0);
  EXPECT_EQ(children[1]->Get(sibling.id)->x, sibling.x);
  EXPECT_LE(layout.Stats().nodes, 1u);
  EXPECT_EQ(layout.Get(unrelated)->y, unrelated_before.y);
  EXPECT_GT(layout.Get(container)->width, 160);
  EXPECT_TRUE(children[0]->RewireInput(children[0]->Order()[0], 0, "new_capture"));
  bool found_new = false;
  for (const auto &edge : layout.Edges(container)) {
    EXPECT_NE(edge.value, "captured");
    found_new |= edge.value == "new_capture";
  }
  EXPECT_TRUE(found_new);
}

TEST(GraphLayout, GraphOutputsCaptureOuterValuesAndLocalInputsShadowThem) {
  GraphProto graph;
  auto node = Node("result", {}, "If");
  auto *attribute = node.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  attribute->mutable_g()->add_output()->set_name("outer");
  attribute->mutable_g()->add_input()->set_name("shadow");
  *attribute->mutable_g()->add_node() = Node("local", {"shadow"});
  GraphLayout layout(graph);
  layout.Load();
  auto id = layout.Append(node);
  ASSERT_EQ(layout.Edges(id).size(), 1u);
  EXPECT_EQ(layout.Edges(id)[0].value, "outer");
}

TEST(GraphLayout, CyclesTerminateWithDiagnostics) {
  GraphProto graph;
  *graph.add_node() = Node("a", {"b"});
  *graph.add_node() = Node("b", {"a"});
  GraphLayout layout(graph);
  layout.Load();
  EXPECT_FALSE(layout.Diagnostics().empty());
  EXPECT_LE(layout.Stats().nodes, 1024u);
}

TEST(GraphLayout, DeepSubgraphsAreBounded) {
  GraphProto graph;
  GraphProto *nested = &graph;
  for (size_t depth = 0; depth < 70; ++depth) {
    auto *node = nested->add_node();
    node->set_op_type("If");
    node->add_output("y");
    auto *attribute = node->add_attribute();
    attribute->set_type(AttributeProto::GRAPH);
    nested = attribute->mutable_g();
  }
  GraphLayout layout(graph);
  layout.Load();
  auto *current = &layout;
  size_t scopes = 1;
  while (!current->Children(current->Order()[0]).empty()) {
    current = current->Children(current->Order()[0])[0];
    ++scopes;
  }
  EXPECT_EQ(scopes, 65u);
  EXPECT_FALSE(current->Diagnostics().empty());
}

TEST(GraphLayout, ReverseIdsAndReverseProtoOrderStillUseLinearTopologicalWork) {
  GraphProto graph;
  constexpr size_t count = 300;
  for (size_t i = count; i > 0; --i) {
    auto *node = graph.add_node();
    *node = Node("v" + std::to_string(i),
                 i == 1 ? std::vector<std::string>{}
                        : std::vector<std::string>{"v" + std::to_string(i - 1)});
    Meta(*node, GraphLayout::IdKey, std::to_string(count - i + 1));
  }
  GraphLayout layout(graph);
  layout.Load();
  EXPECT_TRUE(layout.Diagnostics().empty());
  EXPECT_EQ(layout.Stats().nodes, count);
  EXPECT_EQ(layout.Get(layout.Order()[0])->layer, count - 1);
  EXPECT_EQ(graph.node()[0].output()[0], "v300");
}

TEST(GraphLayout, VariableWidthsAndTallNodesDoNotOverlapAcrossLayers) {
  GraphProto graph;
  GraphLayout layout(graph);
  layout.Load();
  auto short_node = layout.Append(Node("short"));
  auto wide = layout.Append(Node("wide", {}, std::string(60, 'W')));
  auto tall = layout.Append(Node("tall", {"short", "x", "y", "z", "a", "b", "c", "d"}));
  const auto &a = *layout.Get(wide);
  const auto &b = *layout.Get(tall);
  EXPECT_TRUE(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y ||
              b.y + b.height <= a.y);
  EXPECT_EQ(layout.Get(short_node)->y, 0);
}

TEST(GraphLayout, GrandchildMutationUpdatesOnlyContainersAndTheirConsumers) {
  GraphProto graph;
  auto outer = Node("outer", {}, "If");
  auto *outer_attribute = outer.add_attribute();
  outer_attribute->set_type(AttributeProto::GRAPH);
  auto middle = Node("middle", {}, "If");
  auto *middle_attribute = middle.add_attribute();
  middle_attribute->set_type(AttributeProto::GRAPH);
  *middle_attribute->mutable_g()->add_node() = Node("inner");
  *outer_attribute->mutable_g()->add_node() = middle;
  GraphLayout layout(graph);
  layout.Load();
  auto outer_id = layout.Append(outer);
  auto consumer = layout.Append(Node("consumer", {"outer"}));
  auto unrelated = layout.Append(Node("unrelated"));
  const auto before = *layout.Get(unrelated);
  auto *child = layout.Children(outer_id)[0];
  auto *grandchild = child->Children(child->Order()[0])[0];
  grandchild->Append(Node("next", {"inner", "captured"}));
  EXPECT_EQ(layout.Get(unrelated)->y, before.y);
  EXPECT_EQ(layout.Stats().nodes, 2u);
  EXPECT_EQ(child->Stats().nodes, 1u);
  EXPECT_EQ(layout.Edges(consumer)[0].source, outer_id);
  ASSERT_EQ(layout.Edges(outer_id).size(), 1u);
  EXPECT_EQ(layout.Edges(outer_id)[0].value, "captured");
}

TEST(GraphLayout, ReloadRestoresFramesButAutomaticPositionsStillReactToEdits) {
  GraphProto graph;
  GraphLayout original(graph);
  original.Load();
  original.Append(Node("a"));
  auto b = original.Append(Node("b", {"a"}));
  auto pinned = original.Append(Node("pinned"));
  original.SetPosition(pinned, -60, -90);
  original.Persist();
  const auto before = *original.Get(b);
  GraphLayout restored(graph);
  restored.Load();
  EXPECT_EQ(restored.Get(b)->x, before.x);
  EXPECT_EQ(restored.Get(b)->y, before.y);
  restored.RewireInput(b, 0, "late");
  EXPECT_EQ(restored.Get(b)->x, 0);
  auto late = restored.Append(Node("late", {"a"}));
  EXPECT_GT(restored.Get(b)->x, restored.Get(late)->x);
  EXPECT_EQ(restored.Get(pinned)->x, -60);
  EXPECT_EQ(restored.Get(pinned)->y, -90);
}

TEST(GraphLayout, NonrecursiveLoadReusesChildBoundsAndEscapedCaptures) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("inner", {"quoted \" capture\\name"});
  *graph.add_node() = container;
  GraphLayout recursive(graph);
  recursive.Load();
  recursive.Persist();
  const auto id = recursive.Order()[0];
  const auto expected = *recursive.Get(id);
  const auto child_before = attribute->g().node().size();
  GraphLayout shallow(graph);
  shallow.Load(false);
  EXPECT_TRUE(shallow.Children(id).empty());
  EXPECT_TRUE(shallow.Diagnostics().empty());
  EXPECT_EQ(shallow.Stats().scopes, 1u);
  EXPECT_EQ(shallow.Get(id)->width, expected.width);
  EXPECT_EQ(shallow.Get(id)->height, expected.height);
  ASSERT_EQ(shallow.Edges(id).size(), 1u);
  EXPECT_EQ(shallow.Edges(id)[0].value, "quoted \" capture\\name");
  EXPECT_EQ(graph.node()[0].attribute()[0].g().node().size(), child_before);
}

TEST(GraphLayout, NonrecursiveLoadRejectsUnboundedChildSummaryAndFallsBack) {
  for (const auto &summary : {"nan 0 0", "0 0 999999999", "0 0 2 \"missing\"", "0 0 0 trailing"}) {
    GraphProto graph;
    auto *node = graph.add_node();
    *node = Node("outer", {}, "If");
    auto *attribute = node->add_attribute();
    attribute->set_type(AttributeProto::GRAPH);
    auto *child = attribute->mutable_g();
    *child->add_node() = Node("inner", {"capture"});
    Meta(*child, GraphLayout::VersionKey, "1");
    Meta(*child, GraphLayout::ScopeKey, summary);
    GraphLayout layout(graph);
    layout.Load(false);
    auto id = layout.Order()[0];
    EXPECT_FALSE(layout.Diagnostics().empty());
    EXPECT_EQ(layout.Children(id).size(), 1u);
    ASSERT_EQ(layout.Edges(id).size(), 1u);
    EXPECT_EQ(layout.Edges(id)[0].value, "capture");
  }
}

TEST(GraphLayout, LoadingConstructorSupportsChildFirstAndEmptyGraphs) {
  GraphProto graph;
  auto *container = graph.add_node();
  *container = Node("outer", {}, "If");
  auto *attribute = container->add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  auto *child = attribute->mutable_g();
  GraphLayout(*child, false).Persist();
  EXPECT_EQ(child->metadata_props().size(), 3u);
  GraphLayout layout(graph, false);
  layout.Persist();
  auto id = layout.Order()[0];
  EXPECT_TRUE(layout.Children(id).empty());
  EXPECT_TRUE(layout.Diagnostics().empty());
  EXPECT_GT(layout.Get(id)->height, 0);
  EXPECT_EQ(child->node().size(), 0u);
}

TEST(GraphLayout, ExplicitDimensionsPersistAndIncrementallyUpdateConsumers) {
  GraphProto graph;
  GraphLayout layout(graph, true);
  auto source = layout.Append(Node("a", {"x"}));
  auto consumer = layout.Append(Node("b", {"a"}));
  auto unrelated = layout.Append(Node("unrelated"));
  const auto untouched = *layout.Get(unrelated);
  EXPECT_TRUE(layout.SetSize(source, 600, 180));
  EXPECT_EQ(layout.Get(source)->width, 600);
  EXPECT_EQ(layout.Get(source)->height, 180);
  EXPECT_GE(layout.Get(consumer)->x, 680);
  EXPECT_EQ(layout.Get(unrelated)->x, untouched.x);
  EXPECT_EQ(layout.Get(unrelated)->y, untouched.y);
  EXPECT_EQ(layout.Stats().nodes, 2u);
  layout.RewireInput(source, 0, "new_input");
  EXPECT_EQ(layout.Get(source)->width, 600);
  layout.Persist();
  GraphLayout restored(graph, true);
  restored.RewireInput(source, 0, "another_input");
  EXPECT_EQ(restored.Get(source)->width, 600);
  EXPECT_EQ(restored.Get(source)->height, 180);
  EXPECT_FALSE(restored.SetSize(source, 0, 10));
  EXPECT_FALSE(restored.SetSize(source, 10, -1));
  EXPECT_FALSE(restored.SetSize(source, 1e7, 10));
  EXPECT_FALSE(restored.SetSize(source, std::numeric_limits<double>::infinity(), 10));
  EXPECT_FALSE(restored.SetSize(source, 10, std::nan("")));
  EXPECT_FALSE(restored.SetSize(999999, 10, 10));
}

TEST(GraphLayout, NonrecursivePersistDoesNotWriteChildMetadata) {
  GraphProto graph;
  auto *node = graph.add_node();
  *node = Node("outer", {}, "If");
  auto *attribute = node->add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("inner");
  GraphLayout layout(graph, true);
  layout.Persist(false);
  bool version_written = false;
  for (const auto &entry : attribute->g().metadata_props())
    version_written |= entry.key() == GraphLayout::VersionKey;
  EXPECT_FALSE(version_written);
  layout.Persist(true);
  for (const auto &entry : attribute->g().metadata_props())
    version_written |= entry.key() == GraphLayout::VersionKey;
  EXPECT_TRUE(version_written);
}

TEST(GraphLayout, PersistedPinnedContainerTracksChildGrowthAndShrinkWithoutMoving) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("inner");
  *graph.add_node() = container;
  GraphLayout original(graph, true);
  const auto id = original.Order()[0];
  original.SetPosition(id, -200, -300);
  original.Persist();
  GraphLayout restored(graph, true);
  const double initial_width = restored.Get(id)->width;
  auto *child = restored.Children(id)[0];
  auto added = child->Append(Node("next", {"inner"}));
  EXPECT_GT(restored.Get(id)->width, initial_width);
  EXPECT_EQ(restored.Get(id)->x, -200);
  EXPECT_EQ(restored.Get(id)->y, -300);
  child->Erase(added);
  EXPECT_EQ(restored.Get(id)->width, initial_width);
  EXPECT_EQ(restored.Get(id)->x, -200);
  EXPECT_EQ(restored.Get(id)->y, -300);
}

TEST(GraphLayout, SetSizePreservesChildBoundMinimums) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("left");
  *attribute->mutable_g()->add_node() = Node("right");
  *graph.add_node() = container;
  GraphLayout layout(graph, true);
  const auto id = layout.Order()[0];
  auto *child = layout.Children(id)[0];
  ASSERT_TRUE(child->SetPosition(child->Order()[1], 400, 200));
  const auto child_bounds = child->Bounds();

  ASSERT_TRUE(layout.SetSize(id, 160, 64));
  EXPECT_GE(layout.Get(id)->width, child_bounds.right - child_bounds.left + 48);
  EXPECT_GE(layout.Get(id)->height, child_bounds.bottom - child_bounds.top + 96);

  layout.Persist();
  GraphLayout shallow(graph, false);
  EXPECT_EQ(shallow.Get(id)->width, layout.Get(id)->width);
  EXPECT_EQ(shallow.Get(id)->height, layout.Get(id)->height);
}

TEST(GraphLayout, OversizedMetadataIsRejectedBeforeAllocatingParserCopies) {
  GraphProto graph;
  Meta(graph, GraphLayout::VersionKey, "1");
  *graph.add_node() = Node("a");
  Meta(*graph.mutable_node(0), GraphLayout::PositionKey, std::string(2 * 1024 * 1024, '1'));
  Meta(*graph.mutable_node(0), GraphLayout::IdKey, std::string(2 * 1024 * 1024, '1'));
  GraphLayout layout(graph, true);
  EXPECT_FALSE(layout.Diagnostics().empty());
  EXPECT_EQ(layout.Order().size(), 1u);
  EXPECT_GE(layout.Get(layout.Order()[0])->width, 1);
}

TEST(GraphLayout, NewNodesUseOrderedPortBarycentersAndKeepStableAffectedPositions) {
  GraphProto graph;
  GraphLayout layout(graph, true);
  auto source_node = Node("first");
  source_node.add_output("second");
  auto source = layout.Append(source_node);
  layout.SetPosition(source, -400, -200);
  auto consumer = layout.Append(Node("consumer", {"second"}));
  EXPECT_EQ(layout.Get(consumer)->y, -182);
  const double y = layout.Get(consumer)->y;
  layout.SetSize(source, 800, 120);
  EXPECT_EQ(layout.Get(consumer)->y, y);
  EXPECT_GE(layout.Get(consumer)->x, 480);
  EXPECT_EQ(layout.Get(source)->x, -400);
  EXPECT_EQ(layout.Get(source)->y, -200);
}

TEST(GraphLayout, PersistedAutomaticConstraintViolationsAndCollisionsAreRepaired) {
  GraphProto graph;
  GraphLayout initial(graph, true);
  auto source = initial.Append(Node("a"));
  auto consumer = initial.Append(Node("b", {"a"}));
  auto independent = initial.Append(Node("c"));
  initial.SetPosition(independent, 0, 500);
  initial.SetSize(source, 900, 100);
  initial.SetPosition(source, 0, 0);
  initial.Persist();
  for (auto &entry : *graph.mutable_node(1)->mutable_metadata_props())
    if (entry.key() == GraphLayout::PositionKey)
      entry.set_value("0 0 160 66");
  for (auto &entry : *graph.mutable_node(2)->mutable_metadata_props())
    if (entry.key() == GraphLayout::PositionKey)
      entry.set_value("0 0 160 66");
  GraphLayout repaired(graph, true);
  EXPECT_GE(repaired.Get(consumer)->x, 980);
  EXPECT_GE(repaired.Get(independent)->y, 124);
  EXPECT_FALSE(repaired.Diagnostics().empty());
}

TEST(GraphLayout, FoldingConstantsDoesNotAllocateColumnsOrOccupiedRows) {
  GraphProto graph;
  GraphLayout layout(graph, true);
  std::vector<std::string> inputs;
  for (size_t i = 0; i < 30; ++i) {
    inputs.push_back("weight" + std::to_string(i));
    layout.Append(Node(inputs.back(), {}, "Constant"));
  }
  auto consumer = layout.Append(Node("result", inputs));
  EXPECT_EQ(layout.Get(consumer)->layer, 0u);
  EXPECT_EQ(layout.Get(consumer)->x, 0);
  EXPECT_EQ(layout.Get(consumer)->y, 0);
  for (const auto &edge : layout.Edges(consumer))
    EXPECT_TRUE(edge.folded);
}

TEST(GraphLayout, PersistedNextIdPreventsReuseAndRankResumesCorrectly) {
  GraphProto graph;
  GraphLayout initial(graph, true);
  auto a = initial.Append(Node("a"));
  auto b = initial.Append(Node("b", {"a"}));
  auto removed = initial.Append(Node("removed"));
  initial.Erase(removed);
  initial.Persist();
  GraphLayout resumed(graph, true);
  auto added = resumed.Append(Node("new", {"b"}));
  EXPECT_GT(added, removed);
  EXPECT_EQ(resumed.Get(a)->layer, 0u);
  EXPECT_EQ(resumed.Get(b)->layer, 1u);
  EXPECT_EQ(resumed.Get(added)->layer, 2u);
}

TEST(GraphLayout, StaleSignaturesInvalidateGeometryWhenSemanticPortsChange) {
  GraphProto graph;
  GraphLayout initial(graph, true);
  initial.Append(Node("a"));
  auto b = initial.Append(Node("b", {"a"}));
  initial.Persist();
  graph.mutable_node(1)->mutable_input()->Set(0, "external");
  GraphLayout resumed(graph, true);
  EXPECT_EQ(resumed.Get(b)->layer, 0u);
  EXPECT_EQ(resumed.Get(b)->x, 0);
  EXPECT_FALSE(resumed.Diagnostics().empty());
}

TEST(GraphLayout, DuplicateProducerDiagnosticsBoundAndSanitizeUntrustedNames) {
  GraphProto graph;
  std::string output = "\x1b\n";
  output.append(2 * 1024 * 1024, 'X');
  *graph.add_node() = Node(output);
  *graph.add_node() = Node(output);
  GraphLayout layout(graph, true);
  ASSERT_FALSE(layout.Diagnostics().empty());
  for (const auto &message : layout.Diagnostics()) {
    EXPECT_LT(message.size(), 256u);
    EXPECT_EQ(message.find('\x1b'), std::string::npos);
    EXPECT_EQ(message.find('\n'), std::string::npos);
  }
}

TEST(GraphLayout, StaleManualGeometryIsDiscardedButApiEditsKeepExplicitNegativePins) {
  GraphProto graph;
  GraphLayout initial(graph, true);
  auto a = initial.Append(Node("a"));
  initial.SetPosition(a, -800, -200);
  auto b = initial.Append(Node("b", {"a"}));
  initial.SetPosition(b, -400, -200);
  initial.RewireInput(b, 0, "temporary");
  EXPECT_EQ(initial.Get(b)->x, -400);
  EXPECT_EQ(initial.Get(b)->y, -200);
  initial.Persist();
  graph.mutable_node(1)->mutable_input()->Set(0, "a");
  GraphLayout resumed(graph, true);
  EXPECT_GE(resumed.Get(b)->x, resumed.Get(a)->x + resumed.Get(a)->width + 80);
  EXPECT_FALSE(resumed.Diagnostics().empty());
}

TEST(GraphLayout, NegativeChildCoordinatesUseFullCachedSpanAndKeepTheirOrigins) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  *attribute->mutable_g()->add_node() = Node("left");
  *attribute->mutable_g()->add_node() = Node("right");
  *graph.add_node() = container;
  GraphLayout layout(graph, true);
  const auto parent_id = layout.Order()[0];
  auto *child = layout.Children(parent_id)[0];
  const auto left = child->Order()[0], right = child->Order()[1];
  child->SetPosition(left, -500, -300);
  child->SetPosition(right, -200, -100);
  const auto bounds = child->Bounds();
  EXPECT_EQ(bounds.left, -500);
  EXPECT_EQ(bounds.top, -300);
  EXPECT_EQ(bounds.right, -40);
  EXPECT_EQ(bounds.bottom, -34);
  EXPECT_EQ(layout.Get(parent_id)->width, 460 + 48);
  EXPECT_EQ(layout.Get(parent_id)->height, 266 + 96);
  layout.Persist();
  GraphLayout shallow(graph, false);
  EXPECT_TRUE(shallow.Children(parent_id).empty());
  EXPECT_TRUE(shallow.Diagnostics().empty());
  EXPECT_EQ(shallow.Get(parent_id)->width, 460 + 48);
  EXPECT_EQ(shallow.Get(parent_id)->height, 266 + 96);
  EXPECT_EQ(child->Get(left)->x, -500);
  EXPECT_EQ(child->Get(right)->x, -200);
  child->Erase(left);
  EXPECT_EQ(child->Bounds().left, -200);
  EXPECT_EQ(child->Bounds().right, -40);
  EXPECT_EQ(layout.Get(parent_id)->width, 160 + 48);
  EXPECT_EQ(child->Get(right)->y, -100);
}

TEST(GraphLayout, AncestorPropagationStopsWhenBoundsAndExternalCapturesAreUnchanged) {
  GraphProto graph;
  auto outer = Node("outer", {}, "If");
  auto *outer_attribute = outer.add_attribute();
  outer_attribute->set_type(AttributeProto::GRAPH);
  auto middle = Node("middle", {}, "If");
  auto *middle_attribute = middle.add_attribute();
  middle_attribute->set_type(AttributeProto::GRAPH);
  *middle_attribute->mutable_g()->add_node() = Node("inner", {"capture", "capture"});
  *outer_attribute->mutable_g()->add_node() = middle;
  GraphLayout layout(graph, true);
  auto outer_id = layout.Append(outer);
  layout.Append(Node("consumer", {"outer"}));
  layout.Append(Node("unrelated"));
  EXPECT_EQ(layout.Stats().nodes, 1u);
  auto *child = layout.Children(outer_id)[0];
  auto *grandchild = child->Children(child->Order()[0])[0];
  auto inner = grandchild->Order()[0];
  const auto before = layout.Bounds();
  grandchild->RewireInput(inner, 0, "");
  EXPECT_EQ(grandchild->Edges(inner).size(), 1u);
  EXPECT_EQ(child->Stats().nodes, 1u);
  EXPECT_EQ(layout.Stats().nodes, 1u);
  EXPECT_EQ(layout.Bounds().right, before.right);
  EXPECT_EQ(layout.Bounds().bottom, before.bottom);
  grandchild->RewireInput(inner, 1, "different");
  EXPECT_EQ(layout.Stats().nodes, 2u);
  ASSERT_EQ(layout.Edges(outer_id).size(), 1u);
  EXPECT_EQ(layout.Edges(outer_id)[0].value, "different");
}

TEST(GraphLayout, ExplicitChildCapturesDifferFromFormalInputsInitializersAndLocalValues) {
  GraphProto graph;
  auto container = Node("outer", {}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  auto *child_graph = attribute->mutable_g();
  child_graph->add_input()->set_name("formal");
  child_graph->add_initializer()->set_name("weight");
  *child_graph->add_node() = Node("local");
  *child_graph->add_node() = Node("inner", {"captured", "formal", "weight", "local"});
  *graph.add_node() = container;
  GraphLayout layout(graph, true);
  auto container_id = layout.Order()[0];
  auto *child = layout.Children(container_id)[0];
  const auto &edges = child->Edges(child->Order()[1]);
  ASSERT_EQ(edges.size(), 4u);
  EXPECT_TRUE(edges[0].capture);
  EXPECT_FALSE(edges[1].capture);
  EXPECT_FALSE(edges[2].capture);
  EXPECT_FALSE(edges[3].capture);
  EXPECT_TRUE(edges[2].folded);
  ASSERT_EQ(layout.Edges(container_id).size(), 1u);
  EXPECT_TRUE(layout.Edges(container_id)[0].capture);
  EXPECT_EQ(layout.Edges(container_id)[0].value, "captured");
}

TEST(GraphLayout, CustomDomainConstantRetainsOrdinaryOperationLayout) {
  GraphProto graph;
  GraphLayout layout(graph, true);
  auto source = layout.Append(Node("source"));
  auto custom = Node("custom", {"source"}, "Constant");
  custom.set_domain("example.custom");
  auto custom_id = layout.Append(custom);
  auto consumer = layout.Append(Node("consumer", {"custom"}));
  EXPECT_FALSE(layout.Get(custom_id)->folded);
  EXPECT_EQ(layout.Get(custom_id)->layer, 1u);
  EXPECT_GT(layout.Get(custom_id)->x, layout.Get(source)->x);
  EXPECT_FALSE(layout.Edges(consumer)[0].folded);
  EXPECT_EQ(layout.Get(consumer)->layer, 2u);
  auto standard = Node("standard", {}, "Constant");
  standard.set_domain("ai.onnx");
  auto standard_id = layout.Append(standard);
  EXPECT_TRUE(layout.Get(standard_id)->folded);
  auto standard_consumer = layout.Append(Node("result", {"standard"}));
  EXPECT_TRUE(layout.Edges(standard_consumer)[0].folded);
  EXPECT_EQ(layout.Get(standard_consumer)->layer, 0u);
}

TEST(GraphLayout, ContainerDefaultHeightIncludesAllImplicitCapturePorts) {
  GraphProto graph;
  auto container = Node("outer", {"condition"}, "If");
  auto *attribute = container.add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  std::vector<std::string> captures;
  for (size_t i = 0; i < 60; ++i)
    captures.push_back("capture" + std::to_string(i));
  for (const auto &capture : captures)
    attribute->mutable_g()->add_output()->set_name(capture);
  *graph.add_node() = container;
  GraphLayout layout(graph, true);
  auto id = layout.Order()[0];
  const auto &box = *layout.Get(id);
  ASSERT_EQ(layout.Edges(id).size(), 61u);
  EXPECT_GE(box.height, 48 + 18 * 61);
  for (const auto &edge : layout.Edges(id)) {
    ASSERT_FALSE(edge.points.empty());
    EXPECT_LE(edge.points.back().y, box.y + box.height);
  }
  layout.Persist();
  GraphLayout shallow(graph, false);
  EXPECT_EQ(shallow.Get(id)->height, box.height);
}
