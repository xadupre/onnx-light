// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_context.h"
#include <future>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

RuntimeContext EventContext() {
  return RuntimeContext(RuntimeContextOptions{.events_enabled = true});
}

void PutNumber(RuntimeContext &context, const char *name) {
  context.Put(name, Tensor::FromFloat(name, {1}, {1}));
}

} // namespace

TEST(RuntimeEvents, GenericRecordingStampsMetadataAndPreservesPayloadAndTimestamp) {
  SimpleRawBufferAllocator allocator(4);
  RuntimeContext context(RuntimeContextOptions{.allocator = &allocator, .events_enabled = true});
  const Tensor live = Tensor::FromFloat("", {1}, {1}, &allocator);
  context.set_current_node_index(9);
  context.set_current_subgraph(2, "body");
  context.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                       .timestamp_ns = 123,
                       .name = "cache",
                       .storage_allocated_bytes = 64});
  ASSERT_EQ(context.events().size(), 1u);
  const auto &event = context.events().front();
  EXPECT_EQ(event.action, RuntimeEventAction::kPersistentStorage);
  EXPECT_EQ(event.timestamp_ns, 123);
  EXPECT_EQ(event.name, "cache");
  EXPECT_EQ(event.node_index, 9);
  EXPECT_EQ(event.subgraph_node_index, 2);
  EXPECT_EQ(event.subgraph_attr_name, "body");
  EXPECT_EQ(event.allocated_bytes, sizeof(float));
  EXPECT_EQ(event.peak_bytes, sizeof(float));
  EXPECT_EQ(event.storage_allocated_bytes, 64u);

  context.RecordEvent({.action = RuntimeEventAction::kAdd, .kind = RuntimeEventKind::kInput});
  context.RecordEvent({.action = RuntimeEventAction::kAdd, .kind = RuntimeEventKind::kInitializer});
  context.RecordEvent({.action = RuntimeEventAction::kReplace});
  context.RecordEvent({.action = RuntimeEventAction::kRemove});
  context.RecordEvent(
      {.action = RuntimeEventAction::kRunNode, .timestamp_ns = 456, .duration_ns = 20});
  ASSERT_EQ(context.events().size(), 6u);
  EXPECT_EQ(context.events()[1].node_index, -1);
  EXPECT_GT(context.events()[1].timestamp_ns, 0);
  EXPECT_EQ(context.events()[2].node_index, -2);
  EXPECT_EQ(context.events()[3].node_index, 9);
  EXPECT_EQ(context.events()[4].node_index, -1);
  EXPECT_EQ(context.events()[5].node_index, 9);
  EXPECT_EQ(context.events()[5].timestamp_ns, 456);
  EXPECT_EQ(context.events()[5].duration_ns, 20);
}

TEST(RuntimeEvents, GenericRecordingDoesNothingWhenDisabled) {
  RuntimeContext context;
  context.events().push_back({.name = "existing"});
  for (auto action :
       {RuntimeEventAction::kAdd, RuntimeEventAction::kReplace, RuntimeEventAction::kRemove,
        RuntimeEventAction::kRunNode, RuntimeEventAction::kPersistentStorage})
    context.RecordEvent({.action = action, .name = "ignored"});
  ASSERT_EQ(context.events().size(), 1u);
  EXPECT_EQ(context.events().front().name, "existing");
}

TEST(RuntimeEvents, SharesOneLogAcrossNestedContextsAndCopies) {
  auto parent = EventContext();
  parent.set_current_node_index(7);
  auto child = parent.MakeSubgraphContext("body");
  auto function = child.MakeFunctionContext();
  auto copy = function;
  PutNumber(parent, "parent");
  PutNumber(child, "child");
  function.set_current_node_index(3);
  function.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                        .storage_allocations = 1,
                        .storage_allocated_bytes = 64});
  PutNumber(copy, "copy");
  NodeProto node;
  node.set_op_type("Identity");
  copy.RecordRunNodeEvent(node, "", "Identity", 10, 20);
  child.Remove("child");

  EXPECT_EQ(&parent.events(), &child.events());
  EXPECT_EQ(&parent.events(), &function.events());
  EXPECT_EQ(&parent.events(), &copy.events());
  ASSERT_EQ(parent.events().size(), 6u);
  EXPECT_EQ(parent.events()[0].name, "parent");
  EXPECT_EQ(parent.events()[1].name, "child");
  const auto &storage = parent.events()[2];
  EXPECT_EQ(storage.action, RuntimeEventAction::kPersistentStorage);
  EXPECT_EQ(storage.storage_allocated_bytes, 64u);
  EXPECT_EQ(storage.node_index, 3);
  EXPECT_EQ(storage.subgraph_node_index, 7);
  EXPECT_EQ(storage.subgraph_attr_name, "body");
  EXPECT_EQ(parent.events()[3].name, "copy");
  EXPECT_EQ(parent.events()[4].action, RuntimeEventAction::kRunNode);
  EXPECT_EQ(parent.events()[4].timestamp_ns, 10);
  EXPECT_EQ(parent.events()[4].duration_ns, 20);
  EXPECT_EQ(parent.events()[5].action, RuntimeEventAction::kRemove);
  EXPECT_EQ(parent.events()[5].node_index, -1);
  EXPECT_FALSE(parent.Has("copy"));
}

TEST(RuntimeEvents, PreservesEventsBeforeFailureWithoutScopeCleanup) {
  auto parent = EventContext();
  auto child = parent.MakeFunctionContext();
  EXPECT_THROW(
      {
        PutNumber(child, "before failure");
        child.RecordEvent(
            {.action = RuntimeEventAction::kPersistentStorage, .storage_prefix_copied_bytes = 16});
        ASSERT_EQ(parent.events().size(), 2u);
        throw std::runtime_error("kernel failure");
      },
      std::runtime_error);
  ASSERT_EQ(parent.events().size(), 2u);
  EXPECT_EQ(parent.events()[0].name, "before failure");
  EXPECT_EQ(parent.events()[1].storage_prefix_copied_bytes, 16u);
}

TEST(RuntimeEvents, KeepsLogAliveAfterParentDestructionAndMoves) {
  auto child = [] {
    auto parent = EventContext();
    PutNumber(parent, "parent");
    return parent.MakeFunctionContext();
  }();
  auto moved = std::move(child);
  PutNumber(moved, "child");
  ASSERT_EQ(moved.events().size(), 2u);
  EXPECT_EQ(moved.events()[0].name, "parent");
  EXPECT_EQ(moved.events()[1].name, "child");
}

TEST(RuntimeEvents, ClearsSharedLogWithoutClearingOtherContextsValues) {
  auto parent = EventContext();
  auto child = parent.MakeFunctionContext();
  PutNumber(parent, "parent");
  PutNumber(child, "child");
  child.ClearEvents();
  EXPECT_TRUE(parent.events().empty());
  EXPECT_TRUE(parent.Has("parent"));
  EXPECT_TRUE(child.Has("child"));
  child.RecordEvent({.action = RuntimeEventAction::kPersistentStorage, .storage_reuse_count = 1});
  ASSERT_EQ(parent.events().size(), 1u);
  parent.Clear();
  EXPECT_TRUE(child.events().empty());
  EXPECT_FALSE(parent.Has("parent"));
  EXPECT_TRUE(child.Has("child"));
  PutNumber(child, "next");
  ASSERT_EQ(parent.events().size(), 1u);
}

TEST(RuntimeEvents, LeavesIndependentContextsIsolatedAndDisabledContextsSilent) {
  auto first = EventContext();
  auto second = EventContext();
  PutNumber(first, "first");
  EXPECT_TRUE(second.events().empty());

  RuntimeContext disabled;
  auto child = disabled.MakeSubgraphContext("body");
  auto function = child.MakeFunctionContext();
  auto copy = disabled;
  EXPECT_EQ(&disabled.events(), &child.events());
  EXPECT_EQ(&disabled.events(), &function.events());
  EXPECT_EQ(&disabled.events(), &copy.events());
  for (auto *context : {&disabled, &child, &function}) {
    PutNumber(*context, "value");
    context->RecordEvent(
        {.action = RuntimeEventAction::kPersistentStorage, .storage_allocations = 1});
    NodeProto node;
    context->RecordRunNodeEvent(node, "", "Identity", 0, 1);
    context->Remove("value");
    EXPECT_FALSE(context->events_enabled());
    EXPECT_TRUE(context->events().empty());
  }
  disabled.events().push_back(RuntimeEvent{});
  EXPECT_EQ(function.events().size(), 1u);
  copy.ClearEvents();
  EXPECT_TRUE(disabled.events().empty());
  EXPECT_TRUE(child.events().empty());
}

TEST(RuntimeEvents, SerializesRecordingFromConcurrentChildren) {
  auto parent = EventContext();
  auto run = [](RuntimeContext child) {
    for (int i = 0; i < 100; ++i) {
      PutNumber(child, "value");
      child.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                         .storage_allocations = 1,
                         .storage_allocated_bytes = 4});
      NodeProto node;
      child.RecordRunNodeEvent(node, "", "Identity", i, 1);
      child.Remove("value");
    }
  };
  auto first = std::async(std::launch::async, run, parent.MakeFunctionContext());
  auto second = std::async(std::launch::async, run, parent.MakeSubgraphContext("body"));
  first.get();
  second.get();
  ASSERT_EQ(parent.events().size(), 800u);
  uint64_t allocations = 0, allocated_bytes = 0;
  size_t body_events = 0;
  for (const auto &event : parent.events()) {
    allocations += event.storage_allocations;
    allocated_bytes += event.storage_allocated_bytes;
    body_events += event.subgraph_attr_name == "body";
  }
  EXPECT_EQ(allocations, 200u);
  EXPECT_EQ(allocated_bytes, 800u);
  EXPECT_EQ(body_events, 400u);
}
