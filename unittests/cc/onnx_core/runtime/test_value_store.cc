// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::RuntimeValue;
using core::runtime::SubgraphSession;
using core::runtime::Tensor;
using core::runtime::ValueStore;

static_assert(std::is_same_v<decltype(std::declval<const RuntimeSession &>().value_store()),
                             const ValueStore &>);
static_assert(std::is_same_v<decltype(std::declval<const RuntimeContext &>().value_store()),
                             const ValueStore &>);
static_assert(std::is_same_v<decltype(std::declval<const RuntimeSession &>().value_store_owner()),
                             std::shared_ptr<const ValueStore>>);
static_assert(std::is_same_v<decltype(std::declval<const RuntimeContext &>().value_store_owner()),
                             std::shared_ptr<const ValueStore>>);

TEST(ValueStore, SessionsDoNotAdoptStoresFromReusedContexts) {
  ModelProto model;
  model.ref_graph().set_name("empty");
  FunctionProto function;
  function.set_name("empty");
  core::runtime::ExecutionPlan plan(model.graph());
  RuntimeSession first(plan);
  RuntimeSession second(model);
  RuntimeSession third(function);
  SubgraphSession fourth(model.graph());
  RuntimeContext context;
  const auto original_context_store = context.value_store_owner();
  const std::vector<RuntimeSession *> sessions{&first, &second, &third, &fourth};
  std::vector<std::shared_ptr<ValueStore>> owners;
  for (RuntimeSession *session : sessions) {
    auto owner = session->value_store_owner();
    ASSERT_NE(owner, nullptr);
    EXPECT_NE(owner, original_context_store);
    for (const auto &previous : owners)
      EXPECT_NE(owner, previous);
    owners.push_back(std::move(owner));
  }

  const uint32_t slot = first.value_store().Bind("prepared");
  Tensor tensor = first.value_store().AllocateTensor(TensorProto::FLOAT, {1}, sizeof(float));
  tensor.As<float>()[0] = 7.0f;
  first.value_store().Publish(slot, RuntimeValue(std::move(tensor)));
  for (size_t i = 0; i < sessions.size(); ++i) {
    sessions[i]->Run(context);
    EXPECT_EQ(context.value_store_owner(), owners[i]);
    EXPECT_EQ(sessions[i]->value_store_owner(), owners[i]);
  }
  first.Run(context);
  EXPECT_EQ(context.value_store_owner(), owners.front());
  EXPECT_FLOAT_EQ(first.value_store().Get(slot).tensor.As<float>()[0], 7.0f);
  EXPECT_EQ(first.prepared_bytes(), sizeof(float));
  EXPECT_EQ(second.prepared_bytes(), 0u);
  EXPECT_EQ(original_context_store->slot_count(), 0u);
  RuntimeContext replacement;
  first.Run(replacement);
  EXPECT_EQ(replacement.value_store_owner(), owners.front());
  EXPECT_FLOAT_EQ(first.value_store().Get(slot).tensor.As<float>()[0], 7.0f);
}

TEST(ValueStore, ExplicitOwnersSharePreparationsAcrossSessionConstructors) {
  ModelProto model;
  model.ref_graph().set_name("empty");
  FunctionProto function;
  function.set_name("empty");
  core::runtime::ExecutionPlan plan(model.graph());
  auto owner = std::make_shared<ValueStore>();
  RuntimeSession first(plan, {.value_store = owner});
  RuntimeSession second(model, {.value_store = owner});
  RuntimeSession third(function, {.value_store = owner});
  SubgraphSession fourth(model.graph(), nullptr, owner);

  const uint32_t slot = first.value_store().Bind("shared");
  Tensor tensor = first.value_store().AllocateTensor(TensorProto::FLOAT, {1}, sizeof(float));
  tensor.As<float>()[0] = 9.0f;
  first.value_store().Publish(slot, RuntimeValue(std::move(tensor)));
  RuntimeContext context;
  for (RuntimeSession *session : std::vector<RuntimeSession *>{&first, &second, &third, &fourth}) {
    EXPECT_EQ(session->value_store_owner(), owner);
    EXPECT_EQ(&session->value_store(), owner.get());
    session->Run(context);
    EXPECT_EQ(context.value_store_owner(), owner);
    EXPECT_FLOAT_EQ(session->value_store().Get(slot).tensor.As<float>()[0], 9.0f);
    EXPECT_EQ(session->prepared_bytes(), sizeof(float));
  }
}

TEST(ValueStore, PlanlessFunctionDispatchPreservesActiveStore) {
  ModelProto model;
  model.ref_graph().set_name("empty");
  FunctionProto *function = model.add_functions();
  function->set_domain("test.static_preparation");
  function->set_name("Empty");
  RuntimeContext context;
  core::runtime::RegisterModelFunctions(model, context);
  const auto owner = context.value_store_owner();
  const uint32_t slot = owner->Bind("existing");
  NodeProto node;
  node.set_domain(function->domain());
  node.set_op_type(function->name());
  for (size_t i = 0; i < 2; ++i) {
    core::runtime::RunNode(node, context);
    EXPECT_EQ(context.value_store_owner(), owner);
    EXPECT_EQ(context.value_store().Bind("existing"), slot);
    EXPECT_EQ(context.value_store().slot_count(), 1u);
  }
}

TEST(ValueStore, UsesIntegerSlotsAndAllocatorBackedRuntimeValues) {
  ValueStore store;
  const uint32_t first = store.Bind("first");
  const uint32_t shared = store.Bind("first");
  const uint32_t second = store.Bind("second");

  EXPECT_EQ(first, shared);
  EXPECT_NE(first, second);
  EXPECT_EQ(store.slot_count(), 2u);
  EXPECT_FALSE(store.IsReady(first));

  Tensor tensor = store.AllocateTensor(TensorProto::FLOAT, {2}, 2 * sizeof(float));
  const auto *tensor_allocator = tensor.allocation_owner();
  tensor.As<float>()[0] = 1.0f;
  tensor.As<float>()[1] = 2.0f;
  store.Publish(first, RuntimeValue(std::move(tensor)));

  ASSERT_TRUE(store.IsReady(first));
  const RuntimeValue &prepared = store.Get(first);
  ASSERT_EQ(prepared.kind, RuntimeValue::Kind::kTensor);
  EXPECT_EQ(prepared.tensor.allocation_owner(), tensor_allocator);
  EXPECT_FLOAT_EQ(prepared.tensor.As<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(prepared.tensor.As<float>()[1], 2.0f);
  EXPECT_EQ(store.prepared_bytes(), 2 * sizeof(float));
  EXPECT_THROW(store.Publish(first, RuntimeValue()), std::runtime_error);

  Tensor left = store.AllocateTensor(TensorProto::FLOAT, {1}, sizeof(float));
  Tensor right = store.AllocateTensor(TensorProto::FLOAT, {1}, sizeof(float));
  left.As<float>()[0] = 3.0f;
  right.As<float>()[0] = 4.0f;
  std::unordered_map<std::string, RuntimeValue> fields;
  fields.emplace("left", RuntimeValue(std::move(left)));
  fields.emplace("right", RuntimeValue(std::move(right)));
  store.Publish(second, RuntimeValue(std::move(fields)));

  const RuntimeValue &prepared_struct = store.Get(second);
  ASSERT_EQ(prepared_struct.kind, RuntimeValue::Kind::kStruct);
  EXPECT_FLOAT_EQ(prepared_struct.fields.at("left").tensor.As<float>()[0], 3.0f);
  EXPECT_FLOAT_EQ(prepared_struct.fields.at("right").tensor.As<float>()[0], 4.0f);
  EXPECT_EQ(store.prepared_bytes(), 4 * sizeof(float));
}

} // namespace
