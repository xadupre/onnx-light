// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_core/runtime/quantization.h"
#include <future>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

uint64_t StorageTotal(const RuntimeEventLog &events, uint64_t RuntimeEvent::*field) {
  uint64_t result = 0;
  for (const auto &event : events)
    if (event.action == RuntimeEventAction::kPersistentStorage)
      result += event.*field;
  return result;
}

TypeProto FloatType(int64_t size = 1) {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(size);
  return type;
}

RuntimeValue Number(float value) {
  return RuntimeValue(Tensor::FromFloat("", {1}, {value})).Retain();
}

float Number(const RuntimeValue &value) { return value.tensor.AsFloat()[0]; }

TypeProto Structure(std::initializer_list<std::pair<std::string, TypeProto>> fields) {
  TypeProto type;
  auto *structure = type.mutable_struct_type()->mutable_structure();
  for (const auto &[name, field_type] : fields) {
    auto *field = structure->add_field();
    field->set_name(name);
    *field->mutable_type() = field_type;
  }
  return type;
}

void Bind(ModelProto &model, const std::string &input, const std::string &output) {
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name(input);
  binding->set_output_name(output);
}

ModelProto Model(bool structured = false) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("test.feedback");
  opset->set_version(1);
  auto *graph = model.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(structured ? "request" : "past");
  const TypeProto cache = Structure({{"keys", FloatType()}, {"values", FloatType()}});
  *input->mutable_type() =
      structured ? Structure({{"logits", FloatType()}, {"cache", cache}}) : FloatType();
  auto *tokens = graph->add_input();
  tokens->set_name("tokens");
  *tokens->mutable_type() = FloatType();
  auto *output = graph->add_output();
  output->set_name(structured ? "response" : "present");
  *output->mutable_type() =
      structured ? Structure({{"logits", FloatType()}, {"cache", cache}}) : FloatType();
  auto *node = graph->add_node();
  node->set_domain("test.feedback");
  node->set_op_type("Step");
  node->add_input(structured ? "request" : "past");
  node->add_input("tokens");
  node->add_output(structured ? "response" : "present");
  if (structured)
    Bind(model, "request", "response");
  else
    Bind(model, "past", "present");
  return model;
}

void RegisterStep(RuntimeContext &context) {
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        const float next = rt.Get("past").AsFloat()[0] + rt.Get("tokens").AsFloat()[0];
        rt.Put(node.output(0), Tensor::FromFloat("", {1}, {next}, rt.allocator()));
      });
}

void RegisterIdentity(RuntimeContext &context) {
  context.RegisterCustomKernel("", "Identity", [](const NodeProto &node, RuntimeContext &rt) {
    if (rt.Has(node.input(0)))
      rt.Put(node.output(0), rt.Get(node.input(0)).BorrowView());
    else
      rt.values()[node.output(0)] = rt.values().at(node.input(0)).BorrowView();
  });
}

RuntimeValue Cache(float key, float value) {
  return RuntimeValue(RuntimeValueMap{{"keys", Number(key)}, {"values", Number(value)}});
}

RuntimeValue Request(float key, float value) {
  return RuntimeValue(RuntimeValueMap{{"logits", Number(0)}, {"cache", Cache(key, value)}});
}

void RegisterStructuredStep(RuntimeContext &context, const uint8_t **expected = nullptr) {
  context.RegisterCustomKernel(
      "test.feedback", "Step", [expected](const NodeProto &node, RuntimeContext &rt) {
        const auto &request = rt.values().at(node.input(0));
        const auto &cache = request.fields.at("cache");
        if (expected != nullptr) {
          EXPECT_EQ(cache.fields.at("keys").tensor.bytes(), *expected);
        }
        const float token = rt.Get(node.input(1)).AsFloat()[0];
        rt.values()[node.output(0)] = RuntimeValue(
            RuntimeValueMap{{"logits", Number(token)},
                            {"cache", Cache(Number(cache.fields.at("keys")) + token,
                                            Number(cache.fields.at("values")) + token)}});
        if (expected != nullptr)
          *expected =
              rt.values().at(node.output(0)).fields.at("cache").fields.at("keys").tensor.bytes();
      });
}

} // namespace

TEST(PersistentValueState, PersistentStorageAccountingDoesNotDependOnAttention) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.events_enabled = true});
  bool fail = false;
  context.RegisterCustomKernel(
      "test.feedback", "Step", [&](const NodeProto &node, RuntimeContext &rt) {
        RuntimeContext child = rt.MakeFunctionContext();
        EXPECT_EQ(&context.events(), &rt.events());
        EXPECT_EQ(&context.events(), &child.events());
        const auto previous_allocations =
            StorageTotal(context.events(), &RuntimeEvent::storage_allocations);
        Tensor next = child.MakeOutputTensor(0, DataType::FLOAT, {1}, sizeof(float));
        next.AsFloat()[0] = rt.Get("past").AsFloat()[0] + rt.Get("tokens").AsFloat()[0];
        child.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                           .storage_allocations = 1,
                           .storage_allocated_bytes = sizeof(float)});
        EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations),
                  previous_allocations + 1);
        rt.Put(node.output(0), std::move(next));
        if (fail)
          throw std::invalid_argument("failure after reporting persistent storage work");
      });
  PersistentValueState state(model, {{"past", Number(2)}});
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(1)}}).at("present")), 3);
  fail = true;
  EXPECT_THROW(state.Run(context, {{"tokens", Number(1)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 3);
  EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations), 2u);
  state.Reset({{"past", Number(7)}});
  EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations), 2u);
  fail = false;
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(1)}}).at("present")), 8);
  const auto statistics = context.events();
  EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_allocations), 3u);
  EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_allocated_bytes), 3 * sizeof(float));
  EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_prefix_copied_bytes), 0u);
  EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_append_copied_bytes), 0u);
  EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_reuse_count), 0u);
  PersistentValueState other(model, {{"past", Number(0)}});
  other.Run(context, {{"tokens", Number(1)}});
  EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations), 4u);
  context.ClearEvents();
  EXPECT_TRUE(context.events().empty());
  EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations), 0u);
  EXPECT_EQ(Number(state.Values().at("past")), 8);
  other.Run(context, {{"tokens", Number(1)}});
  EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations), 1u);
}

TEST(PersistentValueState, GenericProducerComputesDirectlyIntoPersistentTail) {
  for (bool events_enabled : {false, true}) {
    SCOPED_TRACE(events_enabled);
    ModelProto model = Model();
    TypeProto dynamic;
    dynamic.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    dynamic.mutable_tensor_type()->mutable_shape()->add_dim();
    *model.mutable_graph()->mutable_input(0)->mutable_type() = dynamic;
    *model.mutable_graph()->mutable_output(0)->mutable_type() = dynamic;
    auto arena = IOArena::Create(4);
    SimpleRawBufferAllocator execution(8);
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.allocator = &execution,
                                                 .io_allocator = arena.get(),
                                                 .events_enabled = events_enabled});
    bool fail = false;
    context.RegisterCustomKernel(
        "test.feedback", "Step", [&](const NodeProto &node, RuntimeContext &rt) {
          const Tensor &past = rt.Get(node.input(0));
          const Shape shape{past.shape[0] + 1};
          EXPECT_FALSE(rt.ReservePersistentAppend(past, shape, 0, 1, 0));
          auto reservation = rt.ReservePersistentAppend(past, shape, 0, 0, 0);
          EXT_ENFORCE(reservation.has_value(), "Expected a bound contiguous reservation.");
          const auto tail = reservation->writable_bytes();
          EXT_ENFORCE(tail.size() == sizeof(float), "Expected one new float.");
          *reinterpret_cast<float *>(tail.data()) = rt.Get(node.input(1)).AsFloat()[0] * 2;
          if (fail)
            throw std::invalid_argument("failure after direct tail write");
          rt.Put(node.output(0),
                 rt.CommitPersistentAppend(0, std::move(*reservation), tail.size()));
        });
    RuntimeSessionOptions options;
    options.persistent_tensor_initial_capacity = 4;
    options.check_shapes = true;
    PersistentValueState state(model, {{"past", RuntimeValue(Tensor::FromFloat("", {0}, {}))}},
                               options);
    const uint8_t *previous = nullptr;
    for (int64_t step = 1; step <= 5; ++step) {
      if (step == 3) {
        fail = true;
        EXPECT_THROW(state.Run(context, {{"tokens", Number(9)}}), std::invalid_argument);
        EXPECT_EQ(state.Values().at("past").tensor.shape, Shape{2});
        EXPECT_FLOAT_EQ(state.Values().at("past").tensor.AsFloat()[1], 4);
        fail = false;
      }
      const auto output = state.Run(context, {{"tokens", Number(static_cast<float>(step))}});
      const Tensor &present = output.at("present").tensor;
      EXPECT_EQ(present.shape, Shape{step});
      for (int64_t i = 0; i < step; ++i)
        EXPECT_FLOAT_EQ(present.AsFloat()[i], static_cast<float>(2 * (i + 1)));
      if (step > 1 && step <= 4) {
        EXPECT_EQ(present.bytes(), previous);
      }
      if (step == 5) {
        EXPECT_NE(present.bytes(), previous);
      }
      previous = present.bytes();
    }
    const auto statistics = context.events();
    EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_allocations),
              events_enabled ? 2u : 0u);
    EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_allocated_bytes),
              events_enabled ? 12 * sizeof(float) : 0u);
    EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_prefix_copied_bytes),
              events_enabled ? 4 * sizeof(float) : 0u);
    EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_append_copied_bytes), 0u);
    EXPECT_EQ(StorageTotal(statistics, &RuntimeEvent::storage_reuse_count),
              events_enabled ? 4u : 0u);
    EXPECT_EQ(context.events().empty(), !events_enabled);
    EXPECT_EQ(execution.TotalAllocatedSize(), 0u);
    EXPECT_EQ(arena->leased_count(), 1u);
    state.Close();
    EXPECT_EQ(arena->leased_count(), 0u);
  }
}

TEST(PersistentValueState, RuntimeRejectsPersistentInputsWithoutExactlyOneUse) {
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    ModelProto model = Model();
    GraphProto &graph = *model.mutable_graph();
    if (mode == 0) {
      *graph.mutable_node(0)->mutable_input(0) = "tokens";
    } else if (mode == 1) {
      graph.mutable_node(0)->add_input("past");
    } else if (mode == 2) {
      NodeProto *observer = graph.add_node();
      observer->set_op_type("Shape");
      observer->add_input("past");
      observer->add_output("past_shape");
    } else {
      *graph.add_output() = graph.input(0);
    }
    EXPECT_THROW({ ExecutionPlan plan(graph); }, std::invalid_argument);
    EXPECT_THROW({ RuntimeSession session(model); }, std::invalid_argument);
    EXPECT_THROW(
        { PersistentValueState state(model, {{"past", Number(0)}}); }, std::invalid_argument);
    graph.clear_persistent_bindings();
    EXPECT_NO_THROW({ RuntimeSession session(model); });
  }
}

TEST(PersistentValueState, RuntimeRejectsSecondAppendAttemptEvenWhenCapacityIsDisabled) {
  for (size_t capacity : {0u, 4u}) {
    SCOPED_TRACE(capacity);
    ModelProto model = Model();
    auto *input_type = model.mutable_graph()->mutable_input(0)->mutable_type();
    input_type->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->clear_dim_value();
    *model.mutable_graph()->mutable_output(0)->mutable_type() = *input_type;
    RuntimeSessionOptions options;
    options.persistent_tensor_initial_capacity = capacity;
    PersistentValueState state(model, {{"past", Number(0)}}, options);
    RuntimeContext context(KernelContext(DefaultOpset(18)));
    bool first_attempt_finished = false;
    context.RegisterCustomKernel("test.feedback", "Step",
                                 [&](const NodeProto &node, RuntimeContext &rt) {
                                   const Tensor &past = rt.Get("past");
                                   auto first = rt.ReservePersistentAppend(past, {2}, 0, 0, 0);
                                   EXPECT_EQ(first.has_value(), capacity != 0);
                                   first_attempt_finished = true;
                                   rt.ReservePersistentAppend(past, {2}, 0, 0, 0);
                                   rt.Put(node.output(0), past.BorrowView());
                                 });
    EXPECT_THROW(state.Run(context, {{"tokens", Number(1)}}), std::invalid_argument);
    EXPECT_TRUE(first_attempt_finished);
  }
}

TEST(PersistentValueState, StorageEventUsesExistingActivationMetadataAndClearing) {
  SimpleRawBufferAllocator allocator(4);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &allocator, .events_enabled = true});
  const Tensor value = Tensor::FromFloat("", {1}, {1}, &allocator);
  context.set_current_node_index(7);
  context.set_current_subgraph(3, "body");
  context.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                       .storage_allocations = 1,
                       .storage_allocated_bytes = 64,
                       .storage_prefix_copied_bytes = 8,
                       .storage_append_copied_bytes = 4,
                       .storage_reuse_count = 2});
  ASSERT_EQ(context.events().size(), 1u);
  const auto &event = context.events().front();
  EXPECT_EQ(event.action, RuntimeEventAction::kPersistentStorage);
  EXPECT_STREQ(RuntimeEventActionName(event.action), "persistent_storage");
  EXPECT_EQ(event.node_index, 7);
  EXPECT_EQ(event.subgraph_node_index, 3);
  EXPECT_EQ(event.subgraph_attr_name, "body");
  EXPECT_GT(event.timestamp_ns, 0);
  EXPECT_EQ(event.value_count, 0);
  EXPECT_EQ(event.allocated_bytes, sizeof(float));
  EXPECT_EQ(event.storage_allocated_bytes, 64u);
  EXPECT_EQ(event.storage_prefix_copied_bytes, 8u);
  EXPECT_EQ(event.storage_append_copied_bytes, 4u);
  EXPECT_EQ(event.storage_reuse_count, 2u);
  EXPECT_NE(event.summary().find("storage_bytes=64"), std::string::npos);
  context.ClearEvents();
  EXPECT_TRUE(context.events().empty());
  RuntimeContext disabled(KernelContext(DefaultOpset(18)));
  disabled.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                        .storage_allocations = 1,
                        .storage_allocated_bytes = 64,
                        .storage_prefix_copied_bytes = 8,
                        .storage_append_copied_bytes = 4,
                        .storage_reuse_count = 2});
  EXPECT_TRUE(disabled.events().empty());
}

TEST(PersistentValueState, FunctionAndSubgraphStorageEventsRespectActivationAndSurviveFailure) {
  for (const auto &[subgraph, enabled] : {std::pair{false, false}, std::pair{false, true},
                                          std::pair{true, false}, std::pair{true, true}}) {
    SCOPED_TRACE(subgraph);
    SCOPED_TRACE(enabled);
    ModelProto model = Model();
    auto add_audit = [](NodeProto *node) {
      node->set_domain("test.feedback");
      node->set_op_type("Audit");
      node->add_input("past");
      node->add_input("tokens");
      node->add_output("present");
    };
    if (subgraph) {
      model.add_opset_import()->set_version(18);
      auto *node = model.mutable_graph()->mutable_node(0);
      node->set_domain("");
      node->set_op_type("If");
      node->clear_input();
      node->add_input("condition");
      auto *condition = model.mutable_graph()->add_initializer();
      condition->set_name("condition");
      condition->set_data_type(TensorProto::BOOL);
      condition->add_int32_data(1);
      for (const char *name : {"then_branch", "else_branch"}) {
        auto *attribute = node->add_attribute();
        attribute->set_name(name);
        attribute->set_type(AttributeProto::GRAPH);
        auto *branch = attribute->mutable_g();
        if (std::string(name) == "then_branch") {
          add_audit(branch->add_node());
        } else {
          auto *identity = branch->add_node();
          identity->set_op_type("Identity");
          identity->add_input("tokens");
          identity->add_output("present");
        }
        auto *output = branch->add_output();
        output->set_name("present");
        *output->mutable_type() = FloatType();
      }
    } else {
      auto *function = model.add_functions();
      function->set_domain("test.feedback");
      function->set_name("Step");
      function->add_input("past");
      function->add_input("tokens");
      function->add_output("present");
      auto *opset = function->add_opset_import();
      opset->set_domain("test.feedback");
      opset->set_version(1);
      add_audit(function->add_node());
    }
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.events_enabled = enabled});
    RegisterIdentity(context);
    bool fail = false;
    context.RegisterCustomKernel(
        "test.feedback", "Audit", [&](const NodeProto &node, RuntimeContext &rt) {
          Tensor result = rt.MakeOutputTensor(0, DataType::FLOAT, {1}, sizeof(float));
          result.AsFloat()[0] = rt.Get("past").AsFloat()[0] + rt.Get("tokens").AsFloat()[0];
          rt.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                          .storage_allocations = 1,
                          .storage_allocated_bytes = sizeof(float)});
          if (fail)
            throw std::invalid_argument("failure in nested audited kernel");
          rt.Put(node.output(0), std::move(result));
        });
    PersistentValueState state(model, {{"past", Number(2)}});
    EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(1)}}).at("present")), 3);
    fail = true;
    EXPECT_THROW(state.Run(context, {{"tokens", Number(1)}}), std::invalid_argument);
    EXPECT_EQ(Number(state.Values().at("past")), 3);
    EXPECT_EQ(StorageTotal(context.events(), &RuntimeEvent::storage_allocations),
              enabled ? 2u : 0u);
    EXPECT_EQ(context.events().empty(), !enabled);
    for (const auto &event : context.events()) {
      if (event.action != RuntimeEventAction::kPersistentStorage)
        continue;
      EXPECT_EQ(event.node_index, 0);
      EXPECT_EQ(event.subgraph_node_index, subgraph ? 0 : -1);
      EXPECT_EQ(event.subgraph_attr_name, subgraph ? "then_branch" : "");
    }
  }
}

TEST(PersistentValueState, FunctionDiagnosticMetadataIsCopiedOnlyWhenEventsAreEnabled) {
  for (bool enabled : {false, true}) {
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.events_enabled = enabled});
    const std::string attribute(64, 'x');
    context.set_current_subgraph(7, attribute);
    auto child = context.MakeFunctionContext();
    EXPECT_EQ(child.events_enabled(), enabled);
    EXPECT_EQ(child.current_subgraph_node_index(), enabled ? 7 : -1);
    EXPECT_EQ(child.current_subgraph_attr_name(), enabled ? attribute : "");
  }
}

TEST(PersistentValueState, WholeTensorMatchesManualLoopAndSharesReadOnlyViews) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  RuntimeValueMap initial{{"past", Number(2)}};
  const uint8_t *initial_data = initial.at("past").tensor.bytes();
  PersistentValueState state(model, initial);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), initial_data);
  RuntimeSession manual(model);
  float previous = 2;
  RuntimeValueMap earlier;
  for (float token : {1.f, 4.f, 2.f}) {
    context.Put("past", Number(previous).tensor);
    context.Put("tokens", Number(token).tensor);
    manual.Run(context);
    const auto output = state.Run(context, {{"tokens", Number(token)}});
    previous = context.Get("present").AsFloat()[0];
    EXPECT_EQ(Number(output.at("present")), previous);
    EXPECT_EQ(Number(state.Values().at("past")), previous);
    EXPECT_EQ(state.Values().at("past").tensor.bytes(), output.at("present").tensor.bytes());
    if (earlier.empty())
      earlier = output;
  }
  EXPECT_EQ(Number(earlier.at("present")), 3);
  auto snapshot = state.Values();
  EXPECT_EQ(snapshot.at("past").tensor.bytes(), state.Values().at("past").tensor.bytes());
  EXPECT_EQ(Number(state.Values().at("past")), 9);
  state.Reset({{"past", Number(5)}});
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(2)}}).at("present")), 7);
  state.Close();
  EXPECT_THROW(state.Values(), std::invalid_argument);
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_THROW(state.Reset({{"past", Number(1)}}), std::invalid_argument);
  EXPECT_NO_THROW(state.Close());
}

TEST(PersistentValueState, RetainsWholeStructureAndRequiresSeparateFreshTokens) {
  ModelProto model = Model(true);
  SimpleRawBufferAllocator allocator(20);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &allocator});
  std::weak_ptr<std::vector<float>> retained_logits;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    const auto &cache = request.fields.at("cache");
    float token = rt.Get("tokens").AsFloat()[0];
    auto logits = std::make_shared<std::vector<float>>(1, token);
    retained_logits = logits;
    RuntimeValue response(RuntimeValueMap{
        {"cache", Cache(Number(cache.fields.at("keys")) + token,
                        Number(cache.fields.at("values")) + 2 * token)},
        {"logits", RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                               reinterpret_cast<const uint8_t *>(logits->data()),
                                               sizeof(float), logits))}});
    rt.values()["response"] = std::move(response);
  });
  PersistentValueState state(model, {{"request", Request(0, 10)}});
  auto first = state.Run(context, {{"tokens", Number(2)}});
  EXPECT_FALSE(retained_logits.expired());
  auto first_logits = retained_logits;
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
  auto snapshot = state.Values();
  ASSERT_EQ(snapshot.size(), 1u);
  EXPECT_EQ(Number(snapshot.at("request").fields.at("cache").fields.at("keys")), 2);
  EXPECT_EQ(snapshot.at("request").fields.at("logits").tensor.bytes(),
            first.at("response").fields.at("logits").tensor.bytes());
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_THROW(state.Run(context, {{"request.cache.keys", Number(1)}, {"tokens", Number(3)}}),
               std::invalid_argument);
  auto second = state.Run(context, {{"tokens", Number(3)}});
  EXPECT_EQ(Number(second.at("response").fields.at("cache").fields.at("keys")), 5);
  EXPECT_EQ(Number(first.at("response").fields.at("cache").fields.at("keys")), 2);
  EXPECT_EQ(Number(first.at("response").fields.at("logits")), 2);
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
  first.clear();
  snapshot.clear();
  EXPECT_TRUE(first_logits.expired());
  second.clear();
  EXPECT_FALSE(retained_logits.expired());
  state.Close();
  EXPECT_TRUE(retained_logits.expired());
}

TEST(PersistentValueState, TransfersOwnedTensorOutputsWithoutCopying) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  const uint8_t *produced = nullptr;
  RuntimeValueMap initial{{"past", Number(1)}};
  const uint8_t *expected_input = initial.at("past").tensor.bytes();
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    EXPECT_EQ(rt.Get("past").bytes(), expected_input);
    Tensor output = Tensor::FromFloat("present", {1}, {rt.Get("past").AsFloat()[0] + 1});
    produced = output.bytes();
    expected_input = produced;
    rt.Put("present", std::move(output));
  });
  PersistentValueState state(model, initial);
  auto first = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(first.at("present").tensor.bytes(), produced);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), produced);
  auto second = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(second.at("present").tensor.bytes(), produced);
  EXPECT_EQ(Number(second.at("present")), 3);
  state.Close();
  EXPECT_EQ(Number(first.at("present")), 2);
  EXPECT_EQ(Number(second.at("present")), 3);
}

TEST(PersistentValueState, MovesOwnedInitialMapsWithoutMutatingConstViews) {
  ModelProto model = Model();
  RuntimeValueMap initial;
  initial.emplace("past", RuntimeValue(Tensor::FromFloat("", {1}, {2})));
  const Tensor &source = initial.at("past").tensor;
  const uint8_t *payload = source.bytes();
  auto view = source.BorrowView();
  EXPECT_FALSE(source.is_borrowed());
  EXPECT_EQ(source.data.data(), payload);
  EXPECT_FALSE(view.borrowed_owner());
  PersistentValueState state(model, std::move(initial));
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), payload);
  EXPECT_TRUE(state.Values().at("past").tensor.borrowed_owner());
  RuntimeValueMap reset;
  reset.emplace("past", RuntimeValue(Tensor::FromFloat("", {1}, {4})));
  const uint8_t *next = reset.at("past").tensor.bytes();
  state.Reset(std::move(reset));
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), next);
}

TEST(PersistentValueState, OnlySelectedOutputsBypassOrdinaryAllocationAndMaterialization) {
  ModelProto model = Model();
  auto *ordinary = model.mutable_graph()->add_output();
  ordinary->set_name("ordinary");
  *ordinary->mutable_type() = FloatType();
  model.mutable_graph()->mutable_node(0)->add_output("ordinary");
  SimpleRawBufferAllocator execution(8);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &execution});
  const uint8_t *selected = nullptr;
  const uint8_t *ordinary_source = nullptr;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    EXPECT_TRUE(rt.retains_output("present"));
    EXPECT_FALSE(rt.retains_output("ordinary"));
    EXPECT_FALSE(rt.MakeFunctionContext().retains_output("present"));
    Tensor result = Tensor::FromFloat("", {1}, {rt.Get("past").AsFloat()[0] + 1});
    selected = result.bytes();
    rt.Put("present", std::move(result), RuntimeEventKind::kOutput);
    ordinary_source = rt.Get("tokens").bytes();
    rt.Put("ordinary", rt.Get("tokens").BorrowView(), RuntimeEventKind::kOutput);
    EXPECT_EQ(rt.Get("ordinary").allocation_owner(), &execution);
    EXPECT_NE(rt.Get("ordinary").bytes(), ordinary_source);
  });
  PersistentValueState state(model, {{"past", Number(1)}});
  float token = 7;
  RuntimeValueMap feeds;
  feeds.emplace("tokens", RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                                      reinterpret_cast<const uint8_t *>(&token),
                                                      sizeof(float))));
  auto output = state.Run(context, feeds);
  EXPECT_EQ(output.at("present").tensor.bytes(), selected);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), selected);
  EXPECT_EQ(output.at("ordinary").tensor.allocation_owner(), &execution);
  EXPECT_EQ(Number(output.at("ordinary")), 7);
  EXPECT_EQ(execution.TotalAllocatedSize(), sizeof(float));
  output.clear();
  EXPECT_EQ(execution.TotalAllocatedSize(), 0u);
}

TEST(PersistentValueState, OrdinaryOwnerlessOutputIsMaterializedButSelectedBorrowIsRejected) {
  ModelProto model = Model();
  auto *ordinary = model.mutable_graph()->add_output();
  ordinary->set_name("ordinary");
  *ordinary->mutable_type() = FloatType();
  model.mutable_graph()->mutable_node(0)->add_output("ordinary");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  float external = 7;
  bool invalid_selected = false;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    auto borrow = [&] {
      return Tensor::Borrow("", DataType::FLOAT, {1}, reinterpret_cast<const uint8_t *>(&external),
                            sizeof(float));
    };
    rt.Put("ordinary", borrow());
    rt.Put("present", invalid_selected ? borrow() : Tensor::FromFloat("", {1}, {2}));
  });
  PersistentValueState state(model, {{"past", Number(1)}});
  auto output = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_FALSE(output.at("ordinary").tensor.is_borrowed());
  EXPECT_NE(output.at("ordinary").tensor.bytes(), reinterpret_cast<const uint8_t *>(&external));
  invalid_selected = true;
  const auto previous = state.Values();
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), previous.at("past").tensor.bytes());
}

TEST(PersistentValueState, FunctionAttributesUseOrdinaryBinding) {
  ModelProto model = Model();
  model.add_opset_import()->set_version(18);
  auto *function = model.add_functions();
  function->set_domain("test.feedback");
  function->set_name("Activate");
  function->add_input("x");
  function->add_output("y");
  function->add_attribute("slope");
  auto *opset = function->add_opset_import();
  opset->set_domain("test.feedback");
  opset->set_version(1);
  auto *body = function->add_node();
  body->set_domain("test.feedback");
  body->set_op_type("Scale");
  body->add_input("x");
  body->add_output("y");
  auto *reference = body->add_attribute();
  reference->set_name("alpha");
  reference->set_type(AttributeProto::FLOAT);
  reference->set_ref_attr_name("slope");
  auto *call = model.mutable_graph()->mutable_node(0);
  call->set_op_type("Activate");
  call->clear_input();
  call->add_input("past");
  auto *attribute = call->add_attribute();
  attribute->set_name("slope");
  attribute->set_type(AttributeProto::FLOAT);
  attribute->set_f(0.25f);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel(
      "test.feedback", "Scale", [](const NodeProto &node, RuntimeContext &rt) {
        EXPECT_EQ(node.attribute(0).name(), "alpha");
        EXPECT_EQ(node.attribute(0).f(), 0.25f);
        const float result = rt.Get(node.input(0)).AsFloat()[0] * node.attribute(0).f();
        rt.Put(node.output(0), Tensor::FromFloat("", {1}, {result}));
      });
  PersistentValueState state(model, {{"past", Number(-4)}});
  auto output = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(Number(output.at("present")), -1);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), output.at("present").tensor.bytes());
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(0)}}).at("present")), -0.25f);
}

TEST(PersistentValueState, TransfersOwnedStructuredOutputsWithoutCopying) {
  ModelProto model = Model(true);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  const uint8_t *keys = nullptr;
  const uint8_t *values = nullptr;
  const uint8_t *logits = nullptr;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    const auto &past = rt.values().at("request").fields.at("cache");
    RuntimeValue response(RuntimeValueMap{
        {"cache", Cache(Number(past.fields.at("keys")) + 1, Number(past.fields.at("values")) + 2)},
        {"logits", Number(10)}});
    keys = response.fields.at("cache").fields.at("keys").tensor.bytes();
    values = response.fields.at("cache").fields.at("values").tensor.bytes();
    logits = response.fields.at("logits").tensor.bytes();
    rt.values()["response"] = std::move(response);
  });
  PersistentValueState state(model, {{"request", Request(0, 0)}});
  auto first = state.Run(context, {{"tokens", Number(0)}});
  auto &response = first.at("response");
  EXPECT_EQ(response.fields.at("cache").fields.at("keys").tensor.bytes(), keys);
  EXPECT_EQ(response.fields.at("cache").fields.at("values").tensor.bytes(), values);
  EXPECT_EQ(response.fields.at("logits").tensor.bytes(), logits);
  EXPECT_EQ(state.Values().at("request").fields.at("cache").fields.at("keys").tensor.bytes(), keys);
  EXPECT_EQ(state.Values().at("request").fields.at("logits").tensor.bytes(), logits);
  auto second = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(Number(second.at("response").fields.at("cache").fields.at("keys")), 2);
  state.Close();
  EXPECT_EQ(Number(response.fields.at("logits")), 10);
}

TEST(PersistentValueState, ValidatesDestinationBeforeRetainingOutput) {
  ModelProto model = Model();
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  bool invalid = true;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    if (invalid)
      rt.Put("present", Tensor::FromFloat("", {2}, {3, 4}));
    else
      rt.Put("present", Tensor::FromFloat("", {1}, {rt.Get("past").AsFloat()[0] + 1}));
  });
  PersistentValueState state(model, {{"past", Number(1)}});
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 1);
  invalid = false;
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(0)}}).at("present")), 2);
}

TEST(PersistentValueState, RejectsInvalidDeclarationsAndInitialValues) {
  ModelProto model = Model();
  model.mutable_graph()->clear_persistent_bindings();
  EXPECT_THROW((PersistentValueState(model, {})), std::invalid_argument);
  Bind(model, "unknown", "present");
  EXPECT_THROW((PersistentValueState(model, {})), std::invalid_argument);
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "past", "unknown");
  EXPECT_THROW((PersistentValueState(model, {})), std::invalid_argument);
  model = Model();
  EXPECT_THROW((PersistentValueState(model, {})), std::invalid_argument);
  EXPECT_THROW(
      (PersistentValueState(model, {{"past", RuntimeValue(Tensor::FromInt64("", {1}, {1}))}})),
      std::invalid_argument);
  EXPECT_THROW(
      (PersistentValueState(model, {{"past", RuntimeValue(Tensor::FromFloat("", {2}, {1, 2}))}})),
      std::invalid_argument);
  ModelProto wrong = Model();
  wrong.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT64);
  EXPECT_THROW((PersistentValueState(wrong, {{"past", Number(1)}})), std::invalid_argument);
  wrong = Model();
  *wrong.mutable_graph()->mutable_output(0)->mutable_type() = FloatType(2);
  EXPECT_THROW((PersistentValueState(wrong, {{"past", Number(1)}})), std::invalid_argument);
  ModelProto structured = Model(true);
  Bind(structured, "request.cache.keys", "response.cache.keys");
  EXPECT_THROW((PersistentValueState(structured, {})), std::invalid_argument);
}

TEST(PersistentValueState, DeclarationValidationAgreesWithGraphVerification) {
  for (bool input : {false, true}) {
    ModelProto model = Model(true);
    auto *binding = model.mutable_graph()->mutable_persistent_bindings(0);
    if (input)
      binding->set_input_name("request.cache");
    else
      binding->set_output_name("response.cache");
    EXPECT_THROW(VerifyPersistentBindings(nullptr, model.graph()), std::invalid_argument);
    EXPECT_THROW((PersistentValueState(model, {{"request", Request(1, 2)}})),
                 std::invalid_argument);
  }
  for (bool duplicate_input : {false, true}) {
    ModelProto model = Model();
    auto *output = model.mutable_graph()->add_output();
    output->set_name("other");
    *output->mutable_type() = FloatType();
    Bind(model, duplicate_input ? "past" : "tokens", duplicate_input ? "other" : "present");
    EXPECT_THROW(VerifyPersistentBindings(nullptr, model.graph()), std::invalid_argument);
    EXPECT_THROW((PersistentValueState(model, {{"past", Number(1)}, {"tokens", Number(2)}})),
                 std::invalid_argument);
  }
}

TEST(PersistentValueState, FailsTransactionallyAndKeepsRequestsIndependent) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  int mode = 0;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    const float next = rt.Get("past").AsFloat()[0] + 1;
    if (mode == 1)
      throw std::runtime_error("kernel failure before publishing output");
    if (mode == 2)
      rt.Set("present", Tensor::FromInt64("", {1}, {1}));
    else if (mode == 3)
      rt.Set("present", Tensor::FromFloat("", {2}, {1, 2}));
    else if (mode != 4)
      rt.Set("present", Tensor::FromFloat("", {1}, {next}));
  });
  PersistentValueState a(model, {{"past", Number(0)}});
  PersistentValueState b(model, {{"past", Number(10)}});
  const uint8_t *retained = a.Values().at("past").tensor.bytes();
  for (int failure : {1, 2, 3, 4}) {
    mode = failure;
    EXPECT_ANY_THROW(a.Run(context, {{"tokens", Number(0)}}));
    EXPECT_EQ(Number(a.Values().at("past")), 0);
    EXPECT_EQ(a.Values().at("past").tensor.bytes(), retained);
    EXPECT_EQ(Number(b.Values().at("past")), 10);
  }
  EXPECT_THROW(a.Reset({}), std::invalid_argument);
  EXPECT_EQ(Number(a.Values().at("past")), 0);
  mode = 0;
  EXPECT_EQ(Number(a.Run(context, {{"tokens", Number(0)}}).at("present")), 1);
  EXPECT_EQ(Number(b.Run(context, {{"tokens", Number(0)}}).at("present")), 11);
}

TEST(PersistentValueState, CancellationAndConcurrentOperationsDoNotPublish) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  std::promise<void> entered;
  std::promise<void> resume;
  auto released = resume.get_future();
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    entered.set_value();
    released.wait();
    rt.Set("present", Number(100).tensor);
  });
  PersistentValueState state(model, {{"past", Number(1)}});
  TaskCompletion completion(TaskId{1});
  const uint8_t *retained = state.Values().at("past").tensor.bytes();
  auto running = std::async(
      std::launch::async, [&] { return state.Run(context, {{"tokens", Number(0)}}, &completion); });
  entered.get_future().wait();
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_THROW(state.Reset({{"past", Number(3)}}), std::invalid_argument);
  EXPECT_THROW(state.Close(), std::invalid_argument);
  EXPECT_THROW(state.Values(), std::invalid_argument);
  completion.Cancel();
  resume.set_value();
  EXPECT_THROW(running.get(), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 1);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), retained);
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}, &completion), std::invalid_argument);
}

TEST(PersistentValueState, PublishesThroughTaskCompletion) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  PersistentValueState state(model, {{"past", Number(1)}});
  TaskCompletion completion(TaskId{1});
  state.Run(context, {{"tokens", Number(2)}}, &completion);
  EXPECT_EQ(completion.status(), TaskStatus::kSucceeded);
  EXPECT_EQ(Number(state.Values().at("past")), 3);
}

TEST(PersistentValueState, RejectsReentrantOperationsFromTheActiveKernel) {
  ModelProto model = Model();
  PersistentValueState state(model, {{"past", Number(1)}});
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
    EXPECT_THROW(state.Reset({{"past", Number(3)}}), std::invalid_argument);
    EXPECT_THROW(state.Close(), std::invalid_argument);
    EXPECT_THROW(state.Values(), std::invalid_argument);
    rt.Put("present", Number(2).tensor);
  });
  state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(Number(state.Values().at("past")), 2);
}

TEST(PersistentValueState, MultipleWholeInputsCommitTogether) {
  ModelProto model = Model();
  auto *output = model.mutable_graph()->add_output();
  output->set_name("next_tokens");
  *output->mutable_type() = FloatType();
  model.mutable_graph()->mutable_node(0)->add_output("next_tokens");
  Bind(model, "tokens", "next_tokens");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  bool invalid = true;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    rt.Put("present", Number(30).tensor);
    if (invalid)
      rt.Put("next_tokens", Tensor::FromInt64("", {1}, {40}));
    else
      rt.Put("next_tokens", Number(40).tensor);
  });
  PersistentValueState state(model, {{"past", Number(3)}, {"tokens", Number(4)}});
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 3);
  EXPECT_EQ(Number(state.Values().at("tokens")), 4);
  invalid = false;
  state.Run(context, {});
  EXPECT_EQ(Number(state.Values().at("past")), 30);
  EXPECT_EQ(Number(state.Values().at("tokens")), 40);
}

TEST(PersistentValueState, RetainsInitialOwnerAndRejectsUnleasedExecutionArena) {
  ModelProto model = Model();
  SimpleRawBufferAllocator allocator(10);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &allocator});
  RegisterStep(context);
  auto data = std::make_shared<std::vector<float>>(1, 5.0f);
  std::weak_ptr<std::vector<float>> owner = data;
  RuntimeValueMap initial{
      {"past", RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                           reinterpret_cast<const uint8_t *>(data->data()),
                                           sizeof(float), data))}};
  const uint8_t *original = initial.at("past").tensor.bytes();
  PersistentValueState state(model, initial);
  initial.clear();
  data.reset();
  EXPECT_FALSE(owner.expired());
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), original);
  EXPECT_THROW(state.Run(context, {{"tokens", Number(2)}}), std::invalid_argument);
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), original);
  RuntimeContext different(KernelContext(DefaultOpset(18)));
  RegisterStep(different);
  EXPECT_THROW(state.Run(different, {{"tokens", Number(1)}}), std::invalid_argument);
  state.Close();
  EXPECT_TRUE(owner.expired());
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
}

TEST(PersistentValueState, ValidatesSharedSymbolsAndMalformedTensorStorage) {
  ModelProto model = Model();
  for (auto &input : *model.mutable_graph()->mutable_input()) {
    auto *dimension = input.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0);
    dimension->clear_dim_value();
    dimension->set_dim_param("batch");
  }
  auto *output_dim = model.mutable_graph()
                         ->mutable_output(0)
                         ->mutable_type()
                         ->mutable_tensor_type()
                         ->mutable_shape()
                         ->mutable_dim(0);
  output_dim->clear_dim_value();
  output_dim->set_dim_param("batch");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  PersistentValueState state(model, {{"past", Number(1)}});
  EXPECT_THROW(state.Run(context, {{"tokens", RuntimeValue(Tensor::FromFloat("", {2}, {1, 2}))}}),
               std::invalid_argument);
  RuntimeValue malformed = Number(1);
  malformed.tensor.shape = {2};
  EXPECT_THROW(state.Run(context, {{"tokens", malformed}}), std::invalid_argument);
  malformed.tensor.shape = {-1};
  EXPECT_THROW(state.Run(context, {{"tokens", malformed}}), std::invalid_argument);
  EXPECT_THROW(state.Run(context, {{"unexpected", Number(1)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 1);
}

TEST(PersistentValueState, ExactDottedGraphNamesAreNotOverlappingFields) {
  ModelProto model = Model();
  model.mutable_graph()->mutable_input(1)->set_name("past.tokens");
  auto *second = model.mutable_graph()->add_output();
  second->set_name("present.tokens");
  *second->mutable_type() = FloatType();
  auto *node = model.mutable_graph()->mutable_node(0);
  node->clear_input();
  node->add_input("past");
  node->add_input("past.tokens");
  node->add_output("present.tokens");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const float past = rt.Get("past").AsFloat()[0];
    const float token = rt.Get("past.tokens").AsFloat()[0];
    rt.Put("present", Number(past + token).tensor);
    rt.Put("present.tokens", Number(token + 1).tensor);
  });
  PersistentValueState one(model, {{"past", Number(1)}});
  EXPECT_THROW(one.Run(context, {{R"(past\.tokens)", Number(2)}}), std::invalid_argument);
  EXPECT_EQ(Number(one.Run(context, {{"past.tokens", Number(2)}}).at("present")), 3);
  one.Close();
  ModelProto both_model = model;
  Bind(both_model, "past.tokens", "present.tokens");
  PersistentValueState both(both_model, {{"past", Number(1)}, {"past.tokens", Number(2)}});
  both.Run(context, {});
  EXPECT_EQ(Number(both.Values().at("past")), 3);
  EXPECT_EQ(Number(both.Values().at("past.tokens")), 3);
}

TEST(PersistentValueState, StructuredValuesCrossModelLocalFunctions) {
  ModelProto model = Model(true);
  auto *function = model.add_functions();
  function->set_domain("test.feedback");
  function->set_name("WrappedStep");
  function->add_input("inner_request");
  function->add_input("inner_tokens");
  function->add_output("inner_response");
  auto *opset = function->add_opset_import();
  opset->set_domain("test.feedback");
  opset->set_version(1);
  auto *node = function->add_node();
  node->set_domain("test.feedback");
  node->set_op_type("Step");
  node->add_input("inner_request");
  node->add_input("inner_tokens");
  node->add_output("inner_response");
  model.mutable_graph()->mutable_node(0)->set_op_type("WrappedStep");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RuntimeValueMap initial{{"request", Request(1, 2)}};
  const uint8_t *expected =
      initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes();
  RegisterStructuredStep(context, &expected);
  PersistentValueState state(model, initial);
  state.Run(context, {{"tokens", Number(3)}});
  state.Run(context, {{"tokens", Number(4)}});
  EXPECT_EQ(Number(state.Values().at("request").fields.at("cache").fields.at("keys")), 8);
  EXPECT_EQ(Number(state.Values().at("request").fields.at("cache").fields.at("values")), 9);
  EXPECT_EQ(state.Values().at("request").fields.at("cache").fields.at("keys").tensor.bytes(),
            expected);
}

TEST(PersistentValueState, StructuredValuesCrossIfBranches) {
  ModelProto model = Model(true);
  auto *unused_initializer = model.mutable_graph()->add_initializer();
  unused_initializer->set_name("unused_initializer");
  unused_initializer->set_data_type(DataType::FLOAT);
  unused_initializer->add_dims(1);
  unused_initializer->add_float_data(4);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);
  NodeProto step = model.graph().node(0);
  *step.mutable_output(0) = "next_response";
  GraphProto branch;
  auto *identity = branch.add_node();
  identity->set_op_type("Identity");
  identity->add_input("next_response");
  identity->add_output("response");
  *branch.add_output() = model.graph().output(0);
  auto *cond = model.mutable_graph()->add_input();
  cond->set_name("cond");
  cond->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::BOOL);
  cond->mutable_type()->mutable_tensor_type()->mutable_shape();
  model.mutable_graph()->clear_node();
  *model.mutable_graph()->add_node() = step;
  auto *node = model.mutable_graph()->add_node();
  node->set_op_type("If");
  node->add_input("cond");
  node->add_output("response");
  for (const auto *name : {"then_branch", "else_branch"}) {
    auto *attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::GRAPH);
    *attribute->mutable_g() = branch;
  }
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RuntimeValueMap initial{{"request", Request(1, 2)}};
  const uint8_t *expected =
      initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes();
  RegisterStructuredStep(context, &expected);
  RegisterIdentity(context);
  PersistentValueState state(model, initial);
  for (uint8_t condition : {1, 0})
    state.Run(context, {{"tokens", Number(3)},
                        {"cond", RuntimeValue(Tensor::FromBool("", {}, {condition}))}});
  EXPECT_EQ(Number(state.Values().at("request").fields.at("cache").fields.at("keys")), 7);
  EXPECT_EQ(Number(state.Values().at("request").fields.at("cache").fields.at("values")), 8);
  EXPECT_EQ(state.Values().at("request").fields.at("cache").fields.at("keys").tensor.bytes(),
            expected);
}

TEST(PersistentValueState, SelectedIfInitializerRetainsItsModelStorage) {
  ModelProto model = Model();
  model.add_opset_import()->set_version(18);
  auto *condition = model.mutable_graph()->add_input();
  condition->set_name("condition");
  condition->mutable_type()->mutable_tensor_type()->set_elem_type(DataType::BOOL);
  condition->mutable_type()->mutable_tensor_type()->mutable_shape();
  GraphProto branch;
  auto *constant = branch.add_initializer();
  constant->set_name("constant");
  constant->set_data_type(DataType::FLOAT);
  constant->add_dims(1);
  constant->add_float_data(7);
  auto *result = branch.add_output();
  result->set_name("constant");
  *result->mutable_type() = FloatType();
  model.mutable_graph()->clear_node();
  auto *consume = model.mutable_graph()->add_node();
  consume->set_op_type("Identity");
  consume->add_input("past");
  consume->add_output("previous");
  auto *node = model.mutable_graph()->add_node();
  node->set_op_type("If");
  node->add_input("condition");
  node->add_output("present");
  for (const auto *name : {"then_branch", "else_branch"}) {
    auto *attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::GRAPH);
    *attribute->mutable_g() = branch;
  }
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterIdentity(context);
  PersistentValueState state(model, {{"past", Number(0)}});
  RuntimeValueMap feeds{{"tokens", Number(0)},
                        {"condition", RuntimeValue(Tensor::FromBool("", {}, {1}))}};
  auto first = state.Run(context, feeds);
  auto second = state.Run(context, feeds);
  EXPECT_EQ(first.at("present").tensor.bytes(), second.at("present").tensor.bytes());
  state.Close();
  EXPECT_EQ(Number(first.at("present")), 7);
  EXPECT_EQ(Number(second.at("present")), 7);
}

TEST(PersistentValueState, RejectsExcessiveRuntimeValueNesting) {
  RuntimeValue value = Number(1);
  for (size_t i = 0; i <= RuntimeValue::kMaxDepth; ++i) {
    RuntimeValue parent;
    parent.fields.emplace("child", std::move(value));
    value = std::move(parent);
  }
  EXPECT_THROW(value.DeepCopy(), std::invalid_argument);
  EXPECT_THROW(value.BorrowView(), std::invalid_argument);
}

TEST(PersistentValueState, SequenceViewsCopiesRetentionAndDepth) {
  RuntimeValue value(
      std::vector<RuntimeValue>{RuntimeValue(RuntimeValueMap{{"value", Number(3)}}), Number(4)});
  const auto *bytes = value.elements[0].fields.at("value").tensor.bytes();
  const auto borrowed = value.BorrowView();
  const auto copied = value.DeepCopy();
  EXPECT_EQ(borrowed.kind, RuntimeValue::Kind::kSequence);
  EXPECT_EQ(borrowed.elements[0].fields.at("value").tensor.bytes(), bytes);
  EXPECT_NE(copied.elements[0].fields.at("value").tensor.bytes(), bytes);
  PersistentValue retained(std::move(value));
  const auto view = retained.BorrowView();
  EXPECT_EQ(view.elements[0].fields.at("value").tensor.bytes(), bytes);
  EXPECT_EQ(Number(view.elements[1]), 4);
  RuntimeValue nested = Number(1);
  for (size_t i = 0; i <= RuntimeValue::kMaxDepth; ++i) {
    RuntimeValue parent(std::vector<RuntimeValue>{});
    parent.elements.push_back(std::move(nested));
    nested = std::move(parent);
  }
  EXPECT_THROW(nested.BorrowView(), std::invalid_argument);
  EXPECT_THROW(nested.DeepCopy(), std::invalid_argument);
  EXPECT_THROW(std::move(nested).Retain(), std::invalid_argument);
  EXPECT_THROW(PersistentValue(std::move(nested)), std::invalid_argument);
}

namespace {

EncodedValueProto AffineBlock(int64_t length = 2) {
  EncodedValueProto encoded;
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : {int64_t{1}, int64_t{1}, length, int64_t{1}})
    logical->mutable_shape()->add_dim()->set_dim_value(dim);
  auto *affine = encoded.mutable_affine();
  affine->set_storage_type(TensorProto::INT8);
  affine->mutable_scale()->set_data_type(TensorProto::FLOAT);
  affine->mutable_scale()->add_float_data(0.5f);
  affine->mutable_zero_point()->set_data_type(TensorProto::INT8);
  affine->mutable_zero_point()->add_int32_data(0);
  *encoded.mutable_raw_data() = std::vector<uint8_t>(static_cast<size_t>(length), 2);
  return encoded;
}

TypeProto BlockSequenceType() {
  TypeProto scalar;
  scalar.mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  scalar.mutable_tensor_type()->mutable_shape();
  TypeProto tensor = AffineBlock().logical_type();
  tensor.mutable_tensor_type()->mutable_shape()->mutable_dim(2)->clear_dim_value();
  TypeProto type;
  *type.mutable_sequence_type()->mutable_elem_type() =
      Structure({{"start", scalar}, {"length", scalar}, {"key", tensor}, {"value", tensor}});
  return type;
}

RuntimeValue Block(bool encoded, int64_t length = 2) {
  auto leaf = [=]() {
    return encoded ? RuntimeValue(AffineBlock(length))
                   : RuntimeValue(
                         Tensor::FromFloat("", {1, 1, length, 1}, std::vector<float>(length, 1)));
  };
  return RuntimeValue(RuntimeValueMap{{"start", RuntimeValue(Tensor::FromInt64("", {}, {0}))},
                                      {"length", RuntimeValue(Tensor::FromInt64("", {}, {length}))},
                                      {"key", leaf()},
                                      {"value", leaf()}});
}

} // namespace

TEST(PersistentValueState, TypedBlockSequencesMixDenseAndAffineWithoutMaterializing) {
  ModelProto model = Model();
  const auto type = BlockSequenceType();
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  RuntimeValue blocks(std::vector<RuntimeValue>{Block(false), Block(true, 3)});
  blocks = std::move(blocks).Retain();
  const auto *dense = blocks.elements[0].fields.at("key").tensor.bytes();
  const auto *encoded = blocks.elements[1].fields.at("key").Encoded().raw_data().data();
  PersistentValueState state(model, {{"past", blocks}});
  RuntimeContext context;
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        rt.values()[node.output(0)] = rt.values().at(node.input(0)).BorrowView();
      });
  const auto output = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(output.at("present").elements[0].fields.at("key").tensor.bytes(), dense);
  EXPECT_EQ(output.at("present").elements[1].fields.at("key").Encoded().raw_data().data(), encoded);
  state.Reset({{"past", RuntimeValue(std::vector<RuntimeValue>{})}});
  EXPECT_TRUE(state.Values().at("past").elements.empty());
  EXPECT_TRUE(state.Run(context, {{"tokens", Number(0)}}).at("present").elements.empty());
  state.Close();
  EXPECT_EQ(output.at("present").elements[1].fields.at("key").Encoded().raw_data()[0], 2);

  blocks.elements[1].fields.at("length") = Number(1);
  EXPECT_THROW((PersistentValueState(model, {{"past", blocks}})), std::invalid_argument);
  EXPECT_THROW((PersistentValueState(model, {{"past", Block(false)}})), std::invalid_argument);
}

TEST(PersistentValueState, TypedSequencesRetainPortableAndSharedQuantizationOwners) {
  RuntimeValue retained;
  std::weak_ptr<const QuantizationParameterCatalogue> weak;
  {
    auto model = Model();
    *model.mutable_graph()->mutable_input(0)->mutable_type() = BlockSequenceType();
    *model.mutable_graph()->mutable_output(0)->mutable_type() = BlockSequenceType();
    const auto source = Tensor::FromFloat("", {1, 1, 2, 1}, {1, -2});
    const auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 2, 2);
    auto full = QuantizeTensor(source, plan);
    auto *graph = model.mutable_graph();
    auto *annotation = graph->add_quantization_annotation();
    annotation->set_tensor_name("onnx_light.quantization.parameters:common");
    auto descriptor = [&](const char *name, const std::string &bytes) {
      auto *tensor = graph->add_initializer();
      tensor->set_name(name);
      tensor->set_data_type(TensorProto::UINT8);
      tensor->add_dims(bytes.size());
      tensor->set_raw_data(bytes);
    };
    descriptor("storage_type", full.Encoded().struct_type().SerializeAsString());
    descriptor("logical_type", full.Encoded().logical_type().SerializeAsString());
    auto *scale = graph->add_initializer();
    scale->set_name("scales");
    scale->set_data_type(TensorProto::FLOAT);
    scale->add_float_data(1);
    for (const auto &name : {"storage_type", "logical_type", "scales"}) {
      auto *mapping = annotation->add_quant_parameter_tensor_names();
      mapping->set_key(name);
      mapping->set_value(name);
    }
    VerifyModel(model);
    auto parameters = QuantizationParameterCatalogue::Build(model);
    weak = parameters;
    auto shared = QuantizeTensorShared(source, full.Encoded().struct_type(), "common", parameters);
    EXPECT_THROW(RuntimeValue(shared.Encoded()).Retain(), std::invalid_argument);
    auto wrong = shared.Encoded();
    wrong.set_parameter_ref("missing");
    RuntimeValue bad(std::move(wrong));
    bad.quantization_parameters = parameters;
    EXPECT_THROW(std::move(bad).Retain(), std::invalid_argument);
    auto block = Block(false);
    block.fields.at("key") = full;
    block.fields.at("value") = shared;
    RuntimeValue sequence(std::vector<RuntimeValue>{std::move(block)});
    const auto *payload = shared.Encoded().raw_data().data();
    PersistentValueState state(model, {{"past", sequence}});
    RuntimeContext context;
    context.RegisterCustomKernel(
        "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
          rt.PutValue(node.output(0), rt.values().at(node.input(0)).BorrowView());
        });
    auto output = state.Run(context, {{"tokens", Number(0)}});
    const auto &published = output.at("present").elements[0].fields.at("value");
    EXPECT_EQ(published.Encoded().raw_data().data(), payload);
    EXPECT_EQ(published.quantization_parameters, parameters);
    EXPECT_EQ(state.Values().at("past").elements[0].fields.at("value").quantization_parameters,
              parameters);
    retained = published.BorrowView();
    state.Reset({{"past", RuntimeValue(std::vector<RuntimeValue>{})}});
    state.Close();
  }
  ASSERT_FALSE(weak.expired());
  const auto decoded = DequantizeTensor(retained);
  ASSERT_EQ(decoded.element_count(), 2);
  EXPECT_EQ(decoded.AsFloat()[0], 1);
  EXPECT_EQ(decoded.AsFloat()[1], -2);
  retained = RuntimeValue{};
  EXPECT_TRUE(weak.expired());
}

TEST(PersistentValueState, AffineTensorValidationChecksLogicalTypeShapeSymbolsAndPayload) {
  for (int mode = 0; mode < 6; ++mode) {
    SCOPED_TRACE(mode);
    ModelProto model = Model();
    TypeProto type = AffineBlock().logical_type();
    if (mode == 3) {
      auto *dim = type.mutable_tensor_type()->mutable_shape()->mutable_dim(2);
      dim->set_dim_param("N");
      model.mutable_graph()
          ->mutable_input(1)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(0)
          ->set_dim_param("N");
    }
    *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
    *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
    auto encoded = AffineBlock();
    if (mode == 0)
      encoded.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    if (mode == 1) {
      encoded.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
      encoded.mutable_logical_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->add_dim()
          ->set_dim_value(2);
    }
    if (mode == 2)
      encoded = AffineBlock(3);
    if (mode == 4)
      *encoded.mutable_raw_data() = std::vector<uint8_t>{1};
    if (mode == 5) {
      encoded.clear_raw_data();
      encoded.set_data_location(TensorProto::EXTERNAL);
      auto *location = encoded.add_external_data();
      location->set_key("location");
      location->set_value("unused.bin");
      auto *length = encoded.add_external_data();
      length->set_key("length");
      length->set_value("2");
    }
    if (mode != 3) {
      EXPECT_THROW((PersistentValueState(model, {{"past", RuntimeValue(encoded)}})),
                   std::invalid_argument);
    } else {
      PersistentValueState state(model, {{"past", RuntimeValue(encoded)}});
      RuntimeContext context;
      EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
    }
  }
}

TEST(PersistentValueState, AffineParameterBackingRequiresItsOwnOwner) {
  for (bool scale : {false, true}) {
    auto message = std::make_shared<EncodedValueProto>(AffineBlock());
    auto *parameter = scale ? message->mutable_affine()->mutable_scale()
                            : message->mutable_affine()->mutable_zero_point();
    parameter->clear_float_data();
    parameter->clear_int32_data();
    auto owner = std::make_shared<std::vector<uint8_t>>(scale ? sizeof(float) : 1, 0);
    parameter->mutable_raw_data()->assign_borrowed(owner->data(), owner->size());
    RuntimeValue value = RuntimeValue::FromEncodedView(*message, message);
    EXPECT_THROW(std::move(value).Retain(), std::invalid_argument);
    const auto copy = value.DeepCopy();
    EXPECT_NO_THROW(RuntimeValue(copy).Retain());
    parameter->mutable_raw_data()->assign_borrowed(owner->data(), owner->size(), owner);
    const auto *bytes = owner->data();
    auto retained = std::move(value).Retain();
    owner.reset();
    message.reset();
    const auto &affine = retained.Encoded().affine();
    EXPECT_EQ((scale ? affine.scale() : affine.zero_point()).raw_data().data(), bytes);
  }
}

TEST(PersistentValueState, OrdinarySequenceOutputsMigrateAndDetachNestedPayloads) {
  for (bool use_allocators : {false, true}) {
    SCOPED_TRACE(use_allocators);
    ModelProto model = Model();
    model.mutable_graph()->clear_persistent_bindings();
    *model.mutable_graph()->mutable_output(0)->mutable_type() = BlockSequenceType();
    model.mutable_graph()->mutable_node(0)->add_output("intermediate");
    SimpleRawBufferAllocator execution(20);
    SimpleRawBufferAllocator io(20);
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.allocator = use_allocators ? &execution : nullptr,
                                                 .io_allocator = use_allocators ? &io : nullptr});
    float dense[] = {2, 3};
    uint8_t codes[] = {4, 6};
    context.RegisterCustomKernel(
        "test.feedback", "Step", [&](const NodeProto &node, RuntimeContext &rt) {
          RuntimeValue block = Block(false);
          block.fields.at("key") =
              RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1, 1, 2, 1},
                                          reinterpret_cast<const uint8_t *>(dense), sizeof(dense)));
          auto encoded = AffineBlock();
          encoded.mutable_raw_data()->assign_borrowed(codes, sizeof(codes));
          block.fields.at("value") = RuntimeValue(std::move(encoded));
          RuntimeValue sequence(std::vector<RuntimeValue>{std::move(block)});
          rt.values()[node.output(0)] = sequence;
          rt.values()[node.output(1)] = std::move(sequence);
        });
    context.Put("past", Number(0).tensor);
    context.Put("tokens", Number(0).tensor);
    RuntimeSession session(model);
    session.Run(context);
    const auto &block = context.values().at("present").elements.at(0);
    const auto &key = block.fields.at("key").tensor;
    EXPECT_NE(key.bytes(), reinterpret_cast<const uint8_t *>(dense));
    EXPECT_FALSE(key.is_borrowed());
    const auto &value = block.fields.at("value").Encoded();
    EXPECT_NE(value.raw_data().data(), codes);
    EXPECT_FALSE(value.raw_data().is_borrowed());
    if (use_allocators) {
      EXPECT_EQ(key.allocation_owner(), &io);
      const auto &intermediate =
          context.values().at("intermediate").elements.at(0).fields.at("key").tensor;
      EXPECT_EQ(intermediate.allocation_owner(), &execution);
      EXPECT_NE(intermediate.bytes(), reinterpret_cast<const uint8_t *>(dense));
    }
    dense[0] = 99;
    codes[0] = 99;
    EXPECT_EQ(key.AsFloat()[0], 2);
    EXPECT_EQ(value.raw_data()[0], 4);
  }
}

TEST(PersistentValueState, OrdinarySessionReleasesStructuredIntermediates) {
  ModelProto model = Model(true);
  *model.mutable_graph()->mutable_output(0)->mutable_type() = model.graph().input(0).type();
  model.mutable_graph()->mutable_node(0)->clear_output();
  model.mutable_graph()->mutable_node(0)->add_output("mid");
  auto *last = model.mutable_graph()->add_node();
  last->set_domain("test.feedback");
  last->set_op_type("Step");
  last->add_input("mid");
  last->add_output("response");
  SimpleRawBufferAllocator execution(20);
  SimpleRawBufferAllocator io(20);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &execution,
                                               .io_allocator = &io,
                                               .release_intermediates = true});
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        rt.values()[node.output(0)] = rt.values().at(node.input(0)).DeepCopy();
      });
  context.values()["request"] = Request(2, 3);
  context.Put("tokens", Number(1).tensor);
  RuntimeSession session(model);
  session.Run(context);
  EXPECT_EQ(context.values().count("mid"), 0u);
  ASSERT_EQ(context.values().count("response"), 1u);
  const Tensor &result =
      context.values().at("response").fields.at("cache").fields.at("keys").tensor;
  EXPECT_TRUE(result.has_allocation());
  EXPECT_EQ(result.allocation_owner(), &io);
  EXPECT_EQ(result.AsFloat()[0], 2);
  EXPECT_TRUE(context.Remove("tokens"));
  EXPECT_EQ(execution.TotalAllocatedSize(), 0u);
  EXPECT_TRUE(context.Remove("response"));
  EXPECT_EQ(io.TotalAllocatedSize(), 0u);
}

TEST(PersistentValueState, OrdinarySessionReleasesTypedSequenceAfterLastUse) {
  for (bool release : {false, true}) {
    SCOPED_TRACE(release);
    ModelProto model = Model();
    auto *graph = model.mutable_graph();
    graph->clear_persistent_bindings();
    graph->mutable_node(0)->clear_output();
    graph->mutable_node(0)->add_output("sequence");
    auto *info = graph->add_value_info();
    info->set_name("sequence");
    *info->mutable_type()->mutable_sequence_type()->mutable_elem_type() = FloatType();
    auto *consume = graph->add_node();
    consume->set_domain("test.feedback");
    consume->set_op_type("Consume");
    consume->add_input("sequence");
    consume->add_output("copied");
    auto *finish = graph->add_node();
    finish->set_domain("test.feedback");
    finish->set_op_type("Finish");
    finish->add_input("copied");
    finish->add_output("present");
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.release_intermediates = release});
    std::weak_ptr<std::vector<float>> weak;
    context.RegisterCustomKernel(
        "test.feedback", "Step", [&](const NodeProto &node, RuntimeContext &rt) {
          auto owner = std::make_shared<std::vector<float>>(1, 7);
          weak = owner;
          rt.values()[node.output(0)] =
              RuntimeValue(std::vector<RuntimeValue>{RuntimeValue(Tensor::Borrow(
                  "", DataType::FLOAT, {1}, reinterpret_cast<const uint8_t *>(owner->data()),
                  sizeof(float), owner))});
        });
    context.RegisterCustomKernel(
        "test.feedback", "Consume", [&](const NodeProto &node, RuntimeContext &rt) {
          EXPECT_FALSE(weak.expired());
          const auto &sequence = rt.values().at(node.input(0));
          rt.Put(node.output(0), Tensor::FromFloat("", {1}, {Number(sequence.elements.at(0))}));
        });
    context.RegisterCustomKernel("test.feedback", "Finish",
                                 [&](const NodeProto &node, RuntimeContext &rt) {
                                   EXPECT_EQ(weak.expired(), release);
                                   EXPECT_EQ(rt.values().count("sequence"), release ? 0u : 1u);
                                   rt.Put(node.output(0), rt.Get(node.input(0)).ToOwned());
                                 });
    context.Put("past", Number(0).tensor);
    context.Put("tokens", Number(0).tensor);
    RuntimeSession session(model);
    session.Run(context);
    EXPECT_EQ(context.Get("present").AsFloat()[0], 7);
    EXPECT_EQ(weak.expired(), release);
    if (!release) {
      context.GetExecutionPlan(model.graph()).ReleaseAfter(model.graph().node(1), context);
      EXPECT_TRUE(weak.expired());
      EXPECT_EQ(context.values().count("sequence"), 0u);
    }
  }
}

TEST(PersistentValueState, EncodedFieldsUseNativeCatalogueAndOwnedPayloads) {
  ModelProto model = Model(true);
  auto *format = model.add_struct_types();
  format->set_type_id(7);
  auto *packing = format->mutable_bit_packing();
  packing->set_dimension(1);
  auto *component = packing->add_component();
  component->set_name("code");
  component->set_bit_width(8);
  TypeProto encoded;
  encoded.mutable_struct_type()->set_type_ref(7);
  *model.mutable_graph()->mutable_input(0)->mutable_type() =
      Structure({{"logits", FloatType()}, {"cache", encoded}});
  *model.mutable_graph()->mutable_output(0)->mutable_type() =
      Structure({{"logits", FloatType()}, {"cache", encoded}});
  EncodedValueProto initial;
  initial.mutable_struct_type()->set_type_ref(7);
  initial.set_raw_data(std::string(1, '\x03'));
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const auto &previous = rt.values().at("request").fields.at("cache").Encoded();
    EncodedValueProto value;
    value.mutable_struct_type()->set_type_ref(7);
    value.set_raw_data(std::string(1, static_cast<char>(previous.raw_data()[0] + 1)));
    rt.values()["response"] = RuntimeValue(
        RuntimeValueMap{{"logits", Number(0)}, {"cache", RuntimeValue(std::move(value))}});
  });
  PersistentValueState state(
      model, {{"request", RuntimeValue(RuntimeValueMap{{"logits", Number(0)},
                                                       {"cache", RuntimeValue(initial)}})}});
  initial.set_raw_data(std::string(1, '\x40'));
  auto first = state.Run(context, {{"tokens", Number(1)}});
  state.Run(context, {{"tokens", Number(2)}});
  EXPECT_EQ(state.Values().at("request").fields.at("cache").Encoded().raw_data()[0], 5);
  EXPECT_EQ(first.at("response").fields.at("cache").Encoded().raw_data()[0], 4);
  EncodedValueProto wrong = initial;
  wrong.mutable_struct_type()->set_type_ref(8);
  EXPECT_THROW(
      state.Reset({{"request", RuntimeValue(RuntimeValueMap{{"logits", Number(0)},
                                                            {"cache", RuntimeValue(wrong)}})}}),
      std::invalid_argument);
}

TEST(PersistentValueState, EncodedTransportRetainsTheSamePayloadAcrossResetAndClose) {
  ModelProto model = Model();
  auto *format = model.add_struct_types();
  format->set_type_id(7);
  auto *packing = format->mutable_bit_packing();
  packing->set_dimension(1);
  auto *component = packing->add_component();
  component->set_name("code");
  component->set_bit_width(8);
  TypeProto type;
  type.mutable_struct_type()->set_type_ref(7);
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  EncodedValueProto encoded;
  encoded.mutable_struct_type()->set_type_ref(7);
  encoded.set_raw_data(std::string(1, '\x05'));
  RuntimeValueMap initial;
  initial.emplace("past", RuntimeValue(std::move(encoded)));
  const uint8_t *payload = initial.at("past").Encoded().raw_data().data();
  PersistentValueState state(model, initial);
  EXPECT_EQ(initial.at("past").Encoded().raw_data().data(), payload);
  EXPECT_EQ(state.Values().at("past").Encoded().raw_data().data(), payload);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    EXPECT_EQ(rt.values().at("past").Encoded().raw_data().data(), payload);
    rt.values()["present"] = rt.values().at("past").BorrowView();
  });
  auto output = state.Run(context, {{"tokens", Number(1)}});
  EXPECT_EQ(output.at("present").Encoded().raw_data().data(), payload);
  EXPECT_EQ(state.Values().at("past").Encoded().raw_data().data(), payload);
  state.Reset(initial);
  state.Close();
  initial.clear();
  EXPECT_EQ(output.at("present").Encoded().raw_data().data(), payload);
  EXPECT_EQ(output.at("present").Encoded().raw_data()[0], 5);
}

TEST(PersistentValueState, EncodedExternalViewsRetainTheMessageWithoutMutatingIt) {
  auto message = std::make_shared<EncodedValueProto>();
  message->mutable_struct_type()->set_type_ref(7);
  message->set_raw_data(std::string(1, '\x05'));
  std::weak_ptr<EncodedValueProto> weak = message;
  const EncodedValueProto *original = message.get();
  const uint8_t *payload = message->raw_data().data();
  RuntimeValue view = RuntimeValue::FromEncodedView(*message, message);
  RuntimeValue retained = view.BorrowView();
  EXPECT_EQ(&view.Encoded(), original);
  EXPECT_EQ(&retained.Encoded(), original);
  EXPECT_EQ(message->raw_data().data(), payload);
  EXPECT_EQ(message->struct_type().type_ref(), 7u);
  message.reset();
  view = RuntimeValue();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(retained.Encoded().raw_data().data(), payload);
  EXPECT_EQ(retained.Encoded().raw_data()[0], 5);
  retained = RuntimeValue();
  EXPECT_TRUE(weak.expired());
  EncodedValueProto ownerless;
  EXPECT_THROW(RuntimeValue::FromEncodedView(ownerless, {}), std::invalid_argument);
}

TEST(PersistentValueState, EncodedMessageOwnerDoesNotReplaceBorrowedBackingOwner) {
  uint8_t payload = 5;
  auto message = std::make_shared<EncodedValueProto>();
  auto *array = message->mutable_struct_type()->mutable_array();
  array->set_dimension(1);
  *array->mutable_element_type() = FloatType();
  array->mutable_element_type()->mutable_tensor_type()->set_elem_type(TensorProto::UINT8);
  message->mutable_raw_data()->assign_borrowed(&payload, 1);
  RuntimeValue view = RuntimeValue::FromEncodedView(*message, message);
  try {
    std::move(view).Retain();
    FAIL() << "Ownerless encoded backing was retained.";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("ownerless encoded payload"), std::string::npos);
  }
  const EncodedValueProto &retained = *message;
  EXPECT_EQ(retained.raw_data().data(), &payload);
}

TEST(PersistentValueState, EmptyNumericTensorsShareRatherThanCopy) {
  ModelProto model = Model();
  auto *identity = model.mutable_graph()->mutable_node(0);
  identity->clear_input();
  identity->add_input("past");
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(DataType::FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(0);
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  RuntimeValueMap initial;
  initial.emplace("past", RuntimeValue(Tensor::FromFloat("", {0}, {})));
  const Tensor &source = initial.at("past").tensor;
  const void *payload = source.bytes();
  PersistentValueState state(model, std::move(initial));
  RuntimeContext context;
  context.RegisterCustomKernel("test.feedback", "Step",
                               [](const NodeProto &node, RuntimeContext &rt) {
                                 rt.Put(node.output(0), rt.Get(node.input(0)).BorrowView());
                               });
  const auto output = state.Run(context, {{"tokens", Number(0)}});
  const Tensor &result = output.at("present").tensor;
  EXPECT_EQ(result.bytes(), payload);
  state.Close();
  initial.clear();
  EXPECT_EQ(result.size_bytes(), 0u);
}

TEST(PersistentValueState, RejectsStringDeclarationsBeforeReadingInitialState) {
  TypeProto strings = FloatType();
  strings.mutable_tensor_type()->set_elem_type(DataType::STRING);
  for (const TypeProto &type : {strings, Structure({{"cache", Structure({{"text", strings}})}})}) {
    ModelProto model = Model();
    *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
    *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
    try {
      PersistentValueState state(model, {});
      FAIL() << "String declaration was accepted.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("String tensors cannot be persistent"),
                std::string::npos);
    }
  }
}

TEST(PersistentValueState, RuntimeRetentionRejectsNestedStringsAndEncodedStringConstants) {
  RuntimeValue nested(RuntimeValueMap{
      {"cache", RuntimeValue(RuntimeValueMap{
                    {"text", RuntimeValue(Tensor::MakeString("", {1}, {"hello"}))}})}});
  EXPECT_THROW(std::move(nested).Retain(), std::invalid_argument);
  EncodedValueProto encoded;
  auto *structure = encoded.mutable_struct_type()->mutable_structure();
  auto *payload = structure->add_field();
  payload->set_name("payload");
  *payload->mutable_type() = FloatType();
  auto *constant = structure->add_field();
  constant->set_name("label");
  constant->mutable_constant()->set_data_type(TensorProto::STRING);
  constant->mutable_constant()->add_dims(1);
  constant->mutable_constant()->add_string_data("hello");
  encoded.set_raw_data(std::string(sizeof(float), '\0'));
  StructTypeCatalogue catalogue;
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(encoded));
  RuntimeValue inline_value(encoded);
  EXPECT_THROW(std::move(inline_value).Retain(), std::invalid_argument);
  ModelProto model;
  *model.add_struct_types() = encoded.struct_type();
  model.mutable_struct_types(0)->set_type_id(7);
  catalogue.Build(model);
  encoded.mutable_struct_type()->set_type_ref(7);
  RuntimeValue referenced(encoded);
  EXPECT_THROW(std::move(referenced).Retain(catalogue), std::invalid_argument);
  model.mutable_struct_types(0)
      ->mutable_structure()
      ->mutable_field(1)
      ->mutable_constant()
      ->set_data_type(TensorProto::FLOAT);
  auto *numeric_constant =
      model.mutable_struct_types(0)->mutable_structure()->mutable_field(1)->mutable_constant();
  numeric_constant->clear_string_data();
  numeric_constant->add_float_data(1.0f);
  catalogue.Build(model);
  RuntimeValue numeric(encoded);
  EXPECT_NO_THROW(std::move(numeric).Retain(catalogue));
  encoded.mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::STRING);
  RuntimeValue logical_strings(encoded);
  EXPECT_THROW(std::move(logical_strings).Retain(catalogue), std::invalid_argument);
}

TEST(PersistentValueState, OrdinaryStringFeedsAndOutputsAreNotRetained) {
  ModelProto model = Model();
  auto *input = model.mutable_graph()->add_input();
  input->set_name("text");
  *input->mutable_type() = FloatType();
  input->mutable_type()->mutable_tensor_type()->set_elem_type(DataType::STRING);
  auto *output = model.mutable_graph()->add_output();
  output->set_name("text_out");
  *output->mutable_type() = input->type();
  model.mutable_graph()->mutable_node(0)->add_input("text");
  model.mutable_graph()->mutable_node(0)->add_output("text_out");
  PersistentValueState state(model, {{"past", Number(0)}});
  RuntimeContext context;
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    EXPECT_TRUE(rt.retains_output("present"));
    EXPECT_FALSE(rt.retains_output("text_out"));
    rt.Put("present", Tensor::FromFloat("", {1}, {rt.Get("past").AsFloat()[0] + 1.0f}));
    rt.Put("text_out", rt.Get("text").ToOwned());
  });
  for (float expected : {1.0f, 2.0f}) {
    RuntimeValueMap feeds{{"tokens", Number(0)},
                          {"text", RuntimeValue(Tensor::MakeString("", {1}, {"hello"}))}};
    auto outputs = state.Run(context, feeds);
    feeds.clear();
    EXPECT_EQ(outputs.at("text_out").tensor.AsStrings()[0], "hello");
    EXPECT_EQ(outputs.at("text_out").tensor.borrowed_owner().use_count(), 0);
    const auto retained = state.Values();
    EXPECT_EQ(retained.size(), 1u);
    EXPECT_EQ(Number(retained.at("past")), expected);
  }
}

TEST(PersistentValueState, CustomStringOutputCannotReplaceNumericState) {
  ModelProto model = Model();
  PersistentValueState state(model, {{"past", Number(3)}});
  RuntimeContext context;
  context.RegisterCustomKernel("test.feedback", "Step",
                               [](const NodeProto &node, RuntimeContext &rt) {
                                 rt.Put(node.output(0), Tensor::MakeString("", {1}, {"invalid"}));
                               });
  EXPECT_THROW(state.Run(context, {{"tokens", Number(1)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 3.0f);
}

TEST(PersistentValueState, RejectsOwnerlessInitialAndOutputBorrowsTransactionally) {
  ModelProto model = Model();
  float value = 3;
  const auto borrow = [&] {
    return RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                       reinterpret_cast<const uint8_t *>(&value), sizeof(float)));
  };
  EXPECT_THROW((PersistentValueState(model, {{"past", borrow()}})), std::invalid_argument);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    rt.Put("present", borrow().tensor);
  });
  PersistentValueState state(model, {{"past", Number(1)}});
  const auto snapshot = state.Values();
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), snapshot.at("past").tensor.bytes());
}

TEST(PersistentValueState, IOArenaOutputsRemainLeasedUntilAllViewsAreReleased) {
  ModelProto model = Model();
  RuntimeValueMap output;
  std::weak_ptr<IOArena> weak;
  {
    auto arena = IOArena::Create(4);
    weak = arena;
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.io_allocator = arena.get()});
    RegisterStep(context);
    PersistentValueState state(model, {{"past", Number(1)}});
    output = state.Run(context, {{"tokens", Number(2)}});
    EXPECT_EQ(state.Values().at("past").tensor.bytes(), output.at("present").tensor.bytes());
    EXPECT_EQ(arena->TotalAllocatedSize(), sizeof(float));
    state.Close();
  }
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(Number(output.at("present")), 3);
  output.clear();
  EXPECT_TRUE(weak.expired());
}

TEST(PersistentValueState, InitializerViewsRetainImmutableModelAndPayload) {
  for (bool raw : {false, true}) {
    SCOPED_TRACE(raw);
    RuntimeValueMap output;
    std::weak_ptr<ModelProto> weak;
    const uint8_t *payload = nullptr;
    {
      auto model = std::make_shared<ModelProto>(Model());
      weak = model;
      model->add_opset_import()->set_version(18);
      auto *initializer = model->mutable_graph()->add_initializer();
      initializer->set_name("constant");
      initializer->set_data_type(DataType::FLOAT);
      initializer->add_dims(1);
      if (raw) {
        const float value = 4;
        initializer->set_raw_data(
            std::string(reinterpret_cast<const char *>(&value), sizeof(value)));
        payload = initializer->raw_data().data();
      } else {
        initializer->add_float_data(4);
        payload = reinterpret_cast<const uint8_t *>(initializer->float_data().values().data());
      }
      model->mutable_graph()->clear_node();
      auto *consume = model->mutable_graph()->add_node();
      consume->set_op_type("Identity");
      consume->add_input("past");
      consume->add_output("previous");
      model->mutable_graph()->mutable_output(0)->set_name("constant");
      model->mutable_graph()->mutable_persistent_bindings(0)->set_output_name("constant");
      RuntimeContext context(KernelContext(DefaultOpset(18)));
      RegisterIdentity(context);
      context.set_model_owner(std::make_shared<int>(0));
      std::unique_ptr<PersistentValueState> state;
      const RuntimeValueMap initial{{"past", Number(1)}};
      if (raw)
        state =
            std::make_unique<PersistentValueState>(*model, initial, RuntimeSessionOptions{}, model);
      else
        state = std::make_unique<PersistentValueState>(std::shared_ptr<const ModelProto>(model),
                                                       initial);
      output = state->Run(context, {{"tokens", Number(0)}});
      EXPECT_EQ(output.at("constant").tensor.bytes(), payload);
      EXPECT_EQ(state->Values().at("past").tensor.bytes(), payload);
      state->Close();
      model.reset();
    }
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(output.at("constant").tensor.bytes(), payload);
    EXPECT_EQ(Number(output.at("constant")), 4);
    output.clear();
    EXPECT_TRUE(weak.expired());
  }
}

TEST(PersistentValueState, OrdinarySessionsBorrowInitializersWithoutArenaCopies) {
  for (bool raw : {false, true}) {
    for (bool separate_graph : {false, true}) {
      for (const auto device : {core::symbolic::Device::kUndefined, core::symbolic::Device::kCPU}) {
        SCOPED_TRACE(raw);
        SCOPED_TRACE(separate_graph);
        SCOPED_TRACE(static_cast<int32_t>(device));
        ModelProto model;
        GraphProto source;
        auto *graph = model.mutable_graph();
        auto *initializer = separate_graph ? source.add_initializer() : graph->add_initializer();
        initializer->set_name("constant");
        initializer->set_data_type(DataType::FLOAT);
        initializer->add_dims(1);
        const uint8_t *original = nullptr;
        if (raw) {
          const float value = 4;
          initializer->set_raw_data(
              std::string(reinterpret_cast<const char *>(&value), sizeof(value)));
          original = initializer->raw_data().data();
        } else {
          initializer->add_float_data(4);
          original = reinterpret_cast<const uint8_t *>(initializer->float_data().values().data());
        }
        auto *node = graph->add_node();
        node->set_domain("test.feedback");
        node->set_op_type("ObserveInitializer");
        node->add_input("constant");
        RuntimeSession session(model);
        if (separate_graph)
          session.SetInitializers(source);
        ExecutionArena arena(16);
        RuntimeContext context(KernelContext(DefaultOpset(18)),
                               RuntimeContextOptions{.allocator = &arena, .device = device});
        int calls = 0;
        context.RegisterCustomKernel("test.feedback", "ObserveInitializer",
                                     [&](const NodeProto &, RuntimeContext &rt) {
                                       ++calls;
                                       EXPECT_EQ(rt.Get("constant").bytes(), original);
                                       EXPECT_EQ(rt.Get("constant").AsFloat()[0], 4);
                                       EXPECT_EQ(arena.TotalAllocatedSize(), 0u);
                                     });
        for (int run = 0; run < 2; ++run) {
          session.Run(context);
          EXPECT_TRUE(context.Get("constant").is_borrowed());
          context.Clear();
        }
        EXPECT_EQ(calls, 2);
      }
    }
  }
}

TEST(PersistentValueState, InitializerArenaExemptionIsLimitedToHostExecution) {
  for (const auto device : {core::symbolic::Device::kUndefined, core::symbolic::Device::kCPU,
                            static_cast<core::symbolic::Device>(0)}) {
    SCOPED_TRACE(static_cast<int32_t>(device));
    ExecutionArena arena(16);
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.allocator = &arena, .device = device});
    const float value = 4;
    const auto *original = reinterpret_cast<const uint8_t *>(&value);
    const auto borrow = [&] {
      return Tensor::Borrow("constant", DataType::FLOAT, {1}, original, sizeof(value));
    };
    context.Set("constant", borrow(), RuntimeEventKind::kInitializer);
    const bool host = device != static_cast<core::symbolic::Device>(0);
    EXPECT_EQ(context.Get("constant").bytes() == original, host);
    context.Put("constant", borrow(), RuntimeEventKind::kInitializer);
    EXPECT_EQ(context.Get("constant").bytes() == original, host);
    EXPECT_EQ(context.Get("constant").AsFloat()[0], value);
    context.Set("intermediate", borrow(), RuntimeEventKind::kOutput);
    EXPECT_NE(context.Get("intermediate").bytes(), original);
    EXPECT_TRUE(context.Get("intermediate").has_allocation());
  }
}

TEST(PersistentValueState, ModelOwnerDoesNotReplaceBorrowedInitializerBackingOwner) {
  for (bool retained_backing : {false, true}) {
    auto data = std::make_shared<std::vector<float>>(1, 4.0f);
    std::weak_ptr<std::vector<float>> backing_lifetime = data;
    const auto *payload = reinterpret_cast<const uint8_t *>(data->data());
    auto model = std::make_shared<ModelProto>(Model());
    std::weak_ptr<ModelProto> model_lifetime = model;
    model->add_opset_import()->set_version(18);
    auto *initializer = model->mutable_graph()->add_initializer();
    initializer->set_name("constant");
    initializer->set_data_type(DataType::FLOAT);
    initializer->add_dims(1);
    initializer->mutable_raw_data()->assign_borrowed(
        payload, sizeof(float), retained_backing ? data : std::shared_ptr<void>{});
    model->mutable_graph()->clear_node();
    auto *consume = model->mutable_graph()->add_node();
    consume->set_op_type("Identity");
    consume->add_input("past");
    consume->add_output("previous");
    model->mutable_graph()->mutable_output(0)->set_name("constant");
    model->mutable_graph()->mutable_persistent_bindings(0)->set_output_name("constant");
    PersistentValueState state(std::shared_ptr<const ModelProto>(model), {{"past", Number(1)}});
    RuntimeContext context(KernelContext(DefaultOpset(18)));
    RegisterIdentity(context);
    RuntimeValueMap output;
    if (retained_backing) {
      output = state.Run(context, {{"tokens", Number(0)}});
      EXPECT_EQ(output.at("constant").tensor.bytes(), payload);
    } else {
      const auto before = state.Values();
      EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
      EXPECT_EQ(state.Values().at("past").tensor.bytes(), before.at("past").tensor.bytes());
    }
    state.Close();
    model.reset();
    data.reset();
    EXPECT_TRUE(model_lifetime.expired());
    if (retained_backing) {
      ASSERT_FALSE(backing_lifetime.expired());
      EXPECT_EQ(output.at("constant").tensor.bytes(), payload);
      EXPECT_EQ(Number(output.at("constant")), 4);
      output.clear();
    }
    EXPECT_TRUE(backing_lifetime.expired());
  }
}

TEST(PersistentValueState, WholeStructureFieldsTreatDotsLiterally) {
  ModelProto model = Model(true);
  *model.mutable_graph()->mutable_input(0)->mutable_type() =
      Structure({{"cache.part", FloatType()}, {"logits", FloatType()}});
  *model.mutable_graph()->mutable_output(0)->mutable_type() =
      Structure({{"cache.part", FloatType()}, {"logits", FloatType()}});
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "request", "response");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    rt.values()["response"] = RuntimeValue(RuntimeValueMap{
        {"cache.part", rt.values().at("request").fields.at("cache.part").BorrowView()},
        {"logits", Number(0)}});
  });
  PersistentValueState state(
      model, {{"request",
               RuntimeValue(RuntimeValueMap{{"cache.part", Number(1)}, {"logits", Number(0)}})}});
  const auto before = state.Values();
  const auto output = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(output.at("response").fields.at("cache.part").tensor.bytes(),
            before.at("request").fields.at("cache.part").tensor.bytes());
}

TEST(PersistentValueState, WholeStructuredInitialValuesRejectPartialFeedsAndOverrides) {
  ModelProto model = Model(true);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStructuredStep(context);
  RuntimeValueMap initial{{"request", Request(1, 2)}};
  PersistentValueState state(model, initial);
  EXPECT_EQ(state.Values().at("request").fields.at("cache").fields.at("keys").tensor.bytes(),
            initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes());
  const auto output = state.Run(context, {{"tokens", Number(3)}});
  EXPECT_EQ(Number(output.at("response").fields.at("cache").fields.at("keys")), 4);
  EXPECT_THROW(
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"tokens", Number(3)}})},
                          {"tokens", Number(3)}}),
      std::invalid_argument);
  EXPECT_THROW(state.Run(context, {{"request", Request(9, 9)}, {"tokens", Number(3)}}),
               std::invalid_argument);
  EXPECT_THROW(state.Reset({{"request", Request(1, 2)}, {"request.cache", Cache(1, 2)}}),
               std::invalid_argument);
  EXPECT_THROW(state.Reset({{"request", RuntimeValue(RuntimeValueMap{{"cache", Cache(1, 2)}})}}),
               std::invalid_argument);
  EXPECT_THROW(state.Run(context, {{"request.cache", Cache(1, 2)}, {"tokens", Number(3)}}),
               std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("request").fields.at("cache").fields.at("keys")), 4);
}

TEST(PersistentValueState, LiteralDottedRootAndWholeStructureHaveDistinctStateSlots) {
  ModelProto model = Model(true);
  const TypeProto cache = Structure({{"keys", FloatType()}, {"values", FloatType()}});
  auto *literal_input = model.mutable_graph()->add_input();
  literal_input->set_name("request.cache");
  *literal_input->mutable_type() = cache;
  auto *literal_output = model.mutable_graph()->add_output();
  literal_output->set_name("response.cache");
  *literal_output->mutable_type() = cache;
  Bind(model, "request.cache", "response.cache");
  auto *node = model.mutable_graph()->mutable_node(0);
  node->add_input("request.cache");
  node->add_output("response.cache");
  RuntimeValueMap initial{{"request", Request(1, 2)}, {"request.cache", Cache(10, 20)}};
  const uint8_t *nested = initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes();
  const uint8_t *literal = initial.at("request.cache").fields.at("keys").tensor.bytes();
  PersistentValueState state(model, initial);
  const auto before = state.Values();
  ASSERT_EQ(before.size(), 2u);
  EXPECT_EQ(before.at("request").fields.at("cache").fields.at("keys").tensor.bytes(), nested);
  EXPECT_EQ(before.at("request.cache").fields.at("keys").tensor.bytes(), literal);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    rt.values()["response"] =
        RuntimeValue(RuntimeValueMap{{"cache", request.fields.at("cache").BorrowView()},
                                     {"logits", RuntimeValue(rt.Get("tokens").BorrowView())}});
    rt.values()["response.cache"] = rt.values().at("request.cache").BorrowView();
  });
  const auto output = state.Run(context, {{"tokens", Number(3)}});
  EXPECT_EQ(output.at("response").fields.at("cache").fields.at("keys").tensor.bytes(), nested);
  EXPECT_EQ(output.at("response.cache").fields.at("keys").tensor.bytes(), literal);
  state.Reset(before);
  EXPECT_THROW(state.Reset({{"request.cache", Cache(3, 4)}}), std::invalid_argument);
  EXPECT_EQ(state.Values().at("request.cache").fields.at("keys").tensor.bytes(), literal);
}

TEST(PersistentValueState, WholeStructureKeepsDottedFieldsAndNestedFieldsDistinct) {
  ModelProto model = Model(true);
  const TypeProto type =
      Structure({{"cache.part", FloatType()}, {"cache", Structure({{"part", FloatType()}})}});
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "request", "response");
  RuntimeValueMap initial{
      {"request", RuntimeValue(RuntimeValueMap{
                      {"cache.part", Number(1)},
                      {"cache", RuntimeValue(RuntimeValueMap{{"part", Number(2)}})}})}};
  PersistentValueState state(model, initial);
  const auto before = state.Values();
  EXPECT_EQ(Number(before.at("request").fields.at("cache.part")), 1);
  EXPECT_EQ(Number(before.at("request").fields.at("cache").fields.at("part")), 2);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    rt.values()["response"] = rt.values().at("request").BorrowView();
  });
  const auto output = state.Run(context, {{"tokens", Number(3)}});
  EXPECT_EQ(output.at("response").fields.at("cache.part").tensor.bytes(),
            before.at("request").fields.at("cache.part").tensor.bytes());
  EXPECT_EQ(output.at("response").fields.at("cache").fields.at("part").tensor.bytes(),
            before.at("request").fields.at("cache").fields.at("part").tensor.bytes());
  state.Reset(before);
  EXPECT_THROW(state.Reset({{"request.cache.part", Number(3)}}), std::invalid_argument);
}

TEST(PersistentValueState, CurrentFeedsRequireWholeStructuredInputs) {
  ModelProto model = Model(true);
  *model.mutable_graph()->mutable_input(1)->mutable_type() =
      Structure({{"token.part", FloatType()}});
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    rt.values()["response"] = RuntimeValue(
        RuntimeValueMap{{"cache", request.fields.at("cache").BorrowView()},
                        {"logits", rt.values().at("tokens").fields.at("token.part").BorrowView()}});
  });
  PersistentValueState state(model, {{"request", Request(1, 2)}});
  RuntimeValueMap feeds{{"tokens", RuntimeValue(RuntimeValueMap{{"token.part", Number(3)}})}};
  const uint8_t *token = feeds.at("tokens").fields.at("token.part").tensor.bytes();
  const auto output = state.Run(context, feeds);
  EXPECT_EQ(output.at("response").fields.at("logits").tensor.bytes(), token);
  EXPECT_EQ(Number(output.at("response").fields.at("logits")), 3);
  EXPECT_THROW(state.Run(context, {{R"(tokens.token\.part)", Number(4)}}), std::invalid_argument);
  EXPECT_THROW(state.Run(context, {{"tokens.token.part", Number(4)}}), std::invalid_argument);
}

TEST(PersistentValueState, GraphAndFieldNamesTreatBackslashesAndDotsLiterally) {
  ModelProto model = Model(true);
  const std::string input_name = R"(request\part.cache)";
  const std::string output_name = R"(response\part.cache)";
  const std::string state_field = R"(cache\part.state)";
  const std::string token_field = R"(token.part\suffix)";
  const TypeProto type = Structure({{state_field, FloatType()}, {token_field, FloatType()}});
  model.mutable_graph()->mutable_input(0)->set_name(input_name);
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  model.mutable_graph()->mutable_output(0)->set_name(output_name);
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  auto *node = model.mutable_graph()->mutable_node(0);
  node->clear_input();
  node->add_input(input_name);
  node->clear_output();
  node->add_output(output_name);
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, input_name, output_name);
  RuntimeValueMap initial{{input_name, RuntimeValue(RuntimeValueMap{{state_field, Number(1)},
                                                                    {token_field, Number(2)}})}};
  const uint8_t *payload = initial.at(input_name).fields.at(state_field).tensor.bytes();
  PersistentValueState state(model, initial);
  EXPECT_EQ(state.Values().at(input_name).fields.at(state_field).tensor.bytes(), payload);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        rt.values()[node.output(0)] = rt.values().at(node.input(0)).BorrowView();
      });
  const auto flat_output = state.Run(context, {{"tokens", Number(0)}});
  EXPECT_EQ(flat_output.at(output_name).fields.at(state_field).tensor.bytes(), payload);
  EXPECT_EQ(Number(flat_output.at(output_name).fields.at(token_field)), 2);
  state.Reset(state.Values());
  EXPECT_EQ(state.Values().at(input_name).fields.at(state_field).tensor.bytes(), payload);
  for (const std::string &invalid :
       {std::string(R"(request\\part\.cache)"), input_name + "." + state_field}) {
    EXPECT_THROW(state.Run(context, {{invalid, Number(4)}}), std::invalid_argument);
  }
}

TEST(PersistentValueState, StatesShareExplicitInitialStorageButPublishIndependently) {
  ModelProto model = Model();
  RuntimeValueMap initial{{"past", Number(1)}};
  PersistentValueState first(model, initial);
  PersistentValueState second(model, initial);
  const uint8_t *payload = initial.at("past").tensor.bytes();
  EXPECT_EQ(first.Values().at("past").tensor.bytes(), payload);
  EXPECT_EQ(second.Values().at("past").tensor.bytes(), payload);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  const auto output = first.Run(context, {{"tokens", Number(1)}});
  EXPECT_EQ(first.Values().at("past").tensor.bytes(), output.at("present").tensor.bytes());
  EXPECT_EQ(second.Values().at("past").tensor.bytes(), payload);
  EXPECT_EQ(Number(second.Values().at("past")), 1);
  EXPECT_EQ(Number(first.Values().at("past")), 2);
}

TEST(PersistentValueState, RejectsPersistentDeclarationsInsideSubgraphs) {
  ModelProto model = Model();
  auto *attribute = model.mutable_graph()->mutable_node(0)->add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::GRAPH);
  auto *binding = attribute->mutable_g()->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");
  EXPECT_THROW((PersistentValueState(model, {{"past", Number(1)}})), std::invalid_argument);
}

TEST(PersistentValueState, RetainsExternalOwnersAcrossFailedInitializationUntilClose) {
  auto model = std::make_shared<ModelProto>(Model());
  std::weak_ptr<ModelProto> model_lifetime = model;
  PersistentValueState state(*model, {{"past", Number(1)}});
  state.RetainOwner(model);
  state.RetainOwner(model);
  EXPECT_EQ(model.use_count(), 2);
  auto context = std::make_shared<RuntimeContext>(KernelContext(DefaultOpset(18)));
  std::weak_ptr<RuntimeContext> context_lifetime = context;
  state.RetainOwner(context);
  model.reset();
  EXPECT_THROW(state.Run(*context, {{"tokens", Number(1)}}), std::invalid_argument);
  context.reset();
  EXPECT_FALSE(model_lifetime.expired());
  EXPECT_FALSE(context_lifetime.expired());
  EXPECT_THROW(state.RetainOwner({}), std::invalid_argument);
  state.Close();
  EXPECT_TRUE(model_lifetime.expired());
  EXPECT_TRUE(context_lifetime.expired());
  auto owner = std::make_shared<int>(1);
  EXPECT_THROW(state.RetainOwner(owner), std::invalid_argument);
}

TEST(PersistentValueState, RejectsNullAffinePayloadsBeforeContentValidation) {
  ModelProto model = Model();
  PersistentValueState state(model, {{"past", Number(1)}});
  auto owner = std::make_shared<int>(0);
  for (TensorProto::DataType storage : {TensorProto::INT4, TensorProto::INT8})
    for (int part = 0; part < 5; ++part) {
      SCOPED_TRACE(storage);
      SCOPED_TRACE(part);
      EncodedValueProto encoded;
      *encoded.mutable_logical_type() = FloatType();
      const uint8_t code = 0;
      encoded.set_raw_data(&code, 1);
      auto *affine = encoded.mutable_affine();
      affine->set_storage_type(storage);
      affine->mutable_scale()->set_data_type(TensorProto::FLOAT);
      affine->mutable_scale()->add_float_data(1);
      affine->mutable_zero_point()->set_data_type(storage);
      affine->mutable_zero_point()->set_raw_data(&code, 1);
      if (part == 0)
        encoded.ref_raw_data().assign_borrowed(nullptr, 1, owner);
      else if (part == 1 || part == 3) {
        affine->mutable_scale()->clear_float_data();
        if (part == 3)
          affine->mutable_scale()->set_data_type(TensorProto::FLOAT6E2M3);
        affine->mutable_scale()->ref_raw_data().assign_borrowed(
            nullptr, part == 3 ? 1 : sizeof(float), owner);
      } else {
        if (part == 4)
          affine->mutable_zero_point()->set_data_type(TensorProto::FLOAT6E2M3);
        affine->mutable_zero_point()->ref_raw_data().assign_borrowed(nullptr, 1, owner);
      }
      RuntimeValue value(std::move(encoded));
      EXPECT_THROW(StructTypeCatalogue().ValidateEncodedValue(value.Encoded()),
                   std::invalid_argument);
      EXPECT_THROW((PersistentValueState(model, {{"past", value.BorrowView()}})),
                   std::invalid_argument);
      EXPECT_THROW(state.Reset({{"past", value.BorrowView()}}), std::invalid_argument);
      EXPECT_EQ(Number(state.Values().at("past")), 1);
    }
}
