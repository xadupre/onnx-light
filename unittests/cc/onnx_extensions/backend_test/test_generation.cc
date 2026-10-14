// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/generation.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include <cstring>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

std::vector<int64_t> Tokens(const Tensor &tensor) {
  return {tensor.AsInt64(), tensor.AsInt64() + tensor.element_count()};
}

uint64_t StorageTotal(const RuntimeEventLog &events, uint64_t RuntimeEvent::*field) {
  uint64_t total = 0;
  for (const auto &event : events)
    if (event.action == RuntimeEventAction::kPersistentStorage)
      total += event.*field;
  return total;
}

} // namespace

TEST(Generation, BackendAttentionModels) {
  onnx_kernels::RegisterKernelFunctions();
  auto cases = core::backend_test::CollectTestCasesByName("^test_cc_generation_attention_");
  ASSERT_EQ(cases.size(), 2u);
  for (auto &test : cases) {
    SCOPED_TRACE(test.name);
    const auto &model = test.model();
    const auto original = model.SerializeAsString();
    RuntimeValueMap feeds;
    for (const auto &input : test.data_sets()[0].inputs)
      feeds.emplace(input.name, RuntimeValue(input.ToOwned()));
    const bool persistent = !model.graph().persistent_bindings().empty();
    for (size_t capacity : {size_t{0}, size_t{32}}) {
      SCOPED_TRACE(capacity);
      RuntimeContext context(RuntimeContextOptions{.events_enabled = true});
      RuntimeSession session(model);
      for (const auto &[name, value] : feeds)
        context.PutValue(name, value.BorrowView());
      session.Run(context);
      for (const auto &expected : test.data_sets()[0].outputs) {
        const Tensor &actual = context.Get(expected.name);
        ASSERT_EQ(actual.shape, expected.shape);
        ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
        EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
      }
      context.ClearEvents();
      GenerationOptions options;
      options.max_new_tokens = 4;
      RuntimeSessionOptions session_options;
      session_options.check_shapes = true;
      session_options.persistent_tensor_initial_capacity = capacity;
      auto output = Generate(model, context, feeds, options, session_options);
      EXPECT_EQ(output.shape, (Shape{1, 6}));
      EXPECT_EQ(Tokens(output), (std::vector<int64_t>{0, 1, 1, 1, 1, 1}));
      const auto events = context.events();
      std::vector<int64_t> input_lengths;
      for (const auto &event : events)
        if (event.kind == RuntimeEventKind::kInput && event.name == "input_ids")
          input_lengths.push_back(event.shape[1]);
      EXPECT_EQ(input_lengths,
                (persistent ? std::vector<int64_t>{2, 1, 1, 1} : std::vector<int64_t>{2, 3, 4, 5}));
      EXPECT_EQ(StorageTotal(events, &RuntimeEvent::storage_reuse_count),
                persistent && capacity ? 6u : 0u);
      EXPECT_EQ(StorageTotal(events, &RuntimeEvent::storage_prefix_copied_bytes),
                persistent && !capacity ? 144u : 0u);
      if (!persistent) {
        EXPECT_EQ(StorageTotal(events, &RuntimeEvent::storage_allocations), 0u);
      }
      EXPECT_EQ(feeds.at("input_ids").tensor.shape, (Shape{1, 2}));
      if (persistent) {
        EXPECT_FALSE(context.Has("input_ids"));
        EXPECT_EQ(feeds.at("past_key").tensor.shape, (Shape{1, 1, 0, 2}));
      } else {
        EXPECT_EQ(context.Get("input_ids").shape, (Shape{1, 5}));
      }
      EXPECT_EQ(model.SerializeAsString(), original);
      EXPECT_EQ(Tokens(Generate(model, context, feeds, options, session_options)), Tokens(output));
      options.eos_token_id = 1;
      EXPECT_EQ(Tokens(Generate(model, context, feeds, options, session_options)),
                (std::vector<int64_t>{0, 1, 1}));
    }
  }
}

TEST(Generation, PipelineUsesOneContextAndDeclaredGraphNames) {
  const auto make_model = [](const std::string &operation,
                             const std::vector<std::pair<std::string, int32_t>> &inputs,
                             const std::string &output) {
    ModelProto model;
    model.set_ir_version(10);
    auto *opset = model.add_opset_import();
    opset->set_domain("test.pipeline");
    opset->set_version(1);
    auto *graph = model.mutable_graph();
    auto *node = graph->add_node();
    node->set_domain("test.pipeline");
    node->set_op_type(operation);
    for (const auto &[name, type] : inputs) {
      auto *input = graph->add_input();
      input->set_name(name);
      input->mutable_type()->mutable_tensor_type()->set_elem_type(type);
      node->add_input(name);
    }
    auto *result = graph->add_output();
    result->set_name(output);
    result->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    node->add_output(output);
    return model;
  };
  const auto vision = make_model("Encode", {{"pixels", TensorProto::FLOAT}}, "vision");
  const auto embedding = make_model("Embed", {{"input_ids", TensorProto::INT64}}, "inputs_embeds");
  const auto decoder = make_model(
      "Decode", {{"inputs_embeds", TensorProto::FLOAT}, {"vision", TensorProto::FLOAT}}, "logits");
  RuntimeSession decoder_session(decoder);
  auto executor = decoder_session.cpu_executor();
  RuntimeContext context(RuntimeContextOptions{.cpu_executor = executor.get()});
  RawBufferAllocator *allocator = nullptr;
  std::vector<std::string> calls;
  const auto check_context = [&](const NodeProto &node, RuntimeContext &rt) {
    EXPECT_EQ(&rt, &context);
    EXPECT_EQ(rt.cpu_executor(), executor.get());
    if (allocator == nullptr)
      allocator = rt.execution_allocator();
    EXPECT_EQ(rt.execution_allocator(), allocator);
    calls.push_back(node.op_type());
  };
  context.RegisterCustomKernel("test.pipeline", "Encode",
                               [&](const NodeProto &node, RuntimeContext &rt) {
                                 check_context(node, rt);
                                 rt.Put(node.output(0), rt.Get("pixels").BorrowView());
                               });
  context.RegisterCustomKernel(
      "test.pipeline", "Embed", [&](const NodeProto &node, RuntimeContext &rt) {
        check_context(node, rt);
        const Tensor &ids = rt.Get("input_ids");
        std::vector<float> logits(static_cast<size_t>(ids.element_count()) * 3, 0);
        for (int64_t i = 0; i < ids.element_count(); ++i)
          logits[static_cast<size_t>(i) * 3 + (ids.AsInt64()[i] + 1) % 3] = 4;
        rt.Put(node.output(0), Tensor::FromFloat("", {1, ids.shape[1], 3}, logits));
      });
  context.RegisterCustomKernel("test.pipeline", "Decode",
                               [&](const NodeProto &node, RuntimeContext &rt) {
                                 check_context(node, rt);
                                 EXPECT_FLOAT_EQ(rt.Get("vision").AsFloat()[0], 0);
                                 rt.Put(node.output(0), rt.Get("inputs_embeds").BorrowView());
                               });
  GenerationPipeline pipeline{.context = &context, .prefill = {&vision}, .embedding = &embedding};
  RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))},
                        {"pixels", RuntimeValue(Tensor::FromFloat("", {1, 1, 3}, {0, 0, 0}))}};
  GenerationOptions options;
  options.max_new_tokens = 3;
  for (int repeat = 0; repeat < 2; ++repeat) {
    calls.clear();
    EXPECT_EQ(Tokens(Generate(decoder, context, feeds, options, {}, pipeline)),
              (std::vector<int64_t>{0, 1, 2, 0}));
    EXPECT_EQ(calls, (std::vector<std::string>{"Encode", "Embed", "Decode", "Embed", "Decode",
                                               "Embed", "Decode"}));
  }
}

TEST(Generation, TemperatureMatchesCachedAndFullPrefixDecoding) {
  onnx_kernels::RegisterKernelFunctions();
  auto cases = core::backend_test::CollectTestCasesByName("^test_cc_generation_attention_");
  std::vector<std::vector<int64_t>> results;
  for (auto &test : cases) {
    RuntimeValueMap feeds;
    for (const auto &input : test.data_sets()[0].inputs)
      feeds.emplace(input.name, RuntimeValue(input.ToOwned()));
    RuntimeContext context;
    GenerationOptions options;
    options.max_new_tokens = 12;
    options.temperature = 2.;
    options.seed = 7;
    results.push_back(Tokens(Generate(test.model(), context, feeds, options)));
  }
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0], results[1]);
}

TEST(Generation, RetainsOwnedInitialCounterWithoutChangingCaller) {
  onnx_kernels::RegisterKernelFunctions();
  auto cases =
      core::backend_test::CollectTestCasesByName("^test_cc_generation_attention_persistent$");
  ASSERT_EQ(cases.size(), 1u);
  ModelProto model;
  model.CopyFrom(cases[0].model());
  auto &graph = *model.mutable_graph();
  auto *input = graph.add_input();
  input->set_name("counter");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  input->mutable_type()->mutable_tensor_type()->mutable_shape();
  auto *output = graph.add_output();
  output->CopyFrom(*input);
  output->set_name("next_counter");
  auto *initializer = graph.add_initializer();
  initializer->set_name("one");
  initializer->set_data_type(TensorProto::INT64);
  initializer->add_int64_data(1);
  auto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("counter");
  node->add_input("one");
  node->add_output("next_counter");
  auto *binding = graph.add_persistent_bindings();
  binding->set_input_name("counter");
  binding->set_output_name("next_counter");
  RuntimeValueMap feeds;
  for (const auto &tensor : cases[0].data_sets()[0].inputs)
    feeds.emplace(tensor.name, RuntimeValue(tensor.ToOwned()));
  feeds.emplace("counter", RuntimeValue(Tensor::FromInt64("counter", {}, {0})));
  RuntimeContext context;
  GenerationOptions options;
  options.max_new_tokens = 3;
  EXPECT_EQ(Tokens(Generate(model, context, feeds, options)),
            (std::vector<int64_t>{0, 1, 1, 1, 1}));
  EXPECT_EQ(feeds.at("counter").tensor.AsInt64()[0], 0);
}
