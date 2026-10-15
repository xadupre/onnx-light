// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/generation.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
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

ModelProto PipelineModel(const std::string &operation,
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
}

enum class PrivateValueKind {
  kInitializer,
  kEncodedInitializer,
  kPagedCacheInitializer,
  kIntermediate,
  kNestedIntermediate,
};

void AddPrivateValue(GraphProto &graph, PrivateValueKind kind, int variant) {
  constexpr const char *name = "private_collision";
  switch (kind) {
  case PrivateValueKind::kInitializer: {
    TensorProto *initializer = graph.add_initializer();
    initializer->set_name(name);
    initializer->set_data_type(TensorProto::FLOAT);
    initializer->add_dims(variant + 1);
    for (int index = 0; index <= variant; ++index)
      initializer->add_float_data(static_cast<float>(variant + index));
    break;
  }
  case PrivateValueKind::kEncodedInitializer: {
    EncodedValueProto *initializer = graph.add_encoded_initializer();
    initializer->set_name(name);
    auto *tensor = initializer->mutable_logical_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(variant + 1);
    initializer->set_raw_data(std::string(static_cast<size_t>(variant + 1), char(variant + 1)));
    break;
  }
  case PrivateValueKind::kPagedCacheInitializer: {
    PagedCacheProto *initializer = graph.add_paged_cache_initializer();
    initializer->set_name(name);
    auto *block = initializer->add_blocks();
    block->set_start(0);
    block->set_length(variant + 1);
    break;
  }
  case PrivateValueKind::kIntermediate: {
    NodeProto *node = graph.add_node();
    node->set_domain("test.pipeline");
    node->set_op_type("Private");
    node->add_input(graph.input()[0].name());
    node->add_output(name);
    break;
  }
  case PrivateValueKind::kNestedIntermediate: {
    AttributeProto *attribute = graph.mutable_node(0)->add_attribute();
    attribute->set_name("body");
    attribute->set_type(AttributeProto::GRAPH);
    NodeProto *node = attribute->mutable_g()->add_node();
    node->set_op_type("Identity");
    node->add_input(graph.input()[0].name());
    node->add_output(name);
    break;
  }
  }
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
      EXPECT_EQ(Tokens(Generate(GenerationPipeline{.prefill = &model}, context, feeds, options,
                                session_options)),
                Tokens(output));
      options.eos_token_id = 1;
      EXPECT_EQ(Tokens(Generate(model, context, feeds, options, session_options)),
                (std::vector<int64_t>{0, 1, 1}));
    }
  }
}

TEST(Generation, PipelineUsesOneContextAndDeclaredGraphNames) {
  const auto embedding = PipelineModel("Embed",
                                       {{"input_ids", TensorProto::INT64},
                                        {"attention_mask", TensorProto::INT64},
                                        {"position_ids", TensorProto::INT64}},
                                       "inputs_embeds");
  const auto prefill = PipelineModel("Prefill",
                                     {{"input_ids", TensorProto::INT64},
                                      {"inputs_embeds", TensorProto::FLOAT},
                                      {"vision", TensorProto::FLOAT}},
                                     "logits");
  RuntimeSession prefill_session(prefill);
  auto executor = prefill_session.cpu_executor();
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
  context.RegisterCustomKernel(
      "test.pipeline", "Embed", [&](const NodeProto &node, RuntimeContext &rt) {
        check_context(node, rt);
        const Tensor &ids = rt.Get("input_ids");
        EXPECT_EQ(rt.Get("attention_mask").shape, ids.shape);
        EXPECT_EQ(rt.Get("position_ids").shape, ids.shape);
        EXPECT_EQ(rt.Get("position_ids").AsInt64()[ids.element_count() - 1], ids.shape[1] - 1);
        std::vector<float> logits(static_cast<size_t>(ids.element_count()) * 3, 0);
        for (int64_t i = 0; i < ids.element_count(); ++i)
          logits[static_cast<size_t>(i) * 3 + (ids.AsInt64()[i] + 1) % 3] = 4;
        rt.Put(node.output(0), Tensor::FromFloat("", {1, ids.shape[1], 3}, logits));
      });
  context.RegisterCustomKernel(
      "test.pipeline", "Prefill", [&](const NodeProto &node, RuntimeContext &rt) {
        check_context(node, rt);
        const Tensor &ids = rt.Get("input_ids");
        EXPECT_EQ(ids.shape[1], rt.Get("inputs_embeds").shape[1]);
        EXPECT_EQ(ids.AsInt64()[ids.element_count() - 1], (ids.shape[1] - 1) % 3);
        EXPECT_FLOAT_EQ(rt.Get("vision").AsFloat()[0], 0);
        rt.Put(node.output(0), rt.Get("inputs_embeds").BorrowView());
      });
  GenerationPipeline pipeline{.prefill = &prefill, .embedding = &embedding};
  RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))},
                        {"vision", RuntimeValue(Tensor::FromFloat("", {1, 1, 3}, {0, 0, 0}))}};
  GenerationOptions options;
  options.max_new_tokens = 3;
  for (int repeat = 0; repeat < 2; ++repeat) {
    calls.clear();
    EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)),
              (std::vector<int64_t>{0, 1, 2, 0}));
    EXPECT_EQ(calls, (std::vector<std::string>{"Embed", "Prefill", "Embed", "Prefill", "Embed",
                                               "Prefill"}));
  }
}

TEST(Generation, PipelineRejectsPrivateValueNameCollisionsBeforeExecution) {
  for (bool persistent : {false, true}) {
    for (PrivateValueKind kind :
         {PrivateValueKind::kInitializer, PrivateValueKind::kEncodedInitializer,
          PrivateValueKind::kPagedCacheInitializer, PrivateValueKind::kIntermediate,
          PrivateValueKind::kNestedIntermediate}) {
      SCOPED_TRACE(persistent);
      SCOPED_TRACE(static_cast<int>(kind));
      auto embedding = PipelineModel("Embed", {{"input_ids", TensorProto::INT64}}, "inputs_embeds");
      auto language = PipelineModel("Prefill", {{"inputs_embeds", TensorProto::FLOAT}}, "logits");
      AddPrivateValue(*embedding.mutable_graph(), kind, 0);
      AddPrivateValue(*language.mutable_graph(), kind, 1);
      if (persistent) {
        auto *input = language.mutable_graph()->add_input();
        input->set_name("state");
        input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
        input->mutable_type()->mutable_tensor_type()->mutable_shape();
        auto *output = language.mutable_graph()->add_output();
        output->CopyFrom(*input);
        output->set_name("next_state");
        language.mutable_graph()->mutable_node(0)->add_input("state");
        language.mutable_graph()->mutable_node(0)->add_output("next_state");
        auto *binding = language.mutable_graph()->add_persistent_bindings();
        binding->set_input_name("state");
        binding->set_output_name("next_state");
      }
      RuntimeSession source(language);
      RuntimeContext context(RuntimeContextOptions{.cpu_executor = source.cpu_executor().get()});
      int calls = 0;
      context.RegisterCustomKernel("test.pipeline", "Embed",
                                   [&](const NodeProto &, RuntimeContext &) { ++calls; });
      context.RegisterCustomKernel("test.pipeline", "Prefill",
                                   [&](const NodeProto &, RuntimeContext &) { ++calls; });
      RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))}};
      if (persistent)
        feeds.emplace("state", RuntimeValue(Tensor::FromInt64("", {}, {0})));
      GenerationPipeline pipeline{.prefill = &language, .embedding = &embedding};
      GenerationOptions options;
      options.max_new_tokens = 1;
      for (int repeat = 0; repeat < 2; ++repeat)
        EXPECT_THROW(Generate(pipeline, context, feeds, options), std::invalid_argument);
      EXPECT_EQ(calls, 0);
    }
  }
}

TEST(Generation, PipelineScopesModelFunctionsWithoutChangingTheCallerRegistry) {
  auto embedding = PipelineModel("Stage", {{"input_ids", TensorProto::INT64}}, "inputs_embeds");
  auto *function = embedding.add_functions();
  function->set_domain("test.pipeline");
  function->set_name("Stage");
  function->add_input("X");
  function->add_output("Y");
  auto *opset = function->add_opset_import();
  opset->set_domain("test.pipeline");
  opset->set_version(1);
  auto *node = function->add_node();
  node->set_domain("test.pipeline");
  node->set_op_type("Embed");
  node->add_input("X");
  node->add_output("Y");
  for (bool persistent : {false, true}) {
    SCOPED_TRACE(persistent);
    auto prefill = PipelineModel("Stage", {{"inputs_embeds", TensorProto::FLOAT}}, "logits");
    if (persistent) {
      auto *graph = prefill.mutable_graph();
      auto *input = graph->add_input();
      input->set_name("counter");
      input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
      input->mutable_type()->mutable_tensor_type()->mutable_shape();
      auto *output = graph->add_output();
      output->CopyFrom(*input);
      output->set_name("next_counter");
      graph->mutable_node(0)->add_input("counter");
      graph->mutable_node(0)->add_output("next_counter");
      auto *binding = graph->add_persistent_bindings();
      binding->set_input_name("counter");
      binding->set_output_name("next_counter");
    }
    RuntimeSession source(prefill);
    RuntimeContext context(RuntimeContextOptions{.cpu_executor = source.cpu_executor().get()});
    context.functions().emplace("test.pipeline:Other:", &embedding.functions()[0]);
    const FunctionMap original_functions = context.functions();
    int prefill_calls = 0;
    bool fail = false;
    context.RegisterCustomKernel(
        "test.pipeline", "Embed", [](const NodeProto &node, RuntimeContext &rt) {
          const Tensor &ids = rt.Get(node.input(0));
          std::vector<float> logits(static_cast<size_t>(ids.element_count()) * 3, 0);
          for (int64_t i = 0; i < ids.element_count(); ++i)
            logits[static_cast<size_t>(i) * 3 + (ids.AsInt64()[i] + 1) % 3] = 4;
          rt.Put(node.output(0), Tensor::FromFloat("", {1, ids.shape[1], 3}, logits));
        });
    context.RegisterCustomKernel(
        "test.pipeline", "Stage", [&](const NodeProto &node, RuntimeContext &rt) {
          ++prefill_calls;
          EXT_ENFORCE_INVALID(!fail, "prefill failed");
          rt.Put(node.output(0), rt.Get(node.input(0)).BorrowView());
          if (persistent)
            rt.Put(node.output(1), Tensor::FromInt64("", {}, {rt.Get("counter").AsInt64()[0] + 1}));
        });
    RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))}};
    if (persistent)
      feeds.emplace("counter", RuntimeValue(Tensor::FromInt64("", {}, {0})));
    GenerationPipeline pipeline{.prefill = &prefill, .embedding = &embedding};
    GenerationOptions options;
    options.max_new_tokens = 3;
    for (int repeat = 0; repeat < 2; ++repeat) {
      EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)),
                (std::vector<int64_t>{0, 1, 2, 0}));
      EXPECT_EQ(context.functions(), original_functions);
    }
    EXPECT_EQ(prefill_calls, 6);
    fail = true;
    EXPECT_THROW(Generate(pipeline, context, feeds, options), std::invalid_argument);
    EXPECT_EQ(prefill_calls, 7);
    EXPECT_EQ(context.functions(), original_functions);
  }
}

TEST(Generation, PipelineSeedsExtensionInitializersAndHonorsFeedOverrides) {
  auto embedding = PipelineModel("Embed",
                                 {{"input_ids", TensorProto::INT64},
                                  {"weights", TensorProto::FLOAT},
                                  {"cache", TensorProto::FLOAT}},
                                 "inputs_embeds");
  auto *graph = embedding.mutable_graph();
  auto *weights = graph->add_encoded_initializer();
  weights->set_name("weights");
  auto *logical = weights->mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(1);
  auto *affine = weights->mutable_affine();
  affine->set_storage_type(TensorProto::INT8);
  affine->mutable_scale()->set_data_type(TensorProto::FLOAT);
  affine->mutable_scale()->add_float_data(0.5f);
  affine->mutable_zero_point()->set_data_type(TensorProto::INT8);
  affine->mutable_zero_point()->add_int32_data(0);
  weights->set_raw_data(std::string(1, '\2'));
  *graph->mutable_input(1)->mutable_type() = weights->logical_type();
  auto *cache = graph->add_paged_cache_initializer();
  cache->set_name("cache");
  auto *block = cache->add_blocks();
  block->set_start(0);
  block->set_length(1);
  for (auto *tensor : {block->mutable_key(), block->mutable_value()}) {
    tensor->set_data_type(TensorProto::FLOAT);
    for (int64_t dim : {1, 1, 1, 1})
      tensor->add_dims(dim);
    tensor->add_float_data(5);
  }
  *graph->mutable_input(2)->mutable_type() = TypeProto{};
  graph->mutable_input(2)->mutable_type()->mutable_struct_type();
  const auto prefill = PipelineModel("Prefill", {{"inputs_embeds", TensorProto::FLOAT}}, "logits");
  RuntimeSession source(prefill);
  RuntimeContext context(RuntimeContextOptions{.cpu_executor = source.cpu_executor().get()});
  bool override_weights = false;
  int calls = 0;
  context.RegisterCustomKernel(
      "test.pipeline", "Embed", [&](const NodeProto &node, RuntimeContext &rt) {
        ++calls;
        if (override_weights) {
          EXPECT_FLOAT_EQ(rt.Get("weights").AsFloat()[0], 7);
        } else {
          const auto &value = rt.values().at("weights");
          EXPECT_EQ(value.kind, RuntimeValue::Kind::kEncoded);
          EXPECT_EQ(value.Encoded().raw_data().size(), 1u);
        }
        const auto &value = rt.values().at("cache");
        EXPECT_TRUE(value.HasPagedCacheStructure());
        EXPECT_FLOAT_EQ(value.fields.at("blocks").elements[0].fields.at("key").tensor.AsFloat()[0],
                        5);
        rt.Put(node.output(0), Tensor::FromFloat("", {1, 1, 3}, {0, 4, 0}));
      });
  context.RegisterCustomKernel("test.pipeline", "Prefill",
                               [](const NodeProto &node, RuntimeContext &rt) {
                                 rt.Put(node.output(0), rt.Get(node.input(0)).BorrowView());
                               });
  GenerationPipeline pipeline{.prefill = &prefill, .embedding = &embedding};
  RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))}};
  GenerationOptions options;
  options.max_new_tokens = 2;
  EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)), (std::vector<int64_t>{0, 1, 1}));
  override_weights = true;
  feeds.emplace("weights", RuntimeValue(Tensor::FromFloat("", {1}, {7})));
  EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)), (std::vector<int64_t>{0, 1, 1}));
  EXPECT_EQ(calls, 4);
}

TEST(Generation, PipelinePreservesStructuredSequenceFeeds) {
  auto embedding =
      PipelineModel("Embed", {{"input_ids", TensorProto::INT64}, {"features", TensorProto::FLOAT}},
                    "inputs_embeds");
  auto *type = embedding.mutable_graph()->mutable_input(1)->mutable_type();
  *type = TypeProto{};
  auto *field = type->mutable_sequence_type()
                    ->mutable_elem_type()
                    ->mutable_struct_type()
                    ->mutable_structure()
                    ->add_field();
  field->set_name("value");
  auto *tensor_type = field->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto::FLOAT);
  tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  const auto prefill = PipelineModel("Prefill", {{"inputs_embeds", TensorProto::FLOAT}}, "logits");
  RuntimeSession source(prefill);
  RuntimeContext context(RuntimeContextOptions{.cpu_executor = source.cpu_executor().get()});
  RuntimeValueMap feeds{
      {"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 1}, {0}))},
      {"features", RuntimeValue(std::vector<RuntimeValue>{RuntimeValue(RuntimeValueMap{
                       {"value", RuntimeValue(Tensor::FromFloat("", {1}, {3}))}})})}};
  const auto *payload = feeds.at("features").elements[0].fields.at("value").tensor.bytes();
  size_t count = 1;
  int calls = 0;
  context.RegisterCustomKernel(
      "test.pipeline", "Embed", [&](const NodeProto &node, RuntimeContext &rt) {
        ++calls;
        EXPECT_FALSE(rt.HasSequence("features"));
        const auto &value = rt.values().at("features");
        EXPECT_EQ(value.kind, RuntimeValue::Kind::kSequence);
        EXPECT_EQ(value.elements.size(), count);
        if (count) {
          EXPECT_EQ(value.elements[0].kind, RuntimeValue::Kind::kStruct);
          EXPECT_EQ(value.elements[0].fields.at("value").tensor.bytes(), payload);
          EXPECT_FLOAT_EQ(value.elements[0].fields.at("value").tensor.AsFloat()[0], 3);
        }
        rt.Put(node.output(0), Tensor::FromFloat("", {1, 1, 3}, {0, 4, 0}));
      });
  context.RegisterCustomKernel("test.pipeline", "Prefill",
                               [](const NodeProto &node, RuntimeContext &rt) {
                                 rt.Put(node.output(0), rt.Get(node.input(0)).BorrowView());
                               });
  GenerationPipeline pipeline{.prefill = &prefill, .embedding = &embedding};
  GenerationOptions options;
  options.max_new_tokens = 2;
  EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)), (std::vector<int64_t>{0, 1, 1}));
  count = 0;
  feeds.insert_or_assign("features", RuntimeValue(std::vector<RuntimeValue>{}));
  EXPECT_EQ(Tokens(Generate(pipeline, context, feeds, options)), (std::vector<int64_t>{0, 1, 1}));
  EXPECT_EQ(calls, 4);
}

TEST(Generation, PipelineRequiresItsPrefillModel) {
  RuntimeContext context;
  EXPECT_THROW(Generate(GenerationPipeline{}, context, {}), std::invalid_argument);
}

TEST(Generation, GeneratedMasksAndPositionsOutliveGenerationAndContextValues) {
  const auto model = PipelineModel("Step",
                                   {{"input_ids", TensorProto::INT64},
                                    {"attention_mask", TensorProto::INT64},
                                    {"position_ids", TensorProto::INT64}},
                                   "logits");
  RuntimeSession source(model);
  for (bool use_pipeline : {false, true}) {
    SCOPED_TRACE(use_pipeline);
    RuntimeContext context(RuntimeContextOptions{.cpu_executor = source.cpu_executor().get()});
    context.RegisterCustomKernel(
        "test.pipeline", "Step", [](const NodeProto &node, RuntimeContext &rt) {
          rt.Put(node.output(0), Tensor::FromFloat("", {1, 1, 3}, {0, 4, 0}));
        });
    RuntimeValueMap feeds{{"input_ids", RuntimeValue(Tensor::FromInt64("", {1, 2}, {0, 1}))},
                          {"attention_mask", RuntimeValue(Tensor::FromInt64("", {1, 2}, {0, 1}))},
                          {"position_ids", RuntimeValue(Tensor::FromInt64("", {1, 2}, {0, 7}))}};
    GenerationOptions options;
    options.max_new_tokens = 2;
    const auto tokens =
        use_pipeline ? Generate(GenerationPipeline{.prefill = &model}, context, feeds, options)
                     : Generate(model, context, feeds, options);
    EXPECT_EQ(Tokens(tokens), (std::vector<int64_t>{0, 1, 1, 1}));
    Tensor mask = context.Get("attention_mask").BorrowView();
    Tensor positions = context.Get("position_ids").BorrowView();
    EXPECT_GT(mask.borrowed_owner().use_count(), 0);
    EXPECT_GT(positions.borrowed_owner().use_count(), 0);
    EXPECT_EQ(mask.shape, (Shape{1, 3}));
    EXPECT_EQ(positions.shape, (Shape{1, 3}));
    context.Clear();
    EXPECT_EQ(Tokens(mask), (std::vector<int64_t>{0, 1, 1}));
    EXPECT_EQ(Tokens(positions), (std::vector<int64_t>{0, 7, 8}));
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
