// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/feedback_state.h"
#include <future>
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

TypeProto FloatType(int64_t size = 1) {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(size);
  return type;
}

RuntimeValue Number(float value) { return RuntimeValue(Tensor::FromFloat("", {1}, {value})); }

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
      structured ? Structure({{"tokens", FloatType()}, {"cache", cache}}) : FloatType();
  if (!structured) {
    auto *tokens = graph->add_input();
    tokens->set_name("tokens");
    *tokens->mutable_type() = FloatType();
  }
  auto *output = graph->add_output();
  output->set_name(structured ? "response" : "present");
  *output->mutable_type() =
      structured ? Structure({{"logits", FloatType()}, {"cache", cache}}) : FloatType();
  auto *node = graph->add_node();
  node->set_domain("test.feedback");
  node->set_op_type("Step");
  node->add_input(structured ? "request" : "past");
  if (!structured)
    node->add_input("tokens");
  node->add_output(structured ? "response" : "present");
  return model;
}

void RegisterStep(RuntimeContext &context) {
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        const float next = rt.Get("past").AsFloat()[0] + rt.Get("tokens").AsFloat()[0];
        rt.Put(node.output(0), Tensor::FromFloat("", {1}, {next}, rt.allocator()));
      });
}

RuntimeValue Cache(float key, float value) {
  return RuntimeValue(RuntimeValueMap{{"keys", Number(key)}, {"values", Number(value)}});
}

void RegisterStructuredStep(RuntimeContext &context) {
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        const auto &request = rt.values().at(node.input(0));
        const auto &cache = request.fields.at("cache");
        const float token = Number(request.fields.at("tokens"));
        rt.values()[node.output(0)] = RuntimeValue(
            RuntimeValueMap{{"logits", Number(token)},
                            {"cache", Cache(Number(cache.fields.at("keys")) + token,
                                            Number(cache.fields.at("values")) + token)}});
      });
}

} // namespace

TEST(FeedbackState, WholeTensorMatchesManualLoopAndOwnsSnapshots) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  RuntimeValueMap initial{{"past", Number(2)}};
  FeedbackState state(model, {{"past", "present"}}, initial);
  initial.at("past").tensor.AsFloat()[0] = 100;
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
    if (earlier.empty())
      earlier = output;
  }
  EXPECT_EQ(Number(earlier.at("present")), 3);
  auto snapshot = state.Values();
  snapshot.at("past").tensor.AsFloat()[0] = 100;
  EXPECT_EQ(Number(state.Values().at("past")), 9);
  state.Reset({{"past", Number(5)}});
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(2)}}).at("present")), 7);
  state.Close();
  EXPECT_THROW(state.Values(), std::invalid_argument);
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_THROW(state.Reset({{"past", Number(1)}}), std::invalid_argument);
  EXPECT_NO_THROW(state.Close());
}

TEST(FeedbackState, RetainsOnlySelectedNestedCacheAndRequiresFreshTokens) {
  ModelProto model = Model(true);
  SimpleRawBufferAllocator allocator(20);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &allocator});
  std::weak_ptr<std::vector<float>> unselected;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    const auto &cache = request.fields.at("cache");
    float token = Number(request.fields.at("tokens"));
    auto logits = std::make_shared<std::vector<float>>(1, token);
    unselected = logits;
    RuntimeValue response(RuntimeValueMap{
        {"cache", Cache(Number(cache.fields.at("keys")) + token,
                        Number(cache.fields.at("values")) + 2 * token)},
        {"logits", RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                               reinterpret_cast<const uint8_t *>(logits->data()),
                                               sizeof(float), logits))}});
    rt.values()["response"] = std::move(response);
  });
  FeedbackState state(model, {{"request.cache", "response.cache"}},
                      {{"request.cache", Cache(0, 10)}});
  auto first = state.Run(context, {{"request.tokens", Number(2)}});
  EXPECT_TRUE(unselected.expired());
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
  auto snapshot = state.Values();
  ASSERT_EQ(snapshot.size(), 1u);
  EXPECT_EQ(Number(snapshot.at("request.cache").fields.at("keys")), 2);
  EXPECT_THROW(state.Run(context, {}), std::invalid_argument);
  EXPECT_THROW(
      state.Run(context, {{"request.cache.keys", Number(1)}, {"request.tokens", Number(3)}}),
      std::invalid_argument);
  auto second = state.Run(context, {{"request.tokens", Number(3)}});
  EXPECT_EQ(Number(second.at("response").fields.at("cache").fields.at("keys")), 5);
  EXPECT_EQ(Number(first.at("response").fields.at("cache").fields.at("keys")), 2);
  EXPECT_EQ(Number(first.at("response").fields.at("logits")), 2);
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
}

TEST(FeedbackState, RejectsInvalidMappingsInitialValuesAndModelRewrites) {
  ModelProto model = Model();
  EXPECT_THROW((FeedbackState(model, {}, {})), std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"unknown", "present"}}, {})), std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", "unknown"}}, {})), std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", "present"}}, {})), std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", "present"}},
                              {{"past", RuntimeValue(Tensor::FromInt64("", {1}, {1}))}})),
               std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", "present"}},
                              {{"past", RuntimeValue(Tensor::FromFloat("", {2}, {1, 2}))}})),
               std::invalid_argument);
  ModelProto wrong = Model();
  wrong.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT64);
  EXPECT_THROW((FeedbackState(wrong, {{"past", "present"}}, {{"past", Number(1)}})),
               std::invalid_argument);
  wrong = Model();
  *wrong.mutable_graph()->mutable_output(0)->mutable_type() = FloatType(2);
  EXPECT_THROW((FeedbackState(wrong, {{"past", "present"}}, {{"past", Number(1)}})),
               std::invalid_argument);
  FeedbackState state(model, {{"past", "present"}}, {{"past", Number(1)}});
  model.mutable_graph()->mutable_output(0)->set_name("rewritten");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(state.Run(context, {{"tokens", Number(1)}}), std::invalid_argument);
  EXPECT_THROW(state.Reset({{"past", Number(1)}}), std::invalid_argument);
  ModelProto structured = Model(true);
  EXPECT_THROW((FeedbackState(structured,
                              {{"request.cache", "response.cache"},
                               {"request.cache.keys", "response.cache.keys"}},
                              {})),
               std::invalid_argument);
}

TEST(FeedbackState, FailsTransactionallyAndKeepsRequestsIndependent) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  int mode = 0;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    rt.Get("past").AsFloat()[0] += 1;
    if (mode == 1)
      throw std::runtime_error("kernel failure after mutating invocation input");
    if (mode == 2)
      rt.Set("present", Tensor::FromInt64("", {1}, {1}));
    else if (mode == 3)
      rt.Set("present", Tensor::FromFloat("", {2}, {1, 2}));
    else if (mode != 4)
      rt.Set("present", rt.Get("past"));
  });
  FeedbackState a(model, {{"past", "present"}}, {{"past", Number(0)}});
  FeedbackState b(model, {{"past", "present"}}, {{"past", Number(10)}});
  for (int failure : {1, 2, 3, 4}) {
    mode = failure;
    EXPECT_ANY_THROW(a.Run(context, {{"tokens", Number(0)}}));
    EXPECT_EQ(Number(a.Values().at("past")), 0);
    EXPECT_EQ(Number(b.Values().at("past")), 10);
  }
  EXPECT_THROW(a.Reset({}), std::invalid_argument);
  EXPECT_EQ(Number(a.Values().at("past")), 0);
  mode = 0;
  EXPECT_EQ(Number(a.Run(context, {{"tokens", Number(0)}}).at("present")), 1);
  EXPECT_EQ(Number(b.Run(context, {{"tokens", Number(0)}}).at("present")), 11);
}

TEST(FeedbackState, CancellationAndConcurrentOperationsDoNotPublish) {
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
  FeedbackState state(model, {{"past", "present"}}, {{"past", Number(1)}});
  TaskCompletion completion(TaskId{1});
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
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}, &completion), std::invalid_argument);
}

TEST(FeedbackState, PublishesThroughTaskCompletion) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  FeedbackState state(model, {{"past", "present"}}, {{"past", Number(1)}});
  TaskCompletion completion(TaskId{1});
  state.Run(context, {{"tokens", Number(2)}}, &completion);
  EXPECT_EQ(completion.status(), TaskStatus::kSucceeded);
  EXPECT_EQ(Number(state.Values().at("past")), 3);
}

TEST(FeedbackState, RejectsReentrantOperationsFromTheActiveKernel) {
  ModelProto model = Model();
  FeedbackState state(model, {{"past", "present"}}, {{"past", Number(1)}});
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

TEST(FeedbackState, MultipleSelectedFieldsCommitTogether) {
  ModelProto model = Model(true);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  bool invalid = true;
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    RuntimeValue cache = Cache(30, 40);
    if (invalid)
      cache.fields["values"] = RuntimeValue(Tensor::FromInt64("", {1}, {40}));
    rt.values()["response"] =
        RuntimeValue(RuntimeValueMap{{"cache", std::move(cache)}, {"logits", Number(1)}});
  });
  FeedbackState state(model,
                      {{"request.cache.keys", "response.cache.keys"},
                       {"request.cache.values", "response.cache.values"}},
                      {{"request.cache.keys", Number(3)}, {"request.cache.values", Number(4)}});
  EXPECT_THROW(state.Run(context, {{"request.tokens", Number(1)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("request.cache.keys")), 3);
  EXPECT_EQ(Number(state.Values().at("request.cache.values")), 4);
  invalid = false;
  state.Run(context, {{"request.tokens", Number(1)}});
  EXPECT_EQ(Number(state.Values().at("request.cache.keys")), 30);
  EXPECT_EQ(Number(state.Values().at("request.cache.values")), 40);
}

TEST(FeedbackState, BorrowsNeitherInitialStorageNorExecutionArena) {
  ModelProto model = Model();
  SimpleRawBufferAllocator allocator(10);
  RuntimeContext context(KernelContext(DefaultOpset(18)),
                         RuntimeContextOptions{.allocator = &allocator});
  RegisterStep(context);
  auto data = std::make_shared<std::vector<float>>(1, 5);
  std::weak_ptr<std::vector<float>> owner = data;
  RuntimeValueMap initial{
      {"past", RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                           reinterpret_cast<const uint8_t *>(data->data()),
                                           sizeof(float), data))}};
  FeedbackState state(model, {{"past", "present"}}, initial);
  initial.clear();
  data.reset();
  EXPECT_TRUE(owner.expired());
  auto first = state.Run(context, {{"tokens", Number(2)}});
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
  EXPECT_FALSE(first.at("present").tensor.has_allocation());
  EXPECT_FALSE(state.Values().at("past").tensor.has_allocation());
  auto second = state.Run(context, {{"tokens", Number(3)}});
  EXPECT_EQ(Number(first.at("present")), 7);
  EXPECT_EQ(Number(second.at("present")), 10);
  RuntimeContext different(KernelContext(DefaultOpset(18)));
  RegisterStep(different);
  EXPECT_THROW(state.Run(different, {{"tokens", Number(1)}}), std::invalid_argument);
  state.Close();
  EXPECT_EQ(Number(first.at("present")), 7);
  EXPECT_EQ(allocator.TotalAllocatedSize(), 0u);
}

TEST(FeedbackState, ValidatesSharedSymbolsAndMalformedTensorStorage) {
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
  FeedbackState state(model, {{"past", "present"}}, {{"past", Number(1)}});
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

TEST(FeedbackState, ExactDottedGraphNamesAreNotOverlappingFields) {
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
  FeedbackState one(model, {{"past", "present"}}, {{"past", Number(1)}});
  EXPECT_EQ(Number(one.Run(context, {{"past.tokens", Number(2)}}).at("present")), 3);
  FeedbackState both(model, {{"past", "present"}, {"past.tokens", "present.tokens"}},
                     {{"past", Number(1)}, {"past.tokens", Number(2)}});
  both.Run(context, {});
  EXPECT_EQ(Number(both.Values().at("past")), 3);
  EXPECT_EQ(Number(both.Values().at("past.tokens")), 3);
}

TEST(FeedbackState, StructuredValuesCrossModelLocalFunctions) {
  ModelProto model = Model(true);
  auto *function = model.add_functions();
  function->set_domain("test.feedback");
  function->set_name("WrappedStep");
  function->add_input("inner_request");
  function->add_output("inner_response");
  auto *opset = function->add_opset_import();
  opset->set_domain("test.feedback");
  opset->set_version(1);
  auto *node = function->add_node();
  node->set_domain("test.feedback");
  node->set_op_type("Step");
  node->add_input("inner_request");
  node->add_output("inner_response");
  model.mutable_graph()->mutable_node(0)->set_op_type("WrappedStep");
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStructuredStep(context);
  FeedbackState state(model, {{"request.cache", "response.cache"}},
                      {{"request.cache", Cache(1, 2)}});
  state.Run(context, {{"request.tokens", Number(3)}});
  state.Run(context, {{"request.tokens", Number(4)}});
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("keys")), 8);
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("values")), 9);
}

TEST(FeedbackState, StructuredValuesCrossIfBranches) {
  ModelProto model = Model(true);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);
  GraphProto branch;
  *branch.add_node() = model.graph().node(0);
  *branch.add_output() = model.graph().output(0);
  auto *cond = model.mutable_graph()->add_input();
  cond->set_name("cond");
  cond->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::BOOL);
  cond->mutable_type()->mutable_tensor_type()->mutable_shape();
  model.mutable_graph()->clear_node();
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
  RegisterStructuredStep(context);
  FeedbackState state(model, {{"request.cache", "response.cache"}},
                      {{"request.cache", Cache(1, 2)}});
  for (uint8_t condition : {1, 0})
    state.Run(context, {{"request.tokens", Number(3)},
                        {"cond", RuntimeValue(Tensor::FromBool("", {}, {condition}))}});
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("keys")), 7);
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("values")), 8);
}

TEST(FeedbackState, RejectsExcessiveRuntimeValueNesting) {
  RuntimeValue value = Number(1);
  for (size_t i = 0; i <= RuntimeValue::kMaxDepth; ++i) {
    RuntimeValue parent;
    parent.fields.emplace("child", std::move(value));
    value = std::move(parent);
  }
  EXPECT_THROW(value.DeepCopy(), std::invalid_argument);
}

TEST(FeedbackState, OrdinarySessionReleasesStructuredIntermediates) {
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
  context.values()["request"] =
      RuntimeValue(RuntimeValueMap{{"tokens", Number(1)}, {"cache", Cache(2, 3)}});
  RuntimeSession session(model);
  session.Run(context);
  EXPECT_EQ(context.values().count("mid"), 0u);
  ASSERT_EQ(context.values().count("response"), 1u);
  const Tensor &result =
      context.values().at("response").fields.at("cache").fields.at("keys").tensor;
  EXPECT_TRUE(result.has_allocation());
  EXPECT_EQ(result.allocation_owner(), &io);
  EXPECT_EQ(result.AsFloat()[0], 2);
  EXPECT_EQ(execution.TotalAllocatedSize(), 0u);
  EXPECT_TRUE(context.Remove("response"));
  EXPECT_EQ(io.TotalAllocatedSize(), 0u);
}

TEST(FeedbackState, EncodedFieldsUseNativeCatalogueAndOwnedPayloads) {
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
      Structure({{"tokens", FloatType()}, {"cache", encoded}});
  *model.mutable_graph()->mutable_output(0)->mutable_type() =
      Structure({{"logits", FloatType()}, {"cache", encoded}});
  EncodedValueProto initial;
  initial.mutable_struct_type()->set_type_ref(7);
  initial.set_raw_data(std::string(1, '\x03'));
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    auto value = rt.values().at("request").fields.at("cache").encoded;
    value.set_raw_data(std::string(1, static_cast<char>(value.raw_data()[0] + 1)));
    rt.values()["response"] = RuntimeValue(
        RuntimeValueMap{{"logits", Number(0)}, {"cache", RuntimeValue(std::move(value))}});
  });
  FeedbackState state(model, {{"request.cache", "response.cache"}},
                      {{"request.cache", RuntimeValue(initial)}});
  initial.set_raw_data(std::string(1, '\x40'));
  auto first = state.Run(context, {{"request.tokens", Number(1)}});
  state.Run(context, {{"request.tokens", Number(2)}});
  EXPECT_EQ(state.Values().at("request.cache").encoded.raw_data()[0], 5);
  EXPECT_EQ(first.at("response").fields.at("cache").encoded.raw_data()[0], 4);
  EncodedValueProto wrong = initial;
  wrong.mutable_struct_type()->set_type_ref(8);
  EXPECT_THROW(state.Reset({{"request.cache", RuntimeValue(wrong)}}), std::invalid_argument);
}
