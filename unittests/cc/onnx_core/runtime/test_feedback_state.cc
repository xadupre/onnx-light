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

void Bind(ModelProto &model, const std::string &input, const std::string &output,
          std::initializer_list<std::string> input_fields = {},
          std::initializer_list<std::string> output_fields = {}) {
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name(input);
  binding->set_output_name(output);
  for (const auto &field : input_fields)
    binding->add_input_field_path(field);
  for (const auto &field : output_fields)
    binding->add_output_field_path(field);
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
  if (structured)
    Bind(model, "request", "response", {"cache"}, {"cache"});
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

RuntimeValue Cache(float key, float value) {
  return RuntimeValue(RuntimeValueMap{{"keys", Number(key)}, {"values", Number(value)}});
}

void RegisterStructuredStep(RuntimeContext &context, const uint8_t **expected = nullptr) {
  context.RegisterCustomKernel(
      "test.feedback", "Step", [expected](const NodeProto &node, RuntimeContext &rt) {
        const auto &request = rt.values().at(node.input(0));
        const auto &cache = request.fields.at("cache");
        if (expected != nullptr) {
          EXPECT_EQ(cache.fields.at("keys").tensor.bytes(), *expected);
        }
        const float token = Number(request.fields.at("tokens"));
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

TEST(FeedbackState, WholeTensorMatchesManualLoopAndSharesReadOnlyViews) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  RuntimeValueMap initial{{"past", Number(2)}};
  const uint8_t *initial_data = initial.at("past").tensor.bytes();
  FeedbackState state(model, initial);
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
  FeedbackState state(model, {{"request.cache", Cache(0, 10)}});
  auto first = state.Run(context, {{"request.tokens", Number(2)}});
  EXPECT_FALSE(unselected.expired());
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

TEST(FeedbackState, TransfersOwnedTensorOutputsWithoutCopying) {
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
  FeedbackState state(model, initial);
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

TEST(FeedbackState, TransfersOwnedStructuredOutputsWithoutCopying) {
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
  FeedbackState state(model, {{"request.cache", Cache(0, 0)}});
  auto first = state.Run(context, {{"request.tokens", Number(0)}});
  auto &response = first.at("response");
  EXPECT_EQ(response.fields.at("cache").fields.at("keys").tensor.bytes(), keys);
  EXPECT_EQ(response.fields.at("cache").fields.at("values").tensor.bytes(), values);
  EXPECT_EQ(response.fields.at("logits").tensor.bytes(), logits);
  EXPECT_EQ(state.Values().at("request.cache").fields.at("keys").tensor.bytes(), keys);
  auto second = state.Run(context, {{"request.tokens", Number(0)}});
  EXPECT_EQ(Number(second.at("response").fields.at("cache").fields.at("keys")), 2);
  state.Close();
  EXPECT_EQ(Number(response.fields.at("logits")), 10);
}

TEST(FeedbackState, ValidatesDestinationBeforeRetainingOutput) {
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
  FeedbackState state(model, {{"past", Number(1)}});
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("past")), 1);
  invalid = false;
  EXPECT_EQ(Number(state.Run(context, {{"tokens", Number(0)}}).at("present")), 2);
}

TEST(FeedbackState, RejectsInvalidDeclarationsAndInitialValues) {
  ModelProto model = Model();
  model.mutable_graph()->clear_persistent_bindings();
  EXPECT_THROW((FeedbackState(model, {})), std::invalid_argument);
  Bind(model, "unknown", "present");
  EXPECT_THROW((FeedbackState(model, {})), std::invalid_argument);
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "past", "unknown");
  EXPECT_THROW((FeedbackState(model, {})), std::invalid_argument);
  model = Model();
  EXPECT_THROW((FeedbackState(model, {})), std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", RuntimeValue(Tensor::FromInt64("", {1}, {1}))}})),
               std::invalid_argument);
  EXPECT_THROW((FeedbackState(model, {{"past", RuntimeValue(Tensor::FromFloat("", {2}, {1, 2}))}})),
               std::invalid_argument);
  ModelProto wrong = Model();
  wrong.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT64);
  EXPECT_THROW((FeedbackState(wrong, {{"past", Number(1)}})), std::invalid_argument);
  wrong = Model();
  *wrong.mutable_graph()->mutable_output(0)->mutable_type() = FloatType(2);
  EXPECT_THROW((FeedbackState(wrong, {{"past", Number(1)}})), std::invalid_argument);
  ModelProto structured = Model(true);
  Bind(structured, "request", "response", {"cache", "keys"}, {"cache", "keys"});
  EXPECT_THROW((FeedbackState(structured, {})), std::invalid_argument);
}

TEST(FeedbackState, FailsTransactionallyAndKeepsRequestsIndependent) {
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
  FeedbackState a(model, {{"past", Number(0)}});
  FeedbackState b(model, {{"past", Number(10)}});
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
  FeedbackState state(model, {{"past", Number(1)}});
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

TEST(FeedbackState, PublishesThroughTaskCompletion) {
  ModelProto model = Model();
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStep(context);
  FeedbackState state(model, {{"past", Number(1)}});
  TaskCompletion completion(TaskId{1});
  state.Run(context, {{"tokens", Number(2)}}, &completion);
  EXPECT_EQ(completion.status(), TaskStatus::kSucceeded);
  EXPECT_EQ(Number(state.Values().at("past")), 3);
}

TEST(FeedbackState, RejectsReentrantOperationsFromTheActiveKernel) {
  ModelProto model = Model();
  FeedbackState state(model, {{"past", Number(1)}});
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
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "request", "response", {"cache", "keys"}, {"cache", "keys"});
  Bind(model, "request", "response", {"cache", "values"}, {"cache", "values"});
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
                      {{"request.cache.keys", Number(3)}, {"request.cache.values", Number(4)}});
  EXPECT_THROW(state.Run(context, {{"request.tokens", Number(1)}}), std::invalid_argument);
  EXPECT_EQ(Number(state.Values().at("request.cache.keys")), 3);
  EXPECT_EQ(Number(state.Values().at("request.cache.values")), 4);
  invalid = false;
  state.Run(context, {{"request.tokens", Number(1)}});
  EXPECT_EQ(Number(state.Values().at("request.cache.keys")), 30);
  EXPECT_EQ(Number(state.Values().at("request.cache.values")), 40);
}

TEST(FeedbackState, RetainsInitialOwnerAndRejectsUnleasedExecutionArena) {
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
  const uint8_t *original = initial.at("past").tensor.bytes();
  FeedbackState state(model, initial);
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
  FeedbackState state(model, {{"past", Number(1)}});
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
  FeedbackState one(model, {{"past", Number(1)}});
  EXPECT_THROW(one.Run(context, {{"past.tokens", Number(2)}}), std::invalid_argument);
  EXPECT_EQ(Number(one.Run(context, {{R"(past\.tokens)", Number(2)}}).at("present")), 3);
  one.Close();
  ModelProto both_model = model;
  Bind(both_model, "past.tokens", "present.tokens");
  FeedbackState both(both_model, {{"past", Number(1)}, {R"(past\.tokens)", Number(2)}});
  both.Run(context, {});
  EXPECT_EQ(Number(both.Values().at("past")), 3);
  EXPECT_EQ(Number(both.Values().at(R"(past\.tokens)")), 3);
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
  RuntimeValueMap initial{{"request.cache", Cache(1, 2)}};
  const uint8_t *expected = initial.at("request.cache").fields.at("keys").tensor.bytes();
  RegisterStructuredStep(context, &expected);
  FeedbackState state(model, initial);
  state.Run(context, {{"request.tokens", Number(3)}});
  state.Run(context, {{"request.tokens", Number(4)}});
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("keys")), 8);
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("values")), 9);
  EXPECT_EQ(state.Values().at("request.cache").fields.at("keys").tensor.bytes(), expected);
}

TEST(FeedbackState, StructuredValuesCrossIfBranches) {
  ModelProto model = Model(true);
  auto *unused_initializer = model.mutable_graph()->add_initializer();
  unused_initializer->set_name("unused_initializer");
  unused_initializer->set_data_type(DataType::FLOAT);
  unused_initializer->add_dims(1);
  unused_initializer->add_float_data(4);
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
  RuntimeValueMap initial{{"request.cache", Cache(1, 2)}};
  const uint8_t *expected = initial.at("request.cache").fields.at("keys").tensor.bytes();
  RegisterStructuredStep(context, &expected);
  FeedbackState state(model, initial);
  for (uint8_t condition : {1, 0})
    state.Run(context, {{"request.tokens", Number(3)},
                        {"cond", RuntimeValue(Tensor::FromBool("", {}, {condition}))}});
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("keys")), 7);
  EXPECT_EQ(Number(state.Values().at("request.cache").fields.at("values")), 8);
  EXPECT_EQ(state.Values().at("request.cache").fields.at("keys").tensor.bytes(), expected);
}

TEST(FeedbackState, RejectsExcessiveRuntimeValueNesting) {
  RuntimeValue value = Number(1);
  for (size_t i = 0; i <= RuntimeValue::kMaxDepth; ++i) {
    RuntimeValue parent;
    parent.fields.emplace("child", std::move(value));
    value = std::move(parent);
  }
  EXPECT_THROW(value.DeepCopy(), std::invalid_argument);
  EXPECT_THROW(value.Share(), std::invalid_argument);
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
    const auto &previous = rt.values().at("request").fields.at("cache").Encoded();
    EncodedValueProto value;
    value.mutable_struct_type()->set_type_ref(7);
    value.set_raw_data(std::string(1, static_cast<char>(previous.raw_data()[0] + 1)));
    rt.values()["response"] = RuntimeValue(
        RuntimeValueMap{{"logits", Number(0)}, {"cache", RuntimeValue(std::move(value))}});
  });
  FeedbackState state(model, {{"request.cache", RuntimeValue(initial)}});
  initial.set_raw_data(std::string(1, '\x40'));
  auto first = state.Run(context, {{"request.tokens", Number(1)}});
  state.Run(context, {{"request.tokens", Number(2)}});
  EXPECT_EQ(state.Values().at("request.cache").Encoded().raw_data()[0], 5);
  EXPECT_EQ(first.at("response").fields.at("cache").Encoded().raw_data()[0], 4);
  EncodedValueProto wrong = initial;
  wrong.mutable_struct_type()->set_type_ref(8);
  EXPECT_THROW(state.Reset({{"request.cache", RuntimeValue(wrong)}}), std::invalid_argument);
}

TEST(FeedbackState, EncodedTransportRetainsTheSamePayloadAcrossResetAndClose) {
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
  FeedbackState state(model, initial);
  EXPECT_EQ(initial.at("past").Encoded().raw_data().data(), payload);
  EXPECT_EQ(state.Values().at("past").Encoded().raw_data().data(), payload);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    EXPECT_EQ(rt.values().at("past").Encoded().raw_data().data(), payload);
    rt.values()["present"] = rt.values().at("past").Share();
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

TEST(FeedbackState, EncodedExternalViewsRetainTheMessageWithoutMutatingIt) {
  auto message = std::make_shared<EncodedValueProto>();
  message->mutable_struct_type()->set_type_ref(7);
  message->set_raw_data(std::string(1, '\x05'));
  std::weak_ptr<EncodedValueProto> weak = message;
  const EncodedValueProto *original = message.get();
  const uint8_t *payload = message->raw_data().data();
  RuntimeValue view = RuntimeValue::FromEncodedView(*message, message);
  RuntimeValue retained = view.Share();
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

TEST(FeedbackState, EncodedMessageOwnerDoesNotReplaceBorrowedBackingOwner) {
  uint8_t payload = 5;
  auto message = std::make_shared<EncodedValueProto>();
  message->mutable_struct_type()->set_type_ref(7);
  message->mutable_raw_data()->assign_borrowed(&payload, 1);
  RuntimeValue view = RuntimeValue::FromEncodedView(*message, message);
  EXPECT_THROW(view.Share(), std::invalid_argument);
  const EncodedValueProto &retained = *message;
  EXPECT_EQ(retained.raw_data().data(), &payload);
}

TEST(FeedbackState, StringsAndEmptyTensorsShareRatherThanCopy) {
  for (bool strings : {false, true}) {
    ModelProto model = Model();
    auto *opset = model.add_opset_import();
    opset->set_version(18);
    auto *identity = model.mutable_graph()->mutable_node(0);
    identity->set_domain("");
    identity->set_op_type("Identity");
    identity->clear_input();
    identity->add_input("past");
    TypeProto type;
    type.mutable_tensor_type()->set_elem_type(strings ? DataType::STRING : DataType::FLOAT);
    type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(strings ? 1 : 0);
    *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
    *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
    RuntimeValueMap initial;
    initial.emplace("past", RuntimeValue(strings ? Tensor::MakeString("", {1}, {"hello"})
                                                 : Tensor::FromFloat("", {0}, {})));
    const Tensor &source = initial.at("past").tensor;
    const void *payload = strings ? static_cast<const void *>(source.AsStrings().data())
                                  : static_cast<const void *>(source.bytes());
    FeedbackState state(model, initial);
    RuntimeContext context(KernelContext(DefaultOpset(18)));
    const auto output = state.Run(context, {{"tokens", Number(0)}});
    const Tensor &result = output.at("present").tensor;
    EXPECT_EQ(strings ? static_cast<const void *>(result.AsStrings().data())
                      : static_cast<const void *>(result.bytes()),
              payload);
    state.Close();
    initial.clear();
    if (strings)
      EXPECT_EQ(result.AsStrings()[0], "hello");
    else
      EXPECT_EQ(result.size_bytes(), 0u);
  }
}

TEST(FeedbackState, RejectsOwnerlessInitialAndOutputBorrowsTransactionally) {
  ModelProto model = Model();
  float value = 3;
  const auto borrow = [&] {
    return RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1},
                                       reinterpret_cast<const uint8_t *>(&value), sizeof(float)));
  };
  EXPECT_THROW((FeedbackState(model, {{"past", borrow()}})), std::invalid_argument);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [&](const NodeProto &, RuntimeContext &rt) {
    rt.Put("present", borrow().tensor);
  });
  FeedbackState state(model, {{"past", Number(1)}});
  const auto snapshot = state.Values();
  EXPECT_THROW(state.Run(context, {{"tokens", Number(0)}}), std::invalid_argument);
  EXPECT_EQ(state.Values().at("past").tensor.bytes(), snapshot.at("past").tensor.bytes());
}

TEST(FeedbackState, IOArenaOutputsRemainLeasedUntilAllViewsAreReleased) {
  ModelProto model = Model();
  RuntimeValueMap output;
  std::weak_ptr<IOArena> weak;
  {
    auto arena = IOArena::Create(4);
    weak = arena;
    RuntimeContext context(KernelContext(DefaultOpset(18)),
                           RuntimeContextOptions{.io_allocator = arena.get()});
    RegisterStep(context);
    FeedbackState state(model, {{"past", Number(1)}});
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

TEST(FeedbackState, InitializerViewsRetainImmutableModelAndPayload) {
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
      auto *identity = model->mutable_graph()->mutable_node(0);
      identity->set_domain("");
      identity->set_op_type("Identity");
      identity->clear_input();
      identity->add_input("constant");
      RuntimeContext context(KernelContext(DefaultOpset(18)));
      context.set_model_owner(std::make_shared<int>(0));
      std::unique_ptr<FeedbackState> state;
      const RuntimeValueMap initial{{"past", Number(1)}};
      if (raw)
        state = std::make_unique<FeedbackState>(*model, initial, RuntimeSessionOptions{}, model);
      else
        state = std::make_unique<FeedbackState>(std::shared_ptr<const ModelProto>(model), initial);
      output = state->Run(context, {{"tokens", Number(0)}});
      EXPECT_EQ(output.at("present").tensor.bytes(), payload);
      EXPECT_EQ(state->Values().at("past").tensor.bytes(), payload);
      state->Close();
      model.reset();
    }
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(output.at("present").tensor.bytes(), payload);
    EXPECT_EQ(Number(output.at("present")), 4);
    output.clear();
    EXPECT_TRUE(weak.expired());
  }
}

TEST(FeedbackState, OrdinarySessionInitializersOutliveTheirSourceGraph) {
  for (bool raw : {false, true}) {
    ModelProto model = Model();
    auto *node = model.mutable_graph()->mutable_node(0);
    node->clear_input();
    node->add_input("constant");
    RuntimeSession session(model);
    const uint8_t *original = nullptr;
    {
      GraphProto source;
      auto *initializer = source.add_initializer();
      initializer->set_name("constant");
      initializer->set_data_type(DataType::FLOAT);
      initializer->add_dims(1);
      if (raw) {
        const float value = 4;
        initializer->set_raw_data(
            std::string(reinterpret_cast<const char *>(&value), sizeof(value)));
        original = initializer->raw_data().data();
      } else {
        initializer->add_float_data(4);
        original = reinterpret_cast<const uint8_t *>(initializer->float_data().values().data());
      }
      session.SetInitializers(source);
      source.clear_initializer();
    }
    RuntimeContext context(KernelContext(DefaultOpset(18)));
    context.RegisterCustomKernel(
        "test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
          rt.Put("present", Tensor::FromFloat("", {1}, {rt.Get("constant").AsFloat()[0]}));
        });
    session.Run(context);
    EXPECT_NE(context.Get("constant").bytes(), original);
    EXPECT_EQ(context.Get("present").AsFloat()[0], 4);
  }
}

TEST(FeedbackState, ModelOwnerDoesNotReplaceBorrowedInitializerBackingOwner) {
  for (bool retained_backing : {false, true}) {
    auto data = std::make_shared<std::vector<float>>(1, 4);
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
    auto *identity = model->mutable_graph()->mutable_node(0);
    identity->set_domain("");
    identity->set_op_type("Identity");
    identity->clear_input();
    identity->add_input("constant");
    FeedbackState state(std::shared_ptr<const ModelProto>(model), {{"past", Number(1)}});
    RuntimeContext context(KernelContext(DefaultOpset(18)));
    RuntimeValueMap output;
    if (retained_backing) {
      output = state.Run(context, {{"tokens", Number(0)}});
      EXPECT_EQ(output.at("present").tensor.bytes(), payload);
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
      EXPECT_EQ(output.at("present").tensor.bytes(), payload);
      EXPECT_EQ(Number(output.at("present")), 4);
      output.clear();
    }
    EXPECT_TRUE(backing_lifetime.expired());
  }
}

TEST(FeedbackState, DeclaredFieldComponentsTreatDotsLiterally) {
  ModelProto model = Model(true);
  *model.mutable_graph()->mutable_input(0)->mutable_type() =
      Structure({{"cache.part", FloatType()}, {"tokens", FloatType()}});
  *model.mutable_graph()->mutable_output(0)->mutable_type() =
      Structure({{"cache.part", FloatType()}, {"logits", FloatType()}});
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "request", "response", {"cache.part"}, {"cache.part"});
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    rt.values()["response"] = RuntimeValue(
        RuntimeValueMap{{"cache.part", rt.values().at("request").fields.at("cache.part").Share()},
                        {"logits", Number(0)}});
  });
  FeedbackState state(model, {{R"(request.cache\.part)", Number(1)}});
  const auto before = state.Values();
  const auto output = state.Run(context, {{"request.tokens", Number(0)}});
  EXPECT_EQ(output.at("response").fields.at("cache.part").tensor.bytes(),
            before.at(R"(request.cache\.part)").tensor.bytes());
}

TEST(FeedbackState, RootStructuredInitialValuesAndPartialFeedsRemainZeroCopy) {
  ModelProto model = Model(true);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  RegisterStructuredStep(context);
  RuntimeValueMap initial{{"request", RuntimeValue(RuntimeValueMap{{"cache", Cache(1, 2)}})}};
  FeedbackState state(model, initial);
  EXPECT_EQ(state.Values().at("request.cache").fields.at("keys").tensor.bytes(),
            initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes());
  const auto output =
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"tokens", Number(3)}})}});
  EXPECT_EQ(Number(output.at("response").fields.at("cache").fields.at("keys")), 4);
  EXPECT_THROW(
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"tokens", Number(3)}})},
                          {"request.tokens", Number(3)}}),
      std::invalid_argument);
  EXPECT_THROW(
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"cache", Cache(9, 9)},
                                                                   {"tokens", Number(3)}})}}),
      std::invalid_argument);
  EXPECT_THROW(state.Reset({{"request", RuntimeValue(RuntimeValueMap{{"cache", Cache(1, 2)}})},
                            {"request.cache", Cache(1, 2)}}),
               std::invalid_argument);
}

TEST(FeedbackState, LiteralDottedRootAndStructuredFieldHaveDistinctStateSlots) {
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
  RuntimeValueMap initial{{"request", RuntimeValue(RuntimeValueMap{{"cache", Cache(1, 2)}})},
                          {R"(request\.cache)", Cache(10, 20)}};
  const uint8_t *nested = initial.at("request").fields.at("cache").fields.at("keys").tensor.bytes();
  const uint8_t *literal = initial.at(R"(request\.cache)").fields.at("keys").tensor.bytes();
  FeedbackState state(model, initial);
  const auto before = state.Values();
  ASSERT_EQ(before.size(), 2u);
  EXPECT_EQ(before.at("request.cache").fields.at("keys").tensor.bytes(), nested);
  EXPECT_EQ(before.at(R"(request\.cache)").fields.at("keys").tensor.bytes(), literal);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    rt.values()["response"] =
        RuntimeValue(RuntimeValueMap{{"cache", request.fields.at("cache").Share()},
                                     {"logits", request.fields.at("tokens").Share()}});
    rt.values()["response.cache"] = rt.values().at("request.cache").Share();
  });
  const auto output =
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"tokens", Number(3)}})}});
  EXPECT_EQ(output.at("response").fields.at("cache").fields.at("keys").tensor.bytes(), nested);
  EXPECT_EQ(output.at("response.cache").fields.at("keys").tensor.bytes(), literal);
  state.Reset(before);
  EXPECT_THROW(state.Reset({{"request.cache", Cache(3, 4)}}), std::invalid_argument);
  EXPECT_EQ(state.Values().at(R"(request\.cache)").fields.at("keys").tensor.bytes(), literal);
}

TEST(FeedbackState, CanonicalEscapesDistinguishLiteralDottedFieldsAndNestedPaths) {
  ModelProto model = Model(true);
  const TypeProto type = Structure({{"cache.part", FloatType()},
                                    {"cache", Structure({{"part", FloatType()}})},
                                    {"tokens", FloatType()}});
  *model.mutable_graph()->mutable_input(0)->mutable_type() = type;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
  model.mutable_graph()->clear_persistent_bindings();
  Bind(model, "request", "response", {"cache.part"}, {"cache.part"});
  Bind(model, "request", "response", {"cache", "part"}, {"cache", "part"});
  RuntimeValueMap initial{
      {"request", RuntimeValue(RuntimeValueMap{
                      {"cache.part", Number(1)},
                      {"cache", RuntimeValue(RuntimeValueMap{{"part", Number(2)}})}})}};
  FeedbackState state(model, initial);
  const auto before = state.Values();
  EXPECT_EQ(Number(before.at(R"(request.cache\.part)")), 1);
  EXPECT_EQ(Number(before.at("request.cache.part")), 2);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    rt.values()["response"] = rt.values().at("request").Share();
  });
  const auto output =
      state.Run(context, {{"request", RuntimeValue(RuntimeValueMap{{"tokens", Number(3)}})}});
  EXPECT_EQ(output.at("response").fields.at("cache.part").tensor.bytes(),
            before.at(R"(request.cache\.part)").tensor.bytes());
  EXPECT_EQ(output.at("response").fields.at("cache").fields.at("part").tensor.bytes(),
            before.at("request.cache.part").tensor.bytes());
  state.Reset(before);
  EXPECT_THROW(state.Reset({{"request.cache.part", Number(3)}}), std::invalid_argument);
}

TEST(FeedbackState, CurrentFeedsAddressLiteralDottedFieldsThroughEscapesAndRootMaps) {
  ModelProto model = Model(true);
  const TypeProto cache = Structure({{"keys", FloatType()}, {"values", FloatType()}});
  *model.mutable_graph()->mutable_input(0)->mutable_type() =
      Structure({{"token.part", FloatType()}, {"cache", cache}});
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel("test.feedback", "Step", [](const NodeProto &, RuntimeContext &rt) {
    const auto &request = rt.values().at("request");
    rt.values()["response"] =
        RuntimeValue(RuntimeValueMap{{"cache", request.fields.at("cache").Share()},
                                     {"logits", request.fields.at("token.part").Share()}});
  });
  FeedbackState state(model, {{"request.cache", Cache(1, 2)}});
  RuntimeValueMap feeds{{"request", RuntimeValue(RuntimeValueMap{{"token.part", Number(3)}})}};
  const uint8_t *token = feeds.at("request").fields.at("token.part").tensor.bytes();
  const auto output = state.Run(context, feeds);
  EXPECT_EQ(output.at("response").fields.at("logits").tensor.bytes(), token);
  EXPECT_EQ(Number(output.at("response").fields.at("logits")), 3);
  const auto escaped = state.Run(context, {{R"(request.token\.part)", Number(4)}});
  EXPECT_EQ(Number(escaped.at("response").fields.at("logits")), 4);
  EXPECT_THROW(state.Run(context, {{"request.token.part", Number(4)}}), std::invalid_argument);
}

TEST(FeedbackState, CanonicalSelectorsEscapeBackslashesAndDotsInEveryComponent) {
  ModelProto model = Model(true);
  const std::string input_name = R"(request\part.cache)";
  const std::string output_name = R"(response\part.cache)";
  const std::string state_field = R"(cache\part.state)";
  const std::string token_field = R"(token.part\suffix)";
  const std::string root_key = R"(request\\part\.cache)";
  const std::string state_key = R"(request\\part\.cache.cache\\part\.state)";
  const std::string token_key = R"(request\\part\.cache.token\.part\\suffix)";
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
  Bind(model, input_name, output_name, {state_field}, {state_field});
  RuntimeValueMap initial{{state_key, Number(1)}};
  const uint8_t *payload = initial.at(state_key).tensor.bytes();
  FeedbackState state(model, initial);
  EXPECT_EQ(state.Values().at(state_key).tensor.bytes(), payload);
  RuntimeContext context(KernelContext(DefaultOpset(18)));
  context.RegisterCustomKernel(
      "test.feedback", "Step", [](const NodeProto &node, RuntimeContext &rt) {
        rt.values()[node.output(0)] = rt.values().at(node.input(0)).Share();
      });
  const auto flat_output = state.Run(context, {{token_key, Number(2)}});
  EXPECT_EQ(flat_output.at(output_name).fields.at(state_field).tensor.bytes(), payload);
  EXPECT_EQ(Number(flat_output.at(output_name).fields.at(token_field)), 2);
  const auto root_output =
      state.Run(context, {{root_key, RuntimeValue(RuntimeValueMap{{token_field, Number(3)}})}});
  EXPECT_EQ(Number(root_output.at(output_name).fields.at(token_field)), 3);
  state.Reset(state.Values());
  EXPECT_EQ(state.Values().at(state_key).tensor.bytes(), payload);
  for (const std::string &invalid :
       {std::string(R"(request\q)"), root_key + ".", root_key + "\\", root_key + "..field"}) {
    EXPECT_THROW(state.Run(context, {{invalid, Number(4)}}), std::invalid_argument);
  }
}

TEST(FeedbackState, StatesShareExplicitInitialStorageButPublishIndependently) {
  ModelProto model = Model();
  RuntimeValueMap initial{{"past", Number(1)}};
  FeedbackState first(model, initial);
  FeedbackState second(model, initial);
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

TEST(FeedbackState, RejectsPersistentDeclarationsInsideSubgraphs) {
  ModelProto model = Model();
  auto *attribute = model.mutable_graph()->mutable_node(0)->add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::GRAPH);
  auto *binding = attribute->mutable_g()->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");
  EXPECT_THROW((FeedbackState(model, {{"past", Number(1)}})), std::invalid_argument);
}

TEST(FeedbackState, RetainsExternalOwnersAcrossFailedInitializationUntilClose) {
  auto model = std::make_shared<ModelProto>(Model());
  std::weak_ptr<ModelProto> model_lifetime = model;
  FeedbackState state(*model, {{"past", Number(1)}});
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
