// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::PagedAttention;

namespace {

TypeProto FeedType() {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(DataType::FLOAT);
  for (int64_t dimension : {1, 1, 1, 2})
    type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
  return type;
}

ModelProto PagedModel(bool gate = false, int64_t max_tokens = 8) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("onnx_light");
  opset->set_version(1);
  auto *graph = model.mutable_graph();
  graph->set_name("paged_feedback");
  for (const auto &name : {"Q", "K", "V", "past"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    *input->mutable_type() = std::string(name) == "past" ? PagedKVCacheTypeV1() : FeedType();
  }
  for (const auto &name : {"Y", "present"}) {
    auto *output = graph->add_output();
    output->set_name(name);
    *output->mutable_type() = std::string(name) == "present" ? PagedKVCacheTypeV1() : FeedType();
  }
  auto *node = graph->add_node();
  node->set_domain("onnx_light");
  node->set_op_type("PagedAttention");
  for (const auto &name : {"Q", "K", "V", "past"})
    node->add_input(name);
  for (const auto &name : {"Y", "present"})
    node->add_output(name);
  for (const auto &[name, value] :
       std::vector<std::pair<std::string, int64_t>>{{"block_size", 2},
                                                    {"max_tokens", max_tokens},
                                                    {"key_storage_type", DataType::INT4},
                                                    {"value_storage_type", DataType::INT8}}) {
    auto *attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::INT);
    attribute->set_i(value);
  }
  auto *binding = graph->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");
  if (gate) {
    auto *check = graph->add_node();
    check->set_domain("onnx_light");
    check->set_op_type("Gate");
    check->add_input("present");
    check->add_output("gate");
    auto *output = graph->add_output();
    output->set_name("gate");
    *output->mutable_type() = FeedType();
  }
  return model;
}

void RegisterPaged(RuntimeContext &context) {
  context.RegisterKernelFn(
      "onnx_light", "PagedAttention", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
        auto kernel = std::make_unique<PagedAttention>(rt.kernel_ctx());
        kernel->set_node(node);
        return kernel;
      });
}

RuntimeValueMap Feeds(float token) {
  RuntimeValueMap feeds;
  for (const auto &name : {"Q", "K", "V"})
    feeds.emplace(name, RuntimeValue(Tensor::FromFloat("", {1, 1, 1, 2}, {token, -token})));
  return feeds;
}

const RuntimeSequence &Blocks(const RuntimeValue &cache) {
  return cache.fields.at("blocks").elements;
}

const EncodedValueProto *Key(const RuntimeValue &cache, size_t index = 0) {
  return Blocks(cache).at(index).fields.at("key").encoded.get();
}

std::string Payload(const EncodedValueProto &value) {
  return {reinterpret_cast<const char *>(value.raw_data().data()), value.raw_data().size()};
}

} // namespace

TEST(PagedAttentionFeedback, SerializedCacheInitializerResumesAttention) {
  auto model = PagedModel();
  RuntimeContext original_context;
  RegisterPaged(original_context);
  PersistentValueState original(model, {{"past", PagedAttention::EmptyCache()}});
  original.Run(original_context, Feeds(1));
  auto initial = original.Values().at("past").ToPagedCache("past");
  ASSERT_EQ(initial.blocks_size(), 1);
  EXPECT_TRUE(initial.blocks(0).has_encoded_key());
  EXPECT_TRUE(initial.blocks(0).has_encoded_value());
  auto restored_model = model;
  *restored_model.mutable_graph()->add_paged_cache_initializer() = initial;
  ModelProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(restored_model.SerializeAsString()));
  EXPECT_NO_THROW(VerifyModel(parsed));
  RuntimeContext restored_context;
  RegisterPaged(restored_context);
  PersistentValueState restored(parsed, {});
  auto expected = original.Run(original_context, Feeds(2));
  auto actual = restored.Run(restored_context, Feeds(2));
  ASSERT_EQ(actual.at("Y").tensor.size_bytes(), expected.at("Y").tensor.size_bytes());
  for (size_t i = 0; i < actual.at("Y").tensor.size_bytes() / sizeof(float); ++i)
    EXPECT_FLOAT_EQ(actual.at("Y").tensor.AsFloat()[i], expected.at("Y").tensor.AsFloat()[i]);
  EXPECT_TRUE(actual.at("present").ToPagedCache().Equals(expected.at("present").ToPagedCache()));
  restored.Reset({});
  EXPECT_EQ(restored.Values().at("past").ToPagedCache().blocks_size(), 1);
  auto snapshot = actual.at("present").ToPagedCache();
  restored.Close();
  auto value = RuntimeValue::FromPagedCache(std::move(snapshot));
  EXPECT_EQ(Blocks(value).size(), 2u);
}

TEST(PagedAttentionFeedback, PublishesOwnersWithoutChangingModelOrPriorPartialBlocks) {
  ModelProto model = PagedModel();
  VerifyModel(model);
  const auto original = model.SerializeAsString();
  RuntimeContext context(KernelContext(DefaultOpset(23)),
                         RuntimeContextOptions{.events_enabled = true});
  RegisterPaged(context);
  PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
  auto first = state.Run(context, Feeds(1));
  ASSERT_EQ(Blocks(first.at("present")).size(), 1u);
  const auto *key = Key(first.at("present"));
  const auto payload = Payload(*key);
  EXPECT_EQ(key->affine().storage_type(), DataType::INT4);
  EXPECT_EQ(Blocks(first.at("present"))[0].fields.at("value").Encoded().affine().storage_type(),
            DataType::INT8);
  EXPECT_EQ(Key(state.Values().at("past")), key);
  auto second = state.Run(context, Feeds(2));
  EXPECT_EQ(Blocks(second.at("present")).size(), 2u);
  EXPECT_EQ(Key(second.at("present")), key);
  EXPECT_EQ(Payload(*Key(first.at("present"))), payload);
  EXPECT_EQ(Blocks(first.at("present")).size(), 1u);
  EXPECT_EQ(Blocks(second.at("present"))[1].fields.at("start").tensor.AsInt64()[0], 1);
  EXPECT_EQ(Blocks(second.at("present"))[0].fields.at("length").tensor.AsInt64()[0], 1);
  for (const auto &event : context.events())
    EXPECT_EQ(event.storage_prefix_copied_bytes, 0u);
  EXPECT_EQ(model.SerializeAsString(), original);
}

TEST(PagedAttentionFeedback, NativePublicationReplacesOtherValueKinds) {
  const auto model = PagedModel();
  RuntimeContext context;
  for (auto &[name, value] : Feeds(1))
    context.PutValue(name, std::move(value));
  context.PutValue("past", PagedAttention::EmptyCache());
  context.Put("present", Tensor::FromFloat("", {1}, {0}));
  PagedAttention kernel(context.kernel_ctx());
  kernel.set_node(model.graph().node(0));
  kernel.Run(context);
  EXPECT_FALSE(context.Has("present"));
  ASSERT_EQ(context.values().count("present"), 1u);
  EXPECT_EQ(Blocks(context.values().at("present")).size(), 1u);
}

TEST(PagedAttentionFeedback, QuantizedFormatsMatchUnquantizedReferenceWithinFixtureTolerance) {
  for (int32_t storage : {DataType::INT8, DataType::UINT8, DataType::INT4, DataType::UINT4}) {
    SCOPED_TRACE(storage);
    const bool four_bit = storage == DataType::INT4 || storage == DataType::UINT4;
    const float scale = four_bit ? 0.25f : 0.01f;
    const int64_t zero = storage == DataType::UINT4 ? 8 : storage == DataType::UINT8 ? 128 : 0;
    ModelProto model = PagedModel();
    auto *node = model.mutable_graph()->mutable_node(0);
    for (auto &attribute : *node->mutable_attribute())
      if (attribute.name() == "key_storage_type" || attribute.name() == "value_storage_type")
        attribute.set_i(storage);
    for (const auto &name : {"key_scale", "value_scale"}) {
      auto *attribute = node->add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto::FLOAT);
      attribute->set_f(scale);
    }
    for (const auto &name : {"key_zero_point", "value_zero_point"}) {
      auto *attribute = node->add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto::INT);
      attribute->set_i(zero);
    }
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    RegisterPaged(context);
    PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
    onnx_kernels::kernel::Attention reference(context.kernel_ctx());
    onnx_kernels::kernel::Attention::Attributes attributes;
    attributes.is_causal = true;
    Tensor key = Tensor::FromFloat("", {1, 1, 0, 2}, {});
    Tensor value = Tensor::FromFloat("", {1, 1, 0, 2}, {});
    for (int step = 1; step <= 4; ++step) {
      auto feeds = Feeds(0.173f * step);
      auto expected = reference(feeds.at("Q").tensor, feeds.at("K").tensor, feeds.at("V").tensor,
                                attributes, nullptr, &key, &value);
      const auto actual = state.Run(context, feeds);
      for (size_t i = 0; i < 2; ++i)
        EXPECT_NEAR(actual.at("Y").tensor.AsFloat()[i], expected.Y.AsFloat()[i],
                    four_bit ? 0.15f : 0.01f);
      key = std::move(expected.present_key);
      value = std::move(expected.present_value);
    }
  }
}

TEST(PagedAttentionFeedback, CancellationAndFailureNeverPublishAppendedBlocks) {
  for (bool cancel : {false, true}) {
    SCOPED_TRACE(cancel);
    ModelProto model = PagedModel(true);
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    RegisterPaged(context);
    TaskCompletion completion(TaskId{1});
    bool fail = false;
    RuntimeValue leaked;
    context.RegisterCustomKernel(
        "onnx_light", "Gate", [&](const NodeProto &node, RuntimeContext &rt) {
          if (fail) {
            leaked = rt.values().at("present").BorrowView();
            if (cancel)
              completion.Cancel();
            else
              throw std::invalid_argument("failure after paged append");
          }
          rt.Put(node.output(0), Tensor::FromFloat("", {1, 1, 1, 2}, {0, 0}));
        });
    PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
    const auto first = state.Run(context, Feeds(1));
    const auto *key = Key(first.at("present"));
    fail = true;
    EXPECT_THROW(state.Run(context, Feeds(2), cancel ? &completion : nullptr),
                 std::invalid_argument);
    EXPECT_EQ(Blocks(state.Values().at("past")).size(), 1u);
    EXPECT_EQ(Key(state.Values().at("past")), key);
    ASSERT_EQ(Blocks(leaked).size(), 2u);
    const auto *abandoned = Key(leaked, 1);
    fail = false;
    const auto retry = state.Run(context, Feeds(3));
    EXPECT_EQ(Key(retry.at("present")), key);
    EXPECT_NE(Key(retry.at("present"), 1), abandoned);
    EXPECT_EQ(Key(leaked, 1), abandoned);
    EXPECT_EQ(Blocks(retry.at("present")).size(), 2u);
  }
}

TEST(PagedAttentionFeedback, CapacityAndIncompatibleConsumerLeaveStateUnchanged) {
  ModelProto model = PagedModel(false, 1);
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  RegisterPaged(context);
  PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
  const auto first = state.Run(context, Feeds(1));
  EXPECT_THROW(state.Run(context, Feeds(2)), std::invalid_argument);
  EXPECT_EQ(Key(state.Values().at("past")), Key(first.at("present")));
  EXPECT_EQ(Blocks(state.Values().at("past")).size(), 1u);

  ModelProto unsupported = PagedModel();
  unsupported.mutable_graph()->mutable_node(0)->set_op_type("UnregisteredConsumer");
  PersistentValueState other(unsupported, {{"past", first.at("present").BorrowView()}});
  EXPECT_THROW(other.Run(context, Feeds(2)), std::invalid_argument);
  EXPECT_EQ(Key(other.Values().at("past")), Key(first.at("present")));

  context.RegisterKernelFn(
      "onnx_light", "UnregisteredConsumer", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
        auto kernel = std::make_unique<onnx_kernels::kernel::Attention>(rt.kernel_ctx());
        kernel->set_node(node);
        return kernel;
      });
  PersistentValueState dense_consumer(unsupported, {{"past", first.at("present").BorrowView()}});
  EXPECT_THROW(dense_consumer.Run(context, Feeds(2)), std::invalid_argument);
  EXPECT_EQ(Key(dense_consumer.Values().at("past")), Key(first.at("present")));
}

TEST(PagedAttentionFeedback, ResetCloseAndRequestIsolationRetainLastLiveOwners) {
  ModelProto model = PagedModel();
  auto arena = IOArena::Create(32);
  SimpleRawBufferAllocator execution(8);
  RuntimeContext context(
      KernelContext(DefaultOpset(23)),
      RuntimeContextOptions{.allocator = &execution, .io_allocator = arena.get()});
  RegisterPaged(context);
  PersistentValueState first(model, {{"past", PagedAttention::EmptyCache()}});
  auto output = first.Run(context, Feeds(1));
  auto shared = first.Values();
  std::weak_ptr<const EncodedValueProto> lifetime =
      Blocks(shared.at("past"))[0].fields.at("key").encoded;
  const auto *original = Key(shared.at("past"));
  PersistentValueState second(model, shared);
  output.clear();
  shared.clear();
  auto branch = first.Run(context, Feeds(2));
  EXPECT_EQ(Blocks(second.Values().at("past")).size(), 1u);
  EXPECT_EQ(Key(second.Values().at("past")), original);
  auto other = second.Run(context, Feeds(3));
  EXPECT_EQ(Key(branch.at("present")), Key(other.at("present")));
  EXPECT_NE(Key(branch.at("present"), 1), Key(other.at("present"), 1));
  first.Reset({{"past", PagedAttention::EmptyCache()}});
  second.Close();
  EXPECT_FALSE(lifetime.expired());
  const auto bytes = Payload(*original);
  for (int i = 0; i < 8; ++i) {
    first.Reset({{"past", PagedAttention::EmptyCache()}});
    first.Run(context, Feeds(static_cast<float>(i)));
  }
  EXPECT_EQ(Payload(*original), bytes);
  first.Close();
  branch.clear();
  EXPECT_FALSE(lifetime.expired());
  other.clear();
  EXPECT_TRUE(lifetime.expired());
  EXPECT_THROW(first.Run(context, Feeds(1)), std::invalid_argument);
}

TEST(PagedAttentionFeedback, AllocatorExhaustionDoesNotPublishPartialCache) {
  ModelProto model = PagedModel();
  auto arena = IOArena::Create(2);
  SimpleRawBufferAllocator execution(8);
  RuntimeContext context(
      KernelContext(DefaultOpset(23)),
      RuntimeContextOptions{.allocator = &execution, .io_allocator = arena.get()});
  RegisterPaged(context);
  PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
  // K and V fill the arena, so allocation of Y fails before publication.
  EXPECT_THROW(state.Run(context, Feeds(1)), std::bad_alloc);
  EXPECT_TRUE(Blocks(state.Values().at("past")).empty());
  EXPECT_THROW(state.Run(context, Feeds(2)), std::bad_alloc);
  EXPECT_TRUE(Blocks(state.Values().at("past")).empty());
}

TEST(PagedAttentionFeedback, IntermediateCachesKeepOwnersAcrossArenaRouting) {
  for (int storage :
       {DataType::FLOAT, DataType::INT8, DataType::UINT8, DataType::INT4, DataType::UINT4}) {
    SCOPED_TRACE(storage);
    auto model = PagedModel();
    for (auto &attribute : *model.mutable_graph()->mutable_node(0)->mutable_attribute())
      if (attribute.name() == "key_storage_type" || attribute.name() == "value_storage_type")
        attribute.set_i(storage);
    model.mutable_graph()->mutable_output(1)->set_name("cache_out");
    model.mutable_graph()->mutable_persistent_bindings(0)->set_output_name("cache_out");
    auto *forward = model.mutable_graph()->add_node();
    forward->set_domain("onnx_light");
    forward->set_op_type("Forward");
    forward->add_input("present");
    forward->add_output("cache_out");
    VerifyModel(model);
    SimpleRawBufferAllocator execution(8);
    auto arena = IOArena::Create(32);
    RuntimeContext context(
        KernelContext(DefaultOpset(23)),
        RuntimeContextOptions{.allocator = &execution, .io_allocator = arena.get()});
    RegisterPaged(context);
    context.RegisterCustomKernel(
        "onnx_light", "Forward", [](const NodeProto &node, RuntimeContext &rt) {
          rt.PutValue(node.output(0), rt.values().at(node.input(0)).BorrowView());
        });
    PersistentValueState state(model, {{"past", PagedAttention::EmptyCache()}});
    auto first = state.Run(context, Feeds(1));
    auto second = state.Run(context, Feeds(2));
    EXPECT_EQ(Blocks(first.at("cache_out")).size(), 1u);
    EXPECT_EQ(Blocks(second.at("cache_out")).size(), 2u);
    EXPECT_EQ(&Blocks(first.at("cache_out"))[0], &Blocks(second.at("cache_out"))[0]);
    auto saved = first.at("cache_out").ToPagedCache().SerializeAsString();
    state.Close();
    EXPECT_EQ(first.at("cache_out").ToPagedCache().SerializeAsString(), saved);
  }
}
