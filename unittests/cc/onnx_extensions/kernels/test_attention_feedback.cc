// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/feedback_state.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <future>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

TypeProto FloatType() {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  return type;
}

RuntimeValue Number(float value) {
  return RuntimeValue(Tensor::FromFloat("", {1}, {value})).Retain();
}

float Number(const RuntimeValue &value) { return value.tensor.AsFloat()[0]; }

void Bind(ModelProto &model, const std::string &input, const std::string &output) {
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name(input);
  binding->set_output_name(output);
}

TypeProto AttentionType(int64_t batch, int64_t heads, int64_t sequence) {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(DataType::FLOAT);
  for (int64_t dimension : {batch, heads, sequence, int64_t{2}}) {
    auto *dim = type.mutable_tensor_type()->mutable_shape()->add_dim();
    if (dimension >= 0)
      dim->set_dim_value(dimension);
  }
  return type;
}

ModelProto AttentionModel(int64_t batch = 1, int64_t heads = 1, int64_t query_heads = 1,
                          bool gate = false) {
  onnx_kernels::RegisterKernelFunctions();
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(23);
  auto *graph = model.mutable_graph();
  for (const auto &name : {"Q", "K", "V", "past_key", "past_value"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    *input->mutable_type() = AttentionType(batch, name == std::string("Q") ? query_heads : heads,
                                           std::string(name).find("past") == 0 ? -1 : 1);
  }
  for (const auto &name : {"Y", "present_key", "present_value"}) {
    auto *output = graph->add_output();
    output->set_name(name);
    *output->mutable_type() = AttentionType(batch, name == std::string("Y") ? query_heads : heads,
                                            name == std::string("Y") ? 1 : -1);
  }
  auto *node = graph->add_node();
  node->set_op_type("Attention");
  for (const auto &name : {"Q", "K", "V", "", "past_key", "past_value"})
    node->add_input(name);
  for (const auto &name : {"Y", "present_key", "present_value"})
    node->add_output(name);
  auto *causal = node->add_attribute();
  causal->set_name("is_causal");
  causal->set_type(AttributeProto::INT);
  causal->set_i(1);
  Bind(model, "past_key", "present_key");
  Bind(model, "past_value", "present_value");
  if (gate) {
    auto *opset = model.add_opset_import();
    opset->set_domain("test.feedback");
    opset->set_version(1);
    auto *check = graph->add_node();
    check->set_domain("test.feedback");
    check->set_op_type("Gate");
    check->add_input("present_key");
    check->add_input("present_value");
    check->add_output("gate");
    auto *output = graph->add_output();
    output->set_name("gate");
    *output->mutable_type() = FloatType();
  }
  return model;
}

RuntimeValueMap EmptyAttentionCache(int64_t batch = 1, int64_t heads = 1) {
  RuntimeValueMap result;
  for (const auto &name : {"past_key", "past_value"})
    result.emplace(name, RuntimeValue(Tensor::FromFloat("", {batch, heads, 0, 2}, {})));
  return result;
}

RuntimeValueMap AttentionFeeds(float token, int64_t batch = 1, int64_t heads = 1,
                               int64_t query_heads = 1) {
  RuntimeValueMap result;
  for (const auto &name : {"Q", "K", "V"}) {
    const int64_t count = batch * (name == std::string("Q") ? query_heads : heads) * 2;
    std::vector<float> data(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i)
      data[static_cast<size_t>(i)] = (name == std::string("V") ? -token : token) + i * 0.25f;
    result.emplace(name,
                   RuntimeValue(Tensor::FromFloat(
                       "", {batch, name == std::string("Q") ? query_heads : heads, 1, 2}, data)));
  }
  return result;
}

void EqualAttentionTensor(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  for (int64_t i = 0; i < actual.element_count(); ++i)
    EXPECT_NEAR(actual.AsFloat()[i], expected.AsFloat()[i], 1e-6f);
}

} // namespace

TEST(FeedbackState, AttentionCacheFixedCapacityMatchesFunctionalGQAWithoutPrefixCopies) {
  for (int64_t query_heads : {1, 3}) {
    SCOPED_TRACE(query_heads);
    ModelProto model = AttentionModel(1, 1, query_heads);
    const auto serialized = model.SerializeAsString();
    RuntimeSessionOptions options;
    options.persistent_tensor_initial_capacity = 32;
    FeedbackState state(model, EmptyAttentionCache(), options);
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    onnx_kernels::kernel::Attention reference(context.kernel_ctx());
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor key = Tensor::FromFloat("", {1, 1, 0, 2}, {});
    Tensor value = Tensor::FromFloat("", {1, 1, 0, 2}, {});
    const uint8_t *key_pointer = nullptr;
    const uint8_t *value_pointer = nullptr;
    for (int step = 1; step <= 24; ++step) {
      const auto feeds = AttentionFeeds(step * 0.125f, 1, 1, query_heads);
      auto expected = reference(feeds.at("Q").tensor, feeds.at("K").tensor, feeds.at("V").tensor,
                                attrs, nullptr, &key, &value);
      const auto actual = state.Run(context, feeds);
      EqualAttentionTensor(actual.at("Y").tensor, expected.Y);
      EqualAttentionTensor(actual.at("present_key").tensor, expected.present_key);
      EqualAttentionTensor(actual.at("present_value").tensor, expected.present_value);
      const auto snapshot = state.Values();
      EXPECT_EQ(snapshot.at("past_key").tensor.bytes(), actual.at("present_key").tensor.bytes());
      if (step == 1) {
        key_pointer = actual.at("present_key").tensor.bytes();
        value_pointer = actual.at("present_value").tensor.bytes();
      }
      EXPECT_EQ(actual.at("present_key").tensor.bytes(), key_pointer);
      EXPECT_EQ(actual.at("present_value").tensor.bytes(), value_pointer);
      key = std::move(expected.present_key);
      value = std::move(expected.present_value);
    }
    const auto stats = state.PersistentStorageStats();
    EXPECT_EQ(stats.allocations, 2u);
    EXPECT_EQ(stats.allocated_bytes, 2u * 32 * 2 * sizeof(float));
    EXPECT_EQ(stats.prefix_copied_bytes, 0u);
    EXPECT_EQ(stats.append_copied_bytes, 24u * 2 * 2 * sizeof(float));
    EXPECT_EQ(stats.reuse_count, 46u);
    EXPECT_EQ(model.SerializeAsString(), serialized);
  }
}

TEST(FeedbackState, AttentionCacheSnapshotsAndOutputsBlockWritesAndSurviveResetClose) {
  ModelProto model = AttentionModel();
  FeedbackState state(model, EmptyAttentionCache());
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  auto first = state.Run(context, AttentionFeeds(1));
  auto snapshot = state.Values();
  const Tensor frozen = first.at("present_key").tensor.ToOwned();
  const uint8_t *first_pointer = frozen.bytes();
  EXPECT_NE(first_pointer, first.at("present_key").tensor.bytes());
  first_pointer = first.at("present_key").tensor.bytes();
  first.clear(); // Values() alone must prevent reuse.
  const uint8_t *second_pointer;
  {
    auto second = state.Run(context, AttentionFeeds(2));
    second_pointer = second.at("present_key").tensor.bytes();
    EXPECT_NE(second_pointer, first_pointer);
  }
  auto third = state.Run(context, AttentionFeeds(3));
  EXPECT_EQ(third.at("present_key").tensor.bytes(), second_pointer);
  auto fourth = state.Run(context, AttentionFeeds(4)); // An outstanding output also blocks reuse.
  EXPECT_NE(fourth.at("present_key").tensor.bytes(), second_pointer);
  EqualAttentionTensor(snapshot.at("past_key").tensor, frozen);
  EXPECT_EQ(third.at("present_key").tensor.shape[2], 3);
  EXPECT_FLOAT_EQ(third.at("present_key").tensor.AsFloat()[4], 3);
  const auto before_reset = state.PersistentStorageStats();
  state.Reset(EmptyAttentionCache());
  EXPECT_EQ(state.PersistentStorageStats().allocations, before_reset.allocations);
  state.Run(context, AttentionFeeds(7));
  EXPECT_EQ(state.PersistentStorageStats().allocations, before_reset.allocations + 2);
  state.Close();
  EqualAttentionTensor(snapshot.at("past_key").tensor, frozen);
  EXPECT_FLOAT_EQ(third.at("present_key").tensor.AsFloat()[4], 3);
}

TEST(FeedbackState, AttentionCacheGeometricGrowthCopiesOnlyAtCapacityBoundaries) {
  ModelProto model = AttentionModel();
  RuntimeSessionOptions options;
  options.persistent_tensor_initial_capacity = 2;
  FeedbackState state(model, EmptyAttentionCache(), options);
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  const uint8_t *previous = nullptr;
  for (int step = 1; step <= 9; ++step) {
    const auto result = state.Run(context, AttentionFeeds(static_cast<float>(step)));
    const uint8_t *pointer = result.at("present_key").tensor.bytes();
    if (step == 2 || step == 4 || step == 6 || step == 7 || step == 8) {
      EXPECT_EQ(pointer, previous);
    } else if (step > 1) {
      EXPECT_NE(pointer, previous);
    }
    previous = pointer;
  }
  const auto stats = state.PersistentStorageStats();
  EXPECT_EQ(stats.allocations, 8u);
  EXPECT_EQ(stats.allocated_bytes, 2u * (2 + 4 + 8 + 16) * 2 * sizeof(float));
  EXPECT_EQ(stats.prefix_copied_bytes, 2u * (2 + 4 + 8) * 2 * sizeof(float));
  EXPECT_EQ(stats.append_copied_bytes, 9u * 2 * 2 * sizeof(float));
  EXPECT_EQ(stats.reuse_count, 10u);
}

TEST(FeedbackState, AttentionCacheDenseMultiBatchAndMultiHeadFallbackIsMeasured) {
  for (const auto &geometry : {std::pair{2, 1}, std::pair{1, 2}, std::pair{2, 2}}) {
    const auto [batch, heads] = geometry;
    SCOPED_TRACE(batch);
    SCOPED_TRACE(heads);
    ModelProto model = AttentionModel(batch, heads, heads);
    FeedbackState state(model, EmptyAttentionCache(batch, heads));
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    onnx_kernels::kernel::Attention reference(context.kernel_ctx());
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor key = Tensor::FromFloat("", {batch, heads, 0, 2}, {});
    Tensor value = Tensor::FromFloat("", {batch, heads, 0, 2}, {});
    for (int step = 1; step <= 5; ++step) {
      const auto feeds = AttentionFeeds(step, batch, heads, heads);
      auto expected = reference(feeds.at("Q").tensor, feeds.at("K").tensor, feeds.at("V").tensor,
                                attrs, nullptr, &key, &value);
      const auto result = state.Run(context, feeds);
      EqualAttentionTensor(result.at("Y").tensor, expected.Y);
      EqualAttentionTensor(result.at("present_key").tensor, expected.present_key);
      EqualAttentionTensor(result.at("present_value").tensor, expected.present_value);
      key = std::move(expected.present_key);
      value = std::move(expected.present_value);
    }
    const uint64_t step_bytes = batch * heads * 2 * sizeof(float) * 2;
    const auto stats = state.PersistentStorageStats();
    EXPECT_EQ(stats.allocations, 10u);
    EXPECT_EQ(stats.allocated_bytes, step_bytes * 15);
    EXPECT_EQ(stats.prefix_copied_bytes, step_bytes * 10);
    EXPECT_EQ(stats.append_copied_bytes, step_bytes * 5);
    EXPECT_EQ(stats.reuse_count, 0u);
  }
}

TEST(FeedbackState, AttentionCacheInitialBorrowedOrSharedPayloadIsNeverCertified) {
  ModelProto model = AttentionModel();
  for (bool share : {false, true}) {
    SCOPED_TRACE(share);
    auto backing = std::make_shared<std::vector<float>>(32, 19.f);
    RuntimeValueMap initial;
    for (const auto &name : {"past_key", "past_value"})
      initial.emplace(
          name, RuntimeValue(Tensor::Borrow("", DataType::FLOAT, {1, 1, 1, 2},
                                            reinterpret_cast<const uint8_t *>(backing->data()),
                                            2 * sizeof(float), backing)));
    RuntimeValueMap external;
    if (share)
      external = initial;
    FeedbackState state(model, std::move(initial));
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    const auto output = state.Run(context, AttentionFeeds(2));
    EXPECT_NE(output.at("present_key").tensor.bytes(),
              reinterpret_cast<const uint8_t *>(backing->data()));
    EXPECT_EQ(*backing, std::vector<float>(32, 19.f));
    EXPECT_EQ(state.PersistentStorageStats().prefix_copied_bytes, 4 * sizeof(float));
    EXPECT_EQ(state.PersistentStorageStats().reuse_count, 0u);
  }
}

TEST(FeedbackState, AttentionCacheFailureCancellationAndLeakedOutputAreRetrySafe) {
  for (int failure_mode : {1, 2, 3}) {
    SCOPED_TRACE(failure_mode);
    ModelProto model = AttentionModel(1, 1, 1, true);
    FeedbackState state(model, EmptyAttentionCache());
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    TaskCompletion completion(TaskId{1});
    int mode = 0;
    Tensor leaked;
    context.RegisterCustomKernel("test.feedback", "Gate",
                                 [&](const NodeProto &, RuntimeContext &rt) {
                                   if (mode == 3)
                                     leaked = rt.Get("present_key").BorrowView();
                                   if (mode == 1 || mode == 3)
                                     throw std::invalid_argument("failure after cache append");
                                   if (mode == 2)
                                     completion.Cancel();
                                   rt.Put("gate", Tensor::FromFloat("", {1}, {0}));
                                 });
    state.Run(context, AttentionFeeds(1));
    const uint8_t *original = state.Values().at("past_key").tensor.bytes();
    mode = failure_mode;
    EXPECT_THROW(state.Run(context, AttentionFeeds(9), failure_mode == 2 ? &completion : nullptr),
                 std::invalid_argument);
    EXPECT_EQ(state.Values().at("past_key").tensor.bytes(), original);
    EXPECT_EQ(state.Values().at("past_key").tensor.shape[2], 1);
    EXPECT_FLOAT_EQ(state.Values().at("past_key").tensor.AsFloat()[0], 1);
    EXPECT_EQ(state.PersistentStorageStats().reuse_count, 2u);
    mode = 0;
    const auto retry = state.Run(context, AttentionFeeds(3));
    const auto &key = retry.at("present_key").tensor;
    EXPECT_EQ(key.shape[2], 2);
    EXPECT_FLOAT_EQ(key.AsFloat()[0], 1);
    EXPECT_FLOAT_EQ(key.AsFloat()[2], 3);
    if (failure_mode == 3) {
      EXPECT_NE(key.bytes(), original);
      EXPECT_FLOAT_EQ(leaked.AsFloat()[2], 9);
      EXPECT_EQ(leaked.shape[2], 2);
    } else {
      EXPECT_EQ(key.bytes(), original);
    }
  }
}

TEST(FeedbackState, AttentionCacheOneUsePermitHandlesSamePastKeyAndValue) {
  ModelProto model = AttentionModel();
  *model.mutable_graph()->mutable_node(0)->mutable_input(5) = "past_key";
  FeedbackState state(model, EmptyAttentionCache());
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  state.Run(context, AttentionFeeds(1));
  const auto output = state.Run(context, AttentionFeeds(2));
  const auto &key = output.at("present_key").tensor;
  const auto &value = output.at("present_value").tensor;
  EXPECT_NE(key.bytes(), value.bytes());
  EXPECT_FLOAT_EQ(key.AsFloat()[2], 2);
  EXPECT_FLOAT_EQ(value.AsFloat()[2], -2);
  EXPECT_FLOAT_EQ(key.AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(value.AsFloat()[0], 1);
  EXPECT_EQ(state.PersistentStorageStats().reuse_count, 1u);
  EXPECT_EQ(state.PersistentStorageStats().allocations, 3u);
}

TEST(FeedbackState, AttentionCacheBranchingRequestsCannotOverwriteEachOther) {
  ModelProto model = AttentionModel();
  FeedbackState first(model, EmptyAttentionCache());
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  first.Run(context, AttentionFeeds(1));
  FeedbackState second(model, first.Values());
  const auto a = first.Run(context, AttentionFeeds(2));
  const auto b = second.Run(context, AttentionFeeds(8));
  EXPECT_FLOAT_EQ(a.at("present_key").tensor.AsFloat()[2], 2);
  EXPECT_FLOAT_EQ(b.at("present_key").tensor.AsFloat()[2], 8);
  EXPECT_NE(a.at("present_key").tensor.bytes(), b.at("present_key").tensor.bytes());
}

TEST(FeedbackState, AttentionCacheIOLeasesSurviveStateAndAllocatorOwner) {
  ModelProto model = AttentionModel();
  RuntimeValueMap retained;
  std::weak_ptr<IOArena> weak;
  {
    auto arena = IOArena::Create(8);
    weak = arena;
    SimpleRawBufferAllocator execution(8);
    RuntimeContext context(
        KernelContext(DefaultOpset(23)),
        RuntimeContextOptions{.allocator = &execution, .io_allocator = arena.get()});
    FeedbackState state(model, EmptyAttentionCache());
    const uint8_t *pointer = nullptr;
    for (int step = 1; step <= 4; ++step) {
      auto output = state.Run(context, AttentionFeeds(step));
      if (step == 1)
        pointer = output.at("present_key").tensor.bytes();
      EXPECT_EQ(output.at("present_key").tensor.bytes(), pointer);
      EXPECT_EQ(arena->leased_count(), 2u);
      EXPECT_EQ(arena->allocated_count(), 1u); // Only Y; cache views share the two leases.
      EXPECT_EQ(arena->TotalAllocatedSize(), (2u * 16 * 2 + 2) * sizeof(float));
      const auto snapshot = state.Values();
      EXPECT_EQ(snapshot.at("past_key").tensor.bytes(), pointer);
      EXPECT_EQ(arena->leased_count(), 2u);
      EXPECT_EQ(arena->allocated_count(), 1u);
    }
    EXPECT_EQ(arena->allocated_count(), 0u);
    EXPECT_EQ(arena->TotalAllocatedSize(), 2u * 16 * 2 * sizeof(float));
    retained = state.Values();
    state.Reset(EmptyAttentionCache());
    EXPECT_EQ(arena->leased_count(), 2u);
    state.Close();
    EXPECT_EQ(retained.at("past_key").tensor.bytes(), pointer);
  }
  EXPECT_FALSE(weak.expired());
  EXPECT_FLOAT_EQ(retained.at("past_key").tensor.AsFloat()[6], 4);
  retained.clear();
  EXPECT_TRUE(weak.expired());
}

TEST(FeedbackState, AttentionCacheRejectsUnretainableAllocationAndCapacityOverflow) {
  ModelProto model = AttentionModel();
  SimpleRawBufferAllocator allocator(8);
  RuntimeContext context(KernelContext(DefaultOpset(23)),
                         RuntimeContextOptions{.allocator = &allocator});
  FeedbackState state(model, EmptyAttentionCache());
  EXPECT_THROW(state.Run(context, AttentionFeeds(1)), std::invalid_argument);
  EXPECT_EQ(state.Values().at("past_key").tensor.shape[2], 0);
  EXPECT_EQ(state.PersistentStorageStats().allocations, 1u);
  RuntimeSessionOptions options;
  options.persistent_tensor_initial_capacity = std::numeric_limits<size_t>::max();
  FeedbackState overflow(model, EmptyAttentionCache(), options);
  RuntimeContext ordinary(KernelContext(DefaultOpset(23)));
  EXPECT_THROW(overflow.Run(ordinary, AttentionFeeds(1)), std::invalid_argument);
  EXPECT_EQ(overflow.PersistentStorageStats().allocations, 0u);
}

TEST(FeedbackState, AttentionCacheIndependentConcurrentRequests) {
  ModelProto model = AttentionModel();
  auto run = [&model](float start) {
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    FeedbackState state(model, EmptyAttentionCache());
    for (int step = 0; step < 10; ++step)
      state.Run(context, AttentionFeeds(start + step));
    EXPECT_EQ(state.PersistentStorageStats().allocations, 2u);
    EXPECT_EQ(state.PersistentStorageStats().reuse_count, 18u);
    return state.Values();
  };
  auto first = std::async(std::launch::async, run, 1.f);
  auto second = std::async(std::launch::async, run, 20.f);
  const auto a = first.get();
  const auto b = second.get();
  EXPECT_FLOAT_EQ(a.at("past_key").tensor.AsFloat()[18], 10);
  EXPECT_FLOAT_EQ(b.at("past_key").tensor.AsFloat()[18], 29);
  EXPECT_NE(a.at("past_key").tensor.bytes(), b.at("past_key").tensor.bytes());
}

TEST(FeedbackState, AttentionCacheZeroCapacityUsesMeasuredFunctionalPath) {
  ModelProto model = AttentionModel();
  RuntimeSessionOptions options;
  options.persistent_tensor_initial_capacity = 0;
  FeedbackState state(model, EmptyAttentionCache(), options);
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  for (int i = 0; i < 4; ++i)
    state.Run(context, AttentionFeeds(i + 1));
  const auto stats = state.PersistentStorageStats();
  EXPECT_EQ(stats.allocations, 8u);
  EXPECT_EQ(stats.allocated_bytes, 2u * 10 * 2 * sizeof(float));
  EXPECT_EQ(stats.prefix_copied_bytes, 2u * 6 * 2 * sizeof(float));
  EXPECT_EQ(stats.append_copied_bytes, 2u * 4 * 2 * sizeof(float));
  EXPECT_EQ(stats.reuse_count, 0u);
}

TEST(FeedbackState, AttentionCacheOrdinaryInvocationBorrowsAndCopiesNeverCarryWritePermit) {
  for (bool copy : {false, true}) {
    ModelProto model = AttentionModel();
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    bool indirect = false;
    context.RegisterCustomKernel("", "Attention", [&](const NodeProto &, RuntimeContext &rt) {
      Tensor key = copy ? Tensor(rt.Get("past_key")) : rt.Get("past_key").BorrowView();
      Tensor value = copy ? Tensor(rt.Get("past_value")) : rt.Get("past_value").BorrowView();
      onnx_kernels::kernel::Attention attention(rt.kernel_ctx());
      auto result = attention(rt.Get("Q"), rt.Get("K"), rt.Get("V"),
                              onnx_kernels::kernel::Attention::Attributes{}, nullptr,
                              indirect ? &key : &rt.Get("past_key"),
                              indirect ? &value : &rt.Get("past_value"), nullptr, &rt);
      rt.Put("Y", std::move(result.Y));
      rt.Put("present_key", std::move(result.present_key));
      rt.Put("present_value", std::move(result.present_value));
    });
    FeedbackState state(model, EmptyAttentionCache());
    state.Run(context, AttentionFeeds(1));
    indirect = true;
    for (int step = 0; step < 3; ++step)
      state.Run(context, AttentionFeeds(step));
    EXPECT_EQ(state.PersistentStorageStats().reuse_count, 0u);
    EXPECT_EQ(state.PersistentStorageStats().allocations, 8u);
    EXPECT_EQ(state.PersistentStorageStats().prefix_copied_bytes, 2u * 6 * 2 * sizeof(float));
  }
}

TEST(FeedbackState, AttentionCacheRequiresExactPastToPresentDeclaration) {
  ModelProto model = AttentionModel();
  model.mutable_graph()->mutable_persistent_bindings(0)->set_output_name("present_value");
  model.mutable_graph()->mutable_persistent_bindings(1)->set_output_name("present_key");
  FeedbackState state(model, EmptyAttentionCache());
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  for (int step = 0; step < 3; ++step)
    state.Run(context, AttentionFeeds(step));
  const auto stats = state.PersistentStorageStats();
  EXPECT_EQ(stats.reuse_count, 0u);
  EXPECT_EQ(stats.allocations, 6u);
  EXPECT_EQ(stats.allocated_bytes, 2u * 6 * 2 * sizeof(float));
  EXPECT_EQ(stats.prefix_copied_bytes, 2u * 3 * 2 * sizeof(float));
}

TEST(FeedbackState, AttentionCacheChildContextsAndCopiesCannotUseInvocationPermissions) {
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    ModelProto model = AttentionModel();
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    bool use_child = false;
    context.RegisterCustomKernel("", "Attention", [&](const NodeProto &, RuntimeContext &rt) {
      const auto compute = [](RuntimeContext &active) {
        onnx_kernels::kernel::Attention attention(active.kernel_ctx());
        return attention(active.Get("Q"), active.Get("K"), active.Get("V"),
                         onnx_kernels::kernel::Attention::Attributes{}, nullptr,
                         &active.Get("past_key"), &active.Get("past_value"), nullptr, &active);
      };
      auto result = [&] {
        if (!use_child)
          return compute(rt);
        RuntimeContext child = mode == 0   ? rt.MakeFunctionContext()
                               : mode == 1 ? rt.MakeSubgraphContext("body")
                                           : rt;
        for (const auto &name : {"Q", "K", "V", "past_key", "past_value"})
          child.Put(name, rt.Get(name).BorrowView(), RuntimeEventKind::kInput);
        child.set_current_node_index(rt.current_node_index());
        return compute(child);
      }();
      rt.Put("Y", std::move(result.Y));
      rt.Put("present_key", std::move(result.present_key));
      rt.Put("present_value", std::move(result.present_value));
    });
    FeedbackState state(model, EmptyAttentionCache());
    state.Run(context, AttentionFeeds(1));
    use_child = true;
    for (int step = 2; step <= 3; ++step) {
      const auto output = state.Run(context, AttentionFeeds(step));
      EXPECT_FLOAT_EQ(output.at("present_key").tensor.AsFloat()[2 * (step - 1)], step);
    }
    EXPECT_EQ(state.PersistentStorageStats().reuse_count, 0u);
    EXPECT_EQ(state.PersistentStorageStats().allocations, 6u);
  }
}

TEST(FeedbackState, AttentionCacheDuplicateConsumersConsumePermitOnlyOnce) {
  ModelProto model = AttentionModel();
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  Tensor earlier;
  bool duplicate = false;
  context.RegisterCustomKernel("", "Attention", [&](const NodeProto &, RuntimeContext &rt) {
    onnx_kernels::kernel::Attention attention(rt.kernel_ctx());
    const auto &key = rt.Get("past_key");
    const auto &value = rt.Get("past_value");
    auto first = attention(rt.Get("Q"), rt.Get("K"), rt.Get("V"),
                           onnx_kernels::kernel::Attention::Attributes{}, nullptr, &key, &value,
                           nullptr, &rt);
    if (!duplicate) {
      rt.Put("Y", std::move(first.Y));
      rt.Put("present_key", std::move(first.present_key));
      rt.Put("present_value", std::move(first.present_value));
      return;
    }
    earlier = first.present_key.BorrowView();
    auto second = attention(rt.Get("Q"), rt.Get("V"), rt.Get("K"),
                            onnx_kernels::kernel::Attention::Attributes{}, nullptr, &key, &value,
                            nullptr, &rt);
    EXPECT_NE(first.present_key.bytes(), second.present_key.bytes());
    EXPECT_FLOAT_EQ(first.present_key.AsFloat()[2], 2);
    EXPECT_FLOAT_EQ(second.present_key.AsFloat()[2], -2);
    rt.Put("Y", std::move(second.Y));
    rt.Put("present_key", std::move(second.present_key));
    rt.Put("present_value", std::move(second.present_value));
  });
  FeedbackState state(model, EmptyAttentionCache());
  state.Run(context, AttentionFeeds(1));
  duplicate = true;
  const auto output = state.Run(context, AttentionFeeds(2));
  EXPECT_FLOAT_EQ(earlier.AsFloat()[2], 2);
  EXPECT_FLOAT_EQ(output.at("present_key").tensor.AsFloat()[2], -2);
  EXPECT_EQ(state.PersistentStorageStats().reuse_count, 2u);
  EXPECT_EQ(state.PersistentStorageStats().allocations, 4u);
}

TEST(FeedbackState, AttentionCacheReimportedViewsDoNotCertifyAppendCapacity) {
  for (bool reset : {false, true}) {
    SCOPED_TRACE(reset);
    ModelProto model = AttentionModel();
    RuntimeContext context(KernelContext(DefaultOpset(23)));
    FeedbackState original(model, EmptyAttentionCache());
    original.Run(context, AttentionFeeds(1));
    auto values = original.Values();
    const uint8_t *old_key = values.at("past_key").tensor.bytes();
    if (reset) {
      original.Reset(std::move(values));
      const auto output = original.Run(context, AttentionFeeds(2));
      EXPECT_NE(output.at("present_key").tensor.bytes(), old_key);
      EXPECT_EQ(original.PersistentStorageStats().reuse_count, 0u);
    } else {
      original.Close();
      FeedbackState imported(model, std::move(values));
      const auto output = imported.Run(context, AttentionFeeds(2));
      EXPECT_NE(output.at("present_key").tensor.bytes(), old_key);
      EXPECT_EQ(imported.PersistentStorageStats().reuse_count, 0u);
    }
  }
}

TEST(FeedbackState, AttentionCachePublishesCapacityOnlyForTheExactCandidate) {
  ModelProto model = AttentionModel();
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  context.RegisterCustomKernel("", "Attention", [](const NodeProto &, RuntimeContext &rt) {
    onnx_kernels::kernel::Attention attention(rt.kernel_ctx());
    auto result = attention(rt.Get("Q"), rt.Get("K"), rt.Get("V"),
                            onnx_kernels::kernel::Attention::Attributes{}, nullptr,
                            &rt.Get("past_key"), &rt.Get("past_value"), nullptr, &rt);
    rt.Put("Y", std::move(result.Y));
    // Replacing a candidate with an ordinary tensor must discard its capacity.
    rt.Put("present_key", result.present_key.ToOwned());
    rt.Put("present_value", result.present_value.ToOwned());
  });
  FeedbackState state(model, EmptyAttentionCache());
  for (int step = 1; step <= 3; ++step) {
    const auto output = state.Run(context, AttentionFeeds(step));
    EXPECT_FLOAT_EQ(output.at("present_key").tensor.AsFloat()[2 * (step - 1)], step);
  }
  EXPECT_EQ(state.PersistentStorageStats().reuse_count, 0u);
  EXPECT_EQ(state.PersistentStorageStats().allocations, 6u);
}

TEST(FeedbackState, UnselectedAttentionUsesExecutionArenaWithUnrelatedRetainedState) {
  for (const auto &[intermediate, with_io] : {std::pair{false, false}, std::pair{true, false},
                                              std::pair{false, true}, std::pair{true, true}}) {
    SCOPED_TRACE(intermediate);
    SCOPED_TRACE(with_io);
    ModelProto model = AttentionModel();
    auto *graph = model.mutable_graph();
    graph->clear_persistent_bindings();
    if (intermediate)
      graph->clear_output();
    auto *state_input = graph->add_input();
    state_input->set_name("state");
    *state_input->mutable_type() = FloatType();
    auto *next_output = graph->add_output();
    next_output->set_name("next");
    *next_output->mutable_type() = FloatType();
    auto *opset = model.add_opset_import();
    opset->set_domain("test.feedback");
    opset->set_version(1);
    auto *step = graph->add_node();
    step->set_domain("test.feedback");
    step->set_op_type("Unrelated");
    step->add_input("state");
    step->add_input("Y");
    step->add_output("next");
    Bind(model, "state", "next");
    SimpleRawBufferAllocator execution(20);
    auto io = IOArena::Create(20);
    RuntimeContext context(KernelContext(DefaultOpset(23)),
                           RuntimeContextOptions{.allocator = &execution,
                                                 .io_allocator = with_io ? io.get() : nullptr});
    context.RegisterCustomKernel(
        "test.feedback", "Unrelated", [](const NodeProto &, RuntimeContext &rt) {
          rt.Put("next", Tensor::FromFloat("", {1}, {rt.Get("state").AsFloat()[0] + 1}));
        });
    FeedbackState state(model, {{"state", Number(0)}});
    auto feeds = AttentionFeeds(1);
    for (const auto &name : {"past_key", "past_value"})
      feeds.emplace(name, RuntimeValue(Tensor::FromFloat("", {1, 1, 1, 2}, {0, 0})));
    const auto output = state.Run(context, feeds);
    EXPECT_FLOAT_EQ(Number(output.at("next")), 1);
    const auto stats = state.PersistentStorageStats();
    EXPECT_EQ(stats.allocations, 2u);
    EXPECT_EQ(stats.allocated_bytes, 8 * sizeof(float));
    EXPECT_EQ(stats.prefix_copied_bytes, 4 * sizeof(float));
    EXPECT_EQ(stats.append_copied_bytes, 4 * sizeof(float));
    EXPECT_EQ(stats.reuse_count, 0u);
  }
}

TEST(FeedbackState, AttentionCacheVariableAndEmptyChunksPreserveNonemptyInitialPrefix) {
  ModelProto model = AttentionModel();
  for (int input : {1, 2})
    *model.mutable_graph()->mutable_input(input)->mutable_type() = AttentionType(1, 1, -1);
  Tensor key = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 2, 3, 4});
  Tensor value = Tensor::FromFloat("", {1, 1, 2, 2}, {-1, -2, -3, -4});
  RuntimeValueMap initial{{"past_key", RuntimeValue(key)}, {"past_value", RuntimeValue(value)}};
  RuntimeSessionOptions options;
  options.persistent_tensor_initial_capacity = 8;
  FeedbackState state(model, std::move(initial), options);
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  onnx_kernels::kernel::Attention reference(context.kernel_ctx());
  onnx_kernels::kernel::Attention::Attributes attributes;
  attributes.is_causal = true;
  const uint8_t *pointer = nullptr;
  int64_t length = 2;
  for (int64_t chunk : {2, 0, 3, 2}) {
    SCOPED_TRACE(chunk);
    auto feeds = AttentionFeeds(1);
    std::vector<float> keys(static_cast<size_t>(chunk * 2));
    std::vector<float> values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = length + static_cast<float>(i) * 0.25f;
      values[i] = -keys[i];
    }
    feeds.at("K") = RuntimeValue(Tensor::FromFloat("", {1, 1, chunk, 2}, keys));
    feeds.at("V") = RuntimeValue(Tensor::FromFloat("", {1, 1, chunk, 2}, values));
    auto expected = reference(feeds.at("Q").tensor, feeds.at("K").tensor, feeds.at("V").tensor,
                              attributes, nullptr, &key, &value);
    const auto output = state.Run(context, feeds);
    EqualAttentionTensor(output.at("Y").tensor, expected.Y);
    EqualAttentionTensor(output.at("present_key").tensor, expected.present_key);
    EqualAttentionTensor(output.at("present_value").tensor, expected.present_value);
    if (pointer != nullptr && length + chunk <= 8) {
      EXPECT_EQ(output.at("present_key").tensor.bytes(), pointer);
    }
    pointer = output.at("present_key").tensor.bytes();
    length += chunk;
    key = std::move(expected.present_key);
    value = std::move(expected.present_value);
  }
  const auto stats = state.PersistentStorageStats();
  EXPECT_EQ(stats.allocations, 4u);
  EXPECT_EQ(stats.allocated_bytes, 2u * (8 + 16) * 2 * sizeof(float));
  EXPECT_EQ(stats.prefix_copied_bytes, 2u * (2 + 7) * 2 * sizeof(float));
  EXPECT_EQ(stats.append_copied_bytes, 2u * (2 + 0 + 3 + 2) * 2 * sizeof(float));
  EXPECT_EQ(stats.reuse_count, 4u);
}

TEST(FeedbackState, AttentionCacheZeroWidthValueUsesMeasuredDenseFallback) {
  ModelProto model = AttentionModel();
  for (int input : {2, 4})
    model.mutable_graph()
        ->mutable_input(input)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(3)
        ->set_dim_value(0);
  for (int output : {0, 2})
    model.mutable_graph()
        ->mutable_output(output)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(3)
        ->set_dim_value(0);
  auto initial = EmptyAttentionCache();
  initial.at("past_value") = RuntimeValue(Tensor::FromFloat("", {1, 1, 0, 0}, {}));
  FeedbackState state(model, std::move(initial));
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  for (int64_t length = 1; length <= 3; ++length) {
    auto feeds = AttentionFeeds(static_cast<float>(length));
    feeds.at("V") = RuntimeValue(Tensor::FromFloat("", {1, 1, 1, 0}, {}));
    const auto output = state.Run(context, feeds);
    EXPECT_EQ(output.at("present_value").tensor.shape, (Shape{1, 1, length, 0}));
    EXPECT_EQ(output.at("present_value").tensor.size_bytes(), 0u);
    EXPECT_EQ(output.at("Y").tensor.shape, (Shape{1, 1, 1, 0}));
    EXPECT_EQ(output.at("Y").tensor.size_bytes(), 0u);
  }
  const auto stats = state.PersistentStorageStats();
  EXPECT_EQ(stats.allocations, 4u);
  EXPECT_EQ(stats.allocated_bytes, 16u * 2 * sizeof(float));
  EXPECT_EQ(stats.prefix_copied_bytes, 0u);
  EXPECT_EQ(stats.append_copied_bytes, 3u * 2 * sizeof(float));
  EXPECT_EQ(stats.reuse_count, 2u);
}
