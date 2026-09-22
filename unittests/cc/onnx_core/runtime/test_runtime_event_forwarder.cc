// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/stream_class.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

#ifdef ONNX_LIGHT_TEST_EVENT_ALLOCATION_FAILURE
namespace {
thread_local bool fail_next_allocation = false;
}

ONNX_LIGHT_NOINLINE void *operator new(std::size_t size) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc();
}

ONNX_LIGHT_NOINLINE void *operator new[](std::size_t size) { return ::operator new(size); }
ONNX_LIGHT_NOINLINE void operator delete(void *memory) noexcept { std::free(memory); }
ONNX_LIGHT_NOINLINE void operator delete[](void *memory) noexcept { std::free(memory); }
ONNX_LIGHT_NOINLINE void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
ONNX_LIGHT_NOINLINE void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}
#endif

namespace {

struct KernelFailure {};

RuntimeContext EventContext() {
  return RuntimeContext(RuntimeContextOptions{.events_enabled = true});
}

void AddEvent(RuntimeContext &context, const char *name) {
  RuntimeEvent event;
  event.name = name;
  context.events().push_back(std::move(event));
}

static_assert(std::is_nothrow_move_constructible_v<RuntimeEvent>);
static_assert(std::is_nothrow_move_assignable_v<RuntimeEvent>);
static_assert(std::is_nothrow_destructible_v<RuntimeEventForwarder>);

} // namespace

TEST(RuntimeEventForwarder, AppendsEventsInOrderAndClearsSource) {
  auto parent = EventContext();
  auto child = EventContext();
  AddEvent(parent, "parent");
  {
    const RuntimeEventForwarder forward(child, &parent);
    AddEvent(child, "first");
    AddEvent(child, "second");
  }
  ASSERT_EQ(parent.events().size(), 3u);
  EXPECT_EQ(parent.events()[0].name, "parent");
  EXPECT_EQ(parent.events()[1].name, "first");
  EXPECT_EQ(parent.events()[2].name, "second");
  EXPECT_TRUE(child.events().empty());
}

TEST(RuntimeEventForwarder, ForwardsDuringKernelFailure) {
  auto parent = EventContext();
  auto child = EventContext();
  EXPECT_THROW(
      {
        const RuntimeEventForwarder forward(child, &parent);
        AddEvent(child, "before failure");
        throw KernelFailure{};
      },
      KernelFailure);
  ASSERT_EQ(parent.events().size(), 1u);
  EXPECT_EQ(parent.events()[0].name, "before failure");
  EXPECT_TRUE(child.events().empty());
}

TEST(RuntimeEventForwarder, IgnoresNullDisabledAndIdenticalDestinations) {
  RuntimeContext disabled;
  auto child = EventContext();
  AddEvent(child, "child");
  {
    const RuntimeEventForwarder null_parent(child, nullptr);
    const RuntimeEventForwarder disabled_parent(child, &disabled);
    const RuntimeEventForwarder same_context(child, &child);
  }
  ASSERT_EQ(child.events().size(), 1u);
  EXPECT_EQ(child.events()[0].name, "child");
  EXPECT_TRUE(disabled.events().empty());
}

#ifdef ONNX_LIGHT_TEST_EVENT_ALLOCATION_FAILURE
TEST(RuntimeEventForwarder, ReportsAllocationFailureWithoutLosingEitherLog) {
  for (const bool kernel_fails : {false, true}) {
    auto parent = EventContext();
    auto child = EventContext();
    AddEvent(parent, "parent");
    // Forces forwarding to exceed the parent's capacity on every standard library.
    while (child.events().size() <= parent.events().capacity())
      AddEvent(child, "child");
    const size_t child_count = child.events().size();
    testing::internal::CaptureStderr();
    if (kernel_fails) {
      EXPECT_THROW(
          {
            const RuntimeEventForwarder forward(child, &parent);
            fail_next_allocation = true;
            throw KernelFailure{};
          },
          KernelFailure);
    } else {
      const RuntimeEventForwarder forward(child, &parent);
      fail_next_allocation = true;
    }
    EXPECT_FALSE(fail_next_allocation);
    fail_next_allocation = false;
    const std::string warning = testing::internal::GetCapturedStderr();
    EXPECT_NE(warning.find("insufficient memory to forward runtime events"), std::string::npos);
    EXPECT_NE(warning.find("the parent event log is incomplete"), std::string::npos);
    ASSERT_EQ(parent.events().size(), 1u);
    EXPECT_EQ(parent.events()[0].name, "parent");
    ASSERT_EQ(child.events().size(), child_count);
    for (const auto &event : child.events())
      EXPECT_EQ(event.name, "child");
    {
      const RuntimeEventForwarder retry(child, &parent);
    }
    EXPECT_EQ(parent.events().size(), child_count + 1);
    EXPECT_TRUE(child.events().empty());
  }
}
#endif
