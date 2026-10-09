// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/kernel_preparation_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::runtime::KernelPreparationSlot;
using core::runtime::KernelPreparationStore;
using core::runtime::MakeOutputTensor;
using core::runtime::RuntimeValue;
using core::runtime::Tensor;

static_assert(std::is_same_v<KernelPreparationSlot, uint32_t>);

TEST(KernelPreparationStore, UsesIntegerSlotsAndAllocatorBackedRuntimeValues) {
  KernelPreparationStore store;
  const KernelPreparationSlot first = store.Bind("first");
  const KernelPreparationSlot shared = store.Bind("first");
  const KernelPreparationSlot second = store.Bind("second");

  EXPECT_EQ(first, shared);
  EXPECT_NE(first, second);
  EXPECT_EQ(store.slot_count(), 2u);
  EXPECT_FALSE(store.IsReady(first));

  Tensor tensor = MakeOutputTensor(TensorProto::FLOAT, {2}, 2 * sizeof(float), &store.allocator());
  tensor.As<float>()[0] = 1.0f;
  tensor.As<float>()[1] = 2.0f;
  store.Publish(first, RuntimeValue(std::move(tensor)));

  ASSERT_TRUE(store.IsReady(first));
  const RuntimeValue &prepared = store.Get(first);
  ASSERT_EQ(prepared.kind, RuntimeValue::Kind::kTensor);
  EXPECT_EQ(prepared.tensor.allocation_owner(), &store.allocator());
  EXPECT_FLOAT_EQ(prepared.tensor.As<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(prepared.tensor.As<float>()[1], 2.0f);
  EXPECT_EQ(store.prepared_bytes(), 2 * sizeof(float));
  EXPECT_THROW(store.Publish(first, RuntimeValue()), std::runtime_error);

  Tensor left = MakeOutputTensor(TensorProto::FLOAT, {1}, sizeof(float), &store.allocator());
  Tensor right = MakeOutputTensor(TensorProto::FLOAT, {1}, sizeof(float), &store.allocator());
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
