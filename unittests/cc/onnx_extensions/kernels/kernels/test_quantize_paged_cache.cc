// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::DecodePagedCachePayload;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::PagedAttention;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::QuantizePagedCache;

namespace {

Tensor Input(int64_t length, int64_t width, float offset = 0) {
  std::vector<float> values(static_cast<size_t>(length * width));
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = offset + static_cast<float>(static_cast<int>(i) - 3) * 0.25f;
  return Tensor::FromFloat("", {1, 1, length, width}, values);
}

const RuntimeSequence &Blocks(const RuntimeValue &cache) {
  return cache.fields.at("blocks").elements;
}

Tensor ZeroPoint(int32_t type, uint8_t value) { return Tensor("", type, {}, {value}); }

Tensor TypeMarker(int32_t type) {
  return Tensor("", type, {}, std::vector<uint8_t>(PackedByteSize(type, 1), 0));
}

} // namespace

TEST(QuantizePagedCache, QuantizesSelectedBlocksAndRequantizesEncodedPayloads) {
  KernelContext context(DefaultOpset(23));
  PagedAttention append(context);
  PagedAttention::Options options;
  options.block_size = 2;
  auto initial =
      append(Input(4, 2), Input(4, 2), Input(4, 3, 1), PagedAttention::EmptyCache(), options)
          .present;
  ASSERT_EQ(Blocks(initial).size(), 2u);
  const auto unselected_key = Blocks(initial)[0].fields.at("key").tensor.bytes();
  const auto selected_key = Blocks(initial)[1].fields.at("key").tensor;

  QuantizePagedCache quantize(context);
  auto result = quantize(initial, Tensor::FromInt64("", {1}, {1}),
                         Tensor::FromFloat("", {}, {0.25f}), ZeroPoint(DataType::INT2, 2),
                         Tensor::FromFloat("", {}, {0.25f}), ZeroPoint(DataType::UINT8, 128));

  ASSERT_EQ(Blocks(result).size(), 2u);
  EXPECT_EQ(Blocks(result)[0].fields.at("key").tensor.bytes(), unselected_key);
  EXPECT_EQ(Blocks(initial)[1].fields.at("key").tensor.bytes(), selected_key.bytes());
  ASSERT_TRUE(Blocks(result)[1].fields.at("key").encoded);
  ASSERT_TRUE(Blocks(result)[1].fields.at("value").encoded);
  EXPECT_EQ(Blocks(result)[1].fields.at("key").Encoded().affine().storage_type(), DataType::INT2);
  EXPECT_EQ(Blocks(result)[1].fields.at("value").Encoded().affine().storage_type(),
            DataType::UINT8);

  const Tensor decoded_key = DecodePagedCachePayload(Blocks(result)[1].fields.at("key"), 2);
  ASSERT_EQ(decoded_key.shape, selected_key.shape);
  for (int64_t i = 0; i < decoded_key.element_count(); ++i)
    EXPECT_NEAR(decoded_key.AsFloat()[i], selected_key.AsFloat()[i], 0.25f);

  auto requantized = quantize(result, Tensor::FromInt64("", {1}, {1}),
                              Tensor::FromFloat("", {}, {0.5f}), ZeroPoint(DataType::INT8, 0),
                              Tensor::FromFloat("", {}, {0.5f}), ZeroPoint(DataType::UINT4, 8));
  EXPECT_EQ(Blocks(requantized)[1].fields.at("key").Encoded().affine().storage_type(),
            DataType::INT8);
  EXPECT_EQ(Blocks(requantized)[1].fields.at("value").Encoded().affine().storage_type(),
            DataType::UINT4);

  for (const auto &[key_type, value_type] : {std::pair{DataType::FLOAT, DataType::FLOAT},
                                             std::pair{DataType::FLOAT16, DataType::BFLOAT16}}) {
    auto dequantized =
        quantize(result, Tensor::FromInt64("", {1}, {1}), Tensor::FromFloat("", {}, {1}),
                 TypeMarker(key_type), Tensor::FromFloat("", {}, {1}), TypeMarker(value_type));
    const auto &key = Blocks(dequantized)[1].fields.at("key");
    const auto &value = Blocks(dequantized)[1].fields.at("value");
    EXPECT_EQ(key.kind, RuntimeValue::Kind::kTensor);
    EXPECT_EQ(value.kind, RuntimeValue::Kind::kTensor);
    EXPECT_EQ(key.tensor.data_type, key_type);
    EXPECT_EQ(value.tensor.data_type, value_type);
    const Tensor decoded = DecodePagedCachePayload(key, 2);
    for (int64_t i = 0; i < decoded.element_count(); ++i)
      EXPECT_NEAR(decoded.AsFloat()[i], decoded_key.AsFloat()[i], 0.001f);
    const auto serialized = dequantized.ToPagedCache("converted");
    EXPECT_EQ(serialized.blocks(1).key().data_type(), key_type);
    EXPECT_EQ(serialized.blocks(1).value().data_type(), value_type);
    const auto restored = RuntimeValue::FromPagedCache(serialized);
    EXPECT_EQ(Blocks(restored)[1].fields.at("key").tensor.data_type, key_type);
  }
}

TEST(QuantizePagedCache, RejectsInvalidIndicesAndParametersWithoutChangingInput) {
  KernelContext context(DefaultOpset(23));
  PagedAttention append(context);
  auto cache =
      append(Input(1, 2), Input(1, 2), Input(1, 2), PagedAttention::EmptyCache(), {}).present;
  const auto pointer = Blocks(cache)[0].fields.at("key").tensor.bytes();
  QuantizePagedCache quantize(context);
  const auto scale = Tensor::FromFloat("", {}, {0.25f});
  const auto zero = ZeroPoint(DataType::INT8, 0);

  EXPECT_THROW(quantize(cache, Tensor::FromInt64("", {2}, {0, 0}), scale, zero, scale, zero),
               std::invalid_argument);
  EXPECT_THROW(quantize(cache, Tensor::FromInt64("", {1}, {1}), scale, zero, scale, zero),
               std::invalid_argument);
  EXPECT_THROW(quantize(cache, Tensor::FromInt64("", {1}, {0}), Tensor::FromFloat("", {}, {0}),
                        zero, scale, zero),
               std::invalid_argument);
  EXPECT_EQ(Blocks(cache)[0].fields.at("key").tensor.bytes(), pointer);
  EXPECT_FALSE(Blocks(cache)[0].fields.at("key").encoded);
}
