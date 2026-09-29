// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <gtest/gtest.h>
#include <limits>

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
  EXPECT_TRUE(result.HasPagedCacheStructure());
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
    EXPECT_TRUE(dequantized.HasPagedCacheStructure());
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

TEST(QuantizePagedCache, PartialPageCommitsThroughFixedCapacityPersistentCache) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(23);
  auto *opset = model.add_opset_import();
  opset->set_domain("ai.rt");
  opset->set_version(1);
  auto *graph = model.mutable_graph();
  graph->set_name("convert_typed_cache");
  TypeProto cache_type;
  auto *blocks = cache_type.mutable_struct_type()->mutable_structure()->add_field();
  blocks->set_name("blocks");
  auto *page = blocks->mutable_type()
                   ->mutable_sequence_type()
                   ->mutable_elem_type()
                   ->mutable_struct_type()
                   ->mutable_structure();
  for (const char *name : {"start", "length", "key", "value"}) {
    auto *field = page->add_field();
    field->set_name(name);
    auto *tensor = field->mutable_type()->mutable_tensor_type();
    const bool scalar = std::string(name) == "start" || std::string(name) == "length";
    tensor->set_elem_type(scalar ? DataType::INT64 : DataType::FLOAT);
    tensor->mutable_shape();
    if (!scalar)
      for (int64_t dimension : {1, 1, 2, 2})
        tensor->mutable_shape()->add_dim()->set_dim_value(dimension);
  }

  auto *past = graph->add_input();
  past->set_name("past");
  *past->mutable_type() = cache_type;
  auto *present = graph->add_output();
  present->set_name("present");
  *present->mutable_type() = cache_type;
  auto *binding = graph->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");
  auto *node = graph->add_node();
  node->set_domain("ai.rt");
  node->set_op_type("QuantizePagedCache");
  node->add_input("past");
  node->add_output("present");
  for (const auto &[name, dtype] :
       {std::pair{"indices", DataType::INT64}, std::pair{"key_scale", DataType::FLOAT},
        std::pair{"key_zero", DataType::FLOAT16}, std::pair{"value_scale", DataType::FLOAT},
        std::pair{"value_zero", DataType::BFLOAT16}}) {
    node->add_input(name);
    auto *input = graph->add_input();
    input->set_name(name);
    auto *tensor = input->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(dtype);
    tensor->mutable_shape();
    if (dtype == DataType::INT64)
      tensor->mutable_shape()->add_dim()->set_dim_value(1);
  }
  KernelContext kernel_context(DefaultOpset(23));
  PagedAttention append(kernel_context, [](const Tensor &, const Tensor &, int64_t) {
    return PagedAttention::Formats{{DataType::INT8, 0.25f, 0}, {DataType::INT8, 0.25f, 0}};
  });
  auto cache =
      append(Input(2, 2), Input(2, 2), Input(2, 2), PagedAttention::EmptyCache(), {}).present;
  auto partial = Blocks(cache)[0].BorrowView();
  partial.fields.at("length") = RuntimeValue(Tensor::FromInt64("", {}, {1}));
  cache.fields.at("blocks").elements.Set(0, std::move(partial));
  PersistentValueState state(model, {{"past", cache}});
  RuntimeContext context;
  const RuntimeValueMap feeds{{"indices", RuntimeValue(Tensor::FromInt64("", {1}, {0}))},
                              {"key_scale", RuntimeValue(Tensor::FromFloat("", {}, {1}))},
                              {"value_scale", RuntimeValue(Tensor::FromFloat("", {}, {1}))},
                              {"key_zero", RuntimeValue(TypeMarker(DataType::FLOAT16))},
                              {"value_zero", RuntimeValue(TypeMarker(DataType::BFLOAT16))}};
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto output = state.Run(context, feeds);
    const auto &converted = Blocks(output.at("present"))[0];
    EXPECT_EQ(converted.fields.at("key").tensor.data_type, DataType::FLOAT16);
    EXPECT_EQ(converted.fields.at("value").tensor.data_type, DataType::BFLOAT16);
    EXPECT_EQ(converted.fields.at("key").tensor.shape, (Shape{1, 1, 2, 2}));
    EXPECT_EQ(converted.fields.at("value").tensor.shape, (Shape{1, 1, 2, 2}));
    EXPECT_EQ(converted.fields.at("length").tensor.AsInt64()[0], 1);
    const auto key = DecodePagedCachePayload(converted.fields.at("key"), 1);
    const auto expected = Input(1, 2);
    for (int64_t i = 0; i < key.element_count(); ++i)
      EXPECT_FLOAT_EQ(key.AsFloat()[i], expected.AsFloat()[i]);
  }
  state.Reset(state.Values());
  auto reconstructed = RuntimeValue(state.Values().at("past").fields);
  EXPECT_NO_THROW(state.Reset({{"past", reconstructed}}));
  auto malformed = state.Values().at("past");
  malformed.fields.at("blocks").elements.Set(0, RuntimeValue{});
  EXPECT_THROW(state.Reset({{"past", malformed}}), std::invalid_argument);
  EXPECT_NO_THROW(state.Run(context, feeds));
}

TEST(QuantizePagedCache, PreservesPartialCapacityAcrossAllConversionsWithoutReadingUnusedRows) {
  KernelContext context(DefaultOpset(23));
  QuantizePagedCache quantize(context);
  const float unused = std::numeric_limits<float>::quiet_NaN();
  const auto partial_values = [unused](int64_t width) {
    std::vector<float> values(static_cast<size_t>(2 * 2 * 3 * width), unused);
    for (int64_t outer = 0; outer < 4; ++outer)
      for (int64_t column = 0; column < width; ++column)
        values[static_cast<size_t>(outer * 3 * width + column)] =
            static_cast<float>((outer + column) % 4 - 2) * 0.25f;
    return values;
  };
  RuntimeValue page;
  page.fields.emplace("start", RuntimeValue(Tensor::FromInt64("", {}, {0})));
  page.fields.emplace("length", RuntimeValue(Tensor::FromInt64("", {}, {1})));
  page.fields.emplace("key", RuntimeValue(Tensor::FromFloat("", {2, 2, 3, 3}, partial_values(3))));
  page.fields.emplace("value",
                      RuntimeValue(Tensor::FromFloat("", {2, 2, 3, 2}, partial_values(2))));
  RuntimeValue cache;
  cache.fields.emplace("blocks", RuntimeValue(std::vector<RuntimeValue>{std::move(page)}));
  cache = std::move(cache).Retain();
  const auto indices = Tensor::FromInt64("", {1}, {0});
  const auto scale = Tensor::FromFloat("", {}, {0.25f});
  for (int32_t dtype :
       {DataType::INT2, DataType::UINT2, DataType::INT4, DataType::UINT4, DataType::INT8,
        DataType::UINT8, DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    SCOPED_TRACE(dtype);
    const auto zero = dtype == DataType::UINT2   ? ZeroPoint(dtype, 2)
                      : dtype == DataType::UINT4 ? ZeroPoint(dtype, 8)
                      : dtype == DataType::UINT8 ? ZeroPoint(dtype, 128)
                                                 : TypeMarker(dtype);
    auto converted = quantize(cache, indices, scale, zero, scale, zero);
    for (int iteration = 0; iteration < 2; ++iteration) {
      SCOPED_TRACE(iteration);
      const auto &block = Blocks(converted)[0];
      EXPECT_EQ(block.fields.at("start").tensor.AsInt64()[0], 0);
      EXPECT_EQ(block.fields.at("length").tensor.AsInt64()[0], 1);
      for (const char *name : {"key", "value"}) {
        const int64_t width = std::string(name) == "key" ? 3 : 2;
        const auto decoded = DecodePagedCachePayload(block.fields.at(name), 3);
        ASSERT_EQ(decoded.shape, (Shape{2, 2, 3, width}));
        const auto &original = Blocks(cache)[0].fields.at(name).tensor;
        for (int64_t outer = 0; outer < 4; ++outer)
          for (int64_t row = 0; row < 3; ++row)
            for (int64_t column = 0; column < width; ++column) {
              const int64_t index = (outer * 3 + row) * width + column;
              EXPECT_FLOAT_EQ(decoded.AsFloat()[index], row == 0 ? original.AsFloat()[index] : 0);
            }
      }
      const auto exported = converted.ToPagedCache();
      converted = quantize(RuntimeValue::FromPagedCache(exported), indices, scale,
                           ZeroPoint(DataType::INT8, 0), scale, ZeroPoint(DataType::INT8, 0));
    }
  }
}

TEST(QuantizePagedCache, RejectsMalformedIndexStorageBeforeReadingOrReserving) {
  KernelContext context(DefaultOpset(23));
  PagedAttention append(context);
  auto cache =
      append(Input(1, 2), Input(1, 2), Input(1, 2), PagedAttention::EmptyCache(), {}).present;
  const auto *payload = Blocks(cache)[0].fields.at("key").tensor.bytes();
  QuantizePagedCache quantize(context);
  const auto scale = Tensor::FromFloat("", {}, {0.25f});
  const auto zero = ZeroPoint(DataType::INT8, 0);
  const std::vector<Tensor> invalid{
      Tensor("", DataType::INT64, {-1}, {}),
      Tensor("", DataType::INT64, {INT64_MAX}, {}),
      Tensor("", DataType::INT64, {int64_t{1} << 61}, {}),
      Tensor("", DataType::INT64, {1}, {}),
      Tensor("", DataType::INT64, {1}, std::vector<uint8_t>(7)),
      Tensor("", DataType::INT64, {1}, std::vector<uint8_t>(16)),
      Tensor("", DataType::INT64, {0}, std::vector<uint8_t>(8)),
      Tensor::FromInt64("", {}, {0}),
      Tensor::FromInt64("", {1, 1}, {0}),
      Tensor::FromFloat("", {1}, {0}),
      Tensor::Borrow("", DataType::INT64, {1}, nullptr, sizeof(int64_t)),
  };
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_THROW(quantize(cache, invalid[i], scale, zero, scale, zero), std::invalid_argument);
    EXPECT_EQ(Blocks(cache)[0].fields.at("key").tensor.bytes(), payload);
    EXPECT_EQ(Blocks(cache)[0].fields.at("key").kind, RuntimeValue::Kind::kTensor);
  }
  auto unchanged = quantize(cache, Tensor::FromInt64("", {0}, {}), scale, zero, scale, zero);
  EXPECT_EQ(Blocks(unchanged)[0].fields.at("key").tensor.bytes(), payload);
}

TEST(QuantizePagedCache, ValidatesAllPagesIncludingEmptyAndPartialSelections) {
  KernelContext context(DefaultOpset(23));
  PagedAttention append(context);
  PagedAttention::Options options;
  options.block_size = 1;
  auto cache =
      append(Input(2, 2), Input(2, 2), Input(2, 3), PagedAttention::EmptyCache(), options).present;
  const auto *first_payload = Blocks(cache)[0].fields.at("key").tensor.bytes();
  QuantizePagedCache quantize(context);
  const auto scale = Tensor::FromFloat("", {}, {0.25f});
  const auto zero = ZeroPoint(DataType::INT8, 0);
  for (int failure = 0; failure < 9; ++failure) {
    SCOPED_TRACE(failure);
    auto invalid = cache.BorrowView();
    auto page = Blocks(invalid)[1].BorrowView();
    if (failure == 0)
      page.fields.at("start") = RuntimeValue(Tensor::FromInt64("", {}, {2}));
    else if (failure == 1)
      page.fields.at("start") = RuntimeValue(Tensor::FromInt64("", {}, {INT64_MAX}));
    else if (failure == 2)
      page.fields.at("length") = RuntimeValue(Tensor::FromInt64("", {}, {0}));
    else if (failure == 3)
      page.fields.at("length") = RuntimeValue(Tensor::FromInt64("", {}, {2}));
    else if (failure == 4)
      page.fields.at("value") = RuntimeValue(Input(2, 3)).Retain();
    else if (failure == 5)
      page.fields.at("key") = RuntimeValue(Input(1, 3)).Retain();
    else if (failure == 6)
      page.fields.emplace("extra", RuntimeValue(Tensor::FromInt64("", {}, {0})));
    else if (failure == 7)
      page.fields.at("key") = RuntimeValue(Tensor("", DataType::FLOAT, {1, 1, 1, 2},
                                                  std::vector<uint8_t>(sizeof(float))))
                                  .Retain();
    else
      page.fields.erase("value");
    invalid.fields.at("blocks").elements.Set(1, std::move(page));
    for (const auto &indices : {Tensor::FromInt64("", {0}, {}), Tensor::FromInt64("", {1}, {0})}) {
      SCOPED_TRACE(indices.element_count());
      EXPECT_THROW(quantize(invalid, indices, scale, zero, scale, zero), std::invalid_argument);
      EXPECT_EQ(Blocks(invalid)[0].fields.at("key").tensor.bytes(), first_payload);
      EXPECT_EQ(Blocks(invalid)[0].fields.at("key").kind, RuntimeValue::Kind::kTensor);
    }
  }

  auto encoded = quantize(cache, Tensor::FromInt64("", {2}, {0, 1}), scale, zero, scale, zero);
  auto unchanged = quantize(encoded, Tensor::FromInt64("", {0}, {}), scale, zero, scale, zero);
  for (size_t i = 0; i < Blocks(encoded).size(); ++i)
    EXPECT_EQ(Blocks(unchanged)[i].fields.at("key").Encoded().raw_data().data(),
              Blocks(encoded)[i].fields.at("key").Encoded().raw_data().data());
  EXPECT_EQ(quantize(PagedAttention::EmptyCache(), Tensor::FromInt64("", {0}, {}), scale, zero,
                     scale, zero)
                .ToPagedCache()
                .blocks_size(),
            0);
}
