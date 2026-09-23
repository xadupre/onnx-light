// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::Attention;
using ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::PagedAttention;

namespace {

Tensor Input(int64_t length, int64_t width, float offset = 0) {
  std::vector<float> values(static_cast<size_t>(length * width));
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = offset + static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
  return Tensor::FromFloat("", {1, 1, length, width}, values);
}

const std::vector<RuntimeValue> &Pages(const RuntimeValue &cache) {
  return cache.fields.at("blocks").elements;
}

void Near(const Tensor &actual, const Tensor &expected, float tolerance = 2e-6f) {
  ASSERT_EQ(actual.shape, expected.shape);
  for (int64_t i = 0; i < actual.element_count(); ++i)
    EXPECT_NEAR(actual.AsFloat()[i], expected.AsFloat()[i], tolerance);
}

RuntimeValue CachePage(int64_t start, int64_t length, RuntimeValue key, RuntimeValue value) {
  RuntimeValue page;
  page.fields.emplace("start", RuntimeValue(Tensor::FromInt64("", {}, {start})).Retain());
  page.fields.emplace("length", RuntimeValue(Tensor::FromInt64("", {}, {length})).Retain());
  page.fields.emplace("key", std::move(key));
  page.fields.emplace("value", std::move(value));
  RuntimeValue cache = PagedAttention::EmptyCache();
  cache.fields.at("blocks").elements.push_back(std::move(page));
  return cache;
}

EncodedValueProto Affine(int32_t storage, int axis, uint64_t block = 0) {
  EncodedValueProto encoded;
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(DataType::FLOAT);
  for (int64_t d : {1, 1, 2, 2})
    logical->mutable_shape()->add_dim()->set_dim_value(d);
  auto *affine = encoded.mutable_affine();
  affine->set_storage_type(static_cast<TensorProto::DataType>(storage));
  affine->set_axis(axis);
  auto *scale = affine->mutable_scale();
  scale->set_data_type(DataType::FLOAT);
  if (block) {
    affine->set_block_size(block);
    for (int d : {1, 1, 2, 2})
      scale->add_dims(d);
    for (float f : {0.25f, 0.5f, 0.75f, 1.0f})
      scale->add_float_data(f);
  } else {
    scale->add_dims(2);
    scale->add_float_data(0.25f);
    scale->add_float_data(0.5f);
  }
  if (storage == DataType::INT4 || storage == DataType::UINT4) {
    const uint8_t bytes[] = {0x21, 0x43};
    encoded.set_raw_data(bytes, 2);
  } else {
    const uint8_t bytes[] = {1, 2, 3, 4};
    encoded.set_raw_data(bytes, 4);
  }
  return encoded;
}

} // namespace

TEST(PagedAttention, DenseMatchesAttentionAcrossAppendsAndWindows) {
  KernelContext context(DefaultOpset(23));
  PagedAttention paged(context);
  Attention dense(context);
  for (bool causal : {false, true})
    for (int64_t window : {-1, 0, 2}) {
      PagedAttention::Options options;
      options.block_size = 2;
      options.is_causal = causal;
      options.left_window_size = window;
      Attention::Attributes attributes;
      attributes.is_causal = causal;
      attributes.left_window_size = window;
      RuntimeValue cache = PagedAttention::EmptyCache();
      Tensor past_key = Input(0, 3), past_value = Input(0, 2);
      uint64_t cumulative_copied_bytes = 0;
      for (int step = 0; step < 3; ++step) {
        const Tensor q = Input(3, 3, 0.25f * step), k = Input(3, 3), v = Input(3, 2, step);
        auto actual = paged(q, k, v, cache, options);
        auto expected = dense(q, k, v, attributes, nullptr, &past_key, &past_value);
        Near(actual.Y, expected.Y);
        EXPECT_EQ(actual.statistics.copied_bytes, 3u * (3 + 2) * sizeof(float));
        cumulative_copied_bytes += actual.statistics.copied_bytes;
        EXPECT_EQ(cumulative_copied_bytes,
                  static_cast<uint64_t>(step + 1) * 3 * (3 + 2) * sizeof(float));
        EXPECT_EQ(actual.statistics.dequantized_bytes, 0u);
        EXPECT_EQ(actual.statistics.peak_workspace_bytes, 2u * sizeof(double));
        cache = std::move(actual.present);
        past_key = std::move(expected.present_key);
        past_value = std::move(expected.present_value);
      }
    }
}

TEST(PagedAttention, MixedFormatsKeepPartialPageOwnersAndBytes) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  PagedAttention::Options options;
  options.block_size = 4;
  options.key_format = {DataType::INT4, 0.25f, 0};
  Tensor q = Input(1, 3), k = Input(1, 3), v = Input(1, 2);
  auto first = kernel(q, k, v, PagedAttention::EmptyCache(), options);
  const auto &old_key = Pages(first.present)[0].fields.at("key");
  const auto &old_value = Pages(first.present)[0].fields.at("value").tensor;
  auto alias = first.present.BorrowView().Retain();
  const auto serialized = old_key.Encoded().SerializeAsString();
  const auto pointer = old_value.bytes();
  options.key_format = {};
  options.value_format = {DataType::UINT8, 0.25f, 10};
  auto second = kernel(q, k, v, first.present, options);
  ASSERT_EQ(Pages(second.present).size(), 2u);
  EXPECT_EQ(Pages(second.present)[0].fields.at("key").encoded, old_key.encoded);
  EXPECT_EQ(Pages(second.present)[0].fields.at("value").tensor.bytes(), pointer);
  EXPECT_EQ(old_key.Encoded().SerializeAsString(), serialized);
  EXPECT_EQ(Pages(alias)[0].fields.at("value").tensor.bytes(), pointer);
  EXPECT_EQ(second.statistics.copied_bytes, 3u * sizeof(float) + 2);
  EXPECT_EQ(second.statistics.dequantized_bytes, (3u + 2) * sizeof(float));
  Near(first.Y, second.Y);
}

TEST(PagedAttention, DecodesOnlyIntersectingValidTokens) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  PagedAttention::Options options;
  options.block_size = 2;
  options.key_format = {DataType::INT8, 0.25f, 0};
  options.value_format = {DataType::UINT4, 0.25f, 4};
  auto first = kernel(Input(6, 2), Input(6, 2), Input(6, 3), PagedAttention::EmptyCache(), options);
  options.left_window_size = 1;
  auto second = kernel(Input(2, 2), Input(2, 2), Input(2, 3), first.present, options);
  EXPECT_EQ(second.statistics.dequantized_bytes, 2u * 2 * (2 + 3) * sizeof(float));
  EXPECT_EQ(second.statistics.copied_bytes, 4u + 3);
  EXPECT_EQ(second.statistics.peak_workspace_bytes, 3u * sizeof(double));
}

TEST(PagedAttention, AcceptsSealedPagesWithUnusedCapacity) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  PagedAttention::Options options;
  options.block_size = 4;
  auto key = Input(4, 2), value = Input(4, 2);
  key.AsFloat()[7] = std::numeric_limits<float>::quiet_NaN();
  value.AsFloat()[7] = std::numeric_limits<float>::quiet_NaN();
  auto cache = CachePage(0, 1, RuntimeValue(std::move(key)).Retain(),
                         RuntimeValue(std::move(value)).Retain());
  const auto pointer = Pages(cache)[0].fields.at("key").tensor.bytes();
  auto result = kernel(Input(1, 2), Input(1, 2), Input(1, 2), cache, options);
  EXPECT_EQ(Pages(result.present).size(), 2u);
  EXPECT_EQ(Pages(result.present)[0].fields.at("key").tensor.bytes(), pointer);
  EXPECT_EQ(Pages(result.present)[0].fields.at("key").tensor.AsFloat()[0], -0.75f);
}

TEST(PagedAttention, AffineAxesAndBlockedParametersMatchDense) {
  KernelContext context(DefaultOpset(23));
  PagedAttention kernel(context);
  Attention dense(context);
  for (int32_t storage : {DataType::INT8, DataType::UINT8, DataType::INT4, DataType::UINT4})
    for (int axis : {2, -1})
      for (uint64_t block : {0u, 1u}) {
        const auto encoded = Affine(storage, axis, block);
        std::vector<float> data = block       ? std::vector<float>{0.25f, 1, 2.25f, 4}
                                  : axis == 2 ? std::vector<float>{0.25f, 0.5f, 1.5f, 2}
                                              : std::vector<float>{0.25f, 1, 0.75f, 2};
        Tensor old = Tensor::FromFloat("", {1, 1, 2, 2}, data);
        auto cache =
            CachePage(0, 2, RuntimeValue(encoded).Retain(), RuntimeValue(encoded).Retain());
        PagedAttention::Options options;
        const Tensor q = Input(1, 2), k = Input(1, 2), v = Input(1, 2);
        auto actual = kernel(q, k, v, cache, options);
        Attention::Attributes attributes;
        attributes.is_causal = true;
        auto expected = dense(q, k, v, attributes, nullptr, &old, &old);
        Near(actual.Y, expected.Y);
        EXPECT_EQ(actual.statistics.dequantized_bytes, 8u * sizeof(float));
      }
}

TEST(PagedAttention, QuantizationUsesTiesToEvenAndSaturates) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  for (int32_t storage : {DataType::INT8, DataType::UINT8, DataType::INT4, DataType::UINT4}) {
    PagedAttention::Options options;
    options.key_format = {storage, 1, 0};
    const Tensor k = Tensor::FromFloat("", {1, 1, 1, 5}, {0.5f, 1.5f, 2.5f, 3.5f, 1e30f});
    auto result = kernel(k, k, k, PagedAttention::EmptyCache(), options);
    const auto &raw = Pages(result.present)[0].fields.at("key").Encoded().raw_data();
    if (storage == DataType::INT4 || storage == DataType::UINT4) {
      ASSERT_EQ(raw.size(), 3u);
      EXPECT_EQ(raw.data()[0], 0x20);
      EXPECT_EQ(raw.data()[1], 0x42);
      EXPECT_EQ(raw.data()[2], storage == DataType::INT4 ? 7 : 15);
    } else {
      ASSERT_EQ(raw.size(), 5u);
      EXPECT_EQ(raw.data()[0], 0);
      EXPECT_EQ(raw.data()[1], 2);
      EXPECT_EQ(raw.data()[2], 2);
      EXPECT_EQ(raw.data()[3], 4);
      EXPECT_EQ(raw.data()[4], storage == DataType::INT8 ? 127 : 255);
    }
  }
}

TEST(PagedAttention, RejectsBadOptionsDescriptorsAndOverflow) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  PagedAttention::Options options;
  Tensor input = Input(1, 2);
  const auto empty = PagedAttention::EmptyCache();
  options.max_tokens = 0;
  EXPECT_THROW(kernel(input, input, input, empty, options), std::invalid_argument);
  options = {};
  options.key_format.scale = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(kernel(input, input, input, empty, options), std::invalid_argument);
  options = {};
  options.value_format = {DataType::UINT4, 1, 16};
  EXPECT_THROW(kernel(input, input, input, empty, options), std::invalid_argument);
  options = {};
  auto result = kernel(input, input, input, empty, options);
  options.max_tokens = 1;
  EXPECT_THROW(kernel(input, input, input, result.present, options), std::invalid_argument);
  options = {};
  auto malformed = result.present.DeepCopy();
  malformed.fields.at("blocks").elements[0].fields.at("start") =
      RuntimeValue(Tensor::FromInt64("", {}, {std::numeric_limits<int64_t>::max()})).Retain();
  EXPECT_THROW(kernel(input, input, input, malformed, options), std::invalid_argument);
  auto overflow = Input(0, 2);
  overflow.shape = {1, 1, std::numeric_limits<int64_t>::max(), 2};
  EXPECT_THROW(kernel(overflow, overflow, overflow, empty, options), std::invalid_argument);
  for (int variant = 0; variant < 4; ++variant) {
    Tensor unsupported = Input(1, 2);
    if (variant < 2)
      unsupported.shape[variant] = 2;
    else if (variant == 2)
      unsupported.data_type = DataType::INT32;
    else
      unsupported.shape[3] = 0;
    EXPECT_THROW(kernel(unsupported, input, input, empty, options), std::invalid_argument);
  }
  input.AsFloat()[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(kernel(input, input, input, empty, options), std::invalid_argument);
}

TEST(PagedAttention, RejectsUnsupportedEncodingAndMalformedPayload) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  PagedAttention::Options options;
  const Tensor input = Input(1, 2);
  for (int variant = 0; variant < 5; ++variant) {
    auto encoded = Affine(DataType::INT8, 2);
    if (variant == 0) {
      encoded.clear_affine();
      encoded.mutable_struct_type()->mutable_structure();
    } else if (variant == 1)
      encoded.set_raw_data("x", 1);
    else if (variant == 2)
      encoded.mutable_affine()->set_axis(4);
    else if (variant == 3)
      encoded.mutable_affine()->mutable_scale()->set_data_type(DataType::DOUBLE);
    else
      encoded.mutable_affine()->mutable_scale()->ref_float_data()[0] = -1;
    auto cache =
        CachePage(0, 2, RuntimeValue(std::move(encoded)), RuntimeValue(Input(2, 2)).Retain());
    EXPECT_THROW(kernel(input, input, input, cache, options), std::invalid_argument);
  }
}

TEST(PagedAttention, PackedTypedZeroPointsAndRawScales) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  auto encoded = Affine(DataType::INT4, -1);
  auto *affine = encoded.mutable_affine();
  const uint8_t scales[] = {0, 0, 0x80, 0x3e, 0, 0, 0, 0x3f};
  affine->mutable_scale()->clear_float_data();
  affine->mutable_scale()->set_raw_data(scales, sizeof(scales));
  auto *zero = affine->mutable_zero_point();
  zero->set_data_type(DataType::INT4);
  zero->add_dims(2);
  zero->add_int32_data(0x2f);
  auto cache = CachePage(0, 2, RuntimeValue(encoded).Retain(), RuntimeValue(encoded).Retain());
  const Tensor input = Tensor::FromFloat("", {1, 1, 1, 2}, {0, 0});
  const auto result = kernel(input, input, input, cache, {});
  EXPECT_NEAR(result.Y.AsFloat()[0], (0.5f + 1.0f) / 3, 1e-6f);
  EXPECT_NEAR(result.Y.AsFloat()[1], (0.0f + 1.0f) / 3, 1e-6f);
}

TEST(PagedAttention, RunRejectsUnsupportedAttributesWithoutPublishing) {
  RuntimeContext context(KernelContext(DefaultOpset(23)));
  NodeProto node;
  node.set_domain("onnx_light");
  node.set_op_type("PagedAttention");
  for (const char *name : {"Q", "K", "V", "past"})
    node.add_input(name);
  node.add_output("Y");
  node.add_output("present");
  context.values().emplace("past", PagedAttention::EmptyCache());
  for (const char *name : {"Q", "K", "V"})
    context.Put(name, Input(1, 2));
  auto *attribute = node.add_attribute();
  attribute->set_name("right_window_size");
  attribute->set_type(AttributeProto::INT);
  attribute->set_i(0);
  PagedAttention kernel(context.kernel_ctx());
  kernel.set_node(node);
  EXPECT_THROW(kernel.Run(context), std::invalid_argument);
  EXPECT_EQ(context.tensors().count("Y"), 0u);
  EXPECT_EQ(context.values().count("present"), 0u);
  node.clear_attribute();
  kernel.Run(context);
  EXPECT_EQ(context.tensors().count("Y"), 1u);
  EXPECT_EQ(Pages(context.values().at("present")).size(), 1u);
}

TEST(PagedAttention, EmptyAppendPreservesRetainedPages) {
  PagedAttention kernel(KernelContext(DefaultOpset(23)));
  const Tensor input = Input(1, 2), empty = Input(0, 2);
  const auto first = kernel(input, input, input, PagedAttention::EmptyCache(), {});
  const auto second = kernel(empty, empty, empty, first.present, {});
  EXPECT_EQ(second.Y.size_bytes(), 0u);
  EXPECT_EQ(second.statistics.copied_bytes, 0u);
  EXPECT_EQ(second.statistics.dequantized_bytes, 0u);
  EXPECT_EQ(second.statistics.peak_workspace_bytes, 0u);
  EXPECT_EQ(Pages(second.present)[0].fields.at("key").tensor.bytes(),
            Pages(first.present)[0].fields.at("key").tensor.bytes());
}

TEST(PagedAttention, QuantizedReconstructionAndAttentionTolerance) {
  KernelContext context(DefaultOpset(23));
  PagedAttention kernel(context);
  Attention reference(context);
  Attention::Attributes attributes;
  attributes.is_causal = true;
  const Tensor q = Input(3, 3, 0.031f), key = Input(3, 3, 0.017f), value = Input(3, 2, 0.063f);
  for (int32_t storage : {DataType::INT8, DataType::UINT8, DataType::INT4, DataType::UINT4}) {
    SCOPED_TRACE(storage);
    const bool four_bit = storage == DataType::INT4 || storage == DataType::UINT4;
    const bool is_signed = storage == DataType::INT4 || storage == DataType::INT8;
    const float scale = four_bit ? 0.25f : 0.01f;
    const int zero = is_signed ? 0 : (four_bit ? 8 : 128);
    PagedAttention::Options options;
    options.block_size = 2;
    options.key_format = options.value_format = {storage, scale, zero};
    const auto actual = kernel(q, key, value, PagedAttention::EmptyCache(), options);
    Tensor reconstructed_key = Input(3, 3), reconstructed_value = Input(3, 2);
    for (const auto &page : Pages(actual.present)) {
      const int64_t start = page.fields.at("start").tensor.AsInt64()[0];
      const int64_t length = page.fields.at("length").tensor.AsInt64()[0];
      for (const char *field : {"key", "value"}) {
        const bool is_key = std::string(field) == "key";
        Tensor &reconstructed = is_key ? reconstructed_key : reconstructed_value;
        const Tensor &source = is_key ? key : value;
        const auto &raw = page.fields.at(field).Encoded().raw_data();
        for (int64_t i = 0; i < length * source.shape[3]; ++i) {
          int code = four_bit ? (raw.data()[i / 2] >> (4 * (i % 2))) & 15 : raw.data()[i];
          if (is_signed && code >= (four_bit ? 8 : 128))
            code -= four_bit ? 16 : 256;
          const int64_t index = start * source.shape[3] + i;
          const float decoded = (code - zero) * scale;
          EXPECT_LE(std::abs(decoded - source.AsFloat()[index]), scale / 2 + 1e-7f);
          reconstructed.AsFloat()[index] = decoded;
        }
      }
    }
    Near(actual.Y, reference(q, reconstructed_key, reconstructed_value, attributes).Y, 1e-5f);
    Near(actual.Y, reference(q, key, value, attributes).Y, four_bit ? 0.15f : 0.01f);
  }
}
