// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/runtime/runtime_session.h"
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

PagedCacheProto Cache() {
  PagedCacheProto cache;
  cache.set_name("cache");
  auto *block = cache.add_blocks();
  block->set_start(0);
  block->set_length(1);
  for (auto *tensor : {block->mutable_key(), block->mutable_value()}) {
    tensor->set_data_type(TensorProto::FLOAT);
    for (int64_t dim : {1, 1, 2, 2})
      tensor->add_dims(dim);
    for (float value : {1.f, 2.f, 3.f, 4.f})
      tensor->add_float_data(value);
  }
  return cache;
}

PagedCacheProto EncodedCache() {
  PagedCacheProto cache;
  cache.set_name("encoded_cache");
  auto *block = cache.add_blocks();
  block->set_start(0);
  block->set_length(1);
  for (auto *encoded : {block->mutable_encoded_key(), block->mutable_encoded_value()}) {
    auto *logical = encoded->mutable_logical_type()->mutable_tensor_type();
    logical->set_elem_type(TensorProto::FLOAT);
    for (int64_t dim : {1, 1, 2, 2})
      logical->mutable_shape()->add_dim()->set_dim_value(dim);
    auto *affine = encoded->mutable_affine();
    affine->set_storage_type(TensorProto::INT8);
    auto *scale = affine->mutable_scale();
    scale->set_data_type(TensorProto::FLOAT);
    const float scale_value = 0.25f;
    scale->set_raw_data(&scale_value, sizeof(scale_value));
    auto *zero = affine->mutable_zero_point();
    zero->set_data_type(TensorProto::INT8);
    const uint8_t zero_value = 0;
    zero->set_raw_data(&zero_value, sizeof(zero_value));
    const uint8_t payload[] = {1, 2, 3, 4};
    encoded->set_raw_data(payload, sizeof(payload));
  }
  return cache;
}

ModelProto CacheModel() {
  ModelProto model;
  model.set_ir_version(10);
  model.mutable_graph()->set_name("cache_initializer");
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(23);
  *model.mutable_graph()->add_extended_initializer()->mutable_paged_cache() = Cache();
  auto *output = model.mutable_graph()->add_output();
  output->set_name("cache");
  output->mutable_type()->mutable_struct_type();
  return model;
}

} // namespace

TEST(PagedCacheProto, ModelRoundTripPresenceEqualityAndInitializers) {
  auto model = CacheModel();
  EXPECT_NO_THROW(VerifyModel(model));
  ModelProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(model.SerializeAsString()));
  std::string difference;
  EXPECT_TRUE(model.Equals(parsed, &difference)) << difference;
  EXPECT_TRUE(parsed.graph().extended_initializer(0).paged_cache().blocks(0).has_start());
  auto *block = parsed.mutable_graph()
                    ->mutable_extended_initializer(0)
                    ->mutable_paged_cache()
                    ->mutable_blocks(0);
  block->set_length(2);
  difference.clear();
  EXPECT_FALSE(model.Equals(parsed, &difference));
  EXPECT_NE(difference.find("extended_initializer[0].paged_cache.blocks[0].length"),
            std::string::npos);
  PagedCacheProto empty;
  EXPECT_NO_THROW(StructTypeCatalogue().ValidatePagedCache(empty));
  auto changed = Cache();
  changed.set_doc_string("");
  EXPECT_FALSE(Cache().Equals(changed));
}

TEST(PagedCacheProto, OneofSwitchesAndRoundTrips) {
  auto cache = Cache();
  auto *block = cache.mutable_blocks(0);
  auto *encoded = block->mutable_encoded_key();
  EXPECT_FALSE(block->has_key());
  encoded->mutable_logical_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  encoded->mutable_affine()->set_storage_type(TensorProto::INT8);
  PagedCacheBlockProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(block->SerializeAsString()));
  EXPECT_TRUE(parsed.has_encoded_key());
  EXPECT_FALSE(parsed.has_key());
  EXPECT_TRUE(parsed.Equals(*block));
}

TEST(PagedCacheProto, CopyFromReplacesExistingBlocksAndPresence) {
  auto cache = Cache();
  cache.CopyFrom(cache);
  EXPECT_TRUE(cache.Equals(Cache()));
  PagedCacheBlockProto block = cache.blocks(0);
  block.CopyFrom(EncodedCache().blocks(0));
  EXPECT_TRUE(block.Equals(EncodedCache().blocks(0)));
  block.CopyFrom(cache.blocks(0));
  EXPECT_TRUE(block.Equals(cache.blocks(0)));
  block.CopyFrom(block);
  EXPECT_TRUE(block.Equals(cache.blocks(0)));
  block.CopyFrom(PagedCacheBlockProto{});
  EXPECT_FALSE(block.has_start());
  EXPECT_FALSE(block.has_length());
  EXPECT_FALSE(block.has_key_payload());
  EXPECT_FALSE(block.has_value_payload());
  cache.CopyFrom(PagedCacheProto{});
  EXPECT_TRUE(cache.blocks().empty());
  EXPECT_FALSE(cache.has_name());
}

TEST(PagedCacheProto, RejectsMalformedRangesShapesAndPayloads) {
  for (int failure = 0; failure < 10; ++failure) {
    SCOPED_TRACE(failure);
    auto cache = Cache();
    auto *block = cache.mutable_blocks(0);
    if (failure == 0)
      block->clear_start();
    else if (failure == 1)
      block->set_start(1);
    else if (failure == 2)
      block->set_length(0);
    else if (failure == 3)
      block->set_length(3);
    else if (failure == 4)
      block->clear_key_payload();
    else if (failure == 5)
      block->mutable_key()->set_data_type(TensorProto::DOUBLE);
    else if (failure == 6)
      (*block->mutable_key()->mutable_dims())[2] = INT64_MAX;
    else if (failure == 7)
      (*block->mutable_value()->mutable_dims())[2] = 1;
    else if (failure == 8)
      block->mutable_key()->clear_float_data();
    else
      block->mutable_key()->set_data_location(TensorProto::EXTERNAL);
    EXPECT_THROW(StructTypeCatalogue().ValidatePagedCache(cache), std::invalid_argument);
    EXPECT_THROW(RuntimeValue::FromPagedCache(cache), std::invalid_argument);
  }
}

TEST(PagedCacheProto, RuntimeRetainsStorageAfterProtoDestructionAndExports) {
  RuntimeValue value;
  {
    auto cache = Cache();
    value = RuntimeValue::FromPagedCache(std::move(cache));
  }
  const auto &key = value.fields.at("blocks").elements[0].fields.at("key").tensor;
  EXPECT_TRUE(value.HasPagedCacheStructure());
  EXPECT_TRUE(value.BorrowView().HasPagedCacheStructure());
  EXPECT_TRUE(value.DeepCopy().HasPagedCacheStructure());
  EXPECT_FALSE(RuntimeValue{}.HasPagedCacheStructure());
  EXPECT_FLOAT_EQ(key.AsFloat()[0], 1.f);
  EXPECT_GT(key.borrowed_owner().use_count(), 0);
  auto restored = RuntimeValue::FromPagedCache(value.ToPagedCache("roundtrip"));
  const auto &copy = restored.fields.at("blocks").elements[0].fields.at("key").tensor;
  EXPECT_EQ(copy.shape, key.shape);
  EXPECT_EQ(copy.size_bytes(), key.size_bytes());
  EXPECT_FLOAT_EQ(copy.AsFloat()[3], 4.f);
  EXPECT_EQ(copy.bytes(), key.bytes());
}

TEST(PagedCacheProto, EncodedExportBorrowsPayloadAndAffineParameters) {
  PagedCacheProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(EncodedCache().SerializeAsString()));
  PagedCacheProto exported;
  const uint8_t *payload = nullptr;
  const uint8_t *scale = nullptr;
  const uint8_t *zero = nullptr;
  {
    RuntimeValue value = RuntimeValue::FromPagedCache(std::move(parsed));
    const auto &encoded = value.fields.at("blocks").elements[0].fields.at("key").Encoded();
    payload = encoded.raw_data().data();
    scale = encoded.affine().scale().raw_data().data();
    zero = encoded.affine().zero_point().raw_data().data();
    exported = value.ToPagedCache();
    EXPECT_EQ(exported.blocks(0).encoded_key().raw_data().data(), payload);
    EXPECT_EQ(exported.blocks(0).encoded_key().affine().scale().raw_data().data(), scale);
    EXPECT_EQ(exported.blocks(0).encoded_key().affine().zero_point().raw_data().data(), zero);
  }
  EXPECT_EQ(exported.blocks(0).encoded_key().raw_data().data(), payload);
  EXPECT_EQ(exported.blocks(0).encoded_key().affine().scale().raw_data().data(), scale);
  EXPECT_EQ(exported.blocks(0).encoded_key().affine().zero_point().raw_data().data(), zero);
  EXPECT_EQ(exported.blocks(0).encoded_key().raw_data()[0], 1);
  EXPECT_EQ(exported.blocks(0).encoded_key().affine().scale().raw_data().size(), sizeof(float));
  EXPECT_EQ(exported.blocks(0).encoded_key().affine().zero_point().raw_data()[0], 0);
}

TEST(PagedCacheProto, ImportsTypedFloat16AndBfloat16PagesWithOwnedStorage) {
  for (int32_t dtype : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    SCOPED_TRACE(dtype);
    const uint16_t one = dtype == TensorProto::FLOAT16 ? 0x3c00 : 0x3f80;
    const uint16_t expected[] = {one, static_cast<uint16_t>(one | 0x8000), 0, 0x8000};
    RuntimeValue value;
    {
      auto model = CacheModel();
      auto *block = model.mutable_graph()
                        ->mutable_extended_initializer(0)
                        ->mutable_paged_cache()
                        ->mutable_blocks(0);
      for (auto *tensor : {block->mutable_key(), block->mutable_value()}) {
        tensor->clear_float_data();
        tensor->set_data_type(dtype);
        for (uint16_t bits : expected)
          tensor->add_int32_data(bits);
      }
      EXPECT_NO_THROW(VerifyModel(model));
      value = RuntimeValue::FromPagedCache(model.graph().extended_initializer(0).paged_cache());
      for (int count : {0, 3, 5}) {
        auto malformed = model.graph().extended_initializer(0).paged_cache();
        auto *key = malformed.mutable_blocks(0)->mutable_key();
        key->clear_int32_data();
        for (int i = 0; i < count; ++i)
          key->add_int32_data(one);
        EXPECT_THROW(StructTypeCatalogue().ValidatePagedCache(malformed), std::invalid_argument);
        EXPECT_THROW(RuntimeValue::FromPagedCache(malformed), std::invalid_argument);
      }
    }
    for (const char *name : {"key", "value"}) {
      const auto &tensor = value.fields.at("blocks").elements[0].fields.at(name).tensor;
      EXPECT_EQ(tensor.data_type, dtype);
      ASSERT_EQ(tensor.size_bytes(), sizeof(expected));
      EXPECT_EQ(std::memcmp(tensor.bytes(), expected, sizeof(expected)), 0);
      EXPECT_GT(tensor.borrowed_owner().use_count(), 0);
    }
    const auto exported = value.ToPagedCache();
    EXPECT_EQ(exported.blocks(0).key().data_type(), dtype);
    EXPECT_EQ(exported.blocks(0).key().raw_data().size(), sizeof(expected));
    EXPECT_NO_THROW(RuntimeValue::FromPagedCache(exported));
  }
}

TEST(PagedCacheProto, RecognizesCacheFromStructureWithoutImportMetadata) {
  RuntimeValue empty;
  empty.fields.emplace("blocks", RuntimeValue(std::vector<RuntimeValue>{}));
  EXPECT_TRUE(empty.HasPagedCacheStructure());
  EXPECT_EQ(empty.ToPagedCache().blocks_size(), 0);

  for (const auto &proto : {Cache(), EncodedCache()}) {
    RuntimeValue cache(RuntimeValue::FromPagedCache(proto).fields);
    EXPECT_TRUE(cache.HasPagedCacheStructure());
    EXPECT_EQ(cache.ToPagedCache().blocks_size(), 1);
    auto page = cache.fields.at("blocks").elements[0].BorrowView();
    for (const char *name : {"start", "length", "key", "value"}) {
      auto missing = page.BorrowView();
      missing.fields.erase(name);
      cache.fields.at("blocks").elements.Set(0, std::move(missing));
      EXPECT_FALSE(cache.HasPagedCacheStructure());
    }
    cache.fields.at("blocks").elements.Set(0, page);
    cache.fields.at("blocks").elements.push_back(RuntimeValue{});
    EXPECT_FALSE(cache.HasPagedCacheStructure());
    cache.fields.at("blocks").elements.Set(1, page);
    EXPECT_TRUE(cache.HasPagedCacheStructure());
    // Structural recognition does not replace range validation.
    EXPECT_THROW(cache.ToPagedCache(), std::invalid_argument);
  }

  RuntimeValue ordinary;
  ordinary.fields.emplace("blocks", RuntimeValue(std::vector<RuntimeValue>{
                                        RuntimeValue(Tensor::FromFloat("", {1}, {1.f}))}));
  EXPECT_FALSE(ordinary.HasPagedCacheStructure());
  empty.fields.emplace("metadata", RuntimeValue{});
  EXPECT_FALSE(empty.HasPagedCacheStructure());
}

TEST(PagedCacheProto, ExportRetainsOwnedEncodedPagesAndCopiesOwnerlessInitializerViews) {
  for (bool owned : {false, true}) {
    SCOPED_TRACE(owned);
    PagedCacheProto exported;
    std::weak_ptr<EncodedValueProto> weak;
    std::string expected;
    {
      auto proto = std::make_shared<EncodedValueProto>(EncodedCache().blocks(0).encoded_key());
      weak = proto;
      expected = proto->SerializeAsString();
      RuntimeValue page;
      page.fields.emplace("start", RuntimeValue(Tensor::FromInt64("", {}, {0})));
      page.fields.emplace("length", RuntimeValue(Tensor::FromInt64("", {}, {1})));
      for (const char *name : {"key", "value"})
        page.fields.emplace(name, owned ? RuntimeValue::FromEncodedView(*proto, proto)
                                        : RuntimeValue::FromEncodedView(*proto));
      RuntimeValue cache;
      cache.fields.emplace("blocks", RuntimeValue(std::vector<RuntimeValue>{std::move(page)}));
      exported = cache.ToPagedCache();
      for (const auto *value :
           {&exported.blocks(0).encoded_key(), &exported.blocks(0).encoded_value()}) {
        EXPECT_GT(value->raw_data().owner().use_count(), 0);
        EXPECT_GT(value->affine().scale().raw_data().owner().use_count(), 0);
        EXPECT_GT(value->affine().zero_point().raw_data().owner().use_count(), 0);
        if (owned)
          EXPECT_EQ(value->raw_data().data(), proto->raw_data().data());
        else
          EXPECT_NE(value->raw_data().data(), proto->raw_data().data());
      }
    }
    EXPECT_EQ(weak.expired(), !owned);
    EXPECT_EQ(exported.blocks(0).encoded_key().SerializeAsString(), expected);
    EXPECT_EQ(exported.blocks(0).encoded_value().SerializeAsString(), expected);
    EXPECT_NO_THROW(RuntimeValue::FromPagedCache(exported));
    exported = PagedCacheProto{};
    EXPECT_TRUE(weak.expired());
  }
}

TEST(PagedCacheProto, SessionSeedsAndReseedsWithoutOverridingCallerValues) {
  auto model = CacheModel();
  RuntimeSession session(model);
  RuntimeContext context;
  session.Run(context);
  ASSERT_TRUE(context.values().contains("cache"));
  EXPECT_EQ(context.values().at("cache").ToPagedCache().blocks_size(), 1);
  context.Remove("cache");
  session.Run(context);
  EXPECT_EQ(context.values().at("cache").ToPagedCache().blocks_size(), 1);
  context.PutValue("cache", RuntimeValue::FromPagedCache(PagedCacheProto{}));
  session.Run(context);
  EXPECT_EQ(context.values().at("cache").ToPagedCache().blocks_size(), 0);
}

TEST(PagedCacheProto, GraphBuilderPreservesAndPrunesInitializer) {
  auto model = CacheModel();
  core::builder::GraphBuilder builder(model);
  ASSERT_EQ(builder.ExtendedInitializers().size(), 1u);
  const auto exported = builder.ToModel();
  ASSERT_EQ(exported.graph().extended_initializer_size(), 1);
  EXPECT_TRUE(exported.graph().extended_initializer(0).paged_cache().Equals(Cache()));
  EXPECT_THROW(builder.MakePagedCacheInitializer(Cache()), core::builder::BuilderError);
  EXPECT_THROW(builder.ToStandardModel(), core::builder::BuilderError);
  auto unused = Cache();
  unused.set_name("unused");
  builder.MakePagedCacheInitializer(unused);
  builder.RemoveUnusedNodes();
  EXPECT_EQ(builder.ExtendedInitializers().size(), 1u);
}

TEST(PagedCacheProto, GraphExportRetainsDenseAndEncodedPayloadOwners) {
  for (bool encoded : {false, true}) {
    SCOPED_TRACE(encoded);
    bool released[2] = {false, false};
    const uint8_t *payloads[2] = {nullptr, nullptr};
    ModelProto exported;
    {
      auto model = CacheModel();
      auto *cache = model.mutable_graph()->mutable_extended_initializer(0)->mutable_paged_cache();
      if (encoded) {
        *cache = EncodedCache();
        model.mutable_graph()->mutable_output(0)->set_name(cache->name());
      }
      auto *block = cache->mutable_blocks(0);
      for (size_t i = 0; i < 2; ++i) {
        auto bytes = std::make_shared<std::vector<uint8_t>>(encoded ? 4 : 16, 1);
        payloads[i] = bytes->data();
        auto deleter = [bytes, &released, i]() { released[i] = true; };
        if (encoded) {
          auto *value = i == 0 ? block->mutable_encoded_key() : block->mutable_encoded_value();
          value->set_raw_data_with_deleter(bytes->data(), bytes->size(), std::move(deleter));
        } else {
          auto *value = i == 0 ? block->mutable_key() : block->mutable_value();
          value->clear_float_data();
          value->set_raw_data_with_deleter(bytes->data(), bytes->size(), std::move(deleter));
        }
      }
      core::builder::GraphBuilder builder(model);
      model.Clear();
      exported = builder.ToModel();
    }
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_FALSE(released[i]);
      const auto &block = exported.graph().extended_initializer(0).paged_cache().blocks(0);
      const auto &payload =
          encoded ? (i == 0 ? block.encoded_key().raw_data() : block.encoded_value().raw_data())
                  : (i == 0 ? block.key().raw_data() : block.value().raw_data());
      EXPECT_EQ(payload.data(), payloads[i]);
      ASSERT_EQ(payload.size(), encoded ? 4u : 16u);
      EXPECT_EQ(payload[0], 1);
    }
    exported.Clear();
    EXPECT_TRUE(released[0]);
    EXPECT_TRUE(released[1]);
  }
}

TEST(PagedCacheProto, RejectsInitializerNameCollisionsAndMismatchedDeclaredDefaults) {
  auto model = CacheModel();
  *model.mutable_graph()->add_extended_initializer()->mutable_paged_cache() = Cache();
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  model = CacheModel();
  *model.mutable_graph()->add_initializer() = Cache().blocks(0).key();
  model.mutable_graph()->mutable_initializer(0)->set_name("cache");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  model = CacheModel();
  auto *input = model.mutable_graph()->add_input();
  input->set_name("cache");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  input->mutable_type()->mutable_tensor_type()->mutable_shape();
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(PagedCacheProto, RejectsArrayAndBitPackingDeclarationsIncludingReferences) {
  for (bool array : {false, true}) {
    for (bool reference : {false, true}) {
      SCOPED_TRACE(array);
      SCOPED_TRACE(reference);
      auto model = CacheModel();
      StructTypeProto storage;
      if (array) {
        auto *layout = storage.mutable_array();
        layout->set_dimension(2);
        auto *tensor = layout->mutable_element_type()->mutable_tensor_type();
        tensor->set_elem_type(TensorProto::FLOAT);
        tensor->mutable_shape();
      } else {
        auto *layout = storage.mutable_bit_packing();
        layout->set_dimension(1);
        auto *component = layout->add_component();
        component->set_name("code");
        component->set_bit_width(8);
      }
      TypeProto type;
      if (reference) {
        storage.set_type_id(1);
        *model.add_struct_types() = storage;
        type.mutable_struct_type()->set_type_ref(1);
      } else {
        *type.mutable_struct_type() = storage;
      }
      *model.mutable_graph()->mutable_output(0)->mutable_type() = type;
      auto *input = model.mutable_graph()->add_input();
      input->set_name("cache");
      *input->mutable_type() = type;
      StructTypeCatalogue catalogue;
      catalogue.Build(model);
      EXPECT_NO_THROW(ValidatePersistentType(catalogue, type));
      EXPECT_THROW(catalogue.ValidatePagedCache(Cache(), true, &type), std::invalid_argument);
      EXPECT_THROW(VerifyModel(model), std::invalid_argument);
      model.mutable_graph()->clear_extended_initializer();
      EXPECT_NO_THROW(VerifyModel(model));
    }
  }
}

TEST(PagedCacheProto, RequiresLogicalFloatDeclarationsRegardlessOfPhysicalStorage) {
  for (int32_t physical : {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (int32_t logical : {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      for (const char *payload : {"key", "value"}) {
        SCOPED_TRACE(physical);
        SCOPED_TRACE(logical);
        SCOPED_TRACE(payload);
        auto model = CacheModel();
        auto *block = model.mutable_graph()
                          ->mutable_extended_initializer(0)
                          ->mutable_paged_cache()
                          ->mutable_blocks(0);
        if (physical != TensorProto::FLOAT) {
          for (auto *tensor : {block->mutable_key(), block->mutable_value()}) {
            tensor->set_data_type(physical);
            tensor->clear_float_data();
            const uint16_t zeros[4] = {};
            tensor->set_raw_data(zeros, sizeof(zeros));
          }
        }
        auto *blocks = model.mutable_graph()
                           ->mutable_output(0)
                           ->mutable_type()
                           ->mutable_struct_type()
                           ->mutable_structure()
                           ->add_field();
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
          tensor->set_elem_type(scalar                         ? TensorProto::INT64
                                : std::string(name) == payload ? logical
                                                               : TensorProto::FLOAT);
          tensor->mutable_shape();
          if (!scalar)
            for (int64_t dimension : {1, 1, 2, 2})
              tensor->mutable_shape()->add_dim()->set_dim_value(dimension);
        }
        auto *input = model.mutable_graph()->add_input();
        input->set_name("cache");
        *input->mutable_type() = model.graph().output(0).type();
        if (logical == TensorProto::FLOAT)
          EXPECT_NO_THROW(VerifyModel(model));
        else
          EXPECT_THROW(VerifyModel(model), std::invalid_argument);
      }
    }
  }
}
