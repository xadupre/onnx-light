// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/runtime/runtime_session.h"
#include <gtest/gtest.h>

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

ModelProto CacheModel() {
  ModelProto model;
  model.set_ir_version(10);
  model.mutable_graph()->set_name("cache_initializer");
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(23);
  *model.mutable_graph()->add_paged_cache_initializer() = Cache();
  auto *output = model.mutable_graph()->add_output();
  output->set_name("cache");
  *output->mutable_type() = PagedCacheProto::CacheType();
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
  EXPECT_TRUE(parsed.graph().paged_cache_initializer(0).blocks(0).has_start());
  auto *block = parsed.mutable_graph()->mutable_paged_cache_initializer(0)->mutable_blocks(0);
  block->set_length(2);
  difference.clear();
  EXPECT_FALSE(model.Equals(parsed, &difference));
  EXPECT_NE(difference.find("paged_cache_initializer[0].blocks[0].length"), std::string::npos);
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
  *encoded->mutable_logical_type() = PagedCacheProto::CacheType();
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
  block.CopyFrom(PagedCacheBlockProto{});
  EXPECT_FALSE(block.has_start());
  EXPECT_FALSE(block.has_key_payload());
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
  EXPECT_FLOAT_EQ(key.AsFloat()[0], 1.f);
  EXPECT_GT(key.borrowed_owner().use_count(), 0);
  auto restored = RuntimeValue::FromPagedCache(value.ToPagedCache("roundtrip"));
  const auto &copy = restored.fields.at("blocks").elements[0].fields.at("key").tensor;
  EXPECT_EQ(copy.shape, key.shape);
  EXPECT_EQ(copy.size_bytes(), key.size_bytes());
  EXPECT_FLOAT_EQ(copy.AsFloat()[3], 4.f);
  EXPECT_EQ(copy.bytes(), key.bytes());
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
  ASSERT_EQ(builder.PagedCacheInitializers().size(), 1u);
  const auto exported = builder.ToModel();
  ASSERT_EQ(exported.graph().paged_cache_initializer_size(), 1);
  EXPECT_TRUE(exported.graph().paged_cache_initializer(0).Equals(Cache()));
  EXPECT_THROW(builder.MakePagedCacheInitializer(Cache()), core::builder::BuilderError);
  EXPECT_THROW(builder.ToStandardModel(), core::builder::BuilderError);
  auto unused = Cache();
  unused.set_name("unused");
  builder.MakePagedCacheInitializer(unused);
  builder.RemoveUnusedNodes();
  EXPECT_EQ(builder.PagedCacheInitializers().size(), 1u);
}

TEST(PagedCacheProto, RejectsInitializerNameCollisionsAndMismatchedDeclaredDefaults) {
  auto model = CacheModel();
  *model.mutable_graph()->add_paged_cache_initializer() = Cache();
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
