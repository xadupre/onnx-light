// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning.h"

#include <gtest/gtest.h>

#include <unordered_set>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

KernelTuningKey MakeKey() { return {"onnx_light_cpu", "Gemm", "blocked_avx2", 1, Device::kCPU, 3}; }

KernelTuningParameters MakeDefaults() {
  return {MakeKey(),
          {{"algorithm.tile_m", int64_t{64}},
           {"parallel.minimum_tasks", int64_t{2}},
           {"algorithm.minimum_margin", 0.05},
           {"algorithm.pack_b", true},
           {"algorithm.mode", std::string("blocked")}}};
}

TEST(KernelTuningKey, HashesAllIdentityFields) {
  KernelTuningKey first = MakeKey();
  KernelTuningKey second = first;
  std::unordered_set<KernelTuningKey, KernelTuningKeyHash> keys{first};

  EXPECT_EQ(first, second);
  EXPECT_TRUE(keys.contains(second));

  second.element_type = 11;
  EXPECT_NE(first, second);
  EXPECT_FALSE(keys.contains(second));
}

TEST(KernelTuningParameters, ProvidesStrictTypedAccess) {
  KernelTuningParameters parameters = MakeDefaults();

  EXPECT_TRUE(parameters.Contains("algorithm.tile_m"));
  EXPECT_EQ(parameters.Get<int64_t>("algorithm.tile_m"), 64);
  EXPECT_DOUBLE_EQ(parameters.Get<double>("algorithm.minimum_margin"), 0.05);
  EXPECT_TRUE(parameters.Get<bool>("algorithm.pack_b"));
  EXPECT_EQ(parameters.Get<std::string>("algorithm.mode"), "blocked");
  EXPECT_THROW(parameters.Get<double>("algorithm.tile_m"), std::invalid_argument);
  EXPECT_THROW(parameters.Get<int64_t>("missing"), std::invalid_argument);
}

TEST(KernelTuningSchema, AcceptsCompleteParametersAndRunsValidationHook) {
  KernelTuningSchema schema(MakeDefaults(), [](const KernelTuningParameters &parameters) {
    if (parameters.Get<int64_t>("algorithm.tile_m") <= 0 ||
        parameters.Get<int64_t>("parallel.minimum_tasks") <= 0) {
      throw std::invalid_argument("Tile and task sizes must be positive.");
    }
  });
  KernelTuningParameters parameters = MakeDefaults();
  parameters.values["algorithm.tile_m"] = int64_t{128};

  EXPECT_NO_THROW(schema.Validate(parameters));
  EXPECT_EQ(schema.portable_defaults().Get<int64_t>("algorithm.tile_m"), 64);

  parameters.values["algorithm.tile_m"] = int64_t{0};
  EXPECT_THROW(schema.Validate(parameters), std::invalid_argument);
}

TEST(KernelTuningSchema, RejectsIncompleteUnknownAndMistypedParameters) {
  KernelTuningSchema schema(MakeDefaults());

  KernelTuningParameters missing = MakeDefaults();
  missing.values.erase("algorithm.tile_m");
  EXPECT_THROW(schema.Validate(missing), std::invalid_argument);

  KernelTuningParameters unknown = MakeDefaults();
  unknown.values["algorithm.tile_n"] = int64_t{256};
  EXPECT_THROW(schema.Validate(unknown), std::invalid_argument);

  KernelTuningParameters mistyped = MakeDefaults();
  mistyped.values["algorithm.tile_m"] = 64.0;
  EXPECT_THROW(schema.Validate(mistyped), std::invalid_argument);

  KernelTuningParameters another_key = MakeDefaults();
  another_key.key.element_type = 11;
  EXPECT_THROW(schema.Validate(another_key), std::invalid_argument);
}

TEST(KernelTuningSchema, RejectsInvalidKeysNamesAndPortableDefaults) {
  KernelTuningParameters parameters = MakeDefaults();
  parameters.key.tuning_abi = 0;
  EXPECT_THROW(KernelTuningSchema(std::move(parameters)), std::invalid_argument);

  parameters = MakeDefaults();
  parameters.values["parallel..grain"] = int64_t{1};
  EXPECT_THROW(KernelTuningSchema(std::move(parameters)), std::invalid_argument);

  parameters = MakeDefaults();
  parameters.values.clear();
  EXPECT_THROW(KernelTuningSchema(std::move(parameters)), std::invalid_argument);

  EXPECT_THROW(KernelTuningSchema(MakeDefaults(),
                                  [](const KernelTuningParameters &) {
                                    throw std::invalid_argument("Invalid portable defaults.");
                                  }),
               std::invalid_argument);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
