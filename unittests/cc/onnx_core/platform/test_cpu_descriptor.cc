// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/platform/cpu_descriptor.h"

#include <gtest/gtest.h>

namespace ONNX_LIGHT_NAMESPACE::core::platform {
namespace {

TEST(CpuFeatureSet, StoresAndMatchesFeatures) {
  CpuFeatureSet available;
  available.Add(CpuFeature::kSse2);
  available.Add(CpuFeature::kAvx2);

  CpuFeatureSet required;
  required.Add(CpuFeature::kAvx2);
  CpuFeatureSet excluded;
  excluded.Add(CpuFeature::kSve);

  EXPECT_TRUE(available.Has(CpuFeature::kSse2));
  EXPECT_TRUE(available.ContainsAll(required));
  EXPECT_FALSE(available.Intersects(excluded));
  EXPECT_EQ(CpuFeatureName(CpuFeature::kAvx512F), "avx512f");
  EXPECT_EQ(CpuFeatureFromName("dotprod"), CpuFeature::kDotProduct);
  EXPECT_EQ(CpuFeatureFromName("unknown"), std::nullopt);
}

TEST(CpuSelector, MatchesStableProcessorProperties) {
  CpuDescriptor processor;
  processor.architecture = "x86_64";
  processor.vendor = "intel";
  processor.family = 6;
  processor.model = 0x97;
  processor.stepping = 2;
  processor.microarchitecture = "alder_lake";
  processor.features.Add(CpuFeature::kSse2);
  processor.features.Add(CpuFeature::kAvx2);

  CpuSelector selector;
  selector.architecture = "AMD64";
  selector.vendor = "GenuineIntel";
  selector.family = 6;
  selector.models = {0x8E, 0x97};
  selector.microarchitecture = "Alder Lake";
  selector.required_features.Add(CpuFeature::kAvx2);
  selector.excluded_features.Add(CpuFeature::kAvx512F);
  selector.minimum_threads = 2;
  selector.maximum_threads = 16;

  EXPECT_TRUE(selector.Matches(processor, 8));
  EXPECT_FALSE(selector.Matches(processor));
  EXPECT_FALSE(selector.Matches(processor, 1));
  EXPECT_FALSE(selector.Matches(processor, 32));

  selector.models = {0x8E};
  EXPECT_FALSE(selector.Matches(processor, 8));
}

TEST(CpuSelector, RejectsUnknownRequiredProperties) {
  CpuDescriptor processor;
  processor.architecture = "aarch64";

  CpuSelector selector;
  selector.vendor = "arm";
  EXPECT_FALSE(selector.Matches(processor));

  selector.vendor.reset();
  selector.family = 8;
  EXPECT_FALSE(selector.Matches(processor));

  selector.family.reset();
  selector.models = {0xD0C};
  EXPECT_FALSE(selector.Matches(processor));
}

TEST(CpuDescriptor, DetectsAConsistentProcessDescriptor) {
  const CpuDescriptor &first = GetCpuDescriptor();
  const CpuDescriptor &second = GetCpuDescriptor();

  EXPECT_EQ(&first, &second);
  EXPECT_FALSE(first.architecture.empty());
  if (first.features.Has(CpuFeature::kAvx2)) {
    EXPECT_TRUE(first.features.Has(CpuFeature::kAvx));
  }
  if (first.features.Has(CpuFeature::kAvx512F)) {
    EXPECT_TRUE(first.features.Has(CpuFeature::kAvx));
  }
  if (first.logical_cores.has_value()) {
    EXPECT_GT(*first.logical_cores, 0U);
  }
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::platform
