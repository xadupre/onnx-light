// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning.h"
#include "onnx_core/runtime/kernel_tuning_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
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

std::string OptionalValue(const auto &value) {
  return value.has_value() ? std::to_string(*value) : "-";
}

void WriteExecution(std::ostream &stream, const CpuExecutionDescriptor &execution) {
  const platform::CpuDescriptor &processor = execution.processor;
  stream << "architecture " << std::quoted(processor.architecture) << '\n'
         << "vendor " << std::quoted(processor.vendor) << '\n'
         << "family " << OptionalValue(processor.family) << '\n'
         << "model " << OptionalValue(processor.model) << '\n'
         << "stepping " << OptionalValue(processor.stepping) << '\n'
         << "microarchitecture " << std::quoted(processor.microarchitecture) << '\n'
         << "features " << processor.features.bits() << '\n'
         << "cache_line_bytes " << OptionalValue(processor.cache_line_bytes) << '\n'
         << "l1_data_bytes " << OptionalValue(processor.l1_data_bytes) << '\n'
         << "l2_bytes " << OptionalValue(processor.l2_bytes) << '\n'
         << "l3_bytes " << OptionalValue(processor.l3_bytes) << '\n'
         << "physical_cores " << OptionalValue(processor.physical_cores) << '\n'
         << "logical_cores " << OptionalValue(processor.logical_cores) << '\n'
         << "effective_threads " << execution.effective_threads << '\n';
}

void WriteProfile(std::ostream &stream, const KernelTuningParameters &parameters,
                  const CpuExecutionDescriptor &execution) {
  stream << "profile\n"
         << "library " << std::quoted(parameters.key.library) << '\n'
         << "kernel " << std::quoted(parameters.key.kernel) << '\n'
         << "implementation " << std::quoted(parameters.key.implementation) << '\n'
         << "element_type " << parameters.key.element_type << '\n'
         << "device " << static_cast<int32_t>(parameters.key.device) << '\n'
         << "tuning_abi " << parameters.key.tuning_abi << '\n';
  WriteExecution(stream, execution);
  for (const auto &[name, value] : parameters.values) {
    stream << "value " << std::quoted(name) << ' ' << TuningValueTypeName(value) << ' ';
    std::visit(
        [&stream](const auto &typed_value) {
          using T = std::decay_t<decltype(typed_value)>;
          if constexpr (std::is_same_v<T, bool>) {
            stream << (typed_value ? "true" : "false");
          } else if constexpr (std::is_same_v<T, std::string>) {
            stream << std::quoted(typed_value);
          } else {
            stream << typed_value;
          }
        },
        value);
    stream << '\n';
  }
  stream << "end\n";
}

class TemporaryCache {
public:
  explicit TemporaryCache(std::string_view suffix)
      : path_(std::filesystem::temp_directory_path() /
              ("onnx_light_kernel_tuning_" + std::string(suffix) + ".cache")) {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  ~TemporaryCache() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

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

TEST(KernelTuningRegistry, KeepsPublishedGenerationsImmutable) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  KernelTuningRegistrySnapshot portable = registry.Snapshot();
  KernelTuningParameters tuned = MakeDefaults();
  tuned.values["algorithm.tile_m"] = int64_t{128};

  registry.PublishProfiles(std::span<const KernelTuningParameters>(&tuned, 1));
  KernelTuningRegistrySnapshot published = registry.Snapshot();

  EXPECT_LT(portable.generation(), published.generation());
  ASSERT_NE(portable.Find(MakeKey()), nullptr);
  ASSERT_NE(published.Find(MakeKey()), nullptr);
  EXPECT_EQ(portable.Find(MakeKey())->Get<int64_t>("algorithm.tile_m"), 64);
  EXPECT_EQ(published.Find(MakeKey())->Get<int64_t>("algorithm.tile_m"), 128);
}

TEST(KernelTuningRegistry, RejectsBatchWithoutPartialPublication) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  KernelTuningRegistrySnapshot before = registry.Snapshot();
  KernelTuningParameters valid = MakeDefaults();
  valid.values["algorithm.tile_m"] = int64_t{128};
  KernelTuningParameters invalid = MakeDefaults();
  invalid.values.erase("algorithm.tile_m");
  const std::array profiles{valid, invalid};

  EXPECT_THROW(registry.PublishProfiles(profiles), std::invalid_argument);
  KernelTuningRegistrySnapshot after = registry.Snapshot();
  EXPECT_EQ(before.generation(), after.generation());
  EXPECT_EQ(after.Find(MakeKey())->Get<int64_t>("algorithm.tile_m"), 64);
}

TEST(KernelTuningCache, LoadsCompatibleProfileAndPreservesOldSnapshot) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_load_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  KernelTuningRegistrySnapshot before = GetKernelTuningRegistry().Snapshot();
  KernelTuningParameters tuned = defaults;
  tuned.values["algorithm.tile_m"] = int64_t{192};
  CpuExecutionDescriptor execution{platform::GetCpuDescriptor(), 3};
  TemporaryCache cache("compatible");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, tuned, execution);
  }

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  KernelTuningCacheLoadReport report = LoadKernelTuningCache(selection, {cache.path(), execution});
  KernelTuningRegistrySnapshot after = GetKernelTuningRegistry().Snapshot();

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kLoaded);
  EXPECT_EQ(report.loaded.size(), 1);
  EXPECT_TRUE(report.invalid.empty());
  EXPECT_TRUE(report.missing.empty());
  EXPECT_EQ(after.Find(defaults.key)->Get<int64_t>("algorithm.tile_m"), 192);
  EXPECT_EQ(before.Find(defaults.key)->Get<int64_t>("algorithm.tile_m"), 64);
}

TEST(KernelTuningCache, AllowsSameKeyForDifferentExecutionDescriptors) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_multiple_execution_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  KernelTuningParameters other = defaults;
  other.values["algorithm.tile_m"] = int64_t{96};
  KernelTuningParameters matching = defaults;
  matching.values["algorithm.tile_m"] = int64_t{224};
  CpuExecutionDescriptor execution{platform::GetCpuDescriptor(), 4};
  CpuExecutionDescriptor other_execution = execution;
  other_execution.effective_threads = 2;
  TemporaryCache cache("multiple_execution");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, other, other_execution);
    WriteProfile(stream, matching, execution);
  }

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  KernelTuningCacheLoadReport report = LoadKernelTuningCache(selection, {cache.path(), execution});

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kLoaded);
  EXPECT_EQ(report.loaded.size(), 1);
  EXPECT_EQ(report.incompatible.size(), 1);
  EXPECT_EQ(
      GetKernelTuningRegistry().Snapshot().Find(defaults.key)->Get<int64_t>("algorithm.tile_m"),
      224);
}

TEST(KernelTuningCache, RejectsMalformedFileWithoutPublishing) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_malformed_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  KernelTuningRegistrySnapshot before = GetKernelTuningRegistry().Snapshot();
  TemporaryCache cache("malformed");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\nprofile\nlibrary \"unfinished\"\n";
  }

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  KernelTuningCacheLoadReport report =
      LoadKernelTuningCache(selection, {cache.path(), std::nullopt});

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kMalformed);
  EXPECT_EQ(GetKernelTuningRegistry().Snapshot().generation(), before.generation());
  EXPECT_EQ(
      GetKernelTuningRegistry().Snapshot().Find(defaults.key)->Get<int64_t>("algorithm.tile_m"),
      64);
}

TEST(KernelTuningCache, RejectsInvalidEntryAsAWhole) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_invalid_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  KernelTuningParameters invalid = defaults;
  invalid.values.erase("algorithm.tile_m");
  CpuExecutionDescriptor execution{platform::GetCpuDescriptor(), 2};
  TemporaryCache cache("invalid");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, invalid, execution);
  }

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  KernelTuningCacheLoadReport report = LoadKernelTuningCache(selection, {cache.path(), execution});
  const KernelTuningParameters *resolved = GetKernelTuningRegistry().Snapshot().Find(defaults.key);

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kLoaded);
  EXPECT_EQ(report.invalid.size(), 1);
  EXPECT_EQ(report.missing.size(), 1);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->Get<int64_t>("algorithm.tile_m"), 64);
}

TEST(KernelTuningCache, ReportsMissingFileWithoutPublishing) {
  TemporaryCache cache("missing");
  KernelTuningRegistrySnapshot before = GetKernelTuningRegistry().Snapshot();

  KernelTuningCacheLoadReport report = LoadKernelTuningCache({}, {cache.path(), std::nullopt});

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kNotFound);
  EXPECT_EQ(GetKernelTuningRegistry().Snapshot().generation(), before.generation());
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
