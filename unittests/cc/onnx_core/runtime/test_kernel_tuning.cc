// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_tuning.h"
#include "onnx_core/runtime/kernel_tuning_cache.h"
#include "onnx_core/runtime/parallel_for.h"

#include <gtest/gtest.h>

#include <algorithm>
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

CpuExecutionDescriptor MakeExecution() {
  CpuExecutionDescriptor execution;
  execution.processor.architecture = "x86_64";
  execution.processor.vendor = "intel";
  execution.processor.family = 6;
  execution.processor.model = 0x97;
  execution.processor.microarchitecture = "alder_lake";
  execution.processor.features.Add(platform::CpuFeature::kSse2);
  execution.processor.features.Add(platform::CpuFeature::kAvx2);
  execution.effective_threads = 8;
  return execution;
}

KernelTuningParameters WithTileM(int64_t tile_m) {
  KernelTuningParameters parameters = MakeDefaults();
  parameters.values["algorithm.tile_m"] = tile_m;
  return parameters;
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
    std::filesystem::remove(path_.string() + ".lock", error);
    std::filesystem::remove(path_.string() + ".tmp", error);
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

TEST(KernelTuningRegistry, CountsSnapshotsAndProfileResolutions) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  const KernelTuningRegistryAccessCounts before = registry.AccessCounts();

  const KernelTuningRegistrySnapshot snapshot = registry.Snapshot();
  ASSERT_NE(snapshot.Resolve(MakeKey(), MakeExecution()), nullptr);
  ASSERT_NE(snapshot.Resolve(MakeKey(), MakeExecution()), nullptr);
  const KernelTuningRegistryAccessCounts after = registry.AccessCounts();

  EXPECT_EQ(after.snapshots - before.snapshots, 1u);
  EXPECT_EQ(after.lookups - before.lookups, 2u);
  EXPECT_EQ(after.resolutions - before.resolutions, 2u);
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

TEST(KernelTuningRegistry, ResolvesExactListAndInstructionSetPrecedence) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));

  platform::CpuSelector instruction_set;
  instruction_set.required_features.Add(platform::CpuFeature::kAvx2);
  registry.RegisterProfile(MakeKey(), instruction_set, WithTileM(80));

  platform::CpuSelector processor_list;
  processor_list.vendor = "intel";
  processor_list.family = 6;
  processor_list.models = {0x8E, 0x97};
  registry.RegisterProfile(MakeKey(), processor_list, WithTileM(96));

  platform::CpuSelector exact;
  exact.vendor = "GenuineIntel";
  exact.family = 6;
  exact.models = {0x97};
  registry.RegisterProfile(MakeKey(), exact, WithTileM(128));

  const KernelTuningRegistrySnapshot snapshot = registry.Snapshot();
  CpuExecutionDescriptor execution = MakeExecution();
  ASSERT_NE(snapshot.Resolve(MakeKey(), execution), nullptr);
  EXPECT_EQ(snapshot.Resolve(MakeKey(), execution)->Get<int64_t>("algorithm.tile_m"), 128);

  execution.processor.model = 0x8E;
  EXPECT_EQ(snapshot.Resolve(MakeKey(), execution)->Get<int64_t>("algorithm.tile_m"), 96);

  execution.processor.vendor = "amd";
  execution.processor.family = 25;
  execution.processor.model = 0x21;
  EXPECT_EQ(snapshot.Resolve(MakeKey(), execution)->Get<int64_t>("algorithm.tile_m"), 80);

  execution.processor.features = {};
  EXPECT_EQ(snapshot.Resolve(MakeKey(), execution)->Get<int64_t>("algorithm.tile_m"), 64);
}

TEST(KernelTuningRegistry, UsesPriorityOnlyAtEqualSpecificity) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  platform::CpuSelector avx2;
  avx2.required_features.Add(platform::CpuFeature::kAvx2);

  registry.RegisterProfile(MakeKey(), avx2, WithTileM(80), 0);
  registry.RegisterProfile(MakeKey(), avx2, WithTileM(112), 1);

  const KernelTuningParameters *resolved = registry.Snapshot().Resolve(MakeKey(), MakeExecution());
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->Get<int64_t>("algorithm.tile_m"), 112);
}

TEST(KernelTuningRegistry, RejectsInvalidAndAmbiguousSelectors) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  platform::CpuSelector empty;
  EXPECT_THROW(registry.RegisterProfile(MakeKey(), empty, MakeDefaults()), std::invalid_argument);

  platform::CpuSelector avx2;
  avx2.required_features.Add(platform::CpuFeature::kAvx2);
  registry.RegisterProfile(MakeKey(), avx2, WithTileM(80));
  EXPECT_THROW(registry.RegisterProfile(MakeKey(), avx2, WithTileM(96)), std::invalid_argument);

  platform::CpuSelector impossible;
  impossible.required_features.Add(platform::CpuFeature::kAvx2);
  impossible.excluded_features.Add(platform::CpuFeature::kAvx2);
  EXPECT_THROW(registry.RegisterProfile(MakeKey(), impossible, WithTileM(96)),
               std::invalid_argument);
}

TEST(KernelTuningRegistry, PublishedProfileOverridesProcessorProfile) {
  KernelTuningRegistry registry;
  registry.RegisterSchema(KernelTuningSchema(MakeDefaults()));
  platform::CpuSelector avx2;
  avx2.required_features.Add(platform::CpuFeature::kAvx2);
  registry.RegisterProfile(MakeKey(), avx2, WithTileM(80));
  KernelTuningRegistrySnapshot before = registry.Snapshot();
  KernelTuningParameters published = WithTileM(160);

  registry.PublishProfiles(std::span<const KernelTuningParameters>(&published, 1));
  KernelTuningRegistrySnapshot after = registry.Snapshot();

  EXPECT_EQ(before.Resolve(MakeKey(), MakeExecution())->Get<int64_t>("algorithm.tile_m"), 80);
  EXPECT_EQ(after.Resolve(MakeKey(), MakeExecution())->Get<int64_t>("algorithm.tile_m"), 160);
}

TEST(KernelCalibration, SelectsReportsAndPublishesCallbacks) {
  KernelTuningParameters success_defaults = MakeDefaults();
  success_defaults.key.library = "calibration_batch_test";
  success_defaults.key.kernel = "Success";
  KernelTuningParameters unsupported_defaults = success_defaults;
  unsupported_defaults.key.kernel = "Unsupported";
  RegisterKernelTuningSchema(KernelTuningSchema(success_defaults));
  RegisterKernelTuningSchema(KernelTuningSchema(unsupported_defaults));

  RegisterKernelCalibrationFunction(
      success_defaults.key,
      [success_defaults](const KernelTuningKey &key, const CpuExecutionDescriptor &execution,
                         const CalibrationOptions &options, CalibrationReporter &reporter) {
        EXPECT_EQ(key, success_defaults.key);
        EXPECT_EQ(execution.effective_threads, 3u);
        EXPECT_EQ(options.maximum_duration_ms, 50u);
        reporter.AddDiagnostic("measured crossover");
        reporter.RecordBenchmark(1024, 2000);
        KernelTuningParameters calibrated = success_defaults;
        calibrated.values["algorithm.tile_m"] = int64_t{192};
        return calibrated;
      });

  KernelCalibrationSelection selection;
  selection.library = success_defaults.key.library;
  selection.kernels = {"Success", "Unsupported"};
  CalibrationOptions options;
  options.execution = MakeExecution();
  options.maximum_threads = 3;
  options.maximum_duration_ms = 50;
  const uint64_t generation_before = GetKernelTuningRegistry().Snapshot().generation();

  CalibrationBatchReport report = CalibrateRegisteredKernels(selection, options);

  ASSERT_EQ(report.calibrated.size(), 1u);
  EXPECT_EQ(report.calibrated[0].key, success_defaults.key);
  EXPECT_EQ(report.unsupported, std::vector<KernelTuningKey>({unsupported_defaults.key}));
  EXPECT_TRUE(report.skipped.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_TRUE(std::any_of(report.diagnostics.begin(), report.diagnostics.end(),
                          [&](const KernelCalibrationDiagnostic &diagnostic) {
                            return diagnostic.key == success_defaults.key &&
                                   diagnostic.message == "measured crossover";
                          }));
  ASSERT_EQ(report.resources.size(), 1u);
  EXPECT_EQ(report.resources[0].key, success_defaults.key);
  EXPECT_EQ(report.resources[0].benchmark_cases, 1u);
  EXPECT_EQ(report.resources[0].peak_memory_bytes, 1024u);
  EXPECT_EQ(report.resources[0].measured_duration_ns, 2000u);
  EXPECT_GT(report.published_generation, generation_before);
  EXPECT_EQ(report.successful_profiles().size(), 1u);
  CpuExecutionDescriptor calibrated_execution = *options.execution;
  calibrated_execution.effective_threads = 3;
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(success_defaults.key, calibrated_execution)
                ->Get<int64_t>("algorithm.tile_m"),
            192);
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(success_defaults.key, *options.execution)
                ->Get<int64_t>("algorithm.tile_m"),
            64);
}

TEST(KernelCalibration, BuildsUnaryAndBroadcastingBinaryCases) {
  const std::vector<KernelCalibrationCase> unary =
      MakeElementwiseCalibrationCases(DataType::FLOAT, 1, 16, 16, false);
  ASSERT_EQ(unary.size(), 1u);
  EXPECT_EQ(unary[0].name, "unary");
  EXPECT_EQ(unary[0].inputs.size(), 1u);
  EXPECT_EQ(unary[0].inputs[0].shape, Shape({16}));
  EXPECT_EQ(unary[0].output_shape, Shape({16}));

  const std::vector<KernelCalibrationCase> binary =
      MakeElementwiseCalibrationCases(DataType::UINT32, 2, 16, 16, true);
  ASSERT_EQ(binary.size(), 3u);
  EXPECT_EQ(binary[0].name, "equal_shape");
  EXPECT_EQ(binary[1].name, "scalar_broadcast");
  EXPECT_EQ(binary[1].inputs[1].shape, Shape({}));
  EXPECT_EQ(binary[2].name, "multidirectional_broadcast");
  EXPECT_EQ(binary[2].inputs[0].shape, Shape({4, 4}));
  EXPECT_EQ(binary[2].inputs[1].shape, Shape({1, 4}));
  EXPECT_EQ(binary[2].output_shape, Shape({4, 4}));

  EXPECT_THROW(MakeElementwiseCalibrationCases(DataType::FLOAT, 0, 16, 16, false),
               std::invalid_argument);
  EXPECT_THROW(MakeElementwiseCalibrationCases(DataType::FLOAT, 1, 16, 16, true),
               std::invalid_argument);
}

TEST(KernelCalibration, ValidatesCandidateOutput) {
  if (ParallelForThreadCount() == 1) {
    GTEST_SKIP() << "Parallel calibration is unavailable.";
  }
  KernelTuningParameters defaults = MakeDefaults();
  KernelCalibrationBenchmark benchmark;
  benchmark.portable_parameters = defaults;
  benchmark.parameter_name = "algorithm.tile_m";
  benchmark.cases = MakeElementwiseCalibrationCases(DataType::FLOAT, 1, 16, 16, false);
  benchmark.repetitions = 1;
  benchmark.required_consecutive_wins = 1;
  benchmark.reference.configure = [](int64_t) {};
  benchmark.candidate.configure = [](int64_t) {};
  benchmark.reference.run = [](std::span<const Tensor>, Tensor &output) {
    std::fill_n(output.AsFloat(), output.element_count(), 0.0f);
  };
  benchmark.candidate.run = [](std::span<const Tensor>, Tensor &output) {
    std::fill_n(output.AsFloat(), output.element_count(), 1.0f);
  };
  CpuExecutionDescriptor execution = MakeExecution();
  execution.effective_threads = static_cast<uint32_t>(ParallelForThreadCount());
  CalibrationReporter reporter;

  EXPECT_THROW(CalibrateKernelBenchmark(defaults.key, execution, {}, reporter, benchmark),
               std::runtime_error);
}

TEST(KernelCalibration, CallbackFailurePropagatesWithoutPublishingBatch) {
  KernelTuningParameters success_defaults = MakeDefaults();
  success_defaults.key.library = "calibration_failure_test";
  success_defaults.key.kernel = "Success";
  KernelTuningParameters failure_defaults = success_defaults;
  failure_defaults.key.kernel = "ZFailure";
  RegisterKernelTuningSchema(KernelTuningSchema(success_defaults));
  RegisterKernelTuningSchema(KernelTuningSchema(failure_defaults));
  RegisterKernelCalibrationFunction(
      success_defaults.key,
      [success_defaults](const KernelTuningKey &, const CpuExecutionDescriptor &,
                         const CalibrationOptions &, CalibrationReporter &) {
        KernelTuningParameters calibrated = success_defaults;
        calibrated.values["algorithm.tile_m"] = int64_t{192};
        return calibrated;
      });
  RegisterKernelCalibrationFunction(failure_defaults.key,
                                    [](const KernelTuningKey &, const CpuExecutionDescriptor &,
                                       const CalibrationOptions &,
                                       CalibrationReporter &) -> KernelTuningParameters {
                                      throw std::runtime_error("measurement failed");
                                    });

  KernelCalibrationSelection selection;
  selection.library = success_defaults.key.library;
  CalibrationOptions options;
  options.execution = MakeExecution();
  const uint64_t generation_before = GetKernelTuningRegistry().Snapshot().generation();

  EXPECT_THROW(CalibrateRegisteredKernels(selection, options), std::runtime_error);

  const KernelTuningRegistrySnapshot after = GetKernelTuningRegistry().Snapshot();
  EXPECT_EQ(after.generation(), generation_before);
  EXPECT_EQ(
      after.Resolve(success_defaults.key, *options.execution)->Get<int64_t>("algorithm.tile_m"),
      64);
}

TEST(KernelCalibration, OnlyMissingSkipsPublishedProfile) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "calibration_only_missing_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  RegisterKernelCalibrationFunction(
      defaults.key, [defaults](const KernelTuningKey &, const CpuExecutionDescriptor &,
                               const CalibrationOptions &, CalibrationReporter &) {
        KernelTuningParameters calibrated = defaults;
        calibrated.values["algorithm.tile_m"] = int64_t{128};
        return calibrated;
      });

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  CalibrationOptions options;
  options.execution = MakeExecution();
  EXPECT_EQ(CalibrateRegisteredKernels(selection, options).calibrated.size(), 1u);

  selection.only_missing = true;
  CalibrationBatchReport report = CalibrateRegisteredKernels(selection, options);
  EXPECT_TRUE(report.calibrated.empty());
  EXPECT_EQ(report.skipped, std::vector<KernelTuningKey>({defaults.key}));

  options.execution->effective_threads = 2;
  report = CalibrateRegisteredKernels(selection, options);
  EXPECT_EQ(report.calibrated.size(), 1u);
  EXPECT_TRUE(report.skipped.empty());
}

TEST(KernelCalibration, RejectsInvalidRegistrationAndOptions) {
  KernelTuningRegistry registry;
  EXPECT_THROW(registry.RegisterCalibrationFunction(MakeKey(), {}), std::invalid_argument);
  EXPECT_THROW(registry.RegisterCalibrationFunction(
                   MakeKey(), [](const KernelTuningKey &, const CpuExecutionDescriptor &,
                                 const CalibrationOptions &,
                                 CalibrationReporter &) { return MakeDefaults(); }),
               std::invalid_argument);

  CalibrationOptions options;
  options.maximum_threads = 0;
  EXPECT_THROW(CalibrateRegisteredKernels({}, options), std::invalid_argument);
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
  EXPECT_EQ(after.Resolve(defaults.key, execution)->Get<int64_t>("algorithm.tile_m"), 192);
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
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(defaults.key, execution)
                ->Get<int64_t>("algorithm.tile_m"),
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
  const KernelTuningParameters *resolved =
      GetKernelTuningRegistry().Snapshot().Resolve(defaults.key, execution);

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

TEST(KernelTuningCache, AtomicallyCreatesMergesAndReplacesProfiles) {
  KernelTuningParameters first = MakeDefaults();
  first.key.library = "cache_update_merge_test";
  first.key.kernel = "First";
  KernelTuningParameters second = first;
  second.key.kernel = "Second";
  RegisterKernelTuningSchema(KernelTuningSchema(first));
  RegisterKernelTuningSchema(KernelTuningSchema(second));
  CpuExecutionDescriptor execution = MakeExecution();
  TemporaryCache cache("update_merge");

  first.values["algorithm.tile_m"] = int64_t{96};
  KernelTuningCacheUpdateReport report = UpdateKernelTuningCache(
      std::span<const KernelTuningParameters>(&first, 1), {cache.path(), execution});
  EXPECT_EQ(report.status, KernelTuningCacheUpdateStatus::kUpdated);
  EXPECT_EQ(report.updated, std::vector<KernelTuningKey>({first.key}));

  second.values["algorithm.tile_m"] = int64_t{128};
  report = UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&second, 1),
                                   {cache.path(), execution});
  EXPECT_EQ(report.status, KernelTuningCacheUpdateStatus::kUpdated);

  first.values["algorithm.tile_m"] = int64_t{192};
  report = UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&first, 1),
                                   {cache.path(), execution});
  EXPECT_EQ(report.status, KernelTuningCacheUpdateStatus::kUpdated);

  KernelCalibrationSelection selection;
  selection.library = first.key.library;
  KernelTuningCacheLoadReport loaded = LoadKernelTuningCache(selection, {cache.path(), execution});
  EXPECT_EQ(loaded.loaded.size(), 2u);
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(first.key, execution)
                ->Get<int64_t>("algorithm.tile_m"),
            192);
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(second.key, execution)
                ->Get<int64_t>("algorithm.tile_m"),
            128);
}

TEST(KernelTuningCache, HonorsReadOnlyAndReplacementOptions) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_update_options_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  CpuExecutionDescriptor execution = MakeExecution();
  TemporaryCache cache("update_options");
  KernelTuningParameters initial = defaults;
  initial.values["algorithm.tile_m"] = int64_t{96};
  ASSERT_EQ(UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&initial, 1),
                                    {cache.path(), execution})
                .status,
            KernelTuningCacheUpdateStatus::kUpdated);

  KernelTuningParameters replacement = defaults;
  replacement.values["algorithm.tile_m"] = int64_t{224};
  KernelTuningCacheOptions no_replace{cache.path(), execution};
  no_replace.replace_existing = false;
  KernelTuningCacheUpdateReport report =
      UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&replacement, 1), no_replace);
  EXPECT_TRUE(report.updated.empty());
  EXPECT_EQ(report.preserved, std::vector<KernelTuningKey>({defaults.key}));

  KernelTuningCacheOptions read_only{cache.path(), execution};
  read_only.read_only = true;
  report =
      UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&replacement, 1), read_only);
  EXPECT_EQ(report.status, KernelTuningCacheUpdateStatus::kReadOnly);

  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  ASSERT_EQ(LoadKernelTuningCache(selection, {cache.path(), execution}).loaded.size(), 1u);
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(defaults.key, execution)
                ->Get<int64_t>("algorithm.tile_m"),
            96);
}

TEST(KernelTuningCache, PreservesMalformedCacheAndPrunesStaleAbi) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "cache_update_stale_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  CpuExecutionDescriptor execution = MakeExecution();
  TemporaryCache cache("update_stale");
  {
    std::ofstream stream(cache.path());
    stream << "malformed cache\n";
  }
  const std::string malformed = "malformed cache\n";
  EXPECT_EQ(UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&defaults, 1),
                                    {cache.path(), execution})
                .status,
            KernelTuningCacheUpdateStatus::kMalformed);
  {
    std::ifstream stream(cache.path());
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()),
              malformed);
  }

  KernelTuningParameters stale = defaults;
  stale.key.tuning_abi += 1;
  {
    std::ofstream stream(cache.path(), std::ios::trunc);
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, stale, execution);
  }
  KernelTuningCacheOptions options{cache.path(), execution};
  options.prune_stale_abis = true;
  KernelTuningCacheUpdateReport report =
      UpdateKernelTuningCache(std::span<const KernelTuningParameters>(&defaults, 1), options);
  EXPECT_EQ(report.status, KernelTuningCacheUpdateStatus::kUpdated);
  EXPECT_EQ(report.pruned, std::vector<KernelTuningKey>({stale.key}));
}

TEST(KernelTuningCache, ImportsDeploymentProfileOnlyForExplicitSelector) {
  KernelTuningParameters defaults = MakeDefaults();
  defaults.key.library = "deployment_import_test";
  RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  KernelTuningParameters tuned = defaults;
  tuned.values["algorithm.tile_m"] = int64_t{176};
  TemporaryCache cache("deployment_import");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, tuned, MakeExecution());
  }

  platform::CpuSelector processors;
  processors.vendor = "intel";
  processors.family = 6;
  processors.models = {0x97};
  KernelCalibrationSelection selection;
  selection.library = defaults.key.library;
  KernelTuningDeploymentImportReport report =
      ImportKernelTuningDeploymentProfiles(selection, {cache.path(), processors, 0});

  EXPECT_EQ(report.status, KernelTuningCacheLoadStatus::kLoaded);
  EXPECT_EQ(report.imported, std::vector<KernelTuningKey>({defaults.key}));
  CpuExecutionDescriptor matching = MakeExecution();
  CpuExecutionDescriptor outside = matching;
  outside.processor.model = 0x8E;
  const KernelTuningRegistrySnapshot snapshot = GetKernelTuningRegistry().Snapshot();
  EXPECT_EQ(snapshot.Resolve(defaults.key, matching)->Get<int64_t>("algorithm.tile_m"), 176);
  EXPECT_EQ(snapshot.Resolve(defaults.key, outside)->Get<int64_t>("algorithm.tile_m"), 64);
}

TEST(KernelTuningCache, RejectsDeploymentImportWithoutPartialRegistration) {
  KernelTuningParameters valid = MakeDefaults();
  valid.key.library = "deployment_import_atomic_test";
  valid.key.kernel = "Valid";
  KernelTuningParameters invalid = valid;
  invalid.key.kernel = "Invalid";
  RegisterKernelTuningSchema(KernelTuningSchema(valid));
  RegisterKernelTuningSchema(KernelTuningSchema(invalid));
  valid.values["algorithm.tile_m"] = int64_t{144};
  invalid.values.erase("algorithm.tile_m");
  TemporaryCache cache("deployment_import_atomic");
  {
    std::ofstream stream(cache.path());
    stream << "onnx_light_kernel_tuning_cache 1\n";
    WriteProfile(stream, valid, MakeExecution());
    WriteProfile(stream, invalid, MakeExecution());
  }

  platform::CpuSelector processors;
  processors.vendor = "intel";
  KernelCalibrationSelection selection;
  selection.library = valid.key.library;
  const uint64_t generation_before = GetKernelTuningRegistry().Snapshot().generation();
  KernelTuningDeploymentImportReport report =
      ImportKernelTuningDeploymentProfiles(selection, {cache.path(), processors, 0});

  EXPECT_EQ(report.invalid, std::vector<KernelTuningKey>({invalid.key}));
  EXPECT_TRUE(report.imported.empty());
  EXPECT_EQ(GetKernelTuningRegistry().Snapshot().generation(), generation_before);
  EXPECT_EQ(GetKernelTuningRegistry()
                .Snapshot()
                .Resolve(valid.key, MakeExecution())
                ->Get<int64_t>("algorithm.tile_m"),
            64);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
