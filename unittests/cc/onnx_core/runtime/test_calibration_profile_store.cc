// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/calibration_profile_store.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

class TemporaryProfileStore {
public:
  explicit TemporaryProfileStore(std::string_view suffix)
      : path_(std::filesystem::temp_directory_path() /
              ("onnx_light_calibration_profiles_" + std::string(suffix) + ".cache")) {
    Remove();
  }

  ~TemporaryProfileStore() { Remove(); }

  const std::filesystem::path &path() const { return path_; }

private:
  void Remove() {
    std::error_code error;
    std::filesystem::remove(path_, error);
    std::filesystem::remove(path_.string() + ".lock", error);
    std::filesystem::remove(path_.string() + ".tmp", error);
  }

  std::filesystem::path path_;
};

CalibrationProfileKey ExactKey(std::string digest = "model-sha256") {
  CalibrationProfileKey key;
  key.backend = "example_backend";
  key.operator_name = "Gemm";
  key.implementation_version = "2";
  key.model_digest = std::move(digest);
  key.processor = "x86_64:vendor:model:l1=32768:l2=1048576";
  key.thread_count = 8;
  return key;
}

CalibrationProfileKey PortableKey(std::string shape = "m=dynamic,n=4096,k=4096") {
  CalibrationProfileKey key;
  key.backend = "example_backend";
  key.operator_name = "Gemm";
  key.implementation_version = "2";
  key.kind = CalibrationProfileKind::kPortable;
  key.structural_properties = {{"shape_class", std::move(shape)}, {"dtype", "float32"}};
  return key;
}

CalibrationProfileLookupOptions LookupOptions() {
  CalibrationProfileLookupOptions options;
  options.exact_key = ExactKey();
  options.structural_properties = PortableKey().structural_properties;
  return options;
}

bool AcceptExamplePolicy(std::string_view policy, std::string &error) {
  if (policy.starts_with("algorithm=")) {
    return true;
  }
  error = "missing algorithm";
  return false;
}

TEST(CalibrationProfileStore, RoundTripsExactProfileAcrossInstances) {
  TemporaryProfileStore temporary("round_trip");
  {
    CalibrationProfileStore writer({temporary.path(), true});
    const CalibrationProfileStoreReport stored = writer.Store(
        ExactKey(), {{"median_latency", 1.25, "us"}, {"samples", 7, "count"}},
        [] { return "algorithm=blocked;tile=32"; }, AcceptExamplePolicy);
    EXPECT_EQ(stored.status, CalibrationProfileStoreStatus::kOk);
  }

  CalibrationProfileStore reader({temporary.path(), true});
  const CalibrationProfileLookupReport found = reader.Lookup(LookupOptions(), AcceptExamplePolicy);
  ASSERT_EQ(found.status, CalibrationProfileStoreStatus::kOk);
  ASSERT_TRUE(found.profile.has_value());
  EXPECT_EQ(found.profile->policy, "algorithm=blocked;tile=32");
  ASSERT_EQ(found.profile->measurements.size(), 2u);
  EXPECT_DOUBLE_EQ(found.profile->measurements[0].value, 1.25);
}

#if !defined(_WIN32)
TEST(CalibrationProfileStore, RoundTripsAcrossProcesses) {
  TemporaryProfileStore temporary("process_round_trip");
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    CalibrationProfileStore writer({temporary.path(), true});
    const CalibrationProfileStoreReport report =
        writer.Store(ExactKey(), {}, [] { return "algorithm=child"; });
    ::_exit(report.status == CalibrationProfileStoreStatus::kOk ? 0 : 1);
  }
  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  ASSERT_EQ(WEXITSTATUS(child_status), 0);

  CalibrationProfileStore reader({temporary.path(), true});
  const CalibrationProfileLookupReport found = reader.Lookup(LookupOptions());
  ASSERT_TRUE(found.profile.has_value());
  EXPECT_EQ(found.profile->policy, "algorithm=child");
}
#endif

TEST(CalibrationProfileStore, ResolvesOverridesAndForcedPortableProfiles) {
  TemporaryProfileStore temporary("overrides");
  CalibrationProfileStore store({temporary.path(), true});
  ASSERT_EQ(store.Store(ExactKey(), {}, [] { return "algorithm=exact"; }).status,
            CalibrationProfileStoreStatus::kOk);
  ASSERT_EQ(store.Store(PortableKey(), {}, [] { return "algorithm=portable"; }).status,
            CalibrationProfileStoreStatus::kOk);
  ASSERT_EQ(store.InstallOverride(ExactKey(), [] { return "algorithm=user"; }).status,
            CalibrationProfileStoreStatus::kOk);

  CalibrationProfileLookupOptions options = LookupOptions();
  ASSERT_EQ(store.Lookup(options).profile->policy, "algorithm=user");
  options.force_portable = true;
  ASSERT_EQ(store.Lookup(options).profile->policy, "algorithm=portable");

  ASSERT_EQ(store.ClearOverride(ExactKey()).affected_profiles, 1u);
  options.force_portable = false;
  ASSERT_EQ(store.Lookup(options).profile->policy, "algorithm=exact");
}

TEST(CalibrationProfileStore, InvalidatesByBackendAndImplementationVersion) {
  TemporaryProfileStore temporary("invalidate");
  CalibrationProfileStore store({temporary.path(), true});
  ASSERT_EQ(store.Store(ExactKey(), {}, [] { return "algorithm=current"; }).status,
            CalibrationProfileStoreStatus::kOk);
  CalibrationProfileKey old = ExactKey("old-model");
  old.implementation_version = "1";
  ASSERT_EQ(store.Store(old, {}, [] { return "algorithm=old"; }).status,
            CalibrationProfileStoreStatus::kOk);

  const CalibrationProfileStoreReport invalidated =
      store.Invalidate("example_backend", std::string_view("1"));
  EXPECT_EQ(invalidated.affected_profiles, 1u);
  ASSERT_EQ(store.Inspect().size(), 1u);
  EXPECT_EQ(store.Invalidate("example_backend").affected_profiles, 1u);
  EXPECT_TRUE(store.Inspect().empty());

  CalibrationProfileStore reloaded({temporary.path(), true});
  EXPECT_TRUE(reloaded.Inspect().empty());
}

TEST(CalibrationProfileStore, ReportsCorruptionVersionAndPolicyRejectionExplicitly) {
  TemporaryProfileStore malformed("malformed");
  {
    std::ofstream output(malformed.path());
    output << "onnx_light_calibration_profiles 1\nprofile exact calibrated\n";
  }
  CalibrationProfileStore malformed_store({malformed.path(), true});
  EXPECT_EQ(malformed_store.Reload().status, CalibrationProfileStoreStatus::kMalformed);

  TemporaryProfileStore future("future");
  {
    std::ofstream output(future.path());
    output << "onnx_light_calibration_profiles 99\n";
  }
  CalibrationProfileStore future_store({future.path(), true});
  EXPECT_EQ(future_store.Reload().status, CalibrationProfileStoreStatus::kUnsupportedVersion);

  TemporaryProfileStore rejected("rejected");
  CalibrationProfileStore rejected_store({rejected.path(), true});
  EXPECT_EQ(rejected_store.Store(
                              ExactKey(), {}, [] { return "invalid"; }, AcceptExamplePolicy)
                .status,
            CalibrationProfileStoreStatus::kPolicyRejected);
  EXPECT_TRUE(rejected_store.Inspect().empty());
  ASSERT_EQ(rejected_store.Store(ExactKey(), {}, [] { return "invalid"; }).status,
            CalibrationProfileStoreStatus::kOk);
  const CalibrationProfileLookupReport lookup =
      rejected_store.Lookup(LookupOptions(), AcceptExamplePolicy);
  EXPECT_EQ(lookup.status, CalibrationProfileStoreStatus::kNotFound);
  ASSERT_EQ(lookup.rejections.size(), 1u);
  EXPECT_EQ(lookup.rejections[0].reason, CalibrationProfileRejectionReason::kPolicyRejected);
}

TEST(CalibrationProfileStore, ReportsOutdatedProfilesAsRejections) {
  TemporaryProfileStore temporary("outdated");
  CalibrationProfileStore store({temporary.path(), true});
  CalibrationProfileKey old = ExactKey();
  old.implementation_version = "1";
  ASSERT_EQ(store.Store(old, {}, [] { return "algorithm=old"; }).status,
            CalibrationProfileStoreStatus::kOk);

  const CalibrationProfileLookupReport lookup = store.Lookup(LookupOptions());
  EXPECT_EQ(lookup.status, CalibrationProfileStoreStatus::kNotFound);
  ASSERT_EQ(lookup.rejections.size(), 1u);
  EXPECT_EQ(lookup.rejections[0].reason,
            CalibrationProfileRejectionReason::kOutdatedImplementation);
}

TEST(CalibrationProfileStore, DisabledPersistenceRetainsInMemoryCalibration) {
  TemporaryProfileStore temporary("disabled");
  CalibrationProfileStore store({temporary.path(), false});
  const CalibrationProfileStoreReport stored =
      store.Store(ExactKey(), {}, [] { return "algorithm=memory"; });
  EXPECT_EQ(stored.status, CalibrationProfileStoreStatus::kDisabled);
  ASSERT_EQ(store.Lookup(LookupOptions()).profile->policy, "algorithm=memory");
  EXPECT_FALSE(std::filesystem::exists(temporary.path()));
  EXPECT_EQ(store.Reload().status, CalibrationProfileStoreStatus::kDisabled);
  ASSERT_EQ(store.Lookup(LookupOptions()).profile->policy, "algorithm=memory");
}

TEST(CalibrationProfileStore, ConcurrentWritersMergeCompleteProfiles) {
  TemporaryProfileStore temporary("concurrent");
  constexpr size_t kWriters = 8;
  std::vector<std::unique_ptr<CalibrationProfileStore>> stores;
  stores.reserve(kWriters);
  for (size_t index = 0; index < kWriters; ++index) {
    stores.push_back(std::make_unique<CalibrationProfileStore>(
        CalibrationProfileStoreOptions{temporary.path(), true}));
  }
  std::atomic<size_t> successful{0};
  std::vector<std::thread> threads;
  for (size_t index = 0; index < kWriters; ++index) {
    threads.emplace_back([&, index] {
      CalibrationProfileKey key = ExactKey("model-" + std::to_string(index));
      const CalibrationProfileStoreReport report =
          stores[index]->Store(key, {{"latency", static_cast<double>(index), "us"}},
                               [index] { return "algorithm=" + std::to_string(index); });
      if (report.status == CalibrationProfileStoreStatus::kOk) {
        ++successful;
      }
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(successful, kWriters);
  CalibrationProfileStore merged({temporary.path(), true});
  EXPECT_EQ(merged.Inspect().size(), kWriters);
}

TEST(CalibrationProfileStore, InterruptedTemporaryWriteCannotReplaceExistingStore) {
  TemporaryProfileStore temporary("atomic");
  CalibrationProfileStore store({temporary.path(), true});
  ASSERT_EQ(store.Store(ExactKey(), {}, [] { return "algorithm=existing"; }).status,
            CalibrationProfileStoreStatus::kOk);
  {
    std::ofstream interrupted(temporary.path().string() + ".tmp", std::ios::trunc);
    interrupted << "partial replacement";
  }

  CalibrationProfileStore after_interruption({temporary.path(), true});
  ASSERT_EQ(after_interruption.Lookup(LookupOptions()).profile->policy, "algorithm=existing");
  ASSERT_EQ(after_interruption.Store(PortableKey(), {}, [] { return "algorithm=new"; }).status,
            CalibrationProfileStoreStatus::kOk);
  CalibrationProfileStore final_reader({temporary.path(), true});
  EXPECT_EQ(final_reader.Inspect().size(), 2u);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
