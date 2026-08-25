// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/runtime/tuning/kernel_tuning_cache.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"

#include <algorithm>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace platform = ONNX_LIGHT_NAMESPACE::core::platform;
namespace symbolic = ONNX_LIGHT_NAMESPACE::core::symbolic;

namespace {

rt::CpuExecutionDescriptor LocalExecution(int32_t num_threads) {
  return {platform::GetCpuDescriptor(),
          static_cast<uint32_t>(rt::RuntimeParameters(num_threads).EffectiveNumThreads())};
}

rt::KernelTuningCacheOptions CacheOptions(const std::optional<std::string> &path,
                                          const rt::CpuExecutionDescriptor &execution) {
  rt::KernelTuningCacheOptions options;
  if (path.has_value()) {
    options.path = *path;
  }
  options.execution = execution;
  return options;
}

std::string LoadStatusName(rt::KernelTuningCacheLoadStatus status) {
  switch (status) {
  case rt::KernelTuningCacheLoadStatus::kLoaded:
    return "loaded";
  case rt::KernelTuningCacheLoadStatus::kNotFound:
    return "not_found";
  case rt::KernelTuningCacheLoadStatus::kUnreadable:
    return "unreadable";
  case rt::KernelTuningCacheLoadStatus::kMalformed:
    return "malformed";
  }
  throw std::invalid_argument("Unknown kernel tuning cache load status.");
}

std::string UpdateStatusName(rt::KernelTuningCacheUpdateStatus status) {
  switch (status) {
  case rt::KernelTuningCacheUpdateStatus::kUpdated:
    return "updated";
  case rt::KernelTuningCacheUpdateStatus::kReadOnly:
    return "read_only";
  case rt::KernelTuningCacheUpdateStatus::kUnreadable:
    return "unreadable";
  case rt::KernelTuningCacheUpdateStatus::kMalformed:
    return "malformed";
  case rt::KernelTuningCacheUpdateStatus::kWriteFailed:
    return "write_failed";
  }
  throw std::invalid_argument("Unknown kernel tuning cache update status.");
}

nb::object TuningValueToPython(const rt::TuningValue &value) {
  return std::visit([](const auto &typed_value) { return nb::cast(typed_value); }, value);
}

rt::TuningValue TuningValueFromPython(nb::handle value) {
  if (nb::isinstance<nb::bool_>(value)) {
    return nb::cast<bool>(value);
  }
  if (nb::isinstance<nb::int_>(value)) {
    return nb::cast<int64_t>(value);
  }
  if (nb::isinstance<nb::float_>(value)) {
    return nb::cast<double>(value);
  }
  if (nb::isinstance<nb::str>(value)) {
    return nb::cast<std::string>(value);
  }
  throw nb::type_error("Kernel tuning values must be bool, int, float, or str.");
}

nb::dict ValuesToDict(const rt::KernelTuningParameters &parameters) {
  std::vector<std::string> names;
  names.reserve(parameters.values.size());
  for (const auto &[name, value] : parameters.values) {
    (void)value;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  nb::dict values;
  for (const std::string &name : names) {
    values[nb::str(name.c_str())] = TuningValueToPython(parameters.values.at(name));
  }
  return values;
}

nb::dict KeyToDict(const rt::KernelTuningKey &key) {
  nb::dict result;
  result["library"] = key.library;
  result["kernel"] = key.kernel;
  result["implementation"] = key.implementation;
  result["element_type"] = key.element_type;
  result["device"] = static_cast<int32_t>(key.device);
  result["device_name"] = symbolic::DeviceName(key.device);
  result["tuning_abi"] = key.tuning_abi;
  return result;
}

nb::dict ParallelRegionToDict(const rt::ParallelRegionReportEvent &event) {
  std::ostringstream calling_thread_id;
  calling_thread_id << event.calling_thread_id;
  nb::dict result;
  result["region_id"] = event.region_id;
  result["parent_region_id"] = event.parent_region_id;
  result["run_id"] = event.run_id;
  result["calling_thread_id"] = calling_thread_id.str();
  result["label"] = event.label;
  result["file_name"] = event.file_name;
  result["function_name"] = event.function_name;
  result["line"] = event.line;
  result["column"] = event.column;
  result["total_iterations"] = event.total_iterations;
  result["grain_size"] = event.grain_size;
  result["requested_threads"] = event.requested_threads;
  result["admitted_threads"] = event.admitted_threads;
  result["observed_threads"] = event.observed_threads;
  result["wall_time_ns"] = event.wall_time_ns;
  result["process_cpu_time_ns"] = event.process_cpu_time_ns;
  result["cpu_utilization"] = event.cpu_utilization;
  result["counter_status"] = rt::HardwareCounterStatusName(event.counter_status);
  result["cpu_cycles"] = event.cpu_cycles;
  result["retired_instructions"] = event.retired_instructions;
  result["llc_references"] = event.llc_references;
  result["llc_misses"] = event.llc_misses;
  result["counter_time_enabled"] = event.counter_time_enabled;
  result["counter_time_running"] = event.counter_time_running;
  result["ipc"] = event.ipc;
  result["llc_miss_rate"] = event.llc_miss_rate;
  result["executor_instance_id"] = event.executor_instance_id;
  result["nested_inline"] = event.nested_inline;
  return result;
}

nb::dict ProfileToDict(const rt::CalibratedKernelProfile &profile) {
  nb::dict result = KeyToDict(profile.parameters.key);
  result["values"] = ValuesToDict(profile.parameters);
  result["effective_threads"] = profile.execution.effective_threads;
  result["architecture"] = profile.execution.processor.architecture;
  result["vendor"] = profile.execution.processor.vendor;
  result["microarchitecture"] = profile.execution.processor.microarchitecture;
  return result;
}

nb::list KeysToList(const std::vector<rt::KernelTuningKey> &keys) {
  nb::list result;
  for (const rt::KernelTuningKey &key : keys) {
    result.append(KeyToDict(key));
  }
  return result;
}

rt::KernelCalibrationSelection MakeSelection(const std::optional<std::string> &kernel,
                                             const std::optional<std::string> &library,
                                             const std::optional<std::string> &implementation,
                                             const std::optional<int32_t> &element_type,
                                             const std::optional<int32_t> &device = -1) {
  rt::KernelCalibrationSelection selection;
  selection.library = library;
  if (kernel.has_value()) {
    selection.kernels = {*kernel};
  }
  if (implementation.has_value()) {
    selection.implementations = {*implementation};
  }
  if (element_type.has_value()) {
    selection.element_types = {*element_type};
  }
  if (device.has_value()) {
    const auto selected_device = static_cast<rt::Device>(*device);
    if (symbolic::DeviceName(selected_device) == "Unknown") {
      throw std::invalid_argument("Unknown kernel tuning device value.");
    }
    selection.device = selected_device;
  }
  return selection;
}

bool KeyMatches(const rt::KernelTuningKey &key, const rt::KernelCalibrationSelection &selection,
                const std::optional<uint32_t> &tuning_abi = std::nullopt) {
  return selection.Matches(key) && (!tuning_abi.has_value() || key.tuning_abi == *tuning_abi);
}

std::vector<rt::KernelTuningKey>
SelectedKeys(const rt::KernelCalibrationSelection &selection,
             const std::optional<uint32_t> &tuning_abi = std::nullopt) {
  std::vector<rt::KernelTuningKey> keys;
  for (const rt::KernelTuningKey &key : rt::GetKernelTuningRegistry().RegisteredKeys()) {
    if (KeyMatches(key, selection, tuning_abi)) {
      keys.push_back(key);
    }
  }
  std::sort(keys.begin(), keys.end(), [](const auto &left, const auto &right) {
    return std::tie(left.library, left.kernel, left.implementation, left.element_type, left.device,
                    left.tuning_abi) < std::tie(right.library, right.kernel, right.implementation,
                                                right.element_type, right.device, right.tuning_abi);
  });
  return keys;
}

const rt::CalibratedKernelProfile *
FindCachedProfile(const rt::KernelTuningCacheInspectionReport &inspection,
                  const rt::KernelTuningKey &key, const rt::CpuExecutionDescriptor &execution) {
  const auto found =
      std::find_if(inspection.profiles.begin(), inspection.profiles.end(),
                   [&](const rt::CalibratedKernelProfile &profile) {
                     return profile.parameters.key == key && profile.execution == execution;
                   });
  return found == inspection.profiles.end() ? nullptr : &*found;
}

nb::dict InspectCache(const std::optional<std::string> &path, int32_t num_threads) {
  const rt::CpuExecutionDescriptor execution = LocalExecution(num_threads);
  const rt::KernelTuningCacheInspectionReport inspection =
      rt::InspectKernelTuningCache(CacheOptions(path, execution));
  nb::dict result;
  result["status"] = LoadStatusName(inspection.status);
  result["path"] = inspection.path.string();
  result["diagnostics"] = inspection.diagnostics;
  nb::list profiles;
  for (const rt::CalibratedKernelProfile &profile : inspection.profiles) {
    nb::dict item = ProfileToDict(profile);
    item["local"] = profile.execution == execution;
    profiles.append(std::move(item));
  }
  result["profiles"] = std::move(profiles);
  return result;
}

nb::dict ListParameters(const std::optional<std::string> &kernel,
                        const std::optional<std::string> &library,
                        const std::optional<std::string> &implementation,
                        const std::optional<int32_t> &element_type,
                        const std::optional<std::string> &path, int32_t num_threads,
                        const std::optional<int32_t> &device) {
  const rt::KernelCalibrationSelection selection =
      MakeSelection(kernel, library, implementation, element_type, device);
  const rt::CpuExecutionDescriptor execution = LocalExecution(num_threads);
  const rt::KernelTuningCacheInspectionReport inspection =
      rt::InspectKernelTuningCache(CacheOptions(path, execution));
  const rt::KernelTuningRegistrySnapshot snapshot = rt::GetKernelTuningRegistry().Snapshot();

  nb::dict result;
  result["cache_status"] = LoadStatusName(inspection.status);
  result["cache_path"] = inspection.path.string();
  result["diagnostics"] = inspection.diagnostics;
  nb::list kernels;
  for (const rt::KernelTuningKey &key : SelectedKeys(selection)) {
    const std::shared_ptr<const rt::KernelTuningSchema> schema =
        rt::GetKernelTuningRegistry().FindSchema(key);
    if (schema == nullptr) {
      continue;
    }
    nb::dict item = KeyToDict(key);
    const rt::KernelTuningParameters &defaults = schema->portable_defaults();
    item["defaults"] = ValuesToDict(defaults);
    item["calibratable"] =
        static_cast<bool>(rt::GetKernelTuningRegistry().FindCalibrationFunction(key));
    nb::list calibration_parameters;
    std::vector<std::string> sorted_calibration_parameters =
        rt::GetKernelTuningRegistry().FindCalibrationParameterNames(key);
    std::sort(sorted_calibration_parameters.begin(), sorted_calibration_parameters.end());
    for (const std::string &name : sorted_calibration_parameters) {
      calibration_parameters.append(name);
    }
    item["calibration_parameters"] = std::move(calibration_parameters);
    nb::list names;
    std::vector<std::string> sorted_names;
    for (const auto &[name, value] : defaults.values) {
      (void)value;
      sorted_names.push_back(name);
    }
    std::sort(sorted_names.begin(), sorted_names.end());
    for (const std::string &name : sorted_names) {
      names.append(name);
    }
    item["parameter_names"] = std::move(names);

    const rt::CalibratedKernelProfile *cached = FindCachedProfile(inspection, key, execution);
    if (cached == nullptr) {
      item["cached_values"] = nb::none();
    } else {
      item["cached_values"] = ValuesToDict(cached->parameters);
    }
    const rt::KernelTuningParameters *active = snapshot.Resolve(key, execution);
    if (active == nullptr) {
      item["active_values"] = nb::none();
    } else {
      item["active_values"] = ValuesToDict(*active);
    }
    item["active_source"] =
        active == nullptr ? "none"
                          : (snapshot.HasPublishedProfile(key, execution)
                                 ? "published_profile"
                                 : (active->values == defaults.values ? "portable_default"
                                                                      : "registered_profile"));
    kernels.append(std::move(item));
  }
  result["kernels"] = std::move(kernels);
  return result;
}

nb::dict LoadCache(const std::optional<std::string> &kernel, const std::string &library,
                   const std::optional<std::string> &implementation,
                   const std::optional<int32_t> &element_type,
                   const std::optional<std::string> &path, int32_t num_threads) {
  const rt::KernelCalibrationSelection selection =
      MakeSelection(kernel, library, implementation, element_type);
  const rt::CpuExecutionDescriptor execution = LocalExecution(num_threads);
  const rt::KernelTuningCacheOptions options = CacheOptions(path, execution);
  const rt::KernelTuningCacheLoadReport report = rt::LoadKernelTuningCache(selection, options);
  nb::dict result;
  result["status"] = LoadStatusName(report.status);
  result["path"] =
      (options.path.empty() ? rt::DefaultKernelTuningCachePath() : options.path).string();
  result["published_generation"] = report.published_generation;
  result["loaded"] = KeysToList(report.loaded);
  result["incompatible"] = KeysToList(report.incompatible);
  result["stale"] = KeysToList(report.stale);
  result["invalid"] = KeysToList(report.invalid);
  result["missing"] = KeysToList(report.missing);
  result["diagnostics"] = report.diagnostics;
  return result;
}

nb::dict SetParameters(const std::string &kernel, int32_t element_type, nb::dict values,
                       const std::string &library, const std::string &implementation,
                       const std::optional<uint32_t> &tuning_abi,
                       const std::optional<std::string> &path, int32_t num_threads, bool load) {
  const rt::KernelCalibrationSelection selection =
      MakeSelection(kernel, library, implementation, element_type);
  const std::vector<rt::KernelTuningKey> keys = SelectedKeys(selection, tuning_abi);
  if (keys.empty()) {
    throw nb::key_error("No registered kernel tuning schema matches the requested key.");
  }
  if (keys.size() != 1) {
    throw std::invalid_argument(
        "The requested kernel tuning key is ambiguous; specify tuning_abi.");
  }
  const rt::KernelTuningKey &key = keys[0];
  const std::shared_ptr<const rt::KernelTuningSchema> schema =
      rt::GetKernelTuningRegistry().FindSchema(key);
  if (schema == nullptr) {
    throw nb::key_error("The selected kernel tuning schema is no longer registered.");
  }
  const rt::CpuExecutionDescriptor execution = LocalExecution(num_threads);
  const rt::KernelTuningCacheOptions options = CacheOptions(path, execution);
  const rt::KernelTuningCacheInspectionReport inspection = rt::InspectKernelTuningCache(options);
  rt::KernelTuningParameters updated = schema->portable_defaults();
  if (const rt::CalibratedKernelProfile *cached = FindCachedProfile(inspection, key, execution)) {
    updated = cached->parameters;
  }
  for (auto [name, value] : values) {
    updated.values[nb::cast<std::string>(name)] = TuningValueFromPython(value);
  }
  schema->Validate(updated);

  const rt::KernelTuningCacheUpdateReport update = rt::UpdateKernelTuningCache(
      std::span<const rt::KernelTuningParameters>(&updated, 1), options);
  nb::dict result;
  result["status"] = UpdateStatusName(update.status);
  result["path"] =
      (options.path.empty() ? rt::DefaultKernelTuningCachePath() : options.path).string();
  result["values"] = ValuesToDict(updated);
  result["updated"] = KeysToList(update.updated);
  result["preserved"] = KeysToList(update.preserved);
  result["pruned"] = KeysToList(update.pruned);
  result["diagnostics"] = update.diagnostics;
  if (load && update.status == rt::KernelTuningCacheUpdateStatus::kUpdated) {
    result["load"] = LoadCache(kernel, library, implementation, element_type, path, num_threads);
  } else {
    result["load"] = nb::none();
  }
  return result;
}

nb::dict Calibrate(const std::string &kernel, const std::vector<int32_t> &element_types,
                   const std::string &library, const std::optional<std::string> &implementation,
                   bool only_missing, uint64_t maximum_duration_ms, uint64_t maximum_memory_bytes,
                   bool save, const std::optional<std::string> &path,
                   const std::optional<rt::CpuExecutionPolicy> &cpu_execution,
                   size_t profiling_capacity, bool profiling_hardware_counters,
                   const std::optional<int32_t> &device) {
  rt::KernelCalibrationSelection selection =
      MakeSelection(kernel, library, implementation, std::nullopt, device);
  selection.element_types = element_types;
  selection.only_missing = only_missing;
  rt::CalibrationOptions options;
  options.maximum_duration_ms = maximum_duration_ms;
  options.maximum_memory_bytes = maximum_memory_bytes;
  options.profiling_capacity = profiling_capacity;
  options.profiling_hardware_counters = profiling_hardware_counters;
  std::shared_ptr<rt::CpuExecutor> executor;
  if (cpu_execution.has_value()) {
    executor = rt::GlobalCpuExecutorRegistry().Acquire(*cpu_execution);
    options.execution =
        rt::CpuExecutionDescriptor{platform::GetCpuDescriptor(), executor->effective_threads()};
  }
  const rt::CpuExecutorScope executor_scope(executor == nullptr ? rt::CurrentCpuExecutor()
                                                                : executor.get());
  const rt::CalibrationBatchReport calibration = rt::CalibrateRegisteredKernels(selection, options);

  nb::dict result;
  result["published_generation"] = calibration.published_generation;
  nb::list calibrated;
  for (const rt::KernelTuningParameters &parameters : calibration.calibrated) {
    nb::dict profile = KeyToDict(parameters.key);
    profile["values"] = ValuesToDict(parameters);
    calibrated.append(std::move(profile));
  }
  result["calibrated"] = std::move(calibrated);
  result["skipped"] = KeysToList(calibration.skipped);
  result["unsupported"] = KeysToList(calibration.unsupported);
  nb::list diagnostics;
  for (const rt::KernelCalibrationDiagnostic &diagnostic : calibration.diagnostics) {
    nb::dict item = KeyToDict(diagnostic.key);
    item["message"] = diagnostic.message;
    diagnostics.append(std::move(item));
  }
  result["diagnostics"] = std::move(diagnostics);
  nb::list candidate_diagnostics;
  for (const rt::KernelCalibrationCandidateDiagnostic &diagnostic :
       calibration.candidate_diagnostics) {
    nb::dict item = KeyToDict(diagnostic.key);
    nb::list events;
    for (const rt::ParallelRegionReportEvent &event : diagnostic.parallel_regions.events()) {
      events.append(ParallelRegionToDict(event));
    }
    item["events"] = std::move(events);
    item["dropped_events"] = diagnostic.parallel_regions.dropped_events();
    candidate_diagnostics.append(std::move(item));
  }
  result["candidate_diagnostics"] = std::move(candidate_diagnostics);
  if (save && !calibration.calibrated.empty()) {
    rt::KernelTuningCacheOptions cache_options;
    if (path.has_value()) {
      cache_options.path = *path;
    }
    const rt::KernelTuningCacheUpdateReport update =
        rt::UpdateKernelTuningCache(calibration.successful_profiles(), cache_options);
    nb::dict persisted;
    persisted["status"] = UpdateStatusName(update.status);
    persisted["path"] =
        (path.has_value() ? std::filesystem::path(*path) : rt::DefaultKernelTuningCachePath())
            .string();
    persisted["updated"] = KeysToList(update.updated);
    persisted["diagnostics"] = update.diagnostics;
    result["cache_update"] = std::move(persisted);
  } else {
    result["cache_update"] = nb::none();
  }
  return result;
}

} // namespace

void AddOnnxPyTuning(nb::module_ &rt_mod) {
  rt_mod.def(
      "default_kernel_tuning_cache_path",
      []() { return rt::DefaultKernelTuningCachePath().string(); },
      "Returns the platform-specific default kernel tuning cache path.");
  rt_mod.def("inspect_kernel_tuning_cache", &InspectCache, nb::arg("path") = nb::none(),
             nb::arg("num_threads") = 0,
             "Returns every persisted tuning profile without activating the cache.");
  rt_mod.def("kernel_tuning_parameters", &ListParameters, nb::arg("kernel") = nb::none(),
             nb::arg("library") = "onnx_light", nb::arg("implementation") = nb::none(),
             nb::arg("element_type") = nb::none(), nb::arg("path") = nb::none(),
             nb::arg("num_threads") = 0, nb::arg("device") = -1,
             "Lists registered parameter names, portable defaults, matching cached values, "
             "and active values. None for library or device selects all values.");
  rt_mod.def("load_kernel_tuning_cache", &LoadCache, nb::arg("kernel") = nb::none(),
             nb::arg("library") = "onnx_light", nb::arg("implementation") = nb::none(),
             nb::arg("element_type") = nb::none(), nb::arg("path") = nb::none(),
             nb::arg("num_threads") = 0,
             "Validates and activates compatible profiles from a tuning cache.");
  rt_mod.def("set_kernel_tuning_parameters", &SetParameters, nb::arg("kernel"),
             nb::arg("element_type"), nb::arg("values"), nb::arg("library") = "onnx_light",
             nb::arg("implementation") = "portable", nb::arg("tuning_abi") = nb::none(),
             nb::arg("path") = nb::none(), nb::arg("num_threads") = 0, nb::arg("load") = true,
             "Validates and persists a partial parameter update for the local processor.");
  rt_mod.def("calibrate_kernel_tuning", &Calibrate, nb::arg("kernel"),
             nb::arg("element_types") = std::vector<int32_t>{}, nb::arg("library") = "onnx_light",
             nb::arg("implementation") = nb::none(), nb::arg("only_missing") = false,
             nb::arg("maximum_duration_ms") = uint64_t{0},
             nb::arg("maximum_memory_bytes") = uint64_t{0}, nb::arg("save") = true,
             nb::arg("path") = nb::none(), nb::arg("cpu_execution").none() = nb::none(),
             nb::arg("profiling_capacity") = size_t{0},
             nb::arg("profiling_hardware_counters") = false, nb::arg("device") = -1,
             "Calibrates selected registered keys and optionally persists successful profiles.");
}
