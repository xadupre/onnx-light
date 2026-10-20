// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/runtime_session_loader.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments {
  std::filesystem::path directory;
  size_t weights = 8;
  size_t dimension = 1024;
  size_t iterations = 3;
};

size_t PositiveInteger(const std::string &text, size_t maximum) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected a positive integer: " + text);
  }
  const unsigned long long value = std::stoull(text);
  if (value == 0 || value > maximum) {
    throw std::invalid_argument("argument outside the supported benchmark bounds: " + text);
  }
  return static_cast<size_t>(value);
}

Arguments ParseArguments(int argc, char **argv) {
  Arguments arguments;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) {
      throw std::invalid_argument("missing argument value");
    }
    const std::string flag = argv[i];
    if (flag == "--directory") {
      arguments.directory = argv[i + 1];
    } else if (flag == "--weights") {
      arguments.weights = PositiveInteger(argv[i + 1], 64);
    } else if (flag == "--dimension") {
      arguments.dimension = PositiveInteger(argv[i + 1], 4096);
    } else if (flag == "--iterations") {
      arguments.iterations = PositiveInteger(argv[i + 1], 1000);
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }
  const uint64_t bytes = static_cast<uint64_t>(arguments.weights) * arguments.dimension *
                         arguments.dimension * sizeof(float);
  if (bytes > uint64_t{512} * 1024 * 1024) {
    throw std::invalid_argument("generated weights must not exceed 512 MiB");
  }
  if (arguments.directory.empty()) {
    arguments.directory =
        std::filesystem::temp_directory_path() / "onnx-light-bench-session-preparation";
  }
  return arguments;
}

class GeneratedModel {
public:
  explicit GeneratedModel(const Arguments &arguments) : directory_(arguments.directory) {
    if (!std::filesystem::create_directory(directory_)) {
      throw std::invalid_argument("--directory must name a new directory");
    }
  }

  ~GeneratedModel() {
    std::error_code ignored;
    std::filesystem::remove(directory_ / "model.onnx", ignored);
    std::filesystem::remove(directory_ / "weights.data", ignored);
    std::filesystem::remove(directory_, ignored);
  }

  std::string path() const { return (directory_ / "model.onnx").string(); }

  void Write(const Arguments &arguments) {
    if constexpr (std::endian::native != std::endian::little) {
      throw std::runtime_error("this benchmark generates little-endian FLOAT data");
    }
    ModelProto model;
    model.set_ir_version(9);
    auto *opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(18);
    auto &graph = *model.add_graph();
    graph.set_name("external_gemm_preparation");
    const auto add_value = [&](ValueInfoProto *value, const std::string &name) {
      value->set_name(name);
      auto *type = value->mutable_type()->mutable_tensor_type();
      type->set_elem_type(TensorProto::FLOAT);
      type->mutable_shape()->add_dim()->set_dim_value(1);
      type->mutable_shape()->add_dim()->set_dim_value(static_cast<int64_t>(arguments.dimension));
    };
    add_value(graph.add_input(), "x");
    add_value(graph.add_output(), "y");
    const size_t elements = arguments.dimension * arguments.dimension;
    const size_t bytes = elements * sizeof(float);
    std::vector<float> data(elements);
    std::ofstream weights(directory_ / "weights.data", std::ios::binary);
    weights.exceptions(std::ios::badbit | std::ios::failbit);
    for (size_t i = 0; i < arguments.weights; ++i) {
      // Near-identity matrices keep a long chain finite and nontrivial.
      for (size_t j = 0; j < elements; ++j) {
        data[j] = j / arguments.dimension == j % arguments.dimension
                      ? 0.95f
                      : static_cast<float>(static_cast<int>((j + i) % 17) - 8) /
                            static_cast<float>(arguments.dimension * 64);
      }
      weights.write(reinterpret_cast<const char *>(data.data()),
                    static_cast<std::streamsize>(bytes));
      auto &tensor = *graph.add_initializer();
      const std::string weight_name = "w" + std::to_string(i);
      tensor.set_name(weight_name);
      tensor.set_data_type(TensorProto::FLOAT);
      tensor.add_dims(static_cast<int64_t>(arguments.dimension));
      tensor.add_dims(static_cast<int64_t>(arguments.dimension));
      tensor.set_data_location(TensorProto::EXTERNAL);
      for (const auto &[key, value] :
           std::vector<std::pair<std::string, std::string>>{{"location", "weights.data"},
                                                            {"offset", std::to_string(i * bytes)},
                                                            {"length", std::to_string(bytes)}}) {
        auto *entry = tensor.add_external_data();
        entry->set_key(key);
        entry->set_value(value);
      }
      for (size_t reuse = 0; reuse < 2; ++reuse) {
        const size_t index = i * 2 + reuse;
        const std::string input = index == 0 ? "x" : "v" + std::to_string(index);
        const std::string output =
            index + 1 == arguments.weights * 2 ? "y" : "v" + std::to_string(index + 1);
        graph.add_node(MakeNode("Gemm", {input, weight_name}, {output}));
      }
    }
    weights.close();
    std::string serialized;
    if (!model.SerializeToString(serialized)) {
      throw std::runtime_error("failed to serialize benchmark model");
    }
    std::ofstream output(path(), std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    output.close();
  }

private:
  std::filesystem::path directory_;
};

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

uint64_t ProcessPeakRssBytes() {
#if defined(__linux__) || defined(__APPLE__)
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }
#if defined(__APPLE__)
  return static_cast<uint64_t>(usage.ru_maxrss);
#else
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
#else
  return 0;
#endif
}

struct Interval {
  uint64_t begin;
  uint64_t end;
};

struct Measurement {
  double load_ready_ms = 0;
  double model_load_ms = 0;
  double packing_ms = 0;
  double first_inference_ms = 0;
  double warm_inference_ms = 0;
  double overlap_ms = 0;
  size_t prepared_bytes = 0;
  size_t slots = 0;
  size_t admitted_peak = 0;
  size_t queued_peak = 0;
  size_t executed_packing_bytes = 0;
  uint64_t process_peak_rss = 0;
  std::optional<std::pair<Interval, Interval>> witness;
  std::vector<float> output;
};

Measurement Measure(const Arguments &arguments, const std::string &path, bool overlap) {
  Measurement measurement;
  std::mutex events_mutex;
  std::vector<RuntimeLoadEvent> events;
  events.reserve(arguments.weights * 10 + 8);
  RuntimeContext context;
  RuntimeLoadOptions load_options;
  load_options.overlap_preparation = overlap;
  const size_t matrix_bytes = arguments.dimension * arguments.dimension * sizeof(float);
  // Reserve source and conversion residency, persistent packs, and bounded work.
  load_options.memory_limit = (3 * arguments.weights + 4) * matrix_bytes;
  load_options.trace = [&](const RuntimeLoadEvent &event) {
    const std::lock_guard<std::mutex> lock(events_mutex);
    events.push_back(event);
  };
  RuntimeSessionOptions session_options;
  session_options.parameters.num_threads = 1;
  const auto start = Clock::now();
  LoadedRuntimeSession loaded(path, context, session_options, load_options);
  measurement.load_ready_ms = Milliseconds(Clock::now() - start);
  auto &session = loaded.session();
  if (!session.pending_preparations().empty()) {
    throw std::runtime_error("loaded root session still has pending Gemm preparations");
  }
  measurement.prepared_bytes = session.prepared_bytes();
  measurement.slots = session.value_store().slot_count();
  if (measurement.slots != arguments.weights ||
      measurement.prepared_bytes != arguments.weights * matrix_bytes) {
    throw std::runtime_error("repeated Gemms did not share exactly one packing per weight");
  }
  measurement.admitted_peak = loaded.statistics().peak_admitted_bytes;
  measurement.queued_peak = loaded.statistics().peak_queued_tensors;
  if (measurement.admitted_peak > load_options.memory_limit ||
      measurement.queued_peak > (overlap ? size_t{1} : size_t{0})) {
    throw std::runtime_error("loader exceeded its bounded admission or queue limit");
  }
  if (loaded.statistics().external_tensors != arguments.weights ||
      loaded.statistics().source_bytes != arguments.weights * matrix_bytes) {
    throw std::runtime_error("loader did not read the expected external-weight model");
  }

  // The first real inference feed is supplied only after loading/preparation returns.
  std::vector<float> input(arguments.dimension);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 23.0f;
  }
  context.Set("x", Tensor::FromFloat("x", {1, static_cast<int64_t>(arguments.dimension)}, input));
  const auto first_start = Clock::now();
  session.Run(context);
  measurement.first_inference_ms = Milliseconds(Clock::now() - first_start);
  const auto &result = context.Get("y");
  if (result.data_type != DataType::FLOAT ||
      result.shape != Shape{1, static_cast<int64_t>(arguments.dimension)}) {
    throw std::runtime_error("Gemm chain produced an unexpected output type or shape");
  }
  measurement.output.assign(result.AsFloat(), result.AsFloat() + arguments.dimension);
  context.Put("x", Tensor::FromFloat("x", {1, static_cast<int64_t>(arguments.dimension)}, input),
              RuntimeEventKind::kInput);
  const auto warm_start = Clock::now();
  session.Run(context);
  measurement.warm_inference_ms = Milliseconds(Clock::now() - warm_start);
  const float *warm = context.Get("y").AsFloat();
  if (!std::equal(measurement.output.begin(), measurement.output.end(), warm) ||
      session.prepared_bytes() != measurement.prepared_bytes ||
      session.value_store().slot_count() != measurement.slots) {
    throw std::runtime_error("warm replay changed output or repeated preparation");
  }
  measurement.process_peak_rss = ProcessPeakRssBytes();

  std::sort(events.begin(), events.end(), [](const auto &left, const auto &right) {
    return left.timestamp_ns < right.timestamp_ns;
  });
  const uint64_t origin =
      std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
  std::unordered_map<std::string, uint64_t> reads;
  std::optional<uint64_t> packing_start;
  std::vector<Interval> read_intervals;
  std::vector<Interval> packing_intervals;
  for (const auto &event : events) {
    const uint64_t time = event.timestamp_ns - origin;
    switch (event.phase) {
    case RuntimeLoadPhase::kReadStarted:
      reads.emplace(event.tensor_name, time);
      break;
    case RuntimeLoadPhase::kReadFinished: {
      const auto read = reads.find(event.tensor_name);
      if (read == reads.end()) {
        throw std::runtime_error("read-finished trace has no matching read-started event");
      }
      read_intervals.push_back({read->second, time});
      measurement.model_load_ms = static_cast<double>(time) / 1e6;
      reads.erase(read);
      break;
    }
    case RuntimeLoadPhase::kPreparationStarted:
      packing_start = time;
      break;
    case RuntimeLoadPhase::kPreparationFinished:
      if (!packing_start.has_value()) {
        throw std::runtime_error("preparation trace has no matching start");
      }
      if (event.prepared_bytes_added > 0) {
        packing_intervals.push_back({*packing_start, time});
        measurement.packing_ms += static_cast<double>(time - *packing_start) / 1e6;
        measurement.executed_packing_bytes += event.prepared_bytes_added;
      }
      packing_start.reset();
      break;
    default:
      break;
    }
  }
  if (!reads.empty() || packing_start.has_value() || read_intervals.size() != arguments.weights) {
    throw std::runtime_error("incomplete external-weight loading trace");
  }
  if (measurement.executed_packing_bytes != measurement.prepared_bytes) {
    throw std::runtime_error("trace did not witness every actual packed allocation");
  }
  for (const auto &read : read_intervals) {
    for (const auto &packing : packing_intervals) {
      const uint64_t begin = std::max(read.begin, packing.begin);
      const uint64_t end = std::min(read.end, packing.end);
      if (begin < end) {
        measurement.overlap_ms += static_cast<double>(end - begin) / 1e6;
        if (!measurement.witness.has_value()) {
          measurement.witness = std::make_pair(read, packing);
        }
      }
    }
  }
  return measurement;
}

void Print(const char *mode, size_t iteration, const Measurement &measurement) {
  std::cout << mode << " iteration=" << iteration << " model_load_ms=" << measurement.model_load_ms
            << " load_ready_ms=" << measurement.load_ready_ms
            << " packing_ms=" << measurement.packing_ms
            << " first_inference_ms=" << measurement.first_inference_ms
            << " warm_inference_ms=" << measurement.warm_inference_ms
            << " io_packing_overlap_ms=" << measurement.overlap_ms
            << " admission_peak_bytes=" << measurement.admitted_peak
            << " queued_tensors_peak=" << measurement.queued_peak
            << " process_lifetime_peak_rss_bytes=" << measurement.process_peak_rss
            << " prepared_bytes=" << measurement.prepared_bytes << " slots=" << measurement.slots
            << " executed_packing_bytes=" << measurement.executed_packing_bytes
            << " shared_operation_reuse=PASS warm_replay=PASS\n";
  if (measurement.witness.has_value()) {
    const auto &[read, packing] = *measurement.witness;
    std::cout << "  overlap_witness_ns_since_load_start: io=[" << read.begin << "," << read.end
              << ") preparation=[" << packing.begin << "," << packing.end << ")\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    std::cout << "Usage: bench_session_preparation [--directory new-path] [--weights 1..64]"
                 " [--dimension 1..4096] [--iterations 1..1000]\n";
    return 0;
  }
  onnx_kernels::RegisterKernelFunctions();
  const Arguments arguments = ParseArguments(argc, argv);
  GeneratedModel generated(arguments);
  generated.Write(arguments);
  std::cout << std::fixed << std::setprecision(3)
            << "external FLOAT Gemm chain: weights=" << arguments.weights
            << " dimension=" << arguments.dimension << " nodes=" << arguments.weights * 2
            << " iterations=" << arguments.iterations << " admission_limit_bytes="
            << (3 * arguments.weights + 4) * arguments.dimension * arguments.dimension *
                   sizeof(float)
            << "\n"
            << "model_load_ms ends at the last read (including intervening packing);"
               " load_ready_ms includes all packing.\n"
            << "packing_ms includes preparation dispatch/publication; tracing is enabled.\n"
            << "Files are freshly generated (page cache likely warm); no cold-disk claim.\n"
            << "RSS is process-lifetime high water, including generation and preceding modes;"
               " 0 means unavailable.\n";
  double serial_total = 0;
  double overlap_total = 0;
  bool observed_overlap = false;
  for (size_t iteration = 0; iteration < arguments.iterations; ++iteration) {
    // Alternate order to reduce a systematic page-cache/order advantage.
    Measurement serial;
    Measurement overlap;
    if (iteration % 2 == 0) {
      serial = Measure(arguments, generated.path(), false);
      overlap = Measure(arguments, generated.path(), true);
    } else {
      overlap = Measure(arguments, generated.path(), true);
      serial = Measure(arguments, generated.path(), false);
    }
    Print("serial", iteration + 1, serial);
    Print("overlap", iteration + 1, overlap);
    if (serial.overlap_ms != 0) {
      throw std::runtime_error("serial read/packing intervals unexpectedly overlap");
    }
    float maximum_difference = 0;
    for (size_t i = 0; i < serial.output.size(); ++i) {
      if (!std::isfinite(serial.output[i]) || !std::isfinite(overlap.output[i])) {
        throw std::runtime_error("inference produced nonfinite output");
      }
      maximum_difference =
          std::max(maximum_difference, std::abs(serial.output[i] - overlap.output[i]));
    }
    if (maximum_difference > 1e-6f) {
      throw std::runtime_error("serial and overlap inference outputs differ");
    }
    std::cout << "numerical_equivalence=PASS max_abs_difference=" << std::scientific
              << maximum_difference << std::fixed << "\n";
    serial_total += serial.load_ready_ms;
    overlap_total += overlap.load_ready_ms;
    observed_overlap = observed_overlap || overlap.overlap_ms > 0;
  }
  std::cout << "mean_load_ready_ms: serial=" << serial_total / arguments.iterations
            << " overlap=" << overlap_total / arguments.iterations
            << " genuine_io_overlap_observed=" << (observed_overlap ? "yes" : "no") << "\n";
  return 0;
}
