/**
 * bench_load_external_nocopy.cpp
 *
 * Standalone C++ benchmark for the 2-file (external-data) + no-copy load path.
 * The external weights file is memory-mapped directly into the process address
 * space so that each tensor's raw_data borrows a pointer into the mapping without
 * any memcpy.  This corresponds to the Python benchmark key
 * ``load/2filex1/onnxlight-nocopy`` from ``plot_onnx_time.py``.
 *
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux profiling
 * tools (perf, gprof, valgrind/callgrind) can attribute wall-clock or instruction
 * samples back to named C++ functions.
 *
 * Usage:
 *   bench_load_external_nocopy [-n iters] [-t threads] [-i n_init] [-d dim]
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

static const std::string kOnnxFile = "bench_ext_nc.onnx";
static const std::string kDataFile = "bench_ext_nc.onnx.data";

/**
 * Builds a synthetic ModelProto with n_init float tensors of shape [dim, dim].
 *
 * @param n_init Number of initializer tensors to add to the graph.
 * @param dim Side length of each square float weight matrix.
 * @return A fully populated ModelProto ready for serialization.
 */
static ModelProto build_model(int n_init, int dim) {
  ModelProto model;
  model.set_ir_version(9);
  model.set_producer_name("bench_load_external_nocopy");

  GraphProto *graph = model.add_graph();
  graph->set_name("bench_graph");

  const size_t n_floats = static_cast<size_t>(dim) * static_cast<size_t>(dim);
  const size_t n_bytes = n_floats * sizeof(float);

  std::vector<uint8_t> raw(n_bytes);
  for (size_t i = 0; i < n_bytes; ++i) {
    raw[i] = static_cast<uint8_t>((i * 2654435761ULL) >> 24);
  }

  for (int i = 0; i < n_init; ++i) {
    TensorProto *tensor = graph->add_initializer();
    tensor->set_name("W" + std::to_string(i));
    tensor->set_data_type(TensorProto::DataType::FLOAT);
    tensor->add_dims(static_cast<int64_t>(dim));
    tensor->add_dims(static_cast<int64_t>(dim));
    tensor->set_raw_data(raw);
  }

  return model;
}

/**
 * Saves the model to two files (onnx + external data).
 *
 * @param model ModelProto to serialize.
 */
static void save_model(ModelProto &model) {
  TwoFilesWriteStream wstream(kOnnxFile, kDataFile);
  SerializeOptions sopts;
  sopts.raw_data_threshold = 0;
  SerializeModelProtoToStream(model, wstream, sopts);
}

/**
 * Runs the 2-file no-copy load loop.
 *
 * Each iteration opens the two-file stream, parses the model, and lets every
 * tensor borrow a direct pointer into the memory-mapped external-data file.
 * No per-tensor malloc+memcpy is performed.
 *
 * @param n_iters Number of load iterations.
 * @return Total number of tensor initializers loaded across all iterations.
 */
static size_t run_load_external_nocopy(int n_iters) {
  ParseOptions opts;
  opts.no_copy = true;

  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    TwoFilesStream rstream(kOnnxFile, kDataFile);
    m.ParseFromStream(rstream, opts);
    total_tensors += m.ref_graph().ref_initializer().size();
  }
  return total_tensors;
}

/**
 * Runs the 2-file copy (standard) load loop for comparison.
 *
 * Each iteration allocates one buffer per tensor and copies the tensor bytes.
 *
 * @param n_iters Number of load iterations.
 * @return Total number of tensor initializers loaded across all iterations.
 */
static size_t run_load_external_copy(int n_iters) {
  ParseOptions opts;
  // no_copy stays false

  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    TwoFilesStream rstream(kOnnxFile, kDataFile);
    m.ParseFromStream(rstream, opts);
    total_tensors += m.ref_graph().ref_initializer().size();
  }
  return total_tensors;
}

int main(int argc, char *argv[]) {
  int n_iters = 20;
  int n_threads = 1;
  int n_init = 40;
  int dim = 2048;

  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0) {
      n_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-t") == 0) {
      n_threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-i") == 0) {
      n_init = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-d") == 0) {
      dim = std::atoi(argv[++i]);
    }
  }
  (void)n_threads; // reserved for future parallel variant

  std::cout << "bench_load_external_nocopy\n"
            << "  n_iters  = " << n_iters << "\n"
            << "  n_threads= " << n_threads << "\n"
            << "  n_init   = " << n_init << "\n"
            << "  dim      = " << dim << "\n";

  ModelProto model = build_model(n_init, dim);
  save_model(model);

  if (std::filesystem::exists(kDataFile)) {
    const double data_mb =
        static_cast<double>(std::filesystem::file_size(kDataFile)) / (1024.0 * 1024.0);
    std::cout << "  data_mb  = " << data_mb << " MB\n\n";
  }

  // --- no-copy (mmap) ---
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t nc_tensors = run_load_external_nocopy(n_iters);
  auto t1 = std::chrono::high_resolution_clock::now();
  const double nc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // --- standard copy ---
  auto t2 = std::chrono::high_resolution_clock::now();
  size_t cp_tensors = run_load_external_copy(n_iters);
  auto t3 = std::chrono::high_resolution_clock::now();
  const double cp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "load/2filex1/onnxlight-nocopy : " << nc_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << nc_tensors << ")\n"
            << "load/2filex1/onnxlight        : " << cp_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << cp_tensors << ")\n";

  std::remove(kOnnxFile.c_str());
  std::remove(kDataFile.c_str());
  return 0;
}
