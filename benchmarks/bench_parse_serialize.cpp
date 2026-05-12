/**
 * bench_parse_serialize.cpp
 *
 * Standalone C++ benchmark for the onnx-light parse and serialize paths.
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux
 * profiling tools (perf, gprof, valgrind/callgrind) can attribute wall-clock
 * or instruction samples back to named C++ functions.
 *
 * Build (see CMakeLists.txt ONNX_LIGHT_BUILD_BENCHMARKS option):
 *
 *   cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 *                  -DONNX_LIGHT_BUILD_BENCHMARKS=ON \
 *                  -DONNX_LIGHT_BUILD_PYTHON=OFF
 *   cmake --build build --target bench_parse_serialize -j
 *
 * Usage:
 *   ./build/bench_parse_serialize [OPTIONS]
 *     -n <iters>    Number of parse+serialize iterations  (default: 20)
 *     -t <threads>  Thread count for parallel mode (0=auto, default: 1)
 *     -i <tensors>  Number of initializer tensors          (default: 40)
 *     -d <dim>      Tensor side length in floats           (default: 512)
 *
 * Typical profiling workflow (see docs/examples/core/plot_onnx_profile.py):
 *
 *   # perf stat (counts)
 *   perf stat ./build/bench_parse_serialize -n 20
 *
 *   # perf record + report (flame-graph ready)
 *   perf record -g ./build/bench_parse_serialize -n 500
 *   perf report --stdio --no-children -n | head -60
 *
 *   # gprof  (requires recompile with -pg; see CMakeLists GPROF option)
 *   cmake -B build_gprof -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 *                        -DONNX_LIGHT_BUILD_BENCHMARKS=ON \
 *                        -DONNX_LIGHT_BUILD_PYTHON=OFF \
 *                        -DONNX_LIGHT_BENCH_GPROF=ON
 *   cmake --build build_gprof --target bench_parse_serialize -j
 *   ./build_gprof/bench_parse_serialize -n 20
 *   gprof ./build_gprof/bench_parse_serialize gmon.out | head -40
 *
 *   # valgrind callgrind
 *   valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
 *       ./build/bench_parse_serialize -n 20
 *   callgrind_annotate callgrind.out | head -80
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/**
 * Builds a synthetic ModelProto with *n_init* float tensors of shape
 * [dim, dim] containing random-looking data (not cryptographically random,
 * but sufficient to prevent constant-folding by the compiler).
 *
 * @param n_init  Number of initializer tensors to add to the graph.
 * @param dim     Side length of each square float weight matrix.
 * @return        A fully populated ModelProto ready for serialization.
 */
static ModelProto build_model(int n_init, int dim) {
  ModelProto model;
  model.set_ir_version(9);
  model.set_producer_name("bench_parse_serialize");

  GraphProto &graph = model.add_graph();
  graph.set_name("bench_graph");

  const size_t n_floats = static_cast<size_t>(dim) * static_cast<size_t>(dim);
  const size_t n_bytes = n_floats * sizeof(float);

  // Simple pseudo-random fill so the compiler cannot eliminate the data.
  std::vector<uint8_t> raw(n_bytes);
  for (size_t i = 0; i < n_bytes; ++i) {
    raw[i] = static_cast<uint8_t>((i * 2654435761ULL) >> 24);
  }

  for (int i = 0; i < n_init; ++i) {
    TensorProto &tensor = graph.add_initializer();
    tensor.set_name("W" + std::to_string(i));
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.add_dims(static_cast<int64_t>(dim));
    tensor.add_dims(static_cast<int64_t>(dim));
    tensor.set_raw_data(raw);
  }

  return model;
}

/**
 * Runs the serialize loop: converts *model* to bytes *n_iters* times.
 *
 * @param model     ModelProto to serialize.
 * @param n_iters   Number of serialization iterations to execute.
 * @param n_threads Thread count (1 = sequential, >1 = parallel mode).
 * @return Total number of bytes written across all iterations
 *         (prevents dead-code elimination by the compiler).
 */
static size_t run_serialize(ModelProto &model, int n_iters, int n_threads) {
  SerializeOptions opts;
  if (n_threads != 1) {
    opts.parallel = true;
    opts.num_threads = n_threads;
  }

  size_t total_bytes = 0;
  for (int i = 0; i < n_iters; ++i) {
    std::string out;
    model.SerializeToString(out, opts);
    total_bytes += out.size();
  }
  return total_bytes;
}

/**
 * Runs the parse loop: reconstructs a ModelProto from *serialized* bytes
 * *n_iters* times.
 *
 * @param serialized  Bytes produced by SerializeToString.
 * @param n_iters     Number of parse iterations to execute.
 * @param n_threads   Thread count (1 = sequential, >1 = parallel mode).
 * @return Total number of initializer tensors parsed across all iterations
 *         (prevents dead-code elimination by the compiler).
 */
static size_t run_parse(const std::string &serialized, int n_iters, int n_threads) {
  ParseOptions opts;
  if (n_threads != 1) {
    opts.parallel = true;
    opts.num_threads = n_threads;
  }

  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    m.ParseFromString(serialized, opts);
    total_tensors += m.ref_graph().ref_initializer().size();
  }
  return total_tensors;
}

/**
 * Runs the file-load loop: saves the model to a temp file then loads it
 * *n_iters* times via FileStream.
 *
 * @param serialized  Serialized bytes to write once to the temp file.
 * @param n_iters     Number of load iterations to execute.
 * @param n_threads   Thread count (1 = sequential, >1 = parallel mode).
 * @return Total number of initializer tensors loaded across all iterations.
 */
static size_t run_load_file(const std::string &serialized, int n_iters, int n_threads) {
  // Write the model to a temp file once.
  const std::string tmp_path = "bench_load_tmp.onnx";
  {
    std::ofstream f(tmp_path, std::ios::binary);
    f.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    f.close();
    if (!f.good()) {
      std::cerr << "run_load_file: failed to write temp file '" << tmp_path << "'\n";
      return 0;
    }
  }

  ParseOptions opts;
  if (n_threads != 1) {
    opts.parallel = true;
    opts.num_threads = n_threads;
  }

  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    FileStream rstream(tmp_path);
    if (opts.parallel) {
      rstream.StartThreadPool(opts.num_threads);
    }
    m.ParseFromStream(rstream, opts);
    if (opts.parallel) {
      rstream.WaitForDelayedBlock();
    }
    total_tensors += m.ref_graph().ref_initializer().size();
  }

  std::remove(tmp_path.c_str());
  return total_tensors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  // Defaults
  int n_iters = 20;
  int n_threads = 1;
  int n_init = 40;
  int dim = 2048;

  // Minimal command-line parsing.
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

  std::cout << "bench_parse_serialize\n"
            << "  n_iters  = " << n_iters << "\n"
            << "  n_threads= " << n_threads << "\n"
            << "  n_init   = " << n_init << "\n"
            << "  dim      = " << dim << "\n";

  // Build the model once.
  ModelProto model = build_model(n_init, dim);

  // Serialize once to get the byte string for the parse benchmark.
  std::string serialized;
  {
    SerializeOptions opts;
    model.SerializeToString(serialized, opts);
  }
  double model_mb = static_cast<double>(serialized.size()) / (1024.0 * 1024.0);
  std::cout << "  model_mb = " << model_mb << " MB\n\n";

  // --- serialize benchmark ---
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t bytes_written = run_serialize(model, n_iters, n_threads);
  auto t1 = std::chrono::high_resolution_clock::now();
  double serialize_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // --- parse benchmark ---
  auto t2 = std::chrono::high_resolution_clock::now();
  size_t tensors_read = run_parse(serialized, n_iters, n_threads);
  auto t3 = std::chrono::high_resolution_clock::now();
  double parse_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  // --- file-load benchmark (FileStream) ---
  auto t4 = std::chrono::high_resolution_clock::now();
  size_t tensors_loaded = run_load_file(serialized, n_iters, n_threads);
  auto t5 = std::chrono::high_resolution_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

  // Print results so the linker cannot eliminate the loops.
  std::cout << "serialize: " << serialize_ms / n_iters << " ms/iter"
            << "  (total_bytes=" << bytes_written << ")\n";
  std::cout << "parse    : " << parse_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << tensors_read << ")\n";
  std::cout << "load/mmap: " << load_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << tensors_loaded << ")\n";
  return 0;
}
