/**
 * bench_load_file.cc
 *
 * Standalone C++ benchmark focused on the file-load path (FileStream).
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux
 * profiling tools (perf, gprof, valgrind/callgrind) can attribute wall-clock
 * or instruction samples back to named C++ functions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#ifdef BENCH_HAS_UPSTREAM_ONNX
#include "onnx/onnx_pb.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

static ModelProto build_model(int n_init, int dim) {
  ModelProto model;
  model.set_ir_version(9);
  model.set_producer_name("bench_load_file");

  GraphProto &graph = *model.add_graph();
  graph.set_name("bench_graph");

  const size_t n_floats = static_cast<size_t>(dim) * static_cast<size_t>(dim);
  const size_t n_bytes = n_floats * sizeof(float);

  std::vector<uint8_t> raw(n_bytes);
  for (size_t i = 0; i < n_bytes; ++i) {
    raw[i] = static_cast<uint8_t>((i * 2654435761ULL) >> 24);
  }

  for (int i = 0; i < n_init; ++i) {
    TensorProto &tensor = *graph.add_initializer();
    tensor.set_name("W" + std::to_string(i));
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.add_dims(static_cast<int64_t>(dim));
    tensor.add_dims(static_cast<int64_t>(dim));
    tensor.set_raw_data(raw);
  }

  return model;
}

static size_t run_load_file(const std::string &file_path, int n_iters, int n_threads) {
  ParseOptions opts;
  if (n_threads != 1) {
    opts.num_threads = n_threads;
  }

  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    FileStream rstream(file_path);
    if (opts.is_parallel()) {
      rstream.StartThreadPool(opts.num_threads);
    }
    m.ParseFromStream(rstream, opts);
    if (opts.is_parallel()) {
      rstream.WaitForDelayedBlock();
    }
    total_tensors += m.ref_graph().ref_initializer().size();
  }

  return total_tensors;
}

#ifdef BENCH_HAS_UPSTREAM_ONNX
/**
 * Runs the upstream onnx (protobuf-based) file-load loop. Included in the
 * same binary as the onnx_light loader so that a single profiling run
 * (perf, gprof, valgrind/callgrind) attributes samples to both loaders and
 * the two stacks can be compared side by side.
 */
static size_t run_load_file_onnx(const std::string &file_path, int n_iters) {
  size_t total_tensors = 0;
  for (int i = 0; i < n_iters; ++i) {
    std::ifstream input(file_path, std::ios::binary);
    onnx::ModelProto m;
    m.ParseFromIstream(&input);
    total_tensors += static_cast<size_t>(m.graph().initializer_size());
  }
  return total_tensors;
}
#endif

int main(int argc, char *argv[]) {
  int n_iters = 10;
  int n_threads = 1;
  int n_init = 40;
  int dim = 2048;
  std::string input_file;

  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0) {
      n_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-t") == 0) {
      n_threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-i") == 0) {
      n_init = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-d") == 0) {
      dim = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-f") == 0) {
      input_file = argv[++i];
    }
  }

  std::cout << "bench_load_file\n"
            << "  n_iters  = " << n_iters << "\n"
            << "  n_threads= " << n_threads << "\n";

  std::string tmp_path;
  std::string file_to_load;
  double model_mb = 0.0;

  if (!input_file.empty()) {
    std::cout << "  input_file=" << input_file << "\n";
    file_to_load = input_file;

    std::ifstream f(input_file, std::ios::binary | std::ios::ate);
    if (f.good()) {
      std::streamsize size = f.tellg();
      model_mb = static_cast<double>(size) / (1024.0 * 1024.0);
    }
  } else {
    std::cout << "  n_init   = " << n_init << "\n"
              << "  dim      = " << dim << "\n";

    ModelProto model = build_model(n_init, dim);
    std::string serialized;
    {
      SerializeOptions opts;
      model.SerializeToString(serialized, opts);
    }
    model_mb = static_cast<double>(serialized.size()) / (1024.0 * 1024.0);

    tmp_path = "bench_load_tmp.onnx";
    std::ofstream f(tmp_path, std::ios::binary);
    f.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    f.close();
    if (!f.good()) {
      std::cerr << "main: failed to write temp file '" << tmp_path << "'\n";
      return 1;
    }
    file_to_load = tmp_path;
  }

  std::cout << "  model_mb = " << model_mb << " MB\n\n";

  auto t0 = std::chrono::high_resolution_clock::now();
  size_t tensors_loaded = run_load_file(file_to_load, n_iters, n_threads);
  auto t1 = std::chrono::high_resolution_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "onnx_light load/mmap: " << load_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << tensors_loaded << ")\n";

#ifdef BENCH_HAS_UPSTREAM_ONNX
  auto t2 = std::chrono::high_resolution_clock::now();
  size_t tensors_loaded_onnx = run_load_file_onnx(file_to_load, n_iters);
  auto t3 = std::chrono::high_resolution_clock::now();
  double load_ms_onnx = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "onnx       load:      " << load_ms_onnx / n_iters << " ms/iter"
            << "  (total_tensors=" << tensors_loaded_onnx << ")\n";
#endif

  if (!tmp_path.empty()) {
    std::remove(tmp_path.c_str());
  }

  return 0;
}
