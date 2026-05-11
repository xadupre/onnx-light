/**
 * bench_load_file.cpp
 *
 * Standalone C++ benchmark focused on the file-load path (MmapStream).
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux
 * profiling tools (perf, gprof, valgrind/callgrind) can attribute wall-clock
 * or instruction samples back to named C++ functions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

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

  GraphProto &graph = model.add_graph();
  graph.set_name("bench_graph");

  const size_t n_floats = static_cast<size_t>(dim) * static_cast<size_t>(dim);
  const size_t n_bytes = n_floats * sizeof(float);

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

static size_t run_load_file(const std::string &serialized, int n_iters, int n_threads) {
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
    MmapStream rstream(tmp_path);
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

  std::cout << "bench_load_file\n"
            << "  n_iters  = " << n_iters << "\n"
            << "  n_threads= " << n_threads << "\n"
            << "  n_init   = " << n_init << "\n"
            << "  dim      = " << dim << "\n";

  ModelProto model = build_model(n_init, dim);
  std::string serialized;
  {
    SerializeOptions opts;
    model.SerializeToString(serialized, opts);
  }
  double model_mb = static_cast<double>(serialized.size()) / (1024.0 * 1024.0);
  std::cout << "  model_mb = " << model_mb << " MB\n\n";

  auto t0 = std::chrono::high_resolution_clock::now();
  size_t tensors_loaded = run_load_file(serialized, n_iters, n_threads);
  auto t1 = std::chrono::high_resolution_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "load/mmap: " << load_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << tensors_loaded << ")\n";
  return 0;
}
