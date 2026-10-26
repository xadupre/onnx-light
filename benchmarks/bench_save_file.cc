/**
 * bench_save_file.cc
 *
 * Standalone C++ benchmark focused on the file-save path (TwoFilesWriteStream).
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux
 * profiling tools (perf, gprof, valgrind/callgrind) can attribute wall-clock
 * or instruction samples back to named C++ functions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "onnx_io_policy.h"
#include "stream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::utils;

/**
 * Builds a synthetic ModelProto with *n_init* float tensors of shape [dim, dim].
 *
 * @param n_init Number of initializer tensors to add to the graph.
 * @param dim Side length of each square float weight matrix.
 * @return A fully populated ModelProto ready for serialization.
 */
static ModelProto build_model(int n_init, int dim) {
  ModelProto model;
  model.set_ir_version(9);
  model.set_producer_name("bench_save_file");

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

/** Builds a mixed-size LLM-like payload, not a reconstruction of a pretrained model. */
static ModelProto build_llm_model(int n_layers) {
  ModelProto model;
  model.set_ir_version(9);
  GraphProto &graph = *model.add_graph();
  graph.set_name("llm_initializer_sizes");
  const int64_t mib = 1024 * 1024;
  const std::vector<int64_t> sizes = {4096,    4096,     mib,      2 * mib, 4 * mib,
                                      8 * mib, 12 * mib, 12 * mib, 16 * mib};
  for (int layer = 0; layer < n_layers; ++layer) {
    for (int64_t size : sizes) {
      TensorProto &tensor = *graph.add_initializer();
      tensor.set_name("W" + std::to_string(graph.ref_initializer().size()));
      tensor.set_data_type(TensorProto::DataType::FLOAT);
      tensor.add_dims(size / sizeof(float));
      tensor.set_raw_data(
          std::vector<uint8_t>(static_cast<size_t>(size), static_cast<uint8_t>(layer + 1)));
    }
  }
  TensorProto &embedding = *graph.add_initializer();
  embedding.set_name("embedding");
  embedding.set_data_type(TensorProto::DataType::FLOAT);
  embedding.add_dims(64 * mib / sizeof(float));
  embedding.set_raw_data(std::vector<uint8_t>(64 * mib, 42));
  return model;
}

/**
 * Runs the external-data save loop using TwoFilesWriteStream.
 *
 * @param model ModelProto to serialize.
 * @param n_iters Number of save iterations to execute.
 * @param n_threads Thread count (1 = sequential, >1 = explicit, -1 = automatic).
 * @param min_block_size Minimum parallel block size in bytes (0 = automatic).
 * @param total_save_ms Accumulates save time, excluding output removal.
 * @return Total number of bytes written across all iterations.
 */
static size_t run_save_file(ModelProto &model, int n_iters, int n_threads, int64_t min_block_size,
                            double &total_save_ms) {
  const std::string tmp_path = "bench_save_tmp.onnx";
  const std::string tmp_data_path = "bench_save_tmp.onnx.data";

  SerializeOptions opts;
  opts.num_threads = n_threads;
  opts.min_parallel_block_size = min_block_size;
  IOPolicyTrace trace;
  opts.io_trace = &trace;

  size_t total_bytes = 0;
  for (int i = 0; i < n_iters; ++i) {
    trace = {};
    auto t0 = std::chrono::steady_clock::now();
    {
      TwoFilesWriteStream wstream(tmp_path, tmp_data_path);
      SerializeModelProtoToStream(model, wstream, opts);
    }
    auto t1 = std::chrono::steady_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_save_ms += save_ms;
    std::cout << "save_ms=" << save_ms << " workers=" << trace.resolved_workers
              << " min_block_size=" << trace.resolved_min_block_size
              << " submitted_bytes=" << trace.bytes_in_flight << "\n";
    total_bytes += static_cast<size_t>(std::filesystem::file_size(tmp_path));
    total_bytes += static_cast<size_t>(std::filesystem::file_size(tmp_data_path));
    std::remove(tmp_path.c_str());
    std::remove(tmp_data_path.c_str());
  }

  return total_bytes;
}

int main(int argc, char *argv[]) {
  int n_iters = 20;
  int n_threads = 1;
  int n_init = 40;
  int dim = 2048;
  int64_t min_block_size = 0;
  std::string profile = "uniform";

  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0) {
      n_iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-t") == 0) {
      n_threads = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-i") == 0) {
      n_init = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-d") == 0) {
      dim = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-b") == 0) {
      min_block_size = std::strtoll(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "-p") == 0) {
      profile = argv[++i];
    }
  }

  if (n_iters <= 0 || n_init <= 0 || dim <= 0 || min_block_size < 0 ||
      (profile != "uniform" && profile != "llm")) {
    std::cerr << "Usage: bench_save_file -n iterations -t threads -i tensors_or_layers"
                 " -d dimension -b min_block_bytes -p uniform|llm\n";
    return 1;
  }

  std::cout << "bench_save_file\n"
            << "  n_iters  = " << n_iters << "\n"
            << "  n_threads= " << n_threads << "\n"
            << "  n_init   = " << n_init << "\n"
            << "  dim      = " << dim << "\n"
            << "  profile  = " << profile << "\n"
            << "  min_block_size= " << min_block_size << "\n";

  ModelProto model = profile == "llm" ? build_llm_model(n_init) : build_model(n_init, dim);
  double model_mb = static_cast<double>(model.ByteSizeLong()) / (1024.0 * 1024.0);
  std::cout << "  model_mb = " << model_mb << " MB\n\n";

  double save_ms = 0;
  size_t bytes_written = run_save_file(model, n_iters, n_threads, min_block_size, save_ms);

  std::cout << "save/2filex1: " << save_ms / n_iters << " ms/iter"
            << "  (total_bytes=" << bytes_written << ")\n";
  return 0;
}
