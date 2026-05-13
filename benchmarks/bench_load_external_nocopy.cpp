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
 * **Fair comparison note**:
 * ``mmap(MAP_PRIVATE)`` only creates virtual-address mappings; individual pages are
 * not loaded from disk until first access (demand paging).  The copy path reads all
 * tensor bytes eagerly inside ``ParseFromStream``, while the no-copy path would
 * appear artificially fast if we only counted tensors.  Both loops therefore call
 * ``touch_all_raw_data()`` after parsing to force every mmap page fault to complete
 * before the timer stops.  This ensures both timings represent the cost of having
 * all tensor bytes resident in CPU-accessible memory.
 *
 * The genuine advantage of the no-copy path is that it avoids:
 *   - one ``malloc`` per tensor (or one large ``malloc`` for the whole file), and
 *   - one ``memcpy`` from the kernel page cache into that user-space buffer.
 * Page faults under mmap are served by the same kernel path as the ``read()``-based
 * approach but skip the extra copy step.
 *
 * Usage:
 *   bench_load_external_nocopy [-n iters] [-t threads] [-i n_init] [-d dim]
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <chrono>
#include <cstdint>
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

/** Size of a virtual-memory page; used to stride through tensor bytes when
 *  forcing page faults in ``touch_all_raw_data()``. */
static constexpr size_t kPageSize = 4096;

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
 * Touches every page of every tensor's raw_data to force mmap page faults.
 *
 * Reads one byte from each kPageSize-aligned offset plus the last byte of each
 * tensor, ensuring all virtual-memory pages are populated before the caller
 * stops the timer.  Returns a checksum to prevent dead-code elimination.
 *
 * @param m ModelProto whose initializer raw_data should be touched.
 * @return Simple byte checksum across all sampled bytes.
 */
static uint64_t touch_all_raw_data(const ModelProto &m) {
  uint64_t checksum = 0;
  const auto &inits = m.ref_graph().ref_initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const ByteSpan &rd = inits[i].ref_raw_data();
    const uint8_t *data = rd.data();
    const size_t n = rd.size();
    for (size_t j = 0; j < n; j += kPageSize) {
      checksum += data[j];
    }
    if (n > 0) {
      checksum += data[n - 1]; // ensure the last partial page is faulted in
    }
  }
  return checksum;
}

/**
 * Runs the 2-file no-copy load loop.
 *
 * Each iteration opens the two-file stream, parses the model so that every
 * tensor borrows a direct pointer into the memory-mapped external-data file,
 * and then touches all tensor pages to force page faults within the timed
 * window.  No per-tensor malloc+memcpy is performed.
 *
 * @param n_iters Number of load iterations.
 * @return Pair of (total tensor count, byte checksum) across all iterations.
 */
static std::pair<size_t, uint64_t> run_load_external_nocopy(int n_iters) {
  ParseOptions opts;
  opts.no_copy = true;

  size_t total_tensors = 0;
  uint64_t checksum = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    TwoFilesStream rstream(kOnnxFile, kDataFile);
    m.ParseFromStream(rstream, opts);
    total_tensors += m.ref_graph().ref_initializer().size();
    checksum += touch_all_raw_data(m);
  }
  return {total_tensors, checksum};
}

/**
 * Runs the 2-file copy (standard) load loop for comparison.
 *
 * Each iteration allocates one owned buffer per tensor and reads the bytes
 * directly into it.  A touch pass is performed identically to the no-copy
 * loop so both measurements represent the same amount of work.
 *
 * @param n_iters Number of load iterations.
 * @return Pair of (total tensor count, byte checksum) across all iterations.
 */
static std::pair<size_t, uint64_t> run_load_external_copy(int n_iters) {
  ParseOptions opts;
  // no_copy stays false

  size_t total_tensors = 0;
  uint64_t checksum = 0;
  for (int i = 0; i < n_iters; ++i) {
    ModelProto m;
    TwoFilesStream rstream(kOnnxFile, kDataFile);
    m.ParseFromStream(rstream, opts);
    total_tensors += m.ref_graph().ref_initializer().size();
    checksum += touch_all_raw_data(m);
  }
  return {total_tensors, checksum};
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

  // --- no-copy (mmap + touch) ---
  auto t0 = std::chrono::high_resolution_clock::now();
  auto [nc_tensors, nc_checksum] = run_load_external_nocopy(n_iters);
  auto t1 = std::chrono::high_resolution_clock::now();
  const double nc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // --- standard copy + touch ---
  auto t2 = std::chrono::high_resolution_clock::now();
  auto [cp_tensors, cp_checksum] = run_load_external_copy(n_iters);
  auto t3 = std::chrono::high_resolution_clock::now();
  const double cp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "load/2filex1/onnxlight-nocopy : " << nc_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << nc_tensors << ", checksum=" << nc_checksum << ")\n"
            << "load/2filex1/onnxlight        : " << cp_ms / n_iters << " ms/iter"
            << "  (total_tensors=" << cp_tensors << ", checksum=" << cp_checksum << ")\n";

  std::remove(kOnnxFile.c_str());
  std::remove(kDataFile.c_str());
  return 0;
}
