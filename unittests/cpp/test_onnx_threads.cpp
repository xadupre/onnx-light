#include "onnx.h"
#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include "thread_pool.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace onnx;
using namespace onnx::utils;

TEST(onnx_threads, CreateAndDestroy) {
  ThreadPool pool;
  pool.Start(4);
  EXPECT_EQ(pool.GetThreadCount(), 4);
}

TEST(onnx_threads, SubmitSingleTask) {
  ThreadPool pool;
  pool.Start(2);
  int result = 0;
  auto task = [&result]() {
    for (size_t i = 0; i < 42; ++i) {
      result += 1;
    }
  };
  pool.SubmitTask(task);
  pool.Wait();
  EXPECT_EQ(result, 42);
}

TEST(onnx_threads, SubmitMultipleTasks) {
  ThreadPool pool;
  pool.Start(4);
  constexpr int num_tasks = 100;
  std::atomic<int> counter(0);
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
  }
  pool.Wait();
  EXPECT_EQ(counter.load(), num_tasks);
}

TEST(onnx_threads, ParallelExecution) {
  ThreadPool pool;
  pool.Start(8);

  std::atomic<int> counter(0);
  std::vector<int> thread_ids;
  std::mutex mutex;

  constexpr int num_tasks = 20;
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter, &thread_ids, &mutex]() {
      int thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
      {
        std::lock_guard<std::mutex> lock(mutex);
        thread_ids.push_back(thread_id);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      counter.fetch_add(1, std::memory_order_relaxed);
    });
  }

  pool.Wait();

  EXPECT_EQ(counter.load(), num_tasks);

  std::sort(thread_ids.begin(), thread_ids.end());
  auto unique_end = std::unique(thread_ids.begin(), thread_ids.end());
  int unique_threads = std::distance(thread_ids.begin(), unique_end);
  EXPECT_GT(unique_threads, 1);
}

TEST(onnx_threads, ParallelModelProcessing0) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto &graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto &tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 0;
    ModelProto model_proto2;
    stream.StartThreadPool(0);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(),
              model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}

TEST(onnx_threads, ParallelModelProcessing4_File) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto &graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto &tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    ModelProto model_proto2;
    stream.StartThreadPool(2);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(),
              model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}

TEST(onnx_threads, ParallelModelProcessing4_String) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto &graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto &tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string serialized;
  {
    SerializeOptions options;
    model.SerializeToString(serialized, options);
  }

  // reading
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    ModelProto model_proto2;
    model_proto2.ParseFromString(serialized, options);
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(),
              model_proto2.ref_graph().ref_initializer().size());
    for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
      const auto &tensor1 = model.ref_graph().ref_initializer()[i];
      const auto &tensor2 = model_proto2.ref_graph().ref_initializer()[i];
      EXPECT_EQ(tensor1.ref_raw_data(), tensor2.ref_raw_data());
      EXPECT_EQ(tensor1.ref_data_type(), tensor2.ref_data_type());
    }
  }
}

TEST(onnx_threads, ParallelModelProcessing4_FileExternalData) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto &weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  TensorProto &weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::string temp_filename = "test_tensor_file_stream_read.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.weight.tmp";

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  ModelProto model2;
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    utils::TwoFilesStream rstream(temp_filename, temp_weights);
    rstream.StartThreadPool(2);
    ParseProtoFromStream(model2, rstream, options);
    rstream.WaitForDelayedBlock();
  }

  EXPECT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data());
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_name().as_string(),
              model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}

TEST(onnx_threads, ParallelModelProcessing4_FileExternalDataManyInitializers) {
  ModelProto model;

  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");

  for (uint8_t i = 0; i < 100; ++i) {
    TensorProto &weights = graph.add_initializer();
    weights.set_name("weights");
    weights.set_data_type(TensorProto::DataType::FLOAT);
    weights.ref_dims().push_back(1);
    weights.ref_dims().push_back(1);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
  }

  NodeProto &node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto &attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto &biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);

  NodeProto &nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto &attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto &nested = attrg.add_g();

  for (uint8_t i = 0; i < 100; ++i) {
    TensorProto &weights2 = nested.add_initializer();
    weights2.set_name("weights2");
    weights2.set_data_type(TensorProto::DataType::FLOAT);
    weights2.ref_dims().push_back(1);
    weights2.ref_dims().push_back(1);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
  }

  NodeProto &node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto &attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto &biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);

  std::string temp_filename = "test_tensor_file_stream_read.big.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.weight.big.tmp";

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  int64_t length;
  {
    std::ifstream file(temp_weights, std::ios::binary | std::ios::ate);
    length = static_cast<int64_t>(file.tellg());
  }
  EXT_ENFORCE(length, 100 + 2 * 4);
  {
    std::ifstream file(temp_weights, std::ios::binary);
    std::vector<uint8_t> buffer(length);
    file.read(reinterpret_cast<char *>(buffer.data()), length);
    for (size_t i = 0; i < buffer.size(); ++i) {
      if (i < 4)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>(232)) << " at index " << i;
      else if (i < 8)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>(244)) << " at index " << i;
      else if (i < 408)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>((i - 8) / 4 + 105)) << " at index " << i;
      else
        EXPECT_EQ(buffer[i], static_cast<uint8_t>((i - 408) / 4)) << " at index " << i;
    }
  }

  ModelProto model2;
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    utils::TwoFilesStream rstream(temp_filename, temp_weights);
    rstream.StartThreadPool(2);
    ParseProtoFromStream(model2, rstream, options);
    rstream.WaitForDelayedBlock();
  }

  EXPECT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data());
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_name().as_string(),
              model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}

TEST(onnx_threads, TwoFilesStreamParallelReadDelayedBlocksOnWeights) {
  std::string temp_filename = "test_tensor_file_stream_read.delayed.main.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.delayed.weights.tmp";

  {
    std::ofstream file(temp_filename, std::ios::binary);
    file.put(0);
  }
  {
    std::ofstream file(temp_weights, std::ios::binary);
    for (size_t i = 0; i < 512; ++i) {
      file.put(static_cast<char>(i % 251));
    }
  }

  std::vector<uint8_t> block_data1(64), block_data2(64), block_data3(64);
  {
    utils::TwoFilesStream stream(temp_filename, temp_weights);
    stream.StartThreadPool(3);

    DelayedBlock block1;
    block1.size = block_data1.size();
    block1.data = block_data1.data();
    block1.offset = 7;
    block1.stream_id = 1;
    stream.ReadDelayedBlock(block1);

    DelayedBlock block2;
    block2.size = block_data2.size();
    block2.data = block_data2.data();
    block2.offset = 123;
    block2.stream_id = 1;
    stream.ReadDelayedBlock(block2);

    DelayedBlock block3;
    block3.size = block_data3.size();
    block3.data = block_data3.data();
    block3.offset = 301;
    block3.stream_id = 1;
    stream.ReadDelayedBlock(block3);

    stream.WaitForDelayedBlock();
  }

  for (size_t i = 0; i < block_data1.size(); ++i) {
    EXPECT_EQ(block_data1[i], static_cast<uint8_t>((7 + i) % 251)) << " at index " << i;
    EXPECT_EQ(block_data2[i], static_cast<uint8_t>((123 + i) % 251)) << " at index " << i;
    EXPECT_EQ(block_data3[i], static_cast<uint8_t>((301 + i) % 251)) << " at index " << i;
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}

// -----------------------------------------------------------------------
// Parallel external-data write tests
// -----------------------------------------------------------------------

// Verify that FileWriteStream::pre_allocate creates a file of the exact
// requested size, filled with zeroes at the last byte.
TEST(onnx_threads, FileWriteStreamPreAllocate) {
  std::string temp_file = "test_pre_allocate.tmp";
  const int64_t total_bytes = 1024;
  {
    utils::FileWriteStream stream(temp_file);
    stream.pre_allocate(total_bytes);
  }
  std::ifstream f(temp_file, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(f.is_open());
  EXPECT_EQ(static_cast<int64_t>(f.tellg()), total_bytes);
  std::remove(temp_file.c_str());
}

// Helper: build a model with a given number of float initializers each
// holding `tensor_floats` float values (i.e. tensor_floats * 4 bytes of
// raw_data).  The byte value of every element in tensor i is (i % 256).
static ModelProto MakeModelWithInitializers(int num_tensors, int tensor_floats) {
  ModelProto model;
  model.set_ir_version(7);
  GraphProto &graph = model.add_graph();
  graph.set_name("test_graph");
  for (int i = 0; i < num_tensors; ++i) {
    TensorProto &t = graph.add_initializer();
    t.set_name("w" + std::to_string(i));
    t.set_data_type(TensorProto::DataType::FLOAT);
    t.ref_dims().push_back(tensor_floats);
    std::vector<uint8_t> raw(tensor_floats * 4, static_cast<uint8_t>(i % 256));
    t.set_raw_data(raw);
  }
  return model;
}

// Verify that parallel external-data writing produces byte-for-byte
// identical weights file content to sequential writing.
TEST(onnx_threads, ParallelExternalWriteMatchesSequential) {
  const int num_tensors = 20;
  const int tensor_floats = 64; // 256 bytes per tensor, well above any threshold
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string seq_onnx = "test_par_ext_write_seq.onnx";
  std::string seq_data = "test_par_ext_write_seq.onnx.data";
  std::string par_onnx = "test_par_ext_write_par.onnx";
  std::string par_data = "test_par_ext_write_par.onnx.data";

  // Sequential write
  {
    utils::TwoFilesWriteStream wstream(seq_onnx, seq_data);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    SerializeProtoToStream(model, wstream, opts);
  }

  // Parallel write (4 threads)
  {
    utils::TwoFilesWriteStream wstream(par_onnx, par_data);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    opts.parallel = true;
    opts.num_threads = 4;
    SerializeProtoToStream(model, wstream, opts);
  }

  // Compare the weights files byte-by-byte
  std::ifstream f_seq(seq_data, std::ios::binary);
  std::ifstream f_par(par_data, std::ios::binary);
  ASSERT_TRUE(f_seq.is_open()) << "Sequential weights file not found: " << seq_data;
  ASSERT_TRUE(f_par.is_open()) << "Parallel weights file not found: " << par_data;

  std::vector<uint8_t> seq_bytes((std::istreambuf_iterator<char>(f_seq)),
                                 std::istreambuf_iterator<char>());
  std::vector<uint8_t> par_bytes((std::istreambuf_iterator<char>(f_par)),
                                 std::istreambuf_iterator<char>());

  ASSERT_EQ(seq_bytes.size(), par_bytes.size()) << "Weights file sizes differ";
  EXPECT_EQ(seq_bytes, par_bytes) << "Weights file contents differ";

  std::remove(seq_onnx.c_str());
  std::remove(seq_data.c_str());
  std::remove(par_onnx.c_str());
  std::remove(par_data.c_str());
}

// Same as above but using num_threads = -1 (auto: one thread per core).
TEST(onnx_threads, ParallelExternalWriteAutoThreadsMatchesSequential) {
  const int num_tensors = 16;
  const int tensor_floats = 32; // 128 bytes per tensor
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string seq_onnx = "test_par_ext_write_auto_seq.onnx";
  std::string seq_data = "test_par_ext_write_auto_seq.onnx.data";
  std::string par_onnx = "test_par_ext_write_auto_par.onnx";
  std::string par_data = "test_par_ext_write_auto_par.onnx.data";

  // Sequential write
  {
    utils::TwoFilesWriteStream wstream(seq_onnx, seq_data);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    SerializeProtoToStream(model, wstream, opts);
  }

  // Parallel write (-1 = auto threads)
  {
    utils::TwoFilesWriteStream wstream(par_onnx, par_data);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    opts.parallel = true;
    opts.num_threads = -1;
    SerializeProtoToStream(model, wstream, opts);
  }

  std::ifstream f_seq(seq_data, std::ios::binary);
  std::ifstream f_par(par_data, std::ios::binary);
  ASSERT_TRUE(f_seq.is_open());
  ASSERT_TRUE(f_par.is_open());

  std::vector<uint8_t> seq_bytes((std::istreambuf_iterator<char>(f_seq)),
                                 std::istreambuf_iterator<char>());
  std::vector<uint8_t> par_bytes((std::istreambuf_iterator<char>(f_par)),
                                 std::istreambuf_iterator<char>());

  ASSERT_EQ(seq_bytes.size(), par_bytes.size());
  EXPECT_EQ(seq_bytes, par_bytes);

  std::remove(seq_onnx.c_str());
  std::remove(seq_data.c_str());
  std::remove(par_onnx.c_str());
  std::remove(par_data.c_str());
}

// Verify that a model saved with parallel external writes can be loaded
// back and contains the same initializer data.
TEST(onnx_threads, ParallelExternalWriteRoundTrip) {
  const int num_tensors = 10;
  const int tensor_floats = 16; // 64 bytes per tensor
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string onnx_path = "test_par_ext_write_rt.onnx";
  std::string data_path = "test_par_ext_write_rt.onnx.data";

  // Write in parallel
  {
    utils::TwoFilesWriteStream wstream(onnx_path, data_path);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    opts.parallel = true;
    opts.num_threads = 2;
    SerializeProtoToStream(model, wstream, opts);
  }

  // Read back
  ModelProto model2;
  {
    utils::TwoFilesStream rstream(onnx_path, data_path);
    ParseOptions popts;
    ParseProtoFromStream(model2, rstream, popts);
  }

  ASSERT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data())
        << "Mismatch at initializer " << i;
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_name().as_string(),
              model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(onnx_path.c_str());
  std::remove(data_path.c_str());
}

TEST(onnx_threads, ParallelExternalWriteCanRestartOnSameStream) {
  std::string onnx_path = "test_par_ext_restart.onnx";
  std::string data_path = "test_par_ext_restart.onnx.data";

  utils::TwoFilesWriteStream wstream(onnx_path, data_path);

  std::vector<uint8_t> first = {1, 2, 3, 4};
  std::vector<uint8_t> second = {5, 6, 7, 8};

  wstream.pre_allocate_weights(static_cast<int64_t>(first.size()));
  wstream.StartWriteThreadPool(2);
  wstream.write_raw_bytes_in_second_stream(first.data(), static_cast<int64_t>(first.size()));
  wstream.WaitForWriteCompletion();

  wstream.pre_allocate_weights(static_cast<int64_t>(second.size()));
  wstream.StartWriteThreadPool(2);
  wstream.write_raw_bytes_in_second_stream(second.data(), static_cast<int64_t>(second.size()));
  wstream.WaitForWriteCompletion();

  std::ifstream f_data(data_path, std::ios::binary);
  ASSERT_TRUE(f_data.is_open());
  std::vector<uint8_t> content((std::istreambuf_iterator<char>(f_data)),
                               std::istreambuf_iterator<char>());
  ASSERT_EQ(content.size(), second.size());
  EXPECT_EQ(content, second);

  std::remove(onnx_path.c_str());
  std::remove(data_path.c_str());
}

// -----------------------------------------------------------------------
// ParseModelProtoFromStream thread-pool ownership tests
// -----------------------------------------------------------------------

// Verify that ParseProtoFromStream starts and stops the thread pool
// internally when options.parallel=true, without the caller having to call
// StartThreadPool/WaitForDelayedBlock manually.
TEST(onnx_threads, ParseModelProtoHandlesThreadPoolInternally_File) {
  const int num_tensors = 16;
  const int tensor_floats = 16; // 64 bytes per tensor
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string onnx_path = "test_parse_internal_tp.onnx";
  {
    utils::FileWriteStream wstream(onnx_path);
    SerializeOptions opts;
    model.SerializeToStream(wstream, opts);
  }

  // Parse WITHOUT manually calling StartThreadPool / WaitForDelayedBlock.
  // ParseProtoFromStream (via ParseModelProtoFromStream) must handle these.
  ModelProto model2;
  {
    utils::FileStream rstream(onnx_path);
    ParseOptions opts;
    opts.parallel = true;
    opts.num_threads = 2;
    ParseProtoFromStream(model2, rstream, opts);
  }

  ASSERT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data())
        << "Mismatch at initializer " << i;
  }

  std::remove(onnx_path.c_str());
}

// Same test but using an already-started thread pool on the stream.
// ParseModelProtoFromStream must detect that it is already started and
// skip the second Start() call (which would throw).
TEST(onnx_threads, ParseModelProtoSkipsStartIfAlreadyStarted_File) {
  const int num_tensors = 8;
  const int tensor_floats = 16;
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string onnx_path = "test_parse_skip_start.onnx";
  {
    utils::FileWriteStream wstream(onnx_path);
    SerializeOptions opts;
    model.SerializeToStream(wstream, opts);
  }

  ModelProto model2;
  {
    utils::FileStream rstream(onnx_path);
    ParseOptions opts;
    opts.parallel = true;
    opts.num_threads = 2;
    rstream.StartThreadPool(2); // caller starts manually
    ParseProtoFromStream(model2, rstream, opts);
    // WaitForDelayedBlock already called inside ParseProtoFromStream;
    // calling it again must be safe (idempotent).
    rstream.WaitForDelayedBlock();
  }

  ASSERT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data())
        << "Mismatch at initializer " << i;
  }

  std::remove(onnx_path.c_str());
}

// Verify that ParseProtoFromStream handles the thread pool internally for
// external-data (TwoFilesStream) models as well.
TEST(onnx_threads, ParseModelProtoHandlesThreadPoolInternally_TwoFiles) {
  const int num_tensors = 10;
  const int tensor_floats = 16;
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  std::string onnx_path = "test_parse_internal_tp_ext.onnx";
  std::string data_path = "test_parse_internal_tp_ext.data";
  {
    utils::TwoFilesWriteStream wstream(onnx_path, data_path);
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    SerializeProtoToStream(model, wstream, opts);
  }

  ModelProto model2;
  {
    utils::TwoFilesStream rstream(onnx_path, data_path);
    ParseOptions opts;
    opts.parallel = true;
    opts.num_threads = 2;
    // No manual StartThreadPool / WaitForDelayedBlock needed.
    ParseProtoFromStream(model2, rstream, opts);
  }

  ASSERT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data())
        << "Mismatch at initializer " << i;
  }

  std::remove(onnx_path.c_str());
  std::remove(data_path.c_str());
}

// -----------------------------------------------------------------------
// Parallel SerializeToString tests
// -----------------------------------------------------------------------

// Verify that parallel SerializeToString produces byte-for-byte identical
// output to sequential SerializeToString.
TEST(onnx_threads, ParallelSerializeToStringMatchesSequential) {
  const int num_tensors = 20;
  const int tensor_floats = 64; // 256 bytes per tensor
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  // Sequential
  std::string seq_out;
  {
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    model.SerializeToString(seq_out, opts);
  }

  // Parallel (4 threads)
  std::string par_out;
  {
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    opts.parallel = true;
    opts.num_threads = 4;
    model.SerializeToString(par_out, opts);
  }

  ASSERT_EQ(seq_out.size(), par_out.size()) << "Output sizes differ";
  EXPECT_EQ(seq_out, par_out) << "Output contents differ";
}

// Verify that a model serialized to string with parallel=true can be parsed
// back and contains the same initializer data.
TEST(onnx_threads, ParallelSerializeToStringRoundTrip) {
  const int num_tensors = 10;
  const int tensor_floats = 16; // 64 bytes per tensor
  ModelProto model = MakeModelWithInitializers(num_tensors, tensor_floats);

  // Serialize with parallel
  std::string serialized;
  {
    SerializeOptions opts;
    opts.raw_data_threshold = 4;
    opts.parallel = true;
    opts.num_threads = 2;
    model.SerializeToString(serialized, opts);
  }

  // Parse back
  ModelProto model2;
  {
    ParseOptions popts;
    model2.ParseFromString(serialized, popts);
  }

  ASSERT_EQ(model.ref_graph().ref_initializer().size(),
            model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_raw_data(),
              model2.ref_graph().ref_initializer()[i].ref_raw_data())
        << "Mismatch at initializer " << i;
    EXPECT_EQ(model.ref_graph().ref_initializer()[i].ref_name().as_string(),
              model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }
}
