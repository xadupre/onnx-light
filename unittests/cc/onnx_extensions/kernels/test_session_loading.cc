// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/generation.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/runtime_session_loader.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

class LoadingPreparedIdentity : public KernelBase {
public:
  LoadingPreparedIdentity(const NodeProto &node, std::shared_ptr<size_t> executions)
      : KernelBase(KernelContext{}), executions_(std::move(executions)) {
    set_node(node);
  }

  bool HasPreparations(const std::unordered_set<std::string> &) const override { return true; }

  void DeclarePreparations(const std::unordered_set<std::string> &,
                           std::vector<KernelPreparation> &preparations) override {
    KernelPreparation preparation;
    if (node_->input_size() != 0)
      preparation.info.source_inputs = {node_->input(0)};
    preparation.estimate_memory = [](const RuntimeContext &) { return KernelPreparationMemory{}; };
    preparation.execute = [this](RuntimeContext &) {
      ++*executions_;
      ready_ = true;
    };
    preparations.push_back(std::move(preparation));
  }

  void Run(RuntimeContext &rt) override {
    if (!ready_)
      throw std::logic_error("Preparation consumer ran before publication.");
    rt.Put(node_->output(0), node_->input_size() == 0 ? Tensor::FromFloat("Y0", {}, {1})
                                                      : rt.Get(node_->input(0)).BorrowView());
  }

private:
  std::shared_ptr<size_t> executions_;
  bool ready_ = false;
};

class RuntimeSessionLoading : public testing::Test {
protected:
  void SetUp() override {
    onnx_kernels::RegisterKernelFunctions();
    static std::atomic<size_t> sequence{0};
    directory = std::filesystem::current_path() /
                ("session-loading-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(sequence++));
    std::filesystem::create_directory(directory);
    WriteModel();
  }
  void TearDown() override { std::filesystem::remove_all(directory); }

  void WriteModel() {
    ModelProto model;
    model.set_ir_version(9);
    model.add_opset_import()->set_version(13);
    auto *graph = model.mutable_graph();
    graph->add_input()->set_name("A");
    for (size_t i = 0; i < 2; ++i) {
      const std::string suffix = std::to_string(i);
      graph->add_output()->set_name("Y" + suffix);
      auto *weight = graph->add_initializer();
      weight->set_name("B" + suffix);
      weight->set_data_type(TensorProto::FLOAT);
      weight->add_dims(3);
      weight->add_dims(2);
      weight->set_data_location(TensorProto::EXTERNAL);
      auto *location = weight->add_external_data();
      location->set_key("location");
      location->set_value("weight" + suffix + ".bin");
      auto *length = weight->add_external_data();
      length->set_key("length");
      length->set_value("24");
      const std::vector<float> values{1, 2, 3, 4, 5, 6};
      std::ofstream payload(directory / ("weight" + suffix + ".bin"), std::ios::binary);
      payload.write(reinterpret_cast<const char *>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
      auto *node = graph->add_node();
      node->set_op_type("Gemm");
      node->add_input("A");
      node->add_input("B" + suffix);
      node->add_output("Y" + suffix);
    }
    std::ofstream output(directory / "model.onnx", std::ios::binary);
    const auto bytes = model.SerializeAsString();
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  std::vector<float> Run(RuntimeSession &session, RuntimeContext &rt) {
    rt.Put("A", Tensor::FromFloat("A", {1, 3}, {1, 2, 3}));
    session.Run(rt);
    const float *first = rt.Get("Y0").AsFloat();
    const float *second = rt.Get("Y1").AsFloat();
    return {first[0], first[1], second[0], second[1]};
  }

  std::string Path() const { return (directory / "model.onnx").string(); }
  ModelProto ReadModel() const {
    ModelProto model;
    utils::FileStream stream(Path());
    ParseOptions options;
    if (!model.ParseFromStream(stream, options))
      throw std::runtime_error("Cannot read loader fixture.");
    return model;
  }
  void SaveModel(const ModelProto &model) const {
    std::ofstream output(Path(), std::ios::binary);
    const auto bytes = model.SerializeAsString();
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  PipelineProto MakePipeline(bool embedded = true) const {
    PipelineProto pipeline;
    pipeline.set_max_new_tokens(2);
    pipeline.set_logits_name(embedded ? "Y0" : "A");
    ModelProto token_model;
    token_model.set_ir_version(9);
    token_model.add_opset_import()->set_version(13);
    auto *graph = token_model.mutable_graph();
    auto *input = graph->add_input();
    input->set_name("input_ids");
    input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    auto *output = graph->add_output();
    output->set_name("A");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    auto *axis = graph->add_initializer();
    axis->set_name("embedding_axis");
    axis->set_data_type(TensorProto::INT64);
    axis->add_dims(1);
    axis->add_int64_data(1);
    auto *weight = graph->add_initializer();
    weight->set_name("E");
    weight->set_data_type(TensorProto::FLOAT);
    weight->add_dims(1);
    weight->add_dims(3);
    weight->set_data_location(TensorProto::EXTERNAL);
    auto *location = weight->add_external_data();
    location->set_key("location");
    location->set_value("embedding.bin");
    auto *length = weight->add_external_data();
    length->set_key("length");
    length->set_value("12");
    const std::vector<float> values{1, 2, 3};
    std::ofstream payload(directory / "embedding.bin", std::ios::binary);
    payload.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    auto *cast = graph->add_node();
    cast->set_op_type("Cast");
    cast->add_input("input_ids");
    cast->add_output("embedding_float_ids");
    auto *to = cast->add_attribute();
    to->set_name("to");
    to->set_type(AttributeProto::INT);
    to->set_i(TensorProto::FLOAT);
    auto *sum = graph->add_node();
    sum->set_op_type("ReduceSum");
    sum->add_input("embedding_float_ids");
    sum->add_input("embedding_axis");
    sum->add_output("embedding_sum");
    auto *gemm = graph->add_node();
    gemm->set_op_type("Gemm");
    gemm->add_input("embedding_sum");
    gemm->add_input("E");
    gemm->add_output("A");
    if (embedded) {
      *pipeline.mutable_embedding() = std::move(token_model);
      *pipeline.mutable_prefill() = ReadModel();
      auto *prefill = pipeline.mutable_prefill()->mutable_graph();
      prefill->mutable_input(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
          TensorProto::FLOAT);
      for (auto &value : prefill->ref_output())
        value.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    } else {
      *pipeline.mutable_prefill() = std::move(token_model);
    }
    return pipeline;
  }

  std::string PipelinePath() const { return (directory / "pipeline.onnx").string(); }

  void SavePipeline(const PipelineProto &pipeline) const {
    std::ofstream output(PipelinePath(), std::ios::binary);
    const auto bytes = pipeline.SerializeAsString();
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  std::filesystem::path directory;
};

TEST_F(RuntimeSessionLoading, PipelineSerialAndOverlapGenerateFromOwnedStages) {
  auto pipeline = MakePipeline();
  SavePipeline(pipeline);
  RuntimeSessionOptions session_options;
  session_options.parameters.num_threads = 1;
  session_options.value_store = std::make_shared<ValueStore>();
  RuntimeLoadOptions serial_options;
  serial_options.overlap_preparation = false;
  RuntimeContext serial_rt;
  LoadedRuntimePipeline serial(PipelinePath(), serial_rt, session_options, serial_options);
  RuntimeContext overlap_rt;
  size_t completions = 0;
  RuntimeLoadOptions overlap_options;
  overlap_options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kComplete) {
      ++completions;
      EXPECT_FALSE(overlap_rt.Has("E"));
      EXPECT_FALSE(overlap_rt.Has("B0"));
      EXPECT_FALSE(overlap_rt.Has("B1"));
    }
  };
  LoadedRuntimePipeline overlap(PipelinePath(), overlap_rt, {}, overlap_options);
  EXPECT_EQ(completions, 1u);
  RuntimeValueMap feeds;
  feeds.emplace("input_ids", Tensor::From<int64_t>("input_ids", {1, 1}, {1}));
  LoadExternalDataForModel(*pipeline.mutable_embedding(), directory.string());
  LoadExternalDataForModel(*pipeline.mutable_prefill(), directory.string());
  RuntimeSession explicit_session(pipeline.prefill());
  auto executor = explicit_session.cpu_executor();
  RuntimeContext explicit_rt(RuntimeContextOptions{.cpu_executor = executor.get()});
  auto expected = Generate(pipeline, explicit_rt, feeds);
  EXPECT_EQ(
      (std::vector<int64_t>{expected.AsInt64(), expected.AsInt64() + expected.element_count()}),
      (std::vector<int64_t>{1, 1, 1}));
  std::filesystem::remove_all(directory);
  for (auto *loaded : {&serial, &overlap}) {
    RuntimeContext &rt = loaded == &serial ? serial_rt : overlap_rt;
    const size_t prepared = loaded->session_options().value_store->prepared_bytes();
    const size_t slots = loaded->session_options().value_store->slot_count();
    EXPECT_GT(prepared, 0u);
    EXPECT_GT(slots, 0u);
    for (size_t request = 0; request < 2; ++request) {
      auto actual = Generate(*loaded, rt, feeds);
      ASSERT_EQ(actual.shape, expected.shape);
      EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), expected.size_bytes()), 0);
    }
    GenerationOptions options;
    options.max_new_tokens = 1;
    options.logits_name = "Y0";
    EXPECT_EQ(Generate(*loaded, rt, feeds, options).shape, (Shape{1, 2}));
    EXPECT_EQ(rt.cpu_executor(), nullptr);
    options.on_step = [](const GenerationStep &) -> bool {
      throw std::runtime_error("observer failure");
    };
    EXPECT_THROW(Generate(*loaded, rt, feeds, options), std::runtime_error);
    EXPECT_EQ(rt.cpu_executor(), nullptr);
    EXPECT_EQ(loaded->session_options().value_store->prepared_bytes(), prepared);
    EXPECT_EQ(loaded->session_options().value_store->slot_count(), slots);
    EXPECT_EQ(loaded->statistics().source_bytes, 68u);
    EXPECT_EQ(loaded->statistics().external_tensors, 3u);
    EXPECT_FALSE(rt.Has("E"));
  }
  EXPECT_EQ(serial.statistics().peak_queued_tensors, 0u);
  EXPECT_EQ(overlap.statistics().peak_queued_tensors, 1u);
  EXPECT_EQ(serial.session_options().value_store, session_options.value_store);
}

TEST_F(RuntimeSessionLoading, PipelineInfersExternalLengthsBeforeReadingEitherStage) {
  auto pipeline = MakePipeline();
  for (auto *model : {pipeline.mutable_embedding(), pipeline.mutable_prefill()})
    for (auto &tensor : model->ref_graph().ref_initializer())
      if (tensor.data_location() == TensorProto::EXTERNAL)
        tensor.ref_external_data().resize(1);
  SavePipeline(pipeline);
  for (const auto *name : {"embedding.bin", "weight0.bin", "weight1.bin"})
    std::filesystem::resize_file(directory / name, 4096);
  for (bool overlap : {false, true}) {
    SCOPED_TRACE(overlap);
    size_t reads = 0;
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    options.external_reader = [&](TensorProto &tensor, const std::string &base) {
      ++reads;
      const size_t expected = tensor.name() == "E" ? 12 : 24;
      ASSERT_EQ(tensor.external_data().back().key(), "length");
      EXPECT_EQ(tensor.external_data().back().value(), std::to_string(expected));
      tensor.LoadExternalData(base);
      EXPECT_EQ(tensor.raw_data().size(), expected);
    };
    RuntimeContext rt;
    LoadedRuntimePipeline loaded(PipelinePath(), rt, {}, options);
    EXPECT_EQ(reads, 3u);
    EXPECT_EQ(loaded.statistics().source_bytes, 68u);
    RuntimeValueMap feeds;
    feeds.emplace("input_ids", Tensor::From<int64_t>("input_ids", {1, 1}, {1}));
    const auto actual = Generate(loaded, rt, feeds);
    EXPECT_EQ((std::vector<int64_t>{actual.AsInt64(), actual.AsInt64() + actual.element_count()}),
              (std::vector<int64_t>{1, 1, 1}));
  }
}

TEST_F(RuntimeSessionLoading, PipelineGenerationHonorsCpuPolicyAndRestoresCallerExecutor) {
  SavePipeline(MakePipeline());
  CpuExecutionPolicy caller_policy;
  caller_policy.num_threads = 1;
  auto executor = GlobalCpuExecutorRegistry().Acquire(caller_policy);
  for (bool explicit_policy : {false, true}) {
    RuntimeContext rt(RuntimeContextOptions{.cpu_executor = executor.get()});
    RuntimeSessionOptions session_options;
    if (explicit_policy) {
      CpuExecutionPolicy policy;
      policy.num_threads = 2;
      session_options.cpu_execution = policy;
    }
    LoadedRuntimePipeline loaded(PipelinePath(), rt, session_options);
    RuntimeValueMap feeds;
    feeds.emplace("input_ids", Tensor::From<int64_t>("input_ids", {1, 1}, {1}));
    auto *selected =
        explicit_policy ? loaded.prefill_session().cpu_executor().get() : executor.get();
    if (explicit_policy) {
      EXPECT_NE(selected, executor.get());
    }
    GenerationOptions options;
    options.logits_name = "Y0";
    options.max_new_tokens = 1;
    options.on_step = [&](const GenerationStep &) {
      EXPECT_EQ(rt.cpu_executor(), selected);
      return true;
    };
    EXPECT_EQ(Generate(loaded, rt, feeds, options).shape, (Shape{1, 2}));
    EXPECT_EQ(rt.cpu_executor(), executor.get());
    options.on_step = [&](const GenerationStep &) -> bool {
      EXPECT_EQ(rt.cpu_executor(), selected);
      throw std::runtime_error("observer failure");
    };
    EXPECT_THROW(Generate(loaded, rt, feeds, options), std::runtime_error);
    EXPECT_EQ(rt.cpu_executor(), executor.get());
  }
}

TEST_F(RuntimeSessionLoading, PipelinePrefillOnlyUsesStoredAndExplicitOptions) {
  auto pipeline = MakePipeline(false);
  pipeline.set_top_p(0.9f);
  SavePipeline(pipeline);
  RuntimeContext rt;
  LoadedRuntimePipeline loaded(PipelinePath(), rt);
  RuntimeValueMap feeds;
  feeds.emplace("input_ids", Tensor::From<int64_t>("input_ids", {1, 1}, {1}));
  EXPECT_THROW(Generate(loaded, rt, feeds), std::invalid_argument);
  GenerationOptions options;
  options.logits_name = "A";
  options.max_new_tokens = 2;
  const auto tokens = Generate(loaded, rt, feeds, options);
  EXPECT_EQ((std::vector<int64_t>{tokens.AsInt64(), tokens.AsInt64() + tokens.element_count()}),
            (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(loaded.embedding_session(), nullptr);
  EXPECT_EQ(loaded.statistics().external_tensors, 1u);
}

TEST_F(RuntimeSessionLoading, PipelineRequestsHaveIndependentPersistentState) {
  auto cases =
      core::backend_test::CollectTestCasesByName("^test_cc_generation_attention_persistent$");
  ASSERT_EQ(cases.size(), 1u);
  PipelineProto pipeline;
  *pipeline.mutable_prefill() = std::move(cases[0].model());
  pipeline.set_max_new_tokens(2);
  SavePipeline(pipeline);
  RuntimeContext rt;
  LoadedRuntimePipeline loaded(PipelinePath(), rt);
  RuntimeValueMap feeds;
  for (const auto &input : cases[0].data_sets()[0].inputs)
    feeds.emplace(input.name, RuntimeValue(input.ToOwned()));
  for (size_t request = 0; request < 2; ++request) {
    const auto tokens = Generate(loaded, rt, feeds);
    EXPECT_EQ((std::vector<int64_t>{tokens.AsInt64(), tokens.AsInt64() + tokens.element_count()}),
              (std::vector<int64_t>{0, 1, 1, 1}));
    EXPECT_EQ(feeds.at("past_key").tensor.shape, (Shape{1, 1, 0, 2}));
  }
}

TEST_F(RuntimeSessionLoading, PipelineAdmissionAggregatesStagesAndSharedStoreBeforeReads) {
  SavePipeline(MakePipeline());
  RuntimeContext baseline_rt;
  LoadedRuntimePipeline baseline(PipelinePath(), baseline_rt);
  const auto &statistics = baseline.statistics();
  const size_t reservation = statistics.source_bytes + statistics.conversion_reserve_bytes;
  for (bool overlap : {false, true}) {
    RuntimeContext rt;
    size_t reads = 0;
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
    options.memory_limit = reservation - 1;
    EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt, {}, options), std::length_error);
    EXPECT_EQ(reads, 0u);
    options.memory_limit =
        reservation + baseline.session_options().value_store->prepared_bytes() - 1;
    EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt, baseline.session_options(), options),
                 std::length_error);
    EXPECT_EQ(reads, 0u);
  }
}

TEST_F(RuntimeSessionLoading, PipelinePreparationBudgetAccountsForEarlierStages) {
  SavePipeline(MakePipeline());
  RuntimeContext baseline_rt;
  LoadedRuntimePipeline baseline(PipelinePath(), baseline_rt);
  for (bool overlap : {false, true}) {
    RuntimeContext rt;
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    options.memory_limit = baseline.statistics().peak_admitted_bytes;
    {
      LoadedRuntimePipeline exact(PipelinePath(), rt, {}, options);
      EXPECT_EQ(exact.statistics().peak_admitted_bytes, options.memory_limit);
    }
    --options.memory_limit;
    EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt, {}, options), std::length_error);
    EXPECT_FALSE(rt.Has("E"));
    EXPECT_FALSE(rt.Has("B0"));
    EXPECT_FALSE(rt.Has("B1"));
  }
}

TEST_F(RuntimeSessionLoading, PipelineOverlapsActualPreparationInLaterStage) {
  SavePipeline(MakePipeline());
  RuntimeContext rt;
  std::mutex mutex;
  std::condition_variable changed;
  bool second_started = false;
  bool prepared = false;
  RuntimeLoadOptions options;
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kPreparationStarted && event.tensor_name == "B0") {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return second_started; }))
        throw std::runtime_error("Pipeline second read did not start.");
    }
    if (event.phase == RuntimeLoadPhase::kPreparationFinished && event.tensor_name == "B0") {
      EXPECT_GT(event.prepared_bytes_added, 0u);
      std::lock_guard lock(mutex);
      prepared = true;
      changed.notify_all();
    }
  };
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    if (tensor.name() == "B1") {
      std::unique_lock lock(mutex);
      second_started = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return prepared; }))
        throw std::runtime_error("Pipeline preparation did not overlap second read.");
    }
    tensor.LoadExternalData(base);
  };
  LoadedRuntimePipeline loaded(PipelinePath(), rt, {}, options);
  EXPECT_TRUE(prepared);
  EXPECT_EQ(loaded.statistics().peak_queued_tensors, 1u);
}

TEST_F(RuntimeSessionLoading, PipelineValidatesEveryStageBeforePayloadReads) {
  for (size_t invalid = 0; invalid < 4; ++invalid) {
    SCOPED_TRACE(invalid);
    auto pipeline = MakePipeline();
    if (invalid == 0)
      pipeline = PipelineProto{};
    else if (invalid == 1)
      *pipeline.mutable_decode() = ReadModel();
    else if (invalid == 2)
      *pipeline.mutable_prefill() = ModelProto{};
    else
      pipeline.mutable_prefill()
          ->mutable_graph()
          ->mutable_initializer(0)
          ->ref_external_data()[1]
          .set_value("-1");
    SavePipeline(pipeline);
    RuntimeContext rt;
    size_t reads = 0;
    RuntimeLoadOptions options;
    options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
    EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt, {}, options), std::invalid_argument);
    EXPECT_EQ(reads, 0u);
  }
}

TEST_F(RuntimeSessionLoading, PipelineReaderFailureCleansUpEveryStageAndPreservesOverrides) {
  SavePipeline(MakePipeline());
  RuntimeContext rt;
  rt.Put("B0", Tensor::FromFloat("B0", {3, 2}, {1, 2, 3, 4, 5, 6}));
  std::atomic<size_t> reads{0};
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    ++reads;
    if (tensor.name() == "B1")
      throw std::runtime_error("pipeline reader failure");
    tensor.LoadExternalData(base);
  };
  EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt, {}, options), std::runtime_error);
  EXPECT_EQ(reads, 3u);
  EXPECT_FALSE(rt.Has("E"));
  EXPECT_TRUE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
}

TEST_F(RuntimeSessionLoading, PipelineConfinesExternalPathsInBothStages) {
  for (bool embedding : {false, true}) {
    auto pipeline = MakePipeline();
    auto *stage = embedding ? pipeline.mutable_embedding() : pipeline.mutable_prefill();
    auto *weight = stage->mutable_graph()->mutable_initializer(embedding ? 1 : 0);
    weight->mutable_external_data(0)->set_value("../weight0.bin");
    SavePipeline(pipeline);
    RuntimeContext rt;
    EXPECT_THROW(LoadedRuntimePipeline(PipelinePath(), rt), std::invalid_argument);
    EXPECT_FALSE(rt.Has("E"));
    EXPECT_FALSE(rt.Has("B0"));
  }
}

TEST_F(RuntimeSessionLoading, OverlapsRealGemmPreparationWithSecondRead) {
  RuntimeContext rt;
  std::mutex mutex;
  std::condition_variable changed;
  bool prepared = false;
  bool second_started = false;
  bool second_finished = false;
  std::thread::id io_thread;
  const auto caller = std::this_thread::get_id();
  RuntimeLoadOptions options;
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kPreparationStarted && event.tensor_name == "B0") {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return second_started; }))
        throw std::runtime_error("Second read did not start before preparation.");
    }
    if (event.phase == RuntimeLoadPhase::kPreparationFinished && event.tensor_name == "B0") {
      EXPECT_EQ(std::this_thread::get_id(), caller);
      EXPECT_GT(event.prepared_bytes_added, 0u);
      EXPECT_GT(rt.value_store().prepared_bytes(), 0u);
      std::lock_guard lock(mutex);
      prepared = true;
      EXPECT_FALSE(second_finished);
      changed.notify_all();
    }
  };
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    EXPECT_NE(std::this_thread::get_id(), caller);
    if (tensor.name() == "B1") {
      std::unique_lock lock(mutex);
      EXPECT_EQ(std::this_thread::get_id(), io_thread);
      second_started = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return prepared; }))
        throw std::runtime_error("Preparation did not overlap the blocked second read.");
    } else {
      io_thread = std::this_thread::get_id();
    }
    tensor.LoadExternalData(base);
    if (tensor.name() == "B1") {
      std::lock_guard lock(mutex);
      second_finished = true;
    }
  };
  LoadedRuntimeSession loaded(Path(), rt, {}, options);
  EXPECT_TRUE(prepared);
  EXPECT_TRUE(second_started);
  EXPECT_TRUE(second_finished);
  EXPECT_GT(loaded.session().prepared_bytes(), 0u);
  EXPECT_EQ(loaded.statistics().external_tensors, 2u);
  EXPECT_EQ(loaded.statistics().peak_queued_tensors, 1u);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, SerialOverlapExplicitAndLazyAgreeAndOwnSource) {
  RuntimeContext serial_rt;
  RuntimeLoadOptions serial;
  serial.overlap_preparation = false;
  LoadedRuntimeSession serial_loaded(Path(), serial_rt, {}, serial);
  RuntimeContext overlap_rt;
  LoadedRuntimeSession overlap_loaded(Path(), overlap_rt);

  ModelProto model;
  {
    utils::FileStream stream(Path());
    ParseOptions parse;
    ASSERT_TRUE(model.ParseFromStream(stream, parse));
  }
  LoadExternalDataForModel(model, directory.string());
  RuntimeContext explicit_rt;
  RuntimeSession explicit_session(model);
  explicit_session.Prepare(explicit_rt);
  RuntimeContext lazy_rt;
  RuntimeSession lazy_session(model);
  const auto expected = Run(lazy_session, lazy_rt);
  EXPECT_EQ(Run(explicit_session, explicit_rt), expected);
  // The writer's model was destroyed in SetUp and files are no longer needed.
  std::filesystem::remove_all(directory);
  EXPECT_EQ(Run(serial_loaded.session(), serial_rt), expected);
  EXPECT_EQ(Run(overlap_loaded.session(), overlap_rt), expected);
}

TEST_F(RuntimeSessionLoading, SerialBaselineLoadsAllPayloadsBeforePreparation) {
  RuntimeContext rt;
  size_t finished_reads = 0;
  RuntimeLoadOptions options;
  options.overlap_preparation = false;
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kReadFinished)
      ++finished_reads;
    if (event.phase == RuntimeLoadPhase::kPreparationStarted) {
      EXPECT_EQ(finished_reads, 2u);
    }
  };
  LoadedRuntimeSession loaded(Path(), rt, {}, options);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, LoadsSharedQuantizationSourcesBeforeSessionConstruction) {
  auto model = ReadModel();
  const auto encoded = QuantizeTensor(Tensor::FromFloat("X", {3}, {-2, 0, 2}),
                                      MakeQuantizationPlan(QuantizationFormat::kInt4, 3));
  auto *annotation = model.ref_graph().add_quantization_annotation();
  annotation->set_tensor_name("onnx_light.quantization.parameters:shared");
  const auto add = [&](const std::string &role, const std::string &bytes) {
    auto *tensor = model.ref_graph().add_initializer();
    tensor->set_name(role);
    tensor->set_data_type(TensorProto::UINT8);
    tensor->add_dims(bytes.size());
    tensor->set_raw_data(bytes);
    auto *mapping = annotation->add_quant_parameter_tensor_names();
    mapping->set_key(role);
    mapping->set_value(role);
  };
  add("storage_type", encoded.Encoded().struct_type().SerializeAsString());
  add("logical_type", encoded.Encoded().logical_type().SerializeAsString());
  auto *scales = model.ref_graph().add_initializer();
  scales->set_name("scales");
  scales->set_data_type(TensorProto::DOUBLE);
  scales->set_data_location(TensorProto::EXTERNAL);
  auto *entry = scales->add_external_data();
  entry->set_key("location");
  entry->set_value("scales.bin");
  auto *mapping = annotation->add_quant_parameter_tensor_names();
  mapping->set_key("scales");
  mapping->set_value("scales");
  {
    const double scale = 2;
    std::ofstream payload(directory / "scales.bin", std::ios::binary);
    payload.write(reinterpret_cast<const char *>(&scale), sizeof(scale));
  }
  SaveModel(model);
  for (bool overlap : {false, true}) {
    RuntimeContext rt;
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    bool scales_loaded = false;
    options.trace = [&](const RuntimeLoadEvent &event) {
      if (event.phase == RuntimeLoadPhase::kReadFinished && event.tensor_name == "scales")
        scales_loaded = true;
      if (event.phase == RuntimeLoadPhase::kDeclared) {
        EXPECT_TRUE(scales_loaded);
      }
    };
    LoadedRuntimeSession loaded(Path(), rt, {}, options);
    EXPECT_TRUE(scales_loaded);
    EXPECT_EQ(loaded.statistics().external_tensors, 3u);
    EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
    ASSERT_NE(rt.quantization_parameters(), nullptr);
    EXPECT_NO_THROW(rt.quantization_parameters()->Get("shared"));
  }
}

TEST_F(RuntimeSessionLoading, OverlapsInlinePreparationWithExternalRead) {
  auto model = ReadModel();
  auto &weight = model.ref_graph().ref_initializer()[0];
  weight.LoadExternalData(directory.string());
  weight.set_data_location(TensorProto::DEFAULT);
  weight.clr_external_data();
  SaveModel(model);
  RuntimeContext rt;
  std::mutex mutex;
  std::condition_variable changed;
  bool reading = false;
  bool prepared = false;
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    {
      std::unique_lock lock(mutex);
      reading = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return prepared; }))
        throw std::runtime_error("Inline preparation did not overlap external read.");
    }
    tensor.LoadExternalData(base);
  };
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kPreparationStarted && event.tensor_name == "B0") {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return reading; }))
        throw std::runtime_error("External reader did not start before inline preparation.");
    }
    if (event.phase == RuntimeLoadPhase::kPreparationFinished && event.tensor_name == "B0") {
      EXPECT_GT(event.prepared_bytes_added, 0u);
      EXPECT_GT(rt.value_store().prepared_bytes(), 0u);
      std::lock_guard lock(mutex);
      prepared = true;
      changed.notify_all();
    }
  };
  LoadedRuntimeSession loaded(Path(), rt, {}, options);
  EXPECT_TRUE(prepared);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, AdmitsExpandedSharedQuantizationBeforeReads) {
  const auto original = ReadModel().SerializeAsString();
  for (auto format : {QuantizationFormat::kInt4, QuantizationFormat::kNf4,
                      QuantizationFormat::kOrtMatmulnbitsInt4}) {
    for (int64_t blocks :
         {int64_t{4096}, static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4096)}) {
      SCOPED_TRACE(QuantizationFormatName(format));
      SCOPED_TRACE(blocks);
      ModelProto model;
      ASSERT_TRUE(model.ParseFromString(original));
      const bool ort = format == QuantizationFormat::kOrtMatmulnbitsInt4;
      StructTypeProto storage;
      if (ort) {
        const auto encoded =
            QuantizeTensor(Tensor::FromFloat("", {16, 1}, std::vector<float>(16, 0)),
                           MakeMatMulNBitsPlan(format, 16, 1, 16));
        storage = encoded.Encoded().struct_type();
        for (int field : {1, 2})
          storage.mutable_structure()
              ->mutable_field(field)
              ->mutable_type()
              ->mutable_tensor_type()
              ->mutable_shape()
              ->mutable_dim(1)
              ->set_dim_value(blocks);
      } else {
        storage = MakeQuantizationType(MakeQuantizationPlan(format, 1, 1));
        auto *fields = storage.mutable_structure()
                           ->mutable_field(6)
                           ->mutable_type()
                           ->mutable_struct_type()
                           ->mutable_structure();
        fields->mutable_field(0)->mutable_constant()->ref_int64_data()[0] = blocks;
        fields->mutable_field(1)
            ->mutable_type()
            ->mutable_struct_type()
            ->mutable_array()
            ->set_dimension(blocks);
      }
      TypeProto logical;
      auto *tensor_type = logical.mutable_tensor_type();
      tensor_type->set_elem_type(TensorProto::FLOAT);
      tensor_type->mutable_shape()->add_dim()->set_dim_value(ort ? 16 * blocks : blocks);
      if (ort)
        tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
      auto *annotation = model.ref_graph().add_quantization_annotation();
      annotation->set_tensor_name("onnx_light.quantization.parameters:shared");
      const auto add = [&](const std::string &role, const std::string &bytes) {
        auto *tensor = model.ref_graph().add_initializer();
        tensor->set_name(role);
        tensor->set_data_type(TensorProto::UINT8);
        tensor->add_dims(bytes.size());
        tensor->set_raw_data(bytes);
        auto *mapping = annotation->add_quant_parameter_tensor_names();
        mapping->set_key(role);
        mapping->set_value(role);
      };
      add("storage_type", storage.SerializeAsString());
      add("logical_type", logical.SerializeAsString());
      auto *scales = model.ref_graph().add_initializer();
      scales->set_name("scales");
      scales->set_data_type(TensorProto::DOUBLE);
      scales->set_data_location(TensorProto::EXTERNAL);
      auto *location = scales->add_external_data();
      location->set_key("location");
      location->set_value("scales.bin");
      auto *length = scales->add_external_data();
      length->set_key("length");
      length->set_value("8");
      auto *mapping = annotation->add_quant_parameter_tensor_names();
      mapping->set_key("scales");
      mapping->set_value("scales");
      {
        const double scale = 2;
        std::ofstream payload(directory / "scales.bin", std::ios::binary);
        payload.write(reinterpret_cast<const char *>(&scale), sizeof(scale));
      }
      SaveModel(model);
      for (bool overlap : {false, true}) {
        RuntimeLoadOptions options;
        options.overlap_preparation = overlap;
        options.memory_limit = 64 * 1024;
        size_t reads = 0;
        options.external_reader = [&](TensorProto &tensor, const std::string &base) {
          ++reads;
          tensor.LoadExternalData(base);
        };
        RuntimeContext rejected;
        EXPECT_THROW(LoadedRuntimeSession(Path(), rejected, {}, options), std::length_error);
        EXPECT_EQ(reads, 0u);
        if (blocks != 4096)
          continue;
        options.memory_limit = 16 * 1024 * 1024;
        RuntimeContext rt;
        LoadedRuntimeSession loaded(Path(), rt, {}, options);
        const auto &statistics = loaded.statistics();
        EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
        ASSERT_NE(rt.quantization_parameters(), nullptr);
        const auto &entry = rt.quantization_parameters()->Get("shared");
        EXPECT_EQ(entry.plan.runs[0].blocks.size(), 4096u);
        EXPECT_GE(statistics.quantization_catalogue_bytes,
                  entry.common.size() + 4096 * sizeof(QuantizationBlockParameters));
        EXPECT_GT(statistics.quantization_temporary_bytes, 0u);
        const size_t reservation = statistics.source_bytes + statistics.conversion_reserve_bytes +
                                   statistics.function_binding_bytes +
                                   statistics.quantization_catalogue_bytes +
                                   statistics.quantization_temporary_bytes;
        EXPECT_GE(statistics.peak_admitted_bytes, reservation + loaded.session().prepared_bytes());
        reads = 0;
        options.memory_limit = reservation - 1;
        RuntimeContext below_reservation;
        EXPECT_THROW(LoadedRuntimeSession(Path(), below_reservation, {}, options),
                     std::length_error);
        EXPECT_EQ(reads, 0u);
        options.memory_limit = reservation;
        RuntimeContext no_preparation_budget;
        EXPECT_THROW(LoadedRuntimeSession(Path(), no_preparation_budget, {}, options),
                     std::length_error);
        EXPECT_GT(reads, 0u);
        options.memory_limit = statistics.peak_admitted_bytes;
        RuntimeContext exact;
        LoadedRuntimeSession bounded(Path(), exact, {}, options);
        EXPECT_LE(bounded.statistics().peak_admitted_bytes, options.memory_limit);
        EXPECT_EQ(Run(bounded.session(), exact), (std::vector<float>{22, 28, 22, 28}));
      }
    }
  }
}

TEST_F(RuntimeSessionLoading, AdmissionRejectsBeforeExternalRead) {
  RuntimeContext rt;
  size_t reads = 0;
  RuntimeLoadOptions options;
  options.memory_limit = 95;
  options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::length_error);
  EXPECT_EQ(reads, 0u);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
}

TEST_F(RuntimeSessionLoading, TracksConservativeAdmission) {
  RuntimeContext rt;
  RuntimeLoadOptions options;
  options.memory_limit = 1024 * 1024;
  LoadedRuntimeSession loaded(Path(), rt, {}, options);
  EXPECT_EQ(loaded.statistics().source_bytes, 48u);
  EXPECT_EQ(loaded.statistics().conversion_reserve_bytes, 48u);
  EXPECT_GE(loaded.statistics().peak_admitted_bytes, 96u);
  EXPECT_GT(loaded.statistics().peak_admitted_bytes, 96u + loaded.session().prepared_bytes());
  EXPECT_LE(loaded.statistics().peak_admitted_bytes, options.memory_limit);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, PreparationBudgetRejectsWithoutLazyFallback) {
  RuntimeContext rt;
  RuntimeLoadOptions options;
  options.memory_limit = 100;
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::length_error);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
}

TEST_F(RuntimeSessionLoading, PreservesCallerInitializerOverrides) {
  RuntimeContext rt;
  rt.Put("B0", Tensor::FromFloat("B0", {3, 2}, {2, 2, 2, 2, 2, 2}));
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_TRUE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{12, 12, 22, 28}));
}

TEST_F(RuntimeSessionLoading, DefaultedGraphInputsRemainMutableAfterLoading) {
  auto model = ReadModel();
  model.ref_graph().add_input()->set_name("B0");
  SaveModel(model);
  RuntimeContext rt;
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_EQ(loaded.session().prepared_bytes(), 24u);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
  rt.Put("B0", Tensor::FromFloat("B0", {3, 2}, {2, 2, 2, 2, 2, 2}));
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{12, 12, 22, 28}));
  EXPECT_EQ(loaded.session().prepared_bytes(), 24u);
}

TEST_F(RuntimeSessionLoading, DefersIntermediatePreparationUntilItsConsumer) {
  auto model = ReadModel();
  auto &graph = model.ref_graph();
  graph.ref_node().clear();
  graph.ref_output().clear();
  graph.add_output()->set_name("Y0");
  graph.add_node(MakeNode("Identity", {"B0"}, {"ready_B"}));
  graph.add_node(MakeNode("LoadingPreparedIdentity", {"ready_B"}, {"Y0"}));
  SaveModel(model);
  RuntimeContext rt;
  const auto executions = std::make_shared<size_t>(0);
  ASSERT_TRUE(rt.RegisterKernelFn(
      "ai.onnx", "LoadingPreparedIdentity", core::symbolic::Device::kCPU,
      [executions](const NodeProto &node, RuntimeContext &, const KernelContext &) {
        return std::make_unique<LoadingPreparedIdentity>(node, executions);
      }));
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_EQ(*executions, 0u);
  ASSERT_EQ(loaded.session().pending_preparations().size(), 1u);
  EXPECT_EQ(loaded.session().pending_preparations()[0].source_inputs,
            (std::vector<std::string>{"ready_B"}));
  loaded.session().Run(rt);
  EXPECT_EQ(*executions, 1u);
  EXPECT_TRUE(loaded.session().pending_preparations().empty());
  ASSERT_EQ(rt.Get("Y0").element_count(), 6);
  EXPECT_FLOAT_EQ(rt.Get("Y0").AsFloat()[5], 6);
  loaded.session().Run(rt);
  EXPECT_EQ(*executions, 1u);
}

TEST_F(RuntimeSessionLoading, IndependentWeightsKeepReadinessAndAccountingLinear) {
  auto model = ReadModel();
  const TensorProto original = model.graph().initializer(0);
  auto &graph = model.ref_graph();
  graph.ref_initializer().clear();
  graph.ref_node().clear();
  graph.ref_output().clear();
  constexpr size_t count = 64;
  for (size_t index = 0; index < count; ++index) {
    const std::string source = "B" + std::to_string(index);
    const std::string output = "Y" + std::to_string(index);
    auto *tensor = graph.add_initializer();
    *tensor = original;
    tensor->set_name(source);
    graph.add_output()->set_name(output);
    graph.add_node(MakeNode("Gemm", {"A", source}, {output}));
  }
  SaveModel(model);
  RuntimeContext rt;
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_EQ(loaded.session().prepared_bytes(), count * 6 * sizeof(float));
  EXPECT_TRUE(loaded.session().pending_preparations().empty());
  EXPECT_LE(loaded.session().preparation_readiness_checks(), 2 * count);
  EXPECT_LE(loaded.session().value_store().accounting_allocator_visits(), 4 * count);
}

TEST_F(RuntimeSessionLoading, ExecutesSourceFreePreparationWithoutAnyPayload) {
  auto model = ReadModel();
  auto &graph = model.ref_graph();
  graph.ref_node().clear();
  graph.ref_initializer().clear();
  graph.ref_output().clear();
  graph.add_output()->set_name("Y0");
  graph.add_node(MakeNode("LoadingPreparedIdentity", {}, {"Y0"}));
  SaveModel(model);
  RuntimeContext rt;
  const auto executions = std::make_shared<size_t>(0);
  ASSERT_TRUE(rt.RegisterKernelFn(
      "ai.onnx", "LoadingPreparedIdentity", core::symbolic::Device::kCPU,
      [executions](const NodeProto &node, RuntimeContext &, const KernelContext &) {
        return std::make_unique<LoadingPreparedIdentity>(node, executions);
      }));
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_EQ(*executions, 1u);
  EXPECT_TRUE(loaded.session().pending_preparations().empty());
  loaded.session().Run(rt);
  EXPECT_FLOAT_EQ(rt.Get("Y0").AsFloat()[0], 1);
}

TEST_F(RuntimeSessionLoading, RejectsMismatchedExternalLengthBeforeReading) {
  auto model = ReadModel();
  model.ref_graph().ref_initializer()[0].ref_external_data()[1].set_value("23");
  SaveModel(model);
  RuntimeContext rt;
  size_t reads = 0;
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::invalid_argument);
  EXPECT_EQ(reads, 0u);
}

TEST_F(RuntimeSessionLoading, RejectsExternalRangeOverflowBeforeReading) {
  auto model = ReadModel();
  auto *offset = model.ref_graph().ref_initializer()[0].add_external_data();
  offset->set_key("offset");
  offset->set_value("9223372036854775807");
  for (bool explicit_length : {true, false}) {
    SCOPED_TRACE(explicit_length);
    if (!explicit_length)
      model.ref_graph().ref_initializer()[0].ref_external_data()[1].set_key("unused_metadata");
    SaveModel(model);
    RuntimeContext rt;
    size_t reads = 0;
    RuntimeLoadOptions options;
    options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
    EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::invalid_argument);
    EXPECT_EQ(reads, 0u);
  }
}

TEST_F(RuntimeSessionLoading, LoadsGatherWithoutExternalLength) {
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(13);
  auto *graph = model.mutable_graph();
  graph->add_input()->set_name("indices");
  graph->add_output()->set_name("Y");
  graph->add_node(MakeNode("Gather", {"weight", "indices"}, {"Y"}));
  auto *weight = graph->add_initializer();
  weight->set_name("weight");
  weight->set_data_type(TensorProto::FLOAT);
  weight->add_dims(4);
  weight->add_dims(4);
  weight->set_data_location(TensorProto::EXTERNAL);
  auto *location = weight->add_external_data();
  location->set_key("location");
  location->set_value("weights/decoder.bin");
  std::filesystem::create_directory(directory / "weights");
  const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const auto gather = [](RuntimeSession &session, RuntimeContext &rt) {
    rt.Put("indices", Tensor::FromInt64("indices", {2}, {3, 1}));
    session.Run(rt);
    const auto &output = rt.Get("Y");
    EXPECT_EQ(output.element_count(), 8);
    return std::vector<float>(output.AsFloat(), output.AsFloat() + output.element_count());
  };
  for (size_t offset : {size_t{0}, size_t{16}}) {
    SCOPED_TRACE(offset);
    if (offset != 0) {
      auto *entry = weight->add_external_data();
      entry->set_key("offset");
      entry->set_value(std::to_string(offset));
    }
    {
      std::ofstream payload(directory / "weights" / "decoder.bin", std::ios::binary);
      const std::string prefix(offset, '\0');
      payload.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
      payload.write(reinterpret_cast<const char *>(values.data()), 64);
    }
    SaveModel(model);
    auto standard_model = ReadModel();
    LoadExternalDataForModel(standard_model, directory.string());
    RuntimeContext standard_rt;
    RuntimeSession standard_session(standard_model);
    const auto expected = gather(standard_session, standard_rt);
    EXPECT_EQ(expected, (std::vector<float>{13, 14, 15, 16, 5, 6, 7, 8}));
    for (bool overlap : {false, true}) {
      SCOPED_TRACE(overlap);
      RuntimeLoadOptions options;
      options.overlap_preparation = overlap;
      options.memory_limit = 128;
      RuntimeContext rt;
      LoadedRuntimeSession loaded(Path(), rt, {}, options);
      EXPECT_EQ(loaded.statistics().source_bytes, 64u);
      EXPECT_EQ(loaded.statistics().conversion_reserve_bytes, 64u);
      EXPECT_LE(loaded.statistics().peak_admitted_bytes, options.memory_limit);
      EXPECT_EQ(gather(loaded.session(), rt), expected);
    }
  }
}

TEST_F(RuntimeSessionLoading, AdmitsMissingLengthsBeforeInvokingBoundedReader) {
  auto model = ReadModel();
  for (auto &tensor : model.ref_graph().ref_initializer())
    tensor.ref_external_data()[1].set_key("unused_metadata");
  SaveModel(model);
  // An omitted length must not let the reader allocate an unadmitted file remainder.
  std::filesystem::resize_file(directory / "weight0.bin", 4096);
  for (bool overlap : {false, true}) {
    SCOPED_TRACE(overlap);
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    options.memory_limit = 95;
    size_t reads = 0;
    options.external_reader = [&](TensorProto &tensor, const std::string &base) {
      ++reads;
      ASSERT_EQ(tensor.external_data().back().key(), "length");
      EXPECT_EQ(tensor.external_data().back().value(), "24");
      tensor.LoadExternalData(base);
      EXPECT_EQ(tensor.raw_data().size(), 24u);
    };
    RuntimeContext rejected;
    EXPECT_THROW(LoadedRuntimeSession(Path(), rejected, {}, options), std::length_error);
    EXPECT_EQ(reads, 0u);
    options.memory_limit = 1024 * 1024;
    RuntimeContext rt;
    LoadedRuntimeSession loaded(Path(), rt, {}, options);
    EXPECT_EQ(reads, 2u);
    EXPECT_EQ(loaded.statistics().source_bytes, 48u);
    EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
  }
}

TEST_F(RuntimeSessionLoading, RejectsInvalidInferredLengthsBeforeReading) {
  const auto original = ReadModel().SerializeAsString();
  const std::vector<std::vector<int64_t>> shapes{
      {-1}, {std::numeric_limits<int64_t>::max()}, {std::numeric_limits<int64_t>::max(), 3}};
  for (const auto &shape : shapes) {
    ModelProto model;
    ASSERT_TRUE(model.ParseFromString(original));
    auto &weight = model.ref_graph().ref_initializer()[0];
    weight.ref_external_data()[1].set_key("unused_metadata");
    weight.clr_dims();
    for (int64_t dim : shape)
      weight.add_dims(dim);
    SaveModel(model);
    RuntimeContext rt;
    size_t reads = 0;
    RuntimeLoadOptions options;
    options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
    EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::invalid_argument);
    EXPECT_EQ(reads, 0u);
  }
}

TEST_F(RuntimeSessionLoading, InfersScalarEmptyAndPackedPayloadLengths) {
  const auto original = ReadModel().SerializeAsString();
  struct Case {
    TensorProto::DataType type;
    std::vector<int64_t> shape;
    size_t bytes;
  };
  for (const auto &test : {Case{TensorProto::FLOAT, {}, 4}, Case{TensorProto::FLOAT, {0}, 0},
                           Case{TensorProto::UINT4, {3}, 2}}) {
    SCOPED_TRACE(test.bytes);
    ModelProto model;
    ASSERT_TRUE(model.ParseFromString(original));
    auto &graph = model.ref_graph();
    graph.ref_node().clear();
    graph.ref_input().clear();
    graph.ref_output().clear();
    graph.add_output()->set_name("B0");
    graph.ref_initializer().resize(1);
    auto &weight = graph.ref_initializer()[0];
    weight.ref_external_data()[1].set_key("unused_metadata");
    weight.set_data_type(test.type);
    weight.clr_dims();
    for (int64_t dim : test.shape)
      weight.add_dims(dim);
    SaveModel(model);
    for (bool overlap : {false, true}) {
      RuntimeContext rt;
      RuntimeLoadOptions options;
      options.overlap_preparation = overlap;
      options.memory_limit = 2 * test.bytes;
      LoadedRuntimeSession loaded(Path(), rt, {}, options);
      EXPECT_EQ(loaded.statistics().source_bytes, test.bytes);
      EXPECT_EQ(loaded.model().graph().initializer(0).raw_data().size(), test.bytes);
    }
  }
}

TEST_F(RuntimeSessionLoading, MissingLengthDoesNotHideTruncatedPayloads) {
  auto model = ReadModel();
  model.ref_graph().ref_initializer()[0].ref_external_data()[1].set_key("unused_metadata");
  SaveModel(model);
  std::filesystem::resize_file(directory / "weight0.bin", 23);
  for (bool overlap : {false, true}) {
    RuntimeContext rt;
    RuntimeLoadOptions options;
    options.overlap_preparation = overlap;
    EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::exception);
    EXPECT_FALSE(rt.Has("B0"));
    EXPECT_FALSE(rt.Has("B1"));
  }
}

TEST_F(RuntimeSessionLoading, SupportsExternalSizeAlias) {
  auto model = ReadModel();
  model.ref_graph().ref_initializer()[0].ref_external_data()[1].set_key("size");
  SaveModel(model);
  RuntimeContext rt;
  LoadedRuntimeSession loaded(Path(), rt);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, RejectsEncodedPayloadBeforeReadingOrDeclaring) {
  auto model = ReadModel();
  auto *encoded = model.ref_graph().add_extended_initializer()->mutable_encoded_value();
  encoded->set_name("encoded");
  encoded->set_raw_data(std::string(4096, '\0'));
  SaveModel(model);
  RuntimeContext rt;
  size_t reads = 0;
  bool declared = false;
  RuntimeLoadOptions options;
  options.memory_limit = 1024;
  options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
  options.trace = [&](const RuntimeLoadEvent &event) {
    declared = declared || event.phase == RuntimeLoadPhase::kDeclared;
  };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::invalid_argument);
  EXPECT_EQ(reads, 0u);
  EXPECT_FALSE(declared);
}

TEST_F(RuntimeSessionLoading, RejectsPagedPayloadBeforeReadingOrDeclaring) {
  auto model = ReadModel();
  auto *cache = model.ref_graph().add_extended_initializer()->mutable_paged_cache();
  cache->set_name("cache");
  auto *key = cache->add_blocks()->mutable_key();
  key->set_name("key");
  key->set_data_type(TensorProto::FLOAT);
  key->add_dims(1024);
  key->set_raw_data(std::string(4096, '\0'));
  SaveModel(model);
  RuntimeContext rt;
  size_t reads = 0;
  bool declared = false;
  RuntimeLoadOptions options;
  options.memory_limit = 1024;
  options.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
  options.trace = [&](const RuntimeLoadEvent &event) {
    declared = declared || event.phase == RuntimeLoadPhase::kDeclared;
  };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::invalid_argument);
  EXPECT_EQ(reads, 0u);
  EXPECT_FALSE(declared);
}

TEST_F(RuntimeSessionLoading, WorkerFailurePropagatesAndJoins) {
  RuntimeContext rt;
  std::atomic<bool> active{false};
  std::atomic<size_t> reads{0};
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    active = true;
    ++reads;
    if (tensor.name() == "B1") {
      active = false;
      throw std::runtime_error("test reader failure");
    }
    tensor.LoadExternalData(base);
    active = false;
  };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::runtime_error);
  EXPECT_EQ(reads, 2u);
  EXPECT_FALSE(active);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
}

TEST_F(RuntimeSessionLoading, CallerFailureJoinsBlockedWorker) {
  RuntimeContext rt;
  std::mutex mutex;
  std::condition_variable changed;
  bool second_started = false;
  bool release = false;
  bool exited = false;
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    if (tensor.name() == "B1") {
      std::unique_lock lock(mutex);
      second_started = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return release; }))
        throw std::runtime_error("Caller did not release reader.");
    }
    tensor.LoadExternalData(base);
    if (tensor.name() == "B1") {
      std::lock_guard lock(mutex);
      exited = true;
    }
  };
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kPreparationStarted && event.tensor_name == "B0") {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return second_started; }))
        throw std::runtime_error("Second read did not start.");
      release = true;
      changed.notify_all();
      throw std::runtime_error("test preparation observer failure");
    }
  };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, {}, options), std::runtime_error);
  EXPECT_TRUE(exited);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
}

TEST_F(RuntimeSessionLoading, InvalidGemmPreparationJoinsWorkerAndCleansSources) {
  auto model = ReadModel();
  auto &weight = model.ref_graph().ref_initializer()[0];
  weight.set_data_type(TensorProto::INT64);
  weight.clr_dims();
  weight.add_dims(3);
  weight.add_dims(1);
  SaveModel(model);
  RuntimeContext rt;
  auto store = std::make_shared<ValueStore>();
  RuntimeSessionOptions session_options;
  session_options.value_store = store;
  std::mutex mutex;
  std::condition_variable changed;
  bool second_started = false;
  bool release = false;
  bool exited = false;
  bool preparation_finished = false;
  RuntimeLoadOptions options;
  options.external_reader = [&](TensorProto &tensor, const std::string &base) {
    if (tensor.name() == "B1") {
      std::unique_lock lock(mutex);
      second_started = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return release; }))
        throw std::runtime_error("Gemm preparation did not release the active reader.");
    }
    tensor.LoadExternalData(base);
    if (tensor.name() == "B1") {
      std::lock_guard lock(mutex);
      exited = true;
    }
  };
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kPreparationStarted && event.tensor_name == "B0") {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return second_started; }))
        throw std::runtime_error("Second reader was not active during Gemm preparation.");
      release = true;
      changed.notify_all();
    }
    if (event.phase == RuntimeLoadPhase::kPreparationFinished && event.tensor_name == "B0")
      preparation_finished = true;
  };
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt, session_options, options), std::invalid_argument);
  EXPECT_TRUE(exited);
  EXPECT_FALSE(preparation_finished);
  EXPECT_FALSE(rt.Has("B0"));
  EXPECT_FALSE(rt.Has("B1"));
  EXPECT_EQ(store->prepared_bytes(), 0u);
  EXPECT_EQ(store->slot_count(), 0u);
}

TEST_F(RuntimeSessionLoading, ConfinesExternalPaths) {
  auto model = ReadModel();
  model.ref_graph().ref_initializer()[0].ref_external_data()[0].set_value("../weight0.bin");
  SaveModel(model);
  RuntimeContext rt;
  EXPECT_THROW(LoadedRuntimeSession(Path(), rt), std::exception);
  EXPECT_FALSE(rt.Has("B0"));
}

TEST_F(RuntimeSessionLoading, LoadsNestedAndFunctionAttributePayloads) {
  auto model = ReadModel();
  model.ref_graph().ref_initializer()[0].ref_external_data()[1].set_key("unused_metadata");
  const auto bytes = model.graph().initializer(0).SerializeAsString();
  auto *attribute = model.ref_graph().ref_node()[0].add_attribute();
  attribute->set_name("unused_graph");
  attribute->set_type(AttributeProto::GRAPH);
  auto *nested = attribute->mutable_g()->add_initializer();
  ASSERT_TRUE(nested->ParseFromString(bytes));
  nested->set_name("nested");
  auto *function = model.add_functions();
  function->set_name("UnusedFunction");
  function->set_domain("loading.test");
  function->add_opset_import()->set_version(13);
  auto *default_attribute = function->add_attribute_proto();
  default_attribute->set_name("default_tensor");
  default_attribute->set_type(AttributeProto::TENSOR);
  ASSERT_TRUE(default_attribute->mutable_t()->ParseFromString(bytes));
  default_attribute->mutable_t()->set_name("function_tensor");
  SaveModel(model);
  RuntimeContext rt;
  LoadedRuntimeSession loaded(Path(), rt);
  const auto &graph_tensor = loaded.model().graph().node(0).attribute(0).g().initializer(0);
  const auto &function_tensor = loaded.model().functions(0).attribute_proto(0).t();
  EXPECT_EQ(graph_tensor.raw_data().size(), 24u);
  EXPECT_EQ(function_tensor.raw_data().size(), 24u);
  EXPECT_EQ(graph_tensor.data_location(), TensorProto::DEFAULT);
  EXPECT_EQ(function_tensor.data_location(), TensorProto::DEFAULT);
  EXPECT_EQ(loaded.statistics().external_tensors, 4u);
  EXPECT_EQ(Run(loaded.session(), rt), (std::vector<float>{22, 28, 22, 28}));
}

TEST_F(RuntimeSessionLoading, LoadsFunctionConstantPayloadBeforeBindingCopiesIt) {
  auto original = ReadModel();
  original.ref_graph().ref_initializer()[0].ref_external_data()[1].set_key("unused_metadata");
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(13);
  auto *custom_import = model.add_opset_import();
  custom_import->set_domain("loading.test");
  custom_import->set_version(1);
  auto *graph = model.mutable_graph();
  graph->add_output()->set_name("Y");
  auto *call = graph->add_node();
  call->set_domain("loading.test");
  call->set_op_type("ExternalConstant");
  call->add_output("Y");
  auto *function = model.add_functions();
  function->set_domain("loading.test");
  function->set_name("ExternalConstant");
  function->add_output("out");
  function->add_opset_import()->set_version(13);
  auto *constant = function->add_node();
  constant->set_op_type("Constant");
  constant->add_output("out");
  auto *attribute = constant->add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::TENSOR);
  ASSERT_TRUE(
      attribute->mutable_t()->ParseFromString(original.graph().initializer(0).SerializeAsString()));
  for (const std::string name : {"Y1", "Y2"}) {
    graph->add_output()->set_name(name);
    auto *repeated = graph->add_node();
    repeated->set_domain("loading.test");
    repeated->set_op_type("ExternalConstant");
    repeated->add_output(name);
  }
  SaveModel(model);
  {
    RuntimeContext bounded_rt;
    RuntimeLoadOptions bounded;
    bounded.memory_limit = 119;
    size_t reads = 0;
    bounded.external_reader = [&](TensorProto &, const std::string &) { ++reads; };
    EXPECT_THROW(LoadedRuntimeSession(Path(), bounded_rt, {}, bounded), std::length_error);
    EXPECT_EQ(reads, 0u);
  }
  RuntimeContext rt;
  bool read_before_declaration = false;
  bool declared = false;
  RuntimeLoadOptions options;
  options.memory_limit = 1024;
  options.trace = [&](const RuntimeLoadEvent &event) {
    if (event.phase == RuntimeLoadPhase::kDeclared)
      declared = true;
    if (event.phase == RuntimeLoadPhase::kReadFinished) {
      EXPECT_FALSE(declared);
      read_before_declaration = true;
    }
  };
  LoadedRuntimeSession loaded(Path(), rt, {}, options);
  EXPECT_TRUE(read_before_declaration);
  EXPECT_EQ(loaded.statistics().function_binding_bytes, 72u);
  EXPECT_GE(loaded.statistics().peak_admitted_bytes, 120u);
  EXPECT_LE(loaded.statistics().peak_admitted_bytes, options.memory_limit);
  loaded.session().Run(rt);
  const auto &output = rt.Get("Y");
  ASSERT_EQ(output.element_count(), 6);
  EXPECT_EQ((std::vector<float>(output.AsFloat(), output.AsFloat() + 6)),
            (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

} // namespace
