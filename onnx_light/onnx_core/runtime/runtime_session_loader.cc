// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session_loader.h"
#include "onnx_core/runtime/kernels/value_store.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

size_t Add(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("Runtime loader byte count overflow.");
  return a + b;
}

void Trace(const RuntimeLoadOptions &options, RuntimeLoadPhase phase, const std::string &name = {},
           size_t prepared_bytes_added = 0) {
  if (options.trace) {
    options.trace({phase, name,
                   static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count()),
                   prepared_bytes_added});
  }
}

void CollectGraph(GraphProto &graph, std::vector<TensorProto *> &tensors, size_t depth);

void CollectSparse(SparseTensorProto &sparse, std::vector<TensorProto *> &tensors) {
  if (sparse.has_values())
    tensors.push_back(sparse.mutable_values());
  if (sparse.has_indices())
    tensors.push_back(sparse.mutable_indices());
}

void CollectAttribute(AttributeProto &attribute, std::vector<TensorProto *> &tensors,
                      size_t depth) {
  if (depth > 128)
    throw std::invalid_argument("Runtime loader graph nesting exceeds 128.");
  if (attribute.has_t())
    tensors.push_back(attribute.mutable_t());
  for (auto &tensor : attribute.ref_tensors())
    tensors.push_back(&tensor);
  if (attribute.has_sparse_tensor())
    CollectSparse(attribute.ref_sparse_tensor(), tensors);
  for (auto &sparse : attribute.ref_sparse_tensors())
    CollectSparse(sparse, tensors);
  if (attribute.has_g())
    CollectGraph(attribute.ref_g(), tensors, depth + 1);
  for (auto &graph : attribute.ref_graphs())
    CollectGraph(graph, tensors, depth + 1);
}

template <typename Nodes>
void CollectNodes(Nodes &nodes, std::vector<TensorProto *> &tensors, size_t depth) {
  for (auto &node : nodes)
    for (auto &attribute : node.ref_attribute())
      CollectAttribute(attribute, tensors, depth);
}

void CollectGraph(GraphProto &graph, std::vector<TensorProto *> &tensors, size_t depth) {
  if (!graph.extended_initializer().empty())
    throw std::invalid_argument(
        "Runtime loader does not support encoded or paged-cache initializers.");
  for (auto &tensor : graph.ref_initializer())
    tensors.push_back(&tensor);
  for (auto &sparse : graph.ref_sparse_initializer())
    CollectSparse(sparse, tensors);
  CollectNodes(graph.ref_node(), tensors, depth);
}

size_t InlineBytes(const TensorProto &tensor) {
  size_t result = tensor.raw_data().size();
  const auto count = [&](size_t n, size_t width) {
    if (n > std::numeric_limits<size_t>::max() / width)
      throw std::overflow_error("Runtime loader tensor byte count overflow.");
    result = Add(result, n * width);
  };
  count(tensor.float_data().size(), sizeof(float));
  count(tensor.double_data().size(), sizeof(double));
  count(tensor.int32_data().size(), sizeof(int32_t));
  count(tensor.int64_data().size(), sizeof(int64_t));
  count(tensor.uint64_data().size(), sizeof(uint64_t));
  for (const auto &value : tensor.string_data())
    result = Add(result, Add(value.size(), sizeof(std::string)));
  return result;
}

// Explicit lengths prevent the reader from allocating an unadmitted remainder
// of a file. Missing lengths are rejected rather than silently losing the bound.
size_t ExternalBytes(const TensorProto &tensor) {
  bool found = false;
  bool found_offset = false;
  size_t length = 0;
  uint64_t offset = 0;
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "offset" || entry.key() == "length" || entry.key() == "size") {
      uint64_t value = 0;
      const auto text = entry.value().sv();
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
      if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
          value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Runtime loader invalid external offset/length.");
      if (entry.key() == "offset") {
        if (found_offset)
          throw std::invalid_argument("Runtime loader duplicate external offset.");
        found_offset = true;
        offset = value;
      } else {
        if (found)
          throw std::invalid_argument("Runtime loader duplicate external length.");
        found = true;
        length = static_cast<size_t>(value);
      }
    }
  }
  if (!found)
    throw std::invalid_argument("Runtime loader requires explicit external tensor lengths.");
  if (length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - offset)
    throw std::invalid_argument("Runtime loader external offset plus length overflows.");
  const size_t width = FixedBitWidth(tensor.data_type());
  if (width == 0)
    throw std::invalid_argument("Runtime loader external tensors require a fixed-width type.");
  size_t elements = 1;
  for (int64_t dimension : tensor.dims()) {
    if (dimension < 0 || (dimension != 0 && elements > std::numeric_limits<size_t>::max() /
                                                           static_cast<uint64_t>(dimension)))
      throw std::invalid_argument("Runtime loader invalid external tensor dimensions.");
    elements *= static_cast<size_t>(dimension);
  }
  if (elements > (std::numeric_limits<size_t>::max() - 7) / width ||
      (elements * width + 7) / 8 != length)
    throw std::invalid_argument("Runtime loader external tensor length does not match its shape.");
  return length;
}

size_t TensorBytes(const TensorProto &tensor) {
  return Add(InlineBytes(tensor),
             tensor.data_location() == TensorProto::EXTERNAL && !tensor.is_raw_data()
                 ? ExternalBytes(tensor)
                 : 0);
}

size_t AttributeBytes(const AttributeProto &attribute, size_t depth = 0) {
  if (depth > 128)
    throw std::invalid_argument("Runtime loader graph nesting exceeds 128.");
  size_t bytes = 0;
  const auto sparse = [&](const SparseTensorProto &tensor) {
    if (tensor.has_values())
      bytes = Add(bytes, TensorBytes(tensor.values()));
    if (tensor.has_indices())
      bytes = Add(bytes, TensorBytes(tensor.indices()));
  };
  if (attribute.has_t())
    bytes = Add(bytes, TensorBytes(attribute.t()));
  for (const auto &tensor : attribute.tensors())
    bytes = Add(bytes, TensorBytes(tensor));
  if (attribute.has_sparse_tensor())
    sparse(attribute.sparse_tensor());
  for (const auto &tensor : attribute.sparse_tensors())
    sparse(tensor);
  const auto graph = [&](const GraphProto &value) {
    for (const auto &tensor : value.initializer())
      bytes = Add(bytes, TensorBytes(tensor));
    for (const auto &tensor : value.sparse_initializer())
      sparse(tensor);
    for (const auto &node : value.node())
      for (const auto &nested : node.attribute())
        bytes = Add(bytes, AttributeBytes(nested, depth + 1));
  };
  if (attribute.has_g())
    graph(attribute.g());
  for (const auto &value : attribute.graphs())
    graph(value);
  return bytes;
}

size_t
BoundReferenceBytes(const NodeProto &node,
                    const std::unordered_map<std::string, const AttributeProto *> &attributes,
                    size_t depth = 0) {
  if (depth > 128)
    throw std::invalid_argument("Runtime loader graph nesting exceeds 128.");
  size_t bytes = 0;
  for (const auto &attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty()) {
      const auto found = attributes.find(attribute.ref_attr_name());
      if (found != attributes.end())
        bytes = Add(bytes, AttributeBytes(*found->second));
    } else {
      const auto graph = [&](const GraphProto &value) {
        for (const auto &nested : value.node())
          bytes = Add(bytes, BoundReferenceBytes(nested, attributes, depth + 1));
      };
      if (attribute.has_g())
        graph(attribute.g());
      for (const auto &value : attribute.graphs())
        graph(value);
    }
  }
  return bytes;
}

struct FunctionBindingMemory {
  size_t retained_bytes = 0;
  size_t temporary_bytes = 0;
};

FunctionBindingMemory FunctionBindingBytes(const ModelProto &model, const RuntimeContext &rt) {
  const auto key = [](const std::string &domain, const std::string &name,
                      const std::string &overload) {
    return NormaliseDomain(domain) + ":" + name + ":" + overload;
  };
  FunctionMap functions = rt.functions();
  for (const auto &function : model.functions())
    functions[key(function.domain(), function.name(), function.overload())] = &function;
  FunctionBindingMemory memory;
  for (const auto &call : model.graph().node()) {
    const auto found = functions.find(key(call.domain(), call.op_type(), call.overload()));
    if (found == functions.end())
      continue;
    const auto &function = *found->second;
    size_t body_bytes = 0;
    size_t reference_bytes = 0;
    std::unordered_map<std::string, const AttributeProto *> attributes;
    for (const auto &attribute : function.attribute_proto()) {
      body_bytes = Add(body_bytes, AttributeBytes(attribute));
      attributes[attribute.name()] = &attribute;
    }
    for (const auto &attribute : call.attribute()) {
      attributes[attribute.name()] = &attribute;
      memory.temporary_bytes = std::max(memory.temporary_bytes,
                                        Add(attribute.ByteSizeLong(), AttributeBytes(attribute)));
    }
    for (const auto &node : function.node()) {
      for (const auto &attribute : node.attribute())
        body_bytes = Add(body_bytes, AttributeBytes(attribute));
      reference_bytes = Add(reference_bytes, BoundReferenceBytes(node, attributes));
    }
    memory.retained_bytes = Add(memory.retained_bytes, Add(body_bytes, reference_bytes));
    memory.temporary_bytes =
        std::max(memory.temporary_bytes, Add(function.ByteSizeLong(), body_bytes));
  }
  return memory;
}

struct SeedCleanup {
  RuntimeContext &rt;
  std::vector<std::string> names;
  ~SeedCleanup() {
    for (const auto &name : names)
      rt.Remove(name);
  }
};

// Owns the worker before any caller-side preparation can throw. The only queue
// slot is released when the caller takes the loaded tensor, admitting one ahead.
class Reader {
public:
  Reader(const std::vector<TensorProto *> &tensors, const std::vector<size_t> &lengths,
         const std::string &base, const RuntimeLoadOptions &options)
      : tensors_(tensors), lengths_(lengths), base_(base), options_(options),
        worker_([this] { Work(); }) {}
  ~Reader() {
    {
      std::lock_guard lock(mutex_);
      stopped_ = true;
    }
    changed_.notify_all();
    worker_.join();
  }
  void Take(size_t index) {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return ready_ > index || error_; });
    if (error_)
      std::rethrow_exception(error_);
    consumed_ = index + 1;
    changed_.notify_all();
  }

  static void Read(TensorProto &tensor, size_t length, const std::string &base,
                   const RuntimeLoadOptions &options) {
    Trace(options, RuntimeLoadPhase::kReadStarted, tensor.name());
    if (options.external_reader)
      options.external_reader(tensor, base);
    else
      tensor.LoadExternalData(base);
    if (tensor.raw_data().size() != length)
      throw std::invalid_argument("Runtime loader reader returned an unexpected payload length.");
    Trace(options, RuntimeLoadPhase::kReadFinished, tensor.name());
  }

private:
  void Work() noexcept {
    try {
      for (size_t i = 0; i < tensors_.size(); ++i) {
        {
          std::unique_lock lock(mutex_);
          changed_.wait(lock, [&] { return stopped_ || i <= consumed_; });
          if (stopped_)
            return;
        }
        Read(*tensors_[i], lengths_[i], base_, options_);
        {
          std::lock_guard lock(mutex_);
          ready_ = i + 1;
        }
        changed_.notify_all();
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        error_ = std::current_exception();
      }
      changed_.notify_all();
    }
  }

  const std::vector<TensorProto *> &tensors_;
  const std::vector<size_t> &lengths_;
  const std::string &base_;
  const RuntimeLoadOptions &options_;
  std::mutex mutex_;
  std::condition_variable changed_;
  size_t ready_ = 0;
  size_t consumed_ = 0;
  bool stopped_ = false;
  std::exception_ptr error_;
  std::thread worker_;
};

struct ModelLoadingPlan {
  ModelProto *model;
  std::vector<TensorProto *> tensors;
  std::unordered_set<const TensorProto *> initializers;
  std::unordered_set<std::string> quantization_sources;
  std::vector<TensorProto *> external;
  std::vector<size_t> lengths;
};

std::vector<std::unique_ptr<RuntimeSession>>
LoadSessions(std::span<ModelProto *> models, const std::string &path, RuntimeContext &rt,
             const RuntimeSessionOptions &session_options, const RuntimeLoadOptions &load_options,
             RuntimeLoadStatistics &statistics) {
  std::vector<ModelLoadingPlan> plans;
  plans.reserve(models.size());
  for (auto *model : models) {
    if (!model->has_graph())
      throw std::invalid_argument("Runtime loader requires a model graph in every stage.");
    auto &plan = plans.emplace_back();
    plan.model = model;
    CollectGraph(model->ref_graph(), plan.tensors, 0);
    for (auto &function : model->ref_functions()) {
      CollectNodes(function.ref_node(), plan.tensors, 0);
      for (auto &attribute : function.ref_attribute_proto())
        CollectAttribute(attribute, plan.tensors, 0);
    }
    for (const auto &tensor : model->graph().initializer())
      plan.initializers.insert(&tensor);
    for (const auto &annotation : model->graph().quantization_annotation())
      for (const auto &mapping : annotation.quant_parameter_tensor_names())
        plan.quantization_sources.insert(mapping.value());
    size_t source_bytes = 0;
    for (TensorProto *tensor : plan.tensors) {
      source_bytes = Add(source_bytes, InlineBytes(*tensor));
      if (tensor->data_location() == TensorProto::EXTERNAL) {
        const size_t length = ExternalBytes(*tensor);
        plan.external.push_back(tensor);
        plan.lengths.push_back(length);
        source_bytes = Add(source_bytes, length);
      }
    }
    const auto binding = FunctionBindingBytes(*model, rt);
    statistics.source_bytes = Add(statistics.source_bytes, source_bytes);
    statistics.conversion_reserve_bytes =
        Add(statistics.conversion_reserve_bytes, std::max(source_bytes, binding.temporary_bytes));
    statistics.function_binding_bytes =
        Add(statistics.function_binding_bytes, binding.retained_bytes);
    statistics.external_tensors = Add(statistics.external_tensors, plan.external.size());
  }
  size_t source_reservation = Add(Add(statistics.source_bytes, statistics.conversion_reserve_bytes),
                                  statistics.function_binding_bytes);
  if (source_reservation > load_options.memory_limit)
    throw std::length_error("Runtime loader source payload exceeds memory admission limit.");
  for (auto *model : models) {
    const auto catalogue = QuantizationParameterCatalogue::EstimateMemory(*model);
    statistics.quantization_catalogue_bytes =
        Add(statistics.quantization_catalogue_bytes, catalogue.retained_bytes);
    statistics.quantization_temporary_bytes =
        Add(statistics.quantization_temporary_bytes, catalogue.temporary_bytes);
  }
  source_reservation = Add(source_reservation, Add(statistics.quantization_catalogue_bytes,
                                                   statistics.quantization_temporary_bytes));
  if (source_reservation > load_options.memory_limit)
    throw std::length_error(
        "Runtime loader quantization catalogue exceeds memory admission limit.");
  statistics.peak_admitted_bytes =
      Add(source_reservation,
          session_options.value_store ? session_options.value_store->prepared_bytes() : 0);
  if (statistics.peak_admitted_bytes > load_options.memory_limit)
    throw std::length_error("Runtime loader shared preparation store exceeds memory limit.");
  const size_t preparation_limit = load_options.memory_limit - source_reservation;
  const std::string base = std::filesystem::absolute(path).parent_path().string();
  std::vector<std::unique_ptr<RuntimeSession>> sessions;
  sessions.reserve(models.size());
  for (auto &plan : plans) {
    auto &model = *plan.model;
    auto &tensors = plan.tensors;
    auto &initializers = plan.initializers;
    auto &quantization_sources = plan.quantization_sources;
    auto &external = plan.external;
    auto &lengths = plan.lengths;
    std::vector<TensorProto *> root_external;
    std::vector<TensorProto *> preloaded_initializers;
    std::vector<size_t> root_lengths;
    for (size_t i = 0; i < external.size(); ++i) {
      if (initializers.count(external[i]) && !quantization_sources.contains(external[i]->name())) {
        root_external.push_back(external[i]);
        root_lengths.push_back(lengths[i]);
      } else {
        // Function binding and custom factories may copy attribute payloads during
        // declaration. Quantization catalogues also consume annotated initializers
        // during construction, so only independent root initializers remain unloaded.
        Reader::Read(*external[i], lengths[i], base, load_options);
        if (!initializers.count(external[i])) {
          external[i]->set_data_location(TensorProto::DEFAULT);
          external[i]->clr_external_data();
        } else {
          preloaded_initializers.push_back(external[i]);
        }
      }
    }
    external = std::move(root_external);
    lengths = std::move(root_lengths);
    if (load_options.overlap_preparation && !external.empty())
      statistics.peak_queued_tensors = 1;
    auto session = std::make_unique<RuntimeSession>(model, session_options);
    for (TensorProto *tensor : preloaded_initializers) {
      tensor->set_data_location(TensorProto::DEFAULT);
      tensor->clr_external_data();
    }
    statistics.peak_admitted_bytes = std::max(statistics.peak_admitted_bytes,
                                              Add(source_reservation, session->prepared_bytes()));
    if (statistics.peak_admitted_bytes > load_options.memory_limit)
      throw std::length_error("Runtime loader shared preparation store exceeds memory limit.");
    session->DeclarePreparations(rt);
    Trace(load_options, RuntimeLoadPhase::kDeclared);

    SeedCleanup cleanup{rt, {}};
    const auto prepare = [&](TensorProto *tensor) {
      if (tensor != nullptr && initializers.count(tensor) && !rt.HasValue(tensor->name())) {
        cleanup.names.push_back(tensor->name());
        rt.Set(tensor->name(), Tensor::BorrowInitializer(*tensor), RuntimeEventKind::kInitializer);
      }
      const size_t prepared_before = session->prepared_bytes();
      const std::string name = tensor == nullptr ? std::string{} : tensor->name().value();
      Trace(load_options, RuntimeLoadPhase::kPreparationStarted, name);
      session->ExecutePreparations(rt, preparation_limit,
                                   tensor == nullptr ? std::span<const std::string>{}
                                                     : std::span<const std::string>(&name, 1));
      statistics.peak_admitted_bytes =
          std::max(statistics.peak_admitted_bytes,
                   Add(source_reservation,
                       std::max(session->preparation_peak_bytes(), session->prepared_bytes())));
      const size_t prepared_after = session->prepared_bytes();
      Trace(load_options, RuntimeLoadPhase::kPreparationFinished, name,
            prepared_after > prepared_before ? prepared_after - prepared_before : 0);
    };
    const auto prepare_inline = [&] {
      for (TensorProto *tensor : tensors)
        if (tensor->data_location() != TensorProto::EXTERNAL)
          prepare(tensor);
    };
    if (load_options.overlap_preparation && !external.empty()) {
      Reader reader(external, lengths, base, load_options);
      prepare_inline();
      for (size_t i = 0; i < external.size(); ++i) {
        reader.Take(i);
        prepare(external[i]);
      }
    } else {
      for (size_t i = 0; i < external.size(); ++i)
        Reader::Read(*external[i], lengths[i], base, load_options);
      prepare_inline();
      for (size_t i = 0; i < external.size(); ++i)
        prepare(external[i]);
    }
    // Metadata is changed only after the worker is joined, never concurrently
    // with kernel metadata inspection.
    for (TensorProto *tensor : external) {
      tensor->set_data_location(TensorProto::DEFAULT);
      tensor->clr_external_data();
    }
    prepare(nullptr);
    std::unordered_set<std::string> runtime_sources;
    std::unordered_set<std::string> produced_sources;
    for (const auto &input : model.graph().input())
      runtime_sources.insert(input.name());
    for (const auto &node : model.graph().node())
      for (const auto &output : node.output())
        produced_sources.insert(output);
    std::unordered_set<size_t> deferred;
    for (const auto &preparation : session->pending_preparations()) {
      bool waits_for_runtime = preparation.awaits_runtime_tuning;
      for (const auto &name : preparation.source_inputs) {
        if (produced_sources.contains(name)) {
          waits_for_runtime = true;
          continue;
        }
        if (rt.HasValue(name))
          continue;
        if (!runtime_sources.contains(name))
          throw std::invalid_argument("Runtime loader unknown preparation source: " + name);
        waits_for_runtime = true;
      }
      for (size_t dependency : preparation.dependencies)
        waits_for_runtime = waits_for_runtime || deferred.contains(dependency);
      if (!waits_for_runtime)
        throw std::length_error("Runtime loader cannot admit all ready declared preparations.");
      deferred.insert(preparation.index);
    }
    sessions.push_back(std::move(session));
  }
  Trace(load_options, RuntimeLoadPhase::kComplete);
  return sessions;
}

} // namespace

LoadedRuntimeSession::LoadedRuntimeSession(const std::string &path, RuntimeContext &rt,
                                           RuntimeSessionOptions session_options,
                                           RuntimeLoadOptions load_options) {
  utils::FileStream stream(path);
  ParseOptions parse_options;
  if (!model_.ParseFromStream(stream, parse_options) || !model_.has_graph())
    throw std::invalid_argument("Runtime loader failed to parse a model graph.");
  ModelProto *models[] = {&model_};
  auto sessions = LoadSessions(models, path, rt, session_options, load_options, statistics_);
  session_ = std::move(sessions.front());
}

LoadedRuntimeSession::~LoadedRuntimeSession() = default;

LoadedRuntimePipeline::LoadedRuntimePipeline(const std::string &path, RuntimeContext &rt,
                                             RuntimeSessionOptions session_options,
                                             RuntimeLoadOptions load_options)
    : session_options_(std::move(session_options)) {
  utils::FileStream stream(path);
  ParseOptions parse_options;
  if (!pipeline_.ParseFromStream(stream, parse_options) || !pipeline_.has_prefill())
    throw std::invalid_argument("Runtime pipeline loader requires a prefill model.");
  if (pipeline_.has_decode())
    throw std::invalid_argument(
        "Runtime pipeline loader does not support a separate decode model.");
  if (!session_options_.value_store)
    session_options_.value_store = std::make_shared<ValueStore>();
  std::vector<ModelProto *> models;
  if (pipeline_.has_embedding())
    models.push_back(pipeline_.mutable_embedding());
  models.push_back(pipeline_.mutable_prefill());
  auto sessions = LoadSessions(models, path, rt, session_options_, load_options, statistics_);
  prefill_session_ = std::move(sessions.back());
  if (pipeline_.has_embedding())
    embedding_session_ = std::move(sessions.front());
}

LoadedRuntimePipeline::~LoadedRuntimePipeline() = default;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
