// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/generation.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/run_nodes_internal.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/runtime_session_loader.h"
#include "onnx_lib/common/scoped_resource.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

bool HasInput(const ModelProto &model, const std::string &name) {
  return std::any_of(model.graph().input().begin(), model.graph().input().end(),
                     [&](const auto &input) { return input.name() == name; });
}

void ValidateStorage(const Tensor &tensor) {
  const int64_t count = tensor.shape.product();
  EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(tensor.data_type, count) &&
                          (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                      "Generate: invalid tensor storage.");
}

const Tensor &IntMatrix(const RuntimeValueMap &feeds, const std::string &name) {
  const auto it = feeds.find(name);
  EXT_ENFORCE_INVALID(it != feeds.end() && it->second.kind == RuntimeValue::Kind::kTensor,
                      "Generate: missing tensor input '", name, "'.");
  const Tensor &tensor = it->second.tensor;
  EXT_ENFORCE_INVALID(tensor.data_type == DataType::INT64 && tensor.shape.size() == 2 &&
                          tensor.shape[0] > 0 && tensor.shape[1] > 0,
                      "Generate: '", name, "' must be a nonempty INT64 matrix.");
  ValidateStorage(tensor);
  return tensor;
}

int64_t PositionAxes(const ModelProto &model, const std::string &name, int64_t configured,
                     bool &rank2, bool &rank3) {
  for (const auto &input : model.graph().input()) {
    if (input.name() != name)
      continue;
    EXT_ENFORCE_INVALID(input.type().has_tensor_type(), "Generate: position IDs must be a tensor.");
    const auto &type = input.type().tensor_type();
    EXT_ENFORCE_INVALID(type.elem_type() == TensorProto::INT64,
                        "Generate: position IDs must be INT64.");
    if (!type.has_shape())
      return configured;
    const auto &shape = type.shape();
    EXT_ENFORCE_INVALID(shape.dim_size() == 2 || shape.dim_size() == 3,
                        "Generate: position IDs must have rank 2 or 3.");
    if (shape.dim_size() == 2) {
      rank2 = true;
      EXT_ENFORCE_INVALID(configured == 0, "Generate: position axes require rank-3 input.");
      return 0;
    }
    rank3 = true;
    const auto &axis = shape.dim(0);
    EXT_ENFORCE_INVALID(!axis.has_dim_value() || axis.dim_value() > 0,
                        "Generate: invalid position axis count.");
    EXT_ENFORCE_INVALID(!axis.has_dim_value() || configured == 0 || configured == axis.dim_value(),
                        "Generate: position axis count mismatch.");
    return configured ? configured : (axis.has_dim_value() ? axis.dim_value() : 0);
  }
  return configured;
}

std::vector<int64_t> IntValues(const Tensor &tensor) {
  return {tensor.AsInt64(), tensor.AsInt64() + tensor.shape.product()};
}

void AppendColumn(std::vector<int64_t> &matrix, const std::vector<int64_t> &column) {
  const size_t width = matrix.size() / column.size();
  std::vector<int64_t> next;
  next.reserve(matrix.size() + column.size());
  for (size_t row = 0; row < column.size(); ++row) {
    next.insert(next.end(), matrix.begin() + row * width, matrix.begin() + (row + 1) * width);
    next.push_back(column[row]);
  }
  matrix.swap(next);
}

int64_t Sample(const Tensor &logits, int64_t offset, int64_t vocabulary, double temperature,
               std::mt19937_64 &random) {
  int64_t best = 0;
  double maximum = -std::numeric_limits<double>::infinity();
  if (temperature == 0.) {
    for (int64_t token = 0; token < vocabulary; ++token) {
      const double value = logits.data_type == DataType::DOUBLE ? logits.AsDouble()[offset + token]
                                                                : logits.AsFloat()[offset + token];
      EXT_ENFORCE_INVALID(!std::isnan(value) && value != std::numeric_limits<double>::infinity(),
                          "Generate: logits must not contain NaN or positive infinity.");
      if (value > maximum) {
        maximum = value;
        best = token;
      }
    }
    EXT_ENFORCE_INVALID(std::isfinite(maximum), "Generate: all logits are negative infinity.");
    return best;
  }

  std::vector<double> weights(static_cast<size_t>(vocabulary));
  for (int64_t token = 0; token < vocabulary; ++token) {
    const double value = logits.data_type == DataType::DOUBLE ? logits.AsDouble()[offset + token]
                                                              : logits.AsFloat()[offset + token];
    EXT_ENFORCE_INVALID(!std::isnan(value) && value != std::numeric_limits<double>::infinity(),
                        "Generate: logits must not contain NaN or positive infinity.");
    weights[token] = value;
    if (value > maximum) {
      maximum = value;
      best = token;
    }
  }
  EXT_ENFORCE_INVALID(std::isfinite(maximum), "Generate: all logits are negative infinity.");
  for (double &value : weights)
    value = std::exp((value - maximum) / temperature);
  std::discrete_distribution<int64_t> distribution(weights.begin(), weights.end());
  return distribution(random);
}

void RunGraph(const ModelProto &model, RuntimeContext &invocation, RuntimeValueMap &values,
              RuntimeSession &session) {
  invocation.ResetInvocation();
  for (const auto &input : model.graph().input()) {
    const std::string &name = input.name();
    const auto value = values.find(name);
    const auto has_initializer = [&](const auto &initializers) {
      return std::any_of(initializers.begin(), initializers.end(),
                         [&](const auto &initializer) { return initializer.name() == name; });
    };
    if (value == values.end() && (has_initializer(model.graph().initializer()) ||
                                  has_initializer(model.graph().extended_initializer())))
      continue;
    EXT_ENFORCE_INVALID(value != values.end(), "Generate: missing graph feed '", name, "'.");
    if (input.type().has_sequence_type() &&
        input.type().sequence_type().elem_type().has_tensor_type() &&
        value->second.kind == RuntimeValue::Kind::kSequence &&
        std::all_of(
            value->second.elements.begin(), value->second.elements.end(),
            [](const auto &element) { return element.kind == RuntimeValue::Kind::kTensor; })) {
      Tensors tensors;
      for (const auto &element : value->second.elements) {
        tensors.push_back(element.tensor.BorrowView());
      }
      invocation.PutSequence(
          name, Sequence(name, input.type().sequence_type().elem_type().tensor_type().elem_type(),
                         std::move(tensors)));
    } else {
      invocation.PutValue(name, value->second.BorrowView(), RuntimeEventKind::kInput);
    }
  }
  session.Run(invocation);
  for (const auto &declared : model.graph().output()) {
    const std::string &output = declared.name();
    if (invocation.Has(output)) {
      values.insert_or_assign(output,
                              RuntimeValue(std::move(invocation.Get(output)).RetainStorage()));
    } else if (invocation.HasSequence(output)) {
      std::vector<RuntimeValue> elements;
      for (auto &tensor : invocation.GetSequence(output).values)
        elements.emplace_back(tensor.ToOwned());
      values.insert_or_assign(output, RuntimeValue(std::move(elements)));
    } else {
      EXT_THROW_INVALID("Generate: missing graph output '", output, "'.");
    }
  }
}

void ValidateGraph(const ModelProto &model) {
  std::unordered_set<std::string> destinations;
  for (const auto &output : model.graph().output()) {
    EXT_ENFORCE_INVALID(!output.name().empty() &&
                            (output.type().has_tensor_type() ||
                             (output.type().has_sequence_type() &&
                              output.type().sequence_type().elem_type().has_tensor_type())),
                        "Generate: unsupported graph output '", output.name(), "'.");
    EXT_ENFORCE_INVALID(destinations.insert(output.name()).second,
                        "Generate: duplicate graph output '", output.name(), "'.");
  }
  EXT_ENFORCE_INVALID(!model.graph().output().empty(), "Generate: graph outputs are required.");
}

struct PipelineValueNames {
  std::unordered_set<std::string> all;
  std::unordered_set<std::string> private_values;
};

void InsertName(std::unordered_set<std::string> &names, std::string_view name) {
  if (!name.empty())
    names.emplace(name);
}

void CollectGraphValueNames(const GraphProto &graph, bool top_level,
                            const std::unordered_set<std::string> &public_initializer_outputs,
                            PipelineValueNames &names) {
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  for (const ValueInfoProto &input : graph.input()) {
    InsertName(names.all, input.name());
    inputs.insert(input.name());
    if (!top_level)
      InsertName(names.private_values, input.name());
  }
  for (const ValueInfoProto &output : graph.output()) {
    InsertName(names.all, output.name());
    outputs.insert(output.name());
    if (!top_level)
      InsertName(names.private_values, output.name());
  }
  const auto add_initializer = [&](const std::string &name) {
    InsertName(names.all, name);
    if (!top_level || (!inputs.contains(name) && !public_initializer_outputs.contains(name)))
      InsertName(names.private_values, name);
  };
  for (const TensorProto &initializer : graph.initializer())
    add_initializer(initializer.name());
  for (const SparseTensorProto &initializer : graph.sparse_initializer())
    add_initializer(initializer.values().name());
  for (const auto &initializer : graph.extended_initializer())
    add_initializer(initializer.name());
  for (const NodeProto &node : graph.node()) {
    for (const std::string &output : node.output()) {
      InsertName(names.all, output);
      if (!top_level || !outputs.contains(output))
        InsertName(names.private_values, output);
    }
    for (const AttributeProto &attribute : node.attribute()) {
      if (attribute.has_g()) {
        CollectGraphValueNames(attribute.g(), false, {}, names);
      }
      for (const GraphProto &nested : attribute.graphs())
        CollectGraphValueNames(nested, false, {}, names);
    }
  }
}

std::string ModelFunctionKey(const NodeProto &node) {
  return ONNX_LIGHT_NAMESPACE::NormaliseDomain(node.domain()) + ":" + node.op_type() + ":" +
         node.overload();
}

void CollectFunctionValueNames(const FunctionProto &function, PipelineValueNames &names) {
  std::unordered_set<std::string> formal_outputs(function.output().begin(),
                                                 function.output().end());
  for (const NodeProto &node : function.node()) {
    for (const std::string &output : node.output()) {
      if (!formal_outputs.contains(output)) {
        InsertName(names.all, output);
        InsertName(names.private_values, output);
      }
    }
  }
}

void CollectReachableFunctionGraphNames(const utils::RepeatedProtoField<NodeProto> &nodes,
                                        const FunctionMap &functions, PipelineValueNames &names,
                                        std::unordered_set<std::string> &active,
                                        std::unordered_set<std::string> &completed,
                                        const RuntimeContext &context) {
  for (const NodeProto &node : nodes) {
    for (const AttributeProto &attribute : node.attribute()) {
      if (attribute.has_g()) {
        CollectGraphValueNames(attribute.g(), false, {}, names);
        CollectReachableFunctionGraphNames(attribute.g().node(), functions, names, active,
                                           completed, context);
      }
      for (const GraphProto &nested : attribute.graphs()) {
        CollectGraphValueNames(nested, false, {}, names);
        CollectReachableFunctionGraphNames(nested.node(), functions, names, active, completed,
                                           context);
      }
    }
    const std::string key = ModelFunctionKey(node);
    const auto found = functions.find(key);
    if (found == functions.end() || active.contains(key) ||
        detail::HasRegisteredNodeKernel(node, context.device(), &context.custom_kernels()))
      continue;
    const FunctionProto bound = detail::BindFunctionAttributes(*found->second, node);
    // Calls to one function may bind different graph-valued attributes.
    std::string completed_key = key;
    completed_key.push_back('\0');
    completed_key.append(bound.SerializeAsString());
    if (completed.contains(completed_key))
      continue;
    active.insert(key);
    CollectFunctionValueNames(bound, names);
    CollectReachableFunctionGraphNames(bound.node(), functions, names, active, completed, context);
    active.erase(key);
    completed.insert(std::move(completed_key));
  }
}

PipelineValueNames
CollectPipelineValueNames(const ModelProto &model,
                          const std::unordered_set<std::string> &public_initializer_outputs,
                          const RuntimeContext &context) {
  PipelineValueNames names;
  CollectGraphValueNames(model.graph(), true, public_initializer_outputs, names);
  FunctionMap functions = context.functions();
  for (const auto &[key, function] : detail::ModelFunctions(model))
    functions.insert_or_assign(key, function);
  std::unordered_set<std::string> active;
  std::unordered_set<std::string> completed;
  CollectReachableFunctionGraphNames(model.graph().node(), functions, names, active, completed,
                                     context);
  return names;
}

void ValidatePipelineValueNames(const ModelProto &embedding, const ModelProto &language,
                                const RuntimeContext &context) {
  std::unordered_set<std::string> public_connections;
  for (const ValueInfoProto &output : embedding.graph().output())
    if (HasInput(language, output.name()))
      public_connections.insert(output.name());
  const PipelineValueNames embedding_names =
      CollectPipelineValueNames(embedding, public_connections, context);
  const PipelineValueNames language_names = CollectPipelineValueNames(language, {}, context);
  std::vector<std::string> collisions;
  for (const std::string &name : embedding_names.private_values)
    if (language_names.all.contains(name))
      collisions.push_back(name);
  for (const std::string &name : language_names.private_values)
    if (embedding_names.all.contains(name))
      collisions.push_back(name);
  std::sort(collisions.begin(), collisions.end());
  collisions.erase(std::unique(collisions.begin(), collisions.end()), collisions.end());
  EXT_ENFORCE_INVALID(
      collisions.empty(), "Generate: private value name '", collisions.front(),
      "' is shared by the embedding and language models; private value names must be unique.");
}

struct GenerationPipelineView {
  const ModelProto *prefill = nullptr;
  const ModelProto *embedding = nullptr;
};

void BindParameterReferences(RuntimeValue &value, const ModelProto &model,
                             const StructTypeCatalogue &types,
                             std::shared_ptr<const QuantizationParameterCatalogue> &parameters,
                             size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "Generate: feed value exceeds the maximum nesting depth.");
  if (value.kind == RuntimeValue::Kind::kEncoded && value.Encoded().has_parameter_ref()) {
    if (!parameters)
      parameters = QuantizationParameterCatalogue::Build(model);
    parameters->Validate(value.Encoded(), types);
    value.quantization_parameters = parameters;
  }
  for (auto &[name, field] : value.fields) {
    (void)name;
    BindParameterReferences(field, model, types, parameters, depth + 1);
  }
  value.elements = value.elements.Transform([&](const RuntimeValue &element) {
    RuntimeValue scoped = element.BorrowView();
    BindParameterReferences(scoped, model, types, parameters, depth + 1);
    return scoped;
  });
}

RuntimeValueMap ScopeFeeds(const ModelProto &model, const RuntimeValueMap &feeds) {
  StructTypeCatalogue types;
  types.Build(model);
  std::shared_ptr<const QuantizationParameterCatalogue> parameters;
  RuntimeValueMap result;
  for (const auto &[name, value] : feeds) {
    if (!HasInput(model, name))
      continue;
    RuntimeValue scoped = value.BorrowView();
    BindParameterReferences(scoped, model, types, parameters);
    result.emplace(name, std::move(scoped));
  }
  return result;
}

GenerationOptions PipelineOptions(const PipelineProto &pipeline) {
  GenerationOptions options;
  if (pipeline.has_max_new_tokens())
    options.max_new_tokens = pipeline.max_new_tokens();
  if (pipeline.has_temperature())
    options.temperature = pipeline.temperature();
  if (pipeline.has_seed()) {
    EXT_ENFORCE_INVALID(pipeline.seed() >= 0, "Generate: seed must be nonnegative.");
    options.seed = static_cast<uint64_t>(pipeline.seed());
  }
  if (pipeline.has_eos_token_id())
    options.eos_token_id = pipeline.eos_token_id();
  if (pipeline.has_pad_token_id())
    options.pad_token_id = pipeline.pad_token_id();
  EXT_ENFORCE_INVALID(!pipeline.has_top_k() || pipeline.top_k() == 0,
                      "Generate: top-k sampling is not supported.");
  EXT_ENFORCE_INVALID(!pipeline.has_top_p(), "Generate: top-p sampling is not supported.");
  if (pipeline.has_input_ids_name())
    options.input_ids_name = pipeline.input_ids_name();
  if (pipeline.has_logits_name())
    options.logits_name = pipeline.logits_name();
  if (pipeline.has_attention_mask_name())
    options.attention_mask_name = pipeline.attention_mask_name();
  if (pipeline.has_position_ids_name())
    options.position_ids_name = pipeline.position_ids_name();
  return options;
}

Tensor GenerateLoadedPipeline(const LoadedRuntimePipeline &pipeline, RuntimeContext &context,
                              const RuntimeValueMap &feeds, const GenerationOptions *options) {
  const auto &session_options = pipeline.session_options();
  CpuExecutor *previous = context.cpu_executor();
  const auto executor =
      previous != nullptr && !session_options.cpu_execution
          ? std::shared_ptr<CpuExecutor>{}
          : GlobalCpuExecutorRegistry().Acquire(pipeline.prefill_session().cpu_execution_policy());
  const ScopeExit restore([&context, previous]() noexcept { context.set_cpu_executor(previous); });
  const CpuExecutorScope scope(executor ? executor.get() : previous);
  context.set_cpu_executor(executor ? executor.get() : previous);
  return options == nullptr
             ? Generate(pipeline.pipeline(), context, feeds, session_options)
             : Generate(pipeline.pipeline(), context, feeds, *options, session_options);
}

} // namespace

static Tensor GenerateImpl(const GenerationPipelineView &pipeline, RuntimeContext &context,
                           const RuntimeValueMap &prefill_feeds,
                           const RuntimeValueMap &embedding_feeds, const GenerationOptions &options,
                           RuntimeSessionOptions session_options,
                           const RuntimeValueMap &unscoped_feeds);

static Tensor GeneratePipeline(const GenerationPipelineView &pipeline, RuntimeContext &context,
                               const RuntimeValueMap &feeds, const GenerationOptions &options,
                               RuntimeSessionOptions session_options) {
  EXT_ENFORCE_INVALID(pipeline.prefill != nullptr, "Generate: prefill model is required.");
  RuntimeValueMap prefill_feeds = ScopeFeeds(*pipeline.prefill, feeds);
  RuntimeValueMap embedding_feeds =
      pipeline.embedding == nullptr ? RuntimeValueMap{} : ScopeFeeds(*pipeline.embedding, feeds);
  for (const auto &[name, value] : feeds) {
    (void)value;
    EXT_ENFORCE_INVALID(prefill_feeds.contains(name) || embedding_feeds.contains(name),
                        "Generate: unknown feed '", name, "'.");
  }
  return GenerateImpl(pipeline, context, prefill_feeds, embedding_feeds, options,
                      std::move(session_options), feeds);
}

Tensor Generate(const ModelProto &model, RuntimeContext &context, const RuntimeValueMap &feeds,
                const GenerationOptions &options, RuntimeSessionOptions session_options) {
  return GeneratePipeline(GenerationPipelineView{.prefill = &model}, context, feeds, options,
                          std::move(session_options));
}

Tensor Generate(const PipelineProto &pipeline, RuntimeContext &context,
                const RuntimeValueMap &feeds, RuntimeSessionOptions session_options) {
  return Generate(pipeline, context, feeds, PipelineOptions(pipeline), std::move(session_options));
}

Tensor Generate(const PipelineProto &pipeline, RuntimeContext &context,
                const RuntimeValueMap &feeds, const GenerationOptions &options,
                RuntimeSessionOptions session_options) {
  EXT_ENFORCE_INVALID(pipeline.has_prefill(), "Generate: prefill model is required.");
  EXT_ENFORCE_INVALID(!pipeline.has_decode(),
                      "Generate: a separate decode model is not supported.");
  const GenerationPipelineView view{.prefill = &pipeline.prefill(),
                                    .embedding =
                                        pipeline.has_embedding() ? &pipeline.embedding() : nullptr};
  return GeneratePipeline(view, context, feeds, options, std::move(session_options));
}

Tensor Generate(const ModelProto &prefill, const ModelProto *embedding, RuntimeContext &context,
                const RuntimeValueMap &feeds, const GenerationOptions &options,
                RuntimeSessionOptions session_options) {
  return GeneratePipeline(GenerationPipelineView{.prefill = &prefill, .embedding = embedding},
                          context, feeds, options, std::move(session_options));
}

Tensor Generate(const LoadedRuntimePipeline &pipeline, RuntimeContext &context,
                const RuntimeValueMap &feeds) {
  return GenerateLoadedPipeline(pipeline, context, feeds, nullptr);
}

Tensor Generate(const LoadedRuntimePipeline &pipeline, RuntimeContext &context,
                const RuntimeValueMap &feeds, const GenerationOptions &options) {
  return GenerateLoadedPipeline(pipeline, context, feeds, &options);
}

static Tensor GenerateImpl(const GenerationPipelineView &pipeline, RuntimeContext &context,
                           const RuntimeValueMap &prefill_feeds,
                           const RuntimeValueMap &embedding_feeds, const GenerationOptions &options,
                           RuntimeSessionOptions session_options,
                           const RuntimeValueMap &unscoped_feeds) {
  EXT_ENFORCE_INVALID(pipeline.prefill != nullptr, "Generate: prefill model is required.");
  const ModelProto &model = *pipeline.prefill;
  EXT_ENFORCE_INVALID(options.max_new_tokens >= 0, "Generate: max_new_tokens must be nonnegative.");
  EXT_ENFORCE_INVALID(options.max_logit_bytes == 0 || options.on_step,
                      "Generate: raw-logit capture requires an on_step callback.");
  EXT_ENFORCE_INVALID(std::isfinite(options.temperature) && options.temperature >= 0.,
                      "Generate: temperature must be finite and nonnegative.");
  EXT_ENFORCE_INVALID(!options.eos_token_id || *options.eos_token_id >= 0,
                      "Generate: eos_token_id must be nonnegative.");
  EXT_ENFORCE_INVALID(!options.pad_token_id || *options.pad_token_id >= 0,
                      "Generate: pad_token_id must be nonnegative.");
  const bool embedded = pipeline.embedding != nullptr;
  EXT_ENFORCE_INVALID(embedded ? HasInput(*pipeline.embedding, options.input_ids_name)
                               : HasInput(model, options.input_ids_name),
                      "Generate: input_ids must name a graph input.");
  EXT_ENFORCE_INVALID(
      std::any_of(model.graph().output().begin(), model.graph().output().end(),
                  [&](const auto &output) { return output.name() == options.logits_name; }),
      "Generate: logits must name a graph output.");
  if (embedded)
    ValidateGraph(*pipeline.embedding);
  const auto check_destinations = [&](const ModelProto &stage) {
    for (const auto &output : stage.graph().output()) {
      const std::string &destination = output.name();
      EXT_ENFORCE_INVALID(destination != options.input_ids_name &&
                              destination != options.attention_mask_name &&
                              destination != options.position_ids_name,
                          "Generate: graph output cannot replace generated inputs.");
    }
  };
  if (embedded) {
    check_destinations(*pipeline.embedding);
    for (const auto &output : pipeline.embedding->graph().output()) {
      EXT_ENFORCE_INVALID(HasInput(model, output.name()),
                          "Generate: embedding output must name a decoder input.");
    }
    ValidatePipelineValueNames(*pipeline.embedding, model, context);
  }
  for (const auto &binding : model.graph().persistent_bindings()) {
    if (embedded)
      for (const auto &output : pipeline.embedding->graph().output()) {
        EXT_ENFORCE_INVALID(output.name() != binding.input_name(),
                            "Generate: embedding output cannot replace a persistent input.");
      }
  }
  for (const auto &[name, value] : prefill_feeds) {
    (void)value;
    EXT_ENFORCE_INVALID(HasInput(model, name), "Generate: unknown feed '", name,
                        "' for the language model.");
  }
  for (const auto &[name, value] : embedding_feeds) {
    (void)value;
    EXT_ENFORCE_INVALID(embedded && HasInput(*pipeline.embedding, name), "Generate: unknown feed '",
                        name, "' for the embedding model.");
  }
  const RuntimeValueMap &prompt_feeds = embedded ? embedding_feeds : prefill_feeds;
  const Tensor &prompt = IntMatrix(prompt_feeds, options.input_ids_name);
  const int64_t batch = prompt.shape[0], prompt_length = prompt.shape[1];
  EXT_ENFORCE_INVALID(options.max_new_tokens <= std::numeric_limits<int64_t>::max() - prompt_length,
                      "Generate: requested sequence length overflows.");
  const int64_t max_length = prompt_length + options.max_new_tokens;
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(max_length) <=
                              std::numeric_limits<size_t>::max() / sizeof(int64_t) / batch &&
                          max_length <= std::numeric_limits<int64_t>::max() / batch,
                      "Generate: requested output size overflows.");
  std::vector<int64_t> tokens = IntValues(prompt);
  EXT_ENFORCE_INVALID(std::all_of(tokens.begin(), tokens.end(), [](auto id) { return id >= 0; }),
                      "Generate: token IDs must be nonnegative.");
  const bool has_mask = HasInput(model, options.attention_mask_name) ||
                        (embedded && HasInput(*pipeline.embedding, options.attention_mask_name));
  const bool has_positions = HasInput(model, options.position_ids_name) ||
                             (embedded && HasInput(*pipeline.embedding, options.position_ids_name));
  EXT_ENFORCE_INVALID(options.position_axes >= 0,
                      "Generate: position axis count must be nonnegative.");
  int64_t axes = options.position_axes;
  if (has_positions) {
    bool rank2 = false, rank3 = false;
    axes = PositionAxes(model, options.position_ids_name, axes, rank2, rank3);
    if (embedded)
      axes = PositionAxes(*pipeline.embedding, options.position_ids_name, axes, rank2, rank3);
    EXT_ENFORCE_INVALID(!rank2 || !rank3, "Generate: conflicting position input ranks.");
    EXT_ENFORCE_INVALID(!rank3 || axes > 0,
                        "Generate: dynamic position axis count requires position_axes.");
  }
  const bool multi_axis = axes > 0;
  EXT_ENFORCE_INVALID(!multi_axis ||
                          (axes <= std::numeric_limits<int64_t>::max() / batch / max_length &&
                           static_cast<uint64_t>(axes) <= std::numeric_limits<size_t>::max() /
                                                              sizeof(int64_t) / batch / max_length),
                      "Generate: requested position size overflows.");
  EXT_ENFORCE_INVALID((!has_mask || options.attention_mask_name != options.input_ids_name) &&
                          (!has_positions || (options.position_ids_name != options.input_ids_name &&
                                              (!has_mask || options.position_ids_name !=
                                                                options.attention_mask_name))),
                      "Generate: token, mask and position input names must be distinct.");
  std::vector<int64_t> mask(tokens.size(), 1);
  const auto control_feeds = [&](const std::string &name) -> const RuntimeValueMap & {
    return embedded && HasInput(*pipeline.embedding, name) && embedding_feeds.contains(name)
               ? embedding_feeds
               : prefill_feeds;
  };
  if (has_mask &&
      control_feeds(options.attention_mask_name).contains(options.attention_mask_name)) {
    const auto &tensor =
        IntMatrix(control_feeds(options.attention_mask_name), options.attention_mask_name);
    EXT_ENFORCE_INVALID(tensor.shape == prompt.shape, "Generate: attention mask shape mismatch.");
    mask = IntValues(tensor);
  }
  std::vector<int64_t> positions(tokens.size());
  for (int64_t row = 0; row < batch; ++row) {
    int64_t position = 0;
    bool started = false;
    for (int64_t col = 0; col < prompt_length; ++col) {
      const auto index = row * prompt_length + col;
      EXT_ENFORCE_INVALID(mask[index] == 0 || mask[index] == 1,
                          "Generate: attention mask must contain zeros and ones.");
      EXT_ENFORCE_INVALID(!started || mask[index] == 1,
                          "Generate: only left-padded prompts are supported.");
      started = started || mask[index] == 1;
      positions[index] = mask[index] ? position++ : 0;
    }
    EXT_ENFORCE_INVALID(started, "Generate: each prompt must contain an unmasked token.");
  }
  if (multi_axis) {
    const auto text_positions = positions;
    positions.reserve(text_positions.size() * static_cast<size_t>(axes));
    for (int64_t axis = 1; axis < axes; ++axis)
      positions.insert(positions.end(), text_positions.begin(), text_positions.end());
  }
  const RuntimeValueMap &position_feeds = control_feeds(options.position_ids_name);
  if (has_positions && position_feeds.contains(options.position_ids_name)) {
    const auto &value = position_feeds.at(options.position_ids_name);
    EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kTensor,
                        "Generate: position IDs must be a tensor.");
    const Tensor &tensor = value.tensor;
    EXT_ENFORCE_INVALID(tensor.data_type == DataType::INT64 &&
                            tensor.shape ==
                                (multi_axis ? Shape{axes, batch, prompt_length} : prompt.shape),
                        "Generate: position IDs shape or type mismatch.");
    ValidateStorage(tensor);
    positions = IntValues(tensor);
  }
  for (const auto position : positions)
    EXT_ENFORCE_INVALID(position >= 0 && position <= std::numeric_limits<int64_t>::max() -
                                                         options.max_new_tokens,
                        "Generate: invalid or overflowing position ID.");
  if (options.max_new_tokens == 0)
    return Tensor::From<int64_t>(options.input_ids_name, prompt.shape, tokens);

  if (embedded) {
    EXT_ENFORCE_INVALID(context.cpu_executor() != nullptr,
                        "Generate: a context with a CPU executor is required for split models.");
    if (session_options.cpu_execution_counters)
      context.cpu_executor()->EnableCounters();
    session_options.cpu_execution.reset();
    // Sizes the shared arena before any session captures its allocator.
    const ExecutionPlan *largest = &context.GetExecutionPlan(model);
    const auto include_plan = [&](const ModelProto &graph_model) {
      const auto &plan = context.GetExecutionPlan(graph_model);
      if (plan.arena_capacity() > largest->arena_capacity())
        largest = &plan;
    };
    include_plan(*pipeline.embedding);
    context.BindExecutionPlan(*largest);
  }
  RuntimeValueMap current;
  for (const auto &[name, value] : prefill_feeds)
    current.emplace(name, value.BorrowView());
  RuntimeValueMap embedding_current;
  for (const auto &[name, value] : embedding_feeds)
    embedding_current.emplace(name, value.BorrowView());
  std::unique_ptr<RuntimeSession> embedding_session;
  if (embedded) {
    embedding_session = std::make_unique<RuntimeSession>(*pipeline.embedding, session_options);
  }
  const bool persistent = !model.graph().persistent_bindings().empty();
  RuntimeValueMap initial;
  if (persistent) {
    StructTypeCatalogue initial_types;
    initial_types.Build(model);
    std::shared_ptr<const QuantizationParameterCatalogue> initial_parameters;
    for (const auto &binding : model.graph().persistent_bindings()) {
      const std::string &name = binding.input_name();
      EXT_ENFORCE_INVALID(name != options.input_ids_name &&
                              (!has_mask || name != options.attention_mask_name) &&
                              (!has_positions || name != options.position_ids_name),
                          "Generate: token, mask and position inputs cannot be persistent.");
      const auto value = unscoped_feeds.find(name);
      if (value != unscoped_feeds.end()) {
        RuntimeValue scoped = value->second.BorrowView();
        BindParameterReferences(scoped, model, initial_types, initial_parameters);
        initial.emplace(name, scoped.DeepCopy());
      }
      current.erase(name);
    }
  }
  RuntimeSession session(model, std::move(initial), session_options);
  std::mt19937_64 random(options.seed ? *options.seed : std::random_device{}());
  std::vector<bool> finished(static_cast<size_t>(batch), false);
  std::vector<int64_t> next(static_cast<size_t>(batch));
  std::vector<int64_t> next_positions(static_cast<size_t>(batch * (multi_axis ? axes : 1)));
  if (multi_axis) {
    for (int64_t row = 0; row < batch; ++row) {
      int64_t maximum = 0;
      for (int64_t axis = 0; axis < axes; ++axis)
        for (int64_t col = 0; col < prompt_length; ++col)
          if (mask[row * prompt_length + col])
            maximum = std::max(maximum, positions[(axis * batch + row) * prompt_length + col]);
      for (int64_t axis = 0; axis < axes; ++axis)
        next_positions[axis * batch + row] = maximum;
    }
  }
  std::vector<int64_t> next_mask(static_cast<size_t>(batch));
  size_t remaining_logit_bytes = options.max_logit_bytes;
  int64_t length = prompt_length;
  for (int64_t step = 0; step < options.max_new_tokens; ++step) {
    const auto started_at = options.on_step ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    if (step > 0) {
      AppendColumn(mask, next_mask);
      AppendColumn(positions, next_positions);
    }
    const bool decode = persistent && step > 0;
    const Shape shape{batch, decode ? 1 : length};
    const auto put_control = [&](const std::string &name, RuntimeValue value) {
      if (HasInput(model, name))
        current.insert_or_assign(name, value.BorrowView());
      if (embedded && HasInput(*pipeline.embedding, name))
        embedding_current.insert_or_assign(name, value.BorrowView());
    };
    const Shape position_shape = multi_axis ? Shape{axes, batch, decode ? 1 : length} : shape;
    if (has_mask)
      put_control(
          options.attention_mask_name,
          RuntimeValue(Tensor::From<int64_t>(options.attention_mask_name, {batch, length}, mask)
                           .RetainStorage()));
    if (has_positions)
      put_control(options.position_ids_name,
                  RuntimeValue(Tensor::From<int64_t>(options.position_ids_name, position_shape,
                                                     decode ? next_positions : positions)
                                   .RetainStorage()));
    put_control(
        options.input_ids_name,
        RuntimeValue(Tensor::From<int64_t>(options.input_ids_name, shape, decode ? next : tokens)
                         .RetainStorage()));
    if (embedded) {
      RunGraph(*pipeline.embedding, context, embedding_current, *embedding_session);
      for (const ValueInfoProto &output : pipeline.embedding->graph().output())
        current.insert_or_assign(output.name(), embedding_current.at(output.name()).BorrowView());
    }
    Tensor logits;
    if (persistent) {
      RuntimeValueMap model_feeds;
      for (const auto &input : model.graph().input()) {
        const auto found = current.find(input.name());
        if (found != current.end())
          model_feeds.emplace(input.name(), found->second.BorrowView());
      }
      auto outputs = session.Run(context, model_feeds);
      auto it = outputs.find(options.logits_name);
      EXT_ENFORCE_INVALID(it != outputs.end() && it->second.kind == RuntimeValue::Kind::kTensor,
                          "Generate: logits output must be a tensor.");
      logits = std::move(it->second.tensor);
    } else {
      context.ResetInvocation();
      for (const auto &input : model.graph().input()) {
        const auto found = current.find(input.name());
        if (found == current.end())
          continue;
        const auto &value = found->second;
        if (input.type().has_sequence_type() &&
            input.type().sequence_type().elem_type().has_tensor_type() &&
            value.kind == RuntimeValue::Kind::kSequence &&
            std::all_of(value.elements.begin(), value.elements.end(), [](const auto &element) {
              return element.kind == RuntimeValue::Kind::kTensor;
            })) {
          Tensors tensors;
          for (const auto &element : value.elements)
            tensors.push_back(element.tensor.BorrowView());
          context.PutSequence(
              input.name(),
              Sequence(input.name(),
                       input.type().sequence_type().elem_type().tensor_type().elem_type(),
                       std::move(tensors)));
        } else {
          context.PutValue(input.name(), value.BorrowView(), RuntimeEventKind::kInput);
        }
      }
      session.Run(context);
      EXT_ENFORCE_INVALID(context.Has(options.logits_name), "Generate: missing logits tensor.");
      logits = std::move(context.Get(options.logits_name)).RetainStorage();
    }
    EXT_ENFORCE_INVALID(
        (logits.shape.size() == 2 || logits.shape.size() == 3) && logits.shape[0] == batch &&
            logits.shape.back() > 0 && (logits.shape.size() == 2 || logits.shape[1] > 0),
        "Generate: logits must have shape [batch, vocabulary] or [batch, sequence, vocabulary].");
    EXT_ENFORCE_INVALID(logits.data_type == DataType::FLOAT ||
                            logits.data_type == DataType::DOUBLE ||
                            IsHalfPrecision(logits.data_type),
                        "Generate: logits must have a floating-point type.");
    ValidateStorage(logits);
    const int64_t vocabulary = logits.shape.back();
    size_t row_bytes = 0;
    bool capture_logits = false;
    if (options.on_step && remaining_logit_bytes > 0) {
      row_bytes = PackedByteSize(logits.data_type, vocabulary);
      capture_logits = row_bytes <= remaining_logit_bytes / static_cast<size_t>(batch);
    }
    std::optional<Tensor> original_logits;
    if (capture_logits && IsHalfPrecision(logits.data_type)) {
      original_logits.emplace(std::move(logits));
      logits = PromoteToFloat32(*original_logits);
    } else if (IsHalfPrecision(logits.data_type)) {
      logits = PromoteToFloat32(logits);
    }
    EXT_ENFORCE_INVALID(!options.eos_token_id || *options.eos_token_id < vocabulary,
                        "Generate: eos_token_id is outside the vocabulary.");
    EXT_ENFORCE_INVALID(!options.pad_token_id || *options.pad_token_id < vocabulary,
                        "Generate: pad_token_id is outside the vocabulary.");
    const int64_t stride = logits.shape.product() / batch;
    for (int64_t row = 0; row < batch; ++row) {
      next_mask[row] = finished[row] ? 0 : 1;
      next[row] = finished[row] ? options.pad_token_id.value_or(*options.eos_token_id)
                                : Sample(logits, (row + 1) * stride - vocabulary, vocabulary,
                                         options.temperature, random);
      if (multi_axis) {
        for (int64_t axis = 0; axis < axes; ++axis)
          ++next_positions[axis * batch + row];
      } else {
        next_positions[row] = positions[row * length + length - 1] + 1;
      }
      if (options.eos_token_id && next[row] == *options.eos_token_id)
        finished[row] = true;
    }
    AppendColumn(tokens, next);
    ++length;
    if (options.on_step) {
      GenerationStep observation;
      observation.index = step;
      observation.phase = step == 0 ? GenerationPhase::kPrefill : GenerationPhase::kDecode;
      observation.started_at = started_at;
      observation.completed_at = std::chrono::steady_clock::now();
      observation.token_ids = next;
      if (capture_logits) {
        const Tensor &raw = original_logits ? *original_logits : logits;
        const size_t snapshot_bytes = row_bytes * static_cast<size_t>(batch);
        Tensor snapshot;
        snapshot.name = options.logits_name;
        snapshot.data_type = raw.data_type;
        snapshot.shape = {batch, vocabulary};
        snapshot.data.resize(snapshot_bytes);
        const size_t stride_bytes = raw.size_bytes() / static_cast<size_t>(batch);
        for (size_t row = 0; row < static_cast<size_t>(batch); ++row)
          std::memcpy(snapshot.mutable_bytes() + row * row_bytes,
                      raw.bytes() + (row + 1) * stride_bytes - row_bytes, row_bytes);
        observation.raw_logits.emplace(std::move(snapshot));
        remaining_logit_bytes -= snapshot_bytes;
      }
      if (!options.on_step(observation))
        break;
    }
    if (std::all_of(finished.begin(), finished.end(), [](bool value) { return value; }) ||
        step + 1 == options.max_new_tokens)
      break;
  }
  return Tensor::From<int64_t>(options.input_ids_name, {batch, length}, tokens);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
