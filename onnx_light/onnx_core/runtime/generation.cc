// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/generation.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/run_nodes_internal.h"
#include "onnx_core/runtime/persistent_value_state.h"
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

constexpr const char *kGenerationPrepared = "onnx_light.generation_prepared";

bool StandardDomain(const std::string &domain) { return domain.empty() || domain == "ai.onnx"; }

bool GenerationFloat(int64_t type) {
  return type == TensorProto::FLOAT || type == TensorProto::DOUBLE ||
         type == TensorProto::FLOAT16 || type == TensorProto::BFLOAT16;
}

// Scans nested scopes conservatively: even a shadowed name is reserved.
void GenerationNames(const GraphProto &graph, std::unordered_set<std::string> &names) {
  for (const auto *values : {&graph.input(), &graph.output(), &graph.value_info()})
    for (const auto &value : *values)
      names.insert(value.name());
  for (const auto &value : graph.initializer())
    names.insert(value.name());
  for (const auto &value : graph.sparse_initializer()) {
    names.insert(value.values().name());
    names.insert(value.indices().name());
  }
  for (const auto &value : graph.extended_initializer()) {
    if (value.has_encoded_value())
      names.insert(value.encoded_value().name());
    if (value.has_paged_cache())
      names.insert(value.paged_cache().name());
  }
  for (const auto &binding : graph.persistent_bindings()) {
    names.insert(binding.input_name());
    names.insert(binding.output_name());
  }
  for (const auto &node : graph.node()) {
    names.insert(node.name());
    for (const auto &name : node.input())
      names.insert(name);
    for (const auto &name : node.output())
      names.insert(name);
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g())
        GenerationNames(attribute.g(), names);
      for (const auto &nested : attribute.graphs())
        GenerationNames(nested, names);
    }
  }
}

struct GenerationTensorInfo {
  int rank = -1;
  int32_t type = TensorProto::UNDEFINED;
};

GenerationTensorInfo GenerationInfo(const GraphProto &graph, const std::string &name) {
  GenerationTensorInfo result;
  const auto merge = [&](int rank, int32_t type) {
    EXT_ENFORCE_INVALID((rank < 0 || result.rank < 0 || result.rank == rank) &&
                            (!type || !result.type || result.type == type),
                        "PrepareGenerationModel: conflicting tensor metadata for ", name, ".");
    if (rank >= 0)
      result.rank = rank;
    if (type)
      result.type = type;
  };
  for (const auto *values : {&graph.input(), &graph.output(), &graph.value_info()})
    for (const auto &value : *values)
      if (value.name() == name) {
        EXT_ENFORCE_INVALID(value.type().has_tensor_type(),
                            "PrepareGenerationModel: head values must be tensors.");
        const auto &tensor = value.type().tensor_type();
        merge(tensor.has_shape() ? tensor.shape().dim_size() : -1, tensor.elem_type());
      }
  for (const auto &value : graph.initializer())
    if (value.name() == name)
      merge(value.dims_size(), value.data_type());
  for (const auto &value : graph.extended_initializer())
    if (value.has_encoded_value() && value.encoded_value().name() == name) {
      const auto &type = value.encoded_value().logical_type();
      EXT_ENFORCE_INVALID(type.has_tensor_type(),
                          "PrepareGenerationModel: weight must be a tensor.");
      const auto &tensor = type.tensor_type();
      merge(tensor.has_shape() ? tensor.shape().dim_size() : -1, tensor.elem_type());
    }
  return result;
}

template <class Proto> void ClearGenerationExecutionMetadata(Proto &proto) {
  auto &metadata = *proto.mutable_metadata_props();
  for (size_t index = metadata.size(); index > 0; --index) {
    const std::string &key = metadata[index - 1].key();
    if (key == "onnx_light.inplace_reuse" || key == "onnx_light.release_after" ||
        key == "onnx_light.release_after_shape_tag" || key == "onnx_light.not_used_after" ||
        key == "onnx_light.lifetime_names_encoding" ||
        std::string_view(key).find("onnx_light.peak_memory") == 0) {
      // RepeatedProtoField deliberately has no arbitrary erase.
      for (size_t next = index; next < metadata.size(); ++next)
        metadata.SwapElements(static_cast<int>(next - 1), static_cast<int>(next));
      metadata.RemoveLast();
    }
  }
}

bool HasInput(const ModelProto &model, const std::string &name) {
  return std::any_of(model.graph().input().begin(), model.graph().input().end(),
                     [&](const auto &input) { return input.name() == name; });
}

void ValidateStorage(const Tensor &tensor) {
  const int64_t count = tensor.shape.product(0, tensor.shape.size(), "Generate: tensor storage");
  if (tensor.data_type == DataType::STRING) {
    EXT_ENFORCE_INVALID(tensor.AsStrings().size() == static_cast<uint64_t>(count),
                        "Generate: invalid tensor storage.");
    return;
  }
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

void ValidateGraph(const ModelProto &model, bool persistent_values = false) {
  std::unordered_set<std::string> destinations;
  for (const auto &output : model.graph().output()) {
    EXT_ENFORCE_INVALID(!output.name().empty() &&
                            (output.type().has_tensor_type() ||
                             (persistent_values && (output.type().has_struct_type() ||
                                                    output.type().has_sequence_type())) ||
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
  const ModelProto *decode = nullptr;
};

const ValueInfoProto &GenerationValue(const utils::RepeatedProtoField<ValueInfoProto> &values,
                                      const std::string &name) {
  const ValueInfoProto *found = nullptr;
  for (const auto &value : values)
    if (value.name() == name) {
      EXT_ENFORCE_INVALID(found == nullptr, "Generate: duplicate graph value '", name, "'.");
      found = &value;
    }
  EXT_ENFORCE_INVALID(found != nullptr, "Generate: missing graph value '", name, "'.");
  return *found;
}

bool CompatibleGenerationTypes(const TypeProto &left, const TypeProto &right,
                               const StructTypeCatalogue &left_types,
                               const StructTypeCatalogue &right_types, int sequence_axis = -1,
                               size_t depth = 0) {
  if (depth > RuntimeValue::kMaxDepth || left.value_case() != right.value_case())
    return false;
  if (left.has_tensor_type()) {
    const auto &a = left.tensor_type();
    const auto &b = right.tensor_type();
    if (a.elem_type() != b.elem_type())
      return false;
    for (const auto *tensor : {&a, &b})
      for (const auto &dim : tensor->shape().dim())
        if (dim.has_dim_value() && dim.dim_value() < 0)
          return false;
    if (!a.has_shape() || !b.has_shape())
      return true;
    if (a.shape().dim_size() != b.shape().dim_size())
      return false;
    for (size_t i = 0; i < a.shape().dim().size(); ++i)
      if (static_cast<int>(i) != sequence_axis && a.shape().dim(i).has_dim_value() &&
          b.shape().dim(i).has_dim_value() &&
          a.shape().dim(i).dim_value() != b.shape().dim(i).dim_value())
        return false;
    return true;
  }
  if (left.has_sequence_type())
    return CompatibleGenerationTypes(left.sequence_type().elem_type(),
                                     right.sequence_type().elem_type(), left_types, right_types, -1,
                                     depth + 1);
  if (!left.has_struct_type())
    return false;
  StructTypeProto a = left_types.Resolve(left.struct_type());
  StructTypeProto b = right_types.Resolve(right.struct_type());
  if (a.type_id() != b.type_id() || a.kind_case() != b.kind_case())
    return false;
  if (a.has_structure()) {
    if (a.structure().field_size() != b.structure().field_size())
      return false;
    for (size_t i = 0; i < a.structure().field().size(); ++i) {
      const auto &af = a.structure().field(i);
      const auto &bf = b.structure().field(i);
      if (af.has_type() != bf.has_type() ||
          (af.has_type() && !CompatibleGenerationTypes(af.type(), bf.type(), left_types,
                                                       right_types, -1, depth + 1)))
        return false;
      if (af.has_type()) {
        a.mutable_structure()->mutable_field(static_cast<int>(i))->clear_type();
        b.mutable_structure()->mutable_field(static_cast<int>(i))->clear_type();
      }
    }
  } else if (a.has_array()) {
    if (!CompatibleGenerationTypes(a.array().element_type(), b.array().element_type(), left_types,
                                   right_types, -1, depth + 1))
      return false;
    a.mutable_array()->clear_element_type();
    b.mutable_array()->clear_element_type();
  }
  return a.Equals(b);
}

void ValidateGenerationTensor(const ValueInfoProto &value, int64_t type, const Shape &shape,
                              int varying_axis = -1, int64_t final_length = -1) {
  EXT_ENFORCE_INVALID(value.type().has_tensor_type(), "Generate: '", value.name(),
                      "' must be a tensor.");
  const auto &tensor = value.type().tensor_type();
  EXT_ENFORCE_INVALID(tensor.elem_type() == type, "Generate: dtype mismatch for '", value.name(),
                      "'.");
  if (!tensor.has_shape())
    return;
  EXT_ENFORCE_INVALID(static_cast<size_t>(tensor.shape().dim_size()) == shape.size(),
                      "Generate: rank mismatch for '", value.name(), "'.");
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto &dim = tensor.shape().dim(i);
    EXT_ENFORCE_INVALID(!dim.has_dim_value() ||
                            (dim.dim_value() == shape[i] &&
                             (static_cast<int>(i) != varying_axis || final_length == shape[i])),
                        "Generate: shape mismatch for '", value.name(), "'.");
  }
}

void ValidateSplitGeneration(const GenerationPipelineView &pipeline,
                             const GenerationOptions &options, int64_t batch, int64_t prompt_length,
                             int64_t axes, int64_t largest_eos_token_id) {
  const auto &prefill = *pipeline.prefill;
  const auto &decode = *pipeline.decode;
  StructTypeCatalogue prefill_types, decode_types;
  prefill_types.Build(prefill);
  decode_types.Build(decode);
  VerifyPersistentBindings(&prefill_types, prefill.graph());
  VerifyPersistentBindings(&decode_types, decode.graph());
  EXT_ENFORCE_INVALID(prefill.graph().persistent_bindings().size() ==
                          decode.graph().persistent_bindings().size(),
                      "Generate: prefill and decode require matching persistent bindings.");
  for (const auto &binding : prefill.graph().persistent_bindings()) {
    EXT_ENFORCE_INVALID(std::any_of(decode.graph().persistent_bindings().begin(),
                                    decode.graph().persistent_bindings().end(),
                                    [&](const auto &other) {
                                      return binding.input_name() == other.input_name() &&
                                             binding.output_name() == other.output_name();
                                    }),
                        "Generate: prefill and decode persistent binding names differ.");
    for (const auto *values : {&prefill.graph().input(), &prefill.graph().output()}) {
      const bool input = values == &prefill.graph().input();
      const std::string &name = input ? binding.input_name() : binding.output_name();
      const auto &a = GenerationValue(*values, name).type();
      const auto &b =
          GenerationValue(input ? decode.graph().input() : decode.graph().output(), name).type();
      EXT_ENFORCE_INVALID(CompatibleGenerationTypes(a, b, prefill_types, decode_types),
                          "Generate: incompatible persistent binding type for '", name, "'.");
    }
    EXT_ENFORCE_INVALID(binding.input_name() != options.input_ids_name &&
                            binding.input_name() != options.attention_mask_name &&
                            binding.input_name() != options.position_ids_name,
                        "Generate: token, mask and position inputs cannot be persistent.");
    EXT_ENFORCE_INVALID(CompatibleGenerationTypes(
                            GenerationValue(prefill.graph().output(), binding.output_name()).type(),
                            GenerationValue(decode.graph().input(), binding.input_name()).type(),
                            prefill_types, decode_types),
                        "Generate: prefill persistent output is incompatible with decode input.");
  }
  ValidateGraph(prefill, true);
  ValidateGraph(decode, true);
  for (const auto &name :
       {options.input_ids_name, options.attention_mask_name, options.position_ids_name}) {
    EXT_ENFORCE_INVALID(HasInput(prefill, name) == HasInput(decode, name),
                        "Generate: prefill and decode generated input names differ: '", name, "'.");
    if (!HasInput(prefill, name))
      continue;
    const bool positions = name == options.position_ids_name;
    const bool mask = name == options.attention_mask_name;
    const Shape prefill_shape =
        positions && axes > 0 ? Shape{axes, batch, prompt_length} : Shape{batch, prompt_length};
    const Shape decode_shape =
        positions && axes > 0 ? Shape{axes, batch, 1} : Shape{batch, mask ? prompt_length + 1 : 1};
    const int sequence_axis = positions && axes > 0 ? 2 : 1;
    const auto &a = GenerationValue(prefill.graph().input(), name);
    const auto &b = GenerationValue(decode.graph().input(), name);
    EXT_ENFORCE_INVALID(
        CompatibleGenerationTypes(a.type(), b.type(), prefill_types, decode_types, sequence_axis),
        "Generate: incompatible generated input '", name, "'.");
    ValidateGenerationTensor(a, TensorProto::INT64, prefill_shape);
    ValidateGenerationTensor(b, TensorProto::INT64, decode_shape, mask ? sequence_axis : -1,
                             prompt_length + std::max<int64_t>(1, options.max_new_tokens - 1));
  }
  if (pipeline.embedding != nullptr)
    for (const auto &output : pipeline.embedding->graph().output()) {
      const auto &a = GenerationValue(prefill.graph().input(), output.name());
      const auto &b = GenerationValue(decode.graph().input(), output.name());
      const auto has_sequence = [](const ValueInfoProto &value) {
        return value.type().has_tensor_type() && value.type().tensor_type().has_shape() &&
               value.type().tensor_type().shape().dim_size() >= 3;
      };
      const bool sequence = has_sequence(a) || has_sequence(b);
      EXT_ENFORCE_INVALID(CompatibleGenerationTypes(a.type(), b.type(), prefill_types, decode_types,
                                                    sequence ? 1 : -1),
                          "Generate: incompatible embedding-connected language inputs '",
                          output.name(), "'.");
      for (const auto *value : {&a, &b}) {
        if (!sequence || !value->type().tensor_type().has_shape())
          continue;
        const auto &shape = value->type().tensor_type().shape();
        EXT_ENFORCE_INVALID((!shape.dim(0).has_dim_value() || shape.dim(0).dim_value() == batch) &&
                                (!shape.dim(1).has_dim_value() ||
                                 shape.dim(1).dim_value() == (value == &a ? prompt_length : 1)),
                            "Generate: invalid embedding-connected input sequence specialization.");
      }
    }
  const auto &prefill_logits = GenerationValue(prefill.graph().output(), options.logits_name);
  const auto &decode_logits = GenerationValue(decode.graph().output(), options.logits_name);
  const int logits_sequence_axis =
      prefill_logits.type().has_tensor_type() && prefill_logits.type().tensor_type().has_shape() &&
              prefill_logits.type().tensor_type().shape().dim_size() == 3
          ? 1
          : -1;
  EXT_ENFORCE_INVALID(CompatibleGenerationTypes(prefill_logits.type(), decode_logits.type(),
                                                prefill_types, decode_types, logits_sequence_axis),
                      "Generate: incompatible prefill and decode logits.");
  for (const auto *value : {&prefill_logits, &decode_logits}) {
    EXT_ENFORCE_INVALID(value->type().has_tensor_type() &&
                            GenerationFloat(value->type().tensor_type().elem_type()),
                        "Generate: logits must have a floating-point tensor type.");
    const auto &type = value->type().tensor_type();
    if (!type.has_shape())
      continue;
    const auto &shape = type.shape();
    EXT_ENFORCE_INVALID(shape.dim_size() == 2 || shape.dim_size() == 3,
                        "Generate: logits must have rank 2 or 3.");
    EXT_ENFORCE_INVALID(!shape.dim(0).has_dim_value() || shape.dim(0).dim_value() == batch,
                        "Generate: logits batch mismatch.");
    const auto &vocab = shape.dim(shape.dim_size() - 1);
    EXT_ENFORCE_INVALID(!vocab.has_dim_value() ||
                            (vocab.dim_value() > 0 && largest_eos_token_id < vocab.dim_value() &&
                             (!options.pad_token_id || *options.pad_token_id < vocab.dim_value())),
                        "Generate: invalid logits vocabulary or token ID.");
    if (shape.dim_size() == 3) {
      const auto &sequence = shape.dim(1);
      EXT_ENFORCE_INVALID(!sequence.has_dim_value() || sequence.dim_value() == 1 ||
                              (value == &prefill_logits && sequence.dim_value() == prompt_length),
                          "Generate: invalid logits sequence specialization.");
    }
  }
}

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

RuntimeValueMap ScopeFeeds(const ModelProto &model, const RuntimeValueMap &feeds,
                           bool include_persistent_inputs = true) {
  StructTypeCatalogue types;
  types.Build(model);
  std::shared_ptr<const QuantizationParameterCatalogue> parameters;
  RuntimeValueMap result;
  for (const auto &[name, value] : feeds) {
    if (!HasInput(model, name))
      continue;
    if (!include_persistent_inputs &&
        std::any_of(model.graph().persistent_bindings().begin(),
                    model.graph().persistent_bindings().end(),
                    [&](const auto &binding) { return binding.input_name() == name; }))
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
  options.eos_token_ids.assign(pipeline.eos_token_ids().begin(), pipeline.eos_token_ids().end());
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
  if (session_options.cpu_execution_counters)
    context.cpu_executor()->EnableCounters();
  return options == nullptr
             ? Generate(pipeline.pipeline(), context, feeds, session_options)
             : Generate(pipeline.pipeline(), context, feeds, *options, session_options);
}

} // namespace

static Tensor GenerateImpl(const GenerationPipelineView &pipeline, RuntimeContext &context,
                           const RuntimeValueMap &prefill_feeds,
                           const RuntimeValueMap &decode_feeds,
                           const RuntimeValueMap &embedding_feeds, const GenerationOptions &options,
                           RuntimeSessionOptions session_options,
                           const RuntimeValueMap &unscoped_feeds);

static Tensor GeneratePipeline(const GenerationPipelineView &pipeline, RuntimeContext &context,
                               const RuntimeValueMap &feeds, const GenerationOptions &options,
                               RuntimeSessionOptions session_options) {
  EXT_ENFORCE_INVALID(pipeline.prefill != nullptr, "Generate: prefill model is required.");
  RuntimeValueMap prefill_feeds = ScopeFeeds(*pipeline.prefill, feeds);
  RuntimeValueMap decode_feeds =
      pipeline.decode == nullptr ? RuntimeValueMap{} : ScopeFeeds(*pipeline.decode, feeds, false);
  RuntimeValueMap embedding_feeds =
      pipeline.embedding == nullptr ? RuntimeValueMap{} : ScopeFeeds(*pipeline.embedding, feeds);
  for (const auto &[name, value] : feeds) {
    (void)value;
    EXT_ENFORCE_INVALID(prefill_feeds.contains(name) || decode_feeds.contains(name) ||
                            embedding_feeds.contains(name),
                        "Generate: unknown feed '", name, "'.");
  }
  return GenerateImpl(pipeline, context, prefill_feeds, decode_feeds, embedding_feeds, options,
                      std::move(session_options), feeds);
}

void PrepareGenerationModel(ModelProto &model, const std::string &logits_name) {
  EXT_ENFORCE_INVALID(model.has_graph() && !logits_name.empty(),
                      "PrepareGenerationModel: requires a graph and logits name.");
  for (const auto &entry : model.metadata_props())
    EXT_ENFORCE_INVALID(entry.key() != kGenerationPrepared,
                        "PrepareGenerationModel: model is already prepared.");
  const GraphProto &graph = model.graph();
  const ValueInfoProto *output = nullptr;
  for (const auto &value : graph.output())
    if (value.name() == logits_name) {
      EXT_ENFORCE_INVALID(!output, "PrepareGenerationModel: duplicate logits output.");
      output = &value;
    }
  EXT_ENFORCE_INVALID(output, "PrepareGenerationModel: logits must be a graph output.");
  const auto logits_info = GenerationInfo(graph, logits_name);
  EXT_ENFORCE_INVALID(logits_info.rank == 3 && GenerationFloat(logits_info.type),
                      "PrepareGenerationModel: logits must have floating rank-three metadata.");
  int64_t opset = 0;
  for (const auto &import : model.opset_import())
    if (StandardDomain(import.domain())) {
      EXT_ENFORCE_INVALID(import.version() >= 11,
                          "PrepareGenerationModel: requires standard opset >= 11.");
      EXT_ENFORCE_INVALID(opset == 0 || opset == import.version(),
                          "PrepareGenerationModel: conflicting standard opset imports.");
      opset = import.version();
    }
  EXT_ENFORCE_INVALID(opset >= 11, "PrepareGenerationModel: requires standard opset >= 11.");
  for (const auto &function : model.functions())
    EXT_ENFORCE_INVALID(
        !StandardDomain(function.domain()) ||
            (function.name() != "MatMul" && function.name() != "Gather" &&
             function.name() != "Cast" && function.name() != "Identity"),
        "PrepareGenerationModel: local functions must not override head operators.");

  std::unordered_map<std::string, size_t> producers;
  for (size_t index = 0; index < graph.node().size(); ++index)
    for (const auto &name : graph.node(index).output())
      if (!name.empty())
        EXT_ENFORCE_INVALID(producers.emplace(name, index).second,
                            "PrepareGenerationModel: duplicate producer.");
  std::vector<size_t> chain;
  std::unordered_set<std::string> changed;
  std::string current = logits_name;
  int32_t expected_type = logits_info.type;
  size_t head = 0;
  while (true) {
    const auto producer = producers.find(current);
    EXT_ENFORCE_INVALID(producer != producers.end() && changed.insert(current).second,
                        "PrepareGenerationModel: unsupported or cyclic logits head.");
    const size_t index = producer->second;
    const NodeProto &node = graph.node(index);
    EXT_ENFORCE_INVALID(StandardDomain(node.domain()) && node.overload().empty() &&
                            node.output_size() == 1,
                        "PrepareGenerationModel: requires a standard single-output head.");
    const auto info = GenerationInfo(graph, current);
    EXT_ENFORCE_INVALID((info.rank < 0 || info.rank == 3) &&
                            (!info.type || info.type == expected_type),
                        "PrepareGenerationModel: inconsistent head metadata.");
    chain.push_back(index);
    if (node.op_type() == "MatMul") {
      EXT_ENFORCE_INVALID(node.input_size() == 2 && node.attribute().empty(),
                          "PrepareGenerationModel: unsupported MatMul.");
      for (const auto &name : node.input())
        EXT_ENFORCE_INVALID(!name.empty() && !changed.count(name) &&
                                (!producers.count(name) || producers.at(name) < index),
                            "PrepareGenerationModel: invalid MatMul dependency.");
      const auto lhs = GenerationInfo(graph, node.input(0));
      const auto rhs = GenerationInfo(graph, node.input(1));
      EXT_ENFORCE_INVALID(
          rhs.rank == 2 && (lhs.rank < 0 || lhs.rank == 3) && GenerationFloat(expected_type) &&
              (!lhs.type || lhs.type == expected_type) && (!rhs.type || rhs.type == expected_type),
          "PrepareGenerationModel: requires a floating [B,S,D] x [D,V] MatMul.");
      head = index;
      break;
    }
    EXT_ENFORCE_INVALID(node.input_size() == 1 &&
                            (node.op_type() == "Identity" || node.op_type() == "Cast"),
                        "PrepareGenerationModel: unsupported logits head.");
    if (node.op_type() == "Cast") {
      EXT_ENFORCE_INVALID(node.attribute_size() == 1,
                          "PrepareGenerationModel: unsupported Cast attributes.");
      int64_t to = TensorProto::UNDEFINED;
      for (const auto &attribute : node.attribute()) {
        EXT_ENFORCE_INVALID(attribute.name() == "to" && attribute.type() == AttributeProto::INT,
                            "PrepareGenerationModel: unsupported Cast attributes.");
        to = attribute.i();
      }
      auto source = GenerationInfo(graph, node.input(0));
      // Exporters often omit intermediate metadata; a MatMul weight still fixes its dtype.
      std::string source_name = node.input(0);
      size_t before = index;
      while (!source.type && producers.count(source_name)) {
        const size_t source_index = producers.at(source_name);
        EXT_ENFORCE_INVALID(source_index < before,
                            "PrepareGenerationModel: head must be topologically ordered.");
        const auto &source_node = graph.node(source_index);
        if (!StandardDomain(source_node.domain()))
          break;
        if (source_node.op_type() == "MatMul" && source_node.input_size() == 2) {
          source.type = GenerationInfo(graph, source_node.input(1)).type;
          break;
        }
        if (source_node.op_type() == "Cast" && source_node.attribute_size() == 1) {
          const auto &attribute = source_node.attribute(0);
          if (attribute.name() == "to" && attribute.type() == AttributeProto::INT &&
              GenerationFloat(attribute.i()))
            source.type = static_cast<int32_t>(attribute.i());
          break;
        }
        if (source_node.op_type() != "Identity" || source_node.input_size() != 1)
          break;
        before = source_index;
        source_name = source_node.input(0);
        source.type = GenerationInfo(graph, source_name).type;
      }
      EXT_ENFORCE_INVALID(to == expected_type && GenerationFloat(source.type),
                          "PrepareGenerationModel: Cast requires known floating source dtype.");
      expected_type = source.type;
    } else {
      EXT_ENFORCE_INVALID(node.attribute().empty(),
                          "PrepareGenerationModel: unsupported Identity attributes.");
    }
    EXT_ENFORCE_INVALID(producers.count(node.input(0)) && producers.at(node.input(0)) < index,
                        "PrepareGenerationModel: head must be topologically ordered.");
    current = node.input(0);
  }

  for (const auto &value : graph.input())
    EXT_ENFORCE_INVALID(!changed.count(value.name()),
                        "PrepareGenerationModel: head aliases a graph input.");
  for (const auto &value : graph.output())
    EXT_ENFORCE_INVALID(value.name() == logits_name || !changed.count(value.name()),
                        "PrepareGenerationModel: head has another exposed output.");
  std::unordered_set<std::string> initializer_names;
  for (const auto &value : graph.initializer())
    initializer_names.insert(value.name());
  for (const auto &value : graph.sparse_initializer())
    initializer_names.insert(value.values().name());
  for (const auto &value : graph.extended_initializer()) {
    if (value.has_encoded_value())
      initializer_names.insert(value.encoded_value().name());
    if (value.has_paged_cache())
      initializer_names.insert(value.paged_cache().name());
  }
  for (const auto &name : changed)
    EXT_ENFORCE_INVALID(!initializer_names.count(name),
                        "PrepareGenerationModel: head aliases an initializer.");
  for (const auto &binding : graph.persistent_bindings())
    EXT_ENFORCE_INVALID(!changed.count(binding.input_name()) &&
                            !changed.count(binding.output_name()),
                        "PrepareGenerationModel: head aliases persistent state.");
  std::unordered_map<std::string, size_t> permitted_consumer;
  for (size_t index = 0; index + 1 < chain.size(); ++index)
    permitted_consumer.emplace(graph.node(chain[index + 1]).output(0), chain[index]);
  for (size_t index = 0; index < graph.node().size(); ++index) {
    const auto &node = graph.node(index);
    for (const auto &name : node.input())
      EXT_ENFORCE_INVALID(!changed.count(name) || (permitted_consumer.count(name) &&
                                                   permitted_consumer.at(name) == index),
                          "PrepareGenerationModel: head has another consumer.");
    std::unordered_set<std::string> nested_names;
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g())
        GenerationNames(attribute.g(), nested_names);
      for (const auto &nested : attribute.graphs())
        GenerationNames(nested, nested_names);
    }
    for (const auto &name : changed)
      EXT_ENFORCE_INVALID(!nested_names.count(name),
                          "PrepareGenerationModel: nested graph captures a head value.");
  }

  std::unordered_set<std::string> names;
  GenerationNames(graph, names);
  const auto unique = [&](const std::string &base) {
    std::string name = base;
    for (size_t suffix = 0; !names.insert(name).second; ++suffix)
      name = base + "_" + std::to_string(suffix);
    return name;
  };
  const std::string indices_name = unique(logits_name + "_last_indices");
  const std::string hidden_name = unique(logits_name + "_last_hidden");
  NodeProto gather;
  gather.set_name(unique(logits_name + "_last_position"));
  gather.set_op_type("Gather");
  gather.add_input(graph.node(head).input(0));
  gather.add_input(indices_name);
  gather.add_output(hidden_name);
  auto *axis = gather.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
  TensorProto indices;
  indices.set_name(indices_name);
  indices.set_data_type(TensorProto::INT64);
  indices.add_dims(1);
  indices.add_int64_data(-1);

  // Everything above is read-only. Move node ownership, never copy the graph or weights.
  GraphProto &prepared = *model.mutable_graph();
  *prepared.mutable_node(head)->mutable_input(0) = hidden_name;
  *prepared.add_initializer() = std::move(indices);
  *prepared.add_node() = std::move(gather);
  for (size_t index = prepared.node().size() - 1; index > head; --index)
    prepared.mutable_node()->SwapElements(static_cast<int>(index), static_cast<int>(index - 1));
  for (auto *values : {prepared.mutable_value_info(), prepared.mutable_output()})
    for (auto &value : *values)
      if (changed.count(value.name()) && value.type().tensor_type().has_shape()) {
        auto *sequence =
            value.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
        sequence->clear_dim_param();
        sequence->set_dim_value(1);
      }
  for (auto &node : *prepared.mutable_node())
    ClearGenerationExecutionMetadata(node);
  ClearGenerationExecutionMetadata(prepared);
  model.clear_rewritings();
  auto *marker = model.add_metadata_props();
  marker->set_key(kGenerationPrepared);
  marker->set_value(logits_name);
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
  const GenerationPipelineView view{.prefill = &pipeline.prefill(),
                                    .embedding =
                                        pipeline.has_embedding() ? &pipeline.embedding() : nullptr,
                                    .decode = pipeline.has_decode() ? &pipeline.decode() : nullptr};
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
                           const RuntimeValueMap &decode_feeds,
                           const RuntimeValueMap &embedding_feeds, const GenerationOptions &options,
                           RuntimeSessionOptions session_options,
                           const RuntimeValueMap &unscoped_feeds) {
  EXT_ENFORCE_INVALID(pipeline.prefill != nullptr, "Generate: prefill model is required.");
  const ModelProto &model = *pipeline.prefill;
  const bool split = pipeline.decode != nullptr;
  EXT_ENFORCE_INVALID(options.max_new_tokens >= 0, "Generate: max_new_tokens must be nonnegative.");
  EXT_ENFORCE_INVALID(options.max_logit_bytes == 0 || options.on_step,
                      "Generate: raw-logit capture requires an on_step callback.");
  EXT_ENFORCE_INVALID(std::isfinite(options.temperature) && options.temperature >= 0.,
                      "Generate: temperature must be finite and nonnegative.");
  std::unordered_set<int64_t> eos_token_ids;
  if (!options.eos_token_ids.empty())
    eos_token_ids.reserve(options.eos_token_ids.size());
  int64_t largest_eos_token_id = -1;
  for (int64_t id : options.eos_token_ids) {
    EXT_ENFORCE_INVALID(id >= 0, "Generate: eos_token_ids must be nonnegative; got ", id, ".");
    eos_token_ids.insert(id);
    largest_eos_token_id = std::max(largest_eos_token_id, id);
  }
  auto pad_token_id = options.pad_token_id;
  if (!pad_token_id && !options.eos_token_ids.empty())
    pad_token_id = options.eos_token_ids.front();
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
      if (split)
        EXT_ENFORCE_INVALID(HasInput(*pipeline.decode, output.name()),
                            "Generate: embedding output must name a decode input.");
    }
    ValidatePipelineValueNames(*pipeline.embedding, model, context);
    if (split)
      ValidatePipelineValueNames(*pipeline.embedding, *pipeline.decode, context);
  }
  for (const auto *stage : {&model, pipeline.decode})
    if (stage != nullptr)
      for (const auto &binding : stage->graph().persistent_bindings())
        if (embedded)
          for (const auto &output : pipeline.embedding->graph().output())
            EXT_ENFORCE_INVALID(output.name() != binding.input_name(),
                                "Generate: embedding output cannot replace a persistent input.");
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
  const bool persistent = !model.graph().persistent_bindings().empty();
  for (const auto &name : options.prefill_only_inputs) {
    EXT_ENFORCE_INVALID(embedded && persistent,
                        "Generate: prefill_only_inputs requires an embedding model and cached "
                        "decoding with persistent bindings.");
    EXT_ENFORCE_INVALID(name != options.input_ids_name && name != options.attention_mask_name &&
                            name != options.position_ids_name && !HasInput(model, name) &&
                            (!split || !HasInput(*pipeline.decode, name)),
                        "Generate: prefill-only input '", name,
                        "' must be an embedding-only auxiliary input.");
    const auto found = embedding_feeds.find(name);
    EXT_ENFORCE_INVALID(
        found != embedding_feeds.end() && found->second.kind == RuntimeValue::Kind::kTensor &&
            !found->second.tensor.shape.empty(),
        "Generate: prefill-only input '", name, "' requires an explicit nonscalar tensor feed.");
    ValidateStorage(found->second.tensor);
  }
  const RuntimeValueMap &prompt_feeds = embedded ? embedding_feeds : prefill_feeds;
  const Tensor &prompt = IntMatrix(prompt_feeds, options.input_ids_name);
  const int64_t batch = prompt.shape[0], prompt_length = prompt.shape[1];
  EXT_ENFORCE_INVALID(!split || prompt_length < std::numeric_limits<int64_t>::max(),
                      "Generate: decode attention prefix length overflows.");
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
                        (split && HasInput(*pipeline.decode, options.attention_mask_name)) ||
                        (embedded && HasInput(*pipeline.embedding, options.attention_mask_name));
  const bool has_positions = HasInput(model, options.position_ids_name) ||
                             (split && HasInput(*pipeline.decode, options.position_ids_name)) ||
                             (embedded && HasInput(*pipeline.embedding, options.position_ids_name));
  EXT_ENFORCE_INVALID(options.position_axes >= 0,
                      "Generate: position axis count must be nonnegative.");
  int64_t axes = options.position_axes;
  if (has_positions) {
    bool rank2 = false, rank3 = false;
    axes = PositionAxes(model, options.position_ids_name, axes, rank2, rank3);
    if (split)
      axes = PositionAxes(*pipeline.decode, options.position_ids_name, axes, rank2, rank3);
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
  if (split) {
    ValidateSplitGeneration(pipeline, options, batch, prompt_length, axes, largest_eos_token_id);
    StructTypeCatalogue embedding_types;
    if (embedded)
      embedding_types.Build(*pipeline.embedding);
    for (const auto *stage : {&model, pipeline.decode, pipeline.embedding}) {
      if (stage == nullptr)
        continue;
      StructTypeCatalogue types;
      types.Build(*stage);
      const auto &stage_feeds = stage == &model            ? prefill_feeds
                                : stage == pipeline.decode ? decode_feeds
                                                           : embedding_feeds;
      for (const auto &input : stage->graph().input()) {
        const auto &name = input.name();
        const bool generated = name == options.input_ids_name ||
                               name == options.attention_mask_name ||
                               name == options.position_ids_name;
        const bool retained =
            std::any_of(stage->graph().persistent_bindings().begin(),
                        stage->graph().persistent_bindings().end(),
                        [&](const auto &binding) { return binding.input_name() == name; });
        const ValueInfoProto *connection = nullptr;
        if (embedded && stage != pipeline.embedding)
          for (const auto &output : pipeline.embedding->graph().output())
            if (output.name() == name)
              connection = &output;
        if (connection != nullptr) {
          EXT_ENFORCE_INVALID(
              CompatibleGenerationTypes(connection->type(), input.type(), embedding_types, types),
              "Generate: incompatible embedding connection '", name, "'.");
          continue;
        }
        if (generated || retained)
          continue;
        const auto feed = stage_feeds.find(name);
        const bool initialized =
            std::any_of(stage->graph().initializer().begin(), stage->graph().initializer().end(),
                        [&](const auto &value) { return value.name() == name; }) ||
            std::any_of(stage->graph().extended_initializer().begin(),
                        stage->graph().extended_initializer().end(),
                        [&](const auto &value) { return value.name() == name; });
        EXT_ENFORCE_INVALID(feed != stage_feeds.end() || initialized,
                            "Generate: missing graph feed '", name, "'.");
        if (feed != stage_feeds.end() && input.type().has_tensor_type()) {
          EXT_ENFORCE_INVALID(feed->second.kind == RuntimeValue::Kind::kTensor,
                              "Generate: expected tensor feed '", name, "'.");
          const auto &tensor = feed->second.tensor;
          ValidateStorage(tensor);
          ValidateGenerationTensor(input, static_cast<int64_t>(tensor.data_type), tensor.shape);
          if (stage == pipeline.embedding &&
              std::find(options.prefill_only_inputs.begin(), options.prefill_only_inputs.end(),
                        name) != options.prefill_only_inputs.end()) {
            Shape empty = tensor.shape;
            empty[0] = 0;
            ValidateGenerationTensor(input, static_cast<int64_t>(tensor.data_type), empty);
          }
        }
      }
    }
    if (embedded) {
      for (const auto &name :
           {options.input_ids_name, options.attention_mask_name, options.position_ids_name}) {
        if (!HasInput(*pipeline.embedding, name))
          continue;
        const auto &input = GenerationValue(pipeline.embedding->graph().input(), name);
        const bool position = name == options.position_ids_name;
        const bool mask_input = name == options.attention_mask_name;
        const Shape prompt_shape =
            position && axes > 0 ? Shape{axes, batch, prompt_length} : Shape{batch, prompt_length};
        const Shape decode_shape = position && axes > 0
                                       ? Shape{axes, batch, 1}
                                       : Shape{batch, mask_input ? prompt_length + 1 : 1};
        ValidateGenerationTensor(input, TensorProto::INT64, prompt_shape);
        ValidateGenerationTensor(input, TensorProto::INT64, decode_shape, mask_input ? 1 : -1,
                                 prompt_length + std::max<int64_t>(1, options.max_new_tokens - 1));
      }
    }
  }
  if (options.max_new_tokens == 0)
    return Tensor::From<int64_t>(options.input_ids_name, prompt.shape, tokens);

  if (embedded) {
    EXT_ENFORCE_INVALID(context.cpu_executor() != nullptr,
                        "Generate: a context with a CPU executor is required for split models.");
    if (session_options.cpu_execution_counters)
      context.cpu_executor()->EnableCounters();
    session_options.cpu_execution.reset();
  }
  if (embedded || split) {
    // Sizes the shared arena before any session captures its allocator.
    const ExecutionPlan *largest = &context.GetExecutionPlan(model);
    const auto include_plan = [&](const ModelProto &graph_model) {
      const auto &plan = context.GetExecutionPlan(graph_model);
      if (plan.arena_capacity() > largest->arena_capacity())
        largest = &plan;
    };
    if (embedded)
      include_plan(*pipeline.embedding);
    if (split)
      include_plan(*pipeline.decode);
    context.BindExecutionPlan(*largest);
  }
  RuntimeValueMap current;
  for (const auto &[name, value] : prefill_feeds)
    current.emplace(name, value.BorrowView());
  RuntimeValueMap embedding_current;
  for (const auto &[name, value] : embedding_feeds)
    embedding_current.emplace(name, value.BorrowView());
  RuntimeValueMap decode_current;
  for (const auto &[name, value] : decode_feeds)
    decode_current.emplace(name, value.BorrowView());
  std::unique_ptr<RuntimeSession> embedding_session;
  if (embedded) {
    embedding_session = std::make_unique<RuntimeSession>(*pipeline.embedding, session_options);
  }
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
      decode_current.erase(name);
    }
  }
  RuntimeSession session(model, std::move(initial), session_options);
  std::unique_ptr<RuntimeSession> decode_session;
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
    if (split && step == 1) {
      if (persistent) {
        decode_session = std::make_unique<RuntimeSession>(
            *pipeline.decode, session.persistent_state().Values(), session_options);
        decode_session->persistent_state().TransferFrom(session.persistent_state());
      } else
        decode_session = std::make_unique<RuntimeSession>(*pipeline.decode, session_options);
    }
    const auto started_at = options.on_step ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    if (step > 0) {
      AppendColumn(mask, next_mask);
      AppendColumn(positions, next_positions);
    }
    const bool decode = (persistent || split) && step > 0;
    const ModelProto &active_model = split && step > 0 ? *pipeline.decode : model;
    RuntimeValueMap &active_current = split && step > 0 ? decode_current : current;
    if (step == 1) {
      for (const auto &name : options.prefill_only_inputs) {
        const auto &prefill = embedding_feeds.at(name).tensor;
        Shape empty_shape = prefill.shape;
        empty_shape[0] = 0;
        embedding_current.insert_or_assign(
            name, RuntimeValue(Tensor(name, prefill.data_type, std::move(empty_shape), {})));
      }
    }
    const Shape shape{batch, decode ? 1 : length};
    const auto put_control = [&](const std::string &name, RuntimeValue value) {
      if (HasInput(active_model, name))
        active_current.insert_or_assign(name, value.BorrowView());
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
        active_current.insert_or_assign(output.name(),
                                        embedding_current.at(output.name()).BorrowView());
    }
    Tensor logits;
    if (persistent) {
      if (split)
        context.ResetInvocation();
      RuntimeValueMap model_feeds;
      for (const auto &input : active_model.graph().input()) {
        const auto found = active_current.find(input.name());
        if (found != active_current.end())
          model_feeds.emplace(input.name(), found->second.BorrowView());
      }
      auto outputs = (split && step > 0 ? *decode_session : session).Run(context, model_feeds);
      auto it = outputs.find(options.logits_name);
      EXT_ENFORCE_INVALID(it != outputs.end() && it->second.kind == RuntimeValue::Kind::kTensor,
                          "Generate: logits output must be a tensor.");
      logits = std::move(it->second.tensor);
    } else {
      context.ResetInvocation();
      for (const auto &input : active_model.graph().input()) {
        const auto found = active_current.find(input.name());
        if (found == active_current.end())
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
      (split && step > 0 ? *decode_session : session).Run(context);
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
    EXT_ENFORCE_INVALID(largest_eos_token_id < vocabulary,
                        "Generate: eos_token_ids contains an ID outside the vocabulary: got ",
                        largest_eos_token_id, ", valid range is [0, ", vocabulary, ").");
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
    EXT_ENFORCE_INVALID(!options.pad_token_id || *options.pad_token_id < vocabulary,
                        "Generate: pad_token_id is outside the vocabulary.");
    const int64_t stride = logits.shape.product() / batch;
    for (int64_t row = 0; row < batch; ++row) {
      next_mask[row] = finished[row] ? 0 : 1;
      if (finished[row]) {
        EXT_ENFORCE(pad_token_id.has_value(),
                    "Generate: a finished batch row requires a padding token.");
        next[row] = *pad_token_id;
      } else {
        next[row] = Sample(logits, (row + 1) * stride - vocabulary, vocabulary, options.temperature,
                           random);
      }
      if (multi_axis) {
        for (int64_t axis = 0; axis < axes; ++axis)
          ++next_positions[axis * batch + row];
      } else {
        next_positions[row] = positions[row * length + length - 1] + 1;
      }
      if (eos_token_ids.contains(next[row]))
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
