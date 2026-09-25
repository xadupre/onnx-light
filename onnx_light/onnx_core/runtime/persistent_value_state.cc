// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/quantization.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

using Symbols = std::unordered_map<std::string, int64_t>;
using ValidationMemos = std::unordered_map<const TypeProto *, RuntimeSequence::Memo<Symbols>>;
using Declarations = utils::RepeatedProtoField<ValueInfoProto>;

const ModelProto &RequireModel(const std::shared_ptr<const ModelProto> &model) {
  EXT_ENFORCE_INVALID(model != nullptr, "PersistentValueState: model owner must not be null.");
  return *model;
}

void CheckNestedBindings(const utils::RepeatedProtoField<NodeProto> &nodes, size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "PersistentValueState: maximum subgraph depth exceeded.");
  const auto check = [depth](const GraphProto &graph) {
    EXT_ENFORCE_INVALID(
        graph.persistent_bindings().empty(),
        "PersistentValueState: persistent bindings are restricted to the model root.");
    CheckNestedBindings(graph.node(), depth + 1);
  };
  for (const auto &node : nodes)
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g())
        check(attribute.g());
      for (const auto &graph : attribute.graphs())
        check(graph);
    }
}

const TypeProto &InputType(const std::string &name, const Declarations &declarations) {
  for (const auto &item : declarations)
    if (item.name() == name)
      return item.type();
  EXT_THROW_INVALID("PersistentValueState: unknown graph input name '", name,
                    "'; names are literal and partial field paths are unsupported.");
}

void ValidateTensorShape(const Shape &shape, const TypeProto::Tensor &declared, Symbols &symbols) {
  if (!declared.has_shape())
    return;
  EXT_ENFORCE_INVALID(static_cast<size_t>(declared.shape().dim_size()) == shape.size(),
                      "PersistentValueState: rank mismatch.");
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto &dim = declared.shape().dim(static_cast<int>(i));
    EXT_ENFORCE_INVALID(!dim.has_dim_value() || dim.dim_value() == shape[i],
                        "PersistentValueState: shape mismatch.");
    if (dim.has_dim_param() && !dim.dim_param().empty()) {
      auto [it, inserted] = symbols.emplace(dim.dim_param(), shape[i]);
      EXT_ENFORCE_INVALID(inserted || it->second == shape[i],
                          "PersistentValueState: symbolic shape mismatch.");
    }
  }
}

void Validate(const RuntimeValue &value, const TypeProto &type,
              const StructTypeCatalogue &catalogue, Symbols &symbols, ValidationMemos &memos,
              size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "PersistentValueState: maximum nesting depth exceeded.");
  if (value.kind == RuntimeValue::Kind::kSequence) {
    EXT_ENFORCE_INVALID(type.has_sequence_type() && type.sequence_type().has_elem_type(),
                        "PersistentValueState: expected a sequence type.");
    EXT_ENFORCE_INVALID(value.elements.empty() ||
                            depth + 1 + value.elements.depth() <= RuntimeValue::kMaxDepth,
                        "PersistentValueState: maximum nesting depth exceeded.");
    const auto merge = [](Symbols left, const Symbols &right) {
      for (const auto &[name, dimension] : right) {
        const auto [it, inserted] = left.emplace(name, dimension);
        EXT_ENFORCE_INVALID(inserted || it->second == dimension,
                            "PersistentValueState: symbolic shape mismatch.");
      }
      return left;
    };
    auto sequence_symbols = value.elements.Fold(
        memos[&type],
        [&](const RuntimeValue &element) {
          Symbols local;
          Validate(element, type.sequence_type().elem_type(), catalogue, local, memos, depth + 1);
          return local;
        },
        merge);
    symbols = merge(std::move(symbols), sequence_symbols);
    return;
  }
  if (value.kind == RuntimeValue::Kind::kTensor) {
    EXT_ENFORCE_INVALID(type.has_tensor_type(),
                        "PersistentValueState: expected a structured value.");
    const auto &declared = type.tensor_type();
    const Tensor &tensor = value.tensor;
    EXT_ENFORCE_INVALID(tensor.data_type == declared.elem_type(),
                        "PersistentValueState: dtype mismatch.");
    const int64_t count = tensor.shape.product(0, tensor.shape.size(), "PersistentValueState");
    if (tensor.data_type == DataType::STRING) {
      EXT_ENFORCE_INVALID(tensor.AsStrings().size() == static_cast<size_t>(count),
                          "PersistentValueState: string tensor extent mismatch.");
    } else {
      EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(tensor.data_type, count) &&
                              (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                          "PersistentValueState: tensor byte extent mismatch.");
    }
    ValidateTensorShape(tensor.shape, declared, symbols);
    return;
  }
  if (value.kind == RuntimeValue::Kind::kEncoded && type.has_tensor_type()) {
    const auto &encoded = value.Encoded();
    const auto layout = catalogue.ValidateEncodedValue(encoded);
    EXT_ENFORCE_INVALID(!layout.external && layout.content_verified && encoded.has_logical_type() &&
                            encoded.logical_type().has_tensor_type(),
                        "PersistentValueState: requires an inline encoded tensor payload.");
    const auto &logical = encoded.logical_type().tensor_type();
    EXT_ENFORCE_INVALID(logical.elem_type() == type.tensor_type().elem_type(),
                        "PersistentValueState: dtype mismatch.");
    EXT_ENFORCE_INVALID(logical.has_shape(),
                        "PersistentValueState: encoded values require a concrete shape.");
    Shape shape;
    for (const auto &dim : logical.shape().dim()) {
      EXT_ENFORCE_INVALID(dim.has_dim_value() && !dim.has_dim_param() && dim.dim_value() >= 0,
                          "PersistentValueState: encoded values require concrete dimensions.");
      shape.push_back(dim.dim_value());
    }
    ValidateTensorShape(shape, type.tensor_type(), symbols);
    return;
  }
  EXT_ENFORCE_INVALID(type.has_struct_type(), "PersistentValueState: expected a tensor.");
  const auto &declared = catalogue.Resolve(type.struct_type());
  if (value.kind == RuntimeValue::Kind::kEncoded) {
    const auto layout = catalogue.ValidateEncodedValue(value.Encoded());
    EXT_ENFORCE_INVALID(!layout.external && layout.content_verified &&
                            value.Encoded().has_struct_type(),
                        "PersistentValueState: requires an inline structured encoded payload.");
    const auto &actual = catalogue.Resolve(value.Encoded().struct_type());
    EXT_ENFORCE_INVALID(CompatiblePersistentStructTypes(catalogue, actual, declared),
                        "PersistentValueState: encoded representation mismatch.");
    return;
  }
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kStruct,
                      "PersistentValueState: expected a structured value.");
  EXT_ENFORCE_INVALID(declared.has_structure(),
                      "PersistentValueState: requires named structured fields.");
  size_t expected = 0;
  for (const auto &field : declared.structure().field()) {
    if (!field.has_type())
      continue;
    ++expected;
    auto it = value.fields.find(field.name());
    EXT_ENFORCE_INVALID(it != value.fields.end(), "PersistentValueState: missing field '",
                        field.name(), "'.");
    Validate(it->second, field.type(), catalogue, symbols, memos, depth + 1);
  }
  EXT_ENFORCE_INVALID(value.fields.size() == expected,
                      "PersistentValueState: unexpected structured field.");
}

class Operation {
public:
  explicit Operation(std::atomic_flag &busy) : busy_(busy) {
    EXT_ENFORCE_INVALID(!busy_.test_and_set(std::memory_order_acquire),
                        "PersistentValueState: simultaneous operations are not allowed.");
  }
  ~Operation() { busy_.clear(std::memory_order_release); }
  Operation(const Operation &) = delete;
  Operation &operator=(const Operation &) = delete;

private:
  std::atomic_flag &busy_;
};

} // namespace

PersistentValueState::PersistentValueState(const ModelProto &model, RuntimeValueMap initial,
                                           RuntimeSessionOptions options,
                                           std::shared_ptr<void> model_owner)
    : model_(model),
      model_owner_(model_owner.use_count() != 0
                       ? std::move(model_owner)
                       : std::shared_ptr<void>(const_cast<ModelProto *>(&model), [](void *) {})) {
  catalogue_.Build(model);
  CheckNestedBindings(model.graph().node());
  for (const auto &function : model.functions())
    CheckNestedBindings(function.node());
  EXT_ENFORCE_INVALID(!model.graph().persistent_bindings().empty(),
                      "PersistentValueState: missing graph persistence declarations.");
  for (const auto &input : model.graph().input())
    catalogue_.ValidateType(input.type());
  for (const auto &output : model.graph().output())
    catalogue_.ValidateType(output.type());
  VerifyPersistentBindings(&catalogue_, model.graph());
  for (const auto &declared : model.graph().persistent_bindings()) {
    bindings_.push_back({declared.input_name(), declared.output_name(),
                         &InputType(declared.input_name(), model.graph().input())});
  }
  persistent_tensor_initial_capacity_ = options.persistent_tensor_initial_capacity;
  session_ = std::make_unique<RuntimeSession>(model, std::move(options));
  values_ = ValidateInitial(std::move(initial));
}

PersistentValueState::PersistentValueState(std::shared_ptr<const ModelProto> model,
                                           RuntimeValueMap initial, RuntimeSessionOptions options)
    : PersistentValueState(RequireModel(model), std::move(initial), std::move(options),
                           std::shared_ptr<void>(model, const_cast<ModelProto *>(model.get()))) {}

std::vector<PersistentValue> PersistentValueState::ValidateInitial(RuntimeValueMap initial) const {
  std::shared_ptr<const QuantizationParameterCatalogue> parameters;
  for (const auto &cache : model_.graph().paged_cache_initializer())
    if (!initial.contains(cache.name()) &&
        std::any_of(bindings_.begin(), bindings_.end(),
                    [&](const auto &binding) { return binding.input == cache.name(); })) {
      if (!parameters)
        parameters = QuantizationParameterCatalogue::Build(model_);
      initial.emplace(cache.name(), RuntimeValue::FromPagedCache(cache, catalogue_, parameters));
    }
  std::vector<PersistentValue> result;
  result.reserve(bindings_.size());
  Symbols symbols;
  for (const auto &binding : bindings_) {
    const auto it = initial.find(binding.input);
    EXT_ENFORCE_INVALID(it != initial.end(), "PersistentValueState: missing initial whole input '",
                        binding.input, "'.");
    Validate(it->second, *binding.input_type, catalogue_, symbols, sequence_validation_);
    result.emplace_back(std::move(it->second), catalogue_);
  }
  EXT_ENFORCE_INVALID(
      initial.size() == bindings_.size(),
      "PersistentValueState: initial values must name exactly the retained whole inputs.");
  return result;
}

RuntimeValueMap PersistentValueState::Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                                          const TaskCompletion *completion) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "PersistentValueState: state is closed.");
  EXT_ENFORCE_INVALID(
      completion == nullptr || completion->status() == TaskStatus::kPending,
      "PersistentValueState: invocation was cancelled or completion is not pending.");
  RuntimeContext invocation = context.MakeFunctionContext();
  invocation.persistent_tensor_initial_capacity_ = persistent_tensor_initial_capacity_;
  invocation.persistent_graph_ = &model_.graph();
  invocation.persistent_tensors_ =
      std::make_shared<std::vector<RuntimeContext::PersistentTensorBinding>>();
  RuntimeValueMap inputs;
  for (size_t i = 0; i < bindings_.size(); ++i) {
    if (const auto *tensor = values_[i].tensor()) {
      invocation.persistent_tensors_->push_back({bindings_[i].input, bindings_[i].output, nullptr,
                                                 tensor->PrepareAppend(), std::nullopt});
    }
    inputs.emplace(bindings_[i].input, values_[i].BorrowView());
  }
  for (const auto &[name, value] : feeds) {
    InputType(name, model_.graph().input());
    EXT_ENFORCE_INVALID(inputs.find(name) == inputs.end(),
                        "PersistentValueState: current feed cannot override retained input '", name,
                        "'.");
    inputs.emplace(name, value.BorrowView());
  }
  Symbols symbols;
  std::unordered_set<std::string> retained_outputs;
  for (const auto &binding : bindings_)
    retained_outputs.insert(binding.output);
  invocation.set_retained_outputs(std::move(retained_outputs));
  invocation.set_model_owner(model_owner_);
  RegisterModelFunctions(model_, invocation);
  for (const auto &input : model_.graph().input()) {
    auto it = inputs.find(input.name());
    if (it == inputs.end()) {
      bool initializer = false;
      for (const auto &tensor : model_.graph().initializer())
        initializer = initializer || tensor.name() == input.name();
      for (const auto &encoded : model_.graph().encoded_initializer())
        initializer = initializer || encoded.name() == input.name();
      for (const auto &cache : model_.graph().paged_cache_initializer())
        initializer = initializer || cache.name() == input.name();
      EXT_ENFORCE_INVALID(initializer, "PersistentValueState: missing current input '",
                          input.name(), "'.");
      continue;
    }
    Validate(it->second, input.type(), catalogue_, symbols, sequence_validation_);
    invocation.PutValue(input.name(), std::move(it->second), RuntimeEventKind::kInput);
  }
  for (auto &binding : *invocation.persistent_tensors_)
    binding.input_view = &invocation.Get(binding.input);
  if (allocators_captured_) {
    EXT_ENFORCE_INVALID(execution_allocator_ == context.execution_allocator() &&
                            io_allocator_ == context.io_allocator(),
                        "PersistentValueState: context allocators changed between calls.");
  } else {
    execution_allocator_ = context.execution_allocator();
    io_allocator_ = context.io_allocator();
    allocators_captured_ = true;
  }
  session_->Run(invocation);
  RuntimeValueMap outputs;
  for (const auto &output : model_.graph().output()) {
    RuntimeValue value;
    if (invocation.Has(output.name()))
      value = RuntimeValue(std::move(invocation.Get(output.name())));
    else {
      auto it = invocation.values().find(output.name());
      EXT_ENFORCE_INVALID(it != invocation.values().end(), "PersistentValueState: missing output '",
                          output.name(), "'.");
      value = std::move(it->second);
    }
    Validate(value, output.type(), catalogue_, symbols, sequence_validation_);
    outputs.emplace(output.name(), std::move(value));
  }
  std::vector<PersistentValue> next;
  next.reserve(bindings_.size());
  Symbols next_symbols;
  for (const auto &binding : bindings_) {
    RuntimeValue &value = outputs.at(binding.output);
    Validate(value, *binding.input_type, catalogue_, next_symbols, sequence_validation_);
    value = std::move(value).Retain(catalogue_);
    auto candidate = std::find_if(invocation.persistent_tensors_->begin(),
                                  invocation.persistent_tensors_->end(), [&](const auto &item) {
                                    return item.input == binding.input &&
                                           item.output == binding.output && item.candidate &&
                                           value.kind == RuntimeValue::Kind::kTensor &&
                                           item.candidate->Matches(value.tensor);
                                  });
    if (candidate != invocation.persistent_tensors_->end())
      next.emplace_back(std::move(*candidate->candidate));
    else
      next.emplace_back(value.BorrowView(), catalogue_);
  }
  if (completion != nullptr) {
    EXT_ENFORCE_INVALID(completion->status() == TaskStatus::kPending,
                        "PersistentValueState: invocation was cancelled.");
    // Succeed and Cancel serialize on TaskCompletion's mutex. A lost race throws
    // before the noexcept swap, leaving the retained state untouched.
    completion->Succeed();
  }
  values_.swap(next);
  return outputs;
}

void PersistentValueState::Reset(RuntimeValueMap initial) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "PersistentValueState: state is closed.");
  auto next = ValidateInitial(std::move(initial));
  values_.swap(next);
}

void PersistentValueState::RetainOwner(std::shared_ptr<void> owner) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "PersistentValueState: state is closed.");
  EXT_ENFORCE_INVALID(owner.use_count() != 0,
                      "PersistentValueState: owner token must retain a lifetime.");
  for (const auto &retained : retained_owners_)
    if (!owner.owner_before(retained) && !retained.owner_before(owner))
      return;
  retained_owners_.push_back(std::move(owner));
}

void PersistentValueState::Close() {
  const Operation operation(busy_);
  values_.clear();
  session_.reset();
  retained_owners_.clear();
  model_owner_.reset();
}

RuntimeValueMap PersistentValueState::Values() const {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "PersistentValueState: state is closed.");
  RuntimeValueMap snapshot;
  for (size_t i = 0; i < bindings_.size(); ++i)
    snapshot.emplace(bindings_[i].input, values_[i].BorrowView());
  return snapshot;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
