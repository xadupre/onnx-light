// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/quantization.h"
#include "onnx_lib/common/scoped_resource.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

using Symbols = std::unordered_map<std::string, int64_t>;
using ValidationMemos = std::unordered_map<const TypeProto *, RuntimeSequence::Memo<Symbols>>;
using Declarations = utils::RepeatedProtoField<ValueInfoProto>;

bool IsTensorSequence(const TypeProto &type) {
  return type.has_sequence_type() && type.sequence_type().has_elem_type() &&
         type.sequence_type().elem_type().has_tensor_type();
}

bool IsDenseTensorSequence(const RuntimeValue &value) {
  return value.kind == RuntimeValue::Kind::kSequence &&
         std::all_of(value.elements.begin(), value.elements.end(), [](const RuntimeValue &element) {
           return element.kind == RuntimeValue::Kind::kTensor;
         });
}

void RetainIOBackedStorage(RuntimeValue &value, size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "PersistentValueState: maximum output nesting depth exceeded.");
  if (value.kind == RuntimeValue::Kind::kTensor) {
    if (value.tensor.data_type == DataType::STRING) {
      value.tensor = value.tensor.ToOwned();
      return;
    }
    if (value.tensor.data_type != DataType::STRING && value.tensor.has_allocation() &&
        dynamic_cast<IOArena *>(value.tensor.allocation_owner()) != nullptr)
      value.tensor = std::move(value.tensor).RetainStorage();
  } else if (value.kind == RuntimeValue::Kind::kSequence) {
    if (value.elements.retained())
      return;
    value.elements.TransformInPlace(
        [&](RuntimeValue &element) { RetainIOBackedStorage(element, depth + 1); });
  } else if (value.kind == RuntimeValue::Kind::kStruct) {
    for (auto &[name, field] : value.fields) {
      (void)name;
      RetainIOBackedStorage(field, depth + 1);
    }
  }
}

Sequence ToTensorSequence(const std::string &name, const RuntimeValue &value,
                          const TypeProto::Sequence &type) {
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kSequence,
                      "PersistentValueState: expected a sequence value.");
  Tensors tensors;
  tensors.reserve(value.elements.size());
  for (const RuntimeValue &element : value.elements) {
    EXT_ENFORCE_INVALID(element.kind == RuntimeValue::Kind::kTensor,
                        "PersistentValueState: standard sequences require tensor elements.");
    tensors.push_back(element.tensor.BorrowView());
  }
  return Sequence(name, type.elem_type().tensor_type().elem_type(), std::move(tensors));
}

RuntimeValue FromTensorSequence(Sequence sequence) {
  std::vector<RuntimeValue> elements;
  elements.reserve(sequence.values.size());
  for (Tensor &tensor : sequence.values)
    elements.emplace_back(std::move(tensor));
  return RuntimeValue(std::move(elements));
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
              size_t depth = 0, bool paged_cache = false) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "PersistentValueState: maximum nesting depth exceeded.");
  if (value.kind == RuntimeValue::Kind::kEncoded && value.Encoded().has_parameter_ref()) {
    EXT_ENFORCE_INVALID(value.quantization_parameters != nullptr,
                        "PersistentValueState: missing shared quantization parameter catalogue.");
    value.quantization_parameters->Validate(value.Encoded(), catalogue);
  }
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
          Validate(element, type.sequence_type().elem_type(), catalogue, local, memos, depth + 1,
                   paged_cache);
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
    const bool floating_cache_storage =
        paged_cache && declared.elem_type() == DataType::FLOAT &&
        (tensor.data_type == DataType::FLOAT16 || tensor.data_type == DataType::BFLOAT16);
    EXT_ENFORCE_INVALID(tensor.data_type == declared.elem_type() || floating_cache_storage,
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
  const bool cache_structure = value.HasPagedCacheStructure();
  if (cache_structure) {
    catalogue.ValidatePagedCache(value.ToPagedCache("", catalogue), true, &type);
    paged_cache = true;
  }
  if (!declared.has_structure()) {
    EXT_ENFORCE_INVALID(cache_structure && declared.kind_case() == StructTypeProto::KIND_NOT_SET,
                        "PersistentValueState: ordinary structures require named fields.");
    return;
  }
  ValidationMemos cache_memos;
  auto &field_memos = cache_structure ? cache_memos : memos;
  size_t expected = 0;
  for (const auto &field : declared.structure().field()) {
    if (!field.has_type())
      continue;
    ++expected;
    auto it = value.fields.find(field.name());
    EXT_ENFORCE_INVALID(it != value.fields.end(), "PersistentValueState: missing field '",
                        field.name(), "'.");
    Validate(it->second, field.type(), catalogue, symbols, field_memos, depth + 1, paged_cache);
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
                                           RuntimeSessionOptions options)
    : model_(model) {
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
  values_ = ValidateInitial(std::move(initial));
}

std::vector<PersistentValue> PersistentValueState::ValidateInitial(RuntimeValueMap initial) const {
  std::shared_ptr<const QuantizationParameterCatalogue> parameters;
  for (const auto &initializer : model_.graph().extended_initializer()) {
    if (!initializer.has_paged_cache())
      continue;
    const auto &cache = initializer.paged_cache();
    if (!initial.contains(cache.name()) &&
        std::any_of(bindings_.begin(), bindings_.end(),
                    [&](const auto &binding) { return binding.input == cache.name(); })) {
      if (!parameters)
        parameters = QuantizationParameterCatalogue::Build(model_);
      initial.emplace(cache.name(), RuntimeValue::FromPagedCache(cache, catalogue_, parameters));
    }
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

RuntimeSession::RuntimeSession(const ModelProto &model, RuntimeValueMap initial,
                               RuntimeSessionOptions options)
    : RuntimeSession(model, options) {
  if (!model.graph().persistent_bindings().empty())
    persistent_state_ = std::make_unique<PersistentValueState>(model, std::move(initial), options);
  else
    EXT_ENFORCE_INVALID(initial.empty(),
                        "RuntimeSession: initial persistent values require persistent bindings.");
}

PersistentValueState &RuntimeSession::persistent_state() {
  EXT_ENFORCE_INVALID(persistent_state_ != nullptr,
                      "RuntimeSession: no persistent state was initialized.");
  return *persistent_state_;
}

const PersistentValueState &RuntimeSession::persistent_state() const {
  EXT_ENFORCE_INVALID(persistent_state_ != nullptr,
                      "RuntimeSession: no persistent state was initialized.");
  return *persistent_state_;
}

void RuntimeSession::Reset(RuntimeValueMap initial) {
  persistent_state().Reset(std::move(initial));
}

RuntimeValueMap RuntimeSession::Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                                    const TaskCompletion *completion) {
  PersistentValueState &state = persistent_state();
  const Operation operation(state.busy_);
  EXT_ENFORCE_INVALID(
      completion == nullptr || completion->status() == TaskStatus::kPending,
      "PersistentValueState: invocation was cancelled or completion is not pending.");
  if (state.allocators_captured_)
    EXT_ENFORCE_INVALID(state.execution_allocator_ == context.execution_allocator() &&
                            state.io_allocator_ == context.io_allocator(),
                        "PersistentValueState: context allocators changed between calls.");
  bool started = false;
  ScopeExit cleanup([&]() noexcept {
    if (started) {
      state.execution_allocator_ = context.execution_allocator();
      state.io_allocator_ = context.io_allocator();
      state.allocators_captured_ = true;
    }
    context.persistent_graph_ = nullptr;
    context.persistent_tensors_.reset();
    context.retained_outputs_.clear();
    // Removes invocation borrows without allocating or recording events during unwinding.
    const auto release = [&](const std::string &name) {
      const auto slot = context.value_slots_.find(name);
      if (slot != context.value_slots_.end() && slot->second < context.tensor_slots_.size())
        context.tensor_slots_[slot->second] = nullptr;
      context.tensors_.erase(name);
      context.values_.erase(name);
      context.sequences_.erase(name);
      context.maps_.erase(name);
      context.shapes_.erase(name);
    };
    for (const auto &input : state.model_.graph().input())
      release(input.name());
    for (const auto &output : state.model_.graph().output())
      release(output.name());
  });
  for (const auto &input : state.model_.graph().input())
    context.Remove(input.name());
  for (const auto &node : state.model_.graph().node())
    for (const auto &output : node.output())
      if (!output.empty())
        context.Remove(output);
  context.persistent_tensor_initial_capacity_ = state.persistent_tensor_initial_capacity_;
  context.persistent_graph_ = &state.model_.graph();
  context.persistent_tensors_ =
      std::make_shared<std::vector<RuntimeContext::PersistentTensorBinding>>();
  for (size_t i = 0; i < state.bindings_.size(); ++i) {
    if (const auto *tensor = state.values_[i].tensor()) {
      context.persistent_tensors_->push_back({state.bindings_[i].input, state.bindings_[i].output,
                                              nullptr, tensor->PrepareAppend(), std::nullopt});
    }
  }
  for (const auto &[name, value] : feeds) {
    (void)value;
    InputType(name, state.model_.graph().input());
    EXT_ENFORCE_INVALID(std::none_of(state.bindings_.begin(), state.bindings_.end(),
                                     [&](const auto &binding) { return binding.input == name; }),
                        "PersistentValueState: current feed cannot override retained input '", name,
                        "'.");
  }
  Symbols symbols;
  std::unordered_set<std::string> retained_outputs;
  for (const auto &binding : state.bindings_)
    retained_outputs.insert(binding.output);
  context.set_retained_outputs(std::move(retained_outputs));
  for (const auto &input : state.model_.graph().input()) {
    const auto binding = std::find_if(state.bindings_.begin(), state.bindings_.end(),
                                      [&](const auto &item) { return item.input == input.name(); });
    const auto feed = feeds.find(input.name());
    if (binding == state.bindings_.end() && feed == feeds.end()) {
      bool initializer = false;
      for (const auto &tensor : state.model_.graph().initializer())
        initializer = initializer || tensor.name() == input.name();
      for (const auto &value : state.model_.graph().extended_initializer())
        initializer = initializer || value.name() == input.name();
      EXT_ENFORCE_INVALID(initializer, "PersistentValueState: missing current input '",
                          input.name(), "'.");
      continue;
    }
    RuntimeValue value =
        binding != state.bindings_.end()
            ? state.values_[static_cast<size_t>(binding - state.bindings_.begin())].BorrowView()
            : feed->second.BorrowView();
    Validate(value, input.type(), state.catalogue_, symbols, state.sequence_validation_);
    if (IsTensorSequence(input.type()) && IsDenseTensorSequence(value))
      context.PutSequence(input.name(),
                          ToTensorSequence(input.name(), value, input.type().sequence_type()));
    else
      context.PutValue(input.name(), std::move(value), RuntimeEventKind::kInput);
  }
  for (auto &binding : *context.persistent_tensors_)
    binding.input_view = &context.Get(binding.input);
  started = true;
  Run(context);
  RuntimeValueMap outputs;
  for (const auto &output : state.model_.graph().output()) {
    RuntimeValue value;
    if (context.Has(output.name()))
      value = RuntimeValue(std::move(context.Get(output.name())));
    else if (IsTensorSequence(output.type())) {
      auto sequence = context.sequences().find(output.name());
      if (sequence != context.sequences().end())
        value = FromTensorSequence(std::move(sequence->second));
      else {
        auto structured = context.values().find(output.name());
        EXT_ENFORCE_INVALID(structured != context.values().end(),
                            "PersistentValueState: missing output '", output.name(), "'.");
        value = std::move(structured->second);
      }
    } else {
      auto it = context.values().find(output.name());
      EXT_ENFORCE_INVALID(it != context.values().end(), "PersistentValueState: missing output '",
                          output.name(), "'.");
      value = std::move(it->second);
    }
    Validate(value, output.type(), state.catalogue_, symbols, state.sequence_validation_);
    outputs.emplace(output.name(), std::move(value));
  }
  std::vector<PersistentValue> next;
  next.reserve(state.bindings_.size());
  Symbols next_symbols;
  for (const auto &binding : state.bindings_) {
    RuntimeValue &value = outputs.at(binding.output);
    Validate(value, *binding.input_type, state.catalogue_, next_symbols,
             state.sequence_validation_);
    value = std::move(value).Retain(state.catalogue_);
    auto candidate = std::find_if(context.persistent_tensors_->begin(),
                                  context.persistent_tensors_->end(), [&](const auto &item) {
                                    return item.input == binding.input &&
                                           item.output == binding.output && item.candidate &&
                                           value.kind == RuntimeValue::Kind::kTensor &&
                                           item.candidate->Matches(value.tensor);
                                  });
    if (candidate != context.persistent_tensors_->end())
      next.emplace_back(std::move(*candidate->candidate));
    else
      next.emplace_back(value.BorrowView(), state.catalogue_);
  }
  if (completion != nullptr) {
    EXT_ENFORCE_INVALID(completion->status() == TaskStatus::kPending,
                        "PersistentValueState: invocation was cancelled.");
    // Succeed and Cancel serialize on TaskCompletion's mutex. A lost race throws
    // before the noexcept swap, leaving the retained state untouched.
    completion->Succeed();
  }
  state.values_.swap(next);
  for (auto &[name, value] : outputs) {
    (void)name;
    RetainIOBackedStorage(value);
  }
  return outputs;
}

void PersistentValueState::Reset(RuntimeValueMap initial) {
  const Operation operation(busy_);
  auto next = ValidateInitial(std::move(initial));
  values_.swap(next);
}

RuntimeValueMap PersistentValueState::Values() const {
  const Operation operation(busy_);
  RuntimeValueMap snapshot;
  for (size_t i = 0; i < bindings_.size(); ++i)
    snapshot.emplace(bindings_[i].input, values_[i].BorrowView());
  return snapshot;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
