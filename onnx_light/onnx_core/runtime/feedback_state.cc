// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "feedback_state.h"
#include "onnx_core/runtime/kernels/run_nodes.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

using Symbols = std::unordered_map<std::string, int64_t>;
using Declarations = utils::RepeatedProtoField<ValueInfoProto>;

const ModelProto &RequireModel(const std::shared_ptr<const ModelProto> &model) {
  EXT_ENFORCE_INVALID(model != nullptr, "FeedbackState: model owner must not be null.");
  return *model;
}

void CheckNestedBindings(const utils::RepeatedProtoField<NodeProto> &nodes, size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "FeedbackState: maximum subgraph depth exceeded.");
  const auto check = [depth](const GraphProto &graph) {
    EXT_ENFORCE_INVALID(graph.persistent_bindings().empty(),
                        "FeedbackState: persistent bindings are restricted to the model root.");
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
  EXT_THROW_INVALID("FeedbackState: unknown graph input name '", name,
                    "'; names are literal and partial field paths are unsupported.");
}

void Validate(const RuntimeValue &value, const TypeProto &type,
              const StructTypeCatalogue &catalogue, Symbols &symbols) {
  if (value.kind == RuntimeValue::Kind::kTensor) {
    EXT_ENFORCE_INVALID(type.has_tensor_type(), "FeedbackState: expected a structured value.");
    const auto &declared = type.tensor_type();
    const Tensor &tensor = value.tensor;
    EXT_ENFORCE_INVALID(tensor.data_type == declared.elem_type(), "FeedbackState: dtype mismatch.");
    const int64_t count = tensor.shape.product(0, tensor.shape.size(), "FeedbackState");
    if (tensor.data_type == DataType::STRING) {
      EXT_ENFORCE_INVALID(tensor.AsStrings().size() == static_cast<size_t>(count),
                          "FeedbackState: string tensor extent mismatch.");
    } else {
      EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(tensor.data_type, count) &&
                              (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                          "FeedbackState: tensor byte extent mismatch.");
    }
    if (declared.has_shape()) {
      EXT_ENFORCE_INVALID(static_cast<size_t>(declared.shape().dim_size()) == tensor.shape.size(),
                          "FeedbackState: rank mismatch.");
      for (size_t i = 0; i < tensor.shape.size(); ++i) {
        const auto &dim = declared.shape().dim(static_cast<int>(i));
        EXT_ENFORCE_INVALID(!dim.has_dim_value() || dim.dim_value() == tensor.shape[i],
                            "FeedbackState: shape mismatch.");
        if (dim.has_dim_param() && !dim.dim_param().empty()) {
          auto [it, inserted] = symbols.emplace(dim.dim_param(), tensor.shape[i]);
          EXT_ENFORCE_INVALID(inserted || it->second == tensor.shape[i],
                              "FeedbackState: symbolic shape mismatch.");
        }
      }
    }
    return;
  }
  EXT_ENFORCE_INVALID(type.has_struct_type(), "FeedbackState: expected a tensor.");
  const auto &declared = catalogue.Resolve(type.struct_type());
  if (value.kind == RuntimeValue::Kind::kEncoded) {
    const auto layout = catalogue.ValidateEncodedValue(value.Encoded());
    EXT_ENFORCE_INVALID(!layout.external && layout.content_verified &&
                            value.Encoded().has_struct_type(),
                        "FeedbackState: requires an inline structured encoded payload.");
    const auto &actual = catalogue.Resolve(value.Encoded().struct_type());
    EXT_ENFORCE_INVALID(CompatiblePersistentStructTypes(catalogue, actual, declared),
                        "FeedbackState: encoded representation mismatch.");
    return;
  }
  EXT_ENFORCE_INVALID(declared.has_structure(), "FeedbackState: requires named structured fields.");
  size_t expected = 0;
  for (const auto &field : declared.structure().field()) {
    if (!field.has_type())
      continue;
    ++expected;
    auto it = value.fields.find(field.name());
    EXT_ENFORCE_INVALID(it != value.fields.end(), "FeedbackState: missing field '", field.name(),
                        "'.");
    Validate(it->second, field.type(), catalogue, symbols);
  }
  EXT_ENFORCE_INVALID(value.fields.size() == expected,
                      "FeedbackState: unexpected structured field.");
}

class Operation {
public:
  explicit Operation(std::atomic_flag &busy) : busy_(busy) {
    EXT_ENFORCE_INVALID(!busy_.test_and_set(std::memory_order_acquire),
                        "FeedbackState: simultaneous operations are not allowed.");
  }
  ~Operation() { busy_.clear(std::memory_order_release); }
  Operation(const Operation &) = delete;
  Operation &operator=(const Operation &) = delete;

private:
  std::atomic_flag &busy_;
};

} // namespace

FeedbackState::FeedbackState(const ModelProto &model, RuntimeValueMap initial,
                             RuntimeSessionOptions options, std::shared_ptr<void> model_owner)
    : model_(model),
      model_owner_(model_owner.use_count() != 0
                       ? std::move(model_owner)
                       : std::shared_ptr<void>(const_cast<ModelProto *>(&model), [](void *) {})) {
  catalogue_.Build(model);
  CheckNestedBindings(model.graph().node());
  for (const auto &function : model.functions())
    CheckNestedBindings(function.node());
  EXT_ENFORCE_INVALID(!model.graph().persistent_bindings().empty(),
                      "FeedbackState: missing graph persistence declarations.");
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

FeedbackState::FeedbackState(std::shared_ptr<const ModelProto> model, RuntimeValueMap initial,
                             RuntimeSessionOptions options)
    : FeedbackState(RequireModel(model), std::move(initial), std::move(options),
                    std::shared_ptr<void>(model, const_cast<ModelProto *>(model.get()))) {}

std::vector<PersistentValue> FeedbackState::ValidateInitial(RuntimeValueMap initial) const {
  std::vector<PersistentValue> result;
  result.reserve(bindings_.size());
  Symbols symbols;
  for (const auto &binding : bindings_) {
    const auto it = initial.find(binding.input);
    EXT_ENFORCE_INVALID(it != initial.end(), "FeedbackState: missing initial whole input '",
                        binding.input, "'.");
    Validate(it->second, *binding.input_type, catalogue_, symbols);
    result.emplace_back(std::move(it->second), catalogue_);
  }
  EXT_ENFORCE_INVALID(initial.size() == bindings_.size(),
                      "FeedbackState: initial values must name exactly the retained whole inputs.");
  return result;
}

RuntimeValueMap FeedbackState::Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                                   const TaskCompletion *completion) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  EXT_ENFORCE_INVALID(completion == nullptr || completion->status() == TaskStatus::kPending,
                      "FeedbackState: invocation was cancelled or completion is not pending.");
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
                        "FeedbackState: current feed cannot override retained input '", name, "'.");
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
      EXT_ENFORCE_INVALID(initializer, "FeedbackState: missing current input '", input.name(),
                          "'.");
      continue;
    }
    Validate(it->second, input.type(), catalogue_, symbols);
    if (it->second.kind == RuntimeValue::Kind::kTensor)
      invocation.Put(input.name(), std::move(it->second.tensor), RuntimeEventKind::kInput);
    else
      invocation.values().emplace(input.name(), std::move(it->second));
  }
  for (auto &binding : *invocation.persistent_tensors_)
    binding.input_view = &invocation.Get(binding.input);
  if (allocators_captured_) {
    EXT_ENFORCE_INVALID(execution_allocator_ == context.execution_allocator() &&
                            io_allocator_ == context.io_allocator(),
                        "FeedbackState: context allocators changed between calls.");
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
      EXT_ENFORCE_INVALID(it != invocation.values().end(), "FeedbackState: missing output '",
                          output.name(), "'.");
      value = std::move(it->second);
    }
    Validate(value, output.type(), catalogue_, symbols);
    outputs.emplace(output.name(), std::move(value));
  }
  std::vector<PersistentValue> next;
  next.reserve(bindings_.size());
  Symbols next_symbols;
  for (const auto &binding : bindings_) {
    RuntimeValue &value = outputs.at(binding.output);
    Validate(value, *binding.input_type, catalogue_, next_symbols);
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
                        "FeedbackState: invocation was cancelled.");
    // Succeed and Cancel serialize on TaskCompletion's mutex. A lost race throws
    // before the noexcept swap, leaving the retained state untouched.
    completion->Succeed();
  }
  values_.swap(next);
  return outputs;
}

void FeedbackState::Reset(RuntimeValueMap initial) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  auto next = ValidateInitial(std::move(initial));
  values_.swap(next);
}

void FeedbackState::RetainOwner(std::shared_ptr<void> owner) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  EXT_ENFORCE_INVALID(owner.use_count() != 0, "FeedbackState: owner token must retain a lifetime.");
  for (const auto &retained : retained_owners_)
    if (!owner.owner_before(retained) && !retained.owner_before(owner))
      return;
  retained_owners_.push_back(std::move(owner));
}

void FeedbackState::Close() {
  const Operation operation(busy_);
  values_.clear();
  session_.reset();
  retained_owners_.clear();
  model_owner_.reset();
}

RuntimeValueMap FeedbackState::Values() const {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  RuntimeValueMap snapshot;
  for (size_t i = 0; i < bindings_.size(); ++i)
    snapshot.emplace(bindings_[i].input, values_[i].BorrowView());
  return snapshot;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
