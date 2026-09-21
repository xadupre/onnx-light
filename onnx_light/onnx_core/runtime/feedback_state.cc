// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "feedback_state.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include <algorithm>

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

std::string PathKey(const std::vector<std::string> &parts) {
  std::string result;
  for (const auto &part : parts) {
    if (!result.empty())
      result.push_back('.');
    for (char character : part) {
      if (character == '.' || character == '\\')
        result.push_back('\\');
      result.push_back(character);
    }
  }
  return result;
}

std::vector<std::string> Path(const std::string &path, const Declarations &declarations) {
  std::vector<std::string> result;
  std::string part;
  bool escaped = false;
  for (char character : path) {
    if (escaped) {
      EXT_ENFORCE_INVALID(character == '.' || character == '\\',
                          "FeedbackState: only dots and backslashes may be escaped in selectors.");
      part.push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '.') {
      EXT_ENFORCE_INVALID(!part.empty(), "FeedbackState: empty path component.");
      result.push_back(std::move(part));
      EXT_ENFORCE_INVALID(result.size() <= RuntimeValue::kMaxDepth,
                          "FeedbackState: maximum selector depth exceeded.");
      part.clear();
    } else {
      part.push_back(character);
    }
  }
  EXT_ENFORCE_INVALID(!escaped && !part.empty(), "FeedbackState: incomplete or empty selector.");
  result.push_back(std::move(part));
  bool known_root = false;
  for (const auto &item : declarations)
    known_root = known_root || item.name() == result.front();
  EXT_ENFORCE_INVALID(known_root, "FeedbackState: unknown graph root '", result.front(),
                      "'; escape literal dots and backslashes in graph names.");
  return result;
}

bool OverlappingPaths(const std::vector<std::string> &left, const std::vector<std::string> &right) {
  return left.size() <= right.size() && std::equal(left.begin(), left.end(), right.begin());
}

const TypeProto &Resolve(const std::vector<std::string> &parts, const Declarations &declarations,
                         const StructTypeCatalogue &catalogue) {
  const TypeProto *type = nullptr;
  for (const auto &item : declarations)
    if (item.name() == parts.front())
      type = &item.type();
  EXT_ENFORCE_INVALID(type != nullptr, "FeedbackState: unknown graph name '", parts.front(), "'.");
  for (size_t i = 1; i < parts.size(); ++i) {
    EXT_ENFORCE_INVALID(type->has_struct_type(), "FeedbackState: non-struct path.");
    const auto &structure = catalogue.Resolve(type->struct_type());
    EXT_ENFORCE_INVALID(structure.has_structure(), "FeedbackState: path is not a named structure.");
    type = nullptr;
    for (const auto &field : structure.structure().field())
      if (field.name() == parts[i] && field.has_type())
        type = &field.type();
    EXT_ENFORCE_INVALID(type != nullptr, "FeedbackState: unknown field '", parts[i], "'.");
  }
  return *type;
}

void Compatible(const TypeProto &input, const TypeProto &output,
                const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(CompatiblePersistentTypes(catalogue, input, output),
                      "FeedbackState: incompatible binding types.");
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

const RuntimeValue &Select(const RuntimeValueMap &values, const std::vector<std::string> &path) {
  auto it = values.find(path.front());
  EXT_ENFORCE_INVALID(it != values.end(), "FeedbackState: missing output '", path.front(), "'.");
  const RuntimeValue *value = &it->second;
  for (size_t i = 1; i < path.size(); ++i) {
    EXT_ENFORCE_INVALID(value->kind == RuntimeValue::Kind::kStruct,
                        "FeedbackState: cannot select fields from an encoded payload.");
    auto field = value->fields.find(path[i]);
    EXT_ENFORCE_INVALID(field != value->fields.end(), "FeedbackState: missing output field.");
    value = &field->second;
  }
  return *value;
}

void Insert(RuntimeValueMap &values, const std::vector<std::string> &path,
            const RuntimeValue &value) {
  RuntimeValue *target = &values[path.front()];
  for (size_t i = 1; i < path.size(); ++i) {
    EXT_ENFORCE_INVALID(target->kind == RuntimeValue::Kind::kStruct,
                        "FeedbackState: overlapping current feeds.");
    target = &target->fields[path[i]];
  }
  *target = value.Share();
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

FeedbackState::FeedbackState(const ModelProto &model, const RuntimeValueMap &initial,
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
  for (const auto &declared : model.graph().persistent_bindings()) {
    EXT_ENFORCE_INVALID(!declared.input_name().empty() && !declared.output_name().empty(),
                        "FeedbackState: binding graph names must not be empty.");
    Binding binding{
        declared.input_name(), {declared.input_name()}, {declared.output_name()}, nullptr};
    for (const auto &field : declared.input_field_path()) {
      EXT_ENFORCE_INVALID(!field.empty(), "FeedbackState: empty input field name.");
      binding.input.push_back(field);
    }
    for (const auto &field : declared.output_field_path()) {
      EXT_ENFORCE_INVALID(!field.empty(), "FeedbackState: empty output field name.");
      binding.output.push_back(field);
    }
    binding.key = PathKey(binding.input);
    binding.input_type = &Resolve(binding.input, model.graph().input(), catalogue_);
    Compatible(*binding.input_type, Resolve(binding.output, model.graph().output(), catalogue_),
               catalogue_);
    for (const auto &other : bindings_) {
      EXT_ENFORCE_INVALID(!OverlappingPaths(binding.input, other.input) &&
                              !OverlappingPaths(other.input, binding.input),
                          "FeedbackState: overlapping destination paths.");
    }
    bindings_.push_back(std::move(binding));
  }
  session_ = std::make_unique<RuntimeSession>(model, std::move(options),
                                              RuntimeSession::InitializerMode::kBorrowed);
  values_ = ValidateInitial(initial);
}

FeedbackState::FeedbackState(std::shared_ptr<const ModelProto> model,
                             const RuntimeValueMap &initial, RuntimeSessionOptions options)
    : FeedbackState(RequireModel(model), initial, std::move(options),
                    std::shared_ptr<void>(model, const_cast<ModelProto *>(model.get()))) {}

std::vector<RuntimeValue> FeedbackState::ValidateInitial(const RuntimeValueMap &initial) const {
  std::vector<RuntimeValue> result(bindings_.size());
  std::vector<bool> supplied(bindings_.size(), false);
  Symbols symbols;
  const auto insert = [&](auto &&self, const RuntimeValue &value,
                          const std::vector<std::string> &path) -> void {
    EXT_ENFORCE_INVALID(path.size() <= RuntimeValue::kMaxDepth + 1,
                        "FeedbackState: maximum initial path depth exceeded.");
    bool ancestor = false;
    for (size_t i = 0; i < bindings_.size(); ++i) {
      const auto &binding = bindings_[i];
      if (path == binding.input) {
        EXT_ENFORCE_INVALID(!supplied[i], "FeedbackState: duplicate initial destination.");
        Validate(value, *binding.input_type, catalogue_, symbols);
        result[i] = value.Share();
        supplied[i] = true;
        return;
      }
      ancestor = ancestor || OverlappingPaths(path, binding.input);
    }
    EXT_ENFORCE_INVALID(ancestor && value.kind == RuntimeValue::Kind::kStruct &&
                            !value.fields.empty(),
                        "FeedbackState: initial value does not select a persistent destination; "
                        "escape literal dots and backslashes in selector components.");
    for (const auto &[field, child] : value.fields) {
      auto nested = path;
      nested.push_back(field);
      self(self, child, nested);
    }
  };
  for (const auto &[name, value] : initial) {
    insert(insert, value, Path(name, model_.graph().input()));
  }
  EXT_ENFORCE_INVALID(
      std::all_of(supplied.begin(), supplied.end(), [](bool value) { return value; }),
      "FeedbackState: missing initial values.");
  return result;
}

RuntimeValueMap FeedbackState::Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                                   const TaskCompletion *completion) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  EXT_ENFORCE_INVALID(completion == nullptr || completion->status() == TaskStatus::kPending,
                      "FeedbackState: invocation was cancelled or completion is not pending.");
  RuntimeValueMap inputs;
  std::vector<std::vector<std::string>> feed_paths;
  Symbols feed_symbols;
  const auto insert_feed = [&](auto &&self, const RuntimeValue &value,
                               const std::vector<std::string> &path) -> void {
    EXT_ENFORCE_INVALID(path.size() <= RuntimeValue::kMaxDepth + 1,
                        "FeedbackState: maximum feed path depth exceeded.");
    bool ancestor = false;
    for (const auto &binding : bindings_) {
      EXT_ENFORCE_INVALID(!OverlappingPaths(binding.input, path),
                          "FeedbackState: current feed overlaps retained state.");
      ancestor = ancestor || OverlappingPaths(path, binding.input);
    }
    if (ancestor) {
      EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kStruct,
                          "FeedbackState: partial root feeds require a structured map.");
      for (const auto &[field, child] : value.fields) {
        auto nested = path;
        nested.push_back(field);
        self(self, child, nested);
      }
      return;
    }
    Validate(value, Resolve(path, model_.graph().input(), catalogue_), catalogue_, feed_symbols);
    for (const auto &other : feed_paths)
      EXT_ENFORCE_INVALID(!OverlappingPaths(path, other) && !OverlappingPaths(other, path),
                          "FeedbackState: overlapping current feeds.");
    feed_paths.push_back(path);
    Insert(inputs, path, value);
  };
  for (const auto &[name, value] : feeds)
    insert_feed(insert_feed, value, Path(name, model_.graph().input()));
  for (size_t i = 0; i < bindings_.size(); ++i)
    Insert(inputs, bindings_[i].input, values_[i]);
  Symbols symbols;
  RuntimeContext invocation = context.MakeFunctionContext();
  invocation.set_preserve_value_ownership(true);
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
      invocation.Put(input.name(), std::move(it->second.tensor));
    else
      invocation.values().emplace(input.name(), std::move(it->second));
  }
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
  std::vector<RuntimeValue> next;
  next.reserve(bindings_.size());
  Symbols next_symbols;
  for (const auto &binding : bindings_) {
    const RuntimeValue &value = Select(outputs, binding.output);
    Validate(value, *binding.input_type, catalogue_, next_symbols);
    next.push_back(value.Share());
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

void FeedbackState::Reset(const RuntimeValueMap &initial) {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  auto next = ValidateInitial(initial);
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
    snapshot.emplace(bindings_[i].key, values_[i].Share());
  return snapshot;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
