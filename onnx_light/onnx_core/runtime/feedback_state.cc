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

bool Overlaps(const std::string &a, const std::string &b) {
  return a == b || (b.size() > a.size() && b.compare(0, a.size(), a) == 0 && b[a.size()] == '.');
}

std::vector<std::string> Path(const std::string &path, const Declarations &declarations) {
  std::vector<std::string> result;
  // Exact graph names take precedence; otherwise the longest graph-name prefix wins.
  for (const auto &item : declarations) {
    const std::string name = item.name();
    if (Overlaps(name, path) && (result.empty() || name.size() > result.front().size()))
      result = {name};
  }
  EXT_ENFORCE_INVALID(!result.empty(), "FeedbackState: unknown path '", path, "'.");
  size_t pos = result.front().size();
  while (pos < path.size()) {
    size_t end = path.find('.', pos + 1);
    if (end == std::string::npos)
      end = path.size();
    EXT_ENFORCE_INVALID(end > pos + 1, "FeedbackState: empty path component.");
    result.push_back(path.substr(pos + 1, end - pos - 1));
    pos = end;
  }
  return result;
}

bool OverlappingPaths(const std::string &a, const std::string &b,
                      const Declarations &declarations) {
  const auto left = Path(a, declarations);
  const auto right = Path(b, declarations);
  return left.size() <= right.size() && std::equal(left.begin(), left.end(), right.begin());
}

const TypeProto &Resolve(const std::string &path, const Declarations &declarations,
                         const StructTypeCatalogue &catalogue) {
  const auto parts = Path(path, declarations);
  const TypeProto *type = nullptr;
  for (const auto &item : declarations)
    if (item.name() == parts.front())
      type = &item.type();
  for (size_t i = 1; i < parts.size(); ++i) {
    EXT_ENFORCE_INVALID(type->has_struct_type(), "FeedbackState: non-struct path '", path, "'.");
    const auto &structure = catalogue.Resolve(type->struct_type());
    EXT_ENFORCE_INVALID(structure.has_structure(), "FeedbackState: path is not a named structure.");
    type = nullptr;
    for (const auto &field : structure.structure().field())
      if (field.name() == parts[i] && field.has_type())
        type = &field.type();
    EXT_ENFORCE_INVALID(type != nullptr, "FeedbackState: unknown field in '", path, "'.");
  }
  return *type;
}

void Compatible(const TypeProto &input, const TypeProto &output,
                const StructTypeCatalogue &catalogue) {
  if (input.has_tensor_type() && output.has_tensor_type()) {
    const auto &a = input.tensor_type();
    const auto &b = output.tensor_type();
    EXT_ENFORCE_INVALID(a.elem_type() == b.elem_type(), "FeedbackState: incompatible dtype.");
    if (a.has_shape() && b.has_shape()) {
      EXT_ENFORCE_INVALID(a.shape().dim_size() == b.shape().dim_size(),
                          "FeedbackState: incompatible rank.");
      for (int i = 0; i < a.shape().dim_size(); ++i) {
        const auto &x = a.shape().dim(i);
        const auto &y = b.shape().dim(i);
        EXT_ENFORCE_INVALID(!x.has_dim_value() || !y.has_dim_value() ||
                                x.dim_value() == y.dim_value(),
                            "FeedbackState: incompatible shape.");
      }
    }
    return;
  }
  EXT_ENFORCE_INVALID(input.has_struct_type() && output.has_struct_type(),
                      "FeedbackState: unsupported or incompatible value types.");
  const auto &a = catalogue.Resolve(input.struct_type());
  const auto &b = catalogue.Resolve(output.struct_type());
  if (!a.has_structure() || !b.has_structure()) {
    EXT_ENFORCE_INVALID(a.SerializeAsString() == b.SerializeAsString(),
                        "FeedbackState: incompatible encoded representation.");
    return;
  }
  EXT_ENFORCE_INVALID(a.structure().field_size() == b.structure().field_size(),
                      "FeedbackState: incompatible structured fields.");
  for (int i = 0; i < a.structure().field_size(); ++i) {
    const auto &x = a.structure().field(i);
    const auto &y = b.structure().field(i);
    EXT_ENFORCE_INVALID(x.name() == y.name() && x.has_type() == y.has_type(),
                        "FeedbackState: incompatible structured field.");
    if (x.has_type())
      Compatible(x.type(), y.type(), catalogue);
    else
      EXT_ENFORCE_INVALID(x.constant().SerializeAsString() == y.constant().SerializeAsString(),
                          "FeedbackState: incompatible representation constant.");
  }
  EXT_ENFORCE_INVALID(
      a.has_encoder() == b.has_encoder() && a.has_decoder() == b.has_decoder() &&
          (!a.has_encoder() ||
           a.encoder().SerializeAsString() == b.encoder().SerializeAsString()) &&
          (!a.has_decoder() || a.decoder().SerializeAsString() == b.decoder().SerializeAsString()),
      "FeedbackState: incompatible representation codecs.");
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
    const auto layout = catalogue.ValidateEncodedValue(value.encoded);
    EXT_ENFORCE_INVALID(!layout.external && layout.content_verified &&
                            value.encoded.has_struct_type(),
                        "FeedbackState: requires an inline structured encoded payload.");
    const auto &actual = catalogue.Resolve(value.encoded.struct_type());
    EXT_ENFORCE_INVALID(actual.SerializeAsString() == declared.SerializeAsString(),
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
  *target = value.DeepCopy();
}

void DetachOutputAllocations(RuntimeValue &value, size_t depth = 0) {
  EXT_ENFORCE_INVALID(depth <= RuntimeValue::kMaxDepth,
                      "FeedbackState: maximum output depth exceeded.");
  if (value.kind == RuntimeValue::Kind::kTensor) {
    if (value.tensor.has_allocation())
      value.tensor = value.tensor.ToOwned();
  } else if (value.kind == RuntimeValue::Kind::kStruct) {
    for (auto &[name, field] : value.fields)
      DetachOutputAllocations(field, depth + 1);
  }
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

FeedbackState::FeedbackState(const ModelProto &model, FeedbackBindings bindings,
                             const RuntimeValueMap &initial, RuntimeSessionOptions options)
    : model_(model), model_snapshot_(model.SerializeAsString()), bindings_(std::move(bindings)) {
  catalogue_.Build(model);
  EXT_ENFORCE_INVALID(!bindings_.empty(), "FeedbackState: missing feedback mappings.");
  for (const auto &input : model.graph().input())
    catalogue_.ValidateType(input.type());
  for (const auto &output : model.graph().output())
    catalogue_.ValidateType(output.type());
  for (const auto &[input, output] : bindings_) {
    Compatible(Resolve(input, model.graph().input(), catalogue_),
               Resolve(output, model.graph().output(), catalogue_), catalogue_);
    for (const auto &[other, ignored] : bindings_)
      EXT_ENFORCE_INVALID(input == other || !OverlappingPaths(input, other, model.graph().input()),
                          "FeedbackState: overlapping destination paths.");
  }
  session_ = std::make_unique<RuntimeSession>(model, std::move(options));
  values_ = ValidateInitial(initial);
}

void FeedbackState::CheckModel() const {
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  EXT_ENFORCE_INVALID(model_.SerializeAsString() == model_snapshot_,
                      "FeedbackState: model changed after feedback paths were resolved.");
}

RuntimeValueMap FeedbackState::ValidateInitial(const RuntimeValueMap &initial) const {
  EXT_ENFORCE_INVALID(initial.size() == bindings_.size(), "FeedbackState: missing initial values.");
  RuntimeValueMap result;
  Symbols symbols;
  for (const auto &[input, output] : bindings_) {
    auto value = initial.find(input);
    EXT_ENFORCE_INVALID(value != initial.end(), "FeedbackState: missing initial value '", input,
                        "'.");
    Validate(value->second, Resolve(input, model_.graph().input(), catalogue_), catalogue_,
             symbols);
    result.emplace(input, value->second.DeepCopy());
  }
  return result;
}

RuntimeValueMap FeedbackState::Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                                   const TaskCompletion *completion) {
  const Operation operation(busy_);
  CheckModel();
  EXT_ENFORCE_INVALID(completion == nullptr || completion->status() == TaskStatus::kPending,
                      "FeedbackState: invocation was cancelled or completion is not pending.");
  RuntimeValueMap inputs;
  for (const auto &[name, value] : feeds) {
    Symbols feed_symbols;
    Validate(value, Resolve(name, model_.graph().input(), catalogue_), catalogue_, feed_symbols);
    for (const auto &[destination, ignored] : bindings_)
      EXT_ENFORCE_INVALID(!OverlappingPaths(name, destination, model_.graph().input()) &&
                              !OverlappingPaths(destination, name, model_.graph().input()),
                          "FeedbackState: current feed overlaps retained state.");
    for (const auto &[other, ignored] : feeds)
      EXT_ENFORCE_INVALID(name == other || !OverlappingPaths(name, other, model_.graph().input()),
                          "FeedbackState: overlapping current feeds.");
    Insert(inputs, Path(name, model_.graph().input()), value);
  }
  for (const auto &[name, value] : values_)
    Insert(inputs, Path(name, model_.graph().input()), value);
  Symbols symbols;
  RuntimeContext invocation = context.MakeFunctionContext();
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
    // Run already detached borrowed tensors and encoded payloads. Only arena
    // allocations still need copying; inline output storage transfers as-is.
    DetachOutputAllocations(value);
    Validate(value, output.type(), catalogue_, symbols);
    outputs.emplace(output.name(), std::move(value));
  }
  RuntimeValueMap next;
  Symbols next_symbols;
  for (const auto &[input, output] : bindings_) {
    const auto path = Path(output, model_.graph().output());
    const RuntimeValue &value = Select(outputs, path);
    Validate(value, Resolve(input, model_.graph().input(), catalogue_), catalogue_, next_symbols);
    next.emplace(input, value.DeepCopy());
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
  CheckModel();
  RuntimeValueMap next = ValidateInitial(initial);
  values_.swap(next);
}

void FeedbackState::Close() {
  const Operation operation(busy_);
  values_.clear();
  session_.reset();
}

RuntimeValueMap FeedbackState::Values() const {
  const Operation operation(busy_);
  EXT_ENFORCE_INVALID(session_ != nullptr, "FeedbackState: state is closed.");
  RuntimeValueMap snapshot;
  for (const auto &[name, value] : values_)
    snapshot.emplace(name, value.DeepCopy());
  return snapshot;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
