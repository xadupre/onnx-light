// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

// Recursively builds the TypeProto ``tp`` from ``spec``.
void BuildTypeProto(const TypeSpec &spec, TypeProto &tp) {
  switch (spec.kind) {
  case TypeSpec::Kind::kTensor: {
    TypeProto::Tensor *tt = tp.add_tensor_type();
    tt->set_elem_type(spec.elem_type);
    if (spec.has_shape) {
      TensorShapeProto *sh = tt->add_shape();
      for (int64_t d : spec.shape) {
        sh->add_dim()->set_dim_value(d);
      }
    }
    break;
  }
  case TypeSpec::Kind::kSequence: {
    TypeProto::Sequence *seq = tp.add_sequence_type();
    BuildTypeProto(spec.children.front(), *seq->add_elem_type());
    break;
  }
  case TypeSpec::Kind::kMap: {
    TypeProto::Map *mp = tp.add_map_type();
    mp->set_key_type(spec.elem_type);
    BuildTypeProto(spec.children.front(), *mp->add_value_type());
    break;
  }
  }
}

} // namespace

std::string_view TestCaseKindName(TestCaseKind kind) {
  switch (kind) {
  case TestCaseKind::NODE:
    return "node";
  case TestCaseKind::MODEL:
    return "model";
  }
  throw std::invalid_argument("Unknown TestCaseKind value.");
}

std::string_view TestCaseTagName(TestCaseTag tag) {
  switch (tag) {
  case TestCaseTag::NONE:
    return "";
  case TestCaseTag::AI_ONNX_ML:
    return "ai.onnx.ml";
  case TestCaseTag::AI_ONNX_PREVIEW:
    return "ai.onnx.preview";
  case TestCaseTag::AI_ONNX_PREVIEW_TRAINING:
    return "ai.onnx.preview.training";
  case TestCaseTag::AI_RT:
    return "ai.rt";
  case TestCaseTag::CONSTANT:
    return "constant";
  case TestCaseTag::EMPTY_SHAPE:
    return "empty_shape";
  case TestCaseTag::INFERENCE:
    return "inference";
  case TestCaseTag::INPLACE:
    return "inplace";
  case TestCaseTag::LOCAL_FUNCTION:
    return "local_function";
  case TestCaseTag::NAN_INF:
    return "nan_inf";
  case TestCaseTag::PEAK_MEMORY:
    return "peak_memory";
  case TestCaseTag::RELEASE:
    return "release";
  case TestCaseTag::SHAPE_TAG:
    return "shape_tag";
  }
  throw std::invalid_argument("Unknown TestCaseTag value.");
}

TestCase::TestCase(TestCase &&other) noexcept
    : name(std::move(const_cast<std::string &>(other.name))),
      model_name(std::move(const_cast<std::string &>(other.model_name))), kind(other.kind),
      tag(other.tag), rtol(other.rtol), atol(other.atol), build(std::move(other.build)),
      expected_outputs_generated(other.expected_outputs_generated),
      declared_input_element_counts(std::move(other.declared_input_element_counts)),
      declared_output_element_counts(std::move(other.declared_output_element_counts)),
      data_sets_(std::move(other.data_sets_)), model_(std::move(other.model_)),
      retained_(std::move(other.retained_)), rebuild_(std::move(other.rebuild_)) {}

ModelProto &TestCase::emplace_model() {
  model_ = std::make_shared<ModelProto>();
  return *model_;
}

void TestCase::set_model(ModelProto model) {
  model_ = std::make_shared<ModelProto>(std::move(model));
}

void TestCase::set_expected_outputs_generated(bool value) {
  expected_outputs_generated = value;
  if (!value && materialized() && data_sets_) {
    for (DataSet &data_set : *data_sets_) {
      data_set.expected_outputs_generated = false;
      data_set.outputs.clear();
    }
  }
}

void TestCase::set_rebuild(std::function<BuiltCase(bool)> rebuild) {
  rebuild_ = std::move(rebuild);
}

void TestCase::Materialize() {
  if (model_) {
    return;
  }
  if (!build && !rebuild_) {
    return;
  }
  BuiltCase built =
      build ? build(expected_outputs_generated) : rebuild_(expected_outputs_generated);
  model_ = std::make_shared<ModelProto>(std::move(built.model));
  if (!data_sets_) {
    data_sets_ = std::make_shared<std::vector<DataSet>>(std::move(built.data_sets));
  }
  retained_ = std::move(built.retained);
}

void TestCase::unload() {
  if (!build && !rebuild_) {
    throw std::runtime_error("Cannot unload an eager backend test case.");
  }
  model_.reset();
  data_sets_.reset();
  retained_.reset();
  if (rebuild_) {
    build = nullptr;
  }
}

BuiltCase TestCase::take_materialized() {
  if (!model_) {
    throw std::runtime_error("Cannot take an unmaterialized backend test case.");
  }
  BuiltCase built;
  built.model = std::move(*model_);
  if (data_sets_) {
    built.data_sets = std::move(*data_sets_);
  }
  built.retained = std::move(retained_);
  model_.reset();
  data_sets_.reset();
  retained_.reset();
  return built;
}

std::shared_ptr<ModelProto> TestCase::model_handle() {
  model();
  return model_;
}

std::vector<std::shared_ptr<DataSet>> TestCase::data_set_handles() {
  std::shared_ptr<std::vector<DataSet>> data_sets = data_sets_handle();
  std::vector<std::shared_ptr<DataSet>> handles;
  handles.reserve(data_sets->size());
  for (DataSet &data_set : *data_sets) {
    handles.emplace_back(data_sets, &data_set);
  }
  return handles;
}

ModelProto &TestCase::model() {
  EnsureMaterialized();
  if (!model_) {
    model_ = std::make_shared<ModelProto>();
  }
  return *model_;
}

const ModelProto &TestCase::model() const {
  EnsureMaterialized();
  if (!model_) {
    model_ = std::make_shared<ModelProto>();
  }
  return *model_;
}

std::vector<DataSet> &TestCase::data_sets() {
  EnsureMaterialized();
  if (!data_sets_) {
    data_sets_ = std::make_shared<std::vector<DataSet>>();
  }
  return *data_sets_;
}

const std::vector<DataSet> &TestCase::data_sets() const {
  EnsureMaterialized();
  if (!data_sets_) {
    data_sets_ = std::make_shared<std::vector<DataSet>>();
  }
  return *data_sets_;
}

std::shared_ptr<std::vector<DataSet>> TestCase::data_sets_handle() {
  data_sets();
  return data_sets_;
}

void TestCase::EnsureMaterialized() const { const_cast<TestCase *>(this)->Materialize(); }

TypeSpec TensorTypeSpec(int32_t elem_type) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = false;
  return spec;
}

TypeSpec TensorTypeSpec(int32_t elem_type, std::vector<int64_t> shape) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = true;
  spec.shape = std::move(shape);
  return spec;
}

TypeSpec SequenceTypeSpec(TypeSpec elem) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kSequence;
  spec.children.push_back(std::move(elem));
  return spec;
}

TypeSpec MapTypeSpec(int32_t key_type, TypeSpec value) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kMap;
  spec.elem_type = key_type;
  spec.children.push_back(std::move(value));
  return spec;
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, const TypeSpec &spec) {
  vi.set_name(name);
  BuildTypeProto(spec, *vi.add_type());
}

void InitModel(ModelProto &model, int64_t ir_version, const std::vector<OpsetId> &opset_imports,
               const std::string &producer_name) {
  model.set_ir_version(ir_version);
  model.set_producer_name(producer_name);
  for (const auto &osid : opset_imports) {
    OperatorSetIdProto proto;
    proto.set_domain(osid.domain);
    proto.set_version(osid.version);
    model.add_opset_import(proto);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &shape) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (int64_t d : shape) {
    sh->add_dim()->set_dim_value(d);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, TensorProto::DataType elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendDataSet(TestCase &tc, Tensors inputs, Tensors outputs) {
  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
  tc.data_sets().emplace_back(std::move(ds));
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry);
  }
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterModeMap &entries, TestMode mode) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry, mode);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry, mode);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test