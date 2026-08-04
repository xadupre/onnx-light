// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::TestCase;
using DataTensor = core::runtime::Tensor;

namespace Test {

namespace {

// Extracts the dims of a tensor type as a vector of int64_t (or -1 for unknown).
std::vector<int64_t> DimsOf(const TypeProto::Tensor &tt) {
  std::vector<int64_t> out;
  if (!tt.has_shape()) {
    return out;
  }
  const auto &dims = tt.ref_shape().ref_dim();
  out.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    out.push_back(dims[i].has_dim_value() ? dims[i].ref_dim_value() : -1);
  }
  return out;
}

using MetadataMap = std::unordered_map<std::string, std::string>;

template <typename Proto> MetadataMap MetadataOf(const Proto &proto) {
  MetadataMap out;
  for (const auto &prop : proto.ref_metadata_props()) {
    out[prop.ref_key()] = prop.ref_value();
  }
  return out;
}

// Captures the (elem_type, shape) of each graph output before stripping it
// so that we can compare it to what shape inference recovers.
struct ExpectedOutput {
  std::string name;
  int32_t elem_type = 0;
  std::vector<int64_t> shape;
  bool had_shape = false;
};

// Returns the underlying ``TypeProto::Tensor`` carried by ``type``, drilling
// through an ``optional_type`` or ``sequence_type`` wrapper when present.
// Returns ``nullptr`` if the type does not (transitively) carry a tensor type.
TypeProto::Tensor *MutableTensorTypeOf(TypeProto &type) {
  if (type.has_tensor_type()) {
    return type.mutable_tensor_type();
  }
  if (type.has_optional_type() && type.mutable_optional_type()->mutable_elem_type() != nullptr) {
    return MutableTensorTypeOf(*type.mutable_optional_type()->mutable_elem_type());
  }
  if (type.has_sequence_type() && type.mutable_sequence_type()->mutable_elem_type() != nullptr) {
    return MutableTensorTypeOf(*type.mutable_sequence_type()->mutable_elem_type());
  }
  if (type.has_map_type()) {
    return MutableTensorTypeOf(*type.mutable_map_type()->mutable_value_type());
  }
  return nullptr;
}

const TypeProto::Tensor *TensorTypeOf(const TypeProto &type) {
  if (type.has_tensor_type()) {
    return &type.ref_tensor_type();
  }
  if (type.has_optional_type()) {
    return TensorTypeOf(type.ref_optional_type().ref_elem_type());
  }
  if (type.has_sequence_type()) {
    return TensorTypeOf(type.ref_sequence_type().ref_elem_type());
  }
  if (type.has_map_type()) {
    return TensorTypeOf(type.ref_map_type().ref_value_type());
  }
  return nullptr;
}

const TypeProto::Map *MapTypeOf(const TypeProto &type) {
  if (type.has_map_type()) {
    return &type.ref_map_type();
  }
  if (type.has_optional_type()) {
    return MapTypeOf(type.ref_optional_type().ref_elem_type());
  }
  if (type.has_sequence_type()) {
    return MapTypeOf(type.ref_sequence_type().ref_elem_type());
  }
  return nullptr;
}

std::vector<ExpectedOutput> SnapshotAndStripOutputs(ModelProto &model) {
  std::vector<ExpectedOutput> snapshot;
  auto &outputs = model.mutable_graph()->ref_output();
  snapshot.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &out = outputs[i];
    ExpectedOutput exp;
    exp.name.assign(out.ref_name().data(), out.ref_name().size());
    if (out.has_type()) {
      if (auto *tt = MutableTensorTypeOf(*out.mutable_type()); tt != nullptr) {
        exp.elem_type = static_cast<int32_t>(tt->elem_type());
        exp.had_shape = tt->has_shape();
        exp.shape = DimsOf(*tt);
        // Strip the recorded shape so InferShapes has to recover it.
        // We keep elem_type so the ValueInfo remains well-formed.
        tt->clear_shape();
      }
    }
    snapshot.emplace_back(std::move(exp));
  }
  return snapshot;
}

// Same as :ref:`SnapshotAndStripOutputs` but for the intermediate
// ``value_info`` entries declared by the graph. Used for ``kind == "model"``
// test cases that record expected intermediate shapes in ``value_info``: we
// strip them so shape inference must recover them, then compare the snapshot
// to the post-inference ``value_info``.
std::vector<ExpectedOutput> SnapshotAndStripValueInfo(ModelProto &model) {
  std::vector<ExpectedOutput> snapshot;
  auto &value_infos = model.mutable_graph()->ref_value_info();
  snapshot.reserve(value_infos.size());
  for (size_t i = 0; i < value_infos.size(); ++i) {
    auto &vi = value_infos[i];
    ExpectedOutput exp;
    exp.name.assign(vi.ref_name().data(), vi.ref_name().size());
    if (vi.has_type()) {
      if (auto *tt = MutableTensorTypeOf(*vi.mutable_type()); tt != nullptr) {
        exp.elem_type = static_cast<int32_t>(tt->elem_type());
        exp.had_shape = tt->has_shape();
        exp.shape = DimsOf(*tt);
        // Strip the recorded shape so InferShapes has to recover it.
        // We keep elem_type so the ValueInfo remains well-formed.
        tt->clear_shape();
      }
    }
    snapshot.emplace_back(std::move(exp));
  }
  return snapshot;
}

// Verifies that every entry in ``expected`` is present in
// ``graph.value_info`` after shape inference with the same elem_type and a
// compatible shape (concrete dims must match; symbolic / unknown dims are
// tolerated, mirroring the output check above). When the snapshot recorded
// a shape, shape inference must have populated one as well, so we can detect
// intermediate names for which inference produced nothing at all.
void CheckValueInfoMatchesExpected(const GraphProto &graph,
                                   const std::vector<ExpectedOutput> &expected) {
  std::unordered_map<std::string, const ValueInfoProto *> by_name;
  const auto &value_infos = graph.ref_value_info();
  by_name.reserve(value_infos.size());
  for (size_t i = 0; i < value_infos.size(); ++i) {
    const auto &vi = value_infos[i];
    by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
  }

  for (const auto &exp : expected) {
    auto it = by_name.find(exp.name);
    ASSERT_NE(it, by_name.end()) << "value_info " << exp.name
                                 << " missing from graph after shape inference";
    const auto &vi = *it->second;
    ASSERT_TRUE(vi.has_type()) << "value_info " << exp.name << " missing type";
    const TypeProto::Tensor *tt_ptr = TensorTypeOf(vi.ref_type());
    ASSERT_NE(tt_ptr, nullptr) << "value_info " << exp.name << " not a tensor";
    const auto &tt = *tt_ptr;
    EXPECT_EQ(static_cast<int32_t>(tt.elem_type()), exp.elem_type)
        << "elem_type mismatch on value_info " << exp.name;
    if (exp.had_shape) {
      ASSERT_TRUE(tt.has_shape()) << "value_info " << exp.name
                                  << " has no shape after inference (expected rank "
                                  << exp.shape.size() << ")";
      const auto inferred_dims = DimsOf(tt);
      ASSERT_EQ(inferred_dims.size(), exp.shape.size())
          << "rank mismatch on value_info " << exp.name;
      for (size_t d = 0; d < inferred_dims.size(); ++d) {
        if (inferred_dims[d] != -1 && exp.shape[d] != -1) {
          EXPECT_EQ(inferred_dims[d], exp.shape[d])
              << "dim[" << d << "] mismatch on value_info " << exp.name;
        }
      }
    }
  }
}

} // namespace

TEST(BackendTestCaseShapeInference, ZipMapInfersSequenceOfMapsOutputType) {
  ModelProto model;
  model.set_ir_version(9);

  OperatorSetIdProto *default_opset = model.add_opset_import();
  default_opset->set_domain("");
  default_opset->set_version(13);
  OperatorSetIdProto *ml_opset = model.add_opset_import();
  ml_opset->set_domain("ai.onnx.ml");
  ml_opset->set_version(1);

  GraphProto *graph = model.add_graph();
  graph->set_name("zipmap_graph");

  ValueInfoProto *input = graph->add_input();
  input->set_name("X");
  TypeProto *input_type = input->add_type();
  TypeProto::Tensor *input_tt = input_type->add_tensor_type();
  input_tt->set_elem_type(core::runtime::DataType::FLOAT);
  TensorShapeProto *mutable_input_shape = input_tt->add_shape();
  mutable_input_shape->add_dim()->set_dim_value(2);
  mutable_input_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *output = graph->add_output();
  output->set_name("Z");
  // Leave output type empty so shape inference must populate it.
  output->add_type();

  NodeProto *node = graph->add_node();
  node->set_op_type("ZipMap");
  node->set_domain("ai.onnx.ml");
  node->add_input("X");
  node->add_output("Z");
  AttributeProto *labels = node->add_attribute();
  labels->set_name("classlabels_int64s");
  labels->set_type(AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));
  labels->add_ints(static_cast<int64_t>(2));

  ASSERT_NO_THROW(shape_inference::InferShapes(model));

  ASSERT_EQ(graph->ref_output().size(), 1u);
  const TypeProto &out_type = graph->ref_output()[0].ref_type();
  ASSERT_TRUE(out_type.has_sequence_type());
  const TypeProto &seq_elem_type = out_type.ref_sequence_type().ref_elem_type();
  ASSERT_TRUE(seq_elem_type.has_map_type());
  const TypeProto::Map &map_type = seq_elem_type.ref_map_type();
  ASSERT_TRUE(map_type.ref_value_type().has_tensor_type());
  const TypeProto::Tensor &value_tensor = map_type.ref_value_type().ref_tensor_type();
  EXPECT_EQ(value_tensor.ref_elem_type(), core::runtime::DataType::FLOAT);
  ASSERT_TRUE(value_tensor.has_shape());
  EXPECT_EQ(value_tensor.ref_shape().ref_dim().size(), 0u);
}

TEST(BackendTestCaseShapeInference, ZipMapInfersSequenceOfStringKeyMapsOutputType) {
  ModelProto model;
  model.set_ir_version(9);
  constexpr int64_t kClassCount = 3;

  OperatorSetIdProto *default_opset = model.add_opset_import();
  default_opset->set_domain("");
  default_opset->set_version(13);
  OperatorSetIdProto *ml_opset = model.add_opset_import();
  ml_opset->set_domain("ai.onnx.ml");
  ml_opset->set_version(1);

  GraphProto *graph = model.add_graph();
  graph->set_name("zipmap_graph_string_keys");

  ValueInfoProto *input = graph->add_input();
  input->set_name("X");
  TypeProto *input_type = input->add_type();
  TypeProto::Tensor *input_tt = input_type->add_tensor_type();
  input_tt->set_elem_type(core::runtime::DataType::FLOAT);
  TensorShapeProto *mutable_input_shape = input_tt->add_shape();
  mutable_input_shape->add_dim()->set_dim_value(2);
  mutable_input_shape->add_dim()->set_dim_value(kClassCount);

  ValueInfoProto *output = graph->add_output();
  output->set_name("Z");
  // Leave output type empty so shape inference must populate it.
  output->add_type();

  NodeProto *node = graph->add_node();
  node->set_op_type("ZipMap");
  node->set_domain("ai.onnx.ml");
  node->add_input("X");
  node->add_output("Z");
  AttributeProto *labels = node->add_attribute();
  labels->set_name("classlabels_strings");
  labels->set_type(AttributeProto::AttributeType::STRINGS);
  const std::vector<std::string> class_labels = {"label_a", "label_b", "label_c"};
  ASSERT_EQ(static_cast<int64_t>(class_labels.size()), kClassCount);
  auto &label_strings_ref = labels->strings();
  for (const std::string &name : class_labels) {
    label_strings_ref.push_back(utils::String(name));
  }

  ASSERT_NO_THROW(shape_inference::InferShapes(model));

  ASSERT_EQ(graph->ref_output().size(), 1u);
  const TypeProto &out_type = graph->ref_output()[0].ref_type();
  ASSERT_TRUE(out_type.has_sequence_type());
  const TypeProto &seq_elem_type = out_type.ref_sequence_type().ref_elem_type();
  ASSERT_TRUE(seq_elem_type.has_map_type());
  const TypeProto::Map &map_type = seq_elem_type.ref_map_type();
  ASSERT_TRUE(map_type.ref_value_type().has_tensor_type());
  const TypeProto::Tensor &value_tensor = map_type.ref_value_type().ref_tensor_type();
  EXPECT_EQ(value_tensor.ref_elem_type(), core::runtime::DataType::FLOAT);
  ASSERT_TRUE(value_tensor.has_shape());
  EXPECT_EQ(value_tensor.ref_shape().ref_dim().size(), 0u);
}

TEST(BackendTestCaseShapeInference, AllCollectedCasesInferOutputShapes) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);

    // For ``kind == "model"`` cases the test model also records expected
    // intermediate shapes in ``value_info``. Operate on a deep copy of the
    // model so we can wipe out those intermediate values (and the recorded
    // output shapes) without mutating the original test case, then verify
    // shape inference recovers them.
    ModelProto *model_ptr = &tc.model();
    ModelProto model_copy;
    std::vector<ExpectedOutput> expected_value_info;
    if (tc.kind == "model") {
      std::string serialized;
      tc.model().SerializeToString(serialized);
      model_copy.ParseFromString(serialized);
      model_ptr = &model_copy;
      expected_value_info = SnapshotAndStripValueInfo(*model_ptr);
    }
    const auto expected = SnapshotAndStripOutputs(*model_ptr);

    ASSERT_NO_THROW(shape_inference::InferShapes(*model_ptr)) << "case: " << tc.name;

    const auto &outputs = model_ptr->ref_graph().ref_output();
    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &out = outputs[i];
      ASSERT_TRUE(out.has_type()) << "output " << expected[i].name << " missing type";
      const TypeProto::Tensor *tt_ptr = TensorTypeOf(out.ref_type());
      ASSERT_NE(tt_ptr, nullptr) << "output " << expected[i].name << " not a tensor";
      const auto &tt = *tt_ptr;
      EXPECT_EQ(static_cast<int32_t>(tt.elem_type()), expected[i].elem_type)
          << "elem_type mismatch on output " << expected[i].name;
      const auto inferred_dims = DimsOf(tt);
      if (!inferred_dims.empty() || tt.has_shape()) {
        // Rank must match expected.
        ASSERT_EQ(inferred_dims.size(), expected[i].shape.size())
            << "rank mismatch on output " << expected[i].name;
        for (size_t d = 0; d < inferred_dims.size(); ++d) {
          if (inferred_dims[d] != -1) {
            EXPECT_EQ(inferred_dims[d], expected[i].shape[d])
                << "dim[" << d << "] mismatch on output " << expected[i].name;
          }
        }
      }
    }

    if (tc.kind == "model") {
      CheckValueInfoMatchesExpected(model_ptr->ref_graph(), expected_value_info);
    }

    // Additional pass: for every collected data set, override the graph input
    // shapes with the concrete shapes from the ``DataSet`` input tensors, run
    // shape inference, and verify the inferred output shapes match the
    // ground-truth ``DataSet`` output tensor shapes (i.e. the shapes a runtime
    // would actually observe). This complements the recorded-shape check
    // above, which only validates against the shapes pre-stored in the
    // model's output ``ValueInfo``. Outputs whose graph type is not a plain
    // tensor (sequence / optional / map) are skipped because a single
    // ``DataSet`` tensor does not describe the container shape.
    for (size_t ds_idx = 0; ds_idx < tc.data_sets().size(); ++ds_idx) {
      const DataSet &ds = tc.data_sets()[ds_idx];
      SCOPED_TRACE("data_set[" + std::to_string(ds_idx) + "]");

      std::unordered_map<std::string, const DataTensor *> ds_inputs_by_name;
      for (const DataTensor &t : ds.inputs) {
        if (!t.name.empty()) {
          ds_inputs_by_name.emplace(t.name, &t);
        }
      }
      std::unordered_map<std::string, const DataTensor *> ds_outputs_by_name;
      for (const DataTensor &t : ds.outputs) {
        if (!t.name.empty()) {
          ds_outputs_by_name.emplace(t.name, &t);
        }
      }

      ModelProto ds_model;
      std::string ds_serialized;
      tc.model().SerializeToString(ds_serialized);
      ds_model.ParseFromString(ds_serialized);

      auto &ds_inputs = ds_model.mutable_graph()->ref_input();
      for (size_t i = 0; i < ds_inputs.size(); ++i) {
        auto &vi = ds_inputs[i];
        if (!vi.has_type()) {
          continue;
        }
        TypeProto::Tensor *tt = MutableTensorTypeOf(*vi.mutable_type());
        if (tt == nullptr) {
          continue;
        }
        const std::string in_name(vi.ref_name().data(), vi.ref_name().size());
        auto it = ds_inputs_by_name.find(in_name);
        if (it == ds_inputs_by_name.end()) {
          continue;
        }
        const DataTensor &src = *it->second;
        tt->clear_shape();
        TensorShapeProto *shape = tt->add_shape();
        for (int64_t d : src.shape) {
          shape->add_dim()->set_dim_value(d);
        }
      }

      auto &ds_outputs_pre = ds_model.mutable_graph()->ref_output();
      for (size_t i = 0; i < ds_outputs_pre.size(); ++i) {
        auto &out = ds_outputs_pre[i];
        if (!out.has_type()) {
          continue;
        }
        if (TypeProto::Tensor *tt = MutableTensorTypeOf(*out.mutable_type()); tt != nullptr) {
          tt->clear_shape();
        }
      }
      ds_model.mutable_graph()->mutable_value_info()->clear();

      ASSERT_NO_THROW(shape_inference::InferShapes(ds_model)) << "case: " << tc.name;

      const auto &ds_outputs = ds_model.ref_graph().ref_output();
      for (size_t i = 0; i < ds_outputs.size(); ++i) {
        const auto &out = ds_outputs[i];
        const std::string out_name(out.ref_name().data(), out.ref_name().size());
        auto it = ds_outputs_by_name.find(out_name);
        if (it == ds_outputs_by_name.end()) {
          continue;
        }
        if (!out.has_type() || !out.ref_type().has_tensor_type()) {
          continue;
        }
        const TypeProto::Tensor &tt = out.ref_type().ref_tensor_type();
        if (!tt.has_shape()) {
          // Shape inference produced no shape for this output; mirror the
          // tolerance of the recorded-shape check above. Such cases are
          // pre-existing per-op coverage gaps, out of scope here.
          continue;
        }
        const DataTensor &expected_tensor = *it->second;
        const auto inferred_dims = DimsOf(tt);
        ASSERT_EQ(inferred_dims.size(), expected_tensor.shape.size())
            << "rank mismatch on output " << out_name;
        for (size_t d = 0; d < inferred_dims.size(); ++d) {
          if (inferred_dims[d] != -1) {
            EXPECT_EQ(inferred_dims[d], expected_tensor.shape[d])
                << "dim[" << d << "] mismatch on output " << out_name
                << " (inferred=" << inferred_dims[d] << ", actual=" << expected_tensor.shape[d]
                << ")";
          }
        }
      }
    }
  }
}

// Second pass: replace every input dim_value by a symbolic dim_param keyed on
// the original numeric value (so identical numeric dims across the graph share
// the same symbol in the symbolic run). Run shape inference again, and verify
// that whenever two output dimensions resolve to the same symbol -- or to the
// same concrete value -- in the symbolic run, they were also equal in the
// first run. In other words, shape inference must never declare two dims
// "equal" symbolically unless they really are equal.
TEST(BackendTestCaseShapeInference, AllCollectedCasesPropagateSymbolicDims) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    const auto expected = SnapshotAndStripOutputs(tc.model());

    // Replace every input dim_value with a dim_param keyed by its value so
    // that equal numeric dims across all inputs share the same symbol.
    auto &inputs = tc.model().mutable_graph()->ref_input();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto &vi = inputs[i];
      if (!vi.has_type() || !vi.ref_type().has_tensor_type()) {
        continue;
      }
      auto *tt = vi.mutable_type()->mutable_tensor_type();
      if (!tt->has_shape()) {
        continue;
      }
      auto &dims = tt->mutable_shape()->ref_dim();
      for (size_t d = 0; d < dims.size(); ++d) {
        auto &dim = dims[d];
        if (!dim.has_dim_value()) {
          continue;
        }
        const int64_t value = dim.ref_dim_value();
        const std::string symbol = "sym_v" + std::to_string(value);
        dim.clear_dim_value();
        dim.set_dim_param(symbol);
      }
    }

    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model())) << "case: " << tc.name;

    const auto &outputs = tc.model().ref_graph().ref_output();
    ASSERT_EQ(outputs.size(), expected.size());

    // Describes an output dim in the symbolic run: either a concrete value or
    // a symbol name. ``kind == 0`` means the dim is fully unknown (missing).
    struct SymbolicDim {
      int kind = 0; // 0 = unknown, 1 = dim_value, 2 = dim_param
      int64_t value = 0;
      std::string symbol;
    };

    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &out = outputs[i];
      ASSERT_TRUE(out.has_type()) << "output " << expected[i].name << " missing type";
      const TypeProto::Tensor *tt_ptr = TensorTypeOf(out.ref_type());
      ASSERT_NE(tt_ptr, nullptr) << "output " << expected[i].name << " not a tensor";
      const auto &tt = *tt_ptr;
      EXPECT_EQ(static_cast<int32_t>(tt.elem_type()), expected[i].elem_type)
          << "elem_type mismatch on output " << expected[i].name;
    }

    // Collect (expected_value, symbolic_descriptor) pairs for every output dim
    // across all outputs, then check that pairs with equal expected values
    // also share an "equal" symbolic descriptor.
    std::vector<std::pair<int64_t, SymbolicDim>> dim_records;
    std::vector<std::pair<size_t, size_t>> dim_locations; // (output_index, axis)
    for (size_t i = 0; i < outputs.size(); ++i) {
      const TypeProto::Tensor *tt_ptr = TensorTypeOf(outputs[i].ref_type());
      if (tt_ptr == nullptr || !tt_ptr->has_shape()) {
        continue;
      }
      const auto &dims = tt_ptr->ref_shape().ref_dim();
      if (dims.size() != expected[i].shape.size()) {
        // Rank mismatch is already flagged by the first test; skip here.
        continue;
      }
      for (size_t d = 0; d < dims.size(); ++d) {
        SymbolicDim sd;
        if (dims[d].has_dim_value()) {
          sd.kind = 1;
          sd.value = dims[d].ref_dim_value();
        } else if (dims[d].has_dim_param()) {
          sd.kind = 2;
          sd.symbol.assign(dims[d].ref_dim_param().data(), dims[d].ref_dim_param().size());
        }
        dim_records.emplace_back(expected[i].shape[d], std::move(sd));
        dim_locations.emplace_back(i, d);
      }
    }

    for (size_t a = 0; a < dim_records.size(); ++a) {
      for (size_t b = a + 1; b < dim_records.size(); ++b) {
        const SymbolicDim &sa = dim_records[a].second;
        const SymbolicDim &sb = dim_records[b].second;
        if (sa.kind == 0 || sb.kind == 0) {
          // Tolerate fully unknown dims (data-dependent ops, etc.).
          continue;
        }
        if (sa.kind != sb.kind) {
          continue;
        }
        const int64_t va = dim_records[a].first;
        const int64_t vb = dim_records[b].first;
        if (va == -1 || vb == -1) {
          continue;
        }
        if (sa.kind == 1) {
          // Two outputs resolved to the same concrete dim_value: they must
          // also have been equal in the first run (otherwise inference is
          // inconsistent).
          if (sa.value == sb.value) {
            EXPECT_EQ(va, vb) << "output dims share dim_value " << sa.value
                              << " in the symbolic run but had different values in the first run: "
                              << "output[" << dim_locations[a].first << "].dim["
                              << dim_locations[a].second << "]=" << va << " vs output["
                              << dim_locations[b].first << "].dim[" << dim_locations[b].second
                              << "]=" << vb;
          }
        } else {
          // Same dim_param symbol on two output dims means shape inference
          // has decided those dims are equal; the first run must agree.
          if (sa.symbol == sb.symbol) {
            EXPECT_EQ(va, vb) << "output dims share dim_param '" << sa.symbol
                              << "' in the symbolic run but had different values in the first run: "
                              << "output[" << dim_locations[a].first << "].dim["
                              << dim_locations[a].second << "]=" << va << " vs output["
                              << dim_locations[b].first << "].dim[" << dim_locations[b].second
                              << "]=" << vb;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + nested model-local function call
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_shapes`` shape-inference pipeline correctly handles
// a node whose ``op_type`` references a model-local :cpp:class:`FunctionProto`
// **whose body itself calls another model-local function**. The model is
// built by :cpp:func:`RegisterNestedLocalFunctionAddShapeInferenceCases`:
// its single graph node calls ``local:func_outer_add(X, Y) -> Z`` whose body
// is a single call into ``local:func_inner_add(a, b) -> c { c = Add(a, b) }``.
// After two levels of expansion, the inferred ``Z`` must carry the symbolic
// ``(batch, d_model)`` shape of the inputs.
TEST(BackendTestCaseShapeInference, OnnxOptimSupportsNestedLocalFunctionCall) {
  const std::vector<TestCase> cases = CollectTestCases("shape");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_nested_local_function_add") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    ASSERT_TRUE(tc.model().SerializeToString(serialized))
        << "failed to serialize case: " << tc.name;
    ASSERT_TRUE(model_copy.ParseFromString(serialized)) << "failed to parse case: " << tc.name;

    // Strip the recorded output shape so optim shape inference has to
    // recover it through the two levels of function-body expansion.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    auto *tt = MutableTensorTypeOf(*outputs[0].mutable_type());
    ASSERT_NE(tt, nullptr);
    tt->clear_shape();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *out_tt = TensorTypeOf(out.ref_type());
    ASSERT_NE(out_tt, nullptr);
    EXPECT_EQ(static_cast<int32_t>(out_tt->elem_type()), 1 /* FLOAT */);
    ASSERT_TRUE(out_tt->has_shape());
    const auto &dims = out_tt->ref_shape().ref_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_TRUE(dims[0].has_dim_param());
    EXPECT_EQ(std::string(dims[0].ref_dim_param()), "batch");
    EXPECT_TRUE(dims[1].has_dim_param());
    EXPECT_EQ(std::string(dims[1].ref_dim_param()), "d_model");
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_nested_local_function_add case not registered";
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + local function with Range node
// ---------------------------------------------------------------------------
//
// Verifies that ``onnx_shapes`` shape inference propagates the initializer's
// ``ValueAsShape`` annotation through a local-function call boundary when the
// function body contains a ``Range`` node that uses the function's own input
// as its ``limit``.
//
// The model topology is:
//   Initializer: limit_val : int64[] = 5
//       ↓
//   local:func_range(limit_val) → r_out
//       ↓
//   Abs(r_out) → out
//
// The function body is: start_c=Constant(0), delta_c=Constant(1),
// r = Range(start_c, lim, delta_c).
//
// After inference the output ``out`` must have elem_type=INT64 and shape=[5].
TEST(BackendTestCaseShapeInference, OnnxOptimPropagatesValueAsShapeInLocalFunctionRange) {
  const std::vector<TestCase> cases = CollectTestCases("shape");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_local_function_range") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    ASSERT_TRUE(tc.model().SerializeToString(serialized))
        << "failed to serialize case: " << tc.name;
    ASSERT_TRUE(model_copy.ParseFromString(serialized)) << "failed to parse case: " << tc.name;

    // Strip the recorded output shape so optim shape inference has to
    // recover it by expanding the function body and propagating the
    // initializer's ValueAsShape annotation through the call boundary.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    auto *tt = MutableTensorTypeOf(*outputs[0].mutable_type());
    ASSERT_NE(tt, nullptr);
    tt->clear_shape();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *out_tt = TensorTypeOf(out.ref_type());
    ASSERT_NE(out_tt, nullptr);
    EXPECT_EQ(static_cast<int32_t>(out_tt->elem_type()), 7 /* INT64 */);
    ASSERT_TRUE(out_tt->has_shape());
    const auto &dims = out_tt->ref_shape().ref_dim();
    ASSERT_EQ(dims.size(), 1u);
    ASSERT_TRUE(dims[0].has_dim_value());
    EXPECT_EQ(dims[0].ref_dim_value(), 5);
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_local_function_range case not registered";
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + Loop subgraph
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_shapes`` shape-inference pipeline correctly handles
// ``test_cc_loop_basic_trip_count``: no loop-carried states (N=0) and one
// scan output (K=1). The body is a two-node sub-graph that produces a
// constant INT64 ``[42]`` tensor of shape ``[1]`` each iteration.
//
// After shape inference the scan output ``scan_outputs`` must be INT64 with
// rank 2: a symbolic leading axis (the trip count is a runtime value and
// is not known statically) and a concrete trailing dim of 1 (the
// per-iteration element shape from the body's ``Constant`` node).
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeLoopSubgraph) {
  const std::vector<TestCase> cases = CollectTestCases("Loop");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_loop_basic_trip_count") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape so optim shape inference must
    // recover it by walking the Loop body subgraph.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const auto &out_infos = model_copy.ref_graph().ref_output();
    ASSERT_EQ(out_infos.size(), 1u);

    // scan_outputs: stacked scan output, INT64 [symbolic, 1].
    // The leading axis is symbolic (trip count is a runtime INT64 input);
    // the trailing dim 1 comes from the body Constant node shape [1].
    const ValueInfoProto &out = out_infos[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
    ASSERT_NE(ott, nullptr);
    EXPECT_EQ(static_cast<int32_t>(ott->elem_type()),
              static_cast<int32_t>(core::runtime::DataType::INT64));
    ASSERT_TRUE(ott->has_shape());
    const auto dims = DimsOf(*ott);
    // Rank must be 2: symbolic trip-count axis + concrete [1] element shape.
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[1], 1);
  }
  ASSERT_TRUE(found) << "test_cc_loop_basic_trip_count case not registered";
}

TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeScanSubgraph) {
  const std::vector<TestCase> cases = CollectTestCases("Scan");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_scan_basic_trip_count") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape so optim shape inference must
    // recover it by walking the Scan body subgraph.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const auto &out_infos = model_copy.ref_graph().ref_output();
    ASSERT_EQ(out_infos.size(), 1u);

    // Y: stacked scan output, FLOAT [3, 2].
    // The trip count 3 is inferred from the scan input X's shape [3, 2]
    // (scan axis 0), so the leading axis is concrete.
    const ValueInfoProto &out = out_infos[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
    ASSERT_NE(ott, nullptr);
    EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
    ASSERT_TRUE(ott->has_shape());
    const auto dims = DimsOf(*ott);
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[0], 3);
    EXPECT_EQ(dims[1], 2);
  }
  ASSERT_TRUE(found) << "test_cc_scan_basic_trip_count case not registered";
}

TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapePairwiseDistanceScan) {
  const std::vector<TestCase> cases = CollectTestCases("Scan");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_scan_pairwise_distance") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip recorded output shapes so optim shape inference must recover
    // them by walking the Scan body subgraph.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 2u);
    for (auto &vi : outputs) {
      if (auto *ott = MutableTensorTypeOf(*vi.mutable_type()); ott != nullptr) {
        ott->clear_shape();
      }
    }

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const auto &out_infos = model_copy.ref_graph().ref_output();
    ASSERT_EQ(out_infos.size(), 2u);

    // state_X_final: preserved state, FLOAT [3, 2].
    {
      const ValueInfoProto &out = out_infos[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      const auto dims = DimsOf(*ott);
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_EQ(dims[0], 3);
      EXPECT_EQ(dims[1], 2);
    }

    // dists: stacked scan output, FLOAT [3, 3].
    {
      const ValueInfoProto &out = out_infos[1];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      const auto dims = DimsOf(*ott);
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_EQ(dims[0], 3);
      EXPECT_EQ(dims[1], 3);
    }
  }
  ASSERT_TRUE(found) << "test_cc_scan_pairwise_distance case not registered";
}

TEST(BackendTestCaseShapeInference, OnnxOptimInfersInPlaceReuseOnBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("inplace");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_inplace_reuse") {
      continue;
    }
    found = true;

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(tc.model())) << "case: " << tc.name;

    const std::vector<std::unordered_map<std::string, std::string>> expected_metadata = {
        {{"onnx_light.not_used_after", "X"}},
        {{"onnx_light.inplace_reuse", "0:0:equal"}, {"onnx_light.release_after", "A"}},
        {{"onnx_light.inplace_reuse", "0:0:equal"}, {"onnx_light.release_after", "B"}}};
    const auto &nodes = tc.model().ref_graph().ref_node();
    ASSERT_EQ(nodes.size(), expected_metadata.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(nodes[i]), expected_metadata[i])
          << "metadata mismatch on node " << i << " in case " << tc.name;
    }

    std::vector<std::vector<core::compute::InPlaceReuse>> reuse_with_inputs =
        core::compute::ComputeInPlaceReuse(tc.model().ref_graph(), ctx,
                                           /*allow_input_overwrite=*/true);
    ASSERT_EQ(reuse_with_inputs.size(), 3u);
    ASSERT_EQ(reuse_with_inputs[0].size(), 1u);
    EXPECT_EQ(reuse_with_inputs[0][0],
              (core::compute::InPlaceReuse{0, 0, core::compute::InPlaceReuseKind::kEqual}));
    ASSERT_EQ(reuse_with_inputs[1].size(), 1u);
    EXPECT_EQ(reuse_with_inputs[1][0],
              (core::compute::InPlaceReuse{0, 0, core::compute::InPlaceReuseKind::kEqual}));
    ASSERT_EQ(reuse_with_inputs[2].size(), 1u);
    EXPECT_EQ(reuse_with_inputs[2][0],
              (core::compute::InPlaceReuse{0, 0, core::compute::InPlaceReuseKind::kEqual}));
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_inplace_reuse case not registered";
}

TEST(BackendTestCaseShapeInference, OnnxOptimWritesPeakMemoryMetadataOnBackendCases) {
  const std::vector<TestCase> cases = CollectTestCases("peak_memory");
  ASSERT_FALSE(cases.empty());

  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);

    std::vector<MetadataMap> expected_node_meta;
    for (const auto &node : tc.model().ref_graph().ref_node()) {
      expected_node_meta.push_back(MetadataOf(node));
    }

    ModelProto model_copy;
    std::string serialized;
    ASSERT_TRUE(tc.model().SerializeToString(serialized))
        << "failed to serialize case: " << tc.name;
    ASSERT_TRUE(model_copy.ParseFromString(serialized)) << "failed to parse case: " << tc.name;

    GraphProto *graph = model_copy.mutable_graph();
    ASSERT_NE(graph, nullptr);
    for (auto &node : *graph->mutable_node()) {
      node.mutable_metadata_props()->clear();
    }

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(model_copy)) << "case: " << tc.name;
    ASSERT_NO_THROW(
        core::compute::WritePeakMemoryToMetadata(*graph, ctx, core::symbolic::Device::kCPU))
        << "case: " << tc.name;

    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_nodes[i]), expected_node_meta[i])
          << "metadata mismatch on node " << i << " in case " << tc.name;
    }
  }
}

TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeLoopPairwiseDistance) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_loop_pairwise_distance") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape so optim shape inference must
    // recover it by walking the Loop body subgraph.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const auto &out_infos = model_copy.ref_graph().ref_output();
    ASSERT_EQ(out_infos.size(), 1u);

    // Y: stacked scan output, FLOAT rank 2. The leading axis is the Loop
    // trip count (runtime INT64 input, so the dim is symbolic); the
    // trailing dim is the per-iteration ``[N]`` element shape from the
    // body's ``ReduceSum`` over the squared-difference rows.
    const ValueInfoProto &out = out_infos[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
    ASSERT_NE(ott, nullptr);
    EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
    ASSERT_TRUE(ott->has_shape());
    const auto dims = DimsOf(*ott);
    ASSERT_EQ(dims.size(), 2u);
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_loop_pairwise_distance case not registered";
}

// ---------------------------------------------------------------------------
// onnx shape inference + pairwise-distance / TopK(k=input) / ReduceMean
// ---------------------------------------------------------------------------
//
// Verifies that shape inference propagates **symbolic** dims through the
// ``test_cc_shape_inference_topk_pairwise_distance`` model (broadcasting
// ``Sub`` → ``ReduceSum`` → ``Sqrt`` → ``TopK`` → ``ReduceMean``) and, in
// particular, that ``TopK`` emits a **symbolic** output axis because its
// ``K`` operand is a runtime model input (not a constant). After
// ``ReduceMean`` collapses that axis the model output ``Y`` recovers the
// symbolic ``[N]`` row dimension.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeTopKPairwiseDistance) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_topk_pairwise_distance") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output / intermediate shapes so shape inference must
    // recover them from the symbolic ``[N, D]`` input alone.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    auto &value_infos = model_copy.mutable_graph()->ref_value_info();
    for (size_t i = 0; i < value_infos.size(); ++i) {
      if (auto *tt = MutableTensorTypeOf(*value_infos[i].mutable_type()); tt != nullptr) {
        tt->clear_shape();
      }
    }

    ASSERT_NO_THROW(shape_inference::InferShapes(model_copy)) << "case: " << tc.name;

    // Index every value_info by name to inspect the intermediate TopK output.
    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &inferred_value_infos = model_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < inferred_value_infos.size(); ++i) {
      const auto &vi = inferred_value_infos[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``dist`` — the pairwise distance matrix — must be FLOAT rank 2 with both
    // axes resolved to the symbolic input row dim ``N``.
    {
      auto it = by_name.find("dist");
      ASSERT_NE(it, by_name.end()) << "dist value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
    }

    // ``topk_values`` — must be FLOAT rank 2 whose trailing axis is symbolic
    // (no concrete ``dim_value``) because ``K`` is a runtime input.
    {
      auto it = by_name.find("topk_values");
      ASSERT_NE(it, by_name.end()) << "topk_values value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      const auto &dims = tt->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_FALSE(dims[1].has_dim_value())
          << "TopK output axis must stay symbolic when K is a model input";
    }

    // ``Y`` — the model output — must be FLOAT rank 1: ``ReduceMean`` reduces
    // the symbolic TopK axis away and the surviving row dim is symbolic ``N``.
    {
      const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      const auto &dims = ott->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 1u);
      EXPECT_FALSE(dims[0].has_dim_value())
          << "Y row dim must stay symbolic for a symbolic [N, D] input";
    }

    // Second engine: the symbolic ``onnx_shapes`` shape-inference pipeline must
    // reach the same conclusion — a symbolic TopK axis that ``ReduceMean``
    // collapses to a rank-1 ``Y``.
    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    std::unordered_map<std::string, const ValueInfoProto *> optim_by_name;
    const auto &optim_value_infos = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < optim_value_infos.size(); ++i) {
      const auto &vi = optim_value_infos[i];
      optim_by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    {
      auto it = optim_by_name.find("topk_values");
      ASSERT_NE(it, optim_by_name.end()) << "optim: topk_values value_info missing";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      ASSERT_TRUE(tt->has_shape());
      const auto &dims = tt->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_FALSE(dims[1].has_dim_value())
          << "optim: TopK output axis must stay symbolic when K is a model input";
    }

    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_topk_pairwise_distance case not registered";
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + Loop pairwise-distance / TopK(k=input) /
// ReduceMean
// ---------------------------------------------------------------------------
//
// Verifies that the symbolic ``onnx_shapes`` shape-inference pipeline keeps a
// **symbolic** TopK output axis when the pairwise distance matrix is produced
// by a ``Loop`` body (``test_cc_shape_inference_loop_topk_pairwise_distance``).
// The Loop trip count is a runtime ``Shape(X)[0]`` value, so the stacked
// distance matrix has a symbolic leading axis; ``TopK`` then emits a fresh
// symbolic dim for its trailing axis because ``K`` is a model input, and
// ``ReduceMean`` collapses it to recover a rank-1 ``Y``.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeLoopTopKPairwiseDistance) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_loop_topk_pairwise_distance") {
      continue;
    }
    found = true;

    std::string serialized;
    tc.model().SerializeToString(serialized);

    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    ASSERT_EQ(optim_outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &value_infos = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < value_infos.size(); ++i) {
      const auto &vi = value_infos[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``topk_values`` — FLOAT rank 2 whose trailing axis is symbolic (no
    // concrete ``dim_value``) because ``K`` is a runtime input.
    {
      auto it = by_name.find("topk_values");
      ASSERT_NE(it, by_name.end()) << "topk_values value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      const auto &dims = tt->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_FALSE(dims[1].has_dim_value())
          << "TopK output axis must stay symbolic when K is a model input";
    }

    // ``Y`` — FLOAT rank 1: ``ReduceMean`` reduces the symbolic TopK axis away.
    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_loop_topk_pairwise_distance case not registered";
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + Scan pairwise-distance / TopK(k=input) /
// ReduceMean
// ---------------------------------------------------------------------------
//
// Verifies that the symbolic ``onnx_shapes`` shape-inference pipeline keeps a
// **symbolic** TopK output axis when the pairwise distance matrix is produced
// by a ``Scan`` body (``test_cc_shape_inference_scan_topk_pairwise_distance``).
// The Scan trip count comes from ``X``'s scan axis (symbolic ``N``), so the
// stacked distance matrix is ``[N, N]``; ``TopK`` then emits a fresh symbolic
// dim for its trailing axis because ``K`` is a model input, and ``ReduceMean``
// collapses it to recover a rank-1 ``Y``.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeScanTopKPairwiseDistance) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_scan_topk_pairwise_distance") {
      continue;
    }
    found = true;

    std::string serialized;
    tc.model().SerializeToString(serialized);

    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    ASSERT_EQ(optim_outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &value_infos = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < value_infos.size(); ++i) {
      const auto &vi = value_infos[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``topk_values`` — FLOAT rank 2 whose trailing axis is symbolic (no
    // concrete ``dim_value``) because ``K`` is a runtime input.
    {
      auto it = by_name.find("topk_values");
      ASSERT_NE(it, by_name.end()) << "topk_values value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      const auto &dims = tt->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 2u);
      EXPECT_FALSE(dims[1].has_dim_value())
          << "TopK output axis must stay symbolic when K is a model input";
    }

    // ``Y`` — FLOAT rank 1: ``ReduceMean`` reduces the symbolic TopK axis away.
    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_scan_topk_pairwise_distance case not registered";
}

// ---------------------------------------------------------------------------
// onnx + onnx_shapes shape inference: two sequential TopK nodes sharing
// the same runtime K input, followed by ReduceMean
// ---------------------------------------------------------------------------
//
// Verifies that both the standard ONNX and the ``onnx_shapes`` shape-inference
// pipelines keep **symbolic** TopK output axes when two consecutive TopK nodes
// share the same runtime ``K`` model input
// (``test_cc_shape_inference_two_topk_same_k``). Because K is unknown at
// inference time, each TopK node emits a fresh symbolic dim
// (``TopK_k`` / ``TopK_k_2``); ``ReduceMean`` then collapses
// the second symbolic axis, recovering the rank-1 output ``Y [N]``.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeTwoTopKSameK) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_two_topk_same_k") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip output and intermediate shapes so shape inference must recover them.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    auto &value_infos = model_copy.mutable_graph()->ref_value_info();
    for (size_t i = 0; i < value_infos.size(); ++i) {
      if (auto *tt = MutableTensorTypeOf(*value_infos[i].mutable_type()); tt != nullptr) {
        tt->clear_shape();
      }
    }

    ASSERT_NO_THROW(shape_inference::InferShapes(model_copy)) << "case: " << tc.name;

    // Index every value_info by name to inspect intermediate TopK outputs.
    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &inferred_value_infos = model_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < inferred_value_infos.size(); ++i) {
      const auto &vi = inferred_value_infos[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``values1`` — must be FLOAT rank 2 with symbolic trailing axis.
    {
      auto it = by_name.find("values1");
      ASSERT_NE(it, by_name.end()) << "values1 value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "TopK1 output axis must stay symbolic when K is a model input";
    }

    // ``values2`` — must be FLOAT rank 2 with symbolic trailing axis.
    {
      auto it = by_name.find("values2");
      ASSERT_NE(it, by_name.end()) << "values2 value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "TopK2 output axis must stay symbolic when K is a model input";
    }

    // ``Y`` — must be FLOAT rank 1 (ReduceMean collapsed the symbolic axis).
    {
      const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }

    // onnx_shapes pass must reach the same rank conclusions.
    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    std::unordered_map<std::string, const ValueInfoProto *> optim_by_name;
    const auto &optim_value_infos = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < optim_value_infos.size(); ++i) {
      const auto &vi = optim_value_infos[i];
      optim_by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    {
      auto it = optim_by_name.find("values1");
      ASSERT_NE(it, optim_by_name.end()) << "optim: values1 value_info missing";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "optim: TopK1 output axis must stay symbolic when K is a model input";
    }

    {
      auto it = optim_by_name.find("values2");
      ASSERT_NE(it, optim_by_name.end()) << "optim: values2 value_info missing";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "optim: TopK2 output axis must stay symbolic when K is a model input";
    }

    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_two_topk_same_k case not registered";
}

// ---------------------------------------------------------------------------
// onnx + onnx_shapes shape inference: two sequential TopK nodes with
// different runtime K inputs, followed by ReduceMean
// ---------------------------------------------------------------------------
//
// Verifies that both the standard ONNX and the ``onnx_shapes`` shape-inference
// pipelines keep **symbolic** TopK output axes when two consecutive TopK nodes
// use different runtime K inputs K1 and K2 (K1 > K2)
// (``test_cc_shape_inference_two_topk_different_k``). Each TopK node emits a
// distinct symbolic dim (``TopK_k`` / ``TopK_k_2``);
// ``ReduceMean`` then collapses the second symbolic axis, recovering the
// rank-1 output ``Y [N]``.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeTwoTopKDifferentK) {
  const std::vector<TestCase> cases = CollectTestCases();
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_two_topk_different_k") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip output and intermediate shapes so shape inference must recover them.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    if (auto *ott = MutableTensorTypeOf(*outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    auto &value_infos = model_copy.mutable_graph()->ref_value_info();
    for (size_t i = 0; i < value_infos.size(); ++i) {
      if (auto *tt = MutableTensorTypeOf(*value_infos[i].mutable_type()); tt != nullptr) {
        tt->clear_shape();
      }
    }

    ASSERT_NO_THROW(shape_inference::InferShapes(model_copy)) << "case: " << tc.name;

    // Index every value_info by name.
    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &inferred_value_infos = model_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < inferred_value_infos.size(); ++i) {
      const auto &vi = inferred_value_infos[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``values1`` — FLOAT rank 2 with symbolic trailing axis (K1 unknown).
    {
      auto it = by_name.find("values1");
      ASSERT_NE(it, by_name.end()) << "values1 value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "TopK1 output axis must stay symbolic when K1 is a model input";
    }

    // ``values2`` — FLOAT rank 2 with symbolic trailing axis (K2 unknown).
    {
      auto it = by_name.find("values2");
      ASSERT_NE(it, by_name.end()) << "values2 value_info missing after inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "TopK2 output axis must stay symbolic when K2 is a model input";
    }

    // ``Y`` — must be FLOAT rank 1 (ReduceMean collapsed the symbolic axis).
    {
      const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }

    // onnx_shapes pass must reach the same rank conclusions.
    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    std::unordered_map<std::string, const ValueInfoProto *> optim_by_name;
    const auto &optim_value_infos = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < optim_value_infos.size(); ++i) {
      const auto &vi = optim_value_infos[i];
      optim_by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    {
      auto it = optim_by_name.find("values1");
      ASSERT_NE(it, optim_by_name.end()) << "optim: values1 value_info missing";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "optim: TopK1 output axis must stay symbolic when K1 is a model input";
    }

    {
      auto it = optim_by_name.find("values2");
      ASSERT_NE(it, optim_by_name.end()) << "optim: values2 value_info missing";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 2u);
      EXPECT_FALSE(tt->ref_shape().ref_dim()[1].has_dim_value())
          << "optim: TopK2 output axis must stay symbolic when K2 is a model input";
    }

    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      ASSERT_EQ(ott->ref_shape().ref_dim().size(), 1u);
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_two_topk_different_k case not registered";
}

// ---------------------------------------------------------------------------
// onnx_shapes shape inference + Gather value-as-shape propagation
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_shapes`` shape-inference pipeline correctly propagates
// the *value-as-shape* annotation through a ``Gather`` node. The model is built
// by :cpp:func:`RegisterGatherValueAsShapeShapeInferenceCases`:
//   Shape(x[N,D]) -> shape_x [2]
//   Gather(shape_x, [0], axis=0) -> n_vec [1]   # VAS [N] propagated here
//   Expand(y[1], n_vec) -> expanded [N]
//   Abs(expanded) -> z [N]
//
// ``n_vec`` receives the VAS ``[N]`` sliced from ``shape_x``; ``Expand`` then
// turns that VAS into the concrete output shape so ``z`` carries the symbolic
// dim ``N`` that matches the first axis of ``x``.
TEST(BackendTestCaseShapeInference, OnnxOptimPropagatesGatherValueAsShape) {
  const std::vector<TestCase> cases = CollectTestCases("shape");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_gather_value_as_shape") {
      continue;
    }
    found = true;

    std::string serialized;
    tc.model().SerializeToString(serialized);

    // --- onnx_shapes pass ---
    ModelProto optim_copy;
    optim_copy.ParseFromString(serialized);
    auto &optim_outputs = optim_copy.mutable_graph()->ref_output();
    if (auto *ott = MutableTensorTypeOf(*optim_outputs[0].mutable_type()); ott != nullptr) {
      ott->clear_shape();
    }
    optim_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(optim_copy)) << "case: " << tc.name;

    // Build a name-indexed map for value_info inspection.
    std::unordered_map<std::string, const ValueInfoProto *> by_name;
    const auto &vis = optim_copy.ref_graph().ref_value_info();
    for (size_t i = 0; i < vis.size(); ++i) {
      const auto &vi = vis[i];
      by_name.emplace(std::string(vi.ref_name().data(), vi.ref_name().size()), &vi);
    }

    // ``n_vec`` = Gather(shape_x, [0]) must be INT64 rank 1.
    {
      auto it = by_name.find("n_vec");
      ASSERT_NE(it, by_name.end()) << "optim: n_vec value_info missing after shape inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr) << "optim: n_vec type missing";
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 7 /* INT64 */);
      ASSERT_TRUE(tt->has_shape());
      ASSERT_EQ(tt->ref_shape().ref_dim().size(), 1u) << "optim: n_vec must be rank 1";
    }

    // ``expanded`` = Expand(y, n_vec) must be FLOAT rank 1 with a symbolic axis.
    {
      auto it = by_name.find("expanded");
      ASSERT_NE(it, by_name.end()) << "optim: expanded value_info missing after shape inference";
      const TypeProto::Tensor *tt = TensorTypeOf(it->second->ref_type());
      ASSERT_NE(tt, nullptr) << "optim: expanded type missing";
      EXPECT_EQ(static_cast<int32_t>(tt->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(tt->has_shape());
      const auto &dims = tt->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 1u) << "optim: expanded must be rank 1";
      // The axis must be symbolic (VAS [N] propagated through Gather).
      EXPECT_FALSE(dims[0].has_dim_value())
          << "optim: expanded[0] must be symbolic (N from x), not concrete";
    }

    // ``z`` = Abs(expanded) — model output — must be FLOAT rank 1 with a
    // symbolic axis matching the first axis of ``x``.
    {
      const ValueInfoProto &out = optim_copy.ref_graph().ref_output()[0];
      ASSERT_TRUE(out.has_type());
      const TypeProto::Tensor *ott = TensorTypeOf(out.ref_type());
      ASSERT_NE(ott, nullptr);
      EXPECT_EQ(static_cast<int32_t>(ott->elem_type()), 1 /* FLOAT */);
      ASSERT_TRUE(ott->has_shape());
      const auto &dims = ott->ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), 1u) << "optim: z must be rank 1";
      EXPECT_FALSE(dims[0].has_dim_value()) << "optim: z[0] must be symbolic (N), not concrete";
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_gather_value_as_shape case not registered";
}

namespace {

// Builds a single-node ``ConvTranspose`` model with the given ``group``
// attribute and the input/weight shapes from the example in onnx/onnx#7821.
ModelProto MakeConvTransposeGroupModel(int64_t group) {
  ModelProto model;
  model.set_ir_version(10);

  OperatorSetIdProto *default_opset = model.add_opset_import();
  default_opset->set_domain("");
  default_opset->set_version(22);

  GraphProto *graph = model.add_graph();
  graph->set_name("conv_transpose_group_graph");

  ValueInfoProto *x = graph->add_input();
  x->set_name("x");
  TypeProto::Tensor *x_tt = x->add_type()->add_tensor_type();
  x_tt->set_elem_type(core::runtime::DataType::FLOAT);
  TensorShapeProto *x_shape = x_tt->add_shape();
  for (int64_t dim : {1, 32, 14, 14}) {
    x_shape->add_dim()->set_dim_value(dim);
  }

  ValueInfoProto *w = graph->add_input();
  w->set_name("w");
  TypeProto::Tensor *w_tt = w->add_type()->add_tensor_type();
  w_tt->set_elem_type(core::runtime::DataType::FLOAT);
  TensorShapeProto *w_shape = w_tt->add_shape();
  for (int64_t dim : {32, 64, 3, 3}) {
    w_shape->add_dim()->set_dim_value(dim);
  }

  ValueInfoProto *output = graph->add_output();
  output->set_name("z");
  // Leave the output type empty so shape inference must populate it.
  output->add_type();

  NodeProto *node = graph->add_node();
  node->set_op_type("ConvTranspose");
  node->add_input("x");
  node->add_input("w");
  node->add_output("z");
  AttributeProto *group_attr = node->add_attribute();
  group_attr->set_name("group");
  group_attr->set_type(AttributeProto::AttributeType::INT);
  group_attr->set_i(group);

  return model;
}

} // namespace

// ConvTranspose rejects input channels C that are not divisible by group
// (propagated from onnx/onnx#7821). Mirrors the example: x=[1, 32, 14, 14],
// w=[32, 64, 3, 3], group=3 -> 32 % 3 != 0.
TEST(BackendTestCaseShapeInference, ConvTransposeRejectsIndivisibleGroup) {
  ModelProto model = MakeConvTransposeGroupModel(/*group=*/3);
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(false, 1, false)),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// ConvTranspose rejects a non-positive group attribute (propagated from
// onnx/onnx#7821).
TEST(BackendTestCaseShapeInference, ConvTransposeRejectsNonPositiveGroup) {
  ModelProto model = MakeConvTransposeGroupModel(/*group=*/0);
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(false, 1, false)),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// Verifies that WriteValueAndNodeTagsToMetadata applied to a clean copy of the
// shape-tag backend test case produces metadata that matches the expected
// values pre-embedded in the model.
TEST(BackendTestCaseShapeInference, OnnxOptimWritesShapeTagMetadataOnBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("shape_tag");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_tag_shape_reshape") {
      continue;
    }
    found = true;

    // Collect the expected graph-level metadata pre-embedded in the case model.
    const MetadataMap expected_graph_meta = MetadataOf(tc.model().ref_graph());
    ASSERT_EQ(expected_graph_meta.count(core::compute::kValueTagsMetadataKey), 0u)
        << "feature metadata must not be stored on the graph in case";

    // Collect per-node expected metadata from the pre-embedded model.
    const auto &src_nodes = tc.model().ref_graph().ref_node();
    std::vector<MetadataMap> expected_node_meta;
    expected_node_meta.reserve(src_nodes.size());
    for (const auto &node : src_nodes) {
      expected_node_meta.push_back(MetadataOf(node));
    }

    // Collect expected input and output metadata from the pre-embedded model.
    const auto &src_inputs = tc.model().ref_graph().ref_input();
    std::vector<MetadataMap> expected_input_meta;
    expected_input_meta.reserve(src_inputs.size());
    for (const auto &vi : src_inputs) {
      expected_input_meta.push_back(MetadataOf(vi));
    }
    const auto &src_outputs = tc.model().ref_graph().ref_output();
    std::vector<MetadataMap> expected_output_meta;
    expected_output_meta.reserve(src_outputs.size());
    for (const auto &vi : src_outputs) {
      expected_output_meta.push_back(MetadataOf(vi));
    }

    // Make a clean copy of the model and strip all metadata so
    // WriteValueAndNodeTagsToMetadata starts from a blank slate.
    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    for (size_t n = 0; n < graph->node().size(); ++n) {
      graph->mutable_node(n)->mutable_metadata_props()->clear();
    }
    for (size_t vi = 0; vi < graph->value_info().size(); ++vi) {
      graph->mutable_value_info(vi)->mutable_metadata_props()->clear();
    }
    for (size_t i = 0; i < graph->input().size(); ++i) {
      graph->mutable_input(i)->mutable_metadata_props()->clear();
    }
    for (size_t o = 0; o < graph->output().size(); ++o) {
      graph->mutable_output(o)->mutable_metadata_props()->clear();
    }

    // Run WriteValueAndNodeTagsToMetadata and verify the result matches the
    // pre-embedded expected values.
    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;

    EXPECT_EQ(MetadataOf(*graph), expected_graph_meta)
        << "graph metadata mismatch in case " << tc.name;

    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes onnx_light.value_tag
    // on the value_info entry for "S".
    const auto &result_vis = graph->ref_value_info();
    const auto &src_vis = tc.model().ref_graph().ref_value_info();
    ASSERT_EQ(result_vis.size(), src_vis.size());
    for (size_t vi = 0; vi < result_vis.size(); ++vi) {
      EXPECT_EQ(MetadataOf(result_vis[vi]), MetadataOf(src_vis[vi]))
          << "value_info[" << vi << "] metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes onnx_light.value_tag
    // on input and output entries.
    const auto &result_inputs = graph->ref_input();
    ASSERT_EQ(result_inputs.size(), expected_input_meta.size());
    for (size_t i = 0; i < result_inputs.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_inputs[i]), expected_input_meta[i])
          << "input[" << i << "] metadata mismatch in case " << tc.name;
    }
    const auto &result_outputs = graph->ref_output();
    ASSERT_EQ(result_outputs.size(), expected_output_meta.size());
    for (size_t o = 0; o < result_outputs.size(); ++o) {
      EXPECT_EQ(MetadataOf(result_outputs[o]), expected_output_meta[o])
          << "output[" << o << "] metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_tag_shape_reshape case not registered";
}

// ---------------------------------------------------------------------------
// kReleaseShapeTag event emitted by ComputeContext for the shape-tag release
// backend test case
// ---------------------------------------------------------------------------
//
// Verifies that ``ComputeInPlaceReuseGraph`` emits a ``kRelease`` event for
// the tensor ``S`` at the ``Reshape`` node (node index 1), which is S's only
// consumer.
//
// This test also confirms that the pre-embedded ``onnx_light.release_after``
// node metadata in the backend test case (``test_cc_release_shape_reshape``)
// matches what ``ComputeContext::WriteToMetadata`` would produce.
TEST(BackendTestCaseShapeInference, ReleaseEventEmittedForBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("release");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_release_shape_reshape") {
      continue;
    }
    found = true;

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(tc.model())) << "case: " << tc.name;

    core::compute::ComputeContext inplace;
    inplace.set_events_enabled(true);
    ASSERT_NO_THROW(inplace.ComputeInPlaceReuseGraph(tc.model().ref_graph(), ctx, false, {}))
        << "case: " << tc.name;

    // Exactly one kRelease event must be emitted, for "S" at node 1
    // (the Reshape node, which is S's last consumer).
    using core::compute::ComputeEventAction;
    int release_count = 0;
    for (const auto &ev : inplace.Events()) {
      if (ev.action == ComputeEventAction::kRelease) {
        ++release_count;
        EXPECT_EQ(ev.name, "S") << "release event must name 'S'";
        EXPECT_EQ(ev.node_index, 1u) << "release event must fire at node 1 (Reshape)";
      }
    }
    EXPECT_EQ(release_count, 1) << "expected exactly one kRelease event";

    // Verify that the pre-embedded node metadata matches what WriteToMetadata
    // would produce: node 0 (Shape) has no release metadata, node 1 (Reshape)
    // carries kReleaseAfterMetadataKey for "S".
    const std::vector<MetadataMap> expected_node_meta = {
        {},
        {{std::string(core::compute::kReleaseAfterMetadataKey), "S"},
         {std::string(core::compute::kNotUsedAfterMetadataKey), "X"}}};
    const auto &nodes = tc.model().ref_graph().ref_node();
    ASSERT_EQ(nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_release_shape_reshape case not registered";
}

// Verifies that ``ComputeInPlaceReuseGraph`` correctly reports both a declared
// graph input and a graph initializer under ``kNotUsedAfterMetadataKey`` at
// the node where they reach their last use.
//
// In ``test_cc_release_initializer_add`` the graph is:
//   ``Add(X, W) → Relu``, where ``W`` is an initializer.
// Both ``X`` and ``W`` are consumed only by ``Add`` (node 0), so node 0 must
// carry ``onnx_light.not_used_after = "X;W"``.  ``T`` (the Add output) is
// the sole input to ``Relu`` (node 1) and must be released there.
TEST(BackendTestCaseShapeInference, ReleaseInitializerNotUsedAfterMetadataForBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("release");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_release_initializer_add") {
      continue;
    }
    found = true;

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(tc.model())) << "case: " << tc.name;

    core::compute::ComputeContext inplace;
    inplace.set_events_enabled(true);
    ASSERT_NO_THROW(inplace.ComputeInPlaceReuseGraph(tc.model().ref_graph(), ctx, false, {}))
        << "case: " << tc.name;

    // Exactly one kRelease event must be emitted, for "T" at node 1 (Relu).
    using core::compute::ComputeEventAction;
    int release_count = 0;
    for (const auto &ev : inplace.Events()) {
      if (ev.action == ComputeEventAction::kRelease) {
        ++release_count;
        EXPECT_EQ(ev.name, "T") << "release event must name 'T'";
        EXPECT_EQ(ev.node_index, 1u) << "release event must fire at node 1 (Relu)";
      }
    }
    EXPECT_EQ(release_count, 1) << "expected exactly one kRelease event";

    // Verify that the pre-embedded node metadata matches what WriteToMetadata
    // would produce: node 0 (Add) carries kNotUsedAfterMetadataKey for both
    // the graph input "X" and the initializer "W"; node 1 (Relu) carries
    // kReleaseAfterMetadataKey for "T".
    const std::vector<MetadataMap> expected_node_meta = {
        {{std::string(core::compute::kNotUsedAfterMetadataKey), "X;W"}},
        {{std::string(core::compute::kReleaseAfterMetadataKey), "T"}}};
    const auto &nodes = tc.model().ref_graph().ref_node();
    ASSERT_EQ(nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_release_initializer_add case not registered";
}

// Verifies that WriteValueAndNodeTagsToMetadata applied to a clean copy of
// the shape-tag backend test case (Constant→Reshape) produces metadata that
// matches the expected values pre-embedded in the model.  Specifically it
// confirms that ``S`` (the output of the ``Constant`` node used as
// ``Reshape``'s shape input) receives the ``"shape"`` value tag — because
// "shape" has higher priority than "weight" — and that the ``Constant`` node
// itself is tagged ``"shape"``.
TEST(BackendTestCaseShapeInference, OnnxOptimWritesShapeTagMetadataOnConstantReshapeBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("shape_tag");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_tag_constant_reshape_ambiguous") {
      continue;
    }
    found = true;

    // Collect the expected graph-level metadata pre-embedded in the case model.
    const MetadataMap expected_graph_meta = MetadataOf(tc.model().ref_graph());
    ASSERT_EQ(expected_graph_meta.count(core::compute::kValueTagsMetadataKey), 0u)
        << "feature metadata must not be stored on the graph in case";

    // Collect per-node expected metadata from the pre-embedded model.
    const auto &src_nodes = tc.model().ref_graph().ref_node();
    std::vector<MetadataMap> expected_node_meta;
    expected_node_meta.reserve(src_nodes.size());
    for (const auto &node : src_nodes) {
      expected_node_meta.push_back(MetadataOf(node));
    }

    // Collect expected input and output metadata from the pre-embedded model.
    const auto &src_inputs = tc.model().ref_graph().ref_input();
    std::vector<MetadataMap> expected_input_meta;
    expected_input_meta.reserve(src_inputs.size());
    for (const auto &vi : src_inputs) {
      expected_input_meta.push_back(MetadataOf(vi));
    }
    const auto &src_outputs = tc.model().ref_graph().ref_output();
    std::vector<MetadataMap> expected_output_meta;
    expected_output_meta.reserve(src_outputs.size());
    for (const auto &vi : src_outputs) {
      expected_output_meta.push_back(MetadataOf(vi));
    }

    // Make a clean copy of the model and strip all metadata so
    // WriteValueAndNodeTagsToMetadata starts from a blank slate.
    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    for (size_t n = 0; n < graph->node().size(); ++n) {
      graph->mutable_node(n)->mutable_metadata_props()->clear();
    }
    for (size_t vi = 0; vi < graph->value_info().size(); ++vi) {
      graph->mutable_value_info(vi)->mutable_metadata_props()->clear();
    }
    for (size_t i = 0; i < graph->input().size(); ++i) {
      graph->mutable_input(i)->mutable_metadata_props()->clear();
    }
    for (size_t o = 0; o < graph->output().size(); ++o) {
      graph->mutable_output(o)->mutable_metadata_props()->clear();
    }

    // Run WriteValueAndNodeTagsToMetadata and verify the result matches the
    // pre-embedded expected values.
    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;

    EXPECT_EQ(MetadataOf(*graph), expected_graph_meta)
        << "graph metadata mismatch in case " << tc.name;

    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes
    // onnx_light.value_tag = "shape" on the value_info entry for "S".
    const auto &result_vis = graph->ref_value_info();
    const auto &src_vis = tc.model().ref_graph().ref_value_info();
    ASSERT_EQ(result_vis.size(), src_vis.size());
    for (size_t vi = 0; vi < result_vis.size(); ++vi) {
      EXPECT_EQ(MetadataOf(result_vis[vi]), MetadataOf(src_vis[vi]))
          << "value_info[" << vi << "] metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes onnx_light.value_tag
    // on input and output entries.
    const auto &result_inputs = graph->ref_input();
    ASSERT_EQ(result_inputs.size(), expected_input_meta.size());
    for (size_t i = 0; i < result_inputs.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_inputs[i]), expected_input_meta[i])
          << "input[" << i << "] metadata mismatch in case " << tc.name;
    }
    const auto &result_outputs = graph->ref_output();
    ASSERT_EQ(result_outputs.size(), expected_output_meta.size());
    for (size_t o = 0; o < result_outputs.size(); ++o) {
      EXPECT_EQ(MetadataOf(result_outputs[o]), expected_output_meta[o])
          << "output[" << o << "] metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_tag_constant_reshape_ambiguous case not registered";
}

// Verifies that WriteValueAndNodeTagsToMetadata applied to a clean copy of
// the ``Constant → Mul → Concat → Reshape`` shape-tag backend test case
// produces metadata matching the expected values pre-embedded in the model.
// The propagation path is:
//   S_full (Reshape's shape input) → "shape"
//   Concat backward → S1 and S2 → "shape"
//   Constant (S1) updated → "shape";  Mul inherits "shape" from S1
//   Constant (two) stays "weight"
TEST(BackendTestCaseShapeInference,
     OnnxOptimWritesShapeTagMetadataOnConstantMulConcatReshapeBackendCase) {
  const std::vector<TestCase> cases = CollectTestCases("shape_tag");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_tag_constant_mul_concat_reshape") {
      continue;
    }
    found = true;

    // Collect the expected graph-level metadata pre-embedded in the case model.
    const MetadataMap expected_graph_meta = MetadataOf(tc.model().ref_graph());
    ASSERT_EQ(expected_graph_meta.count(core::compute::kValueTagsMetadataKey), 0u)
        << "feature metadata must not be stored on the graph in case";

    // Collect per-node expected metadata from the pre-embedded model.
    const auto &src_nodes = tc.model().ref_graph().ref_node();
    std::vector<MetadataMap> expected_node_meta;
    expected_node_meta.reserve(src_nodes.size());
    for (const auto &node : src_nodes) {
      expected_node_meta.push_back(MetadataOf(node));
    }

    // Collect expected input and output metadata from the pre-embedded model.
    const auto &src_inputs = tc.model().ref_graph().ref_input();
    std::vector<MetadataMap> expected_input_meta;
    expected_input_meta.reserve(src_inputs.size());
    for (const auto &vi : src_inputs) {
      expected_input_meta.push_back(MetadataOf(vi));
    }
    const auto &src_outputs = tc.model().ref_graph().ref_output();
    std::vector<MetadataMap> expected_output_meta;
    expected_output_meta.reserve(src_outputs.size());
    for (const auto &vi : src_outputs) {
      expected_output_meta.push_back(MetadataOf(vi));
    }

    // Make a clean copy of the model and strip all metadata so
    // WriteValueAndNodeTagsToMetadata starts from a blank slate.
    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    for (size_t n = 0; n < graph->node().size(); ++n) {
      graph->mutable_node(n)->mutable_metadata_props()->clear();
    }
    for (size_t vi = 0; vi < graph->value_info().size(); ++vi) {
      graph->mutable_value_info(vi)->mutable_metadata_props()->clear();
    }
    for (size_t i = 0; i < graph->input().size(); ++i) {
      graph->mutable_input(i)->mutable_metadata_props()->clear();
    }
    for (size_t o = 0; o < graph->output().size(); ++o) {
      graph->mutable_output(o)->mutable_metadata_props()->clear();
    }

    // Run WriteValueAndNodeTagsToMetadata and verify the result matches the
    // pre-embedded expected values.
    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;

    EXPECT_EQ(MetadataOf(*graph), expected_graph_meta)
        << "graph metadata mismatch in case " << tc.name;

    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes
    // onnx_light.value_tag on each value_info entry.
    const auto &result_vis = graph->ref_value_info();
    const auto &src_vis = tc.model().ref_graph().ref_value_info();
    ASSERT_EQ(result_vis.size(), src_vis.size());
    for (size_t vi = 0; vi < result_vis.size(); ++vi) {
      EXPECT_EQ(MetadataOf(result_vis[vi]), MetadataOf(src_vis[vi]))
          << "value_info[" << vi << "] metadata mismatch in case " << tc.name;
    }

    // Verify that WriteValueAndNodeTagsToMetadata also writes onnx_light.value_tag
    // on input and output entries.
    const auto &result_inputs = graph->ref_input();
    ASSERT_EQ(result_inputs.size(), expected_input_meta.size());
    for (size_t i = 0; i < result_inputs.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_inputs[i]), expected_input_meta[i])
          << "input[" << i << "] metadata mismatch in case " << tc.name;
    }
    const auto &result_outputs = graph->ref_output();
    ASSERT_EQ(result_outputs.size(), expected_output_meta.size());
    for (size_t o = 0; o < result_outputs.size(); ++o) {
      EXPECT_EQ(MetadataOf(result_outputs[o]), expected_output_meta[o])
          << "output[" << o << "] metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_tag_constant_mul_concat_reshape case not registered";
}

// Verifies that WriteValueAndNodeTagsToMetadata correctly annotates a graph
// whose output is directly a shape tensor (Shape(X) → Y, where Y is the
// graph output). In particular, the per-output onnx_light.value_tag = "shape"
// must be written by WriteValueAndNodeTagsToMetadata even when the output
// starts with no pre-existing metadata.
TEST(BackendTestCaseShapeInference, OnnxOptimWritesShapeTagToOutputWhenOutputIsShapeTensor) {
  const std::vector<TestCase> cases = CollectTestCases("shape_tag");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_tag_output_is_shape") {
      continue;
    }
    found = true;

    // Collect the expected metadata from the pre-embedded model.
    const MetadataMap expected_graph_meta = MetadataOf(tc.model().ref_graph());
    ASSERT_EQ(expected_graph_meta.count(core::compute::kValueTagsMetadataKey), 0u)
        << "feature metadata must not be stored on the graph in case";

    const auto &src_nodes = tc.model().ref_graph().ref_node();
    std::vector<MetadataMap> expected_node_meta;
    expected_node_meta.reserve(src_nodes.size());
    for (const auto &node : src_nodes) {
      expected_node_meta.push_back(MetadataOf(node));
    }

    // Collect expected input and output metadata from the pre-embedded model.
    const auto &src_inputs = tc.model().ref_graph().ref_input();
    std::vector<MetadataMap> expected_input_meta;
    expected_input_meta.reserve(src_inputs.size());
    for (const auto &vi : src_inputs) {
      expected_input_meta.push_back(MetadataOf(vi));
    }
    const auto &src_outputs = tc.model().ref_graph().ref_output();
    std::vector<MetadataMap> expected_output_meta;
    expected_output_meta.reserve(src_outputs.size());
    for (const auto &vi : src_outputs) {
      expected_output_meta.push_back(MetadataOf(vi));
    }

    // Make a clean copy and strip ALL metadata so WriteValueAndNodeTagsToMetadata
    // has to infer everything from scratch.
    ModelProto model_copy;
    std::string serialized;
    tc.model().SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    for (size_t n = 0; n < graph->node().size(); ++n) {
      graph->mutable_node(n)->mutable_metadata_props()->clear();
    }
    for (size_t vi = 0; vi < graph->value_info().size(); ++vi) {
      graph->mutable_value_info(vi)->mutable_metadata_props()->clear();
    }
    for (size_t i = 0; i < graph->input().size(); ++i) {
      graph->mutable_input(i)->mutable_metadata_props()->clear();
    }
    for (size_t o = 0; o < graph->output().size(); ++o) {
      graph->mutable_output(o)->mutable_metadata_props()->clear();
    }

    // Run WriteValueAndNodeTagsToMetadata from a blank slate.
    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;

    // Verify graph-level metadata.
    EXPECT_EQ(MetadataOf(*graph), expected_graph_meta)
        << "graph metadata mismatch in case " << tc.name;

    // Verify node metadata.
    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_nodes[i]), expected_node_meta[i])
          << "node " << i << " metadata mismatch in case " << tc.name;
    }

    // Verify input metadata.
    const auto &result_inputs = graph->ref_input();
    ASSERT_EQ(result_inputs.size(), expected_input_meta.size());
    for (size_t i = 0; i < result_inputs.size(); ++i) {
      EXPECT_EQ(MetadataOf(result_inputs[i]), expected_input_meta[i])
          << "input[" << i << "] metadata mismatch in case " << tc.name;
    }

    // Verify output metadata — specifically that Y receives value_tag = "shape".
    const auto &result_outputs = graph->ref_output();
    ASSERT_EQ(result_outputs.size(), expected_output_meta.size());
    for (size_t o = 0; o < result_outputs.size(); ++o) {
      EXPECT_EQ(MetadataOf(result_outputs[o]), expected_output_meta[o])
          << "output[" << o << "] metadata mismatch in case " << tc.name;
    }
  }
  ASSERT_TRUE(found) << "test_cc_shape_tag_output_is_shape case not registered";
}

// Guards two invariants for every ``test_cc_shape_tag_*`` fixture:
//   * WriteValueAndNodeTagsToMetadata stamps ``onnx_light.value_tag`` on every
//     graph input, output, and initializer (always-known values), and
//   * each fixture pre-embeds the *expected* value_tag on those same values
//     with a value that matches what is recomputed.
// The second check catches fixtures that forget the expected shape tags on
// inputs, outputs, or initializers (which would otherwise pass silently).
TEST(BackendTestCaseShapeInference, OnnxOptimWritesValueTagOnEveryGraphValueInShapeTagCases) {
  const std::vector<TestCase> cases = CollectTestCases("shape_tag");
  bool found = false;
  for (const TestCase &tc : cases) {
    found = true;

    ModelProto model_copy;
    std::string serialized;
    ASSERT_TRUE(tc.model().SerializeToString(serialized))
        << "failed to serialize case: " << tc.name;
    ASSERT_TRUE(model_copy.ParseFromString(serialized)) << "failed to parse case: " << tc.name;

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    const auto clear_metadata = [](auto *mutable_entries) {
      for (size_t idx = 0; idx < mutable_entries->size(); ++idx) {
        (*mutable_entries)[idx].mutable_metadata_props()->clear();
      }
    };
    clear_metadata(graph->mutable_node());
    clear_metadata(graph->mutable_value_info());
    clear_metadata(graph->mutable_input());
    clear_metadata(graph->mutable_output());
    clear_metadata(graph->mutable_initializer());

    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;

    const GraphProto &original_graph = tc.model().ref_graph();

    const auto has_value_tag = [&](const auto &value) {
      return MetadataOf(value).contains(core::compute::kValueTagMetadataKey);
    };
    const auto has_node_tag = [&](const auto &node) {
      return MetadataOf(node).contains(core::compute::kNodeTagMetadataKey);
    };
    // Inputs, outputs, and initializers must always carry a value_tag after
    // WriteValueAndNodeTagsToMetadata: this information is always known.
    const auto expect_all_have_value_tag = [&](const auto &values, const char *kind) {
      for (size_t idx = 0; idx < values.size(); ++idx) {
        ASSERT_TRUE(has_value_tag(values[idx]))
            << "missing value_tag on " << kind << "[" << idx << "] in case " << tc.name;
      }
    };
    const auto expect_matching_value_tags = [&](const auto &values, const auto &expected_values,
                                                const char *kind) {
      ASSERT_EQ(values.size(), expected_values.size())
          << "size mismatch on " << kind << " in case " << tc.name;
      for (size_t idx = 0; idx < values.size(); ++idx) {
        const bool expected_tagged = has_value_tag(expected_values[idx]);
        EXPECT_EQ(has_value_tag(values[idx]), expected_tagged)
            << "value tag presence mismatch on " << kind << "[" << idx << "] in case " << tc.name;
      }
    };
    const auto expect_matching_node_tags = [&](const auto &nodes, const auto &expected_nodes) {
      ASSERT_EQ(nodes.size(), expected_nodes.size()) << "node size mismatch in case " << tc.name;
      for (size_t idx = 0; idx < nodes.size(); ++idx) {
        const bool expected_tagged = has_node_tag(expected_nodes[idx]);
        EXPECT_EQ(has_node_tag(nodes[idx]), expected_tagged)
            << "node tag presence mismatch on node[" << idx << "] in case " << tc.name;
      }
    };

    const auto value_tag_of = [&](const auto &value) {
      return MetadataOf(value).at(core::compute::kValueTagMetadataKey);
    };
    // Inputs, outputs, and initializers must carry a *pre-embedded* (expected)
    // value_tag whose value matches what WriteValueAndNodeTagsToMetadata
    // recomputes. This guards against fixtures that forget to embed the
    // expected shape tags on these always-known graph values.
    const auto expect_matching_value_tag_values =
        [&](const auto &values, const auto &expected_values, const char *kind) {
          ASSERT_EQ(values.size(), expected_values.size())
              << "size mismatch on " << kind << " in case " << tc.name;
          for (size_t idx = 0; idx < values.size(); ++idx) {
            ASSERT_TRUE(has_value_tag(expected_values[idx]))
                << "missing expected value_tag on " << kind << "[" << idx << "] in case "
                << tc.name;
            EXPECT_EQ(value_tag_of(values[idx]), value_tag_of(expected_values[idx]))
                << "value_tag mismatch on " << kind << "[" << idx << "] in case " << tc.name;
          }
        };

    expect_matching_node_tags(graph->ref_node(), original_graph.ref_node());
    // Every input, output, and initializer must have a value_tag (always known).
    expect_all_have_value_tag(graph->ref_input(), "input");
    expect_all_have_value_tag(graph->ref_output(), "output");
    expect_all_have_value_tag(graph->ref_initializer(), "initializer");
    // The pre-embedded (expected) tags on those values must be present and equal
    // to the recomputed ones.
    expect_matching_value_tag_values(graph->ref_input(), original_graph.ref_input(), "input");
    expect_matching_value_tag_values(graph->ref_output(), original_graph.ref_output(), "output");
    expect_matching_value_tag_values(graph->ref_initializer(), original_graph.ref_initializer(),
                                     "initializer");
    // For intermediate value_info, check that presence matches the pre-embedded data.
    expect_matching_value_tags(graph->ref_value_info(), original_graph.ref_value_info(),
                               "value_info");
  }
  ASSERT_TRUE(found) << "no shape_tag backend cases were collected";
}

// ---------------------------------------------------------------------------
// Big-model smoke tests
//
// These tests collect only big-model cases (CollectTestCases("", true)
// filtered to names containing "_big_") and verify that the four main
// optimisation passes complete without throwing. For ``kind == "model"``
// cases the expected output shapes and intermediate ``value_info`` shapes
// recorded in the model are also validated after shape inference.
// ---------------------------------------------------------------------------

namespace {

// Deep-copies a ModelProto by serializing and re-parsing it.
// Asserts on failure so the test is immediately aborted.
ModelProto DeepCopyModel(const ModelProto &src, const std::string &tc_name) {
  std::string serialized;
  EXPECT_TRUE(src.SerializeToString(serialized)) << "failed to serialize case: " << tc_name;
  ModelProto copy;
  EXPECT_TRUE(copy.ParseFromString(serialized)) << "failed to parse case: " << tc_name;
  return copy;
}

} // namespace

TEST(BackendTestCaseShapeInference, BigModelsOptimShapeInference) {
  const std::vector<TestCase> cases = CollectTestCases("", /*include_big=*/true);
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name.find("_big_") == std::string::npos) {
      continue;
    }
    found = true;
    SCOPED_TRACE(tc.name);

    ModelProto model_copy = DeepCopyModel(tc.model(), tc.name);

    // For ``kind == "model"`` cases, capture the expected intermediate shapes
    // recorded in ``value_info`` so we can verify inference recovers them.
    const auto expected = SnapshotAndStripOutputs(model_copy);
    std::vector<ExpectedOutput> expected_value_info;
    if (tc.kind == "model") {
      expected_value_info = SnapshotAndStripValueInfo(model_copy);
    }
    model_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(core::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const auto &outputs = model_copy.ref_graph().ref_output();
    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &out = outputs[i];
      ASSERT_TRUE(out.has_type()) << "output " << expected[i].name << " missing type";
      const TypeProto::Tensor *tt_ptr = TensorTypeOf(out.ref_type());
      ASSERT_NE(tt_ptr, nullptr) << "output " << expected[i].name << " not a tensor";
      const auto &tt = *tt_ptr;
      EXPECT_EQ(static_cast<int32_t>(tt.elem_type()), expected[i].elem_type)
          << "elem_type mismatch on output " << expected[i].name;
      const auto inferred_dims = DimsOf(tt);
      if (!inferred_dims.empty() || tt.has_shape()) {
        ASSERT_EQ(inferred_dims.size(), expected[i].shape.size())
            << "rank mismatch on output " << expected[i].name;
        for (size_t d = 0; d < inferred_dims.size(); ++d) {
          if (inferred_dims[d] != -1 && expected[i].shape[d] != -1) {
            EXPECT_EQ(inferred_dims[d], expected[i].shape[d])
                << "dim[" << d << "] mismatch on output " << expected[i].name;
          }
        }
      }
    }

    if (tc.kind == "model") {
      CheckValueInfoMatchesExpected(model_copy.ref_graph(), expected_value_info);
    }
  }
  EXPECT_TRUE(found) << "no big-model backend cases were collected";
}

TEST(BackendTestCaseShapeInference, BigModelsReleaseInfo) {
  const std::vector<TestCase> cases = CollectTestCases("", /*include_big=*/true);
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name.find("_big_") == std::string::npos) {
      continue;
    }
    found = true;
    SCOPED_TRACE(tc.name);

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(tc.model())) << "case: " << tc.name;

    core::compute::ComputeContext inplace;
    inplace.set_events_enabled(true);
    ASSERT_NO_THROW(inplace.ComputeInPlaceReuseGraph(tc.model().ref_graph(), ctx, false, {}))
        << "case: " << tc.name;
  }
  EXPECT_TRUE(found) << "no big-model backend cases were collected";
}

TEST(BackendTestCaseShapeInference, BigModelsShapeTag) {
  const std::vector<TestCase> cases = CollectTestCases("", /*include_big=*/true);
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name.find("_big_") == std::string::npos) {
      continue;
    }
    found = true;
    SCOPED_TRACE(tc.name);

    ModelProto model_copy = DeepCopyModel(tc.model(), tc.name);

    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    const auto clear_meta = [](auto *entries) {
      for (size_t i = 0; i < entries->size(); ++i) {
        (*entries)[i].mutable_metadata_props()->clear();
      }
    };
    clear_meta(graph->mutable_node());
    clear_meta(graph->mutable_value_info());
    clear_meta(graph->mutable_input());
    clear_meta(graph->mutable_output());
    clear_meta(graph->mutable_initializer());

    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;
  }
  EXPECT_TRUE(found) << "no big-model backend cases were collected";
}

TEST(BackendTestCaseShapeInference, BigModelsInplaceInfo) {
  const std::vector<TestCase> cases = CollectTestCases("", /*include_big=*/true);
  // Per-node metadata keys produced by the value-tag / in-place-reuse passes
  // that a big-model case may embed as a golden expectation.
  const std::array<const char *, 4> checked_node_keys = {
      core::compute::kNodeTagMetadataKey, core::compute::kInPlaceReuseMetadataKey,
      core::compute::kReleaseAfterMetadataKey, core::compute::kReleaseAfterShapeTagMetadataKey};
  const auto checked_subset = [&](const MetadataMap &full) {
    MetadataMap subset;
    for (const char *key : checked_node_keys) {
      auto it = full.find(key);
      if (it != full.end()) {
        subset.emplace(*it);
      }
    }
    return subset;
  };

  bool found = false;
  bool verified_expected = false;
  for (const TestCase &tc : cases) {
    if (tc.name.find("_big_") == std::string::npos) {
      continue;
    }
    found = true;
    SCOPED_TRACE(tc.name);

    core::shapes::ShapesContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapeModel(tc.model())) << "case: " << tc.name;

    ASSERT_NO_THROW(core::compute::ComputeInPlaceReuse(tc.model().ref_graph(), ctx,
                                                       /*allow_input_overwrite=*/true))
        << "case: " << tc.name;

    // Capture the golden metadata embedded in the case (if any). A case that
    // pre-embeds the expected per-value/per-node tags and in-place-reuse
    // metadata is verified against a fresh recomputation below.
    const GraphProto &expected_graph = tc.model().ref_graph();
    const MetadataMap expected_graph_meta = MetadataOf(expected_graph);
    bool has_expected = false;
    std::vector<MetadataMap> expected_node_meta;
    for (const auto &node : expected_graph.ref_node()) {
      MetadataMap subset = checked_subset(MetadataOf(node));
      if (!subset.empty()) {
        has_expected = true;
      }
      expected_node_meta.push_back(std::move(subset));
    }
    // A case may embed only per-value tags (onnx_light.value_tag) without any
    // in-place-reuse node metadata; treat those as golden too.
    for (const auto &vi : expected_graph.ref_value_info()) {
      if (MetadataOf(vi).count(core::compute::kValueTagMetadataKey) != 0) {
        has_expected = true;
        break;
      }
    }
    if (!has_expected) {
      continue;
    }
    verified_expected = true;

    // Recompute the metadata from scratch on a clean copy and confirm it
    // matches the golden values embedded in the case.
    ModelProto model_copy = DeepCopyModel(tc.model(), tc.name);
    GraphProto *graph = model_copy.mutable_graph();
    graph->mutable_metadata_props()->clear();
    const auto clear_meta = [](auto *entries) {
      for (size_t i = 0; i < entries->size(); ++i) {
        (*entries)[i].mutable_metadata_props()->clear();
      }
    };
    clear_meta(graph->mutable_node());
    clear_meta(graph->mutable_value_info());
    clear_meta(graph->mutable_input());
    clear_meta(graph->mutable_output());
    clear_meta(graph->mutable_initializer());

    core::shapes::ShapesContext recompute_ctx;
    ASSERT_NO_THROW(recompute_ctx.ComputeShapeModel(model_copy)) << "case: " << tc.name;
    const auto value_tags = core::compute::InferValueAndNodeTags(*graph).first;
    ASSERT_NO_THROW(core::compute::WriteValueAndNodeTagsToMetadata(*graph)) << "case: " << tc.name;
    ASSERT_NO_THROW(core::compute::WriteInPlaceReuseToMetadata(*graph, recompute_ctx, value_tags))
        << "case: " << tc.name;

    const MetadataMap computed_graph_meta = MetadataOf(*graph);
    EXPECT_EQ(computed_graph_meta.count(core::compute::kValueTagsMetadataKey), 0u)
        << "value_tags aggregate must not be written on the graph metadata for case " << tc.name;

    // Verify per-value shape/weight tags: the ``onnx_light.value_tag`` metadata
    // that the pass writes onto each ValueInfoProto (inputs, value_info,
    // outputs) and initializer TensorProto. A case that pre-embeds these golden
    // per-value tags (e.g. cases_qwen3_4_layers_like.cc / cases_tiny_llm.cc) has
    // them recomputed and asserted here, so the tags live on each value and not
    // only in the graph-level JSON.
    const auto per_value_tags = [](const GraphProto &g) {
      std::unordered_map<std::string, std::string> tags;
      const auto collect = [&](const auto &entries) {
        for (const auto &e : entries) {
          const MetadataMap meta = MetadataOf(e);
          auto it = meta.find(core::compute::kValueTagMetadataKey);
          if (it != meta.end()) {
            tags[e.name()] = it->second;
          }
        }
      };
      collect(g.ref_input());
      collect(g.ref_value_info());
      collect(g.ref_output());
      collect(g.ref_initializer());
      return tags;
    };
    const auto expected_per_value = per_value_tags(expected_graph);
    if (!expected_per_value.empty()) {
      EXPECT_EQ(per_value_tags(*graph), expected_per_value)
          << "per-value value_tag metadata mismatch on case " << tc.name;
    }
    const auto &result_nodes = graph->ref_node();
    ASSERT_EQ(result_nodes.size(), expected_node_meta.size());
    for (size_t i = 0; i < result_nodes.size(); ++i) {
      EXPECT_EQ(checked_subset(MetadataOf(result_nodes[i])), expected_node_meta[i])
          << "in-place-reuse metadata mismatch on node " << i << " in case " << tc.name;
    }
  }
  EXPECT_TRUE(found) << "no big-model backend cases were collected";
  EXPECT_TRUE(verified_expected)
      << "no big-model backend case carried expected in-place-reuse metadata";
}

} // namespace Test
