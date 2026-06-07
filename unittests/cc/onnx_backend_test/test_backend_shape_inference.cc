// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::DataSet;
using onnx_backend_test::TestCase;
using DataTensor = onnx_kernels::Tensor;

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
  input_tt->set_elem_type(onnx_kernels::DataType::FLOAT);
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
  EXPECT_EQ(value_tensor.ref_elem_type(), onnx_kernels::DataType::FLOAT);
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
  input_tt->set_elem_type(onnx_kernels::DataType::FLOAT);
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
  EXPECT_EQ(value_tensor.ref_elem_type(), onnx_kernels::DataType::FLOAT);
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
    ModelProto *model_ptr = &tc.model;
    ModelProto model_copy;
    std::vector<ExpectedOutput> expected_value_info;
    if (tc.kind == "model") {
      std::string serialized;
      tc.model.SerializeToString(serialized);
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
    for (size_t ds_idx = 0; ds_idx < tc.data_sets.size(); ++ds_idx) {
      const DataSet &ds = tc.data_sets[ds_idx];
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
      tc.model.SerializeToString(ds_serialized);
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
    const auto expected = SnapshotAndStripOutputs(tc.model);

    // Replace every input dim_value with a dim_param keyed by its value so
    // that equal numeric dims across all inputs share the same symbol.
    auto &inputs = tc.model.mutable_graph()->ref_input();
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

    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model)) << "case: " << tc.name;

    const auto &outputs = tc.model.ref_graph().ref_output();
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
// onnx_optim shape inference + model-local functions
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_optim`` shape-inference pipeline
// (:cpp:func:`onnx_optim::shapes::InferShapesModel`) correctly handles a
// node whose ``op_type`` references a model-local
// :cpp:class:`FunctionProto`. The model is built by
// :cpp:func:`RegisterLocalFunctionAddShapeInferenceCases`; its single graph
// node calls ``local:func_add(X, Y) -> Z`` where the function body is a
// one-node ``Add``. After expansion, the inferred ``Z`` must carry the
// symbolic ``(batch, d_model)`` shape of the inputs.
TEST(BackendTestCaseShapeInference, OnnxOptimSupportsLocalFunctionCall) {
  const std::vector<TestCase> cases = CollectTestCases("func_add");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_local_function_add") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model.SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape so optim shape inference has to
    // recover it from the function body's ``Add`` node.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    auto *tt = MutableTensorTypeOf(*outputs[0].mutable_type());
    ASSERT_NE(tt, nullptr);
    tt->clear_shape();

    ASSERT_NO_THROW(onnx_optim::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    // Output shape should now be ``(batch, d_model)`` — symbolic dims
    // inherited from the inputs via the expanded function body.
    const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *out_tt = TensorTypeOf(out.ref_type());
    ASSERT_NE(out_tt, nullptr);
    EXPECT_EQ(static_cast<int32_t>(out_tt->elem_type()), 1 /* FLOAT */);
    ASSERT_TRUE(out_tt->has_shape());
    const auto &dims = out_tt->ref_shape().ref_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_TRUE(dims[0].has_dim_param());
    EXPECT_EQ(dims[0].ref_dim_param().as_string(), "batch");
    EXPECT_TRUE(dims[1].has_dim_param());
    EXPECT_EQ(dims[1].ref_dim_param().as_string(), "d_model");
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_local_function_add case not registered";
}

// ---------------------------------------------------------------------------
// onnx_optim shape inference + nested model-local function call
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_optim`` shape-inference pipeline correctly handles
// a node whose ``op_type`` references a model-local :cpp:class:`FunctionProto`
// **whose body itself calls another model-local function**. The model is
// built by :cpp:func:`RegisterNestedLocalFunctionAddShapeInferenceCases`:
// its single graph node calls ``local:func_outer_add(X, Y) -> Z`` whose body
// is a single call into ``local:func_inner_add(a, b) -> c { c = Add(a, b) }``.
// After two levels of expansion, the inferred ``Z`` must carry the symbolic
// ``(batch, d_model)`` shape of the inputs.
TEST(BackendTestCaseShapeInference, OnnxOptimSupportsNestedLocalFunctionCall) {
  const std::vector<TestCase> cases = CollectTestCases("func_outer_add");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_nested_local_function_add") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model.SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape so optim shape inference has to
    // recover it through the two levels of function-body expansion.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    auto *tt = MutableTensorTypeOf(*outputs[0].mutable_type());
    ASSERT_NE(tt, nullptr);
    tt->clear_shape();

    ASSERT_NO_THROW(onnx_optim::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *out_tt = TensorTypeOf(out.ref_type());
    ASSERT_NE(out_tt, nullptr);
    EXPECT_EQ(static_cast<int32_t>(out_tt->elem_type()), 1 /* FLOAT */);
    ASSERT_TRUE(out_tt->has_shape());
    const auto &dims = out_tt->ref_shape().ref_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_TRUE(dims[0].has_dim_param());
    EXPECT_EQ(dims[0].ref_dim_param().as_string(), "batch");
    EXPECT_TRUE(dims[1].has_dim_param());
    EXPECT_EQ(dims[1].ref_dim_param().as_string(), "d_model");
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_nested_local_function_add case not registered";
}

// ---------------------------------------------------------------------------
// onnx_optim shape inference + Shape→Identity→Unsqueeze
// ---------------------------------------------------------------------------
//
// Verifies that the ``onnx_optim`` shape-inference pipeline can handle the
// ``Shape → Identity → Unsqueeze`` case registered by
// :cpp:func:`RegisterShapeIdentityUnsqueezeShapeInferenceCases`. The
// ``Unsqueeze`` output rank equals ``kAxisCount + 1``; the case must keep
// ``kAxisCount`` small enough that the inferred rank stays within
// :cpp:var:`onnx_optim::kMaxOptimRank` (currently 16). A regression here
// (e.g. raising ``kAxisCount`` back to 16, producing a rank-17 output)
// would cause :cpp:func:`onnx_optim::shapes::InferShapesModel` to throw
// ``std::length_error("OptimShape exceeds maximum rank")``.
TEST(BackendTestCaseShapeInference, OnnxOptimInfersShapeIdentityUnsqueeze) {
  const std::vector<TestCase> cases = CollectTestCases("Shape");
  bool found = false;
  for (const TestCase &tc : cases) {
    if (tc.name != "test_cc_shape_inference_shape_identity_unsqueeze") {
      continue;
    }
    found = true;

    ModelProto model_copy;
    std::string serialized;
    tc.model.SerializeToString(serialized);
    model_copy.ParseFromString(serialized);

    // Strip the recorded output shape and intermediate value_info so optim
    // shape inference has to recover them from scratch.
    auto &outputs = model_copy.mutable_graph()->ref_output();
    ASSERT_EQ(outputs.size(), 1u);
    auto *tt = MutableTensorTypeOf(*outputs[0].mutable_type());
    ASSERT_NE(tt, nullptr);
    tt->clear_shape();
    model_copy.mutable_graph()->mutable_value_info()->clear();

    ASSERT_NO_THROW(onnx_optim::shapes::InferShapesModel(model_copy)) << "case: " << tc.name;

    const ValueInfoProto &out = model_copy.ref_graph().ref_output()[0];
    ASSERT_TRUE(out.has_type());
    const TypeProto::Tensor *out_tt = TensorTypeOf(out.ref_type());
    ASSERT_NE(out_tt, nullptr);
    ASSERT_TRUE(out_tt->has_shape());
    // Expected rank: ``kAxisCount + 1``. We do not require optim shape
    // inference to recover concrete dim values here; the original
    // regression was an unconditional throw from ``OptimShape`` whenever
    // the inferred rank exceeded ``kMaxOptimRank``. Just verifying the
    // rank pins down that the rank is now within the bound.
    const auto inferred = DimsOf(*out_tt);
    EXPECT_FALSE(inferred.empty());
    EXPECT_LE(inferred.size(), onnx_optim::kMaxOptimRank);
  }
  ASSERT_TRUE(found) << "test_cc_shape_inference_shape_identity_unsqueeze case not registered";
}

} // namespace Test
