// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_extensions/backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Reshape`` — the simplest pattern that exercises the shape-tag
// annotation path. ``Shape(X)`` produces an INT64 tensor whose *value*
// represents the shape of its input, so ``WriteValueAndNodeTagsToMetadata``
// must emit:
//
//   * ``onnx_light.node_tag = "shape"`` on the ``Shape`` node.
//   * ``onnx_light.node_tag = "weight"`` on the ``Reshape`` node (inherited
//     from its first input X which is seeded "weight" as a graph input).
//   * ``onnx_light.value_tags = {"S":"shape","X":"weight","Y":"weight"}`` on
//     the graph (X is a graph input → "weight"; Y inherits from X).
//
// The expected metadata is pre-embedded into the model so consumers can
// verify that ``WriteValueAndNodeTagsToMetadata`` reproduces it exactly.
// ---------------------------------------------------------------------------
void RegisterShapeTagCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_shape_reshape";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::Shape kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::Reshape kernel_2{ctx_2};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddNode(*graph, "Shape", {"X"}, {"S"});
    AddNode(*graph, "Reshape", {"X", "S"}, {"Y"});

    // Concrete input shape [2, 3] so the model is executable end-to-end.
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    AppendValueInfo(*graph->add_value_info(), "S", DataType::INT64, {DimSpec(2)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // Pre-embed the expected shape-tag metadata so tests can verify that
    // WriteValueAndNodeTagsToMetadata produces identical results.
    // X is a graph input so CollectGraphSeedTags tags it as "weight";
    // Y inherits "weight" from X through Reshape; S is tagged "shape" by Shape.
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "shape");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "shape");
    // Reshape (node[1]) inherits "weight" from its first input X.
    (*graph->mutable_node())[1].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    // S (value_info[0]) also receives onnx_light.value_tag = "shape".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // X (input[0]) receives onnx_light.value_tag = "weight".
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (output[0]) receives onnx_light.value_tag = "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor s = kernel_1(x, onnx_kernels::kernel::Shape::Attributes{});
    s.name = "S";
    Tensor y = kernel_2(x, s);
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Constant → Reshape`` — exercises the shape-tag-wins-over-weight-tag path.
// The ``Constant`` node produces the shape tensor ``S`` (INT64 [2]), initially
// tagged "weight". The ``Reshape`` node then consumes ``S`` as its *shape*
// input, pushing tag "shape" onto ``S``. Because "shape" has higher priority
// than "weight", ``S`` is tagged "shape".  The ``Constant`` node itself
// inherits the "shape" output tag on the second inference pass.
// Graph input ``X`` is seeded "weight"; Reshape inherits "weight" from X.
//
// ``WriteValueAndNodeTagsToMetadata`` must therefore emit:
//
//   * ``onnx_light.node_tag = "shape"`` on the ``Constant`` node.
//   * ``onnx_light.node_tag = "weight"`` on the ``Reshape`` node.
//   * ``onnx_light.value_tags = {"S":"shape","X":"weight","Y":"weight"}`` on the graph.
//
// The expected metadata is pre-embedded into the model so consumers can
// verify that ``WriteValueAndNodeTagsToMetadata`` reproduces it exactly.
// ---------------------------------------------------------------------------
void RegisterShapeTagAmbiguousCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_constant_reshape_ambiguous";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_3{opset};
    const onnx_kernels::kernel::Reshape kernel_3{ctx_3};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // Constant node: produces S (INT64 [2] = {2, 3}) from a tensor attribute.
    NodeProto &const_node = AddNode(*graph, "Constant", {}, {"S"});
    {
      const Tensor s_value = Tensor::FromInt64("", {2}, {2, 3});
      AttributeProto *attr = const_node.add_attribute();
      attr->set_name("value");
      attr->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = attr->add_t();
      t->set_data_type(static_cast<DataType>(s_value.data_type));
      for (int64_t d : s_value.shape) {
        t->add_dims(static_cast<uint64_t>(d));
      }
      t->set_raw_data(utils::ByteSpan(s_value.data));
    }
    AddNode(*graph, "Reshape", {"X", "S"}, {"Y"});

    // Concrete input shape [6] so the model is executable end-to-end.
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(6)});
    AppendValueInfo(*graph->add_value_info(), "S", DataType::INT64, {DimSpec(2)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // Pre-embed the expected shape-tag metadata so tests can verify that
    // WriteValueAndNodeTagsToMetadata produces identical results.
    // X is seeded as "weight" (graph input). S is produced by Constant ("weight")
    // but consumed as Reshape's shape input ("shape"): "shape" has higher priority
    // than "weight", so S is tagged "shape". The Constant node picks up "shape" on
    // the second inference pass. Reshape (node[1]) inherits "weight" from X.
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "shape");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "shape");
    // Reshape (node[1]) inherits "weight" from X.
    (*graph->mutable_node())[1].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    // S (value_info[0]) receives onnx_light.value_tag = "shape".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // X (input[0]) receives onnx_light.value_tag = "weight".
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (output[0]) receives onnx_light.value_tag = "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    // S is not a graph input (it is produced by the Constant node at runtime),
    // so only X appears in the DataSet inputs.
    const Tensor x = Tensor::FromFloat("X", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor s = Tensor::FromInt64("S", {2}, {2, 3});
    Tensor y = kernel_3(x, s);
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Constant → Mul → Concat → Reshape`` — exercises the shape-tag propagation
// through arithmetic and concatenation.  One ``Constant`` node produces the
// shape tensor ``S1`` (INT64 [1] = {3}), which is multiplied element-wise by
// 2 (a second ``Constant`` node ``two``) to produce ``S2 = {6}``.  ``S1``
// and ``S2`` are concatenated along axis 0 to form the full shape tensor
// ``S_full = {3, 6}``.  ``S_full`` is then used as the *shape* input of
// ``Reshape``, which reshapes the 18-element input ``X`` into a [3, 6] matrix.
// Graph input ``X`` is seeded "weight" and ``Y`` inherits "weight" from X.
//
// Expected value tags (backward-propagation through Concat, then Mul):
//   * ``S1``, ``S2``, ``S_full`` → ``"shape"``
//   * ``two`` → ``"weight"`` (only used as Mul's second input; Mul only
//     backward-propagates the "weight" tag, not "shape", so the "shape"
//     tag on S2 does not flow back to ``two``)
//   * ``X``, ``Y`` → ``"weight"`` (X seeded as graph input; Y inherited)
//
// Expected node tags:
//   * node[0] (Constant → S1) → ``"shape"``  (output overridden by "shape")
//   * node[1] (Constant → two) → ``"weight"`` (output stays "weight")
//   * node[2] (Mul) → ``"shape"``  (inherited from first input S1)
//   * node[3] (Concat) → ``"shape"`` (inherited from first input S1)
//   * node[4] (Reshape) → ``"weight"`` (inherited from first input X)
// ---------------------------------------------------------------------------
void RegisterShapeTagConstantMulConcatReshapeCases(std::vector<TestCase> &registry,
                                                   TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_constant_mul_concat_reshape";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_4{opset};
    const onnx_kernels::kernel::Mul kernel_4{ctx_4};
    const KernelContext ctx_5{opset};
    const onnx_kernels::kernel::Concat kernel_5{ctx_5};
    const KernelContext ctx_6{opset};
    const onnx_kernels::kernel::Reshape kernel_6{ctx_6};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // node[0]: Constant → S1 (INT64 [1] = {3})
    NodeProto &const_s1_node = AddNode(*graph, "Constant", {}, {"S1"});
    {
      const Tensor s1_value = Tensor::FromInt64("", {1}, {3});
      AttributeProto *attr = const_s1_node.add_attribute();
      attr->set_name("value");
      attr->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = attr->add_t();
      t->set_data_type(static_cast<DataType>(s1_value.data_type));
      for (int64_t d : s1_value.shape) {
        t->add_dims(static_cast<uint64_t>(d));
      }
      t->set_raw_data(utils::ByteSpan(s1_value.data));
    }

    // node[1]: Constant → two (INT64 [1] = {2})
    NodeProto &const_two_node = AddNode(*graph, "Constant", {}, {"two"});
    {
      const Tensor two_value = Tensor::FromInt64("", {1}, {2});
      AttributeProto *attr = const_two_node.add_attribute();
      attr->set_name("value");
      attr->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = attr->add_t();
      t->set_data_type(static_cast<DataType>(two_value.data_type));
      for (int64_t d : two_value.shape) {
        t->add_dims(static_cast<uint64_t>(d));
      }
      t->set_raw_data(utils::ByteSpan(two_value.data));
    }

    // node[2]: Mul(S1, two) → S2
    AddNode(*graph, "Mul", {"S1", "two"}, {"S2"});

    // node[3]: Concat([S1, S2], axis=0) → S_full
    NodeProto &concat_node = AddNode(*graph, "Concat", {"S1", "S2"}, {"S_full"});
    AddAxisAttribute(concat_node, 0);

    // node[4]: Reshape(X, S_full) → Y
    AddNode(*graph, "Reshape", {"X", "S_full"}, {"Y"});

    // Concrete input shape [18] so the model is executable end-to-end.
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(18)});
    AppendValueInfo(*graph->add_value_info(), "S1", DataType::INT64, {DimSpec(1)});
    AppendValueInfo(*graph->add_value_info(), "two", DataType::INT64, {DimSpec(1)});
    AppendValueInfo(*graph->add_value_info(), "S2", DataType::INT64, {DimSpec(1)});
    AppendValueInfo(*graph->add_value_info(), "S_full", DataType::INT64, {DimSpec(2)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(3), DimSpec(6)});

    // Pre-embed the expected shape-tag metadata so tests can verify that
    // WriteValueAndNodeTagsToMetadata produces identical results.
    // X is seeded as "weight" (graph input). S_full is the shape input of
    // Reshape → "shape".  Concat backward-propagates "shape" to S1 and S2.
    // S1's Constant picks up "shape" on the next pass.
    // Mul inherits "shape" from S1 (input 0) and S2 is already "shape".
    // "two" stays "weight" (Mul does not backward-propagate through input 1).
    // Reshape (node[4]) inherits "weight" from X (input 0) → Y = "weight".
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey,
                                             "shape"); // Constant → S1
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey,
                                             "shape"); // Constant → S1
    (*graph->mutable_node())[1].add_metadata(core::compute::kNodeTagMetadataKey,
                                             "weight"); // Constant → two
    (*graph->mutable_node())[1].add_metadata(core::compute::kValueTagMetadataKey,
                                             "weight"); // Constant → two
    (*graph->mutable_node())[2].add_metadata(core::compute::kNodeTagMetadataKey,
                                             "shape"); // Mul
    (*graph->mutable_node())[2].add_metadata(core::compute::kValueTagMetadataKey,
                                             "shape"); // Mul
    (*graph->mutable_node())[3].add_metadata(core::compute::kNodeTagMetadataKey,
                                             "shape"); // Concat
    (*graph->mutable_node())[3].add_metadata(core::compute::kValueTagMetadataKey,
                                             "shape"); // Concat
    // node[4] (Reshape) inherits "weight" from X.
    (*graph->mutable_node())[4].add_metadata(core::compute::kNodeTagMetadataKey,
                                             "weight"); // Reshape
    (*graph->mutable_node())[4].add_metadata(core::compute::kValueTagMetadataKey,
                                             "weight"); // Reshape

    // S1 (value_info[0]) → "shape".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // two (value_info[1]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(1)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // S2 (value_info[2]) → "shape".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(2)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // S_full (value_info[3]) → "shape".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(3)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // X (input[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (output[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    // S1, two, S2, S_full are not graph inputs (produced by Constant/Mul/Concat),
    // so only X appears in the DataSet inputs.
    const Tensor s1 = Tensor::FromInt64("S1", {1}, {3});
    const Tensor two = Tensor::FromInt64("two", {1}, {2});
    Tensor s2 = kernel_4(s1, two);
    s2.name = "S2";
    Tensor s_full = kernel_5({s1, s2}, 0);
    s_full.name = "S_full";

    const std::vector<float> x_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                       7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                       13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
    const Tensor x = Tensor::FromFloat("X", {18}, x_data);
    Tensor y = kernel_6(x, s_full);
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Shape`` — the model output is directly a shape tensor. The graph has a
// single ``Shape(X)`` node whose output ``Y`` is also the graph output.
// Because ``Y`` is produced by a ``Shape`` node,
// ``WriteValueAndNodeTagsToMetadata`` must emit:
//
//   * ``onnx_light.node_tag = "shape"`` on the ``Shape`` node.
//   * ``onnx_light.value_tags = {"X":"weight","Y":"shape"}`` on the graph.
//   * ``onnx_light.value_tag = "weight"`` on the graph input ``X``.
//   * ``onnx_light.value_tag = "shape"`` on the graph output ``Y``.
//
// This exercises the code path that writes the per-ValueInfo shape tag to a
// graph output (rather than only to intermediate ``value_info`` entries).
// ---------------------------------------------------------------------------
void RegisterShapeTagOutputAsShapeCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_output_is_shape";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_7{opset};
    const onnx_kernels::kernel::Shape kernel_7{ctx_7};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddNode(*graph, "Shape", {"X"}, {"Y"});

    // X is a graph input so CollectGraphSeedTags tags it as "weight".
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    // Y is the direct output of Shape — it carries the shape of X ([2, 3]).
    AppendValueInfo(*graph->add_output(), "Y", DataType::INT64, {DimSpec(2)});

    // Pre-embed the expected shape-tag metadata so tests can verify that
    // WriteValueAndNodeTagsToMetadata produces identical results.
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "shape");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "shape");
    // X (input[0]) receives onnx_light.value_tag = "weight".
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (output[0]) receives onnx_light.value_tag = "shape".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    // Y = Shape(X) = [2, 3].
    const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = kernel_7(x, onnx_kernels::kernel::Shape::Attributes{});
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Concat (weight wins)`` — verifies the "weight wins" rule for Concat:
// when at least one input carries the "weight" tag, the output is "weight"
// regardless of the tags on other inputs (which may be untagged).
//
// Model: PAST (FLOAT [1,2,4] graph input, seeded "weight") and
//        KH (FLOAT [1,3,4] initializer → "weight") are concatenated along
//        axis 1 to produce C (FLOAT [1,5,4]).  Because KH is "weight",
//        Concat's output C is "weight" (weight wins).  Backward propagation
//        then tags PAST as "weight" as well.
//
// Expected value tags: C="weight", KH="weight", PAST="weight"
// Expected node tags:  node[0] (Concat) → "weight"
// ---------------------------------------------------------------------------
void RegisterShapeTagConcatWeightWinsCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_concat_weight_wins";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_8{opset};
    const onnx_kernels::kernel::Concat kernel_8{ctx_8};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // node[0]: Concat(PAST, KH, axis=1) → C
    NodeProto &concat_node = AddNode(*graph, "Concat", {"PAST", "KH"}, {"C"});
    AddAxisAttribute(concat_node, 1);

    // PAST: FLOAT [1,2,4] (rank-3) → seeded "weight" as a graph input.
    AppendValueInfo(*graph->add_input(), "PAST", DataType::FLOAT,
                    {DimSpec(1), DimSpec(2), DimSpec(4)});
    // C: FLOAT [1,5,4] is the graph output.
    AppendValueInfo(*graph->add_output(), "C", DataType::FLOAT,
                    {DimSpec(1), DimSpec(5), DimSpec(4)});

    // KH: FLOAT [1,3,4] initializer → seeded "weight".
    std::vector<float> kh_data(1 * 3 * 4, 1.0f);
    AddInitializer<float>(*graph, "KH", {1, 3, 4}, kh_data);

    // Pre-embed the expected shape-tag metadata so tests can verify that
    // WriteValueAndNodeTagsToMetadata produces identical results.
    // PAST is "weight" (seeded as graph input). KH is "weight" (initializer seed).
    // Concat forward: weight wins → C="weight".
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    // PAST (input[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // C (output[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // KH (initializer[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_initializer(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    const Tensor past =
        Tensor::FromFloat("PAST", {1, 2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    const Tensor kh = Tensor::FromFloat("KH", {1, 3, 4}, kh_data);
    Tensor c = kernel_8({past, kh}, /*axis=*/1);
    c.name = "C";

    AppendDataSet(tc, {past}, {c});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Cast (backward tag propagation)`` — verifies that backward propagation
// flows through ``Cast``.
//
// Model: X (INT64 [4] graph input, seeded "weight") → Cast(to=FLOAT) → Y (FLOAT [4]).
//        W (FLOAT [4] initializer → "weight") is the first operand of
//        Add(W, Y) → Z.  Because W is "weight", Z inherits "weight"; Add
//        backward then tags Y as "weight"; Cast backward then tags X as
//        "weight".
//
// Expected value tags: W="weight", X="weight", Y="weight", Z="weight"
// Expected node tags:  node[0] (Cast) → "weight", node[1] (Add) → "weight"
// ---------------------------------------------------------------------------
void RegisterShapeTagCastBackwardCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_cast_backward";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_9{opset};
    const onnx_kernels::kernel::Cast kernel_9{ctx_9};
    const KernelContext ctx_10{opset};
    const onnx_kernels::kernel::Add kernel_10{ctx_10};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // node[0]: Cast(X, to=FLOAT) → Y
    NodeProto &cast_node = AddNode(*graph, "Cast", {"X"}, {"Y"});
    AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));

    // node[1]: Add(W, Y) → Z
    //   W is input[0] so inherited_tag = W = "weight" → Z = "weight".
    AddNode(*graph, "Add", {"W", "Y"}, {"Z"});

    // X: INT64 [4] graph input → seeded "weight".
    AppendValueInfo(*graph->add_input(), "X", DataType::INT64, {DimSpec(4)});
    // Z: FLOAT [4] graph output.
    AppendValueInfo(*graph->add_output(), "Z", DataType::FLOAT, {DimSpec(4)});

    // Y (intermediate)
    AppendValueInfo(*graph->add_value_info(), "Y", DataType::FLOAT, {DimSpec(4)});

    // W: FLOAT [4] initializer → seeded "weight".
    AddInitializer<float>(*graph, "W", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Pre-embed the expected shape-tag metadata.
    // W is "weight" (initializer seed).
    // Add forward: inherited from W (input[0]) → Z = "weight".
    // Add backward: Z="weight" → Y gets "weight".
    // Cast backward: Y="weight" → X gets "weight".
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    // X (input[0]) → "weight" (via Cast backward).
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Z (output[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (value_info[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // W (initializer[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_initializer(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    const Tensor x = Tensor::FromInt64("X", {4}, {1, 2, 3, 4});
    const Tensor w = Tensor::FromFloat("W", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = kernel_9(x, static_cast<int32_t>(DataType::FLOAT));
    y.name = "Y";
    Tensor z = kernel_10(w, y);
    z.name = "Z";

    AppendDataSet(tc, {x}, {z});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// ``Reshape (backward tag propagation)`` — verifies that backward propagation
// flows through ``Reshape`` to its first input.
//
// Model: X (FLOAT [6] graph input, seeded "weight") is reshaped by
//        S (INT64 [2] initializer = {2,3}) into Y (FLOAT [2,3]).  S is
//        automatically upgraded to "shape" by Reshape's input[1] role rule.
//        W (FLOAT [2,3] initializer → "weight") is the first operand of
//        Add(W, Y) → Z.  Because W is "weight", Z inherits "weight"; Add
//        backward then tags Y as "weight"; Reshape backward then tags X as
//        "weight".
//
// Expected value tags: S="shape", W="weight", X="weight", Y="weight", Z="weight"
// Expected node tags:  node[0] (Reshape) → "weight", node[1] (Add) → "weight"
// ---------------------------------------------------------------------------
void RegisterShapeTagReshapeBackwardCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_tag_reshape_backward";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_11{opset};
    const onnx_kernels::kernel::Reshape kernel_11{ctx_11};
    const KernelContext ctx_12{opset};
    const onnx_kernels::kernel::Add kernel_12{ctx_12};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::SHAPE_TAG);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // node[0]: Reshape(X, S) → Y
    AddNode(*graph, "Reshape", {"X", "S"}, {"Y"});

    // node[1]: Add(W, Y) → Z
    //   W is input[0] so inherited_tag = W = "weight" → Z = "weight".
    AddNode(*graph, "Add", {"W", "Y"}, {"Z"});

    // X: FLOAT [6] graph input → seeded "weight".
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(6)});
    // Z: FLOAT [2,3] graph output.
    AppendValueInfo(*graph->add_output(), "Z", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // Y (intermediate)
    AppendValueInfo(*graph->add_value_info(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // S: INT64 [2] initializer = {2,3} → seeded "weight", then upgraded to "shape".
    AddInitializer<int64_t>(*graph, "S", {2}, {2, 3});
    // W: FLOAT [2,3] initializer → seeded "weight".
    AddInitializer<float>(*graph, "W", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Pre-embed the expected shape-tag metadata.
    // S is "weight" (seed) then upgraded to "shape" by Reshape's input[1] rule.
    // W is "weight" (seed).
    // Add forward: inherited from W (input[0]) → Z = "weight".
    // Add backward: Z="weight" → Y gets "weight".
    // Reshape backward: Y="weight" → X gets "weight".
    (*graph->mutable_node())[0].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[0].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kNodeTagMetadataKey, "weight");
    (*graph->mutable_node())[1].add_metadata(core::compute::kValueTagMetadataKey, "weight");
    // X (input[0]) → "weight" (via Reshape backward).
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Z (output[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // Y (value_info[0]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // S (initializer[0]) → "shape".
    {
      StringStringEntryProto *entry = graph->mutable_initializer(0)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("shape");
    }
    // W (initializer[1]) → "weight".
    {
      StringStringEntryProto *entry = graph->mutable_initializer(1)->add_metadata_props();
      entry->set_key(core::compute::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Build the reference DataSet so the case is executable end-to-end.
    const Tensor x = Tensor::FromFloat("X", {6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor s = Tensor::FromInt64("S", {2}, {2, 3});
    const Tensor w = Tensor::FromFloat("W", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = kernel_11(x, s);
    y.name = "Y";
    Tensor z = kernel_12(w, y);
    z.name = "Z";

    AppendDataSet(tc, {x}, {z});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
