// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/constant_info.h"
#include "onnx_extensions/backend_test/cases_for_shapes/constant/include_constant_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

// Adds ``onnx_light.constant = "1"`` to a value's (or initializer's)
// ``metadata_props``.
template <typename T> void MarkConstant(T &value) {
  StringStringEntryProto *entry = value.add_metadata_props();
  entry->set_key(core::compute::kConstantMetadataKey);
  entry->set_value("1");
}

} // namespace

// ---------------------------------------------------------------------------
// Two constant-information backend cases:
//
//   1. ``test_cc_constant_add_chain`` — ``Add(C, C) → D`` where ``C`` is an
//      initializer, so ``D`` (and the node) are constant. ``Add(X, D) → Y``
//      mixes the runtime input ``X`` with the constant ``D``, so ``Y`` and its
//      node are not constant.
//
//   2. ``test_cc_constant_node_source`` — a ``Constant`` node produces ``K``
//      (constant); ``Add(X, K) → Y`` is not constant because ``X`` is a runtime
//      input.
// ---------------------------------------------------------------------------
void RegisterConstantInfoCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(18);
  const onnx_kernels::kernel::KernelContext ctx{opset};

  // ---- case 1: Add(C, C) → D ; Add(X, D) → Y --------------------------------
  {
    const std::string name = "test_cc_constant_add_chain";

    TestCase tc(name, name, "model", "constant");
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddNode(*graph, "Add", {"C", "C"}, {"D"});
    AddNode(*graph, "Add", {"X", "D"}, {"Y"});

    AddInitializer<float>(*graph, "C", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    AppendValueInfo(*graph->add_value_info(), "D", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // Pre-embed the expected constant-information metadata.
    // node[0] Add(C, C): both inputs are the constant initializer C → constant.
    // node[1] Add(X, D): X is a runtime input → not constant.
    (*graph->mutable_node())[0].add_metadata(core::compute::kConstantMetadataKey, "1");
    MarkConstant(*graph->mutable_initializer(0)); // C
    MarkConstant(*graph->mutable_value_info(0));  // D

    const Tensor c = Tensor::FromFloat("C", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor x = Tensor::FromFloat("X", {2, 3}, {-1.0f, -2.0f, -3.0f, 0.5f, 1.5f, 2.5f});
    Tensor d = onnx_kernels::kernel::Add(ctx)(c, c);
    d.name = "D";
    Tensor y = onnx_kernels::kernel::Add(ctx)(x, d);
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    registry.emplace_back(std::move(tc));
  }

  // ---- case 2: Constant → K ; Add(X, K) → Y ---------------------------------
  {
    const std::string name = "test_cc_constant_node_source";

    TestCase tc(name, name, "model", "constant");
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    const Tensor k_value =
        Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    NodeProto &const_node = AddNode(*graph, "Constant", {}, {"K"});
    {
      AttributeProto *attr = const_node.add_attribute();
      attr->set_name("value");
      attr->set_type(AttributeProto::AttributeType::TENSOR);
      TensorProto *t = attr->add_t();
      t->set_data_type(static_cast<DataType>(k_value.data_type));
      for (int64_t d : k_value.shape) {
        t->add_dims(static_cast<uint64_t>(d));
      }
      t->set_raw_data(utils::ByteSpan(k_value.data));
    }
    AddNode(*graph, "Add", {"X", "K"}, {"Y"});

    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    AppendValueInfo(*graph->add_value_info(), "K", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

    // Pre-embed the expected constant-information metadata.
    // node[0] Constant: no inputs, deterministic → constant, K constant.
    // node[1] Add(X, K): X is a runtime input → not constant.
    (*graph->mutable_node())[0].add_metadata(core::compute::kConstantMetadataKey, "1");
    MarkConstant(*graph->mutable_value_info(0)); // K

    const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f});
    Tensor k = k_value;
    k.name = "K";
    Tensor y = onnx_kernels::kernel::Add(ctx)(x, k);
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
