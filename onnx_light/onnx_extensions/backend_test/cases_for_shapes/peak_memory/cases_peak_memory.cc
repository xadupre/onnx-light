// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_extensions/backend_test/cases_for_shapes/peak_memory/include_peak_memory_cases.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;
constexpr int64_t kStaticAttentionPeakMemoryBytes = 2 * 4 * 8 * 32 * 4;

} // namespace

void RegisterPeakMemoryCases(std::vector<TestCase> &registry, TestMode mode) {
  (void)mode;
  const OpsetId opset = DefaultOpset(23);

  {
    const std::string name = "test_cc_peak_memory_attention_static";
    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::PEAK_MEMORY);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    NodeProto &node = AddNode(*graph, "Attention", {"Q", "K", "V"}, {"Y"});

    AppendValueInfo(*graph->add_input(), "Q", DataType::FLOAT,
                    {DimSpec(2), DimSpec(4), DimSpec(8), DimSpec(16)});
    AppendValueInfo(*graph->add_input(), "K", DataType::FLOAT,
                    {DimSpec(2), DimSpec(4), DimSpec(32), DimSpec(16)});
    AppendValueInfo(*graph->add_input(), "V", DataType::FLOAT,
                    {DimSpec(2), DimSpec(4), DimSpec(32), DimSpec(16)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT,
                    {DimSpec(2), DimSpec(4), DimSpec(8), DimSpec(16)});

    node.add_metadata(core::compute::kNodePeakMemoryMetadataKey,
                      std::to_string(kStaticAttentionPeakMemoryBytes));
    registry.emplace_back(std::move(tc));
  }

  {
    const std::string name = "test_cc_peak_memory_attention_symbolic";
    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::PEAK_MEMORY);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddNode(*graph, "Attention", {"Q", "K", "V"}, {"Y"});

    AppendValueInfo(*graph->add_input(), "Q", DataType::FLOAT,
                    {DimSpec("batch"), DimSpec(4), DimSpec(8), DimSpec(16)});
    AppendValueInfo(*graph->add_input(), "K", DataType::FLOAT,
                    {DimSpec("batch"), DimSpec(4), DimSpec(32), DimSpec(16)});
    AppendValueInfo(*graph->add_input(), "V", DataType::FLOAT,
                    {DimSpec("batch"), DimSpec(4), DimSpec(32), DimSpec(16)});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT,
                    {DimSpec("batch"), DimSpec(4), DimSpec(8), DimSpec(16)});

    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
