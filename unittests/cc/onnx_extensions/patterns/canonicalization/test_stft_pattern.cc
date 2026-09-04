// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/stft_pattern.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

template <typename T>
std::vector<T> DftWeights(int64_t bins, int64_t frame_length, bool imaginary,
                          bool corrupt = false) {
  const std::vector<double> window = {0.25, 0.5, 1.0, 0.5, 0.25};
  std::vector<T> values(static_cast<std::size_t>(bins * frame_length));
  for (int64_t k = 0; k < bins; ++k) {
    for (int64_t n = 0; n < frame_length; ++n) {
      const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) *
                           static_cast<double>(n) / static_cast<double>(frame_length);
      values[static_cast<std::size_t>(k * frame_length + n)] =
          static_cast<T>((imaginary ? std::sin(angle) : std::cos(angle)) * window[n]);
    }
  }
  if (corrupt) {
    values.back() += static_cast<T>(0.25);
  }
  return values;
}

template <typename T>
void BuildGraph(core::builder::GraphBuilder &builder, int64_t bins, bool corrupt = false,
                bool shared = false, bool invalid_attributes = false, bool invalid_topology = false,
                int64_t signal_components = 1) {
  constexpr int64_t frame_length = 5;
  builder.MakeInput("signal",
                    std::is_same_v<T, float> ? core::symbolic::TensorType::kFloat
                                             : core::symbolic::TensorType::kDouble,
                    Shape({2, 13, signal_components}));
  builder.MakeInitializer(MakeInitializer<T>("real_weights", {bins, 1, frame_length},
                                             DftWeights<T>(bins, frame_length, false, corrupt)));
  builder.MakeInitializer(MakeInitializer<T>("imag_weights", {bins, 1, frame_length},
                                             DftWeights<T>(bins, frame_length, true)));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {3}));

  NodeProto first = MakeNode("Transpose", {"signal"}, {"transposed"});
  AddAttribute<std::vector<int64_t>>(first, "perm",
                                     invalid_topology ? std::vector<int64_t>{0, 1, 2}
                                                      : std::vector<int64_t>{0, 2, 1});
  builder.MakeNode("Transpose", {"signal"}, {"transposed"}, "", "", first.attribute());

  NodeProto real = MakeNode("Conv", {"transposed", "real_weights"}, {"real"});
  AddAttribute<std::vector<int64_t>>(real, "strides", {2});
  if (invalid_attributes) {
    AddAttribute<int64_t>(real, "bogus", 1);
  }
  builder.MakeNode("Conv", {"transposed", "real_weights"}, {"real"}, "", "", real.attribute());

  NodeProto imag = MakeNode("Conv", {"transposed", "imag_weights"}, {"imag"});
  AddAttribute<std::vector<int64_t>>(imag, "strides", {2});
  builder.MakeNode("Conv", {"transposed", "imag_weights"}, {"imag"}, "", "", imag.attribute());

  builder.MakeNode("Unsqueeze", {"real", "axes"}, {"real_u"});
  builder.MakeNode("Unsqueeze", {"imag", "axes"}, {"imag_u"});
  NodeProto concat = MakeNode("Concat", {"real_u", "imag_u"}, {"complex"});
  AddAttribute<int64_t>(concat, "axis", 3);
  builder.MakeNode("Concat", {"real_u", "imag_u"}, {"complex"}, "", "", concat.attribute());
  NodeProto last = MakeNode("Transpose", {"complex"}, {"output"});
  AddAttribute<std::vector<int64_t>>(last, "perm", {0, 2, 1, 3});
  builder.MakeNode("Transpose", {"complex"}, {"output"}, "", "final", last.attribute());
  if (shared) {
    builder.MakeNode("Identity", {"real"}, {"shared"});
  }
  builder.MakeOutput("output");
}

TEST(STFTFusionPattern, FusesFloatAndDoubleHalfAndFullSpectrum) {
  for (int opset : {17, 18}) {
    for (bool use_double : {false, true}) {
      for (int64_t bins : {3, 5}) {
        core::builder::GraphBuilder builder("g", SchemaLookup());
        builder.SetOpsetVersion("", opset);
        if (use_double) {
          BuildGraph<double>(builder, bins);
        } else {
          BuildGraph<float>(builder, bins);
        }

        core::builder::GraphGraph graph(builder);
        onnx_patterns::STFTFusionPattern pattern;
        const NodeProto &candidate = builder.Nodes()[builder.Nodes().size() - 1];
        const auto match = pattern.Match(graph, candidate);
        ASSERT_EQ(match.pattern, &pattern);
        ASSERT_EQ(match.nodes.size(), 7u);
        const auto replacements = pattern.Apply(graph, match.nodes);

        ASSERT_EQ(replacements.size(), 1u);
        const NodeProto &stft = replacements[0];
        EXPECT_EQ(stft.op_type().value(), "STFT");
        EXPECT_EQ(stft.domain().value(), "");
        EXPECT_EQ(stft.input()[0].value(), "signal");
        EXPECT_EQ(stft.output()[0].value(), "output");
        EXPECT_EQ(GetAttributeOr<int64_t>(stft, "onesided", -1), bins == 3 ? 1 : 0);
        ASSERT_EQ(builder.Initializers().size(), 6u);
        EXPECT_EQ(builder.Initializers()[5].data_type(),
                  use_double ? TensorProto::DataType::DOUBLE : TensorProto::DataType::FLOAT);
        EXPECT_EQ(builder.Initializers()[5].dims(0), 5);
      }
    }
  }
}

TEST(STFTFusionPattern, RejectsInvalidTopologyAttributesWeightsAndSharing) {
  struct TestCase {
    bool corrupt;
    bool shared;
    bool invalid_attributes;
    bool invalid_topology;
    int64_t signal_components;
  };
  const std::vector<TestCase> cases = {
      {false, false, false, true, 1},  {false, false, true, false, 1},
      {true, false, false, false, 1},  {false, true, false, false, 1},
      {false, false, false, false, 2},
  };
  for (const TestCase &test : cases) {
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    BuildGraph<float>(builder, 3, test.corrupt, test.shared, test.invalid_attributes,
                      test.invalid_topology, test.signal_components);
    core::builder::GraphGraph graph(builder);
    onnx_patterns::STFTFusionPattern pattern;
    const NodeProto &candidate = builder.Nodes()[6];
    EXPECT_EQ(pattern.Match(graph, candidate).pattern, nullptr);
  }
}

TEST(STFTFusionPattern, RejectsUnsupportedOpsetAndType) {
  core::builder::GraphBuilder old_builder("old", SchemaLookup());
  old_builder.SetOpsetVersion("", 16);
  BuildGraph<float>(old_builder, 3);
  core::builder::GraphGraph old_graph(old_builder);
  onnx_patterns::STFTFusionPattern pattern;
  EXPECT_EQ(pattern.Match(old_graph, old_builder.Nodes()[old_builder.Nodes().size() - 1]).pattern,
            nullptr);

  core::builder::GraphBuilder type_builder("type", SchemaLookup());
  type_builder.SetOpsetVersion("", 18);
  type_builder.MakeInput("signal", core::symbolic::TensorType::kFloat16, Shape({2, 13, 1}));
  type_builder.MakeInitializer(
      MakeInitializer<uint16_t>("real_weights", {3, 1, 5}, std::vector<uint16_t>(15, 0)));
  type_builder.MakeInitializer(
      MakeInitializer<uint16_t>("imag_weights", {3, 1, 5}, std::vector<uint16_t>(15, 0)));
  type_builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {3}));
  NodeProto first = MakeNode("Transpose", {"signal"}, {"transposed"});
  AddAttribute<std::vector<int64_t>>(first, "perm", {0, 2, 1});
  type_builder.MakeNode("Transpose", {"signal"}, {"transposed"}, "", "", first.attribute());
  NodeProto real = MakeNode("Conv", {"transposed", "real_weights"}, {"real"});
  AddAttribute<std::vector<int64_t>>(real, "strides", {2});
  type_builder.MakeNode("Conv", {"transposed", "real_weights"}, {"real"}, "", "", real.attribute());
  NodeProto imag = MakeNode("Conv", {"transposed", "imag_weights"}, {"imag"});
  AddAttribute<std::vector<int64_t>>(imag, "strides", {2});
  type_builder.MakeNode("Conv", {"transposed", "imag_weights"}, {"imag"}, "", "", imag.attribute());
  type_builder.MakeNode("Unsqueeze", {"real", "axes"}, {"real_u"});
  type_builder.MakeNode("Unsqueeze", {"imag", "axes"}, {"imag_u"});
  NodeProto concat = MakeNode("Concat", {"real_u", "imag_u"}, {"complex"});
  AddAttribute<int64_t>(concat, "axis", 3);
  type_builder.MakeNode("Concat", {"real_u", "imag_u"}, {"complex"}, "", "", concat.attribute());
  NodeProto last = MakeNode("Transpose", {"complex"}, {"output"});
  AddAttribute<std::vector<int64_t>>(last, "perm", {0, 2, 1, 3});
  type_builder.MakeNode("Transpose", {"complex"}, {"output"}, "", "", last.attribute());
  type_builder.MakeOutput("output");

  core::builder::GraphGraph type_graph(type_builder);
  EXPECT_EQ(
      pattern.Match(type_graph, type_builder.Nodes()[type_builder.Nodes().size() - 1]).pattern,
      nullptr);
}

} // namespace
} // namespace Test
