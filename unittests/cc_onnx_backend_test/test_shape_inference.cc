// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

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
};

std::vector<ExpectedOutput> SnapshotAndStripOutputs(ModelProto &model) {
  std::vector<ExpectedOutput> snapshot;
  auto &outputs = model.mutable_graph()->ref_output();
  snapshot.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &out = outputs[i];
    ExpectedOutput exp;
    exp.name.assign(out.ref_name().data(), out.ref_name().size());
    if (out.has_type() && out.ref_type().has_tensor_type()) {
      auto *tt = out.mutable_type()->mutable_tensor_type();
      exp.elem_type = static_cast<int32_t>(tt->elem_type());
      exp.shape = DimsOf(*tt);
      // Strip the recorded shape so InferShapes has to recover it.
      // We keep elem_type so the ValueInfo remains well-formed.
      tt->clear_shape();
    }
    snapshot.emplace_back(std::move(exp));
  }
  return snapshot;
}

} // namespace

TEST(BackendTestCaseShapeInference, AllCollectedCasesInferOutputShapes) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    const auto expected = SnapshotAndStripOutputs(tc.model);

    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model)) << "case: " << tc.name;

    const auto &outputs = tc.model.ref_graph().ref_output();
    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &out = outputs[i];
      ASSERT_TRUE(out.has_type()) << "output " << expected[i].name << " missing type";
      ASSERT_TRUE(out.ref_type().has_tensor_type())
          << "output " << expected[i].name << " not a tensor";
      const auto &tt = out.ref_type().ref_tensor_type();
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
  }
}

// Second pass: replace every input dim_value by a unique symbolic dim_param
// (string), record the symbol -> value binding, then run shape inference and
// verify that whenever an output dim is resolved as a symbol, the value bound
// to that symbol matches the originally declared expected output dim value.
// This pins down that shape inference propagates symbolic shapes coherently.
TEST(BackendTestCaseShapeInference, AllCollectedCasesPropagateSymbolicDims) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    const auto expected = SnapshotAndStripOutputs(tc.model);

    // Replace every input dim_value with a unique dim_param "sym_<input>_<axis>"
    // and remember the value originally carried by that symbol.
    std::unordered_map<std::string, int64_t> symbol_values;
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
      const std::string vname(vi.ref_name().data(), vi.ref_name().size());
      auto &dims = tt->mutable_shape()->ref_dim();
      for (size_t d = 0; d < dims.size(); ++d) {
        auto &dim = dims[d];
        if (!dim.has_dim_value()) {
          continue;
        }
        const int64_t value = dim.ref_dim_value();
        const std::string symbol = "sym_" + vname + "_" + std::to_string(d);
        symbol_values[symbol] = value;
        dim.clear_dim_value();
        dim.set_dim_param(symbol);
      }
    }

    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model)) << "case: " << tc.name;

    const auto &outputs = tc.model.ref_graph().ref_output();
    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &out = outputs[i];
      ASSERT_TRUE(out.has_type()) << "output " << expected[i].name << " missing type";
      ASSERT_TRUE(out.ref_type().has_tensor_type())
          << "output " << expected[i].name << " not a tensor";
      const auto &tt = out.ref_type().ref_tensor_type();
      EXPECT_EQ(static_cast<int32_t>(tt.elem_type()), expected[i].elem_type)
          << "elem_type mismatch on output " << expected[i].name;
      if (!tt.has_shape()) {
        continue;
      }
      const auto &dims = tt.ref_shape().ref_dim();
      ASSERT_EQ(dims.size(), expected[i].shape.size())
          << "rank mismatch on output " << expected[i].name;
      for (size_t d = 0; d < dims.size(); ++d) {
        const auto &dim = dims[d];
        const int64_t expected_value = expected[i].shape[d];
        if (dim.has_dim_value()) {
          EXPECT_EQ(dim.ref_dim_value(), expected_value)
              << "dim[" << d << "] value mismatch on output " << expected[i].name;
        } else if (dim.has_dim_param()) {
          const std::string symbol(dim.ref_dim_param().data(), dim.ref_dim_param().size());
          auto it = symbol_values.find(symbol);
          if (it != symbol_values.end()) {
            EXPECT_EQ(it->second, expected_value)
                << "symbol '" << symbol << "' resolves to " << it->second << " but expected dim["
                << d << "] of output " << expected[i].name << " is " << expected_value;
          }
          // Unknown symbol (introduced by inference itself): tolerate.
        }
        // Otherwise dim is fully unknown (e.g. data-dependent ops): tolerate.
      }
    }
  }
}

} // namespace Test
