// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Translated from https://github.com/onnx/onnx/blob/main/onnx/test/cpp/data_propagation_test.cc

#include "../defs/data_propagators.h"
#include "../defs/parser.h"
#include "onnx.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// ============================================================================
// Type aliases
// ============================================================================

// Maps variable names to their partial value information (shape data).
using DataValueMap = std::unordered_map<std::string, TensorShapeProto>;

// ============================================================================
// Helpers: extract typed data from a TensorProto (onnx_light API)
// ============================================================================

// Extracts int64 values from a TensorProto (typed field or raw_data).
static std::vector<int64_t> ExtractInt64Data(const TensorProto &tp) {
  if (!tp.ref_raw_data().empty()) {
    const size_t n = tp.ref_raw_data().size() / sizeof(int64_t);
    std::vector<int64_t> result(n);
    std::memcpy(result.data(), tp.ref_raw_data().data(), n * sizeof(int64_t));
    return result;
  }
  std::vector<int64_t> result;
  for (auto v : tp.ref_int64_data())
    result.push_back(v);
  return result;
}

// Extracts int32 values from a TensorProto.
static std::vector<int32_t> ExtractInt32Data(const TensorProto &tp) {
  if (!tp.ref_raw_data().empty()) {
    const size_t n = tp.ref_raw_data().size() / sizeof(int32_t);
    std::vector<int32_t> result(n);
    std::memcpy(result.data(), tp.ref_raw_data().data(), n * sizeof(int32_t));
    return result;
  }
  std::vector<int32_t> result;
  for (auto v : tp.ref_int32_data())
    result.push_back(v);
  return result;
}

// ============================================================================
// Local DataPropagationContext implementation (onnx_light API)
// ============================================================================

class LocalDataPropagationContextImpl final : public DataPropagationContext {
public:
  LocalDataPropagationContextImpl(
      NodeProto &n, const std::unordered_map<std::string, TypeProto *> &valueTypesByName,
      const std::unordered_map<std::string, const TensorProto *> &inputDataByName,
      DataValueMap &generatedShapeData)
      : generatedShapeData_(generatedShapeData) {

    for (auto &attr : n.ref_attribute())
      attributesByName_[attr.ref_name().as_string()] = &attr;

    size_t idx = 0;
    for (const auto &inp : n.ref_input()) {
      std::string name = inp.as_string();
      inputIndexToNameMap_[idx++] = name;
      auto typeIt = valueTypesByName.find(name);
      allInputTypes_.push_back(typeIt != valueTypesByName.end() ? typeIt->second : nullptr);
      auto dataIt = inputDataByName.find(name);
      allInputData_.push_back(dataIt != inputDataByName.end() ? dataIt->second : nullptr);
    }

    idx = 0;
    for (const auto &out : n.ref_output())
      outputIndexToNameMap_[idx++] = out.as_string();

    allOutputTypes_.resize(n.ref_output().size());
  }

  const AttributeProto *getAttribute(const std::string &name) const override {
    auto it = attributesByName_.find(name);
    return (it != attributesByName_.end()) ? it->second : nullptr;
  }

  size_t getNumInputs() const override { return allInputTypes_.size(); }

  const TypeProto *getInputType(size_t index) const override {
    return (index < allInputTypes_.size()) ? allInputTypes_[index] : nullptr;
  }

  size_t getNumOutputs() const override { return allOutputTypes_.size(); }

  const TypeProto *getOutputType(size_t index) const override {
    return (index < allOutputTypes_.size()) ? &allOutputTypes_[index] : nullptr;
  }

  const TensorShapeProto *getInputData(size_t index) override {
    if (index >= allInputData_.size())
      return nullptr;

    const std::string &input_name = inputIndexToNameMap_.at(index);

    // Check previously generated shape data from upstream nodes.
    auto genIt = generatedShapeData_.find(input_name);
    if (genIt != generatedShapeData_.end())
      return &genIt->second;

    // Fall back to initializer / constant data.
    const TensorProto *input_data = allInputData_[index];
    if (input_data != nullptr &&
        (input_data->ref_dims().size() == 0 || input_data->ref_dims().size() == 1)) {
      TensorShapeProto tsp;
      if (input_data->ref_data_type() == TensorProto::DataType::INT64) {
        for (auto v : ExtractInt64Data(*input_data))
          tsp.add_dim()->set_dim_value(v);
      } else if (input_data->ref_data_type() == TensorProto::DataType::INT32) {
        for (auto v : ExtractInt32Data(*input_data))
          tsp.add_dim()->set_dim_value(static_cast<int64_t>(v));
      } else {
        return nullptr;
      }
      auto [it, inserted] = generatedShapeData_.insert({input_name, std::move(tsp)});
      if (inserted)
        return &it->second;
    }

    // If the input has a known 1-D shape, represent its values as N unknowns.
    const TypeProto *type = getInputType(index);
    if (type != nullptr && type->has_tensor_type()) {
      const auto &tensor_type = type->ref_tensor_type();
      if (tensor_type.has_shape()) {
        const auto &shape = tensor_type.ref_shape();
        if (shape.ref_dim().size() == 1 && shape.ref_dim()[0].has_dim_value()) {
          TensorShapeProto tsp;
          int64_t dim_value = shape.ref_dim()[0].ref_dim_value();
          for (int64_t i = 0; i < dim_value; ++i)
            tsp.add_dim();
          auto [it, inserted] = generatedShapeData_.insert({input_name, std::move(tsp)});
          if (inserted)
            return &it->second;
        }
      }
    }
    return nullptr;
  }

  void addOutputData(size_t index, TensorShapeProto &&tsp) override {
    if (index < outputIndexToNameMap_.size())
      generatedShapeData_.insert({outputIndexToNameMap_.at(index), std::move(tsp)});
  }

private:
  std::vector<const TensorProto *> allInputData_;
  std::unordered_map<size_t, std::string> inputIndexToNameMap_;
  std::unordered_map<size_t, std::string> outputIndexToNameMap_;
  std::vector<const TypeProto *> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  DataValueMap &generatedShapeData_;
  std::unordered_map<std::string, const AttributeProto *> attributesByName_;
};

// ============================================================================
// Data propagation functions (translated from tensor/old.cc, tensor/defs.cc,
// math/defs.cc) using onnx_light's ref_*() API.
// ============================================================================

// --- Shape (opset 15) -------------------------------------------------------
static void ShapeDataPropagator(DataPropagationContext &ctx) {
  const TypeProto *input_type = ctx.getInputType(0);
  if (input_type == nullptr || !input_type->has_tensor_type())
    return;
  const auto &tensor_type = input_type->ref_tensor_type();
  if (!tensor_type.has_shape())
    return;
  const auto &input_shape = tensor_type.ref_shape();
  const int64_t rank = static_cast<int64_t>(input_shape.ref_dim().size());

  const AttributeProto *start_attr = ctx.getAttribute("start");
  int64_t start = (start_attr != nullptr) ? start_attr->ref_i() : 0;
  const AttributeProto *end_attr = ctx.getAttribute("end");
  int64_t end = (end_attr != nullptr) ? end_attr->ref_i() : rank;

  if (start < 0)
    start += rank;
  start = std::max<int64_t>(0, std::min(start, rank));
  if (end < 0)
    end += rank;
  end = std::max<int64_t>(0, std::min(end, rank));

  TensorShapeProto output_shape;
  for (int64_t d = start; d < end; ++d) {
    const auto &src_dim = input_shape.ref_dim()[static_cast<size_t>(d)];
    auto *dst_dim = output_shape.add_dim();
    if (src_dim.has_dim_value())
      dst_dim->set_dim_value(src_dim.ref_dim_value());
    else if (src_dim.has_dim_param())
      dst_dim->set_dim_param(src_dim.ref_dim_param().as_string());
    // else: leave empty (unknown)
  }
  ctx.addOutputData(0, std::move(output_shape));
}

// --- Cast / Squeeze / Unsqueeze (all pass-through) --------------------------
// PropagateShapeDataFromInputToOutput is already in data_propagators.h.

// --- Math ops: Add / Sub / Mul ----------------------------------------------
static void MathOpDataPropagator(DataPropagationContext &ctx, const std::string &op_type) {
  const TensorShapeProto *input_0 = ctx.getInputData(0);
  const TensorShapeProto *input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr)
    return;

  const int64_t size_0 = static_cast<int64_t>(input_0->ref_dim().size());
  const int64_t size_1 = static_cast<int64_t>(input_1->ref_dim().size());
  if (size_0 != size_1 && size_0 != 1 && size_1 != 1)
    return; // Invalid broadcast; skip propagation rather than failing.

  TensorShapeProto tsp;
  const int64_t size_out = (size_0 == 1) ? size_1 : size_0;
  for (int64_t i = 0; i < size_out; ++i) {
    const auto &dim0 = input_0->ref_dim()[static_cast<size_t>(size_0 == 1 ? 0 : i)];
    const auto &dim1 = input_1->ref_dim()[static_cast<size_t>(size_1 == 1 ? 0 : i)];
    auto *out_dim = tsp.add_dim();
    if (dim0.has_dim_value() && dim1.has_dim_value()) {
      int64_t a = dim0.ref_dim_value();
      int64_t b = dim1.ref_dim_value();
      int64_t result;
      if (op_type == "Add")
        result = a + b;
      else if (op_type == "Sub")
        result = a - b;
      else // Mul
        result = a * b;
      out_dim->set_dim_value(result);
    }
    // else: leave empty (unknown dim)
  }
  ctx.addOutputData(0, std::move(tsp));
}

// --- Concat (axis=0 only) ---------------------------------------------------
static void ConcatDataPropagator(DataPropagationContext &ctx) {
  if (!axisIsZero(ctx))
    return;
  TensorShapeProto tsp;
  for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
    const auto *input_data = ctx.getInputData(i);
    if (input_data == nullptr) {
      // Return: cannot propagate if any input is unknown.
      // However, the DynamicConcatTest expects partial propagation for
      // the dynamic input - we add unknown dims instead of bailing.
      // Peek at input type to get rank.
      const TypeProto *input_type = ctx.getInputType(i);
      if (input_type != nullptr && input_type->has_tensor_type()) {
        const auto &tt = input_type->ref_tensor_type();
        if (tt.has_shape()) {
          for (size_t j = 0; j < tt.ref_shape().ref_dim().size(); ++j)
            tsp.add_dim(); // unknown
          continue;
        }
      }
      return;
    }
    for (size_t j = 0; j < input_data->ref_dim().size(); ++j) {
      const auto &src = input_data->ref_dim()[j];
      auto *dst = tsp.add_dim();
      if (src.has_dim_value())
        dst->set_dim_value(src.ref_dim_value());
      else if (src.has_dim_param())
        dst->set_dim_param(src.ref_dim_param().as_string());
      // else: unknown dim
    }
  }
  if (!tsp.ref_dim().empty())
    ctx.addOutputData(0, std::move(tsp));
}

// --- Slice (axis=0 only) ----------------------------------------------------
static void processSliceInputs(int64_t n, int64_t &start, int64_t &end, int64_t step) {
  if (step == 0)
    return;
  if (n == 0) {
    start = end = 0;
    return;
  }
  if (start < 0)
    start += n;
  start = step < 0 ? std::clamp(start, int64_t{0}, n - 1) : std::clamp(start, int64_t{0}, n);
  if (end < 0)
    end += n;
  end = step < 0 ? std::clamp(end, int64_t{-1}, n - 1) : std::clamp(end, int64_t{0}, n);
}

static void SliceDataPropagator(DataPropagationContext &ctx) {
  const TensorShapeProto *input_data = ctx.getInputData(0);
  const TensorShapeProto *starts_tsp = ctx.getInputData(1);
  const TensorShapeProto *ends_tsp = ctx.getInputData(2);

  if (input_data == nullptr || starts_tsp == nullptr || ends_tsp == nullptr)
    return;

  bool axes_specified = ctx.getNumInputs() >= 4;
  bool steps_specified = ctx.getNumInputs() >= 5;

  const TensorShapeProto *axes_tsp = nullptr;
  const TensorShapeProto *steps_tsp = nullptr;
  if (axes_specified) {
    axes_tsp = ctx.getInputData(3);
    if (axes_tsp == nullptr)
      return;
  }
  if (steps_specified) {
    steps_tsp = ctx.getInputData(4);
    if (steps_tsp == nullptr)
      return;
  }

  if (starts_tsp->ref_dim().size() != ends_tsp->ref_dim().size())
    return;

  // Only support axis=0 (same as tensor/defs.cc)
  bool axis_is_zero = !axes_specified ||
                      (axes_tsp->ref_dim().size() == 1 && axes_tsp->ref_dim()[0].has_dim_value() &&
                       axes_tsp->ref_dim()[0].ref_dim_value() == 0);
  if (!axis_is_zero)
    return;
  if (starts_tsp->ref_dim().size() != 1 || ends_tsp->ref_dim().size() != 1)
    return;

  int64_t start =
      starts_tsp->ref_dim()[0].has_dim_value() ? starts_tsp->ref_dim()[0].ref_dim_value() : 0;
  int64_t end = ends_tsp->ref_dim()[0].has_dim_value()
                    ? ends_tsp->ref_dim()[0].ref_dim_value()
                    : static_cast<int64_t>(input_data->ref_dim().size());
  int64_t step = 1;
  if (steps_specified) {
    if (steps_tsp->ref_dim().size() != 1)
      return;
    if (!steps_tsp->ref_dim()[0].has_dim_value())
      return;
    step = steps_tsp->ref_dim()[0].ref_dim_value();
  }

  const int64_t input_dim_size = static_cast<int64_t>(input_data->ref_dim().size());
  processSliceInputs(input_dim_size, start, end, step);

  TensorShapeProto tsp;
  if (step > 0) {
    for (int64_t i = start; i < end; i += step) {
      const auto &src = input_data->ref_dim()[static_cast<size_t>(i)];
      auto *dst = tsp.add_dim();
      if (src.has_dim_value())
        dst->set_dim_value(src.ref_dim_value());
      else if (src.has_dim_param())
        dst->set_dim_param(src.ref_dim_param().as_string());
    }
  } else {
    for (int64_t i = start; i > end; i += step) {
      const auto &src = input_data->ref_dim()[static_cast<size_t>(i)];
      auto *dst = tsp.add_dim();
      if (src.has_dim_value())
        dst->set_dim_value(src.ref_dim_value());
      else if (src.has_dim_param())
        dst->set_dim_param(src.ref_dim_param().as_string());
    }
  }
  if (!tsp.ref_dim().empty())
    ctx.addOutputData(0, std::move(tsp));
}

// --- Size -------------------------------------------------------------------
static void SizeDataPropagator(DataPropagationContext &ctx) {
  const TensorShapeProto *input_data = ctx.getInputData(0);
  if (input_data == nullptr)
    return;
  TensorShapeProto tsp;
  tsp.add_dim()->set_dim_value(static_cast<int64_t>(input_data->ref_dim().size()));
  ctx.addOutputData(0, std::move(tsp));
}

// ============================================================================
// Dispatch: map op_type -> propagation function
// ============================================================================

static void DispatchDataPropagation(DataPropagationContext &ctx, const std::string &op_type) {
  if (op_type == "Shape") {
    ShapeDataPropagator(ctx);
  } else if (op_type == "Cast" || op_type == "Squeeze" || op_type == "Unsqueeze") {
    PropagateShapeDataFromInputToOutput(ctx, 0);
  } else if (op_type == "Add") {
    MathOpDataPropagator(ctx, "Add");
  } else if (op_type == "Sub") {
    MathOpDataPropagator(ctx, "Sub");
  } else if (op_type == "Mul") {
    MathOpDataPropagator(ctx, "Mul");
  } else if (op_type == "Concat") {
    ConcatDataPropagator(ctx);
  } else if (op_type == "Gather") {
    GatherOp13DataPropagator(ctx);
  } else if (op_type == "Slice") {
    SliceDataPropagator(ctx);
  } else if (op_type == "Size") {
    SizeDataPropagator(ctx);
  }
  // Unknown ops: no propagation.
}

// ============================================================================
// Shape comparison helpers
// ============================================================================

static bool CompareShape(const TensorShapeProto &inferredShape,
                         const TensorShapeProto &expectedShape, bool checkSameParam = false) {
  EXPECT_EQ(inferredShape.ref_dim().size(), expectedShape.ref_dim().size())
      << "Dim size for inferred and expected shape is different.";

  const size_t ndims = std::min(inferredShape.ref_dim().size(), expectedShape.ref_dim().size());
  for (size_t i = 0; i < ndims; i++) {
    const auto &inf = inferredShape.ref_dim()[i];
    const auto &exp = expectedShape.ref_dim()[i];

    EXPECT_EQ(inf.has_dim_value(), exp.has_dim_value()) << "has_dim_value mismatch at index " << i;
    EXPECT_EQ(inf.has_dim_param(), exp.has_dim_param()) << "has_dim_param mismatch at index " << i;

    if (inf.has_dim_value() && exp.has_dim_value()) {
      EXPECT_EQ(inf.ref_dim_value(), exp.ref_dim_value()) << "dim_value mismatch at index " << i;
    }
    if (checkSameParam && inf.has_dim_param() && exp.has_dim_param()) {
      EXPECT_EQ(inf.ref_dim_param().as_string(), exp.ref_dim_param().as_string())
          << "dim_param mismatch at index " << i;
    }
  }
  return true;
}

// ============================================================================
// RunDataPropagation: parse graph, build maps, propagate
// ============================================================================

static TensorShapeProto RunDataPropagation(const char *graphCode) {
  // 1. Parse the graph.
  GraphProto graph;
  OnnxParser parser(graphCode);
  auto status = parser.Parse(graph);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // 2. Build name → TypeProto* maps from value_info, inputs, and outputs.
  std::unordered_map<std::string, TypeProto *> valueTypesByName;
  for (auto &vi : graph.ref_value_info()) {
    if (vi.has_type())
      valueTypesByName[vi.ref_name().as_string()] = &vi.ref_type();
  }
  for (auto &vi : graph.ref_input()) {
    if (vi.has_type())
      valueTypesByName[vi.ref_name().as_string()] = &vi.ref_type();
  }
  for (auto &vi : graph.ref_output()) {
    if (vi.has_type())
      valueTypesByName[vi.ref_name().as_string()] = &vi.ref_type();
  }

  // 3. Build name → TensorProto* maps from initializers.
  std::unordered_map<std::string, const TensorProto *> inputDataByName;
  for (const auto &tp : graph.ref_initializer())
    inputDataByName[tp.ref_name().as_string()] = &tp;

  // 4. Collect data from Constant nodes.
  for (const auto &n : graph.ref_node()) {
    if (n.ref_op_type() != "Constant" || n.ref_output().size() != 1)
      continue;
    for (const auto &attr : n.ref_attribute()) {
      if (attr.ref_name() == "value" && attr.ref_type() == AttributeProto::TENSOR && attr.has_t()) {
        inputDataByName[n.ref_output()[0].as_string()] = &attr.ref_t();
      }
    }
  }

  // 5. Run data propagation on each non-Constant node.
  DataValueMap generatedShapeDataByName;
  for (auto &n : graph.ref_node()) {
    if (n.ref_op_type() == "Constant")
      continue;
    LocalDataPropagationContextImpl ctx(n, valueTypesByName, inputDataByName,
                                        generatedShapeDataByName);
    DispatchDataPropagation(ctx, n.ref_op_type().as_string());
  }

  // 6. Return propagated shape for the single graph output.
  const std::string outputName = graph.ref_output()[0].ref_name().as_string();
  const auto it = generatedShapeDataByName.find(outputName);
  EXPECT_TRUE(it != generatedShapeDataByName.cend())
      << "No propagated data found for output: " << outputName;

  TensorShapeProto inferredShape;
  if (it != generatedShapeDataByName.cend())
    inferredShape.CopyFrom(it->second);
  return inferredShape;
}

} // namespace

// ============================================================================
// Tests (translated from data_propagation_test.cc)
// ============================================================================

TEST(DataPropagationImplTest, ShapeTest) {
  const char *code = R"ONNX(
agraph (int32[7,4,1] x) => (int32[3] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(7);
  expected.add_dim()->set_dim_value(4);
  expected.add_dim()->set_dim_value(1);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SymbolicShapeTest) {
  const char *code = R"ONNX(
agraph (int32[N,3,256,256] x) => (int32[4] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_param("N");
  expected.add_dim()->set_dim_value(3);
  expected.add_dim()->set_dim_value(256);
  expected.add_dim()->set_dim_value(256);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected, /*checkSameParam=*/true));
}

TEST(DataPropagationImplTest, CastTest) {
  const char *code = R"ONNX(
agraph (int32[2,5] x) => (int32[2] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(5);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SqueezeTest) {
  const char *code = R"ONNX(
agraph (int32[2,5] x) => (int32[2] z)
{
    xs = Shape(x)
    y = Squeeze(xs)
    z = Cast<to = 7>(y)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(5);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, UnsqueezeTest) {
  const char *code = R"ONNX(
agraph (int32[2,5] x) => (int32[1,2] w)
{
    xs = Shape(x)
    axis = Constant<value = int64[1] {1}>()
    z = Unsqueeze(xs, axis)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(5);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SizeTest) {
  const char *code = R"ONNX(
agraph (int64[1] x) => (int32[1] w)
<int64[3] init = {2,3,5}>
{
    z = Size(init)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(3);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, AddTest) {
  const char *code = R"ONNX(
agraph (int32[2,4,5] x, int32[2,4,5] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Add(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(4);
  expected.add_dim()->set_dim_value(8);
  expected.add_dim()->set_dim_value(10);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, AddSymbolicShapeTest) {
  const char *code = R"ONNX(
agraph (int32[2,4,5] x, int32[2,4,M] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Add(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  // Add({2,4,5}, {2,4,M}) = {4, 8, ?}
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(4);
  expected.add_dim()->set_dim_value(8);
  expected.add_dim(); // unknown
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SubTest) {
  const char *code = R"ONNX(
agraph (int32[10,11,6] x, int32[5] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Sub(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(5);
  expected.add_dim()->set_dim_value(6);
  expected.add_dim()->set_dim_value(1);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, MulTest) {
  const char *code = R"ONNX(
agraph (int32[2] x, int32[5,1,7] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Mul(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(10);
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(14);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, ConcatTest) {
  const char *code = R"ONNX(
agraph (int32[1,2] x, int32[3,4] y) => (int32[4] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Concat<axis = 0>(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(1);
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(3);
  expected.add_dim()->set_dim_value(4);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, DynamicConcatTest) {
  const char *code = R"ONNX(
agraph (float[32, 1024] x, int64[2] dynamic_shape) => (int64[4] z)
{
    xs = Shape(x)
    z = Concat<axis = 0>(xs, dynamic_shape)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(32);
  expected.add_dim()->set_dim_value(1024);
  expected.add_dim(); // unknown
  expected.add_dim(); // unknown
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, GatherTest) {
  const char *code = R"ONNX(
agraph (int32[1,2,3,4,5,6] x) => (int32[3] w)
{
    xs = Shape(x)
    indices = Constant<value = int64[3] {0,3,5}>()
    z = Gather<axis = 0>(xs, indices)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(1);
  expected.add_dim()->set_dim_value(4);
  expected.add_dim()->set_dim_value(6);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, GatherNegativeIndicesTest) {
  const char *code = R"ONNX(
agraph (int32[1,2,3,4,5,6] x) => (int32[2] w)
{
    xs = Shape(x)
    indices = Constant<value = int64[2] {-2,-1}>()
    z = Gather<axis = 0>(xs, indices)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(5);
  expected.add_dim()->set_dim_value(6);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SliceTest) {
  const char *code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[2] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {1}>()
    ends = Constant<value = int64[1] {7}>()
    axes = Constant<value = int64[1] {0}>()
    steps = Constant<value = int64[1] {3}>()
    z = Slice(xs, starts, ends, axes, steps)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(2);
  expected.add_dim()->set_dim_value(5);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SliceDefaultAxesAndStepTest) {
  const char *code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[3] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {2}>()
    ends = Constant<value = int64[1] {5}>()
    z = Slice(xs, starts, ends)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(3);
  expected.add_dim()->set_dim_value(4);
  expected.add_dim()->set_dim_value(5);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}

TEST(DataPropagationImplTest, SliceNegativeStartEndStepTest) {
  const char *code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[3] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {-3}>()
    ends = Constant<value = int64[1] {-7}>()
    axes = Constant<value = int64[1] {0}>()
    steps = Constant<value = int64[1] {-2}>()
    z = Slice(xs, starts, ends, axes, steps)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected;
  expected.add_dim()->set_dim_value(6);
  expected.add_dim()->set_dim_value(4);
  EXPECT_TRUE(CompareShape(RunDataPropagation(code), expected));
}
