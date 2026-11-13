// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/dequantize_packed.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/quantization/shape_quantization.h"
#include "onnx_proto/onnx_helper.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;

namespace {

// Derived from the upstream 64-block corpus, verified against its independent
// GGML-C SHA256 oracles at 572a5d9ebb2194a1cb8775f1cc867efccf5661db.
// FNV-1a hashes canonical little-endian FP32 bits (including signed zeros).
constexpr uint64_t kOracle[] = {
    0x347df38fe6b197c8ULL, 0x402503a4032dd771ULL, 0xe63f7ce1222a5832ULL, 0xc55ad0ec27813b7eULL,
    0xc66b383211b9acd9ULL, 0x53a6fda8ae6e3cc0ULL, 0x7a5dff012509ed0cULL, 0x3358ccb8d9cbcf70ULL,
    0x302bdf61e35eda82ULL, 0x5b48080e8e112110ULL, 0xf797c553c90eaadfULL, 0xc18061311fd7cff0ULL,
    0x18c414c17f6a6e45ULL, 0xe41ddc4d9f0e7907ULL, 0x2146ae7f760d7dffULL, 0x6a03a2146d7c1174ULL,
    0x52bad54854a68148ULL, 0x28da03729d0f4b0fULL, 0x7e3cff214f8e6461ULL, 0x7ed7e2a104676269ULL,
    0xe3762fec5e716cc4ULL, 0x4ae879376d08d00aULL, 0xe8d21cc3d67cbbcaULL, 0x4121096600239e0fULL,
    0xcababe2463b6289aULL, 0x731d5623f5ed41a5ULL, 0xaf234791af5b326cULL};

void Put16(uint8_t *p, uint16_t bits) {
  p[0] = bits & 255;
  p[1] = bits >> 8;
}

Tensor Corpus(const PackedQuantizationProfile &profile) {
  const int type = profile.ggml_type;
  const size_t size = profile.block_bytes;
  std::vector<uint8_t> bytes(64 * size);
  constexpr double scales[] = {0.3125, -0.625, 0, 1.125, 0.1, -3.25, 0x1p-24, 65504};
  for (size_t i = 0; i < 64; ++i) {
    uint8_t *p = bytes.data() + i * size;
    for (size_t j = 0; j < size; ++j)
      p[j] = (j * 73 + i * 29 + (j / 7) * 17 + (i / 3) * 11) & 255;
    if (type != 15 && type != 29 && type != 39 && type != 40) {
      const size_t offset = type == 10   ? 80
                            : type == 11 ? 108
                            : type == 14 ? 208
                            : type == 34 ? 52
                            : type == 35 ? 64
                                         : 0;
      Put16(p + offset, FloatToFloat16Bits(static_cast<float>(scales[i % 8])));
      if (type == 3 || type == 7 || type == 10 || type == 12 || type == 13)
        Put16(p + offset + 2, FloatToFloat16Bits(static_cast<float>(scales[(i + 1) % 8])));
    }
    if (type == 15) {
      const uint32_t bits = std::bit_cast<uint32_t>(static_cast<float>(scales[i % 8] / 3));
      for (size_t j = 0; j < 4; ++j)
        p[j] = (bits >> (8 * j)) & 255;
    } else if (type == 29) {
      const uint16_t bits = FloatToFloat16Bits(static_cast<float>(scales[i % 8]));
      for (size_t j = 0; j < 4; ++j) {
        const uint16_t word = p[48 + 2 * j] | (p[49 + 2 * j] << 8);
        Put16(p + 48 + 2 * j, (word & 0x0fff) | (((bits >> (4 * j)) & 15) << 12));
      }
    } else if (type == 39)
      p[0] = (i * 17) & 255;
  }
  const int64_t count = bytes.size();
  return Tensor("", TensorProto::UINT8, {count}, std::move(bytes));
}

NodeProto PackedNode(const std::string &format, const std::vector<int64_t> &shape,
                     int64_t dtype = TensorProto::FLOAT) {
  NodeProto node;
  node.set_op_type("DequantizePacked");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::string>(node, "format", format);
  AddAttribute<std::vector<int64_t>>(node, "shape", shape);
  AddAttribute<int64_t>(node, "output_dtype", dtype);
  return node;
}

} // namespace

TEST(DequantizePacked, AllProfilesMatchIndependentGgmlOraclesAndOutputConversions) {
  size_t profile_index = 0;
  for (const auto &profile : PackedQuantizationProfiles()) {
    SCOPED_TRACE(profile.format);
    const Tensor input = Corpus(profile);
    const Shape shape{2, static_cast<int64_t>(32 * profile.block_size)};
    const auto output = DequantizePackedTensor(input, profile.format, shape, TensorProto::FLOAT);
    uint64_t hash = 14695981039346656037ULL;
    for (int64_t i = 0; i < output.element_count(); ++i) {
      const uint32_t bits = std::bit_cast<uint32_t>(output.AsFloat()[i]);
      for (size_t j = 0; j < 4; ++j)
        hash = (hash ^ ((bits >> (8 * j)) & 255)) * 1099511628211ULL;
    }
    EXPECT_EQ(hash, kOracle[profile_index++]);
    std::vector<uint8_t> unaligned(input.size_bytes() + 1);
    std::memcpy(unaligned.data() + 1, input.bytes(), input.size_bytes());
    const Tensor borrowed = Tensor::Borrow("", TensorProto::UINT8, input.shape,
                                           unaligned.data() + 1, input.size_bytes());
    const auto unaligned_output =
        DequantizePackedTensor(borrowed, profile.format, shape, TensorProto::FLOAT);
    EXPECT_EQ(std::memcmp(output.bytes(), unaligned_output.bytes(), output.size_bytes()), 0);
    for (const int dtype : {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::DOUBLE}) {
      const auto converted = DequantizePackedTensor(input, profile.format, shape, dtype);
      EXPECT_EQ(converted.shape, shape);
      for (int64_t i = 0; i < output.element_count(); ++i) {
        const float value = output.AsFloat()[i];
        if (dtype == TensorProto::DOUBLE)
          EXPECT_EQ(std::bit_cast<uint64_t>(converted.As<double>()[i]),
                    std::bit_cast<uint64_t>(static_cast<double>(value)));
        else {
          uint16_t bits;
          std::memcpy(&bits, converted.bytes() + 2 * i, 2);
          EXPECT_EQ(bits, dtype == TensorProto::FLOAT16 ? FloatToFloat16Bits(value)
                                                        : FloatToBfloat16Bits(value));
        }
      }
    }
  }
  EXPECT_EQ(profile_index, size_t{27});
}

TEST(DequantizePacked, RejectsInvalidLayoutsBeforeEmptyFastPaths) {
  const Tensor empty("", TensorProto::UINT8, {0}, {});
  const int64_t max = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(DequantizePackedTensor(empty, "gguf:Q4_0", {0, 32}, TensorProto::FLOAT),
               std::invalid_argument);
  for (const Shape &shape : {Shape{}, Shape{-1, 32}, Shape{0, 31}, Shape{max, 32},
                             Shape{max / 32, 32}, Shape{0, -1, 32}, Shape{-1, 0, 32}})
    EXPECT_THROW(DequantizePackedTensor(empty, "gguf:q4_0", shape, TensorProto::FLOAT),
                 std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(empty, "gguf:q4_0", {0, 32}, TensorProto::INT32),
               std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(empty, "gguf:q4_0", {0, 32}, int64_t{1} << 32),
               std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(Tensor("", TensorProto::INT8, {0}, {}), "gguf:q4_0", {0, 32},
                                      TensorProto::FLOAT),
               std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(Tensor("", TensorProto::UINT8, {0, 1}, {}), "gguf:q4_0",
                                      {0, 32}, TensorProto::FLOAT),
               std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(Tensor("", TensorProto::UINT8, {0}, {1}), "gguf:q4_0",
                                      {0, 32}, TensorProto::FLOAT),
               std::invalid_argument);
  EXPECT_THROW(DequantizePackedTensor(Tensor("", TensorProto::UINT8, {-1}, {}), "gguf:q4_0",
                                      {0, 32}, TensorProto::FLOAT),
               std::invalid_argument);
  for (const Shape &shape : {Shape{0, int64_t{1} << 62, 32}, Shape{max, 2, 0}, Shape{max, 0, 32}}) {
    const auto output = DequantizePackedTensor(empty, "gguf:q4_0", shape, TensorProto::FLOAT);
    EXPECT_EQ(output.shape, shape);
    EXPECT_EQ(output.element_count(), 0);
    EXPECT_EQ(output.shape.product(0, output.shape.size(), "empty"), 0);
    EXPECT_EQ(output.size_bytes(), size_t{0});
  }
  for (const auto &profile : PackedQuantizationProfiles()) {
    SCOPED_TRACE(profile.format);
    const Shape shape{0, static_cast<int64_t>(profile.block_size)};
    for (const int dtype :
         {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::DOUBLE}) {
      const auto output = DequantizePackedTensor(empty, profile.format, shape, dtype);
      EXPECT_EQ(output.shape, shape);
      EXPECT_EQ(output.data_type, dtype);
      EXPECT_EQ(output.size_bytes(), size_t{0});
    }
    auto input = Corpus(profile);
    input.data.resize(input.data.size() - 1);
    EXPECT_THROW(DequantizePackedTensor(input, profile.format,
                                        {static_cast<int64_t>(64 * profile.block_size)},
                                        TensorProto::FLOAT),
                 std::invalid_argument);
    input = Corpus(profile);
    input.shape[0]++;
    EXPECT_THROW(DequantizePackedTensor(input, profile.format,
                                        {static_cast<int64_t>(64 * profile.block_size)},
                                        TensorProto::FLOAT),
                 std::invalid_argument);
  }
}

TEST(DequantizePacked, PreservesNonfiniteAndSignedZeroScales) {
  for (const uint16_t scale :
       {uint16_t{0x7c00}, uint16_t{0xfc00}, uint16_t{0x7e01}, uint16_t{0x8000}}) {
    std::vector<uint8_t> bytes(18, 0x98);
    Put16(bytes.data(), scale);
    const Tensor input("", TensorProto::UINT8, {18}, bytes);
    const auto output = DequantizePackedTensor(input, "gguf:q4_0", {32}, TensorProto::FLOAT);
    const float d = Float16BitsToFloat(scale);
    if (std::isinf(d)) {
      EXPECT_TRUE(std::isnan(output.AsFloat()[0]));
      EXPECT_EQ(output.AsFloat()[16], d);
    } else if (std::isnan(d))
      EXPECT_TRUE(std::isnan(output.AsFloat()[16]));
    else {
      EXPECT_EQ(std::bit_cast<uint32_t>(output.AsFloat()[0]), 0x80000000U);
      EXPECT_EQ(std::bit_cast<uint32_t>(output.AsFloat()[16]), 0x80000000U);
    }
  }
}

TEST(DequantizePacked, RunsThroughRuntimeAndOptimizedShapeDispatch) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_shapes::RegisterShapeFunctions();
  for (const auto &profile : PackedQuantizationProfiles()) {
    SCOPED_TRACE(profile.format);
    ModelProto model;
    model.set_ir_version(13);
    model.add_opset_import()->set_version(29);
    auto *graph = model.mutable_graph();
    graph->set_name("packed");
    graph->add_input()->set_name("X");
    graph->add_output()->set_name("Y");
    *graph->add_node() =
        PackedNode(std::string(profile.format), {static_cast<int64_t>(64 * profile.block_size)});
    const Tensor input = Corpus(profile);
    RuntimeSession session(model);
    RuntimeContext context;
    context.Put("X", input);
    session.Run(context);
    EXPECT_EQ(context.Get("Y").size_bytes(), 64 * profile.block_size * sizeof(float));

    core::shapes::ShapesContext shapes;
    core::symbolic::SymShape packed;
    packed.PushBack(core::symbolic::SymDim(input.shape[0]));
    shapes.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUint8,
                                              std::move(packed)));
    shapes.ComputeShapeNode(graph->node(0));
    EXPECT_EQ(shapes.Get("Y").Shape()[0].AsInt(), static_cast<int64_t>(64 * profile.block_size));
    EXPECT_EQ(shapes.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  }
}

TEST(DequantizePacked, OptimizedShapeChecksContractsAndEmptyDimensions) {
  using namespace core::symbolic;
  const SymShape partially_symbolic{SymDim(int64_t{0}), SymDim("bytes")};
  EXPECT_THROW(partially_symbolic.NumElements(), std::runtime_error);
  const int64_t max = std::numeric_limits<int64_t>::max();
  core::shapes::ShapesContext shapes;
  shapes.Set("X", SymTensor(nullptr, TensorType::kUint8, SymShape{SymDim("bytes")}));
  for (const int dtype :
       {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::DOUBLE}) {
    const auto node = PackedNode("gguf:q4_0", {max, 2, 0}, dtype);
    onnx_shapes::shapes::quantization::ComputeShapeDequantizePacked(shapes, node, "X");
    EXPECT_EQ(shapes.Get("Y").Shape().Rank(), size_t{3});
    EXPECT_EQ(shapes.Get("Y").Shape()[0].AsInt(), max);
    EXPECT_EQ(shapes.Get("Y").Shape()[2].AsInt(), 0);
    EXPECT_EQ(shapes.Get("Y").Shape().NumElements(), 0);
    EXPECT_EQ(shapes.Get("Y").Dtype(),
              DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype)));
  }

  for (const auto &node : {PackedNode("gguf:Q4_0", {0, 32}), PackedNode("gguf:q4_0", {0, -1, 32}),
                           PackedNode("gguf:q4_0", {0, 31}), PackedNode("gguf:q4_0", {max, 32}),
                           PackedNode("gguf:q4_0", {0, 32}, TensorProto::INT8)}) {
    EXPECT_THROW(onnx_shapes::shapes::quantization::ComputeShapeDequantizePacked(shapes, node, "X"),
                 std::invalid_argument);
  }
  const auto node = PackedNode("gguf:q4_0", {32});
  for (const SymTensor &input :
       {SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(18)}),
        SymTensor(nullptr, TensorType::kUint8, SymShape{}),
        SymTensor(nullptr, TensorType::kUint8, SymShape{SymDim(1), SymDim(18)}),
        SymTensor(nullptr, TensorType::kUint8, SymShape{SymDim(17)}),
        SymTensor(nullptr, TensorType::kUint8, SymShape{SymDim(-1)})}) {
    shapes.Set("X", SymTensor(input));
    EXPECT_THROW(onnx_shapes::shapes::quantization::ComputeShapeDequantizePacked(shapes, node, "X"),
                 std::invalid_argument);
  }
  shapes.Set("X", SymTensor(nullptr, TensorType::kUint8, SymShape{SymDim(18)}));
  for (const std::string name : {"format", "shape", "output_dtype"}) {
    NodeProto invalid = node;
    for (auto &attribute : *invalid.mutable_attribute())
      if (attribute.name() == name)
        attribute.set_type(AttributeProto::FLOAT);
    EXPECT_THROW(
        onnx_shapes::shapes::quantization::ComputeShapeDequantizePacked(shapes, invalid, "X"),
        std::invalid_argument);
  }
}

TEST(DequantizePacked, RuntimeAcceptsHugeEmptyOutputShape) {
  onnx_kernels::RegisterKernelFunctions();
  const int64_t max = std::numeric_limits<int64_t>::max();
  ModelProto model;
  model.set_ir_version(13);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("empty_packed");
  graph->add_input()->set_name("X");
  graph->add_output()->set_name("Y");
  *graph->add_node() = PackedNode("gguf:q4_0", {max, 2, 0});
  RuntimeSession session(model);
  RuntimeContext context;
  context.Put("X", Tensor("", TensorProto::UINT8, {0}, {}));
  session.Run(context);
  EXPECT_EQ(context.Get("Y").shape, (Shape{max, 2, 0}));
  EXPECT_EQ(context.Get("Y").element_count(), 0);
  EXPECT_EQ(context.Get("Y").size_bytes(), size_t{0});
}
