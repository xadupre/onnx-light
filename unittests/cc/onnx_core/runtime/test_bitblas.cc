// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

BitblasWeights Fixture(bool symmetric = false) {
  BitblasWeights result;
  result.layout = "row-packed-v1";
  result.method = "gptq";
  result.in_features = result.out_features = 32;
  result.group_size = -1;
  result.bits = 2;
  result.symmetric = symmetric;
  // Pinned mlx_packing.repack_gptq_bitblas decodes E4 as raw codes 0,1,2,3.
  result.tensors.emplace(
      "qweight", Tensor("qweight", TensorProto::INT8, {32, 8}, std::vector<uint8_t>(256, 0xe4)));
  result.tensors.emplace("scales",
                         MakeFloat16Tensor("scales", {32, 1}, std::vector<float>(32, 0.5f)));
  result.tensors.emplace("qzeros",
                         Tensor("qzeros", TensorProto::INT8, symmetric ? Shape{0} : Shape{1, 8},
                                std::vector<uint8_t>(symmetric ? 0 : 8, 0x55)));
  return result;
}

float At(const Tensor &tensor, size_t index) {
  float result;
  std::memcpy(&result, tensor.bytes() + index * 4, 4);
  return result;
}

} // namespace

TEST(BitblasWeights, RawCodesZerosSignednessAndBytes) {
  for (const bool symmetric : {false, true}) {
    const auto checkpoint = Fixture(symmetric);
    const auto value = ImportBitblasWeights(checkpoint);
    const auto codes = UnpackBitblasCodes(value);
    const auto reconstructed = ReconstructBitblasWeights(value);
    for (size_t i = 0; i < 1024; ++i) {
      EXPECT_EQ(codes.bytes()[i], i % 4);
      const int code = static_cast<int>(i % 4);
      EXPECT_EQ(At(reconstructed, i), 0.5f * (symmetric ? (code < 2 ? code : code - 4) : code - 1));
    }
    const auto exported = ExportBitblasWeights(value);
    EXPECT_EQ(exported.layout, checkpoint.layout);
    EXPECT_EQ(exported.method, checkpoint.method);
    for (const auto &[name, tensor] : checkpoint.tensors) {
      const auto &actual = exported.tensors.at(name);
      EXPECT_EQ(actual.data_type, tensor.data_type);
      EXPECT_EQ(actual.shape, tensor.shape);
      ASSERT_EQ(actual.size_bytes(), tensor.size_bytes());
      if (tensor.size_bytes())
        EXPECT_EQ(std::memcmp(actual.bytes(), tensor.bytes(), tensor.size_bytes()), 0);
    }
    auto metadata = Fixture(symmetric);
    metadata.tensors.clear();
    const auto packed = PackBitblasWeights(metadata, codes, checkpoint.tensors.at("scales"),
                                           checkpoint.tensors.at("qzeros"));
    EXPECT_EQ(packed.SerializeAsString(), value.SerializeAsString());
    EXPECT_EQ(At(DequantizeTensor(value), 2), At(reconstructed, 2));
  }
}

TEST(BitblasWeights, RejectsInvalidExtentsAndMetadataBeforeAllocation) {
  auto checkpoint = Fixture();
  checkpoint.in_features = checkpoint.out_features = int64_t{1} << 62;
  EXPECT_THROW(ImportBitblasWeights(checkpoint, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  checkpoint = Fixture();
  checkpoint.bits = 8;
  checkpoint.in_features = int64_t{1} << 56;
  EXPECT_THROW(ImportBitblasWeights(checkpoint, std::numeric_limits<size_t>::max()),
               std::invalid_argument);
  checkpoint = Fixture();
  EXPECT_THROW(ImportBitblasWeights(checkpoint, 4095), std::invalid_argument);
  checkpoint.layout = "bitblas";
  EXPECT_THROW(ImportBitblasWeights(checkpoint), std::invalid_argument);
  checkpoint = Fixture();
  checkpoint.tensors.at("qweight").shape = {4, 64};
  EXPECT_THROW(ImportBitblasWeights(checkpoint), std::invalid_argument);
  checkpoint = Fixture();
  checkpoint.tensors.at("qweight") =
      Tensor::Borrow("qweight", TensorProto::INT8, {32, 8}, nullptr, 256);
  EXPECT_THROW(ImportBitblasWeights(checkpoint), std::invalid_argument);
}

TEST(BitblasWeights, RejectsCorruptSchemasAndPayloadsOnEveryConsumer) {
  const auto original = ImportBitblasWeights(Fixture());
  for (int kind = 0; kind < 5; ++kind) {
    auto value = original;
    if (kind == 0)
      value.mutable_raw_data()->resize(original.raw_data().size() - 1);
    else if (kind == 1)
      value.mutable_raw_data()->resize(original.raw_data().size() + 1);
    else if (kind == 2)
      value.mutable_struct_type()->set_name("onnx_light.bitblas_weights.v2");
    else if (kind == 3)
      value.mutable_struct_type()
          ->mutable_structure()
          ->mutable_field(6)
          ->mutable_constant()
          ->ref_int64_data()[0] = 2;
    else
      value.mutable_logical_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(0)
          ->set_dim_value(64);
    EXPECT_THROW(ExportBitblasWeights(value), std::invalid_argument);
    EXPECT_THROW(UnpackBitblasCodes(value), std::invalid_argument);
    EXPECT_THROW(ReconstructBitblasWeights(value), std::invalid_argument);
  }
}
