// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/test_case_registry.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_lib/defs/function.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;

namespace packing_tests {
using Shape = core::runtime::Shape;
using Tensor = core::runtime::Tensor;

namespace {

const onnx_kernels::kernel::Pack pack{KernelContext{DefaultOpset(29)}};
const onnx_kernels::kernel::Unpack unpack{KernelContext{DefaultOpset(29)}};

EncodedValueProto BitPacked(uint32_t width = 3, uint64_t groups = 8,
                            std::vector<uint8_t> bytes = {0x88, 0xC6, 0xFA}) {
  EncodedValueProto result;
  auto *packing = result.mutable_struct_type()->mutable_bit_packing();
  packing->set_dimension(groups);
  auto *component = packing->add_component();
  component->set_name("code");
  component->set_bit_width(width);
  result.set_raw_data(bytes.data(), bytes.size());
  return result;
}

void LogicalShape(EncodedValueProto &value, const Shape &shape,
                  int32_t dtype = TensorProto::UINT8) {
  auto *tensor = value.mutable_logical_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  tensor->mutable_shape()->clear_dim();
  for (int64_t dimension : shape)
    tensor->mutable_shape()->add_dim()->set_dim_value(dimension);
}

void EqualTensor(const Tensor &actual, const Tensor &expected) {
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  if (actual.size_bytes()) {
    EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
  }
}

Tensor Oracle(const Tensor &x, int64_t radix, int64_t group_size, int64_t axis,
              int32_t output_type) {
  if (axis < 0)
    axis += x.shape.size();
  uint64_t limit = 1;
  for (int64_t j = 0; j < group_size; ++j)
    limit *= radix;
  int64_t width = 0;
  for (uint64_t v = limit - 1; v; v >>= 1)
    ++width;
  const int64_t word_bits = output_type == TensorProto::UINT8 ? 8 : 32;
  const int64_t n = x.shape[axis], groups = (n + group_size - 1) / group_size;
  Shape shape = x.shape;
  shape[axis] = (groups * width + word_bits - 1) / word_bits;
  std::vector<uint32_t> words(shape.product(), 0);
  const int64_t inner = shape.product(axis + 1, shape.size(), "oracle");
  const int64_t outer = shape.product(0, axis, "oracle");
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t i = 0; i < inner; ++i)
      for (int64_t g = 0; g < groups; ++g) {
        uint64_t code = 0;
        // Horner's rule independently computes the least-significant-first group.
        for (int64_t j = std::min(group_size, n - g * group_size); j > 0; --j)
          code = code * radix + x.bytes()[(o * n + g * group_size + j - 1) * inner + i];
        for (int64_t b = 0; b < width; ++b) {
          const int64_t bit = g * width + b;
          words[(o * shape[axis] + bit / word_bits) * inner + i] |=
              static_cast<uint32_t>((code >> b) & 1) << (bit % word_bits);
        }
      }
  if (output_type == TensorProto::UINT32)
    return Tensor::FromUint32("", shape, words);
  std::vector<uint8_t> bytes;
  bytes.reserve(words.size());
  for (const uint32_t word : words)
    bytes.push_back(static_cast<uint8_t>(word));
  return Tensor::FromUint8("", shape, bytes);
}

} // namespace

TEST(KernelPacking, BitsGroupsAxesWordsAndEmptyShapesMatchIndependentOracle) {
  struct Parameters {
    std::optional<int64_t> bits;
    int64_t radix;
    int64_t group_size;
  };
  std::vector<Parameters> parameters;
  for (int64_t bits = 1; bits <= 8; ++bits)
    parameters.push_back({bits, int64_t{1} << bits, 1});
  for (auto [radix, group] :
       {std::pair{3, 1}, {3, 5}, {5, 3}, {7, 2}, {16, 2}, {256, 1}, {2, 62}, {3, 39}, {5, 27}})
    parameters.push_back({std::nullopt, radix, group});
  for (const auto &p : parameters)
    for (const Shape &shape : {Shape{2, 11, 17}, Shape{0, 3, 9}, Shape{2, 0, 9}, Shape{2, 3, 0}})
      for (int64_t axis = -3; axis < 3; ++axis)
        for (int32_t type : {TensorProto::UINT8, TensorProto::UINT32}) {
          SCOPED_TRACE(::testing::Message()
                       << p.radix << "/" << p.group_size << " axis=" << axis << " type=" << type);
          std::vector<uint8_t> values(shape.product());
          for (size_t i = 0; i < values.size(); ++i)
            values[i] = (i * 17 + 3) % p.radix;
          Tensor x = Tensor::FromUint8("", shape, values);
          const std::optional<int64_t> radix = p.bits ? std::nullopt : std::optional{p.radix};
          const std::optional<int64_t> group = p.bits ? std::nullopt : std::optional{p.group_size};
          const Tensor packed = pack(x, p.bits, radix, group, axis, type);
          EqualTensor(packed, Oracle(x, p.radix, p.group_size, axis, type));
          const auto count = Tensor::FromInt64("", {}, {shape[axis < 0 ? axis + 3 : axis]});
          EqualTensor(unpack(packed, count, p.bits, radix, group, axis), x);
        }
}

TEST(KernelPacking, LongRowsRemainBoundedAndCrossWordGroupsRoundTrip) {
  for (int64_t group : {1, 5, 39})
    for (int32_t type : {TensorProto::UINT8, TensorProto::UINT32}) {
      std::vector<uint8_t> values(32 * group * 1001 + 1);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = i % 3;
      const auto x = Tensor::FromUint8("", {1, static_cast<int64_t>(values.size())}, values);
      const auto packed = pack(x, std::nullopt, 3, group, -1, type);
      EqualTensor(packed, Oracle(x, 3, group, -1, type));
      EqualTensor(
          unpack(packed, Tensor::FromInt64("", {}, {x.shape.back()}), std::nullopt, 3, group), x);
    }
}

TEST(KernelPacking, IgnoresUnusedDigitsAndPaddingButRejectsInvalidGroupIndices) {
  const auto two = Tensor::FromInt64("", {}, {2});
  EqualTensor(unpack(Tensor::FromUint8("", {1}, {242}), two, std::nullopt, 3, 5),
              Tensor::FromUint8("", {2}, {2, 2}));
  EqualTensor(unpack(Tensor::FromUint8("", {1}, {0xF8}), two, std::nullopt, 3, 2),
              Tensor::FromUint8("", {2}, {2, 2}));
  EqualTensor(unpack(Tensor::FromUint8("", {1}, {0xFF}), two, 3),
              Tensor::FromUint8("", {2}, {7, 7}));
  const auto all_bits = Tensor::FromUint32("", {2}, {0xFFFFFFFF, 0xFFFFFFFF});
  EqualTensor(unpack(all_bits, Tensor::FromInt64("", {}, {7}), 5),
              Tensor::FromUint8("", {7}, std::vector<uint8_t>(7, 31)));
  EXPECT_THROW(unpack(Tensor::FromUint8("", {1}, {243}), two, std::nullopt, 3, 5),
               std::invalid_argument);
  EXPECT_THROW(
      unpack(Tensor::FromUint8("", {1}, {3}), Tensor::FromInt64("", {}, {1}), std::nullopt, 3, 1),
      std::invalid_argument);
}

TEST(KernelPacking, AcceptsUnalignedBorrowedCountAndPackedWords) {
  alignas(int64_t) uint8_t count_bytes[sizeof(int64_t) + 1]{};
  alignas(uint32_t) uint8_t word_bytes[sizeof(uint32_t) + 1]{};
  const int64_t count_value = 2;
  const uint32_t word_value = 0xFFFFFFFF;
  std::memcpy(count_bytes + 1, &count_value, sizeof(count_value));
  std::memcpy(word_bytes + 1, &word_value, sizeof(word_value));
  const Tensor count =
      Tensor::Borrow("", TensorProto::INT64, {}, count_bytes + 1, sizeof(count_value));
  const Tensor words =
      Tensor::Borrow("", TensorProto::UINT32, {1}, word_bytes + 1, sizeof(word_value));
  EqualTensor(unpack(words, count, 3), Tensor::FromUint8("", {2}, {7, 7}));
}

TEST(KernelPacking, RejectsInvalidAttributesEvenForEmptyInputs) {
  const auto x = Tensor::FromUint8("", {0}, {});
  const auto count = Tensor::FromInt64("", {}, {0});
  struct Attributes {
    std::optional<int64_t> bits, radix, group;
  };
  for (const Attributes &a :
       {Attributes{}, Attributes{0, {}, {}}, Attributes{-1, {}, {}}, Attributes{9, {}, {}},
        Attributes{3, 3, 5}, Attributes{3, 3, {}}, Attributes{3, {}, 5}, Attributes{{}, 3, {}},
        Attributes{{}, {}, 5}, Attributes{{}, 1, 5}, Attributes{{}, 257, 1}, Attributes{{}, 3, 0},
        Attributes{{}, 3, -1}, Attributes{{}, 256, 8}, Attributes{{}, 2, 63}, Attributes{{}, 3, 40},
        Attributes{{}, 2, std::numeric_limits<int64_t>::max()}}) {
    EXPECT_THROW(pack(x, a.bits, a.radix, a.group), std::invalid_argument);
    EXPECT_THROW(unpack(x, count, a.bits, a.radix, a.group), std::invalid_argument);
  }
  for (int64_t axis : {int64_t{-2}, int64_t{1}, std::numeric_limits<int64_t>::min()}) {
    EXPECT_THROW(pack(x, 3, {}, {}, axis), std::invalid_argument);
    EXPECT_THROW(unpack(x, count, 3, {}, {}, axis), std::invalid_argument);
  }
  for (int64_t type : {int64_t(TensorProto::INT8), int64_t(TensorProto::UINT64),
                       std::numeric_limits<int64_t>::max()})
    EXPECT_THROW(pack(x, 3, {}, {}, -1, type), std::invalid_argument);
}

TEST(KernelPacking, ValidatesTypesCountsShapesBuffersAndOverflowBeforeEmptyFastPath) {
  const auto x = Tensor::FromUint8("", {1}, {0});
  const auto count = Tensor::FromInt64("", {}, {1});
  for (const auto &bad_count :
       {Tensor::FromInt64("", {}, {-1}), Tensor::FromInt64("", {1}, {1}),
        Tensor::FromInt32("", {}, {1}), Tensor("", TensorProto::INT64, {}, std::vector<uint8_t>{})})
    EXPECT_THROW(unpack(x, bad_count, 3), std::invalid_argument);
  for (const auto &bad : {Tensor::FromUint8("", {}, {0}), Tensor::FromInt8("", {1}, {0}),
                          Tensor("", TensorProto::UINT8, {2}, std::vector<uint8_t>{0}),
                          Tensor("", TensorProto::UINT8, {-1}, std::vector<uint8_t>{})}) {
    EXPECT_THROW(pack(bad, 3), std::invalid_argument);
    EXPECT_THROW(unpack(bad, count, 3), std::invalid_argument);
  }
  EXPECT_THROW(pack(Tensor::FromUint8("", {1}, {8}), 3), std::invalid_argument);
  EXPECT_THROW(pack(Tensor::FromUint8("", {1}, {3}), {}, 3, 5), std::invalid_argument);
  EXPECT_THROW(unpack(x, Tensor::FromInt64("", {}, {0}), 3), std::invalid_argument);
  EXPECT_THROW(unpack(x, Tensor::FromInt64("", {}, {3}), 3), std::invalid_argument);
  const auto empty_wrong_words = Tensor::FromUint8("", {0, 2}, {});
  EXPECT_THROW(unpack(empty_wrong_words, count, 3), std::invalid_argument);
  const int64_t maximum = std::numeric_limits<int64_t>::max();
  const Tensor huge_empty("", TensorProto::UINT8, {0, maximum}, std::vector<uint8_t>{});
  EqualTensor(pack(huge_empty, 8), huge_empty);
  EqualTensor(unpack(huge_empty, Tensor::FromInt64("", {}, {maximum}), 8), huge_empty);
  const Tensor huge_packed = pack(huge_empty, 7, {}, {}, -1, TensorProto::UINT32);
  const int64_t expected_words = maximum / 32 * 7 + ((maximum % 32) * 7 + 31) / 32;
  EXPECT_EQ(huge_packed.shape, (Shape{0, expected_words}));
  EqualTensor(unpack(huge_packed, Tensor::FromInt64("", {}, {maximum}), 7), huge_empty);
  EXPECT_THROW(pack(huge_empty, {}, 256, 6), std::invalid_argument);
  EXPECT_THROW(unpack(huge_empty, Tensor::FromInt64("", {}, {maximum}), {}, 256, 6),
               std::invalid_argument);
  EXPECT_THROW(unpack(x, Tensor::FromInt64("", {}, {maximum}), 8), std::invalid_argument);
  const Tensor overflow("", TensorProto::UINT8, {maximum, 2}, std::vector<uint8_t>{});
  EXPECT_THROW(pack(overflow, 3), std::invalid_argument);
  EXPECT_THROW(unpack(overflow, count, 3), std::invalid_argument);
  const Tensor word_byte_overflow("", TensorProto::UINT32, {maximum}, std::vector<uint8_t>{});
  EXPECT_THROW(unpack(word_byte_overflow, count, 3), std::invalid_argument);
}

TEST(KernelPacking, RuntimeRejectsNonIntegerAttributesWithoutNarrowing) {
  for (const std::string op : {"Pack", "Unpack"})
    for (const std::string attribute : {"bits", "radix", "group_size", "axis", "output_type"}) {
      if (op == "Unpack" && attribute == "output_type")
        continue;
      NodeProto node = op == "Pack" ? MakeNode(op.c_str(), {"X"}, {"Y"})
                                    : MakeNode(op.c_str(), {"X", "count"}, {"Y"});
      if (attribute != "bits")
        AddAttribute<int64_t>(node, "bits", 3);
      AddAttribute<float>(node, attribute.c_str(), 3.0f);
      RuntimeContext context;
      context.Set("X", Tensor::FromUint8("", {0}, {}));
      context.Set("count", Tensor::FromInt64("", {}, {0}));
      if (op == "Pack") {
        onnx_kernels::kernel::Pack kernel{KernelContext{DefaultOpset(29)}};
        kernel.set_node(node);
        EXPECT_THROW(kernel.Run(context), std::invalid_argument);
      } else {
        onnx_kernels::kernel::Unpack kernel{KernelContext{DefaultOpset(29)}};
        kernel.set_node(node);
        EXPECT_THROW(kernel.Run(context), std::invalid_argument);
      }
    }
}

TEST(KernelPacking, EncodedBitPackingAndRadixGroupsRespectDeclaredRows) {
  const auto count = Tensor::FromInt64("", {}, {8});
  auto encoded = BitPacked();
  const auto digits = Tensor::FromUint8("", {8}, {0, 1, 2, 3, 4, 5, 6, 7});
  EqualTensor(unpack(encoded, count, 3), digits);
  LogicalShape(encoded, {8});
  EqualTensor(unpack(encoded, count, 3), digits);
  EXPECT_THROW(unpack(encoded, count, 2), std::invalid_argument);
  EXPECT_THROW(unpack(encoded, Tensor::FromInt64("", {}, {7}), 3), std::invalid_argument);
  LogicalShape(encoded, {3});
  EXPECT_THROW(unpack(encoded, count, 3), std::invalid_argument);
  LogicalShape(encoded, {8}, TensorProto::UINT32);
  EXPECT_THROW(unpack(encoded, count, 3), std::invalid_argument);

  auto ternary = BitPacked(8, 2, {102, 242});
  EqualTensor(unpack(ternary, Tensor::FromInt64("", {}, {7}), {}, 3, 5),
              Tensor::FromUint8("", {7}, {0, 1, 2, 0, 1, 2, 2}));
  EXPECT_THROW(unpack(ternary, Tensor::FromInt64("", {}, {5}), {}, 3, 5), std::invalid_argument);
  EXPECT_THROW(unpack(ternary, Tensor::FromInt64("", {}, {7}), {}, 3, 4), std::invalid_argument);
  ternary.set_raw_data("\xF3\x00", 2);
  EXPECT_THROW(unpack(ternary, Tensor::FromInt64("", {}, {10}), {}, 3, 5), std::invalid_argument);

  auto rows = BitPacked();
  rows.set_raw_data("\x88\xC6\xFA\x88\xC6\xFA", 6);
  LogicalShape(rows, {2, 8});
  EqualTensor(unpack(rows, count, 3),
              Tensor::FromUint8("", {2, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_THROW(unpack(rows, count, 3, {}, {}, 0), std::invalid_argument);
  EXPECT_THROW(unpack(rows, Tensor::FromInt64("", {}, {16}), 3), std::invalid_argument);
}

TEST(KernelPacking, OnlyUnpackLightweightSchemaAcceptsStructuredStorage) {
  using namespace core::schema;
  ValueInfoProto encoded;
  encoded.set_name("X");
  *encoded.mutable_type()->mutable_struct_type() = BitPacked().struct_type();
  std::vector<std::optional<SchemaInputValue>> inputs{encoded};
  for (const std::string op : {"Pack", "Unpack"}) {
    const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory(op);
    auto schema = std::find_if(schemas.begin(), schemas.end(),
                               [](const auto &value) { return value.since_version() == 29; });
    ASSERT_NE(schema, schemas.end());
    auto node =
        op == "Unpack" ? MakeNode("Unpack", {"X", "count"}, {"Y"}) : MakeNode("Pack", {"X"}, {"Y"});
    AddAttribute<int64_t>(node, "bits", 3);
    if (op == "Unpack") {
      EXPECT_NO_THROW(schema->Verify(node, &inputs));
    } else {
      EXPECT_THROW(schema->Verify(node, &inputs), core::schema::SchemaError);
    }
  }
}

TEST(KernelPacking, EncodedLayoutsRejectMalformedAndUnsupportedMetadataEvenWhenEmpty) {
  const auto count = Tensor::FromInt64("", {}, {8});
  std::vector<EncodedValueProto> invalid;
  auto add = [&](auto mutate) {
    auto encoded = BitPacked(3, 8, {});
    mutate(encoded);
    invalid.push_back(std::move(encoded));
  };
  add([](auto &v) { v.mutable_struct_type()->mutable_bit_packing()->set_dimension(0); });
  add([](auto &v) {
    v.mutable_struct_type()->mutable_bit_packing()->mutable_component(0)->set_name("");
  });
  add([](auto &v) {
    v.mutable_struct_type()->mutable_bit_packing()->mutable_component(0)->set_bit_width(0);
  });
  add([](auto &v) {
    auto *component = v.mutable_struct_type()->mutable_bit_packing()->add_component();
    component->set_name("other");
    component->set_bit_width(5);
  });
  add([](auto &v) { v.mutable_struct_type()->mutable_decoder()->set_name("decode"); });
  add([](auto &v) { v.mutable_struct_type()->mutable_encoder()->set_name("encode"); });
  add([](auto &v) { v.mutable_struct_type()->set_type_ref(42); });
  add([](auto &v) { v.mutable_struct_type()->mutable_structure(); });
  add([](auto &v) { v.mutable_affine(); });
  add([](auto &v) {
    v.set_parameter_ref("shared");
    LogicalShape(v, {0, 8});
  });
  add([](auto &v) {
    v.set_data_location(TensorProto::EXTERNAL);
    v.clear_raw_data();
    auto *location = v.add_external_data();
    location->set_key("location");
    location->set_value("unopened.bin");
    auto *length = v.add_external_data();
    length->set_key("length");
    length->set_value("0");
  });
  add([](auto &v) { LogicalShape(v, {0, 7}); });
  add([](auto &v) {
    LogicalShape(v, {0, 8});
    v.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_param(
        "N");
  });
  add([](auto &v) {
    LogicalShape(v, {0, 8});
    auto *dimension =
        v.mutable_logical_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
    dimension->clear_dim_value();
    dimension->set_dim_param("N");
  });
  add([](auto &v) {
    v.mutable_struct_type()->mutable_bit_packing()->set_dimension(
        std::numeric_limits<uint64_t>::max());
  });
  add([](auto &v) {
    auto *packing = v.mutable_struct_type()->mutable_bit_packing();
    packing->set_dimension(uint64_t{1} << 63);
    packing->mutable_component(0)->set_bit_width(1);
  });
  add([](auto &v) { v.set_raw_data("\x00", 1); });
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_THROW(unpack(invalid[i], count, 3), std::invalid_argument);
  }
  auto empty = BitPacked(3, 8, {});
  LogicalShape(empty, {0, 8});
  EqualTensor(unpack(empty, count, 3), Tensor::FromUint8("", {0, 8}, {}));
  EXPECT_THROW(unpack(empty, count, 0), std::invalid_argument);
  EXPECT_THROW(unpack(empty, count, 3, 3, 5), std::invalid_argument);
  EXPECT_THROW(unpack(empty, count, 3, {}, {}, 2), std::invalid_argument);
  EXPECT_THROW(unpack(empty, Tensor::FromInt64("", {}, {-1}), 3), std::invalid_argument);
  EXPECT_THROW(unpack(empty, Tensor::FromInt32("", {}, {8}), 3), std::invalid_argument);
  EXPECT_THROW(unpack(empty, Tensor::FromInt64("", {}, {std::numeric_limits<int64_t>::max()}), 3),
               std::invalid_argument);
}

TEST(KernelPacking, EncodedArraysValidateNestedReferencesCodecsAlignmentAndPhysicalDtypes) {
  const auto count = Tensor::FromInt64("", {}, {8});
  ModelProto model;
  auto *leaf = model.add_struct_types();
  *leaf = BitPacked().struct_type();
  leaf->set_type_id(1);
  auto *root = model.add_struct_types();
  root->set_type_id(2);
  root->mutable_array()->set_dimension(2);
  root->mutable_array()->mutable_element_type()->mutable_struct_type()->set_type_ref(1);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  EncodedValueProto encoded;
  encoded.mutable_struct_type()->set_type_ref(2);
  encoded.set_raw_data("\x88\xC6\xFA\x88\xC6\xFA", 6);
  EqualTensor(unpack(encoded, count, 3, {}, {}, -1, catalogue),
              Tensor::FromUint8("", {2, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_THROW(unpack(encoded, count, 3), std::invalid_argument);

  model.mutable_struct_types(0)->mutable_decoder()->set_name("nested_codec");
  catalogue.Build(model);
  EXPECT_THROW(unpack(encoded, count, 3, {}, {}, -1, catalogue), std::invalid_argument);
  model.mutable_struct_types(0)->clear_decoder();
  model.mutable_struct_types(0)->mutable_bit_packing()->set_dimension(4);
  catalogue.Build(model);
  encoded.set_raw_data("\0\0\0", 3);
  EXPECT_THROW(unpack(encoded, Tensor::FromInt64("", {}, {4}), 3, {}, {}, -1, catalogue),
               std::invalid_argument);

  for (int32_t dtype :
       {TensorProto::UINT8, TensorProto::UINT32, TensorProto::INT8, TensorProto::FLOAT}) {
    EncodedValueProto storage;
    auto *array = storage.mutable_struct_type()->mutable_array();
    array->set_dimension(1);
    auto *tensor = array->mutable_element_type()->mutable_tensor_type();
    tensor->set_elem_type(dtype);
    tensor->mutable_shape();
    storage.set_raw_data(dtype == TensorProto::UINT8 || dtype == TensorProto::INT8
                             ? std::string("\xE4", 1)
                             : std::string("\xE4\x00\x00\x00", 4));
    if (dtype == TensorProto::UINT8 || dtype == TensorProto::UINT32) {
      EqualTensor(unpack(storage, Tensor::FromInt64("", {}, {4}), 2),
                  Tensor::FromUint8("", {4}, {0, 1, 2, 3}));
    } else {
      EXPECT_THROW(unpack(storage, Tensor::FromInt64("", {}, {4}), 2), std::invalid_argument);
    }
    tensor->mutable_shape()->add_dim()->set_dim_param("physical");
    EXPECT_THROW(unpack(storage, Tensor::FromInt64("", {}, {4}), 2), std::invalid_argument);
    tensor->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    EXPECT_THROW(unpack(storage, Tensor::FromInt64("", {}, {4}), 2), std::invalid_argument);
  }
}

TEST(KernelPacking, BackendCasesExecuteNativeAndExpandedFunctionsInRuntimeSession) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  for (const std::string op : {"Pack", "Unpack"}) {
    auto cases = core::backend_test::CollectTestCases(op);
    ASSERT_EQ(cases.size(), 5u);
    // Keep synthetic empty-row dimensions representable on 32-bit platforms.
    const int64_t large_empty_row_dim =
        std::min<int64_t>(int64_t{1} << 61, std::numeric_limits<std::ptrdiff_t>::max() - 1);
    for (int64_t bits = 1; bits <= 8; ++bits)
      for (int64_t n : {int64_t{0}, int64_t{9}, large_empty_row_dim, large_empty_row_dim + 1}) {
        const Tensor digits = Tensor::FromUint8("", {0, n}, {});
        const int64_t words = n / 8 * bits + ((n % 8) * bits + 7) / 8;
        const Tensor packed = Tensor::FromUint8("", {0, words}, {});
        NodeProto node = op == "Pack" ? MakeNode(op.c_str(), {"X"}, {"Y"})
                                      : MakeNode(op.c_str(), {"X", "count"}, {"Y"});
        AddAttribute<int64_t>(node, "bits", bits);
        core::backend_test::Expect(
            cases, std::move(node),
            "test_" + op + "_empty_rows_" + std::to_string(bits) + "_" + std::to_string(n),
            {DefaultOpset(29)}, [digits, packed, n, op]() -> core::backend_test::IoData {
              if (op == "Unpack")
                return {{packed, Tensor::FromInt64("", {}, {n})}, {digits}};
              return {{digits}, {packed}};
            });
      }
    for (const auto &test_case : cases) {
      SCOPED_TRACE(test_case.name);
      ASSERT_EQ(test_case.model().opset_import(0).version(), 29);
      for (bool expanded : {false, true}) {
        SCOPED_TRACE(expanded);
        ModelProto model = test_case.model();
        if (expanded) {
          const NodeProto node = model.graph().node(0);
          std::vector<TypeProto> types;
          for (const auto &input : model.graph().input())
            types.push_back(input.type());
          const auto *schema = OpSchemaRegistry::Schema(op, 29, "");
          ASSERT_NE(schema, nullptr);
          FunctionBodyBuildContextImpl context(node, types);
          FunctionProto function;
          ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, 29));
          model.mutable_graph()->clear_node();
          FunctionExpandHelper(node, function, *model.mutable_graph());
          ASSERT_GT(model.graph().node_size(), 1);
        }
        for (const auto &data : test_case.data_sets()) {
          RuntimeContext context;
          for (const auto &input : data.inputs)
            context.Set(input.name, input);
          RuntimeSession session(model);
          session.Run(context);
          EqualTensor(context.Get(data.outputs[0].name), data.outputs[0]);
        }
      }
    }
  }
}

TEST(KernelPacking, ExpandedWideGroupsCrossWordsAndCaptureLoopValues) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  for (auto [radix, group] : {std::pair{3, 39}, {5, 27}})
    for (int32_t storage : {TensorProto::UINT8, TensorProto::UINT32})
      for (int64_t axis : {0, 1, -1})
        for (const Shape &shape : {Shape{2, 81, 3}, Shape{0, 81, 3}, Shape{2, 0, 3}}) {
          SCOPED_TRACE(::testing::Message()
                       << radix << "/" << group << " axis=" << axis << " storage=" << storage
                       << " size=" << shape.product());
          std::vector<uint8_t> values(shape.product());
          for (size_t i = 0; i < values.size(); ++i)
            values[i] = radix == 5 ? 4 : (i + 2) % radix;
          Tensor digits = Tensor::FromUint8("X", shape, values);
          Tensor packed = pack(digits, {}, radix, group, axis, storage);
          EqualTensor(packed, Oracle(digits, radix, group, axis, storage));
          packed.name = "X";
          for (const std::string op : {"Pack", "Unpack"}) {
            const Tensor &input = op == "Pack" ? digits : packed;
            const Tensor &expected = op == "Pack" ? packed : digits;
            ModelProto model;
            model.set_ir_version(IR_VERSION);
            model.add_opset_import()->set_version(29);
            auto *graph = model.mutable_graph();
            graph->set_name("wide_group_packing");
            auto *info = graph->add_input();
            info->set_name("X");
            auto *type = info->mutable_type()->mutable_tensor_type();
            type->set_elem_type(input.data_type);
            for (int64_t dimension : input.shape)
              type->mutable_shape()->add_dim()->set_dim_value(dimension);
            NodeProto node = op == "Pack" ? MakeNode(op.c_str(), {"X"}, {"Y"})
                                          : MakeNode(op.c_str(), {"X", "count"}, {"Y"});
            AddAttribute<int64_t>(node, "radix", radix);
            AddAttribute<int64_t>(node, "group_size", group);
            AddAttribute<int64_t>(node, "axis", axis);
            if (op == "Pack") {
              AddAttribute<int64_t>(node, "output_type", storage);
            } else {
              auto *count_info = graph->add_input();
              count_info->set_name("count");
              auto *count_type = count_info->mutable_type()->mutable_tensor_type();
              count_type->set_elem_type(TensorProto::INT64);
              count_type->mutable_shape();
            }
            auto *output = graph->add_output();
            output->set_name("Y");
            auto *output_type = output->mutable_type()->mutable_tensor_type();
            output_type->set_elem_type(expected.data_type);
            for (int64_t dimension : expected.shape)
              output_type->mutable_shape()->add_dim()->set_dim_value(dimension);
            std::vector<TypeProto> types;
            for (const auto &value : graph->input())
              types.push_back(value.type());
            const auto *schema = OpSchemaRegistry::Schema(op, 29, "");
            ASSERT_NE(schema, nullptr);
            FunctionBodyBuildContextImpl build_context(node, types);
            FunctionProto function;
            ASSERT_TRUE(schema->BuildContextDependentFunction(build_context, function, 29));
            FunctionExpandHelper(node, function, *graph);
            RuntimeContext context;
            context.Set("X", input);
            context.Set("count", Tensor::FromInt64("", {}, {shape[axis < 0 ? axis + 3 : axis]}));
            RuntimeSession session(model);
            session.Run(context);
            EqualTensor(context.Get("Y"), expected);
          }
        }
}

} // namespace packing_tests
