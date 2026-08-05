// Unit tests for the low-level field serialization helpers used by the proto
// codecs: read_field, read_field_limit_parallel, size_field, write_field and
// write_field_limit (plus the size_field_limit companion). They are exercised
// through the public proto serialization API, which is the only supported entry
// point into these helpers.
#include "onnx.h"
#include "onnx_light_helpers.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Encodes a value as a base-128 varint (protobuf wire format).
std::string EncodeVarint(uint64_t value) {
  std::string out;
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
  return out;
}

// Builds a protobuf field tag (field_number << 3 | wire_type) as a varint.
std::string FieldTag(int field_number, int wire_type) {
  return EncodeVarint((static_cast<uint64_t>(field_number) << 3) |
                      static_cast<uint64_t>(wire_type));
}

// Builds a TensorProto raw_data payload (float TensorProto) of the given size.
TensorProto MakeRawDataTensor(const std::string &name, size_t num_bytes) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(static_cast<int64_t>(num_bytes / sizeof(float)));
  tensor.ref_raw_data().resize(num_bytes);
  for (size_t i = 0; i < num_bytes; ++i) {
    tensor.ref_raw_data().data()[i] = static_cast<uint8_t>(i & 0xFF);
  }
  return tensor;
}

} // namespace

// ---------------------------------------------------------------------------
// write_field / read_field / size_field round-trips for the scalar, string and
// sub-message specializations. size_field correctness is checked by comparing
// the precomputed SerializeSize against the actual serialized byte count.
// ---------------------------------------------------------------------------

TEST(onnx_field_serialization, WriteReadSizeField_Int64) {
  AttributeProto attr;
  attr.set_name("i_attr");
  attr.ref_i() = 123456789;

  std::string serialized;
  attr.SerializeToString(serialized);
  EXPECT_EQ(serialized.size(), attr.SerializeSize().size());

  AttributeProto parsed;
  parsed.ParseFromString(serialized);
  EXPECT_TRUE(parsed.has_i());
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(onnx_field_serialization, WriteReadSizeField_Float) {
  AttributeProto attr;
  attr.set_name("f_attr");
  attr.ref_f() = 3.5f;

  std::string serialized;
  attr.SerializeToString(serialized);
  EXPECT_EQ(serialized.size(), attr.SerializeSize().size());

  AttributeProto parsed;
  parsed.ParseFromString(serialized);
  EXPECT_TRUE(parsed.has_f());
  EXPECT_FLOAT_EQ(parsed.ref_f(), 3.5f);
}

TEST(onnx_field_serialization, WriteReadSizeField_String) {
  AttributeProto attr;
  attr.set_name("s_attr");
  attr.ref_s() = utils::String("string-bytes", 12);

  std::string serialized;
  attr.SerializeToString(serialized);
  EXPECT_EQ(serialized.size(), attr.SerializeSize().size());

  AttributeProto parsed;
  parsed.ParseFromString(serialized);
  EXPECT_EQ(parsed.ref_name(), "s_attr");
  EXPECT_EQ(parsed.ref_s(), "string-bytes");
}

TEST(onnx_field_serialization, WriteReadSizeField_SubMessage) {
  // SparseTensorProto.values / indices are non-optional, non-repeated
  // sub-message fields, so they route through the templated
  // write_field<T>/read_field<T>/size_field<T> overloads.
  SparseTensorProto sparse;
  sparse.ref_values().set_name("values");
  sparse.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.ref_indices().set_name("indices");
  sparse.ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse.ref_dims().push_back(4);

  std::string serialized;
  sparse.SerializeToString(serialized);
  EXPECT_EQ(serialized.size(), sparse.SerializeSize().size());

  SparseTensorProto parsed;
  parsed.ParseFromString(serialized);
  EXPECT_EQ(parsed.ref_values().ref_name(), "values");
  EXPECT_EQ(parsed.ref_indices().ref_name(), "indices");
  ASSERT_EQ(parsed.ref_dims().size(), 1u);
  EXPECT_EQ(parsed.ref_dims()[0], 4);
}

// ---------------------------------------------------------------------------
// read_field wire-type validation: a known field number carrying the wrong
// wire type must be rejected by read_field's EXT_ENFORCE check.
// ---------------------------------------------------------------------------

TEST(onnx_field_serialization, ReadField_String_WrongWireType_Throws) {
  // AttributeProto.name (field 1) is a string and must use FIELD_FIXED_SIZE
  // (wire type 2); emitting it as a varint (wire type 0) must throw.
  std::string bytes = FieldTag(1, 0) + EncodeVarint(42);
  AttributeProto parsed;
  EXPECT_THROW(parsed.ParseFromString(bytes), std::runtime_error);
}

TEST(onnx_field_serialization, ReadField_Int64_WrongWireType_Throws) {
  // AttributeProto.i (field 3) is an int64 and must use FIELD_VARINT
  // (wire type 0); emitting it as length-delimited (wire type 2) must throw.
  std::string payload = "xy";
  std::string bytes = FieldTag(3, 2) + EncodeVarint(payload.size()) + payload;
  AttributeProto parsed;
  EXPECT_THROW(parsed.ParseFromString(bytes), std::runtime_error);
}

TEST(onnx_field_serialization, ReadFieldLimitParallel_RawData_WrongWireType_Throws) {
  // TensorProto.raw_data (field 9) is read via read_field_limit_parallel and
  // must use FIELD_FIXED_SIZE (wire type 2); a varint (wire type 0) must throw.
  std::string bytes = FieldTag(9, 0) + EncodeVarint(1);
  TensorProto parsed;
  EXPECT_THROW(parsed.ParseFromString(bytes), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Repeated numeric fields must be accepted in both the packed and the unpacked
// wire format, regardless of how onnx-light itself would emit them. Some
// producers pack TensorProto.dims (field 1) as a length-delimited block; the
// reader has to decode it (see gh_issue_24203).
// ---------------------------------------------------------------------------

TEST(onnx_field_serialization, ReadRepeatedInt64_PackedWireFormat) {
  // TensorProto.dims (field 1) emitted packed: wire type 2, one varint payload.
  std::string payload = EncodeVarint(3) + EncodeVarint(5) + EncodeVarint(7);
  std::string bytes = FieldTag(1, 2) + EncodeVarint(payload.size()) + payload;
  TensorProto parsed;
  parsed.ParseFromString(bytes);
  ASSERT_EQ(parsed.ref_dims().size(), 3u);
  EXPECT_EQ(parsed.ref_dims()[0], 3);
  EXPECT_EQ(parsed.ref_dims()[1], 5);
  EXPECT_EQ(parsed.ref_dims()[2], 7);
}

TEST(onnx_field_serialization, ReadRepeatedInt64_UnpackedWireFormat) {
  // The same field emitted unpacked: repeated varint entries (wire type 0).
  std::string bytes = FieldTag(1, 0) + EncodeVarint(3) + FieldTag(1, 0) + EncodeVarint(5) +
                      FieldTag(1, 0) + EncodeVarint(7);
  TensorProto parsed;
  parsed.ParseFromString(bytes);
  ASSERT_EQ(parsed.ref_dims().size(), 3u);
  EXPECT_EQ(parsed.ref_dims()[0], 3);
  EXPECT_EQ(parsed.ref_dims()[1], 5);
  EXPECT_EQ(parsed.ref_dims()[2], 7);
}

// AttributeProto.ints (field 8) is declared unpacked in the ONNX schema, but
// producers such as protobuf/onnxruntime emit it packed. TreeEnsemble stores
// its tree topology (nodes_featureids, tree_roots, ...) in ``ints`` attributes,
// so the packed wire format must be accepted to load such models (see the
// onnxruntime ``TreeEnsembleLeafLike`` test).
TEST(onnx_field_serialization, ReadAttributeInts_PackedWireFormat) {
  std::string payload =
      EncodeVarint(0) + EncodeVarint(1) + EncodeVarint(0) + EncodeVarint(1) + EncodeVarint(2);
  std::string bytes = FieldTag(8, 2) + EncodeVarint(payload.size()) + payload;
  AttributeProto parsed;
  parsed.ParseFromString(bytes);
  ASSERT_EQ(parsed.ref_ints().size(), 5u);
  EXPECT_EQ(parsed.ref_ints()[0], 0);
  EXPECT_EQ(parsed.ref_ints()[1], 1);
  EXPECT_EQ(parsed.ref_ints()[2], 0);
  EXPECT_EQ(parsed.ref_ints()[3], 1);
  EXPECT_EQ(parsed.ref_ints()[4], 2);
}

TEST(onnx_field_serialization, ReadAttributeInts_UnpackedWireFormat) {
  std::string bytes = FieldTag(8, 0) + EncodeVarint(0) + FieldTag(8, 0) + EncodeVarint(1) +
                      FieldTag(8, 0) + EncodeVarint(2);
  AttributeProto parsed;
  parsed.ParseFromString(bytes);
  ASSERT_EQ(parsed.ref_ints().size(), 3u);
  EXPECT_EQ(parsed.ref_ints()[0], 0);
  EXPECT_EQ(parsed.ref_ints()[1], 1);
  EXPECT_EQ(parsed.ref_ints()[2], 2);
}

// ---------------------------------------------------------------------------
// write_field_limit / size_field_limit / read_field_limit_parallel: raw_data
// behavior under the default, skip_raw_data and parse-skip configurations.
// ---------------------------------------------------------------------------

TEST(onnx_field_serialization, WriteFieldLimit_RawData_DefaultRoundTrip) {
  TensorProto tensor = MakeRawDataTensor("raw_default", 32);

  std::string serialized;
  SerializeOptions sopts;
  utils::StringWriteStream size_stream;
  tensor.SerializeToString(serialized, sopts);
  EXPECT_EQ(serialized.size(), tensor.SerializeSize(size_stream, sopts).size());

  TensorProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_EQ(parsed.ref_raw_data().size(), 32u);
  EXPECT_EQ(std::memcmp(parsed.ref_raw_data().data(), tensor.ref_raw_data().data(), 32), 0);
}

TEST(onnx_field_serialization, WriteFieldLimit_SkipRawData_AboveThresholdOmitted) {
  // raw_data larger than the threshold is omitted entirely when
  // skip_raw_data is set; both the serialized bytes and the precomputed size
  // must reflect the omission.
  TensorProto tensor = MakeRawDataTensor("raw_skip", 64);

  std::string with_raw;
  SerializeOptions default_opts;
  tensor.SerializeToString(with_raw, default_opts);

  std::string without_raw;
  SerializeOptions skip_opts;
  skip_opts.skip_raw_data = true;
  skip_opts.raw_data_threshold = 16; // 64-byte raw_data exceeds the threshold.
  utils::StringWriteStream size_stream;
  tensor.SerializeToString(without_raw, skip_opts);

  EXPECT_LT(without_raw.size(), with_raw.size());
  EXPECT_EQ(without_raw.size(), tensor.SerializeSize(size_stream, skip_opts).size());

  TensorProto parsed;
  parsed.ParseFromString(without_raw);
  EXPECT_EQ(parsed.ref_raw_data().size(), 0u);
  EXPECT_EQ(parsed.ref_name(), "raw_skip");
}

TEST(onnx_field_serialization, WriteFieldLimit_SkipRawData_BelowThresholdPreserved) {
  // raw_data smaller than the threshold is still written even with
  // skip_raw_data, exercising the size-based guard in write_field_limit /
  // size_field_limit.
  TensorProto tensor = MakeRawDataTensor("raw_small", 8);

  std::string serialized;
  SerializeOptions skip_opts;
  skip_opts.skip_raw_data = true;
  skip_opts.raw_data_threshold = 1024; // 8-byte raw_data stays below threshold.
  utils::StringWriteStream size_stream;
  tensor.SerializeToString(serialized, skip_opts);
  EXPECT_EQ(serialized.size(), tensor.SerializeSize(size_stream, skip_opts).size());

  TensorProto parsed;
  parsed.ParseFromString(serialized);
  ASSERT_EQ(parsed.ref_raw_data().size(), 8u);
  EXPECT_EQ(std::memcmp(parsed.ref_raw_data().data(), tensor.ref_raw_data().data(), 8), 0);
}

TEST(onnx_field_serialization, ReadFieldLimitParallel_SkipRawData_AboveThresholdSkipped) {
  // On the read side, raw_data above the threshold is skipped when
  // skip_raw_data is requested, leaving the rest of the message intact.
  TensorProto tensor = MakeRawDataTensor("raw_read_skip", 64);
  std::string serialized;
  SerializeOptions sopts;
  tensor.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.skip_raw_data = true;
  popts.raw_data_threshold = 16;
  parsed.ParseFromString(serialized, popts);
  EXPECT_EQ(parsed.ref_raw_data().size(), 0u);
  EXPECT_EQ(parsed.ref_name(), "raw_read_skip");
  EXPECT_EQ(parsed.ref_data_type(), TensorProto::DataType::FLOAT);
}

TEST(onnx_field_serialization, ReadFieldLimitParallel_SkipRawData_BelowThresholdKept) {
  // raw_data below the read threshold is retained even with skip_raw_data.
  TensorProto tensor = MakeRawDataTensor("raw_read_keep", 8);
  std::string serialized;
  SerializeOptions sopts;
  tensor.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.skip_raw_data = true;
  popts.raw_data_threshold = 1024;
  parsed.ParseFromString(serialized, popts);
  ASSERT_EQ(parsed.ref_raw_data().size(), 8u);
  EXPECT_EQ(std::memcmp(parsed.ref_raw_data().data(), tensor.ref_raw_data().data(), 8), 0);
}
