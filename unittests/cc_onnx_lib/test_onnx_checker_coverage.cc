// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Additional C++ tests for onnx_light/onnx_lib/checker.cc to cover branches
// that the upstream-mirrored checker_test.cc does not exercise: validation of
// ValueInfo / Tensor / SparseTensor / Sequence / Map / Optional / Attribute /
// Node / Graph / Model protos as well as cycle detection in model-local
// functions and the experimental-op classifier.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "onnx_lib/checker.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

#ifndef ONNX_NO_EXCEPTIONS

namespace {

using checker::CheckerContext;
using checker::LexicalScopeContext;
using checker::ValidationError;

// Helper: build a default CheckerContext for the current IR version with an
// ONNX opset import so that node/graph checks have a domain to resolve.
CheckerContext MakeCtx(int ir_version = IR_VERSION) {
  CheckerContext ctx;
  ctx.set_ir_version(ir_version);
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  ctx.set_opset_imports(opsets);
  return ctx;
}

// Helper: build a minimal valid TensorProto holding a single float scalar.
TensorProto MakeFloatScalar(const std::string &name, float value) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.add_float_data(value);
  return t;
}

} // namespace

// ---------------------------------------------------------------------------
// check_value_info
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, ValueInfoEmptyNameRejected) {
  ValueInfoProto vi;
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoMissingTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorOK) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  tt->mutable_shape(); // empty shape (scalar) is fine
  EXPECT_NO_THROW(checker::check_value_info(vi, MakeCtx()));
}

TEST(CHECKER_COVERAGE, ValueInfoTensorNegativeElemTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(static_cast<TensorProto::DataType>(-1));
  tt->mutable_shape();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorElemTypeAboveLimitRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(static_cast<TensorProto::DataType>(2049));
  tt->mutable_shape();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoTensorMissingShapeRejected) {
  ValueInfoProto vi;
  vi.set_name("x");
  auto *tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  // No mutable_shape() call -> has_shape() is false.
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoSubgraphRelaxed) {
  // For a non-main graph, only the name is required.
  ValueInfoProto vi;
  vi.set_name("x");
  CheckerContext ctx = MakeCtx();
  ctx.set_is_main_graph(false);
  EXPECT_NO_THROW(checker::check_value_info(vi, ctx));
}

TEST(CHECKER_COVERAGE, ValueInfoSequenceMissingElemTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("seq");
  vi.mutable_type()->mutable_sequence_type();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoOptionalOK) {
  ValueInfoProto vi;
  vi.set_name("opt");
  auto *opt = vi.mutable_type()->mutable_optional_type();
  auto *inner_tt = opt->mutable_elem_type()->mutable_tensor_type();
  inner_tt->set_elem_type(TensorProto::FLOAT);
  inner_tt->mutable_shape();
  EXPECT_NO_THROW(checker::check_value_info(vi, MakeCtx()));
}

TEST(CHECKER_COVERAGE, ValueInfoMapMissingKeyTypeRejected) {
  ValueInfoProto vi;
  vi.set_name("m");
  // mutable_map_type() with no key_type / value_type set
  vi.mutable_type()->mutable_map_type();
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, ValueInfoSparseTensorMissingShapeRejected) {
  ValueInfoProto vi;
  vi.set_name("s");
  auto *st = vi.mutable_type()->mutable_sparse_tensor_type();
  st->set_elem_type(TensorProto::FLOAT);
  EXPECT_THROW(checker::check_value_info(vi, MakeCtx()), ValidationError);
}

// ---------------------------------------------------------------------------
// check_tensor
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, TensorUndefinedDataTypeRejected) {
  TensorProto t;
  // Default-constructed TensorProto has data_type==UNDEFINED.
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorFloatOK) {
  TensorProto t = MakeFloatScalar("ok", 1.0f);
  EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
}

TEST(CHECKER_COVERAGE, TensorRawDataOK) {
  TensorProto t;
  t.set_name("raw");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  float v = 1.5f;
  std::vector<uint8_t> bytes(sizeof(v));
  std::memcpy(bytes.data(), &v, sizeof(v));
  for (auto b : bytes) {
    t.ref_raw_data().push_back(b);
  }
  EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
}

TEST(CHECKER_COVERAGE, TensorMultipleValueFieldsRejected) {
  TensorProto t;
  t.set_name("multi");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.add_float_data(1.0f);
  t.add_int32_data(2);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorZeroElementsWithDataRejected) {
  TensorProto t;
  t.set_name("z");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(uint64_t{0});
  t.add_float_data(1.0f);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorWrongFieldForDataTypeRejected) {
  // INT32 dtype but values stored in float_data field.
  TensorProto t;
  t.set_name("badfield");
  t.set_data_type(TensorProto::INT32);
  t.add_dims(1);
  t.add_float_data(1.0f);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorStringInRawDataRejected) {
  TensorProto t;
  t.set_name("s");
  t.set_data_type(TensorProto::STRING);
  t.add_dims(1);
  t.ref_raw_data().push_back('x');
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalMissingLocationRejected) {
  TensorProto t;
  t.set_name("ext");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  // No "location" entry in external_data.
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalWithEmbeddedDataRejected) {
  TensorProto t;
  t.set_name("ext_with_data");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  t.add_float_data(1.0f);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value("weights.bin");
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorExternalInvalidLocationRejected) {
  TensorProto t;
  t.set_name("ext_bad_loc");
  t.set_data_type(TensorProto::FLOAT);
  t.add_dims(1);
  t.set_data_location(TensorProto::EXTERNAL);
  auto *e = t.add_external_data();
  e->set_key("location");
  e->set_value(".."); // rejected by resolve_external_data_location
  CheckerContext ctx = MakeCtx();
  ctx.set_model_dir("localfolder");
  EXPECT_THROW(checker::check_tensor(t, ctx), ValidationError);
}

TEST(CHECKER_COVERAGE, TensorUnrecognizedDataTypeRejected) {
  TensorProto t;
  t.set_name("bogus");
  // Pick a value not part of the switch in check_tensor (and not UNDEFINED).
  t.set_data_type(static_cast<TensorProto::DataType>(1234));
  t.add_dims(1);
  t.add_int32_data(1);
  EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);
}

// Reject packed sub-byte tensors whose raw_data payload is too small.
TEST(CHECKER_COVERAGE, TensorPackedSubByteRawDataTooSmall) {
  // 4-bit types: 2 elements per byte.
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    for (int i = 0; i < 4; ++i) {
      t.ref_raw_data().push_back('\0'); // 1 byte too short (need 5)
    }
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    for (int i = 0; i < 5; ++i) {
      ok.ref_raw_data().push_back('\0'); // ceil(10/2) = 5
    }
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }

  // 2-bit types: 4 elements per byte.
  for (TensorProto::DataType dtype : {TensorProto::INT2, TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    for (int i = 0; i < 2; ++i) {
      t.ref_raw_data().push_back('\0'); // 1 byte too short (need 3)
    }
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    for (int i = 0; i < 3; ++i) {
      ok.ref_raw_data().push_back('\0'); // ceil(10/4) = 3
    }
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }
}

// Reject packed sub-byte tensors whose int32_data payload is too small.
TEST(CHECKER_COVERAGE, TensorPackedSubByteInt32DataTooSmall) {
  // 4-bit types: 8 elements per int32.
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(10);
    t.add_int32_data(0); // 1 int32, need 2
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    ok.add_int32_data(0);
    ok.add_int32_data(0); // ceil(10/8) = 2
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }

  // 2-bit types: 16 elements per int32.
  for (TensorProto::DataType dtype : {TensorProto::INT2, TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(20);
    t.add_int32_data(0); // 1 int32, need 2
    EXPECT_THROW(checker::check_tensor(t, MakeCtx()), ValidationError);

    TensorProto ok;
    ok.set_name("t");
    ok.set_data_type(dtype);
    ok.add_dims(20);
    ok.add_int32_data(0);
    ok.add_int32_data(0); // ceil(20/16) = 2
    EXPECT_NO_THROW(checker::check_tensor(ok, MakeCtx()));
  }
}

// Zero-element packed tensors with empty payload must be valid.
TEST(CHECKER_COVERAGE, TensorPackedSubByteZeroElems) {
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1, TensorProto::INT2,
        TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("t");
    t.set_data_type(dtype);
    t.add_dims(uint64_t{0});
    EXPECT_NO_THROW(checker::check_tensor(t, MakeCtx()));
  }
}

// ---------------------------------------------------------------------------
// check_sparse_tensor
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, SparseTensorDefaultRejected) {
  // Default-constructed SparseTensorProto: values has rank 0 but the checker
  // requires rank 1 — exercises the "must have rank 1" failure branch.
  SparseTensorProto s;
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorValuesRankNot1Rejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  v->add_float_data(3.0f);
  v->add_float_data(4.0f);
  s.add_dims(4);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorZeroDenseRankRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(uint64_t{0});
  // No dense dims => rank 0.
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorNonPositiveDimRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(uint64_t{0});
  s.add_dims(int64_t{0}); // not > 0
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorNonEmptyNoIndicesRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesWrongDtypeRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT32); // must be INT64
  idx->add_dims(2);
  idx->add_int32_data(0);
  idx->add_int32_data(1);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1OK) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_int64_data(0);
  idx->add_int64_data(2);
  EXPECT_NO_THROW(checker::check_sparse_tensor(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1OutOfRangeRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_int64_data(100); // dense_size is 4
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank1UnsortedRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_int64_data(3);
  idx->add_int64_data(1);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesNnzMismatchRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(4);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1); // mismatched with NNZ=2
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank2OK) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(2);
  v->add_float_data(1.0f);
  v->add_float_data(2.0f);
  s.add_dims(2);
  s.add_dims(3);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(2);
  idx->add_dims(2);
  // [[0,0],[1,2]]
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  idx->add_int64_data(1);
  idx->add_int64_data(2);
  EXPECT_NO_THROW(checker::check_sparse_tensor(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank2RankMismatchRejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(2);
  s.add_dims(3);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_dims(3); // dense_rank is 2, not 3
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SparseTensorIndicesRank3Rejected) {
  SparseTensorProto s;
  auto *v = &s.ref_values();
  v->set_name("vals");
  v->set_data_type(TensorProto::FLOAT);
  v->add_dims(1);
  v->add_float_data(1.0f);
  s.add_dims(2);
  auto *idx = &s.ref_indices();
  idx->set_name("idx");
  idx->set_data_type(TensorProto::INT64);
  idx->add_dims(1);
  idx->add_dims(1);
  idx->add_dims(1);
  idx->add_int64_data(0);
  EXPECT_THROW(checker::check_sparse_tensor(s, MakeCtx()), ValidationError);
}

// ---------------------------------------------------------------------------
// check_sequence / check_optional / check_map
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, SequenceMissingElemTypeRejected) {
  SequenceProto s;
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SequenceUndefinedElemTypeRejected) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::UNDEFINED);
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, SequenceTensorOK) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::TENSOR);
  *s.add_tensor_values() = MakeFloatScalar("v0", 1.0f);
  EXPECT_NO_THROW(checker::check_sequence(s, MakeCtx()));
}

TEST(CHECKER_COVERAGE, SequenceTensorInvalidElementRejected) {
  SequenceProto s;
  s.set_elem_type(SequenceProto::TENSOR);
  auto *bad = s.add_tensor_values();
  bad->set_data_type(TensorProto::UNDEFINED); // triggers check_tensor failure
  EXPECT_THROW(checker::check_sequence(s, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, OptionalDefaultUndefinedAccepted) {
  // Default OptionalProto has elem_type=UNDEFINED which is allowed (returns
  // early without further checks).
  OptionalProto o;
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalUndefinedAccepted) {
  OptionalProto o;
  o.set_elem_type(OptionalProto::UNDEFINED);
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalTensorOK) {
  OptionalProto o;
  o.set_elem_type(OptionalProto::TENSOR);
  *o.mutable_tensor_value() = MakeFloatScalar("v", 1.0f);
  EXPECT_NO_THROW(checker::check_optional(o, MakeCtx()));
}

TEST(CHECKER_COVERAGE, OptionalInvalidElemTypeRejected) {
  OptionalProto o;
  o.set_elem_type(static_cast<OptionalProto::DataType>(999));
  EXPECT_THROW(checker::check_optional(o, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapDefaultRejected) {
  // Default-constructed MapProto has key_type==UNDEFINED.
  MapProto m;
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapInvalidKeyTypeRejected) {
  MapProto m;
  m.set_key_type(TensorProto::FLOAT); // disallowed for map keys
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapBothKeyVariantsRejected) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  *m.add_string_keys() = utils::String("k");
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  *values->add_tensor_values() = MakeFloatScalar("v", 1.0f);
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapKeyValueLengthMismatchRejected) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  // 0 values, 1 key -> mismatch
  EXPECT_THROW(checker::check_map(m, MakeCtx()), ValidationError);
}

TEST(CHECKER_COVERAGE, MapInt64KeysOK) {
  MapProto m;
  m.set_key_type(TensorProto::INT64);
  m.add_keys(1);
  m.add_keys(2);
  auto *values = &m.ref_values();
  values->set_elem_type(SequenceProto::TENSOR);
  *values->add_tensor_values() = MakeFloatScalar("a", 1.0f);
  *values->add_tensor_values() = MakeFloatScalar("b", 2.0f);
  EXPECT_NO_THROW(checker::check_map(m, MakeCtx()));
}

// ---------------------------------------------------------------------------
// check_attribute
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, AttributeEmptyNameRejected) {
  AttributeProto attr;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

// Note: AttributeProto.type is a required FIELD that is default-initialised
// to UNDEFINED, so the "type required" branch in check_attribute is not
// reachable from valid proto construction in onnx-light. We therefore only
// test the "type field and data field mismatch" branch below.

TEST(CHECKER_COVERAGE, AttributeTypeFieldMismatchRejected) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::INT); // says INT
  attr.set_f(1.0f);                   // but stores a float
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeMultipleValueFieldsRejected) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_f(1.0f);
  attr.set_i(1);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeIntOK) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::INT);
  attr.set_i(7);
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_attribute(attr, MakeCtx(), lex));
}

TEST(CHECKER_COVERAGE, AttributeFunctionBodyRefWithValueRejected) {
  // Attributes of nodes inside function bodies that have ref_attr_name set
  // must not also have a value field set.
  AttributeProto attr;
  attr.set_name("a");
  attr.set_ref_attr_name("outer");
  attr.set_type(AttributeProto::INT);
  attr.set_i(1);
  LexicalScopeContext lex;
  CheckerContext ctx = MakeCtx();
  ctx.set_is_main_graph(false);
  EXPECT_THROW(checker::check_attribute(attr, ctx, lex), ValidationError);
}

TEST(CHECKER_COVERAGE, AttributeTensorPropagatesError) {
  AttributeProto attr;
  attr.set_name("a");
  attr.set_type(AttributeProto::TENSOR);
  attr.mutable_t()->set_data_type(TensorProto::UNDEFINED); // invalid
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_attribute(attr, MakeCtx(), lex), ValidationError);
}

// ---------------------------------------------------------------------------
// check_node
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, NodeEmptyOpTypeRejected) {
  NodeProto node;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeNoInputsOrOutputsRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeMissingOpsetImportRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeDuplicateAttributeNamesRejected) {
  NodeProto node;
  node.set_op_type("Identity");
  *node.add_input() = "x";
  *node.add_output() = "y";
  auto *a1 = node.add_attribute();
  a1->set_name("dup");
  a1->set_type(AttributeProto::INT);
  a1->set_i(1);
  auto *a2 = node.add_attribute();
  a2->set_name("dup");
  a2->set_type(AttributeProto::INT);
  a2->set_i(2);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeUnknownOnnxOpRejected) {
  // Built-in domain with a name that has no registered schema must fail.
  NodeProto node;
  node.set_op_type("ThisOpDoesNotExist_xyz");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeUnknownCustomDomainAcceptedByDefault) {
  // Unknown ops in unknown domains are accepted unless check_custom_domain.
  NodeProto node;
  node.set_op_type("MyOp");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  CheckerContext ctx = MakeCtx();
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  opsets["custom.domain"] = 1;
  ctx.set_opset_imports(opsets);
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_node(node, ctx, lex));
}

TEST(CHECKER_COVERAGE, NodeUnknownCustomDomainRejectedWhenStrict) {
  NodeProto node;
  node.set_op_type("MyOp");
  node.set_domain("custom.domain");
  *node.add_input() = "x";
  *node.add_output() = "y";
  CheckerContext ctx = MakeCtx();
  std::unordered_map<std::string, int> opsets;
  opsets[""] = 21;
  opsets["custom.domain"] = 1;
  ctx.set_opset_imports(opsets);
  ctx.set_check_custom_domain(true);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_node(node, ctx, lex), ValidationError);
}

TEST(CHECKER_COVERAGE, NodeExperimentalOpSkipsSchemaLookup) {
  // Experimental ops short-circuit and must not require a registered schema.
  NodeProto node;
  node.set_op_type("ATen");
  *node.add_input() = "x";
  *node.add_output() = "y";
  LexicalScopeContext lex;
  EXPECT_NO_THROW(checker::check_node(node, MakeCtx(), lex));
}

// ---------------------------------------------------------------------------
// check_graph
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, GraphEmptyNameRejected) {
  GraphProto g;
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphUnconnectedOutputRejected) {
  GraphProto g;
  g.set_name("g");
  auto *in = g.add_input();
  in->set_name("x");
  auto *ttin = in->mutable_type()->mutable_tensor_type();
  ttin->set_elem_type(TensorProto::FLOAT);
  ttin->mutable_shape();
  // Output 'y' is not produced by any node nor declared as input.
  auto *out = g.add_output();
  out->set_name("y");
  auto *ttout = out->mutable_type()->mutable_tensor_type();
  ttout->set_elem_type(TensorProto::FLOAT);
  ttout->mutable_shape();
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphDuplicateInputNamesRejected) {
  GraphProto g;
  g.set_name("g");
  for (int i = 0; i < 2; ++i) {
    auto *in = g.add_input();
    in->set_name("x"); // duplicate
    auto *tt = in->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);
    tt->mutable_shape();
  }
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphDuplicateInitializerNamesRejected) {
  GraphProto g;
  g.set_name("g");
  for (int i = 0; i < 2; ++i) {
    *g.add_initializer() = MakeFloatScalar("w", 1.0f);
  }
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphInitializerNotInInputForOldIRRejected) {
  // For ir_version <= 3, every initializer must also be an input.
  GraphProto g;
  g.set_name("g");
  *g.add_initializer() = MakeFloatScalar("w", 1.0f);
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(3), lex), ValidationError);
}

TEST(CHECKER_COVERAGE, GraphTopologicallyUnsortedNodeRejected) {
  // Node consumes 'z' which is never defined as input/initializer/prior output.
  GraphProto g;
  g.set_name("g");
  auto *node = g.add_node();
  node->set_op_type("Identity");
  *node->add_input() = "z";
  *node->add_output() = "y";
  // Declare an output so the graph can otherwise be valid.
  auto *out = g.add_output();
  out->set_name("y");
  auto *tt = out->mutable_type()->mutable_tensor_type();
  tt->set_elem_type(TensorProto::FLOAT);
  tt->mutable_shape();
  LexicalScopeContext lex;
  EXPECT_THROW(checker::check_graph(g, MakeCtx(), lex), ValidationError);
}

// ---------------------------------------------------------------------------
// check_function_call_cycles
// ---------------------------------------------------------------------------

namespace {

// Add a callee node referring to function (local_domain::name) to function `f`.
void AddCalleeNode(FunctionProto &f, const std::string &domain, const std::string &op_type) {
  auto *n = f.add_node();
  n->set_op_type(op_type);
  n->set_domain(domain);
}

void InitFunction(FunctionProto &f, const std::string &name) {
  f.set_name(name);
  f.set_domain("local");
}

} // namespace

TEST(CHECKER_COVERAGE, FunctionCallCyclesAcyclicOK) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f1");
  AddCalleeNode(*f1, "local", "f2");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f2");
  EXPECT_NO_THROW(checker::check_function_call_cycles(model));
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesSelfLoopRejected) {
  ModelProto model;
  auto *f = model.add_functions();
  InitFunction(*f, "f");
  AddCalleeNode(*f, "local", "f"); // self-reference
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesMutualRejected) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f1");
  AddCalleeNode(*f1, "local", "f2");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f2");
  AddCalleeNode(*f2, "local", "f1");
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

TEST(CHECKER_COVERAGE, FunctionCallCyclesDuplicateImplIdRejected) {
  ModelProto model;
  auto *f1 = model.add_functions();
  InitFunction(*f1, "f");
  auto *f2 = model.add_functions();
  InitFunction(*f2, "f"); // same implementation id (domain + name)
  EXPECT_THROW(checker::check_function_call_cycles(model), ValidationError);
}

// ---------------------------------------------------------------------------
// check_model
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, ModelMissingIrVersionRejected) {
  ModelProto m;
  m.set_ir_version(0); // explicit "unset" sentinel for the checker
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelIrVersionTooHighRejected) {
  ModelProto m;
  m.set_ir_version(IR_VERSION + 1);
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelMissingOpsetImportRejected) {
  ModelProto m;
  m.set_ir_version(7);
  m.mutable_graph()->set_name("g");
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelDuplicateMetadataKeysRejected) {
  ModelProto m;
  m.set_ir_version(7);
  auto *o = m.add_opset_import();
  o->set_domain("");
  o->set_version(21);
  m.mutable_graph()->set_name("g");
  for (int i = 0; i < 2; ++i) {
    auto *kv = m.add_metadata_props();
    kv->set_key("same_key");
    kv->set_value("v");
  }
  EXPECT_THROW(checker::check_model(m), ValidationError);
}

TEST(CHECKER_COVERAGE, ModelMinimalValidAccepted) {
  ModelProto m;
  m.set_ir_version(7);
  auto *o = m.add_opset_import();
  o->set_domain("");
  o->set_version(21);
  auto *g = m.mutable_graph();
  g->set_name("g");
  EXPECT_NO_THROW(checker::check_model(m));
}

// ---------------------------------------------------------------------------
// check_is_experimental_op
// ---------------------------------------------------------------------------

TEST(CHECKER_COVERAGE, IsExperimentalOpTrueForAtenInDefaultDomain) {
  NodeProto node;
  node.set_op_type("ATen");
  EXPECT_TRUE(checker::check_is_experimental_op(node));
}

TEST(CHECKER_COVERAGE, IsExperimentalOpFalseForCustomDomain) {
  NodeProto node;
  node.set_op_type("ATen");
  node.set_domain("custom");
  EXPECT_FALSE(checker::check_is_experimental_op(node));
}

TEST(CHECKER_COVERAGE, IsExperimentalOpFalseForNonExperimentalOp) {
  NodeProto node;
  node.set_op_type("Add");
  EXPECT_FALSE(checker::check_is_experimental_op(node));
}

#endif // ONNX_NO_EXCEPTIONS

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
