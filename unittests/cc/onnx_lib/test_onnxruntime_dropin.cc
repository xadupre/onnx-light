// Regression tests for the onnx-light drop-in fixes that let onnxruntime build
// against onnx-light instead of onnx/protobuf:
//
//   * The flat protobuf-style enum aliases (TensorProto_DataType, ...) must be
//     reachable through the canonical <onnx_lib/common/onnx_pb.h> header (the
//     one the <onnx/onnx_pb.h> compatibility shim forwards to). onnxruntime
//     headers reference ONNX_NAMESPACE::TensorProto_DataType and expect them.
//   * Parsing must still round-trip after wire_type (uint64_t) is narrowed to
//     the int parameter expected by the read_* helpers in the READ_* macros.
//   * ISchemaRegistry::GetSchema takes `const int maxInclusiveVersion`, matching
//     upstream onnx so onnxruntime's IOnnxRuntimeOpSchemaCollection override
//     (which uses `const int`) lines up without a virtual-override mismatch.
#include "onnx_lib/common/onnx_pb.h"

#include "onnx_lib/defs/schema.h"

#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE;

// ---------------------------------------------------------------------------
// Flat enum aliases are visible through the canonical onnx_pb.h include chain.
// ---------------------------------------------------------------------------

// These are the exact aliases onnxruntime relies on. If onnx_alias.h stops
// being pulled in by common/onnx_pb.h these static_asserts fail to compile,
// mirroring the "'TensorProto_DataType' is not a member of 'onnx_light'" error
// that onnxruntime hit before the fix.
static_assert(std::is_same_v<TensorProto_DataType, TensorProto::DataType>,
              "TensorProto_DataType alias must be reachable via onnx_pb.h");
static_assert(std::is_same_v<TensorShapeProto_Dimension, TensorShapeProto::Dimension>,
              "TensorShapeProto_Dimension alias must be reachable via onnx_pb.h");
static_assert(std::is_same_v<TypeProto_Tensor, TypeProto::Tensor>,
              "TypeProto_Tensor alias must be reachable via onnx_pb.h");
static_assert(std::is_same_v<TypeProto_Sequence, TypeProto::Sequence>,
              "TypeProto_Sequence alias must be reachable via onnx_pb.h");

TEST(onnxruntime_dropin, FlatDataTypeAliasMatchesScopedEnum) {
  TensorProto_DataType dt = TensorProto_DataType_FLOAT;
  EXPECT_EQ(dt, TensorProto::DataType::FLOAT);
  EXPECT_EQ(TensorProto_DataType_INT64, TensorProto::DataType::INT64);
  EXPECT_EQ(TensorProto_DataType_UNDEFINED, TensorProto::DataType::UNDEFINED);
}

// ---------------------------------------------------------------------------
// Parsing still round-trips after the wire_type narrowing casts in the READ_*
// macros (repeated / enum / plain / optional-proto / raw_data fields).
// ---------------------------------------------------------------------------

TEST(onnxruntime_dropin, TensorProtoRoundTripAcrossFieldKinds) {
  TensorProto t;
  t.set_name("weights");                         // plain string field
  t.set_data_type(TensorProto::DataType::FLOAT); // enum field
  t.add_dims(2);                                 // packed repeated field
  t.add_dims(3);
  t.ref_raw_data() = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}; // raw_data path

  std::string blob;
  ASSERT_TRUE(t.SerializeToString(blob));
  ASSERT_FALSE(blob.empty());
  EXPECT_EQ(blob.size(), t.ByteSizeLong());

  TensorProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(blob));
  EXPECT_EQ(parsed.ref_name(), "weights");
  EXPECT_EQ(parsed.ref_data_type(), TensorProto::DataType::FLOAT);
  ASSERT_EQ(parsed.ref_dims().size(), 2u);
  EXPECT_EQ(parsed.ref_dims()[0], 2);
  EXPECT_EQ(parsed.ref_dims()[1], 3);
  ASSERT_EQ(parsed.ref_raw_data().size(), 8u);
  EXPECT_EQ(std::memcmp(parsed.ref_raw_data().data(), t.ref_raw_data().data(), 8), 0);
}

// ---------------------------------------------------------------------------
// ISchemaRegistry::GetSchema(const std::string&, const int, const std::string&)
// can be overridden with the same `const int` signature onnxruntime uses.
// ---------------------------------------------------------------------------

namespace {

// Mirrors onnxruntime::IOnnxRuntimeOpSchemaCollection: overrides the pure
// virtual GetSchema with a `const int maxInclusiveVersion` parameter and marks
// it `override`, which only compiles if the base signature matches.
class CountingSchemaRegistry : public ISchemaRegistry {
public:
  using ISchemaRegistry::GetSchema;

  const OpSchema *GetSchema(const std::string &key, const int maxInclusiveVersion,
                            const std::string &domain) const override {
    last_key_ = key;
    last_version_ = maxInclusiveVersion;
    last_domain_ = domain;
    ++calls_;
    return nullptr;
  }

  mutable std::string last_key_;
  mutable int last_version_ = -1;
  mutable std::string last_domain_;
  mutable int calls_ = 0;
};

} // namespace

TEST(onnxruntime_dropin, SchemaRegistryConstIntOverrideDispatches) {
  CountingSchemaRegistry registry;
  const ISchemaRegistry &base = registry;

  EXPECT_EQ(base.GetSchema("Conv", 13, "ai.onnx"), nullptr);
  EXPECT_EQ(registry.calls_, 1);
  EXPECT_EQ(registry.last_key_, "Conv");
  EXPECT_EQ(registry.last_version_, 13);
  EXPECT_EQ(registry.last_domain_, "ai.onnx");

  // The String / RefString convenience overloads forward to the std::string one.
  EXPECT_EQ(
      base.GetSchema(utils::String(std::string("Gemm")), 11, utils::String(std::string("ai.onnx"))),
      nullptr);
  EXPECT_EQ(registry.calls_, 2);
  EXPECT_EQ(registry.last_key_, "Gemm");
  EXPECT_EQ(registry.last_version_, 11);
}

// ---------------------------------------------------------------------------
// TensorProto_DataType_IsValid: onnxruntime calls the flat protobuf-style name
// and expects real range checking (only the declared enumerators are valid).
// It is a set of free functions, not a macro, so that a class member with the
// same name (onnxruntime's provider bridge) still compiles.
// ---------------------------------------------------------------------------

static_assert(TensorProto_DataType_IsValid(TensorProto_DataType_FLOAT),
              "declared enumerators must be valid");
static_assert(!TensorProto_DataType_IsValid(-100), "out-of-range values must be invalid");

namespace {

// Mirrors onnxruntime's provider bridge, which declares a member function named
// TensorProto_DataType_IsValid. This only compiles if the flat name is not a
// function-like macro.
struct ProviderHostLike {
  bool TensorProto_DataType_IsValid(int value) const {
    return ONNX_LIGHT_NAMESPACE::TensorProto_DataType_IsValid(value);
  }
};

} // namespace

TEST(onnxruntime_dropin, DataTypeIsValidAcceptsDeclaredEnumerators) {
  for (int i = static_cast<int>(TensorProto::DataType::UNDEFINED);
       i <= static_cast<int>(TensorProto::DataType::FLOAT6E3M2); ++i) {
    EXPECT_TRUE(TensorProto::DataType_IsValid(i)) << "value " << i << " must be valid";
    EXPECT_TRUE(TensorProto_DataType_IsValid(i)) << "value " << i << " must be valid";
  }
  EXPECT_TRUE(
      TensorProto_DataType_IsValid(static_cast<TensorProto::DataType>(TensorProto_DataType_INT64)));
}

TEST(onnxruntime_dropin, DataTypeIsValidRejectsOutOfRangeValues) {
  const int first_invalid = static_cast<int>(TensorProto::DataType::FLOAT6E3M2) + 1;
  for (int value : {-100, -1, first_invalid, first_invalid + 1, 1000}) {
    EXPECT_FALSE(TensorProto::DataType_IsValid(value)) << "value " << value << " must be invalid";
    EXPECT_FALSE(TensorProto_DataType_IsValid(value)) << "value " << value << " must be invalid";
  }
}

TEST(onnxruntime_dropin, DataTypeIsValidUsableAsMemberFunctionName) {
  ProviderHostLike host;
  EXPECT_TRUE(host.TensorProto_DataType_IsValid(TensorProto_DataType_FLOAT));
  EXPECT_FALSE(host.TensorProto_DataType_IsValid(-100));
}
