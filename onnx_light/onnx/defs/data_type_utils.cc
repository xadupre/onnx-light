// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "data_type_utils.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ONNX_NAMESPACE {
namespace Utils {
namespace {

// Singleton wrapper around allowed data types.
// This implements construct on first use which is needed to ensure
// static objects are initialized before use.
class TypesWrapper final {
public:
  static TypesWrapper &GetTypesWrapper();

  std::unordered_set<std::string> &GetAllowedDataTypes();
  std::unordered_map<std::string, int32_t> &TypeStrToTensorDataType();
  std::unordered_map<int32_t, std::string> &TensorDataTypeToTypeStr();

  ~TypesWrapper() = default;
  TypesWrapper(const TypesWrapper &) = delete;
  void operator=(const TypesWrapper &) = delete;

private:
  TypesWrapper();

  std::unordered_map<std::string, int32_t> type_str_to_tensor_data_type_;
  std::unordered_map<int32_t, std::string> tensor_data_type_to_type_str_;
  std::unordered_set<std::string> allowed_data_types_;
};

// Simple class which contains pointers to an external string buffer and a size.
class StringRange final {
public:
  StringRange(const char *data, size_t size);
  // NOLINTNEXTLINE(google-explicit-constructor)
  StringRange(const std::string &str);
  // NOLINTNEXTLINE(google-explicit-constructor)
  StringRange(const char *data);
  const char *Data() const;
  size_t Size() const;
  [[maybe_unused]] bool Empty() const;
  bool StartsWith(const StringRange &str) const;
  bool EndsWith(const StringRange &str) const;
  bool LStrip();
  bool LStrip(size_t size);
  bool LStrip(StringRange str);
  bool RStrip();
  bool RStrip(size_t size);
  bool RStrip(StringRange str);
  bool LAndRStrip();
  void ParensWhitespaceStrip();
  size_t Find(char ch) const;

private:
  const char *data_;
  size_t size_;
};

} // namespace

std::unordered_map<std::string, TypeProto> &DataTypeUtils::GetTypeStrToProtoMap() {
  static std::unordered_map<std::string, TypeProto> map;
  return map;
}

std::mutex &DataTypeUtils::GetTypeStrLock() {
  static std::mutex lock;
  return lock;
}

DataType DataTypeUtils::ToType(const TypeProto &type_proto) {
  auto typeStr = ToString(type_proto);
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(typeStr);
  if (it == GetTypeStrToProtoMap().end()) {
    TypeProto type;
    FromString(typeStr, type);
    GetTypeStrToProtoMap()[typeStr] = type;
    it = GetTypeStrToProtoMap().find(typeStr);
  }
  return &(it->first);
}

DataType DataTypeUtils::ToType(const std::string &type_str) {
  TypeProto type;
  FromString(type_str, type);
  return ToType(type);
}

const TypeProto &DataTypeUtils::ToTypeProto(const DataType &data_type) {
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(*data_type);
  if (GetTypeStrToProtoMap().end() == it) {
    ONNX_THROW_EX(std::invalid_argument("Invalid data type " + *data_type));
  }
  return it->second;
}

std::string DataTypeUtils::ToString(const TypeProto &type_proto, const std::string &left,
                                    const std::string &right) {
  if (type_proto.has_tensor_type()) {
    const auto &ttype = type_proto.ref_tensor_type();
    return left + "tensor(" + ToDataTypeString(static_cast<int32_t>(ttype.ref_elem_type())) + ")" +
           right;
  }
  if (type_proto.has_sequence_type()) {
    return ToString(type_proto.ref_sequence_type().ref_elem_type(), left + "seq(", ")" + right);
  }
  if (type_proto.has_optional_type()) {
    return ToString(type_proto.ref_optional_type().ref_elem_type(), left + "optional(",
                    ")" + right);
  }
  if (type_proto.has_map_type()) {
    const auto &mtype = type_proto.ref_map_type();
    std::string map_str = "map(" + ToDataTypeString(mtype.ref_key_type()) + ",";
    return ToString(mtype.ref_value_type(), left + map_str, ")" + right);
  }
  if (type_proto.has_sparse_tensor_type()) {
    const auto &stype = type_proto.ref_sparse_tensor_type();
    return left + "sparse_tensor(" + ToDataTypeString(static_cast<int32_t>(stype.ref_elem_type())) +
           ")" + right;
  }
  ONNX_THROW_EX(std::invalid_argument("Unsupported type proto: no type set"));
}

std::string DataTypeUtils::ToDataTypeString(int32_t tensor_data_type) {
  TypesWrapper &t = TypesWrapper::GetTypesWrapper();
  auto iter = t.TensorDataTypeToTypeStr().find(tensor_data_type);
  if (t.TensorDataTypeToTypeStr().end() == iter) {
    ONNX_THROW_EX(std::invalid_argument("Invalid tensor data type " +
                                        std::to_string(tensor_data_type) + "."));
  }
  return iter->second;
}

void DataTypeUtils::FromString(const std::string &type_str, TypeProto &type_proto) {
  StringRange s(type_str);
  type_proto = TypeProto{};
  if (s.LStrip("seq")) {
    s.ParensWhitespaceStrip();
    TypeProto elem;
    FromString(std::string(s.Data(), s.Size()), elem);
    type_proto.ref_sequence_type().ref_elem_type().CopyFrom(elem);
    return;
  }
  if (s.LStrip("optional")) {
    s.ParensWhitespaceStrip();
    TypeProto elem;
    FromString(std::string(s.Data(), s.Size()), elem);
    type_proto.ref_optional_type().ref_elem_type().CopyFrom(elem);
    return;
  }
  if (s.LStrip("map")) {
    s.ParensWhitespaceStrip();
    size_t key_size = s.Find(',');
    StringRange k(s.Data(), key_size);
    std::string key(k.Data(), k.Size());
    s.LStrip(key_size);
    s.LStrip(",");
    StringRange v(s.Data(), s.Size());
    int32_t key_type = FromDataTypeString(key);
    type_proto.ref_map_type().set_key_type(key_type);
    TypeProto val_type;
    FromString(std::string(v.Data(), v.Size()), val_type);
    type_proto.ref_map_type().ref_value_type().CopyFrom(val_type);
    return;
  }
  if (s.LStrip("sparse_tensor")) {
    s.ParensWhitespaceStrip();
    int32_t e = FromDataTypeString(std::string(s.Data(), s.Size()));
    type_proto.ref_sparse_tensor_type().set_elem_type(static_cast<TensorProto::DataType>(e));
  } else if (s.LStrip("tensor")) {
    s.ParensWhitespaceStrip();
    int32_t e = FromDataTypeString(std::string(s.Data(), s.Size()));
    type_proto.ref_tensor_type().set_elem_type(e);
    // Create shape with zero dimensions for scalar
    type_proto.ref_tensor_type().ref_shape();
  } else {
    // Scalar
    int32_t e = FromDataTypeString(std::string(s.Data(), s.Size()));
    type_proto.ref_tensor_type().set_elem_type(e);
    // Create shape with zero dimensions for scalar
    type_proto.ref_tensor_type().ref_shape();
  }
}

bool DataTypeUtils::IsValidDataTypeString(const std::string &type_str) {
  TypesWrapper &t = TypesWrapper::GetTypesWrapper();
  const auto &allowedSet = t.GetAllowedDataTypes();
  return (allowedSet.find(type_str) != allowedSet.end());
}

int32_t DataTypeUtils::FromDataTypeString(const std::string &type_str) {
  if (!IsValidDataTypeString(type_str)) {
    ONNX_THROW_EX(std::invalid_argument(
        "DataTypeUtils::FromDataTypeString - Received invalid data type string '" + type_str +
        "'."));
  }
  TypesWrapper &t = TypesWrapper::GetTypesWrapper();
  return t.TypeStrToTensorDataType()[type_str];
}

namespace {

StringRange::StringRange(const char *p_data, size_t p_size) : data_(p_data), size_(p_size) {
  assert(p_data != nullptr);
  LAndRStrip();
}

StringRange::StringRange(const std::string &p_str) : data_(p_str.data()), size_(p_str.size()) {
  LAndRStrip();
}

StringRange::StringRange(const char *p_data) : data_(p_data), size_(strlen(p_data)) {
  LAndRStrip();
}

const char *StringRange::Data() const { return data_; }

size_t StringRange::Size() const { return size_; }

bool StringRange::Empty() const { return size_ == 0; }

bool StringRange::StartsWith(const StringRange &str) const {
  return ((size_ >= str.size_) && (memcmp(data_, str.data_, str.size_) == 0));
}

bool StringRange::EndsWith(const StringRange &str) const {
  return ((size_ >= str.size_) && (memcmp(data_ + (size_ - str.size_), str.data_, str.size_) == 0));
}

bool StringRange::LStrip() {
  size_t count = 0;
  const char *ptr = data_;
  while (count < size_ && std::isspace(static_cast<unsigned char>(*ptr))) {
    count++;
    ptr++;
  }
  if (count > 0) {
    return LStrip(count);
  }
  return false;
}

bool StringRange::LStrip(size_t size) {
  if (size <= size_) {
    data_ += size;
    size_ -= size;
    return true;
  }
  return false;
}

bool StringRange::LStrip(StringRange str) {
  if (StartsWith(str)) {
    return LStrip(str.size_);
  }
  return false;
}

bool StringRange::RStrip() {
  size_t count = 0;
  const char *ptr = data_ + size_ - 1;
  while (count < size_ && std::isspace(static_cast<unsigned char>(*ptr))) {
    ++count;
    --ptr;
  }
  if (count > 0) {
    return RStrip(count);
  }
  return false;
}

bool StringRange::RStrip(size_t size) {
  if (size_ >= size) {
    size_ -= size;
    return true;
  }
  return false;
}

bool StringRange::RStrip(StringRange str) {
  if (EndsWith(str)) {
    return RStrip(str.size_);
  }
  return false;
}

bool StringRange::LAndRStrip() {
  bool l = LStrip();
  bool r = RStrip();
  return l || r;
}

void StringRange::ParensWhitespaceStrip() {
  LStrip();
  LStrip("(");
  LAndRStrip();
  RStrip(")");
  RStrip();
}

size_t StringRange::Find(const char ch) const {
  size_t idx = 0;
  while (idx < size_) {
    if (data_[idx] == ch) {
      return idx;
    }
    idx++;
  }
  return std::string::npos;
}

TypesWrapper &TypesWrapper::GetTypesWrapper() {
  static TypesWrapper types;
  return types;
}

std::unordered_set<std::string> &TypesWrapper::GetAllowedDataTypes() { return allowed_data_types_; }

std::unordered_map<std::string, int32_t> &TypesWrapper::TypeStrToTensorDataType() {
  return type_str_to_tensor_data_type_;
}

std::unordered_map<int32_t, std::string> &TypesWrapper::TensorDataTypeToTypeStr() {
  return tensor_data_type_to_type_str_;
}

TypesWrapper::TypesWrapper() {
  // DataType strings. These should match the DataTypes defined in onnx.proto.
  type_str_to_tensor_data_type_["float"] = static_cast<int32_t>(TensorProto::DataType::FLOAT);
  type_str_to_tensor_data_type_["float16"] = static_cast<int32_t>(TensorProto::DataType::FLOAT16);
  type_str_to_tensor_data_type_["bfloat16"] = static_cast<int32_t>(TensorProto::DataType::BFLOAT16);
  type_str_to_tensor_data_type_["double"] = static_cast<int32_t>(TensorProto::DataType::DOUBLE);
  type_str_to_tensor_data_type_["int8"] = static_cast<int32_t>(TensorProto::DataType::INT8);
  type_str_to_tensor_data_type_["int16"] = static_cast<int32_t>(TensorProto::DataType::INT16);
  type_str_to_tensor_data_type_["int32"] = static_cast<int32_t>(TensorProto::DataType::INT32);
  type_str_to_tensor_data_type_["int64"] = static_cast<int32_t>(TensorProto::DataType::INT64);
  type_str_to_tensor_data_type_["uint8"] = static_cast<int32_t>(TensorProto::DataType::UINT8);
  type_str_to_tensor_data_type_["uint16"] = static_cast<int32_t>(TensorProto::DataType::UINT16);
  type_str_to_tensor_data_type_["uint32"] = static_cast<int32_t>(TensorProto::DataType::UINT32);
  type_str_to_tensor_data_type_["uint64"] = static_cast<int32_t>(TensorProto::DataType::UINT64);
  type_str_to_tensor_data_type_["complex64"] =
      static_cast<int32_t>(TensorProto::DataType::COMPLEX64);
  type_str_to_tensor_data_type_["complex128"] =
      static_cast<int32_t>(TensorProto::DataType::COMPLEX128);
  type_str_to_tensor_data_type_["string"] = static_cast<int32_t>(TensorProto::DataType::STRING);
  type_str_to_tensor_data_type_["bool"] = static_cast<int32_t>(TensorProto::DataType::BOOL);
  type_str_to_tensor_data_type_["float8e4m3fn"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT8E4M3FN);
  type_str_to_tensor_data_type_["float8e4m3fnuz"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT8E4M3FNUZ);
  type_str_to_tensor_data_type_["float8e5m2"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT8E5M2);
  type_str_to_tensor_data_type_["float8e5m2fnuz"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT8E5M2FNUZ);
  type_str_to_tensor_data_type_["float8e8m0"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT8E8M0);
  type_str_to_tensor_data_type_["uint4"] = static_cast<int32_t>(TensorProto::DataType::UINT4);
  type_str_to_tensor_data_type_["int4"] = static_cast<int32_t>(TensorProto::DataType::INT4);
  type_str_to_tensor_data_type_["uint2"] = static_cast<int32_t>(TensorProto::DataType::UINT2);
  type_str_to_tensor_data_type_["int2"] = static_cast<int32_t>(TensorProto::DataType::INT2);
  type_str_to_tensor_data_type_["float4e2m1"] =
      static_cast<int32_t>(TensorProto::DataType::FLOAT4E2M1);

  for (auto &[type_str, data_type] : type_str_to_tensor_data_type_) {
    tensor_data_type_to_type_str_[data_type] = type_str;
    allowed_data_types_.insert(type_str);
  }
}

} // namespace

} // namespace Utils
} // namespace ONNX_NAMESPACE
