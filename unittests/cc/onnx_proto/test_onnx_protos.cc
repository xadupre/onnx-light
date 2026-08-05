#include "../common/proto_utils.h"
#include "../common/string_utils.h"
#include "blake3/blake3_hash.h"
#include "onnx.h"
#include "onnx_alias.h"
#include "onnx_helper.h"
#include "onnx_light_helpers.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>
#include <string_view>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE;

static_assert(std::string_view(TensorProto::DataType_Name(TensorProto::DataType::FLOAT)) ==
              "FLOAT");
static_assert(std::string_view(AttributeProto::AttributeType_Name(
                  AttributeProto::AttributeType::SPARSE_TENSORS)) == "SPARSE_TENSORS");
static_assert(std::string_view(AttributeProto_AttributeType_Name(
                  AttributeProto::AttributeType::TYPE_PROTO)) == "TYPE_PROTO");

TEST(onnx_compatibility, NamespaceMacros) {
  EXPECT_TRUE((std::is_same<ONNX_LIGHT_NAMESPACE::ModelProto, ModelProto>::value));
  EXPECT_TRUE((std::is_same<ONNX_LIGHT_NAMESPACE::TensorProto, TensorProto>::value));
}

TEST(onnx_compatibility, StringUtilsMakeString) {
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::MakeString("ab", 3, 'c'), "ab3c");
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::MakeString(std::string("xyz")), "xyz");
}

TEST(onnx_compatibility, ProtoUtilsParseAndRetrieve) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g1");
  std::string serialized;
  model.SerializeToString(serialized);

  ModelProto parsed;
  EXPECT_TRUE(
      ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes(&parsed, serialized.data(), serialized.size()));
  EXPECT_TRUE(parsed.has_graph());
  EXPECT_EQ(parsed.ref_graph().ref_name(), "g1");
  EXPECT_FALSE(ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes<ModelProto>(nullptr, serialized.data(),
                                                                     serialized.size()));

  AttributeProto attr;
  attr.set_name("attr");
  attr.ref_ints().push_back(3);
  attr.ref_ints().push_back(4);
  attr.ref_floats().push_back(1.5f);
  attr.ref_strings().push_back(utils::String("aa", 2));
  attr.ref_strings().push_back(utils::String("bb", 2));

  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<int64_t>(attr)), (std::vector<int64_t>{3, 4}));
  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<float>(attr)), (std::vector<float>{1.5f}));
  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<std::string>(attr)),
            (std::vector<std::string>{"aa", "bb"}));

  const std::string debug = ONNX_LIGHT_NAMESPACE::ProtoDebugString(attr);
  EXPECT_FALSE(debug.empty());
}

TEST(onnx_string, RefString_Constructors) {
  utils::RefString original("test", 4);
  utils::RefString copied(original);
  EXPECT_EQ(copied.size(), 4);
  EXPECT_EQ(copied.data(), original.data());
  EXPECT_EQ(copied, original);

  const char *text = "hello";
  utils::RefString rs(text, 5);
  EXPECT_EQ(rs.size(), 5);
  EXPECT_EQ(rs.data(), text);
}

TEST(onnx_string, RefString_Assignment) {
  utils::RefString a("abc", 3);
  utils::RefString b("xyz", 3);
  b = a;
  EXPECT_EQ(b.data(), a.data());
  EXPECT_EQ(b.size(), 3);

  utils::String s("def", 3);
  utils::RefString c("123", 3);
  c = s;
  EXPECT_EQ(c.data(), s.data());
  EXPECT_EQ(c.size(), 3);
}

TEST(onnx_string, RefString_Methods) {
  utils::RefString a("hello", 5);
  EXPECT_EQ(a.size(), 5);
  EXPECT_EQ(a.c_str(), a.data());
  EXPECT_FALSE(a.empty());
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(a[0], 'h');
  EXPECT_EQ(a[4], 'o');
}

TEST(onnx_string, RefString_Equality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  std::string e("test");
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty == nullptr);
}

TEST(onnx_string, RefString_Inequality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  utils::String e("diff", 4);
  std::string f("test");
  std::string g("diff");
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);
  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);
  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx_string, RefString_AsString) {
  utils::RefString a("hello", 5);
  std::string str = a;
  EXPECT_EQ(str, "hello");

  utils::RefString empty(nullptr, 0);
  std::string emptyStr = empty;
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx_string, String_Constructors) {
  utils::String defaultStr;
  EXPECT_EQ(defaultStr.size(), 0);
  EXPECT_EQ(defaultStr, "");
  EXPECT_TRUE(defaultStr.empty());
  utils::RefString ref("test", 4);
  utils::String fromRef(ref);
  EXPECT_EQ(fromRef.size(), 4);
  EXPECT_NE(fromRef.data(), ref.data());
  EXPECT_EQ(fromRef, ref);

  utils::String fromCharPtr("hello", 5);
  EXPECT_EQ(fromCharPtr.size(), 5);
  EXPECT_EQ(fromCharPtr, "hello");

  utils::String withNull("abc\0", 4);
  EXPECT_EQ(withNull.size(), 4);
  EXPECT_EQ(withNull, std::string("abc\0", 4));

  std::string stdStr = "world";
  utils::String fromStdStr(stdStr);
  EXPECT_EQ(fromStdStr.size(), 5);
  EXPECT_EQ(fromStdStr, stdStr);
}

TEST(onnx_string, String_Assignment) {
  utils::String s;

  s = "abc";
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, "abc");

  utils::RefString ref("def", 3);
  s = ref;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, ref);

  utils::String other("xyz", 3);
  s = other;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data());

  std::string stdStr = "hello";
  s = stdStr;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s, stdStr);
}

TEST(onnx_string, String_Methods) {
  utils::String s("hello", 5);
  EXPECT_EQ(s.size(), 5);
  EXPECT_NE(s.data(), nullptr);
  EXPECT_FALSE(s.empty());
  utils::String empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(s[0], 'h');
  EXPECT_EQ(s[4], 'o');
  s.clear();
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s, "");
  EXPECT_TRUE(s.empty());
}

TEST(onnx_string, String_Equality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  std::string e("test");

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::String empty;
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty.empty());
}

TEST(onnx_string, String_Inequality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  utils::RefString e("diff", 4);
  std::string f("test");
  std::string g("diff");

  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);

  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);

  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx_string, String_AsString) {
  utils::String a("hello", 5);
  std::string str = a;
  EXPECT_EQ(str, "hello");

  utils::String empty;
  std::string emptyStr = empty;
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx_string, String_EdgeCases) {
  utils::String empty("");
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);

  utils::String null("", 0);
  EXPECT_TRUE(null.empty());
  EXPECT_EQ(null.size(), 0);

  utils::String withNulls("abc\0def", 7);
  EXPECT_EQ(withNulls.size(), 7);
  EXPECT_EQ(withNulls[3], '\0');
  EXPECT_EQ(withNulls[4], 'd');
}

TEST(onnx_string, RefString) {
  utils::RefString a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
}

TEST(onnx_string, RefString_ConstructorFromConstCharPtr) {
  const char *text = "hello world";
  utils::RefString rs(text, 11);
  EXPECT_EQ(rs.size(), 11);
  EXPECT_EQ(rs.data(), text);
  EXPECT_FALSE(rs.empty());
  EXPECT_EQ(rs, "hello world");

  utils::RefString empty_rs("", 0);
  EXPECT_EQ(empty_rs.size(), 0);
  EXPECT_TRUE(empty_rs.empty());
  EXPECT_EQ(empty_rs, "");

  utils::RefString null_rs(nullptr, 0);
  EXPECT_EQ(null_rs.size(), 0);
  EXPECT_TRUE(null_rs.empty());
  EXPECT_EQ(null_rs.data(), nullptr);
}

TEST(onnx_string, RefString_AssignmentVariations) {
  utils::RefString source("source", 6);
  utils::RefString target("target", 6);
  EXPECT_EQ(target, "target");
  target = source;
  EXPECT_EQ(target, "source");
  EXPECT_EQ(target.data(), source.data());
  EXPECT_EQ(target.size(), source.size());

  utils::RefString empty(nullptr, 0);
  utils::RefString non_empty("data", 4);
  non_empty = empty;
  EXPECT_TRUE(non_empty.empty());
  EXPECT_EQ(non_empty.size(), 0);
  EXPECT_EQ(non_empty.data(), nullptr);
}

TEST(onnx_string, String_ConstructorFromConstCharPtrVariations) {
  utils::String s1("test string", 11);
  EXPECT_EQ(s1.size(), 11);
  EXPECT_EQ(s1, "test string");

  utils::String s2("embedded\0null", 13);
  EXPECT_EQ(s2.size(), 13);
  EXPECT_EQ(s2[8], '\0');
  EXPECT_EQ(s2[9], 'n');

  utils::String s3("", 0);
  EXPECT_EQ(s3.size(), 0);
  EXPECT_TRUE(s3.empty());

  utils::String s4("", 0);
  EXPECT_EQ(s4.size(), 0);
  EXPECT_TRUE(s4.empty());
}

TEST(onnx_string, String_ConstructorFromStdString) {
  std::string std_str = "hello std::string";
  utils::String s1(std_str);
  EXPECT_EQ(s1.size(), std_str.size());
  EXPECT_EQ(s1, std_str);

  std::string empty_str = "";
  utils::String s2(empty_str);
  EXPECT_EQ(s2.size(), 0);
  EXPECT_TRUE(s2.empty());

  std::string null_str(10, '\0');
  utils::String s3(null_str);
  EXPECT_EQ(s3.size(), 10);
  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(s3[i], '\0');
  }
}

TEST(onnx_string, String_ConstructorFromRefString) {
  utils::RefString ref1("reference data", 14);
  utils::String s1(ref1);
  EXPECT_EQ(s1.size(), ref1.size());
  EXPECT_EQ(s1, ref1);
  EXPECT_NE(s1.data(), ref1.data());

  utils::RefString ref2(nullptr, 0);
  utils::String s2(ref2);
  EXPECT_EQ(s2.size(), 0);
  EXPECT_TRUE(s2.empty());

  char data[5] = {'a', '\0', 'b', '\0', 'c'};
  utils::RefString ref3(data, 5);
  utils::String s3(ref3);
  EXPECT_EQ(s3.size(), 5);
  EXPECT_EQ(s3[0], 'a');
  EXPECT_EQ(s3[1], '\0');
  EXPECT_EQ(s3[2], 'b');
  EXPECT_EQ(s3[3], '\0');
  EXPECT_EQ(s3[4], 'c');
}

TEST(onnx_string, String_CopyConstructor) {
  utils::String original("original data", 13);
  utils::String copy(original);
  EXPECT_EQ(copy.size(), original.size());
  EXPECT_EQ(copy, original);
  EXPECT_NE(copy.data(), original.data());

  original = "changed data";
  EXPECT_EQ(copy, "original data");
  EXPECT_NE(copy, original);

  utils::String empty_original;
  utils::String empty_copy(empty_original);
  EXPECT_EQ(empty_copy.size(), 0);
  EXPECT_TRUE(empty_copy.empty());
}

TEST(onnx_string, String_MoveConstructor) {
  utils::String original("move this data", 14);
  utils::String moved(std::move(original));

  EXPECT_EQ(moved.size(), 14);
  EXPECT_EQ(moved, "move this data");
  EXPECT_TRUE(original.empty() || original.size() == 0);
}

TEST(onnx_string, String_AssignmentOperators) {
  utils::String s;

  s = "1234567890123456789";
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, "1234567890123456789");

  utils::RefString ref("1234567890123456789012", 22);
  s = ref;
  EXPECT_EQ(s.size(), 22);
  EXPECT_EQ(s, ref);
  EXPECT_NE(s.data(), ref.data());

  utils::String other("A234567890123456789", 19);
  s = other;
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data());

  std::string std_str = "assigned from std::string";
  s = std_str;
  EXPECT_EQ(s.size(), std_str.size());
  EXPECT_EQ(s, std_str);

  s = s;
  EXPECT_EQ(s, "assigned from std::string");
}

TEST(onnx_string, String_SelfAssignmentSafety) {
  utils::String s("12345678901234567890", 19);

  s = s;
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, "1234567890123456789");
  EXPECT_EQ(s, "1234567890123456789");

  utils::String *ptr = &s;
  *ptr = *ptr;
  EXPECT_EQ(*ptr, "1234567890123456789");
}

TEST(onnx_string, RefString_EqualityEdgeCases) {
  char data1[] = {'t', 'e', 's', 't', '\0', '!'};
  char data2[] = {'t', 'e', 's', 't', '\0', '?'};

  utils::RefString rs1(data1, 6);
  utils::RefString rs2(data2, 6);
  utils::RefString rs3(data1, 4);

  EXPECT_NE(rs1, rs2);
  EXPECT_NE(rs1, rs3);

  utils::RefString null_rs(nullptr, 0);
  utils::RefString empty_rs("", 0);

  EXPECT_EQ(null_rs, empty_rs);
  EXPECT_EQ(null_rs, nullptr);
  EXPECT_EQ(null_rs, "");
  EXPECT_NE(rs1, nullptr);
  EXPECT_NE(rs1, "");
}

TEST(onnx_string, String_EqualityEdgeCases) {
  utils::String s1("test\0!", 6);
  utils::String s2("test\0?", 6);
  utils::String s3("test", 4);

  EXPECT_NE(s1, s2);
  EXPECT_NE(s1, s3);

  utils::String empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty, "");
  EXPECT_FALSE(s1.empty());
  EXPECT_NE(s1, "");

  utils::RefString rs("test", 4);
  EXPECT_EQ(s3, rs);
  EXPECT_NE(s1, rs);
}

TEST(onnx_string, String_NullVersusSizeZero) {
  // String is now a plain std::string: there is no null state, only empty.
  utils::String empty_string;
  EXPECT_TRUE(empty_string.empty());
  EXPECT_EQ(empty_string.size(), 0);
  EXPECT_EQ(empty_string, "");

  // Presence semantics live in OptionalString.
  utils::OptionalString unset;
  EXPECT_TRUE(unset.null());
  EXPECT_TRUE(unset.empty());
  EXPECT_EQ(unset.size(), 0);
  EXPECT_EQ(unset.data(), nullptr);

  utils::OptionalString present("");
  EXPECT_FALSE(present.null());
  EXPECT_TRUE(present.empty());
  EXPECT_EQ(present.size(), 0);
  EXPECT_NE(present.data(), nullptr);

  // value() returns a const std::string& and never throws, even when unset.
  static_assert(std::is_same_v<decltype(unset.value()), const std::string &>,
                "OptionalString::value() must return const std::string&");
  EXPECT_EQ(unset.value(), std::string());
  EXPECT_TRUE(unset.value().empty());

  utils::OptionalString set_value("hello");
  EXPECT_EQ(set_value.value(), "hello");
  const std::string &bound = set_value.value();
  EXPECT_EQ(&bound, &set_value.value());
}

TEST(onnx_string, OptionalString_ImplicitConstStringRef) {
  // A set value implicitly converts to a const std::string& bound to the stored value.
  utils::OptionalString set_value("hello");
  const std::string &bound = set_value;
  EXPECT_EQ(bound, "hello");
  EXPECT_EQ(&bound, &set_value.value());

  // An unset value implicitly converts to a shared empty string without throwing.
  utils::OptionalString unset;
  const std::string &empty_bound = unset;
  EXPECT_TRUE(empty_bound.empty());
  EXPECT_EQ(&empty_bound, &utils::OptionalString::empty_value());

  // The implicit conversion also works when passing to a const std::string& parameter.
  auto takes_string_ref = [](const std::string &s) { return s.size(); };
  EXPECT_EQ(takes_string_ref(set_value), 5u);
  EXPECT_EQ(takes_string_ref(unset), 0u);
}

TEST(onnx_proto, NodeProtoDomainKeepsExplicitEmptyString) {
  NodeProto with_empty_domain;
  with_empty_domain.set_op_type("Constant");
  with_empty_domain.set_domain("");
  EXPECT_FALSE(with_empty_domain.ref_domain().null());

  std::string serialized_empty_domain;
  with_empty_domain.SerializeToString(serialized_empty_domain);

  NodeProto without_domain;
  without_domain.set_op_type("Constant");
  std::string serialized_without_domain;
  without_domain.SerializeToString(serialized_without_domain);
  EXPECT_NE(serialized_empty_domain, serialized_without_domain);

  NodeProto parsed;
  parsed.ParseFromString(serialized_empty_domain);
  EXPECT_EQ(parsed.ref_domain().size(), 0);
  EXPECT_FALSE(parsed.ref_domain().null());
}

TEST(onnx_proto, StrAccessorReturnsConstStringRef) {
  NodeProto node;
  node.set_op_type("Constant");
  // str_op_type() returns a const std::string& bound to the underlying value.
  const std::string &op_type = node.str_op_type();
  EXPECT_TRUE((std::is_same<decltype(node.str_op_type()), const std::string &>::value));
  EXPECT_EQ(op_type, "Constant");

  // An unset field returns a shared empty string without throwing.
  EXPECT_FALSE(node.has_domain());
  EXPECT_TRUE(node.str_domain().empty());
}

TEST(onnx_string, RefString_AsStringEdgeCases) {
  utils::RefString rs1("regular strings", 14);
  std::string s1 = rs1;
  EXPECT_EQ(s1, "regular string");

  char data[] = {'t', 'e', 's', 't', '\0', '!'};
  utils::RefString rs2(data, 6);
  std::string s2 = rs2;
  EXPECT_EQ(s2.size(), 6);
  EXPECT_EQ(s2[4], '\0');

  utils::RefString null_rs(nullptr, 0);
  std::string s3 = null_rs;
  EXPECT_TRUE(s3.empty());
  EXPECT_EQ(s3, "");
}

TEST(onnx_string, String_AsStringEdgeCases) {
  utils::String s1("1234567890123", 13);
  std::string std_s1 = s1;
  EXPECT_EQ(std_s1, "1234567890123");

  utils::String s2("test\0!", 6);
  std::string std_s2 = s2;
  EXPECT_EQ(std_s2.size(), 6);
  EXPECT_EQ(std_s2[4], '\0');

  // Empty string
  utils::String empty;
  std::string std_s3 = empty;
  EXPECT_TRUE(std_s3.empty());
  EXPECT_EQ(std_s3, "");
}

TEST(onnx_string, String) {
  utils::String a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
  std::string s("iii");
  utils::String b(s);
  EXPECT_EQ(b.size(), 3);
  EXPECT_FALSE(b.empty());
  EXPECT_EQ(b, a);
  EXPECT_EQ(b, "iii");
}

TEST(onnx_proto, TensorProtoName1) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.ref_name().data(), nullptr);
  EXPECT_EQ(tp.ref_name().size(), 0);
  std::string name("test");
  tp.name_ = name;
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  EXPECT_EQ(tp.order_name(), 8);
}

TEST(onnx_proto, TensorProtoContentHash) {
  auto make = [](const std::string &name, const std::vector<float> &data) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto::DataType::FLOAT);
    tp.add_dims(static_cast<uint64_t>(data.size()));
    for (float v : data) {
      tp.add_float_data(v);
    }
    return tp;
  };

  const TensorProto a = make("a", {1.0f, 2.0f, 3.0f});
  const TensorProto b = make("b", {1.0f, 2.0f, 3.0f});
  const TensorProto c = make("c", {1.0f, 2.0f, 9.0f});

  // The name is never part of the hash.
  EXPECT_EQ(a.ContentHash(false), b.ContentHash(false));
  EXPECT_EQ(a.ContentHash(true), b.ContentHash(true));

  // Same shape/type but different content: the size-only hash collides while the
  // content hash discriminates.
  EXPECT_EQ(a.ContentHash(false), c.ContentHash(false));
  EXPECT_NE(a.ContentHash(true), c.ContentHash(true));

  // A different shape changes both hashes.
  const TensorProto d = make("d", {1.0f, 2.0f});
  EXPECT_NE(a.ContentHash(false), d.ContentHash(false));
  EXPECT_NE(a.ContentHash(true), d.ContentHash(true));

  // The return type is int64_t.
  static_assert(std::is_same<decltype(a.ContentHash(true)), int64_t>::value);
}

TEST(onnx_proto, Blake3HasherMatchesOfficialVectors) {
  // The content hash reduces the official BLAKE3 digest to its first eight bytes
  // read little-endian. These expectations pin the vendored library (and its
  // parallel join) to the canonical BLAKE3 output for the standard test inputs
  // (message byte i == i % 251).
  auto reduce = [](const std::vector<uint8_t> &digest) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= static_cast<uint64_t>(digest[i]) << (8 * i);
    }
    return static_cast<int64_t>(value);
  };

  {
    // Empty input: af1349b9f5f9a1a6...
    utils::Blake3Hasher hasher;
    EXPECT_EQ(hasher.Finalize64(), reduce({0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6}));
  }
  {
    // 1024-byte input crosses a full BLAKE3 chunk: 42214739f095a406...
    std::vector<uint8_t> input(1024);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<uint8_t>(i % 251);
    }
    utils::Blake3Hasher hasher;
    hasher.Update(input.data(), input.size());
    EXPECT_EQ(hasher.Finalize64(), reduce({0x42, 0x21, 0x47, 0x39, 0xf0, 0x95, 0xa4, 0x06}));
  }
}

TEST(onnx_proto, TensorProtoContentHashParallelRawData) {
  // A payload larger than the parallel threshold (256 KiB) exercises the
  // multi-threaded BLAKE3 tree join. The digest must be deterministic and
  // independent of the number of threads used.
  const std::size_t size = 4 * 1024 * 1024; // 4 MiB
  auto make = [size](uint8_t seed) {
    TensorProto tp;
    tp.set_data_type(TensorProto::DataType::UINT8);
    tp.add_dims(static_cast<uint64_t>(size));
    std::string raw(size, static_cast<char>(0));
    for (std::size_t i = 0; i < size; ++i) {
      raw[i] = static_cast<char>((i + seed) % 251);
    }
    tp.set_raw_data(raw);
    return tp;
  };

  const TensorProto a = make(0);
  const TensorProto b = make(0);
  const TensorProto c = make(1);

  // Identical content hashes equal, and the result is stable across repeated
  // (independently parallelized) calls.
  EXPECT_EQ(a.ContentHash(true), b.ContentHash(true));
  EXPECT_EQ(a.ContentHash(true), a.ContentHash(true));

  // A different payload of the same size yields a different content hash while
  // the metadata-only hash still collides.
  EXPECT_EQ(a.ContentHash(false), c.ContentHash(false));
  EXPECT_NE(a.ContentHash(true), c.ContentHash(true));

  // Flipping a single byte flips the content hash.
  std::string raw(size, static_cast<char>(0));
  for (std::size_t i = 0; i < size; ++i) {
    raw[i] = static_cast<char>(i % 251);
  }
  raw[size / 2] = static_cast<char>(raw[size / 2] ^ 0x01);
  TensorProto flipped;
  flipped.set_data_type(TensorProto::DataType::UINT8);
  flipped.add_dims(static_cast<uint64_t>(size));
  flipped.set_raw_data(raw);
  EXPECT_NE(a.ContentHash(true), flipped.ContentHash(true));
}

TEST(onnx_proto, TensorProtoName2) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.ref_name().data(), nullptr);
  EXPECT_EQ(tp.ref_name().size(), 0);
  std::string name("test");
  tp.set_name(name);
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  std::string check = tp.name_;
  EXPECT_EQ(name, check);
  std::string check4 = tp.ref_name();
  EXPECT_EQ(name, check4);
  name = "TEST2";
  tp.set_name(name);
  std::string check2 = tp.name_;
  EXPECT_EQ(name, check2);
}

TEST(onnx_proto, TensorProtoNameStringToString1) {
  {
    TensorProto tp;
    tp.name_ = "test";
    if (tp.ref_name().size() == 4) {
      TensorProto tp2;
      tp2.set_name(tp.ref_name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp.name_.reset();
    }
    EXPECT_EQ(tp.name_.size(), 4);
    EXPECT_NE(tp.name_.data(), nullptr);
    EXPECT_EQ(tp.name_.data()[0], 't');
    EXPECT_EQ(tp.order_name(), 8);
    EXPECT_EQ(tp.name_, "test");
  }
}

TEST(onnx_proto, TensorProtoNameStringToString2) {
  {
    TensorProto tp2;
    if (tp2.ref_name().size() == 0) {
      TensorProto tp;
      tp.name_ = "test";
      tp2.set_name(tp.ref_name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp2.name_.reset();
    }
    EXPECT_EQ(tp2.name_.size(), 4);
    EXPECT_NE(tp2.name_.data(), nullptr);
    EXPECT_EQ(tp2.name_.data()[0], 't');
    EXPECT_EQ(tp2.order_name(), 8);
    EXPECT_EQ(tp2.name_, "test");
  }
}

TEST(onnx_proto, TensorProtoName00) { TensorProto tp; }
TEST(onnx_proto, TensorProtoName01) {
  TensorProto tp;
  tp.set_name("rt");
}

TEST(onnx_proto, serialization_StringStringEntryProto) {
  StringStringEntryProto proto;
  proto.ref_key() = "key__";
  proto.ref_value() = "value__";
  EXPECT_EQ(proto.ref_key(), "key__");
  EXPECT_EQ(proto.ref_value(), "value__");
  std::string serialized;
  proto.SerializeToString(serialized);
  EXPECT_EQ(serialized.size(), proto.SerializeSize().size());
  StringStringEntryProto proto2;
  proto2.ParseFromString(serialized);
  EXPECT_EQ(proto.ref_key().sv(), proto2.ref_key().sv());
  EXPECT_EQ(proto.ref_value().sv(), proto2.ref_value().sv());
  std::string serialized2;
  proto2.SerializeToString(serialized2);
  EXPECT_EQ(serialized2.size(), proto2.SerializeSize().size());
  EXPECT_EQ(serialized, serialized2);
}

TEST(onnx_proto, serialization_StringStringEntryProto_Twice) {
  StringStringEntryProto proto;
  proto.set_key("key_");
  proto.set_value("value_");
  EXPECT_EQ(proto.ref_key(), "key_");
  EXPECT_EQ(proto.ref_value(), "value_");
  proto.set_key("key__");
  proto.set_value("value__");
  EXPECT_EQ(proto.ref_key(), "key__");
  EXPECT_EQ(proto.ref_value(), "value__");
  proto.ref_key() = "key___";
  proto.ref_value() = "value___";
  EXPECT_EQ(proto.ref_key(), "key___");
  EXPECT_EQ(proto.ref_value(), "value___");
}

TEST(onnx_proto, TensorShapeProto1) {
  TensorShapeProto shape;
  TensorShapeProto::Dimension *dim = shape.add_dim();
  dim->set_dim_value(5);
  TensorShapeProto::Dimension &dim2 = shape.ref_dim().add();
  dim2.set_dim_param("dime");
  dim2.ref_denotation() = "jj";
  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_param().size(), 0);
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "dime");
  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "jj");
}

TEST(onnx_stream, ZigZagEncoding) {
  int64_t original_values[] = {0, -1, 1, -2, 2, INT64_MAX, INT64_MIN};

  for (auto val : original_values) {
    uint64_t encoded = utils::encodeZigZag64(val);
    int64_t decoded = utils::decodeZigZag64(encoded);
    EXPECT_EQ(decoded, val) << "ZigZag encoding/decoding failed for value: " << val;
  }
}

TEST(onnx_stream, FieldNumber) {
  utils::FieldNumber fn;
  fn.field_number = 5;
  fn.wire_type = 2;

  std::string str = fn.string();
  EXPECT_FALSE(str.empty());
  EXPECT_NE(str.find("field_number=5"), std::string::npos);
  EXPECT_NE(str.find("wire_type=2"), std::string::npos);
}

class onnx_stream_2 : public ::testing::Test {
protected:
  void SetUp() override {
    data = {0x96, 0x01,
            // int64_t
            0x2A,
            // int32_t
            0x18,
            // float: 3.14
            0xC3, 0xF5, 0x48, 0x40,
            // double: 2.71828
            0x4D, 0xFB, 0x21, 0x09, 0x40, 0x05, 0x5D, 0x40,
            // field number: 10, wire_type: 2 -> (10 << 3) | 2 = 82
            0x52,
            // string length: 5
            0x05,
            // string "hello"
            'h', 'e', 'l', 'l', 'o'};

    stream.Setup(data.data(), data.size());
  }

  std::vector<uint8_t> data;
  utils::StringStream stream;
};

TEST_F(onnx_stream_2, NextUInt64) {
  uint64_t value = stream.next_uint64();
  EXPECT_EQ(value, 150);
}

TEST_F(onnx_stream_2, NextInt64) {
  stream.next_uint64();

  int64_t value = stream.next_int64();
  EXPECT_EQ(value, 42);
}

TEST_F(onnx_stream_2, NextInt32) {
  stream.next_uint64();
  stream.next_int64();

  int32_t value = stream.next_int32();
  EXPECT_EQ(value, 24);
}

TEST_F(onnx_stream_2, NextFloat) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  float value = stream.next_float();
  EXPECT_NEAR(value, 3.14f, 0.0001f);
}

TEST_F(onnx_stream_2, NextField) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();

  utils::FieldNumber field = stream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);
}

TEST_F(onnx_stream_2, NextString) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();
  stream.next_field();

  utils::RefString value = stream.next_string();
  EXPECT_EQ(value.size(), 5);
  EXPECT_EQ(value, "hello");
}

TEST_F(onnx_stream_2, ReadBytes) {
  const uint8_t *bytes = stream.read_bytes(2);
  EXPECT_EQ(bytes[0], 0x96);
  EXPECT_EQ(bytes[1], 0x01);
}

TEST_F(onnx_stream_2, CanRead) {
  stream.CanRead(data.size(), "Test message");
  stream.read_bytes(10);
  stream.CanRead(data.size() - 10, "Test message");
  EXPECT_THROW(stream.CanRead(data.size(), "Test message"), std::runtime_error);
}

TEST_F(onnx_stream_2, NotEnd) {
  EXPECT_TRUE(stream.NotEnd());
  stream.read_bytes(data.size() - 1);
  EXPECT_TRUE(stream.NotEnd());
  stream.read_bytes(1);
  EXPECT_FALSE(stream.NotEnd());
}

TEST_F(onnx_stream_2, Tell) {
  EXPECT_EQ(stream.tell(), 0);

  stream.read_bytes(5);
  EXPECT_EQ(stream.tell(), 5);

  stream.read_bytes(10);
  EXPECT_EQ(stream.tell(), 15);
}

TEST(onnx_stream, StringWriteStream) {
  utils::StringWriteStream stream;

  stream.write_variant_uint64(150);
  stream.write_int64(42);
  stream.write_int32(24);
  stream.write_float(3.14f);
  stream.write_double(2.71828);
  stream.write_field_header(10, 2);
  stream.write_string("hello");
  EXPECT_GT(stream.size(), 0);
  EXPECT_NE(stream.data(), nullptr);

  utils::StringStream readStream(stream.data(), stream.size());

  EXPECT_EQ(readStream.next_uint64(), 150);
  EXPECT_EQ(readStream.next_int64(), 42);
  EXPECT_EQ(readStream.next_int32(), 24);
  EXPECT_NEAR(readStream.next_float(), 3.14f, 0.0001f);
  EXPECT_NEAR(readStream.next_double(), 2.71828, 0.0001);

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);

  utils::RefString str = readStream.next_string();
  EXPECT_EQ(str, "hello");
}

TEST(onnx_stream, StringWriteStreamStrings) {
  utils::StringWriteStream stream;

  std::string stdStr = "standard string";
  stream.write_string(stdStr);
  utils::String str("custom string", 13);
  stream.write_string(str);
  utils::RefString refStr("reference string", 16);
  stream.write_string(refStr);
  utils::StringStream readStream(stream.data(), stream.size());

  utils::RefString read1 = readStream.next_string();
  EXPECT_EQ(read1, "standard string");

  utils::RefString read2 = readStream.next_string();
  EXPECT_EQ(read2, "custom string");

  utils::RefString read3 = readStream.next_string();
  EXPECT_EQ(read3, "reference string");
}

TEST(onnx_stream, BorrowedWriteStream) {
  std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
  utils::BorrowedWriteStream stream(data.data(), data.size());
  EXPECT_EQ(stream.size(), 5);
  EXPECT_EQ(stream.data(), data.data());
  EXPECT_THROW(stream.write_raw_bytes(nullptr, 0), std::runtime_error);
}

TEST(onnx_stream, BorrowedStringWriteStream) {
  std::vector<uint8_t> data(64, 0);
  utils::BorrowedStringWriteStream stream(data.data(), data.size());

  stream.write_variant_uint64(150);
  stream.write_int64(42);
  stream.write_string("hello");

  EXPECT_GT(stream.size(), 0);
  EXPECT_LE(stream.size(), static_cast<int64_t>(data.size()));
  EXPECT_EQ(stream.data(), data.data());

  utils::StringStream readStream(stream.data(), stream.size());
  EXPECT_EQ(readStream.next_uint64(), 150);
  EXPECT_EQ(readStream.next_int64(), 42);
  EXPECT_EQ(readStream.next_string(), "hello");
}

TEST(onnx_stream, BorrowedStringWriteStreamNoReallocation) {
  std::vector<uint8_t> data(2, 0);
  utils::BorrowedStringWriteStream stream(data.data(), data.size());
  EXPECT_THROW(stream.write_string("hello"), std::runtime_error);
}

TEST(onnx_stream, NestedStringWriteStreams) {
  utils::StringWriteStream innerStream;

  innerStream.write_string("inner data");
  utils::StringWriteStream outerStream;
  outerStream.write_field_header(15, 2);

  outerStream.write_string_stream(innerStream);

  utils::StringStream readStream(outerStream.data(), outerStream.size());

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 15);
  EXPECT_EQ(field.wire_type, 2);

  uint64_t length = readStream.next_uint64();
  readStream.LimitToNext(length);
  utils::RefString str = readStream.next_string();
  readStream.Restore();
  EXPECT_EQ(str, "inner data");
}

TEST(onnx_stream, NextPackedElement) {
  std::vector<uint8_t> data = {// a float: 3.14
                               0xC3, 0xF5, 0x48, 0x40,
                               // int32: 42
                               0x2A, 0x00, 0x00, 0x00};

  utils::StringStream stream(data.data(), data.size());

  float f;
  stream.next_packed_element(f);
  EXPECT_NEAR(f, 3.14f, 0.0001f);

  int32_t i;
  stream.next_packed_element(i);
  EXPECT_EQ(i, 42);
}

TEST(onnx_stream, ErrorCases) {
  std::vector<uint8_t> badData = {0x80, 0x80, 0x80};
  utils::StringStream badStream(badData.data(), badData.size());

  EXPECT_THROW(badStream.next_uint64(), std::runtime_error);

  std::vector<uint8_t> smallData = {0x01, 0x02};
  utils::StringStream smallStream(smallData.data(), smallData.size());

  EXPECT_THROW(smallStream.CanRead(3, "Test message"), std::runtime_error);
}

TEST(onnx_proto, StringStringEntryProto_Basic) {
  StringStringEntryProto entry;

  EXPECT_TRUE(entry.ref_key().empty());
  EXPECT_TRUE(entry.ref_value().empty());
  EXPECT_FALSE(entry.has_key());
  EXPECT_FALSE(entry.has_value());

  entry.set_key("test_key");
  entry.set_value("test_value");

  EXPECT_EQ(entry.ref_key(), "test_key");
  EXPECT_EQ(entry.ref_value(), "test_value");
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx_proto, StringStringEntryProto_Serialization) {
  StringStringEntryProto entry;
  entry.set_key("test_key");
  entry.set_value("test_value");

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize().size());

  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), "test_key");
  EXPECT_EQ(entry2.ref_value(), "test_value");
}

TEST(onnx_proto, IntIntListEntryProto_Basic) {
  IntIntListEntryProto entry;

  EXPECT_EQ(entry.ref_key(), 0);
  EXPECT_EQ(entry.ref_value().size(), 0);
  EXPECT_FALSE(entry.has_value());

  entry.set_key(42);
  entry.ref_value().push_back(1);
  entry.ref_value().push_back(2);
  entry.ref_value().push_back(3);

  EXPECT_EQ(entry.ref_key(), 42);
  EXPECT_EQ(entry.ref_value().size(), 3);
  EXPECT_EQ(entry.ref_value()[0], 1);
  EXPECT_EQ(entry.ref_value()[1], 2);
  EXPECT_EQ(entry.ref_value()[2], 3);
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx_proto, IntIntListEntryProto_Serialization) {
  IntIntListEntryProto entry;
  entry.set_key(42);
  entry.ref_value().push_back(1);
  entry.ref_value().push_back(2);
  entry.ref_value().push_back(3);

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize().size());

  IntIntListEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), 42);
  EXPECT_EQ(entry2.ref_value().size(), 3);
  EXPECT_EQ(entry2.ref_value()[0], 1);
  EXPECT_EQ(entry2.ref_value()[1], 2);
  EXPECT_EQ(entry2.ref_value()[2], 3);
}

TEST(onnx_proto, TensorAnnotation_Basic) {
  TensorAnnotation annotation;

  EXPECT_TRUE(annotation.ref_tensor_name().empty());
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names().size(), 0);

  annotation.set_tensor_name("my_tensor");
  StringStringEntryProto *entry = annotation.add_quant_parameter_tensor_names();
  entry->set_key("scale");
  entry->set_value("scale_tensor");

  EXPECT_EQ(annotation.ref_tensor_name(), "my_tensor");
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names().size(), 1);
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names()[0].ref_key(), "scale");
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names()[0].ref_value(), "scale_tensor");
}

TEST(onnx_proto, DeviceConfigurationProto_Basic) {
  DeviceConfigurationProto config;

  EXPECT_TRUE(config.ref_name().empty());
  EXPECT_EQ(config.ref_num_devices(), 0);
  EXPECT_EQ(config.ref_device().size(), 0);

  config.set_name("CPU");
  config.set_num_devices(2);
  *config.add_device() = "device0";
  *config.add_device() = "device1";

  EXPECT_EQ(config.ref_name(), "CPU");
  EXPECT_EQ(config.ref_num_devices(), 2);
  EXPECT_EQ(config.ref_device().size(), 2);
  EXPECT_EQ(config.ref_device()[0], "device0");
  EXPECT_EQ(config.ref_device()[1], "device1");
}

TEST(onnx_proto, SimpleShardedDimProto_Basic) {
  SimpleShardedDimProto dim;

  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.ref_dim_param().empty());
  EXPECT_EQ(dim.ref_num_shards(), 0);

  dim.set_dim_value(100);
  dim.set_dim_param("batch");
  dim.set_num_shards(4);

  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.ref_dim_value(), 100);
  EXPECT_EQ(dim.ref_dim_param(), "batch");
  EXPECT_EQ(dim.ref_num_shards(), 4);
}

TEST(onnx_proto, ShardedDimProto_Basic) {
  ShardedDimProto dim;

  EXPECT_EQ(dim.ref_axis(), 0);
  EXPECT_EQ(dim.ref_simple_sharding().size(), 0);

  dim.set_axis(1);
  SimpleShardedDimProto *simple_dim = dim.add_simple_sharding();
  simple_dim->set_dim_value(100);
  simple_dim->set_num_shards(4);

  EXPECT_EQ(dim.ref_axis(), 1);
  EXPECT_EQ(dim.ref_simple_sharding().size(), 1);
  EXPECT_EQ(dim.ref_simple_sharding()[0].ref_dim_value(), 100);
  EXPECT_EQ(dim.ref_simple_sharding()[0].ref_num_shards(), 4);
}

TEST(onnx_proto, ShardingSpecProto_Basic) {
  ShardingSpecProto spec;

  EXPECT_TRUE(spec.ref_tensor_name().empty());
  EXPECT_EQ(spec.ref_device().size(), 0);
  EXPECT_EQ(spec.ref_index_to_device_group_map().size(), 0);
  EXPECT_EQ(spec.ref_sharded_dim().size(), 0);

  spec.set_tensor_name("my_tensor");

  spec.ref_device().push_back(0);
  spec.ref_device().push_back(1);

  IntIntListEntryProto *map_entry = spec.add_index_to_device_group_map();
  map_entry->set_key(0);
  map_entry->ref_value().push_back(0);

  ShardedDimProto *dim = spec.add_sharded_dim();
  dim->set_axis(0);

  EXPECT_EQ(spec.ref_tensor_name(), "my_tensor");
  EXPECT_EQ(spec.ref_device().size(), 2);
  EXPECT_EQ(spec.ref_device()[0], 0);
  EXPECT_EQ(spec.ref_device()[1], 1);
  EXPECT_EQ(spec.ref_index_to_device_group_map().size(), 1);
  EXPECT_EQ(spec.ref_index_to_device_group_map()[0].ref_key(), 0);
  EXPECT_EQ(spec.ref_sharded_dim().size(), 1);
  EXPECT_EQ(spec.ref_sharded_dim()[0].ref_axis(), 0);
}

TEST(onnx_proto, NodeDeviceConfigurationProto_Basic) {
  NodeDeviceConfigurationProto config;

  EXPECT_TRUE(config.ref_configuration_id().empty());
  EXPECT_EQ(config.ref_sharding_spec().size(), 0);
  EXPECT_FALSE(config.has_pipeline_stage());

  config.set_configuration_id("config1");
  config.add_sharding_spec();
  config.set_pipeline_stage(2);

  EXPECT_EQ(config.ref_configuration_id(), "config1");
  EXPECT_EQ(config.ref_sharding_spec().size(), 1);
  EXPECT_TRUE(config.has_pipeline_stage());
  EXPECT_EQ(config.ref_pipeline_stage(), 2);
}

TEST(onnx_proto, OperatorSetIdProto_Basic) {
  OperatorSetIdProto op_set;

  EXPECT_TRUE(op_set.ref_domain().empty());
  EXPECT_EQ(op_set.ref_version(), 0);

  op_set.set_domain("ai.onnx");
  op_set.set_version(12);

  EXPECT_EQ(op_set.ref_domain(), "ai.onnx");
  EXPECT_EQ(op_set.ref_version(), 12);
}

TEST(onnx_proto, TensorShapeProto_Basic) {
  TensorShapeProto shape;

  EXPECT_EQ(shape.ref_dim().size(), 0);

  TensorShapeProto::Dimension *dim1 = shape.add_dim();
  dim1->set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.ref_dim().add();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_TRUE(shape.ref_dim()[0].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_FALSE(shape.ref_dim()[0].has_dim_param());

  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "batch");
}

TEST(onnx_proto, TensorShapeProtoDimension) {
  TensorShapeProto::Dimension dim;

  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.ref_dim_param().empty());
  EXPECT_TRUE(dim.ref_denotation().empty());

  dim.set_dim_value(10);
  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.ref_dim_value(), 10);

  dim.set_dim_param("batch_size");
  EXPECT_EQ(dim.ref_dim_param(), "batch_size");

  dim.set_denotation("batch");
  EXPECT_EQ(dim.ref_denotation(), "batch");
}

TEST(onnx_proto, TensorProto_Basic) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::UNDEFINED);
  EXPECT_EQ(tensor.ref_dims().size(), 0);
  EXPECT_TRUE(tensor.ref_name().empty());

  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);
  tensor.set_name("my_tensor");

  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);
  tensor.ref_float_data().push_back(3.0f);
  tensor.ref_float_data().push_back(4.0f);
  tensor.ref_float_data().push_back(5.0f);
  tensor.ref_float_data().push_back(6.0f);

  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 3);
  EXPECT_EQ(tensor.ref_name(), "my_tensor");
  EXPECT_EQ(tensor.ref_float_data().size(), 6);
  EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor.ref_float_data()[5], 6.0f);
}

TEST(onnx_proto, TensorProto_DataTypes) {
  TensorProto tensor;

  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);
  EXPECT_EQ(tensor.ref_float_data().size(), 2);
  EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor.ref_float_data()[1], 2.0f);

  tensor.set_data_type(TensorProto::DataType::INT32);
  tensor.ref_int32_data().push_back(10);
  tensor.ref_int32_data().push_back(20);
  EXPECT_EQ(tensor.ref_int32_data().size(), 2);
  EXPECT_EQ(tensor.ref_int32_data()[0], 10);
  EXPECT_EQ(tensor.ref_int32_data()[1], 20);

  tensor.set_data_type(TensorProto::DataType::STRING);
  *tensor.add_string_data() = "hello";
  *tensor.add_string_data() = "world";
  EXPECT_EQ(tensor.ref_string_data().size(), 2);
  EXPECT_EQ(tensor.ref_string_data()[0], "hello");
  EXPECT_EQ(tensor.ref_string_data()[1], "world");

  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.ref_int64_data().push_back(100);
  tensor.ref_int64_data().push_back(200);
  EXPECT_EQ(tensor.ref_int64_data().size(), 2);
  EXPECT_EQ(tensor.ref_int64_data()[0], 100);
  EXPECT_EQ(tensor.ref_int64_data()[1], 200);

  tensor.set_data_type(TensorProto::DataType::DOUBLE);
  tensor.ref_double_data().push_back(1.5);
  tensor.ref_double_data().push_back(2.5);
  EXPECT_EQ(tensor.ref_double_data().size(), 2);
  EXPECT_EQ(tensor.ref_double_data()[0], 1.5);
  EXPECT_EQ(tensor.ref_double_data()[1], 2.5);

  tensor.set_data_type(TensorProto::DataType::UINT64);
  tensor.ref_uint64_data().push_back(1000);
  tensor.ref_uint64_data().push_back(2000);
  EXPECT_EQ(tensor.ref_uint64_data().size(), 2);
  EXPECT_EQ(tensor.ref_uint64_data()[0], 1000);
  EXPECT_EQ(tensor.ref_uint64_data()[1], 2000);
}

TEST(onnx_proto, TensorProto_Segment) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_segment().ref_begin(), 0);
  EXPECT_EQ(tensor.ref_segment().ref_end(), 0);

  tensor.ref_segment().set_begin(5);
  tensor.ref_segment().set_end(10);

  EXPECT_EQ(tensor.ref_segment().ref_begin(), 5);
  EXPECT_EQ(tensor.ref_segment().ref_end(), 10);
}

TEST(onnx_proto, TensorProto_RawData) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_raw_data().size(), 0);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  tensor.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  EXPECT_EQ(tensor.ref_raw_data().size(), data.size() * sizeof(float));

  const float *raw_data_ptr = reinterpret_cast<const float *>(tensor.ref_raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

TEST(onnx_proto, TensorProto_Serialization) {
  TensorProto tensor1;
  tensor1.set_name("test_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);
  tensor1.ref_float_data().push_back(3.0f);
  tensor1.ref_float_data().push_back(4.0f);

  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), tensor1.SerializeSize().size());

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  EXPECT_EQ(tensor2.ref_name(), "test_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_dims()[0], 2);
  EXPECT_EQ(tensor2.ref_dims()[1], 2);
  EXPECT_EQ(tensor2.ref_float_data().size(), 4);
  EXPECT_EQ(tensor2.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.ref_float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.ref_float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.ref_float_data()[3], 4.0f);
}

TEST(onnx_proto, SparseTensorProto_Basic) {
  SparseTensorProto sparse;

  EXPECT_EQ(sparse.ref_dims().size(), 0);

  sparse.ref_dims().push_back(3);
  sparse.ref_dims().push_back(4);

  sparse.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.ref_values().ref_float_data().push_back(5.0f);
  sparse.ref_values().ref_float_data().push_back(6.0f);

  sparse.ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(0);
  sparse.ref_indices().ref_int64_data().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(1);
  sparse.ref_indices().ref_int64_data().push_back(3);

  EXPECT_EQ(sparse.ref_dims().size(), 2);
  EXPECT_EQ(sparse.ref_dims()[0], 3);
  EXPECT_EQ(sparse.ref_dims()[1], 4);

  EXPECT_EQ(sparse.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[0], 5.0f);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[1], 6.0f);

  EXPECT_EQ(sparse.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data().size(), 4);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[0], 0);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[1], 2);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[2], 1);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[3], 3);
}

TEST(onnx_proto, TypeProto_Tensor) {
  TypeProto type;

  EXPECT_FALSE(type.has_tensor_type());

  type.add_tensor_type()->set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_FALSE(type.ref_tensor_type().has_shape());
  TensorShapeProto *shape = type.ref_tensor_type().add_shape();
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  TensorShapeProto::Dimension *dim = shape->add_dim();
  dim->set_dim_value(3);

  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 1);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
}

TEST(onnx_proto, CreateTensorProto) {
  TensorProto tensor;
  tensor.set_name("test_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);

  for (int i = 0; i < 6; ++i) {
    tensor.ref_float_data().push_back(static_cast<float>(i + 1));
  }

  EXPECT_EQ(tensor.ref_name(), "test_tensor");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 3);
  EXPECT_EQ(tensor.ref_float_data().size(), 6);
}

TEST(onnx_proto, SerializeDeserializeTensorProto) {
  TensorProto tensor1;
  tensor1.set_name("serialized_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);
  tensor1.ref_float_data().push_back(3.0f);
  tensor1.ref_float_data().push_back(4.0f);

  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), tensor1.SerializeSize().size());

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  EXPECT_EQ(tensor2.ref_name(), "serialized_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_dims()[0], 2);
  EXPECT_EQ(tensor2.ref_dims()[1], 2);
  EXPECT_EQ(tensor2.ref_float_data().size(), 4);
  EXPECT_EQ(tensor2.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.ref_float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.ref_float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.ref_float_data()[3], 4.0f);
}

TEST(onnx_proto, TypeProtoOperations) {
  TypeProto type;

  type.add_tensor_type()->set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());

  TensorShapeProto *shape = type.ref_tensor_type().add_shape();

  TensorShapeProto::Dimension *dim1 = shape->add_dim();
  dim1->set_dim_value(3);

  TensorShapeProto::Dimension *dim2 = shape->add_dim();
  dim2->set_dim_param("batch_size");

  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "batch_size");
}

TEST(onnx_proto, StringStringEntryProtoOperations) {
  StringStringEntryProto entry;
  entry.set_key("metadata_key");
  entry.set_value("metadata_value");

  EXPECT_EQ(entry.ref_key(), "metadata_key");
  EXPECT_EQ(entry.ref_value(), "metadata_value");

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize().size());

  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), "metadata_key");
  EXPECT_EQ(entry2.ref_value(), "metadata_value");
}

TEST(onnx_proto, TensorProtoWithRawData) {
  TensorProto tensor;
  tensor.set_name("raw_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  tensor.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  EXPECT_EQ(tensor.ref_name(), "raw_data_tensor");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 2);
  EXPECT_EQ(tensor.ref_raw_data().size(), data.size() * sizeof(float));

  const float *raw_data_ptr = reinterpret_cast<const float *>(tensor.ref_raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

TEST(onnx_proto, SparseTensorProtoOperations) {
  SparseTensorProto sparse;

  sparse.ref_dims().push_back(3);
  sparse.ref_dims().push_back(4);

  sparse.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.ref_values().ref_float_data().push_back(5.0f);
  sparse.ref_values().ref_float_data().push_back(6.0f);

  sparse.ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(0);
  sparse.ref_indices().ref_int64_data().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(1);
  sparse.ref_indices().ref_int64_data().push_back(3);

  EXPECT_EQ(sparse.ref_dims().size(), 2);
  EXPECT_EQ(sparse.ref_dims()[0], 3);
  EXPECT_EQ(sparse.ref_dims()[1], 4);

  EXPECT_EQ(sparse.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[0], 5.0f);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[1], 6.0f);

  EXPECT_EQ(sparse.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data().size(), 4);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[0], 0);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[1], 2);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[2], 1);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[3], 3);

  std::string serialized;
  sparse.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), sparse.SerializeSize().size());

  SparseTensorProto sparse2;
  sparse2.ParseFromString(serialized);

  EXPECT_EQ(sparse2.ref_dims().size(), 2);
  EXPECT_EQ(sparse2.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse2.ref_indices().ref_int64_data().size(), 4);
}

TEST(onnx_proto, TensorShapeProtoOperations) {
  TensorShapeProto shape;

  TensorShapeProto::Dimension *dim1 = shape.add_dim();
  dim1->set_dim_value(5);

  TensorShapeProto::Dimension *dim2 = shape.add_dim();
  dim2->set_dim_param("N");
  dim2->set_denotation("batch");

  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_TRUE(shape.ref_dim()[0].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_FALSE(shape.ref_dim()[0].has_dim_param());

  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "batch");

  std::string serialized;
  shape.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), shape.SerializeSize().size());

  TensorShapeProto shape2;
  shape2.ParseFromString(serialized);

  EXPECT_EQ(shape2.ref_dim().size(), 2);
  EXPECT_EQ(shape2.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_EQ(shape2.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape2.ref_dim()[1].ref_denotation(), "batch");
}

TEST(onnx_proto, TensorProtoDataTypes) {
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.ref_float_data().push_back(1.0f);
    tensor.ref_float_data().push_back(2.0f);
    EXPECT_EQ(tensor.ref_float_data().size(), 2);
    EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
    EXPECT_EQ(tensor.ref_float_data()[1], 2.0f);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT32);
    tensor.ref_int32_data().push_back(10);
    tensor.ref_int32_data().push_back(20);
    EXPECT_EQ(tensor.ref_int32_data().size(), 2);
    EXPECT_EQ(tensor.ref_int32_data()[0], 10);
    EXPECT_EQ(tensor.ref_int32_data()[1], 20);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::STRING);
    *tensor.add_string_data() = "hello";
    *tensor.add_string_data() = "world";
    EXPECT_EQ(tensor.ref_string_data().size(), 2);
    EXPECT_EQ(tensor.ref_string_data()[0], "hello");
    EXPECT_EQ(tensor.ref_string_data()[1], "world");
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT64);
    tensor.ref_int64_data().push_back(100);
    tensor.ref_int64_data().push_back(200);
    EXPECT_EQ(tensor.ref_int64_data().size(), 2);
    EXPECT_EQ(tensor.ref_int64_data()[0], 100);
    EXPECT_EQ(tensor.ref_int64_data()[1], 200);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::DOUBLE);
    tensor.ref_double_data().push_back(1.5);
    tensor.ref_double_data().push_back(2.5);
    EXPECT_EQ(tensor.ref_double_data().size(), 2);
    EXPECT_EQ(tensor.ref_double_data()[0], 1.5);
    EXPECT_EQ(tensor.ref_double_data()[1], 2.5);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT64);
    tensor.ref_uint64_data().push_back(1000);
    tensor.ref_uint64_data().push_back(2000);
    EXPECT_EQ(tensor.ref_uint64_data().size(), 2);
    EXPECT_EQ(tensor.ref_uint64_data()[0], 1000);
    EXPECT_EQ(tensor.ref_uint64_data()[1], 2000);
  }
}

static TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
  case TensorProto_DataType_FLOAT:
    t.add_float_data((float)value);
    break;
  case TensorProto_DataType_DOUBLE:
    t.add_double_data(value);
    break;
  default:
    assert(false);
  }
  return t;
}

TEST(onnxonnx, DataType) {
  TensorProto proto = ToTensor(4.5, TensorProto_DataType_FLOAT);
  EXPECT_EQ(proto.ref_float_data().size(), 1);
  EXPECT_EQ(proto.ref_float_data()[0], 4.5);
  EXPECT_EQ(proto.ref_data_type(), TensorProto_DataType_FLOAT);
}

TEST(onnx_string, StringStringEntryProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::StringStringEntryProto proto;
  proto.set_key("test_key");
  proto.set_value("test_value");
  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  EXPECT_TRUE(serialized.find("test_key") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_value") != std::string::npos);
}

TEST(onnx_string, IntIntListEntryProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::IntIntListEntryProto proto;
  proto.set_key(42);
  proto.ref_value().push_back(1);
  proto.ref_value().push_back(2);
  proto.ref_value().push_back(3);
  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  EXPECT_TRUE(serialized.find("42") != std::string::npos);
  EXPECT_TRUE(serialized.find("1") != std::string::npos);
  EXPECT_TRUE(serialized.find("2") != std::string::npos);
  EXPECT_TRUE(serialized.find("3") != std::string::npos);
}

TEST(onnx_string, TensorAnnotation) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TensorAnnotation proto;
  proto.set_tensor_name("my_tensor");
  auto *entry = proto.add_quant_parameter_tensor_names();
  entry->set_key("scale");
  entry->set_value("scale_tensor");
  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  EXPECT_TRUE(serialized.find("my_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale_tensor") != std::string::npos);
}

TEST(onnx_string, DeviceConfigurationProto) {
  utils::PrintOptions options;
  DeviceConfigurationProto config;
  config.set_name("test_device_config");
  config.set_num_devices(3);
  *config.add_device() = "device1";
  *config.add_device() = "device2";
  *config.add_device() = "device3";

  std::stringstream ss_result;
  config.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();

  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundNumDevices = false;
  bool foundDevices = false;

  std::string item = serialized;
  if (item.find("name:") != std::string::npos &&
      item.find("test_device_config") != std::string::npos) {
    foundName = true;
  }
  if (item.find("num_devices:") != std::string::npos && item.find("3") != std::string::npos) {
    foundNumDevices = true;
  }
  if (item.find("device:") != std::string::npos && item.find("device1") != std::string::npos &&
      item.find("device2") != std::string::npos && item.find("device3") != std::string::npos) {
    foundDevices = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundNumDevices);
  EXPECT_TRUE(foundDevices);
}

TEST(onnx_string, SimpleShardedDimProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::SimpleShardedDimProto proto;
  proto.set_dim_value(100);
  proto.set_dim_param("batch_size");
  proto.set_num_shards(4);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("dim_value:") != std::string::npos);
  EXPECT_TRUE(serialized.find("100") != std::string::npos);
  EXPECT_TRUE(serialized.find("dim_param:") != std::string::npos);
  EXPECT_TRUE(serialized.find("batch_size") != std::string::npos);
  EXPECT_TRUE(serialized.find("num_shards:") != std::string::npos);
  EXPECT_TRUE(serialized.find("4") != std::string::npos);
}

TEST(onnx_string, ShardedDimProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::ShardedDimProto proto;
  proto.set_axis(2);

  auto *simple_dim1 = proto.add_simple_sharding();
  simple_dim1->set_dim_value(100);
  simple_dim1->set_num_shards(4);

  auto *simple_dim2 = proto.add_simple_sharding();
  simple_dim2->set_dim_param("height");
  simple_dim2->set_num_shards(2);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("axis:") != std::string::npos);
  EXPECT_TRUE(serialized.find("2") != std::string::npos);
  EXPECT_TRUE(serialized.find("simple_sharding") != std::string::npos);
}

TEST(onnx_string, ShardingSpecProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::ShardingSpecProto proto;
  proto.set_tensor_name("sharded_tensor");

  proto.ref_device().push_back(0);
  proto.ref_device().push_back(1);
  proto.ref_device().push_back(2);

  auto *map_entry = proto.add_index_to_device_group_map();
  map_entry->set_key(0);
  map_entry->ref_value().push_back(0);
  map_entry->ref_value().push_back(1);

  auto *dim = proto.add_sharded_dim();
  dim->set_axis(1);
  auto *simple_dim = dim->add_simple_sharding();
  simple_dim->set_dim_value(64);
  simple_dim->set_num_shards(4);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("tensor_name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("sharded_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("device:") != std::string::npos);
  EXPECT_TRUE(serialized.find("index_to_device_group_map") != std::string::npos);
  EXPECT_TRUE(serialized.find("sharded_dim") != std::string::npos);
}

TEST(onnx_string, NodeDeviceConfigurationProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::NodeDeviceConfigurationProto proto;
  proto.set_configuration_id("node_config_1");
  proto.set_pipeline_stage(3);

  auto *spec = proto.add_sharding_spec();
  spec->set_tensor_name("input_tensor");
  spec->ref_device().push_back(0);
  spec->ref_device().push_back(1);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("configuration_id:") != std::string::npos);
  EXPECT_TRUE(serialized.find("node_config_1") != std::string::npos);
  EXPECT_TRUE(serialized.find("pipeline_stage:") != std::string::npos);
  EXPECT_TRUE(serialized.find("3") != std::string::npos);
  EXPECT_TRUE(serialized.find("sharding_spec") != std::string::npos);
}

TEST(onnx_string, OperatorSetIdProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::OperatorSetIdProto proto;
  proto.set_domain("ai.onnx");
  proto.set_version(15);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("domain:") != std::string::npos);
  EXPECT_TRUE(serialized.find("ai.onnx") != std::string::npos);
  EXPECT_TRUE(serialized.find("version:") != std::string::npos);
  EXPECT_TRUE(serialized.find("15") != std::string::npos);
}

TEST(onnx_string, TensorShapeProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TensorShapeProto proto;

  auto *dim1 = proto.add_dim();
  dim1->set_dim_value(64);

  auto *dim2 = proto.add_dim();
  dim2->set_dim_param("batch");
  dim2->set_denotation("N");

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundDim1 = false;
  bool foundDim2 = false;
  bool foundDenotation = false;

  std::string item = serialized;
  if (item.find("dim") != std::string::npos && item.find("dim_value: 64") != std::string::npos) {
    foundDim1 = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDim2 = true;
  }
  if (item.find("denotation: \"N\"") != std::string::npos) {
    foundDenotation = true;
  }

  EXPECT_TRUE(foundDim1);
  EXPECT_TRUE(foundDim2);
  EXPECT_TRUE(foundDenotation);
}

TEST(onnx_string, TensorProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TensorProto proto;
  proto.set_name("test_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_dims().push_back(3);
  proto.ref_dims().push_back(4);

  for (int i = 0; i < 12; ++i) {
    proto.ref_float_data().push_back(static_cast<float>(i * 0.5f));
  }

  proto.ref_doc_string() = "Un tenseur de test";

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("data_type:") != std::string::npos);
  EXPECT_TRUE(serialized.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
              std::string::npos);
  EXPECT_TRUE(serialized.find("dims:") != std::string::npos);
  EXPECT_TRUE(serialized.find("doc_string:") != std::string::npos);
  EXPECT_TRUE(serialized.find("Un tenseur de test") != std::string::npos);
  EXPECT_TRUE(serialized.find("float_data") != std::string::npos);
}

TEST(onnx_string, SparseTensorProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::SparseTensorProto proto;

  proto.ref_dims().push_back(5);
  proto.ref_dims().push_back(5);

  proto.ref_values().set_name("values_tensor");
  proto.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_values().ref_float_data().push_back(1.5f);
  proto.ref_values().ref_float_data().push_back(2.5f);
  proto.ref_values().ref_float_data().push_back(3.5f);

  proto.ref_indices().set_name("indices_tensor");
  proto.ref_indices().set_data_type(TensorProto::DataType::INT64);
  proto.ref_indices().ref_dims().push_back(3);
  proto.ref_indices().ref_dims().push_back(2);

  proto.ref_indices().ref_int64_data().push_back(0);
  proto.ref_indices().ref_int64_data().push_back(1);
  proto.ref_indices().ref_int64_data().push_back(2);
  proto.ref_indices().ref_int64_data().push_back(3);
  proto.ref_indices().ref_int64_data().push_back(4);
  proto.ref_indices().ref_int64_data().push_back(2);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("dims:") != std::string::npos);
  EXPECT_TRUE(serialized.find("5") != std::string::npos);
  EXPECT_TRUE(serialized.find("values") != std::string::npos);
  EXPECT_TRUE(serialized.find("values_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("indices") != std::string::npos);
  EXPECT_TRUE(serialized.find("indices_tensor") != std::string::npos);
}

TEST(onnx_string, TypeProto) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TypeProto proto;

  proto.add_tensor_type()->set_elem_type(1); // FLOAT

  auto *shape = proto.ref_tensor_type().add_shape();

  auto *dim1 = shape->add_dim();
  dim1->set_dim_value(10);

  auto *dim2 = shape->add_dim();
  dim2->set_dim_param("batch");
  dim2->set_denotation("N");

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundTensorType = false;
  bool foundElemType = false;
  bool foundShape = false;
  bool foundDimValue = false;
  bool foundDimParam = false;

  std::string item = serialized;
  if (item.find("tensor_type") != std::string::npos) {
    foundTensorType = true;
  }
  if (item.find("elem_type: 1") != std::string::npos) {
    foundElemType = true;
  }
  if (item.find("shape") != std::string::npos) {
    foundShape = true;
  }
  if (item.find("dim_value: 10") != std::string::npos) {
    foundDimValue = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDimParam = true;
  }

  EXPECT_TRUE(foundTensorType);
  EXPECT_TRUE(foundElemType);
  EXPECT_TRUE(foundShape);
  EXPECT_TRUE(foundDimValue);
  EXPECT_TRUE(foundDimParam);
}

TEST(onnx_string, TensorProto_WithRawData) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TensorProto proto;
  proto.set_name("raw_data_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_dims().push_back(2);
  proto.ref_dims().push_back(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  proto.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(proto.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("raw_data_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("data_type:") != std::string::npos);
  EXPECT_TRUE(serialized.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
              std::string::npos);
  EXPECT_TRUE(serialized.find("raw_data:") != std::string::npos);
}

TEST(onnx_string, TensorProto_WithSegment) {
  utils::PrintOptions options;
  ONNX_LIGHT_NAMESPACE::TensorProto proto;
  proto.set_name("segmented_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);

  proto.ref_segment().set_begin(5);
  proto.ref_segment().set_end(10);

  std::stringstream ss_result;
  proto.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundSegmentBegin = false;
  bool foundSegmentEnd = false;

  std::string item = serialized;
  if (item.find("name:") != std::string::npos &&
      item.find("segmented_tensor") != std::string::npos) {
    foundName = true;
  }
  if (item.find("segment") != std::string::npos && item.find("begin: 5") != std::string::npos) {
    foundSegmentBegin = true;
  }
  if (item.find("segment") != std::string::npos && item.find("end: 10") != std::string::npos) {
    foundSegmentEnd = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundSegmentBegin);
  EXPECT_TRUE(foundSegmentEnd);
}

TEST(onnx_proto, ValueInfoProto_Basic) {
  ValueInfoProto value_info;

  EXPECT_TRUE(value_info.ref_name().empty());
  EXPECT_TRUE(value_info.ref_doc_string().empty());
  EXPECT_FALSE(value_info.has_type());

  value_info.set_name("input_1");
  value_info.set_doc_string("Input tensor documentation");

  TypeProto *type = value_info.add_type();
  type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  TensorShapeProto::Dimension *dim = shape->add_dim();
  dim->set_dim_value(3);

  EXPECT_EQ(value_info.ref_name(), "input_1");
  EXPECT_EQ(value_info.ref_doc_string(), "Input tensor documentation");
  EXPECT_TRUE(value_info.has_type());
  EXPECT_TRUE(value_info.ref_type().has_tensor_type());
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(value_info.ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 1);
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
}

TEST(onnx_proto, ValueInfoProto_Serialization) {
  ValueInfoProto value_info1;
  value_info1.set_name("output_1");
  value_info1.set_doc_string("Output tensor documentation");

  TypeProto *type = value_info1.add_type();
  type->add_tensor_type()->set_elem_type(7); // INT64
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  shape->add_dim()->set_dim_value(2);
  shape->add_dim()->set_dim_param("dynamic_dim");

  std::string serialized;
  value_info1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), value_info1.SerializeSize().size());

  ValueInfoProto value_info2;
  value_info2.ParseFromString(serialized);

  EXPECT_EQ(value_info2.ref_name(), "output_1");
  EXPECT_EQ(value_info2.ref_doc_string(), "Output tensor documentation");
  EXPECT_TRUE(value_info2.has_type());
  EXPECT_TRUE(value_info2.ref_type().has_tensor_type());
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_elem_type(), 7);
  EXPECT_TRUE(value_info2.ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 2);
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(),
            "dynamic_dim");
}

TEST(onnx_proto, ValueInfoProto_PrintToStringStream) {
  utils::PrintOptions options;
  ValueInfoProto value_info;
  value_info.set_name("feature_vector");
  value_info.set_doc_string("Feature vector description");

  TypeProto *type = value_info.add_type();
  type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(512);

  std::stringstream ss_result;
  value_info.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundDocString = false;
  bool foundType = false;

  if (serialized.find("name:") != std::string::npos &&
      serialized.find("feature_vector") != std::string::npos) {
    foundName = true;
  }
  if (serialized.find("doc_string:") != std::string::npos &&
      serialized.find("Feature vector description") != std::string::npos) {
    foundDocString = true;
  }
  if (serialized.find("type") != std::string::npos &&
      serialized.find("elem_type: 1") != std::string::npos) {
    foundType = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDocString);
  EXPECT_TRUE(foundType);
}

TEST(onnx_proto, CopyFrom_TensorProto) {
  TensorProto source;
  source.set_name("source_tensor");
  source.set_data_type(TensorProto::DataType::FLOAT);
  source.ref_dims().push_back(2);
  source.ref_dims().push_back(3);
  source.ref_float_data().push_back(1.0f);
  source.ref_float_data().push_back(2.0f);
  source.ref_float_data().push_back(3.0f);
  source.ref_raw_data().resize(12);
  source.set_doc_string("Source tensor documentation");

  TensorProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_tensor");
  EXPECT_EQ(target.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(target.ref_dims().size(), 2);
  EXPECT_EQ(target.ref_dims()[0], 2);
  EXPECT_EQ(target.ref_dims()[1], 3);
  EXPECT_EQ(target.ref_float_data().size(), 3);
  EXPECT_EQ(target.ref_float_data()[0], 1.0f);
  EXPECT_EQ(target.ref_float_data()[1], 2.0f);
  EXPECT_EQ(target.ref_float_data()[2], 3.0f);
  EXPECT_EQ(target.ref_raw_data().size(), 12);
  EXPECT_EQ(target.ref_doc_string(), "Source tensor documentation");
}

TEST(onnx_proto, CopyFrom_ValueInfoProto) {
  ValueInfoProto source;
  source.set_name("source_info");
  source.set_doc_string("Source documentation");
  TypeProto *type = source.add_type();
  type->add_tensor_type()->set_elem_type(1);

  ValueInfoProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_info");
  EXPECT_EQ(target.ref_doc_string(), "Source documentation");
  EXPECT_TRUE(target.has_type());
  EXPECT_TRUE(target.ref_type().has_tensor_type());
  EXPECT_EQ(target.ref_type().ref_tensor_type().ref_elem_type(), 1);
}

TEST(onnx_proto, CopyFrom_TypeProto) {
  TypeProto source;
  source.add_tensor_type()->set_elem_type(7);
  TensorShapeProto *shape = source.ref_tensor_type().add_shape();
  shape->add_dim()->set_dim_value(10);
  shape->add_dim()->set_dim_param("N");

  TypeProto target;
  target.CopyFrom(source);

  EXPECT_TRUE(target.has_tensor_type());
  EXPECT_EQ(target.ref_tensor_type().ref_elem_type(), 7);
  EXPECT_TRUE(target.ref_tensor_type().has_shape());
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 10);
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "N");
}

TEST(onnx_proto, CopyFrom_SparseTensorProto) {
  SparseTensorProto source;
  source.ref_dims().push_back(4);
  source.ref_dims().push_back(4);

  source.ref_indices().set_name("indices");
  source.ref_indices().set_data_type(TensorProto::DataType::INT64);
  source.ref_indices().ref_int64_data().push_back(0);
  source.ref_indices().ref_int64_data().push_back(1);

  source.ref_values().set_name("values");
  source.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  source.ref_values().ref_float_data().push_back(1.5f);

  SparseTensorProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_dims().size(), 2);
  EXPECT_EQ(target.ref_dims()[0], 4);
  EXPECT_EQ(target.ref_dims()[1], 4);
  EXPECT_EQ(target.ref_indices().ref_name(), "indices");
  EXPECT_EQ(target.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(target.ref_indices().ref_int64_data().size(), 2);
  EXPECT_EQ(target.ref_values().ref_name(), "values");
  EXPECT_EQ(target.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(target.ref_values().ref_float_data().size(), 1);
  EXPECT_EQ(target.ref_values().ref_float_data()[0], 1.5f);
}

TEST(onnx_proto, AttributeProto_Basic) {
  AttributeProto attribute;

  EXPECT_TRUE(attribute.ref_name().empty());
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::UNDEFINED);
  EXPECT_FALSE(attribute.has_i());
  EXPECT_FALSE(attribute.has_f());
  EXPECT_FALSE(attribute.has_s());
  EXPECT_EQ(attribute.ref_ints().size(), 0);
  EXPECT_EQ(attribute.ref_floats().size(), 0);
  EXPECT_EQ(attribute.ref_strings().size(), 0);

  attribute.set_name("weight_decay");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.01f);

  EXPECT_EQ(attribute.ref_name(), "weight_decay");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_TRUE(attribute.has_f());
  EXPECT_EQ(attribute.ref_f(), 0.01f);
}

TEST(onnx_proto, AttributeProto_IntAttribute) {
  AttributeProto attribute;

  attribute.set_name("axis");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(2);

  EXPECT_EQ(attribute.ref_name(), "axis");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_TRUE(attribute.has_i());
  EXPECT_EQ(attribute.ref_i(), 2);
  EXPECT_FALSE(attribute.has_f());
  EXPECT_FALSE(attribute.has_s());
}

TEST(onnx_proto, AttributeProto_StringAttribute) {
  AttributeProto attribute;

  attribute.set_name("mode");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s("constant");

  EXPECT_EQ(attribute.ref_name(), "mode");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::STRING);
  EXPECT_TRUE(attribute.has_s());
  EXPECT_EQ(attribute.ref_s(), "constant");
  EXPECT_FALSE(attribute.has_i());
  EXPECT_FALSE(attribute.has_f());
}

TEST(onnx_proto, AttributeProto_IntsAttribute) {
  AttributeProto attribute;

  attribute.set_name("pads");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  attribute.ref_ints().push_back(0);
  attribute.ref_ints().push_back(0);
  attribute.ref_ints().push_back(1);
  attribute.ref_ints().push_back(1);

  EXPECT_EQ(attribute.ref_name(), "pads");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(attribute.ref_ints().size(), 4);
  EXPECT_EQ(attribute.ref_ints()[0], 0);
  EXPECT_EQ(attribute.ref_ints()[1], 0);
  EXPECT_EQ(attribute.ref_ints()[2], 1);
  EXPECT_EQ(attribute.ref_ints()[3], 1);
}

TEST(onnx_proto, AttributeProto_FloatsAttribute) {
  AttributeProto attribute;

  attribute.set_name("scales");
  attribute.set_type(AttributeProto::AttributeType::FLOATS);
  attribute.ref_floats().push_back(1.0f);
  attribute.ref_floats().push_back(2.0f);
  attribute.ref_floats().push_back(3.0f);

  EXPECT_EQ(attribute.ref_name(), "scales");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(attribute.ref_floats().size(), 3);
  EXPECT_EQ(attribute.ref_floats()[0], 1.0f);
  EXPECT_EQ(attribute.ref_floats()[1], 2.0f);
  EXPECT_EQ(attribute.ref_floats()[2], 3.0f);
}

TEST(onnx_proto, AttributeProto_StringsAttribute) {
  AttributeProto attribute;

  attribute.set_name("tags");
  attribute.set_type(AttributeProto::AttributeType::STRINGS);
  *attribute.add_strings() = "tag1";
  *attribute.add_strings() = "tag2";
  *attribute.add_strings() = "tag3";

  EXPECT_EQ(attribute.ref_name(), "tags");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::STRINGS);
  EXPECT_EQ(attribute.ref_strings().size(), 3);
  EXPECT_EQ(attribute.ref_strings()[0], "tag1");
  EXPECT_EQ(attribute.ref_strings()[1], "tag2");
  EXPECT_EQ(attribute.ref_strings()[2], "tag3");
}

TEST(onnx_proto, AttributeProto_TensorAttribute) {
  AttributeProto attribute;

  attribute.set_name("value");
  attribute.set_type(AttributeProto::AttributeType::TENSOR);

  TensorProto *tensor = attribute.add_t();
  tensor->set_name("const_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_dims().push_back(2);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);
  tensor->ref_float_data().push_back(3.0f);
  tensor->ref_float_data().push_back(4.0f);

  EXPECT_EQ(attribute.ref_name(), "value");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TENSOR);
  EXPECT_TRUE(attribute.has_t());
  EXPECT_EQ(attribute.ref_t().ref_name(), "const_tensor");
  EXPECT_EQ(attribute.ref_t().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(attribute.ref_t().ref_dims().size(), 2);
  EXPECT_EQ(attribute.ref_t().ref_float_data().size(), 4);
}

TEST(onnx_proto, AttributeProto_Serialization) {
  AttributeProto attribute;
  attribute.set_name("test_attribute");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(42);
  attribute.set_doc_string("Test attribute documentation");

  std::string serialized;
  attribute.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), attribute.SerializeSize().size());

  AttributeProto attribute2;
  attribute2.ParseFromString(serialized);

  EXPECT_EQ(attribute2.ref_name(), "test_attribute");
  EXPECT_EQ(attribute2.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attribute2.ref_i(), 42);
  EXPECT_EQ(attribute2.ref_doc_string(), "Test attribute documentation");
}

TEST(onnx_string, AttributeProto) {
  utils::PrintOptions options;
  AttributeProto attribute;
  attribute.set_name("dropout_ratio");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.5f);
  attribute.set_doc_string("Dropout ratio documentation");

  std::stringstream ss_result;
  attribute.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundValue = false;

  if (serialized.find("dropout_ratio") != std::string::npos) {
    foundName = true;
  }
  if (serialized.find(": 0.5") != std::string::npos) {
    foundValue = true;
  }
  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundValue);
}

TEST(onnx_proto, AttributeProto_CopyFrom) {
  AttributeProto source;
  source.set_name("source_attribute");
  source.set_type(AttributeProto::AttributeType::INTS);
  source.ref_ints().push_back(10);
  source.ref_ints().push_back(20);
  source.set_doc_string("Source documentation");

  AttributeProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_attribute");
  EXPECT_EQ(target.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(target.ref_ints().size(), 2);
  EXPECT_EQ(target.ref_ints()[0], 10);
  EXPECT_EQ(target.ref_ints()[1], 20);
  EXPECT_EQ(target.ref_doc_string(), "Source documentation");
}

TEST(onnx_proto, AttributeProto_GraphAttribute) {
  AttributeProto attribute;

  attribute.set_name("body");
  attribute.set_type(AttributeProto::AttributeType::GRAPH);

  // Assuming GraphProto has methods similar to TensorProto
  attribute.add_g()->set_name("subgraph");

  EXPECT_EQ(attribute.ref_name(), "body");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::GRAPH);
  EXPECT_TRUE(attribute.has_g());
  EXPECT_EQ(attribute.ref_g().ref_name(), "subgraph");
}

// NodeProto

TEST(onnx_proto, NodeProto_Basic) {
  NodeProto node;

  EXPECT_TRUE(node.ref_name().empty());
  EXPECT_TRUE(node.ref_op_type().empty());
  EXPECT_TRUE(node.ref_domain().empty());
  EXPECT_EQ(node.ref_input().size(), 0);
  EXPECT_EQ(node.ref_output().size(), 0);
  EXPECT_EQ(node.ref_attribute().size(), 0);
  EXPECT_TRUE(node.ref_doc_string().empty());

  node.set_name("test_node");
  node.set_op_type("Conv");
  node.set_domain("ai.onnx");
  node.set_doc_string("Test node documentation");

  EXPECT_EQ(node.ref_name(), "test_node");
  EXPECT_EQ(node.ref_op_type(), "Conv");
  EXPECT_EQ(node.ref_domain(), "ai.onnx");
  EXPECT_EQ(node.ref_doc_string(), "Test node documentation");
}

TEST(onnx_proto, NodeProto_InputOutput) {
  NodeProto node;

  EXPECT_EQ(node.ref_input().size(), 0);
  EXPECT_EQ(node.ref_output().size(), 0);

  // Add inputs
  *node.add_input() = "X";
  *node.add_input() = "W";
  *node.add_input() = "B";

  // Add outputs
  *node.add_output() = "Y";

  EXPECT_EQ(node.ref_input().size(), 3);
  EXPECT_EQ(node.ref_input()[0], "X");
  EXPECT_EQ(node.ref_input()[1], "W");
  EXPECT_EQ(node.ref_input()[2], "B");

  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_EQ(node.ref_output()[0], "Y");

  // Test clear_input and clear_output
  node.clr_input();
  EXPECT_EQ(node.ref_input().size(), 0);

  node.clr_output();
  EXPECT_EQ(node.ref_output().size(), 0);
}

TEST(onnx_proto, NodeProto_Attributes) {
  NodeProto node;

  EXPECT_EQ(node.ref_attribute().size(), 0);

  // Add attributes
  AttributeProto *attr1 = node.add_attribute();
  attr1->set_name("kernel_shape");
  attr1->set_type(AttributeProto::AttributeType::INTS);
  attr1->ref_ints().push_back(3);
  attr1->ref_ints().push_back(3);

  AttributeProto *attr2 = node.add_attribute();
  attr2->set_name("strides");
  attr2->set_type(AttributeProto::AttributeType::INTS);
  attr2->ref_ints().push_back(1);
  attr2->ref_ints().push_back(1);

  AttributeProto *attr3 = node.add_attribute();
  attr3->set_name("pads");
  attr3->set_type(AttributeProto::AttributeType::INTS);
  attr3->ref_ints().push_back(1);
  attr3->ref_ints().push_back(1);
  attr3->ref_ints().push_back(1);
  attr3->ref_ints().push_back(1);

  EXPECT_EQ(node.ref_attribute().size(), 3);
  EXPECT_EQ(node.ref_attribute()[0].ref_name(), "kernel_shape");
  EXPECT_EQ(node.ref_attribute()[0].ref_ints().size(), 2);
  EXPECT_EQ(node.ref_attribute()[1].ref_name(), "strides");
  EXPECT_EQ(node.ref_attribute()[1].ref_ints().size(), 2);
  EXPECT_EQ(node.ref_attribute()[2].ref_name(), "pads");
  EXPECT_EQ(node.ref_attribute()[2].ref_ints().size(), 4);
}

TEST(onnx_proto, NodeProto_Serialization) {
  NodeProto node1;
  node1.set_name("conv1");
  node1.set_op_type("Conv");
  node1.set_domain("ai.onnx");

  *node1.add_input() = "X";
  *node1.add_input() = "W";

  *node1.add_output() = "Y";

  AttributeProto *attr = node1.add_attribute();
  attr->set_name("kernel_shape");
  attr->set_type(AttributeProto::AttributeType::INTS);
  attr->ref_ints().push_back(3);
  attr->ref_ints().push_back(3);

  std::string serialized;
  node1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node1.SerializeSize().size());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_EQ(node2.ref_name(), "conv1");
  EXPECT_EQ(node2.ref_op_type(), "Conv");
  EXPECT_EQ(node2.ref_domain(), "ai.onnx");
  EXPECT_EQ(node2.ref_input().size(), 2);
  EXPECT_EQ(node2.ref_input()[0], "X");
  EXPECT_EQ(node2.ref_input()[1], "W");
  EXPECT_EQ(node2.ref_output().size(), 1);
  EXPECT_EQ(node2.ref_output()[0], "Y");
  EXPECT_EQ(node2.ref_attribute().size(), 1);
  EXPECT_EQ(node2.ref_attribute()[0].ref_name(), "kernel_shape");
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints().size(), 2);
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints()[0], 3);
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints()[1], 3);
}

TEST(onnx_string, NodeProto_PrintToStringStream) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("relu1");
  node.set_op_type("Relu");
  *node.add_input() = "X";
  *node.add_output() = "Y";
  node.set_doc_string("Simple ReLU activation");

  std::stringstream ss_result;
  node.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundOpType = false;
  bool foundInput = false;
  bool foundOutput = false;
  bool foundDocString = false;

  if (serialized.find("name:") != std::string::npos &&
      serialized.find("relu1") != std::string::npos) {
    foundName = true;
  }

  if (serialized.find("op_type:") != std::string::npos &&
      serialized.find("Relu") != std::string::npos) {
    foundOpType = true;
  }

  if (serialized.find("input:") != std::string::npos && serialized.find("X") != std::string::npos) {
    foundInput = true;
  }

  if (serialized.find("output:") != std::string::npos &&
      serialized.find("Y") != std::string::npos) {
    foundOutput = true;
  }

  if (serialized.find("doc_string:") != std::string::npos &&
      serialized.find("Simple ReLU activation") != std::string::npos) {
    foundDocString = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundOpType);
  EXPECT_TRUE(foundInput);
  EXPECT_TRUE(foundOutput);
  EXPECT_TRUE(foundDocString);
}

TEST(onnx_proto, NodeProto_CopyFrom) {
  NodeProto source;
  source.set_name("source_node");
  source.set_op_type("Add");
  source.set_domain("ai.onnx");
  *source.add_input() = "A";
  *source.add_input() = "B";
  *source.add_output() = "C";

  AttributeProto *attr = source.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);

  source.set_doc_string("Source node documentation");

  NodeProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_node");
  EXPECT_EQ(target.ref_op_type(), "Add");
  EXPECT_EQ(target.ref_domain(), "ai.onnx");
  EXPECT_EQ(target.ref_input().size(), 2);
  EXPECT_EQ(target.ref_input()[0], "A");
  EXPECT_EQ(target.ref_input()[1], "B");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0], "C");
  EXPECT_EQ(target.ref_attribute().size(), 1);
  EXPECT_EQ(target.ref_attribute()[0].ref_name(), "axis");
  EXPECT_EQ(target.ref_attribute()[0].ref_i(), 1);
  EXPECT_EQ(target.ref_doc_string(), "Source node documentation");
}

TEST(onnx_proto, NodeProto_ComplexModel) {
  // Create a more complex node to test multiple attributes, inputs, outputs
  NodeProto node;
  node.set_name("gemm1");
  node.set_op_type("Gemm");
  node.set_domain("ai.onnx");

  // Add inputs
  *node.add_input() = "A";
  *node.add_input() = "B";
  *node.add_input() = "C";

  // Add outputs
  *node.add_output() = "Y";

  // Add attributes
  AttributeProto *alpha = node.add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(AttributeProto::AttributeType::FLOAT);
  alpha->set_f(0.5f);

  AttributeProto *beta = node.add_attribute();
  beta->set_name("beta");
  beta->set_type(AttributeProto::AttributeType::FLOAT);
  beta->set_f(0.8f);

  AttributeProto *transA = node.add_attribute();
  transA->set_name("transA");
  transA->set_type(AttributeProto::AttributeType::INT);
  transA->set_i(1);

  AttributeProto *transB = node.add_attribute();
  transB->set_name("transB");
  transB->set_type(AttributeProto::AttributeType::INT);
  transB->set_i(0);

  node.set_doc_string("GEMM operation: Y = alpha * A' * B + beta * C");

  EXPECT_EQ(node.ref_name(), "gemm1");
  EXPECT_EQ(node.ref_op_type(), "Gemm");
  EXPECT_EQ(node.ref_domain(), "ai.onnx");

  EXPECT_EQ(node.ref_input().size(), 3);
  EXPECT_EQ(node.ref_input()[0], "A");
  EXPECT_EQ(node.ref_input()[1], "B");
  EXPECT_EQ(node.ref_input()[2], "C");

  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_EQ(node.ref_output()[0], "Y");

  EXPECT_EQ(node.ref_attribute().size(), 4);
  EXPECT_EQ(node.ref_attribute()[0].ref_name(), "alpha");
  EXPECT_EQ(node.ref_attribute()[0].ref_f(), 0.5f);
  EXPECT_EQ(node.ref_attribute()[1].ref_name(), "beta");
  EXPECT_EQ(node.ref_attribute()[1].ref_f(), 0.8f);
  EXPECT_EQ(node.ref_attribute()[2].ref_name(), "transA");
  EXPECT_EQ(node.ref_attribute()[2].ref_i(), 1);
  EXPECT_EQ(node.ref_attribute()[3].ref_name(), "transB");
  EXPECT_EQ(node.ref_attribute()[3].ref_i(), 0);

  EXPECT_EQ(node.ref_doc_string(), "GEMM operation: Y = alpha * A' * B + beta * C");
}

TEST(onnx_proto, NodeProto_EmptyStrings) {
  NodeProto node;
  node.set_name("");
  node.set_op_type("Identity");
  *node.add_input() = "";
  *node.add_output() = "";

  EXPECT_TRUE(node.ref_name().empty());
  EXPECT_EQ(node.ref_op_type(), "Identity");
  EXPECT_EQ(node.ref_input().size(), 1);
  EXPECT_TRUE(node.ref_input()[0].empty());
  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_TRUE(node.ref_output()[0].empty());

  std::string serialized;
  node.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node.SerializeSize().size());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_TRUE(node2.ref_name().empty());
  EXPECT_EQ(node2.ref_op_type(), "Identity");
  EXPECT_EQ(node2.ref_input().size(), 1);
  EXPECT_TRUE(node2.ref_input()[0].empty());
  EXPECT_EQ(node2.ref_output().size(), 1);
  EXPECT_TRUE(node2.ref_output()[0].empty());
}

// GraphProto

TEST(onnx_proto, GraphProto_Basic) {
  GraphProto graph;

  EXPECT_TRUE(graph.ref_name().empty());
  EXPECT_EQ(graph.ref_node().size(), 0);
  EXPECT_EQ(graph.ref_initializer().size(), 0);
  EXPECT_EQ(graph.ref_input().size(), 0);
  EXPECT_EQ(graph.ref_output().size(), 0);
  EXPECT_EQ(graph.ref_value_info().size(), 0);
  EXPECT_TRUE(graph.ref_doc_string().empty());

  graph.set_name("test_graph");
  graph.set_doc_string("Test graph documentation");

  EXPECT_EQ(graph.ref_name(), "test_graph");
  EXPECT_EQ(graph.ref_doc_string(), "Test graph documentation");
}

TEST(onnx_proto, GraphProto_Nodes) {
  GraphProto graph;
  graph.set_name("test_graph");

  // Add nodes
  NodeProto *node1 = graph.add_node();
  node1->set_name("conv1");
  node1->set_op_type("Conv");
  *node1->add_input() = "X";
  *node1->add_input() = "W";
  *node1->add_output() = "Y";

  NodeProto *node2 = graph.add_node();
  node2->set_name("relu1");
  node2->set_op_type("Relu");
  *node2->add_input() = "Y";
  *node2->add_output() = "Z";

  EXPECT_EQ(graph.ref_node().size(), 2);
  EXPECT_EQ(graph.ref_node()[0].ref_name(), "conv1");
  EXPECT_EQ(graph.ref_node()[0].ref_op_type(), "Conv");
  EXPECT_EQ(graph.ref_node()[1].ref_name(), "relu1");
  EXPECT_EQ(graph.ref_node()[1].ref_op_type(), "Relu");
}

TEST(onnx_proto, GraphProto_Inputs) {
  GraphProto graph;

  ValueInfoProto *input1 = graph.add_input();
  input1->set_name("X");
  TypeProto *type1 = input1->add_type();
  type1->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape1 = type1->ref_tensor_type().add_shape();
  shape1->add_dim()->set_dim_value(1);
  shape1->add_dim()->set_dim_value(3);
  shape1->add_dim()->set_dim_value(224);
  shape1->add_dim()->set_dim_value(224);

  ValueInfoProto *input2 = graph.add_input();
  input2->set_name("W");
  TypeProto *type2 = input2->add_type();
  type2->add_tensor_type()->set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_input().size(), 2);
  EXPECT_EQ(graph.ref_input()[0].ref_name(), "X");
  EXPECT_EQ(graph.ref_input()[1].ref_name(), "W");
  EXPECT_TRUE(graph.ref_input()[0].has_type());
  EXPECT_TRUE(graph.ref_input()[0].ref_type().has_tensor_type());
  EXPECT_EQ(graph.ref_input()[0].ref_type().ref_tensor_type().ref_elem_type(), 1);
}

TEST(onnx_proto, GraphProto_Outputs) {
  GraphProto graph;

  ValueInfoProto *output = graph.add_output();
  output->set_name("Z");
  TypeProto *type = output->add_type();
  type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(64);
  shape->add_dim()->set_dim_value(112);
  shape->add_dim()->set_dim_value(112);

  EXPECT_EQ(graph.ref_output().size(), 1);
  EXPECT_EQ(graph.ref_output()[0].ref_name(), "Z");
  EXPECT_TRUE(graph.ref_output()[0].has_type());
  EXPECT_TRUE(graph.ref_output()[0].ref_type().has_tensor_type());
  EXPECT_TRUE(graph.ref_output()[0].ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(graph.ref_output()[0].ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 4);
}

TEST(onnx_proto, GraphProto_ValueInfo) {
  GraphProto graph;

  ValueInfoProto *value_info = graph.add_value_info();
  value_info->set_name("Y");
  TypeProto *type = value_info->add_type();
  type->add_tensor_type()->set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_value_info().size(), 1);
  EXPECT_EQ(graph.ref_value_info()[0].ref_name(), "Y");
}

TEST(onnx_proto, GraphProto_Initializers) {
  GraphProto graph;

  TensorProto *initializer = graph.add_initializer();
  initializer->set_name("W");
  initializer->set_data_type(TensorProto::DataType::FLOAT);
  initializer->ref_dims().push_back(64);
  initializer->ref_dims().push_back(3);
  initializer->ref_dims().push_back(3);
  initializer->ref_dims().push_back(3);

  for (int i = 0; i < 64 * 3 * 3 * 3; ++i) {
    initializer->ref_float_data().push_back(static_cast<float>(i) * 0.01f);
  }

  EXPECT_EQ(graph.ref_initializer().size(), 1);
  EXPECT_EQ(graph.ref_initializer()[0].ref_name(), "W");
  EXPECT_EQ(graph.ref_initializer()[0].ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(graph.ref_initializer()[0].ref_dims().size(), 4);
  EXPECT_EQ(graph.ref_initializer()[0].ref_float_data().size(), 64 * 3 * 3 * 3);
}

TEST(onnx_proto, GraphProto_Serialization) {
  GraphProto graph1;
  graph1.set_name("serialization_test");

  NodeProto *node = graph1.add_node();
  node->set_name("node1");
  node->set_op_type("Identity");
  *node->add_input() = "X";
  *node->add_output() = "Y";

  ValueInfoProto *input = graph1.add_input();
  input->set_name("X");

  ValueInfoProto *output = graph1.add_output();
  output->set_name("Y");

  std::string serialized;
  graph1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), graph1.SerializeSize().size());

  GraphProto graph2;
  graph2.ParseFromString(serialized);

  EXPECT_EQ(graph2.ref_name(), "serialization_test");
  EXPECT_EQ(graph2.ref_node().size(), 1);
  EXPECT_EQ(graph2.ref_node()[0].ref_name(), "node1");
  EXPECT_EQ(graph2.ref_node()[0].ref_op_type(), "Identity");
  EXPECT_EQ(graph2.ref_input().size(), 1);
  EXPECT_EQ(graph2.ref_input()[0].ref_name(), "X");
  EXPECT_EQ(graph2.ref_output().size(), 1);
  EXPECT_EQ(graph2.ref_output()[0].ref_name(), "Y");
}

TEST(onnx_proto, GraphProto_PrintToStringStream) {
  utils::PrintOptions options;
  GraphProto graph;
  graph.set_name("vector_serialization_test");
  graph.set_doc_string("Test graph for vector serialization");

  NodeProto *node = graph.add_node();
  node->set_name("add_node");
  node->set_op_type("Add");
  *node->add_input() = "A";
  *node->add_input() = "B";
  *node->add_output() = "C";

  std::stringstream ss_result;
  graph.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  bool foundName = false;
  bool foundDocString = false;
  bool foundNode = false;

  if (serialized.find("name:") != std::string::npos &&
      serialized.find("vector_serialization_test") != std::string::npos) {
    foundName = true;
  }

  if (serialized.find("doc_string:") != std::string::npos &&
      serialized.find("Test graph for vector serialization") != std::string::npos) {
    foundDocString = true;
  }

  if (serialized.find("node") != std::string::npos &&
      serialized.find("add_node") != std::string::npos) {
    foundNode = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDocString);
  EXPECT_TRUE(foundNode);
}

TEST(onnx_proto, GraphProto_CopyFrom) {
  GraphProto source;
  source.set_name("source_graph");
  source.set_doc_string("Source graph documentation");

  NodeProto *node = source.add_node();
  node->set_name("test_node");
  node->set_op_type("Test");

  ValueInfoProto *input = source.add_input();
  input->set_name("input_tensor");

  ValueInfoProto *output = source.add_output();
  output->set_name("output_tensor");

  TensorProto *initializer = source.add_initializer();
  initializer->set_name("weights");
  initializer->set_data_type(TensorProto::DataType::FLOAT);

  GraphProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_graph");
  EXPECT_EQ(target.ref_doc_string(), "Source graph documentation");
  EXPECT_EQ(target.ref_node().size(), 1);
  EXPECT_EQ(target.ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(target.ref_input().size(), 1);
  EXPECT_EQ(target.ref_input()[0].ref_name(), "input_tensor");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0].ref_name(), "output_tensor");
  EXPECT_EQ(target.ref_initializer().size(), 1);
  EXPECT_EQ(target.ref_initializer()[0].ref_name(), "weights");
}

TEST(onnx_proto, GraphProto_ComplexModel) {
  GraphProto graph;
  graph.set_name("complex_model");

  // Create input
  ValueInfoProto *input = graph.add_input();
  input->set_name("data");
  TypeProto *input_type = input->add_type();
  input_type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *input_shape = input_type->ref_tensor_type().add_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(224);
  input_shape->add_dim()->set_dim_value(224);

  // Create weights initializer
  TensorProto *weights = graph.add_initializer();
  weights->set_name("conv1_weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(64);
  weights->ref_dims().push_back(3);
  weights->ref_dims().push_back(7);
  weights->ref_dims().push_back(7);

  // Create bias initializer
  TensorProto *bias = graph.add_initializer();
  bias->set_name("conv1_bias");
  bias->set_data_type(TensorProto::DataType::FLOAT);
  bias->ref_dims().push_back(64);

  // Add Conv node
  NodeProto *conv = graph.add_node();
  conv->set_name("conv1");
  conv->set_op_type("Conv");
  *conv->add_input() = "data";
  *conv->add_input() = "conv1_weights";
  *conv->add_input() = "conv1_bias";
  *conv->add_output() = "conv1_output";

  AttributeProto *strides = conv->add_attribute();
  strides->set_name("strides");
  strides->set_type(AttributeProto::AttributeType::INTS);
  strides->ref_ints().push_back(2);
  strides->ref_ints().push_back(2);

  AttributeProto *kernel_shape = conv->add_attribute();
  kernel_shape->set_name("kernel_shape");
  kernel_shape->set_type(AttributeProto::AttributeType::INTS);
  kernel_shape->ref_ints().push_back(7);
  kernel_shape->ref_ints().push_back(7);

  AttributeProto *pads = conv->add_attribute();
  pads->set_name("pads");
  pads->set_type(AttributeProto::AttributeType::INTS);
  pads->ref_ints().push_back(3);
  pads->ref_ints().push_back(3);
  pads->ref_ints().push_back(3);
  pads->ref_ints().push_back(3);

  // Add ReLU node
  NodeProto *relu = graph.add_node();
  relu->set_name("relu1");
  relu->set_op_type("Relu");
  *relu->add_input() = "conv1_output";
  *relu->add_output() = "relu1_output";

  // Add output
  ValueInfoProto *output = graph.add_output();
  output->set_name("relu1_output");
  TypeProto *output_type = output->add_type();
  output_type->add_tensor_type()->set_elem_type(1); // FLOAT

  // Add intermediate value info
  ValueInfoProto *intermediate = graph.add_value_info();
  intermediate->set_name("conv1_output");
  TypeProto *intermediate_type = intermediate->add_type();
  intermediate_type->add_tensor_type()->set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_name(), "complex_model");
  EXPECT_EQ(graph.ref_node().size(), 2);
  EXPECT_EQ(graph.ref_initializer().size(), 2);
  EXPECT_EQ(graph.ref_input().size(), 1);
  EXPECT_EQ(graph.ref_output().size(), 1);
  EXPECT_EQ(graph.ref_value_info().size(), 1);
}

// FunctionProto

TEST(onnx_proto, FunctionProto_Basic) {
  FunctionProto function;

  EXPECT_TRUE(function.ref_name().empty());
  EXPECT_TRUE(function.ref_domain().empty());
  EXPECT_EQ(function.ref_input().size(), 0);
  EXPECT_EQ(function.ref_output().size(), 0);
  EXPECT_EQ(function.ref_attribute().size(), 0);
  EXPECT_EQ(function.ref_node().size(), 0);
  EXPECT_TRUE(function.ref_doc_string().empty());

  function.set_name("test_function");
  function.set_domain("ai.custom");
  function.set_doc_string("Test function documentation");

  // Add inputs
  *function.add_input() = "X";
  *function.add_input() = "W";

  // Add outputs
  *function.add_output() = "Y";

  // Add attributes
  *function.add_attribute() = "alpha";
  *function.add_attribute() = "beta";

  EXPECT_EQ(function.ref_name(), "test_function");
  EXPECT_EQ(function.ref_domain(), "ai.custom");
  EXPECT_EQ(function.ref_doc_string(), "Test function documentation");
  EXPECT_EQ(function.ref_input().size(), 2);
  EXPECT_EQ(function.ref_input()[0], "X");
  EXPECT_EQ(function.ref_input()[1], "W");
  EXPECT_EQ(function.ref_output().size(), 1);
  EXPECT_EQ(function.ref_output()[0], "Y");
  EXPECT_EQ(function.ref_attribute().size(), 2);
  EXPECT_EQ(function.ref_attribute()[0], "alpha");
  EXPECT_EQ(function.ref_attribute()[1], "beta");
}

TEST(onnx_proto, FunctionProto_Nodes) {
  FunctionProto function;
  function.set_name("custom_op");

  // Add nodes
  NodeProto *node1 = function.add_node();
  node1->set_name("mul");
  node1->set_op_type("Mul");
  *node1->add_input() = "X";
  *node1->add_input() = "W";
  *node1->add_output() = "XW";

  NodeProto *node2 = function.add_node();
  node2->set_name("add");
  node2->set_op_type("Add");
  *node2->add_input() = "XW";
  *node2->add_input() = "B";
  *node2->add_output() = "Y";

  EXPECT_EQ(function.ref_node().size(), 2);
  EXPECT_EQ(function.ref_node()[0].ref_name(), "mul");
  EXPECT_EQ(function.ref_node()[0].ref_op_type(), "Mul");
  EXPECT_EQ(function.ref_node()[1].ref_name(), "add");
  EXPECT_EQ(function.ref_node()[1].ref_op_type(), "Add");
}

TEST(onnx_proto, FunctionProto_Serialization) {
  FunctionProto function1;
  function1.set_name("serialization_function");
  function1.set_domain("ai.test");
  function1.set_doc_string("Function for serialization testing");

  *function1.add_input() = "X";
  *function1.add_output() = "Y";
  *function1.add_attribute() = "param";

  NodeProto *node = function1.add_node();
  node->set_name("op1");
  node->set_op_type("CustomOp");
  *node->add_input() = "X";
  *node->add_output() = "Y";

  std::string serialized;
  function1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), function1.SerializeSize().size());

  FunctionProto function2;
  function2.ParseFromString(serialized);

  EXPECT_EQ(function2.ref_name(), "serialization_function");
  EXPECT_EQ(function2.ref_domain(), "ai.test");
  EXPECT_EQ(function2.ref_doc_string(), "Function for serialization testing");
  EXPECT_EQ(function2.ref_input().size(), 1);
  EXPECT_EQ(function2.ref_input()[0], "X");
  EXPECT_EQ(function2.ref_output().size(), 1);
  EXPECT_EQ(function2.ref_output()[0], "Y");
  EXPECT_EQ(function2.ref_attribute().size(), 1);
  EXPECT_EQ(function2.ref_attribute()[0], "param");
  EXPECT_EQ(function2.ref_node().size(), 1);
  EXPECT_EQ(function2.ref_node()[0].ref_name(), "op1");
}

TEST(onnx_proto, FunctionProto_CopyFrom) {
  FunctionProto source;
  source.set_name("source_function");
  source.set_domain("ai.source");
  *source.add_input() = "X";
  *source.add_output() = "Y";
  *source.add_attribute() = "attr1";

  NodeProto *node = source.add_node();
  node->set_op_type("Identity");

  source.set_doc_string("Source function documentation");

  FunctionProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_function");
  EXPECT_EQ(target.ref_domain(), "ai.source");
  EXPECT_EQ(target.ref_input().size(), 1);
  EXPECT_EQ(target.ref_input()[0], "X");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0], "Y");
  EXPECT_EQ(target.ref_attribute().size(), 1);
  EXPECT_EQ(target.ref_attribute()[0], "attr1");
  EXPECT_EQ(target.ref_node().size(), 1);
  EXPECT_EQ(target.ref_node()[0].ref_op_type(), "Identity");
  EXPECT_EQ(target.ref_doc_string(), "Source function documentation");
}

TEST(onnx_string, FunctionProto) {
  utils::PrintOptions options;
  FunctionProto function;
  function.set_name("my_function");
  function.set_domain("ai.custom");
  *function.add_input() = "input1";
  *function.add_input() = "input2";
  *function.add_output() = "output";
  *function.add_attribute() = "attr";
  function.set_doc_string("Custom function implementation");

  NodeProto *node = function.add_node();
  node->set_name("operation");
  node->set_op_type("MatMul");

  std::stringstream ss_result;
  function.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("name: \"my_function\"") != std::string::npos);
  EXPECT_TRUE(serialized.find("domain: \"ai.custom\"") != std::string::npos);
  EXPECT_TRUE(serialized.find("input:") != std::string::npos);
  EXPECT_TRUE(serialized.find("output:") != std::string::npos);
  EXPECT_TRUE(serialized.find("attribute:") != std::string::npos);
  EXPECT_TRUE(serialized.find("node: [") != std::string::npos);
  EXPECT_TRUE(serialized.find("doc_string:") != std::string::npos);
}

// ModelProto

TEST(onnx_proto, ModelProto_Basic) {
  ModelProto model;

  EXPECT_TRUE(model.ref_producer_name().empty());
  EXPECT_TRUE(model.ref_producer_version().empty());
  EXPECT_TRUE(model.ref_domain().empty());
  EXPECT_EQ(model.ref_model_version(), 0);
  EXPECT_TRUE(model.ref_doc_string().empty());
  EXPECT_FALSE(model.has_graph());
  EXPECT_EQ(model.ref_opset_import().size(), 0);
  EXPECT_EQ(model.ref_metadata_props().size(), 0);

  model.set_ir_version(1);
  model.set_producer_name("test_producer");
  model.set_producer_version("1.0.0");
  model.set_domain("ai.test");
  model.set_model_version(1);
  model.set_doc_string("Test model documentation");

  EXPECT_EQ(model.ref_ir_version(), 1);
  EXPECT_EQ(model.ref_producer_name(), "test_producer");
  EXPECT_EQ(model.ref_producer_version(), "1.0.0");
  EXPECT_EQ(model.ref_domain(), "ai.test");
  EXPECT_EQ(model.ref_model_version(), 1);
  EXPECT_EQ(model.ref_doc_string(), "Test model documentation");
}

TEST(onnx_proto, ModelProto_Graph) {
  ModelProto model;

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Add");

  EXPECT_TRUE(model.has_graph());
  EXPECT_EQ(model.ref_graph().ref_name(), "test_graph");
  EXPECT_EQ(model.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(model.ref_graph().ref_node()[0].ref_op_type(), "Add");
}

TEST(onnx_proto, ModelProto_OpsetImport) {
  ModelProto model;

  OperatorSetIdProto *opset1 = model.add_opset_import();
  opset1->set_domain("ai.onnx");
  opset1->set_version(12);

  OperatorSetIdProto *opset2 = model.add_opset_import();
  opset2->set_domain("ai.onnx.ml");
  opset2->set_version(2);

  EXPECT_EQ(model.ref_opset_import().size(), 2);
  EXPECT_EQ(model.ref_opset_import()[0].ref_domain(), "ai.onnx");
  EXPECT_EQ(model.ref_opset_import()[0].ref_version(), 12);
  EXPECT_EQ(model.ref_opset_import()[1].ref_domain(), "ai.onnx.ml");
  EXPECT_EQ(model.ref_opset_import()[1].ref_version(), 2);
}

TEST(onnx_proto, ModelProto_MetadataProps) {
  ModelProto model;

  StringStringEntryProto *metadata1 = model.add_metadata_props();
  metadata1->set_key("author");
  metadata1->set_value("test_author");

  StringStringEntryProto *metadata2 = model.add_metadata_props();
  metadata2->set_key("description");
  metadata2->set_value("test description");

  EXPECT_EQ(model.ref_metadata_props().size(), 2);
  EXPECT_EQ(model.ref_metadata_props()[0].ref_key(), "author");
  EXPECT_EQ(model.ref_metadata_props()[0].ref_value(), "test_author");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_key(), "description");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_value(), "test description");
}

TEST(onnx_proto, ModelProto_Serialization) {
  ModelProto model1;
  model1.set_ir_version(1);
  model1.set_producer_name("serialization_test");
  model1.set_model_version(42);

  GraphProto *graph = model1.add_graph();
  graph->set_name("serialized_graph");

  NodeProto *node = graph->add_node();
  node->set_name("test_node");
  node->set_op_type("Identity");

  StringStringEntryProto *metadata = model1.add_metadata_props();
  metadata->set_key("test_key");
  metadata->set_value("test_value");

  std::string serialized;
  model1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), model1.SerializeSize().size());

  ModelProto model2;
  model2.ParseFromString(serialized);

  EXPECT_EQ(model2.ref_ir_version(), 1);
  EXPECT_EQ(model2.ref_producer_name(), "serialization_test");
  EXPECT_EQ(model2.ref_model_version(), 42);
  EXPECT_TRUE(model2.has_graph());
  EXPECT_EQ(model2.ref_graph().ref_name(), "serialized_graph");
  EXPECT_EQ(model2.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model2.ref_graph().ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(model2.ref_metadata_props().size(), 1);
  EXPECT_EQ(model2.ref_metadata_props()[0].ref_key(), "test_key");
  EXPECT_EQ(model2.ref_metadata_props()[0].ref_value(), "test_value");
}

TEST(onnx_proto, ModelProto_PrintToStringStream) {
  utils::PrintOptions options;
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_producer");
  model.set_doc_string("Model documentation");
  model.add_graph()->set_name("test_graph");

  std::stringstream ss_result;
  model.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("ir_version:") != std::string::npos);
  EXPECT_TRUE(serialized.find("producer_name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_producer") != std::string::npos);
  EXPECT_TRUE(serialized.find("doc_string:") != std::string::npos);
  EXPECT_TRUE(serialized.find("Model documentation") != std::string::npos);
  EXPECT_TRUE(serialized.find("graph") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_graph") != std::string::npos);
}

TEST(onnx_proto, ModelProto_CopyFrom) {
  ModelProto source;
  source.set_ir_version(2);
  source.set_producer_name("source_producer");
  source.set_model_version(123);
  source.add_graph()->set_name("source_graph");

  OperatorSetIdProto *opset = source.add_opset_import();
  opset->set_domain("ai.onnx");
  opset->set_version(15);

  StringStringEntryProto *metadata = source.add_metadata_props();
  metadata->set_key("source_key");
  metadata->set_value("source_value");

  ModelProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_ir_version(), 2);
  EXPECT_EQ(target.ref_producer_name(), "source_producer");
  EXPECT_EQ(target.ref_model_version(), 123);
  EXPECT_TRUE(target.has_graph());
  EXPECT_EQ(target.ref_graph().ref_name(), "source_graph");
  EXPECT_EQ(target.ref_opset_import().size(), 1);
  EXPECT_EQ(target.ref_opset_import()[0].ref_domain(), "ai.onnx");
  EXPECT_EQ(target.ref_opset_import()[0].ref_version(), 15);
  EXPECT_EQ(target.ref_metadata_props().size(), 1);
  EXPECT_EQ(target.ref_metadata_props()[0].ref_key(), "source_key");
  EXPECT_EQ(target.ref_metadata_props()[0].ref_value(), "source_value");
}

TEST(onnx_proto, ModelProto_ComplexModel) {
  ModelProto model;
  model.set_ir_version(3);
  model.set_producer_name("complex_model_producer");
  model.set_producer_version("1.0.0");
  model.set_model_version(1);

  OperatorSetIdProto *opset = model.add_opset_import();
  opset->set_domain("ai.onnx");
  opset->set_version(13);

  GraphProto *graph = model.add_graph();
  graph->set_name("complex_model_graph");

  // Add input
  ValueInfoProto *input = graph->add_input();
  input->set_name("input_tensor");
  TypeProto *input_type = input->add_type();
  input_type->add_tensor_type()->set_elem_type(1); // FLOAT

  // Add initializer
  TensorProto *weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(TensorProto::DataType::FLOAT);
  weights->ref_dims().push_back(3);
  weights->ref_dims().push_back(3);

  // Add node
  NodeProto *node = graph->add_node();
  node->set_name("matmul_node");
  node->set_op_type("MatMul");
  *node->add_input() = "input_tensor";
  *node->add_input() = "weights";
  *node->add_output() = "output_tensor";

  // Add output
  ValueInfoProto *output = graph->add_output();
  output->set_name("output_tensor");

  // Add metadata
  StringStringEntryProto *metadata = model.add_metadata_props();
  metadata->set_key("framework");
  metadata->set_value("test_framework");

  EXPECT_EQ(model.ref_ir_version(), 3);
  EXPECT_EQ(model.ref_producer_name(), "complex_model_producer");
  EXPECT_EQ(model.ref_model_version(), 1);
  EXPECT_TRUE(model.has_graph());

  EXPECT_EQ(model.ref_graph().ref_input().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_initializer().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_output().size(), 1);

  EXPECT_EQ(model.ref_opset_import().size(), 1);
  EXPECT_EQ(model.ref_opset_import()[0].ref_version(), 13);

  EXPECT_EQ(model.ref_metadata_props().size(), 1);
  EXPECT_EQ(model.ref_metadata_props()[0].ref_key(), "framework");
}

TEST(onnx_proto, AttributeProto_InNodeProto1) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  AttributeProto *attr1 = node.add_attribute();
  attr1->set_type(AttributeProto::AttributeType::INT);
  attr1->ref_i() = 2;
  AttributeProto att2;
  att2.set_type(AttributeProto::AttributeType::INT);
  att2.ref_i() = 2;
  node.ref_attribute().push_back(att2);
  std::stringstream ss_s1;
  node.ref_attribute()[0].PrintToStringStream(ss_s1, options);
  std::string s1 = ss_s1.str();
  std::stringstream ss_s2;
  node.ref_attribute()[1].PrintToStringStream(ss_s2, options);
  std::string s2 = ss_s2.str();
  EXPECT_EQ(s1, s2);
  std::stringstream ss_s4;
  att2.PrintToStringStream(ss_s4, options);
  std::string s4 = ss_s4.str();
  EXPECT_EQ(s1, s4);
}

TEST(onnx_proto, AttributeProto_InNodeProto2) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  AttributeProto *attr1 = node.add_attribute();
  attr1->set_type(AttributeProto::AttributeType::INT);
  attr1->ref_i() = 2;
  AttributeProto *att2 = node.add_attribute();
  att2->set_type(AttributeProto::AttributeType::INT);
  att2->ref_i() = 2;
  std::stringstream ss_s1;
  node.ref_attribute()[0].PrintToStringStream(ss_s1, options);
  std::string s1 = ss_s1.str();
  std::stringstream ss_s2;
  node.ref_attribute()[1].PrintToStringStream(ss_s2, options);
  std::string s2 = ss_s2.str();
  EXPECT_EQ(s1, s2);
  std::stringstream ss_s4;
  att2->PrintToStringStream(ss_s4, options);
  std::string s4 = ss_s4.str();
  EXPECT_EQ(s1, s4);
}

TEST(onnx_proto, TensorProto_SkipRawData) {
  TensorProto tensor1;
  tensor1.set_name("skip_raw_test");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);

  // Add raw data
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  tensor1.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor1.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::string serialized1;
  SerializeOptions options;
  utils::StringWriteStream st;
  tensor1.SerializeToString(serialized1, options);
  EXPECT_EQ(serialized1.size(), tensor1.SerializeSize(st, options).size());

  std::string serialized2;
  SerializeOptions options2;
  options2.skip_raw_data = true;
  options2.raw_data_threshold = 0;
  tensor1.SerializeToString(serialized2, options2);
  EXPECT_EQ(serialized1.size(), 39);
  EXPECT_EQ(serialized2.size(), 21);
  EXPECT_EQ(serialized2.size(), tensor1.SerializeSize(st, options2).size());

  // Test with skip_raw_data = false (default behavior)
  ParseOptions parse_options;
  parse_options.skip_raw_data = true;
  parse_options.raw_data_threshold = 0;
  TensorProto tensor2;
  tensor2.ParseFromString(serialized1, parse_options);

  EXPECT_EQ(tensor2.ref_name(), "skip_raw_test");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_raw_data().size(), 0);
}

TEST(onnx_proto, TensorProto_NoCopyRawData) {
  // Build a TensorProto with raw data.
  TensorProto tensor1;
  tensor1.set_name("no_copy_test");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  tensor1.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor1.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  // Serialize to an owned string that we will keep alive.
  std::string serialized;
  SerializeOptions sopts;
  tensor1.SerializeToString(serialized, sopts);

  // Parse with no_copy=true.
  ParseOptions no_copy_opts;
  no_copy_opts.no_copy = true;
  TensorProto tensor2;
  tensor2.ParseFromString(serialized, no_copy_opts);

  // raw_data_ should be in borrowed mode; nc ptr should be set.
  EXPECT_TRUE(tensor2.ref_raw_data().is_borrowed());
  EXPECT_EQ(tensor2.ref_raw_data().size(), data.size() * sizeof(float));
  EXPECT_TRUE(tensor2.has_raw_data());

  // The pointer should point inside `serialized`.
  const uint8_t *ser_start = reinterpret_cast<const uint8_t *>(serialized.data());
  const uint8_t *ser_end = ser_start + serialized.size();
  const utils::ByteSpan &raw_span = tensor2.ref_raw_data(); // use const ref for read-only access
  EXPECT_GE(raw_span.data(), ser_start);
  EXPECT_LT(raw_span.data(), ser_end);

  // Data values should be correct.
  const float *raw_ptr = reinterpret_cast<const float *>(raw_span.data());
  EXPECT_FLOAT_EQ(raw_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(raw_ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(raw_ptr[2], 3.0f);
  EXPECT_FLOAT_EQ(raw_ptr[3], 4.0f);

  // Other fields should be correctly populated.
  EXPECT_EQ(tensor2.ref_name(), "no_copy_test");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2u);

  // A round-trip serialization from a no-copy tensor should produce the same bytes.
  std::string reserialized;
  tensor2.SerializeToString(reserialized, sopts);
  EXPECT_EQ(serialized, reserialized);

  // SerializeSize should also be consistent.
  utils::StringWriteStream st;
  EXPECT_EQ(reserialized.size(), tensor2.SerializeSize(st, sopts).size());
}

TEST(onnx_stream, FileWriteStream) {
  std::string temp_filename = "test_file_write_stream.tmp";

  // read
  {
    utils::FileWriteStream stream(temp_filename);

    stream.write_variant_uint64(150);
    stream.write_int64(42);
    stream.write_int32(24);
    stream.write_float(3.14f);
    stream.write_string("hello");

    EXPECT_GT(stream.size(), 0);
  }

  // check content
  {
    utils::FileStream readStream(temp_filename);

    EXPECT_EQ(readStream.next_uint64(), 150);
    EXPECT_EQ(readStream.next_int64(), 42);
    EXPECT_EQ(readStream.next_int32(), 24);
    EXPECT_NEAR(readStream.next_float(), 3.14f, 0.0001f);

    utils::RefString str = readStream.next_string();
    EXPECT_EQ(str, "hello");
    EXPECT_FALSE(readStream.NotEnd());
  }

  // Clean up
  std::remove(temp_filename.c_str());
}

TEST(onnx_stream, FileStream_TensorProto) {
  std::string temp_filename = "test_tensor_file_stream.tmp";

  // create and save a TensorProto to a file
  {
    TensorProto tensor;
    tensor.set_name("test_tensor");
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.ref_dims().push_back(2);
    tensor.ref_dims().push_back(3);

    // Add float data
    tensor.ref_float_data().push_back(1.1f);
    tensor.ref_float_data().push_back(2.2f);
    tensor.ref_float_data().push_back(3.3f);
    tensor.ref_float_data().push_back(4.4f);
    tensor.ref_float_data().push_back(5.5f);
    tensor.ref_float_data().push_back(6.6f);

    // Serialize to a file
    utils::FileWriteStream stream(temp_filename);
    std::string serialized;
    tensor.SerializeToString(serialized);

    // Write the size followed by the serialized data
    stream.write_variant_uint64(serialized.size());
    stream.write_raw_bytes(reinterpret_cast<const uint8_t *>(serialized.data()), serialized.size());
  }

  // Read and deserialize the TensorProto from the file
  {
    utils::FileStream stream(temp_filename);

    // Read the size and data
    uint64_t size = stream.next_uint64();
    std::vector<uint8_t> buffer(size);
    const uint8_t *data = stream.read_bytes(size);
    std::memcpy(buffer.data(), data, size);

    // Deserialize the TensorProto
    TensorProto tensor;
    tensor.ParseFromString(
        std::string(reinterpret_cast<const char *>(buffer.data()), buffer.size()));

    // Check properties
    EXPECT_EQ(tensor.ref_name(), "test_tensor");
    EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
    EXPECT_EQ(tensor.ref_dims().size(), 2);
    EXPECT_EQ(tensor.ref_dims()[0], 2);
    EXPECT_EQ(tensor.ref_dims()[1], 3);

    // Check data
    ASSERT_EQ(tensor.ref_float_data().size(), 6);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[0], 1.1f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[1], 2.2f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[2], 3.3f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[3], 4.4f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[4], 5.5f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[5], 6.6f);
  }

  // Clean up
  std::remove(temp_filename.c_str());
}

TEST(onnx_proto, AttributeProto_TensorsAttribute) {
  AttributeProto attribute;

  attribute.set_name("weights");
  attribute.set_type(AttributeProto::AttributeType::TENSORS);

  TensorProto *tensor1 = attribute.add_tensors();
  tensor1->set_name("tensor1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_dims().push_back(2);
  tensor1->ref_dims().push_back(3);
  tensor1->ref_float_data().push_back(1.0f);
  tensor1->ref_float_data().push_back(2.0f);
  tensor1->ref_float_data().push_back(3.0f);
  tensor1->ref_float_data().push_back(4.0f);
  tensor1->ref_float_data().push_back(5.0f);
  tensor1->ref_float_data().push_back(6.0f);

  TensorProto *tensor2 = attribute.add_tensors();
  tensor2->set_name("tensor2");
  tensor2->set_data_type(TensorProto::DataType::INT32);
  tensor2->ref_dims().push_back(2);
  tensor2->ref_int32_data().push_back(10);
  tensor2->ref_int32_data().push_back(20);

  EXPECT_EQ(attribute.ref_name(), "weights");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TENSORS);
  EXPECT_EQ(attribute.ref_tensors().size(), 2);
  EXPECT_EQ(attribute.ref_tensors()[0].ref_name(), "tensor1");
  EXPECT_EQ(attribute.ref_tensors()[0].ref_float_data().size(), 6);
  EXPECT_EQ(attribute.ref_tensors()[1].ref_name(), "tensor2");
  EXPECT_EQ(attribute.ref_tensors()[1].ref_int32_data().size(), 2);
}

TEST(onnx_proto, AttributeProto_GraphsAttribute) {
  AttributeProto attribute;

  attribute.set_name("branches");
  attribute.set_type(AttributeProto::AttributeType::GRAPHS);

  GraphProto *graph1 = attribute.add_graphs();
  graph1->set_name("if_branch");
  NodeProto *node1 = graph1->add_node();
  node1->set_name("add_node");
  node1->set_op_type("Add");

  GraphProto *graph2 = attribute.add_graphs();
  graph2->set_name("else_branch");
  NodeProto *node2 = graph2->add_node();
  node2->set_name("mul_node");
  node2->set_op_type("Mul");

  EXPECT_EQ(attribute.ref_name(), "branches");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::GRAPHS);
  EXPECT_EQ(attribute.ref_graphs().size(), 2);
  EXPECT_EQ(attribute.ref_graphs()[0].ref_name(), "if_branch");
  EXPECT_EQ(attribute.ref_graphs()[0].ref_node()[0].ref_op_type(), "Add");
  EXPECT_EQ(attribute.ref_graphs()[1].ref_name(), "else_branch");
  EXPECT_EQ(attribute.ref_graphs()[1].ref_node()[0].ref_op_type(), "Mul");
}

TEST(onnx_proto, AttributeProto_DocString) {
  AttributeProto attribute;
  attribute.set_name("dropout_ratio");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.5f);
  attribute.set_doc_string("Controls the rate at which activations are dropped");

  EXPECT_EQ(attribute.ref_name(), "dropout_ratio");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_EQ(attribute.ref_f(), 0.5f);
  EXPECT_EQ(attribute.ref_doc_string(), "Controls the rate at which activations are dropped");
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_INT) {
  {
    // Test INT attribute
    AttributeProto int_attr;
    int_attr.set_name("int_attr");
    int_attr.set_type(AttributeProto::AttributeType::INT);
    int_attr.set_i(42);

    std::string serialized;
    int_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "int_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::INT);
    EXPECT_EQ(deserialized.ref_i(), 42);
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_FLOAT) {
  {
    // Test FLOAT attribute
    AttributeProto float_attr;
    float_attr.set_name("float_attr");
    float_attr.set_type(AttributeProto::AttributeType::FLOAT);
    float_attr.set_f(3.14f);

    std::string serialized;
    float_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "float_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::FLOAT);
    EXPECT_FLOAT_EQ(deserialized.ref_f(), 3.14f);
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_STRING) {
  {
    // Test STRING attribute
    AttributeProto string_attr;
    string_attr.set_name("string_attr");
    string_attr.set_type(AttributeProto::AttributeType::STRING);
    string_attr.set_s("test_string");

    std::string serialized;
    string_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "string_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::STRING);
    EXPECT_EQ(deserialized.ref_s(), "test_string");
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_INTS) {
  {
    // Test INTS attribute
    AttributeProto ints_attr;
    ints_attr.set_name("ints_attr");
    ints_attr.set_type(AttributeProto::AttributeType::INTS);
    ints_attr.ref_ints().push_back(1);
    ints_attr.ref_ints().push_back(2);
    ints_attr.ref_ints().push_back(3);

    std::string serialized;
    ints_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "ints_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::INTS);
    EXPECT_EQ(deserialized.ref_ints().size(), 3);
    EXPECT_EQ(deserialized.ref_ints()[0], 1);
    EXPECT_EQ(deserialized.ref_ints()[1], 2);
    EXPECT_EQ(deserialized.ref_ints()[2], 3);
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_FLOATS) {
  {
    // Test FLOATS attribute
    AttributeProto floats_attr;
    floats_attr.set_name("floats_attr");
    floats_attr.set_type(AttributeProto::AttributeType::FLOATS);
    floats_attr.ref_floats().push_back(1.1f);
    floats_attr.ref_floats().push_back(2.2f);

    std::string serialized;
    floats_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "floats_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::FLOATS);
    EXPECT_EQ(deserialized.ref_floats().size(), 2);
    EXPECT_FLOAT_EQ(deserialized.ref_floats()[0], 1.1f);
    EXPECT_FLOAT_EQ(deserialized.ref_floats()[1], 2.2f);
  }
}

// Regression test for the unpacked repeated float/double wire-type bug.
// AttributeProto.floats (field 7) and .doubles (field 11) must be written using
// the standard protobuf wire format so that third-party parsers (google
// protobuf, onnxruntime, ...) can read the resulting bytes. That means the
// per-element tag must use wire type 5 (FIELD_FIXED32) for float and wire
// type 1 (FIELD_FIXED64) for double — not wire type 2 (FIELD_FIXED_SIZE).
// Reading must still accept FIELD_FIXED_SIZE for buffers produced by older
// (buggy) writers, so a hand-crafted "legacy" buffer must round-trip too.
TEST(onnx_proto, AttributeProto_FloatsWireFormatIsFixed32) {
  AttributeProto attr;
  attr.set_name("floats");
  attr.set_type(AttributeProto::AttributeType::FLOATS);
  attr.ref_floats().push_back(1.0f);
  attr.ref_floats().push_back(-2.5f);

  std::string serialized;
  attr.SerializeToString(serialized);

  // Every occurrence of field 7 must use wire type 5 (fixed32), encoded as
  // (7 << 3) | 5 = 0x3D, followed by 4 raw little-endian bytes per element.
  const uint8_t expected_tag = static_cast<uint8_t>((7 << 3) | 5);
  const uint8_t bad_tag = static_cast<uint8_t>((7 << 3) | 2);
  int tag_hits = 0;
  for (size_t i = 0; i < serialized.size(); ++i) {
    const uint8_t b = static_cast<uint8_t>(serialized[i]);
    EXPECT_NE(b, bad_tag) << "field 7 must not be emitted with FIELD_FIXED_SIZE (wire type 2)";
    if (b == expected_tag) {
      ++tag_hits;
    }
  }
  EXPECT_EQ(tag_hits, 2) << "expected one FIELD_FIXED32 tag per float element";

  AttributeProto deserialized;
  deserialized.ParseFromString(serialized);
  EXPECT_EQ(deserialized.ref_floats().size(), 2);
  EXPECT_FLOAT_EQ(deserialized.ref_floats()[0], 1.0f);
  EXPECT_FLOAT_EQ(deserialized.ref_floats()[1], -2.5f);
}

TEST(onnx_proto, AttributeProto_FloatsLegacyWireTypeStillParses) {
  // Hand-crafted AttributeProto bytes that use the legacy (incorrect)
  // FIELD_FIXED_SIZE wire type 2 for the per-element tag of field 7
  // (floats). Older versions of onnx-light produced such buffers; the
  // reader must keep accepting them for backward compatibility.
  std::string buf;
  // name = "f" (field 1, wire type 2 = length-delimited).
  buf.push_back(static_cast<char>((1 << 3) | 2));
  buf.push_back(1);
  buf.push_back('f');
  // type = FLOATS = 6 (field 20, wire type 0 = varint).
  // Tag varint for field 20: (20 << 3) | 0 = 160 = 0xA0, needs 2-byte varint.
  buf.push_back(static_cast<char>(0xA0));
  buf.push_back(static_cast<char>(0x01));
  buf.push_back(6);
  // First floats element: legacy tag (7 << 3) | 2, followed directly by the
  // 4 raw little-endian bytes of the float value 1.0f (no length prefix —
  // the old buggy writer emitted FIELD_FIXED_SIZE without one).
  const float v0 = 1.0f;
  buf.push_back(static_cast<char>((7 << 3) | 2));
  buf.append(reinterpret_cast<const char *>(&v0), sizeof(v0));
  // Second floats element: same legacy encoding for -2.5f.
  const float v1 = -2.5f;
  buf.push_back(static_cast<char>((7 << 3) | 2));
  buf.append(reinterpret_cast<const char *>(&v1), sizeof(v1));

  AttributeProto deserialized;
  deserialized.ParseFromString(buf);
  EXPECT_EQ(deserialized.ref_name(), "f");
  EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::FLOATS);
  ASSERT_EQ(deserialized.ref_floats().size(), 2);
  EXPECT_FLOAT_EQ(deserialized.ref_floats()[0], 1.0f);
  EXPECT_FLOAT_EQ(deserialized.ref_floats()[1], -2.5f);
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_TENSOR) {
  {
    // Test TENSOR attribute
    AttributeProto tensor_attr;
    tensor_attr.set_name("tensor_attr");
    tensor_attr.set_type(AttributeProto::AttributeType::TENSOR);
    tensor_attr.ref_t().set_data_type(TensorProto::DataType::FLOAT);
    tensor_attr.ref_t().add_dims(2);
    tensor_attr.ref_t().add_dims(3);
    tensor_attr.ref_t().ref_float_data().add() = 1.1f;
    tensor_attr.ref_t().ref_float_data().add() = 2.2f;

    std::string serialized;
    tensor_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "tensor_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::TENSOR);
    EXPECT_EQ(deserialized.ref_t().ref_data_type(), TensorProto::DataType::FLOAT);
    EXPECT_EQ(deserialized.ref_t().ref_dims().size(), 2);
    EXPECT_EQ(deserialized.ref_t().ref_dims()[0], 2);
    EXPECT_EQ(deserialized.ref_t().ref_dims()[1], 3);
    EXPECT_EQ(deserialized.ref_t().ref_float_data().size(), 2);
    EXPECT_FLOAT_EQ(deserialized.ref_t().ref_float_data()[0], 1.1f);
    EXPECT_FLOAT_EQ(deserialized.ref_t().ref_float_data()[1], 2.2f);
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_STRINGS) {
  {
    // Test STRINGS attribute
    AttributeProto strings_attr;
    strings_attr.set_name("strings_attr");
    strings_attr.set_type(AttributeProto::AttributeType::STRINGS);
    strings_attr.ref_strings().push_back(utils::String("test_string_1"));
    strings_attr.ref_strings().push_back(utils::String("test_string_2"));

    std::string serialized;
    strings_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "strings_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::STRINGS);
    EXPECT_EQ(deserialized.ref_strings().size(), 2);
    EXPECT_EQ(deserialized.ref_strings()[0], "test_string_1");
    EXPECT_EQ(deserialized.ref_strings()[1], "test_string_2");
  }
}

TEST(onnx_proto, AttributeProto_Serialization_AllTypes_GRAPH) {
  {
    // Test GRAPH attribute
    AttributeProto graph_attr;
    graph_attr.set_name("graph_attr");
    graph_attr.set_type(AttributeProto::AttributeType::GRAPH);
    graph_attr.ref_g().set_name("test_graph");

    std::string serialized;
    graph_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "graph_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::GRAPH);
    EXPECT_EQ(deserialized.ref_g().ref_name(), "test_graph");
  }
}

TEST(onnx_proto, AttributeProto_PrintToStringStream_AllTypes) {
  utils::PrintOptions options;

  {
    // Test INT attribute print
    AttributeProto int_attr;
    int_attr.set_name("int_attr");
    int_attr.set_type(AttributeProto::AttributeType::INT);
    int_attr.set_i(42);

    std::stringstream ss_result;
    int_attr.PrintToStringStream(ss_result, options);
    std::string serialized = ss_result.str();

    EXPECT_TRUE(serialized.find("int_attr: 42") != std::string::npos);
  }

  {
    // Test INTS attribute print
    AttributeProto ints_attr;
    ints_attr.set_name("ints_attr");
    ints_attr.set_type(AttributeProto::AttributeType::INTS);
    ints_attr.ref_ints().push_back(1);
    ints_attr.ref_ints().push_back(2);
    ints_attr.ref_ints().push_back(3);

    std::stringstream ss_result;
    ints_attr.PrintToStringStream(ss_result, options);
    std::string serialized = ss_result.str();

    EXPECT_TRUE(serialized.find("ints_attr: [1, 2, 3]") != std::string::npos);
  }

  {
    // Test FLOATS attribute print
    AttributeProto floats_attr;
    floats_attr.set_name("floats_attr");
    floats_attr.set_type(AttributeProto::AttributeType::FLOATS);
    floats_attr.ref_floats().push_back(1.1f);
    floats_attr.ref_floats().push_back(2.2f);

    std::stringstream ss_result;
    floats_attr.PrintToStringStream(ss_result, options);
    std::string serialized = ss_result.str();

    EXPECT_TRUE(serialized.find("floats_attr: [1.1, 2.2]") != std::string::npos);
  }
}

TEST(onnx_proto, PrintOptions_InlineThreshold) {
  // ``inline_threshold`` controls which repeated fields are written inline.
  // All output is flat (no newlines); the threshold only affects whether the
  // field appears as a bracketed list or is omitted from flat output.
  TensorProto tensor;
  tensor.set_name("t");
  for (int64_t i = 0; i < 6; ++i)
    tensor.ref_dims().push_back(i);

  {
    utils::PrintOptions options;
    options.inline_threshold = 6;
    std::stringstream ss_serialized;
    tensor.PrintToStringStream(ss_serialized, options);
    std::string serialized = ss_serialized.str();
    EXPECT_TRUE(serialized.find("dims: [0, 1, 2, 3, 4, 5]") != std::string::npos);
    EXPECT_TRUE(serialized.find('\n') == std::string::npos);
  }

  {
    utils::PrintOptions options;
    options.inline_threshold = 5;
    std::stringstream ss_serialized;
    tensor.PrintToStringStream(ss_serialized, options);
    std::string serialized = ss_serialized.str();
    EXPECT_TRUE(serialized.find("dims:") != std::string::npos);
    EXPECT_TRUE(serialized.find('\n') == std::string::npos);
  }
}

TEST(onnx_proto, PrintOptions_FlatOutput) {
  // PrintToStringStream always writes a flat single-line representation to a
  // stringstream; there are no newlines regardless of message size or nesting.
  NodeProto node;
  node.set_name("relu1");
  node.set_op_type("Relu");
  *node.add_input() = "X";
  *node.add_output() = "Y";

  utils::PrintOptions options;
  std::stringstream ss;
  node.PrintToStringStream(ss, options);
  std::string serialized = ss.str();
  EXPECT_TRUE(serialized.find('\n') == std::string::npos);
  EXPECT_TRUE(serialized.find("name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("relu1") != std::string::npos);
}

TEST(onnx_proto, PrintOptions_FlatModelOutput) {
  // PrintToStringStream writes flat output without newlines.
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_producer");
  model.set_doc_string("Model documentation");
  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");
  NodeProto *node = graph->add_node();
  node->set_name("relu1");
  node->set_op_type("Relu");
  *node->add_input() = "X";
  *node->add_output() = "Y";

  utils::PrintOptions options;
  std::stringstream ss;
  model.PrintToStringStream(ss, options);
  std::string serialized = ss.str();
  EXPECT_TRUE(serialized.find('\n') == std::string::npos);
  EXPECT_TRUE(serialized.find("ir_version: 7") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_producer") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_graph") != std::string::npos);
  EXPECT_TRUE(serialized.find("relu1") != std::string::npos);
  EXPECT_FALSE(serialized.empty());
}

TEST(onnx_proto, AttributeProto_EmptyCollectionAttributes) {
  // Test empty INTS
  AttributeProto ints_attr;
  ints_attr.set_name("empty_ints");
  ints_attr.set_type(AttributeProto::AttributeType::INTS);

  EXPECT_EQ(ints_attr.ref_name(), "empty_ints");
  EXPECT_EQ(ints_attr.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(ints_attr.ref_ints().size(), 0);

  // Test empty FLOATS
  AttributeProto floats_attr;
  floats_attr.set_name("empty_floats");
  floats_attr.set_type(AttributeProto::AttributeType::FLOATS);

  EXPECT_EQ(floats_attr.ref_name(), "empty_floats");
  EXPECT_EQ(floats_attr.ref_type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(floats_attr.ref_floats().size(), 0);

  // Test empty STRINGS
  AttributeProto strings_attr;
  strings_attr.set_name("empty_strings");
  strings_attr.set_type(AttributeProto::AttributeType::STRINGS);

  EXPECT_EQ(strings_attr.ref_name(), "empty_strings");
  EXPECT_EQ(strings_attr.ref_type(), AttributeProto::AttributeType::STRINGS);
  EXPECT_EQ(strings_attr.ref_strings().size(), 0);

  // Test empty TENSORS
  AttributeProto tensors_attr;
  tensors_attr.set_name("empty_tensors");
  tensors_attr.set_type(AttributeProto::AttributeType::TENSORS);

  EXPECT_EQ(tensors_attr.ref_name(), "empty_tensors");
  EXPECT_EQ(tensors_attr.ref_type(), AttributeProto::AttributeType::TENSORS);
  EXPECT_EQ(tensors_attr.ref_tensors().size(), 0);
}

TEST(onnx_proto, AttributeProto_RefVersusAccessors) {
  AttributeProto attr;
  attr.set_name("test_attr");

  // Test INT
  attr.set_type(AttributeProto::AttributeType::INT);
  attr.set_i(42);
  EXPECT_EQ(attr.ref_i(), 42);
  EXPECT_TRUE(attr.has_i());

  // Test FLOAT
  attr.set_type(AttributeProto::AttributeType::FLOAT);
  attr.set_f(3.14f);
  EXPECT_FLOAT_EQ(attr.ref_f(), 3.14f);
  EXPECT_TRUE(attr.has_f());

  // Test STRING
  attr.set_type(AttributeProto::AttributeType::STRING);
  attr.set_s("test_string");
  EXPECT_EQ(attr.ref_s(), "test_string");
  EXPECT_TRUE(attr.has_s());

  // Test TENSOR
  attr.set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tensor = attr.add_t();
  tensor->set_name("tensor_name");
  EXPECT_EQ(attr.ref_t().ref_name(), "tensor_name");
  EXPECT_TRUE(attr.has_t());

  // Test GRAPH
  attr.set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *graph = attr.add_g();
  graph->set_name("graph_name");
  EXPECT_EQ(attr.ref_g().ref_name(), "graph_name");
  EXPECT_TRUE(attr.has_g());
}

// check size of AttributeProto serialization with the function returning the size

TEST(onnx_proto, SerializeSize_AttributeProto) {
  AttributeProto attribute;
  attribute.set_name("test_attribute");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(42);
  attribute.set_doc_string("Test attribute documentation");

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options).size());
}

TEST(onnx_proto, SerializeSize_AttributeProto_EmptyStrings) {
  AttributeProto attribute;
  attribute.set_name("");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s("");
  attribute.set_doc_string("");

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options).size());
}

TEST(onnx_proto, SerializeSize_AttributeProto_NullStrings) {
  AttributeProto attribute;
  // Do not set name, s, or doc_string to simulate null strings
  attribute.set_type(AttributeProto::AttributeType::STRING);

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options).size());
}

TEST(onnx_proto, SerializeSize_String) {
  utils::String test_string("hello world", 11);

  std::string serialized;
  utils::StringWriteStream write_stream;
  write_stream.write_string(test_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(),
            test_string.size() + write_stream.size_variant_uint64(test_string.size()));
  EXPECT_EQ(read_string, test_string);
}

TEST(onnx_proto, SerializeSize_EmptyString) {
  utils::String empty_string("", 0);

  utils::StringWriteStream write_stream;
  write_stream.write_string(empty_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), write_stream.size_variant_uint64(0));
  EXPECT_EQ(read_string.size(), 0);
  EXPECT_TRUE(read_string.empty());
}

TEST(onnx_proto, SerializeSize_NullString) {
  utils::String null_string;

  utils::StringWriteStream write_stream;
  write_stream.write_string(null_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), write_stream.size_variant_uint64(0));
  EXPECT_EQ(read_string.size(), 0);
  EXPECT_TRUE(read_string.empty());
}

TEST(onnx_proto, SerializeSize_StringWithNulls) {
  std::vector<char> data = {'t', 'e', 's', 't', '\0', 'n', 'u', 'l', 'l'};
  utils::String string_with_nulls(data.data(), data.size());

  utils::StringWriteStream write_stream;
  write_stream.write_string(string_with_nulls);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(),
            string_with_nulls.size() + write_stream.size_variant_uint64(string_with_nulls.size()));
  EXPECT_EQ(read_string.size(), string_with_nulls.size());
}

TEST(onnx_proto, SerializeSize_AttributeProto_IntFloatTensors) {
  AttributeProto attribute;
  attribute.set_name("complex_attribute");
  attribute.set_type(AttributeProto::AttributeType::TENSORS);

  TensorProto *tensor1 = attribute.add_tensors();
  tensor1->set_name("tensor1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_dims().push_back(2);
  tensor1->ref_dims().push_back(3);
  tensor1->ref_float_data().push_back(1.0f);
  tensor1->ref_float_data().push_back(2.0f);

  TensorProto *tensor2 = attribute.add_tensors();
  tensor2->set_name("tensor2");
  tensor2->set_data_type(TensorProto::DataType::INT32);
  tensor2->ref_dims().push_back(2);
  tensor2->ref_int32_data().push_back(10);
  tensor2->ref_int32_data().push_back(20);

  TensorProto *tensor3 = attribute.add_tensors();
  tensor3->set_name("tensor3");
  tensor3->set_data_type(TensorProto::DataType::INT64);
  tensor3->ref_dims().push_back(1);
  tensor3->ref_int64_data().push_back(10);

  TensorProto *tensor4 = attribute.add_tensors();
  tensor4->set_name("tensor4");
  tensor4->set_data_type(TensorProto::DataType::INT32);
  tensor4->ref_dims().push_back(1);
  tensor4->ref_int32_data().push_back(10);

  SerializeOptions options;
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor2->SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor2->SerializeSize(stream, options).size());
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor3->SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor3->SerializeSize(stream, options).size());
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor4->SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor4->SerializeSize(stream, options).size());
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor1->SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor1->SerializeSize(stream, options).size());
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    attribute.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options).size());
  }
}

TEST(onnx_proto, SerializeSize_ConsistencyAcrossTypes) {
  // Test with NodeProto
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  *node.add_input() = "input";
  *node.add_output() = "output";

  std::string node_serialized;
  node.SerializeToString(node_serialized);
  utils::StringWriteStream node_stream;
  SerializeOptions options;
  SerializeSizeResult node_size = node.SerializeSize(node_stream, options);
  EXPECT_EQ(node_serialized.size(), node_size.size());
  EXPECT_EQ(node_serialized.size(), static_cast<size_t>(node_size.proto_size));
  EXPECT_EQ(0, node_size.small_data_size);
  EXPECT_EQ(0, node_size.big_data_size);

  // Test with GraphProto
  GraphProto graph;
  graph.set_name("test_graph");
  NodeProto *graph_node = graph.add_node();
  graph_node->set_name("node_in_graph");

  std::string graph_serialized;
  graph.SerializeToString(graph_serialized);
  utils::StringWriteStream graph_stream;
  SerializeSizeResult graph_size = graph.SerializeSize(graph_stream, options);
  EXPECT_EQ(graph_serialized.size(), graph_size.size());
  EXPECT_EQ(graph_serialized.size(), static_cast<size_t>(graph_size.proto_size));
  EXPECT_EQ(0, graph_size.small_data_size);
  EXPECT_EQ(0, graph_size.big_data_size);

  // Test with ModelProto
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_model");
  GraphProto *model_graph = model.add_graph();
  model_graph->set_name("graph_in_model");

  std::string model_serialized;
  model.SerializeToString(model_serialized);
  utils::StringWriteStream model_stream;
  SerializeSizeResult model_size = model.SerializeSize(model_stream, options);
  EXPECT_EQ(model_serialized.size(), model_size.size());
  EXPECT_EQ(model_serialized.size(), static_cast<size_t>(model_size.proto_size));
  EXPECT_EQ(0, model_size.small_data_size);
  EXPECT_EQ(0, model_size.big_data_size);
}

TEST(onnx_proto, SerializeSizeResult_OperatorPlus) {
  SerializeSizeResult left{3, 5, 7};
  SerializeSizeResult right{11, 13, 17};

  SerializeSizeResult total = left + right;

  EXPECT_EQ(14, total.small_data_size);
  EXPECT_EQ(18, total.big_data_size);
  EXPECT_EQ(24, total.proto_size);
  EXPECT_EQ(56, total.size());
}

TEST(onnx_proto, SerializeSizeResult_SplitsExternalTensorData) {
  TensorProto tensor;
  tensor.set_name("external_size_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  tensor.ref_dims().push_back(2);
  tensor.ref_raw_data() = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8};

  StringStringEntryProto *location = tensor.add_external_data();
  location->set_key("location");
  location->set_value("serialize_size_result_weights.bin");
  StringStringEntryProto *offset = tensor.add_external_data();
  offset->set_key("offset");
  offset->set_value("0");
  StringStringEntryProto *length = tensor.add_external_data();
  length->set_key("length");
  length->set_value("8");

  SerializeOptions options;
  options.raw_data_threshold = kSmallTensorDataThresholdBytes;

  const std::string proto_path = "serialize_size_result_tensor.onnx";
  const std::string weights_path = "serialize_size_result_weights.bin";
  SerializeSizeResult size;
  {
    utils::TwoFilesWriteStream stream(proto_path, weights_path);
    size = tensor.SerializeSize(stream, options);
    tensor.SerializeToStream(stream, options);

    EXPECT_EQ(static_cast<uintmax_t>(stream.size()), static_cast<uintmax_t>(size.proto_size));
    EXPECT_EQ(static_cast<uintmax_t>(stream.weights_size()),
              static_cast<uintmax_t>(size.small_data_size + size.big_data_size));
  }
  EXPECT_EQ(8, size.small_data_size);
  EXPECT_EQ(0, size.big_data_size);
  EXPECT_EQ(size.proto_size + size.small_data_size + size.big_data_size, size.size());

  std::remove(proto_path.c_str());
  std::remove(weights_path.c_str());
}

TEST(onnx_proto, SerializeSizeResult_SplitsBigExternalTensorData) {
  TensorProto tensor;
  tensor.set_name("external_big_size_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  tensor.ref_dims().push_back(16);
  tensor.ref_raw_data() = std::vector<uint8_t>(64, 1);

  StringStringEntryProto *location = tensor.add_external_data();
  location->set_key("location");
  location->set_value("serialize_big_size_result_weights.bin");
  StringStringEntryProto *offset = tensor.add_external_data();
  offset->set_key("offset");
  offset->set_value("0");
  StringStringEntryProto *length = tensor.add_external_data();
  length->set_key("length");
  length->set_value("64");

  SerializeOptions options;
  options.raw_data_threshold = kSmallTensorDataThresholdBytes;

  const std::string proto_path = "serialize_big_size_result_tensor.onnx";
  const std::string weights_path = "serialize_big_size_result_weights.bin";
  SerializeSizeResult size;
  {
    utils::TwoFilesWriteStream stream(proto_path, weights_path);
    size = tensor.SerializeSize(stream, options);
    tensor.SerializeToStream(stream, options);

    EXPECT_EQ(static_cast<uintmax_t>(stream.size()), static_cast<uintmax_t>(size.proto_size));
    EXPECT_EQ(static_cast<uintmax_t>(stream.weights_size()),
              static_cast<uintmax_t>(size.small_data_size + size.big_data_size));
  }
  EXPECT_EQ(0, size.small_data_size);
  EXPECT_EQ(64, size.big_data_size);
  EXPECT_EQ(size.proto_size + size.small_data_size + size.big_data_size, size.size());

  std::remove(proto_path.c_str());
  std::remove(weights_path.c_str());
}

TEST(onnx_file, LoadOnnxFile_OldProtobuf) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / "data" / "test_ai_onnx_ml_binarizer.onnx";

  ModelProto model;
  utils::FileStream stream(file_path.string());
  ONNX_LIGHT_NAMESPACE::ParseOptions opts;
  EXPECT_TRUE(model.ParseFromStream(stream, opts));

  utils::PrintOptions pr;
  std::stringstream ss_text;
  model.PrintToStringStream(ss_text, pr);
  std::string text = ss_text.str();
  EXPECT_NE(text.find("Binarizer"), std::string::npos);
}

TEST(onnx_file, LoadOnnxFile_Expanded) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / "data" / "test_softmax_example_expanded.onnx";

  ModelProto model;
  utils::FileStream stream(file_path.string());
  ONNX_LIGHT_NAMESPACE::ParseOptions opts;
  EXPECT_TRUE(model.ParseFromStream(stream, opts));

  utils::PrintOptions pr;
  std::stringstream ss_text;
  model.PrintToStringStream(ss_text, pr);
  std::string text = ss_text.str();
  EXPECT_NE(text.find("ReduceSum"), std::string::npos);
}

TEST(onnx_file, LoadOnnxFile_Constant) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / "data" / "test_softmax_example_expanded.Constant.onnx";

  NodeProto node;
  utils::FileStream stream(file_path.string());
  ONNX_LIGHT_NAMESPACE::ParseOptions opts;
  EXPECT_TRUE(node.ParseFromStream(stream, opts));

  utils::PrintOptions pr;
  std::stringstream ss_text;
  node.PrintToStringStream(ss_text, pr);
  std::string text = ss_text.str();
  EXPECT_NE(text.find("Constant"), std::string::npos);
}

TEST(onnx_file, LoadOnnxFile_ConstantAsString) {
  std::vector<uint8_t> data = {18,  3,   65,  65,  65,  26,  2,  78,  78, 34, 8,   67,  111,
                               110, 115, 116, 97,  110, 116, 42, 39,  10, 5,  118, 97,  108,
                               117, 101, 42,  16,  8,   1,   16, 6,   58, 10, 255, 255, 255,
                               255, 255, 255, 255, 255, 255, 1,  106, 3,  68, 79,  67,  160,
                               1,   4,   170, 1,   3,   82,  69, 70,  58, 1,  77};
  std::string data_str(data.begin(), data.end());
  EXPECT_EQ(data_str.size(), data.size());
  NodeProto node;
  node.ParseFromString(data_str);

  utils::PrintOptions pr;
  std::stringstream ss_text;
  node.PrintToStringStream(ss_text, pr);
  std::string text = ss_text.str();
  EXPECT_NE(text.find("Constant"), std::string::npos);
}

TEST(onnx_proto, TensorProto_uint64) {
  TensorProto tensor = TensorProto();
  tensor.set_name("tensor");
  tensor.set_data_type(TensorProto::DataType::UINT64);
  tensor.ref_dims().push_back(2);
  tensor.ref_uint64_data().push_back(4);
  tensor.ref_uint64_data().push_back(5);

  SerializeOptions options;
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto t2 = TensorProto();
  ParseOptions parse_options;
  t2.ParseFromString(serialized, parse_options);

  EXPECT_EQ(t2.ref_name().sv(), tensor.ref_name().sv());
  EXPECT_EQ(t2.ref_data_type(), tensor.ref_data_type());
  EXPECT_EQ(t2.ref_dims().size(), tensor.ref_dims().size());
  EXPECT_EQ(t2.ref_uint64_data().size(), tensor.ref_uint64_data().size());
  EXPECT_EQ(t2.ref_uint64_data()[0], 4);
  EXPECT_EQ(t2.ref_uint64_data()[1], 5);
  utils::StringWriteStream stream;
  EXPECT_EQ(serialized.size(), tensor.SerializeSize(stream, options).size());
}

TEST(onnx_proto, AttributeProto_float) {
  AttributeProto attribute = AttributeProto();
  attribute.set_name("attribute");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.01f);

  SerializeOptions options;
  std::string serialized;
  attribute.SerializeToString(serialized);

  AttributeProto t2 = AttributeProto();
  ParseOptions parse_options;
  t2.ParseFromString(serialized, parse_options);

  EXPECT_EQ(t2.ref_name().sv(), attribute.ref_name().sv());
  EXPECT_EQ(t2.ref_type(), attribute.ref_type());
  EXPECT_EQ(t2.ref_f(), attribute.ref_f());
  utils::StringWriteStream stream;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options).size());
}

//

TEST(onnx_proto, AttributeProto_TypeAttribute) {
  AttributeProto attribute;

  attribute.set_name("input_type");
  attribute.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  TypeProto *type = attribute.add_tp();
  type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  TensorShapeProto::Dimension *dim1 = shape->add_dim();
  dim1->set_dim_value(3);
  TensorShapeProto::Dimension *dim2 = shape->add_dim();
  dim2->set_dim_param("N");

  EXPECT_EQ(attribute.ref_name(), "input_type");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(attribute.has_tp());
  EXPECT_TRUE(attribute.ref_tp().has_tensor_type());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(attribute.ref_tp().ref_tensor_type().has_shape());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "N");
}

TEST(onnx_proto, AttributeProto_TypesAttribute) {
  AttributeProto attribute;

  attribute.set_name("output_types");
  attribute.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  // First type
  TypeProto *type1 = attribute.add_tp();
  type1->add_tensor_type()->set_elem_type(static_cast<int32_t>(1)); // FLOAT
  TensorShapeProto *shape1 = type1->ref_tensor_type().add_shape();
  shape1->add_dim()->set_dim_value(2);
  shape1->add_dim()->set_dim_value(3);

  EXPECT_EQ(attribute.ref_name(), "output_types");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(attribute.ref_tp().has_tensor_type());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 2);
}

TEST(onnx_proto, AttributeProto_Serialization_TypeProto) {
  AttributeProto type_attr;
  type_attr.set_name("type_attr");
  type_attr.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  TypeProto *type = type_attr.add_tp();
  type->add_tensor_type()->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = type->ref_tensor_type().add_shape();
  shape->add_dim()->set_dim_value(4);
  shape->add_dim()->set_dim_param("dynamic_dim");

  std::string serialized;
  type_attr.SerializeToString(serialized);

  AttributeProto deserialized;
  deserialized.ParseFromString(serialized);

  EXPECT_EQ(deserialized.ref_name(), "type_attr");
  EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(deserialized.has_tp());
  EXPECT_TRUE(deserialized.ref_tp().has_tensor_type());
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(deserialized.ref_tp().ref_tensor_type().has_shape());
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 4);
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(),
            "dynamic_dim");
}

//

TEST(onnx_proto, TensorProto_DataLocation) {
  // Create a TensorProto with external location
  TensorProto tensor;
  tensor.set_name("external_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);

  // By default, the data location is DEFAULT
  EXPECT_EQ(tensor.ref_data_location(), TensorProto::DataLocation::DEFAULT);

  // Set the location as external
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor.ref_data_location(), TensorProto::DataLocation::EXTERNAL);

  // Serialize and deserialize
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Verify that the data location is preserved
  EXPECT_EQ(tensor2.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor2.ref_name(), "external_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);

  // Test all possible values of DataLocation
  TensorProto tensor3;
  tensor3.set_data_location(TensorProto::DataLocation::DEFAULT);
  EXPECT_EQ(tensor3.ref_data_location(), TensorProto::DataLocation::DEFAULT);

  tensor3.set_data_location(TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor3.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
}

TEST(onnx_proto, TensorProto_ExternalData) {
  // Create a TensorProto with external data
  TensorProto tensor;
  tensor.set_name("external_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);

  // Add information about external data
  StringStringEntryProto *entry1 = tensor.add_external_data();
  entry1->set_key("location");
  entry1->set_value("weights.bin");

  StringStringEntryProto *entry2 = tensor.add_external_data();
  entry2->set_key("offset");
  entry2->set_value("0");

  StringStringEntryProto *entry3 = tensor.add_external_data();
  entry3->set_key("length");
  entry3->set_value("1024");

  // Verify entries
  EXPECT_EQ(tensor.ref_external_data().size(), 3);
  EXPECT_EQ(tensor.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(tensor.ref_external_data()[0].ref_value(), "weights.bin");
  EXPECT_EQ(tensor.ref_external_data()[1].ref_key(), "offset");
  EXPECT_EQ(tensor.ref_external_data()[1].ref_value(), "0");
  EXPECT_EQ(tensor.ref_external_data()[2].ref_key(), "length");
  EXPECT_EQ(tensor.ref_external_data()[2].ref_value(), "1024");

  // Serialize and deserialize
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Verify that external information is preserved
  EXPECT_EQ(tensor2.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor2.ref_external_data().size(), 3);
  EXPECT_EQ(tensor2.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(tensor2.ref_external_data()[0].ref_value(), "weights.bin");
}

TEST(onnx_proto, TensorProto_DataLocationPrintToStringStream) {
  utils::PrintOptions options;
  TensorProto tensor;
  tensor.set_name("external_print_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);

  StringStringEntryProto *entry = tensor.add_external_data();
  entry->set_key("location");
  entry->set_value("external_file.bin");

  // Generate the text representation
  std::stringstream ss_result;
  tensor.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  // Verify that the output contains the data location information
  bool foundDataLocation = false;
  bool foundExternalData = false;

  if (serialized.find("data_location:") != std::string::npos &&
      serialized.find(std::to_string(static_cast<int>(TensorProto::DataLocation::EXTERNAL))) !=
          std::string::npos) {
    foundDataLocation = true;
  }

  if (serialized.find("external_data") != std::string::npos &&
      serialized.find("location") != std::string::npos &&
      serialized.find("external_file.bin") != std::string::npos) {
    foundExternalData = true;
  }

  EXPECT_TRUE(foundDataLocation);
  EXPECT_TRUE(foundExternalData);
}

TEST(onnx_proto, TensorProto_CopyFromWithDataLocation) {
  TensorProto source;
  source.set_name("source_external_tensor");
  source.set_data_type(TensorProto::DataType::FLOAT);
  source.set_data_location(TensorProto::DataLocation::EXTERNAL);

  StringStringEntryProto *entry = source.add_external_data();
  entry->set_key("location");
  entry->set_value("source_file.bin");

  // Copy the data to a new instance
  TensorProto target;
  target.CopyFrom(source);

  // Verify that all properties related to data location are copied
  EXPECT_EQ(target.ref_name(), "source_external_tensor");
  EXPECT_EQ(target.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(target.ref_external_data().size(), 1);
  EXPECT_EQ(target.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(target.ref_external_data()[0].ref_value(), "source_file.bin");
}

// Helper: build a minimal EXTERNAL TensorProto with the given location string.
static TensorProto MakeExternalTensor(const std::string &location) {
  TensorProto t;
  t.set_name("test_tensor");
  t.set_data_type(TensorProto::DataType::FLOAT);
  t.set_data_location(TensorProto::DataLocation::EXTERNAL);
  StringStringEntryProto *e = t.add_external_data();
  e->set_key("location");
  e->set_value(location);
  return t;
}

TEST(onnx_proto, LoadExternalData_RejectsPathTraversal) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_load_traversal_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Create a real weights file inside the directory.
  {
    std::ofstream ofs(dir / "weights.bin", std::ios::binary);
    ofs << "data";
  }

  TensorProto t1 = MakeExternalTensor("../escape.bin");
  EXPECT_THROW(t1.LoadExternalData(dir.string()), std::exception);

  TensorProto t2 = MakeExternalTensor("/etc/passwd");
  EXPECT_THROW(t2.LoadExternalData(dir.string()), std::exception);

  // Lexically normalizes to ../outside, must be rejected.
  TensorProto t3 = MakeExternalTensor("subdir/../../outside.bin");
  EXPECT_THROW(t3.LoadExternalData(dir.string()), std::exception);

  fs::remove_all(dir);
#endif
}

#ifndef _WIN32
TEST(onnx_proto, LoadExternalData_RejectsSymlink) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_load_symlink_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Create a real target file.
  fs::path target = dir / "real.bin";
  {
    std::ofstream ofs(target, std::ios::binary);
    ofs.write("test", 4);
  }

  // Create a symlink pointing to the real file inside the directory.
  fs::path link = dir / "link.bin";
  fs::create_symlink(target, link);

  TensorProto t = MakeExternalTensor("link.bin");
  EXPECT_THROW(t.LoadExternalData(dir.string()), std::exception);

  fs::remove_all(dir);
#endif
}

TEST(onnx_proto, LoadExternalData_RejectsSymlinkOutside) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_load_symlink_out_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  fs::path outside = fs::temp_directory_path() / "onnx_light_load_symlink_out_secret";
  fs::remove_all(outside);
  fs::create_directories(outside);
  {
    std::ofstream ofs(outside / "secret.bin", std::ios::binary);
    ofs.write("secret", 6);
  }

  // Symlink inside dir pointing to a file outside dir.
  fs::path link = dir / "evil.bin";
  fs::create_symlink(outside / "secret.bin", link);

  TensorProto t = MakeExternalTensor("evil.bin");
  EXPECT_THROW(t.LoadExternalData(dir.string()), std::exception);

  fs::remove_all(dir);
  fs::remove_all(outside);
#endif
}

TEST(onnx_proto, LoadExternalData_RejectsParentDirSymlink) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_load_parentsym_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  fs::path outside = fs::temp_directory_path() / "onnx_light_load_parentsym_outside";
  fs::remove_all(outside);
  fs::create_directories(outside);
  {
    std::ofstream ofs(outside / "secret.bin", std::ios::binary);
    ofs.write("secret", 6);
  }

  // Create a directory symlink inside dir pointing outside.
  fs::path symlink_subdir = dir / "subdir";
  fs::create_directory_symlink(outside, symlink_subdir);

  // "subdir/secret.bin" resolves outside dir via the symlink.
  TensorProto t = MakeExternalTensor("subdir/secret.bin");
  EXPECT_THROW(t.LoadExternalData(dir.string()), std::exception);

  fs::remove_all(dir);
  fs::remove_all(outside);
#endif
}
#endif // !_WIN32

TEST(onnx_proto, LoadExternalData_RejectsHardlink) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_load_hardlink_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Create original file outside dir to simulate a sensitive file.
  fs::path original = fs::temp_directory_path() / "onnx_light_hardlink_original.bin";
  {
    std::ofstream ofs(original, std::ios::binary);
    ofs.write("sensitive", 9);
  }

  // Create a hard link inside dir pointing to the original file.
  fs::path hardlink = dir / "weights.bin";
  std::error_code ec;
  fs::create_hard_link(original, hardlink, ec);
  if (ec) {
    // Hard links across filesystems are not supported; skip this test.
    fs::remove_all(dir);
    fs::remove(original);
    GTEST_SKIP() << "Hard links not supported across filesystems on this platform.";
  }

  TensorProto t = MakeExternalTensor("weights.bin");
  EXPECT_THROW(t.LoadExternalData(dir.string()), std::exception);

  fs::remove_all(dir);
  fs::remove(original);
#endif
}

#ifndef _WIN32
// GHSA-8qff-7g33-75mx: FileWriteStream must refuse to open a symlink as target.
TEST(onnx_stream, FileWriteStream_RejectsSymlink) {
#ifndef ONNX_NO_EXCEPTIONS
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "onnx_light_write_symlink_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  fs::path real_file = dir / "real.bin";
  {
    std::ofstream ofs(real_file, std::ios::binary);
    ofs.write("real", 4);
  }

  // Place a symlink at the intended write target.
  fs::path link = dir / "link.bin";
  fs::create_symlink(real_file, link);

  EXPECT_THROW(utils::FileWriteStream stream(link.string()), std::exception);

  fs::remove_all(dir);
#endif
}
#endif // !_WIN32

TEST(onnx_proto, SequenceProto_Basic) {
  SequenceProto sequence;

  EXPECT_EQ(sequence.ref_elem_type(), 0);
  EXPECT_EQ(sequence.ref_tensor_values().size(), 0);
  EXPECT_EQ(sequence.ref_sparse_tensor_values().size(), 0);
  EXPECT_EQ(sequence.ref_sequence_values().size(), 0);
  EXPECT_EQ(sequence.ref_map_values().size(), 0);

  sequence.set_elem_type(1); // FLOAT
  EXPECT_EQ(sequence.ref_elem_type(), 1);

  // Add a tensor to the sequence
  TensorProto *tensor = sequence.add_tensor_values();
  tensor->set_name("tensor_in_sequence");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_dims().push_back(3);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);
  tensor->ref_float_data().push_back(3.0f);

  EXPECT_EQ(sequence.ref_tensor_values().size(), 1);
  EXPECT_EQ(sequence.ref_tensor_values()[0].ref_name(), "tensor_in_sequence");
  EXPECT_EQ(sequence.ref_tensor_values()[0].ref_float_data().size(), 3);
}

TEST(onnx_proto, SequenceProto_SparseTensorValues) {
  SequenceProto sequence;
  sequence.set_elem_type(1); // FLOAT

  // Add a sparse tensor to the sequence
  SparseTensorProto *sparse_tensor = sequence.add_sparse_tensor_values();
  sparse_tensor->ref_dims().push_back(5);
  sparse_tensor->ref_dims().push_back(5);

  sparse_tensor->ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse_tensor->ref_values().ref_float_data().push_back(1.5f);
  sparse_tensor->ref_values().ref_float_data().push_back(2.5f);

  sparse_tensor->ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse_tensor->ref_indices().ref_int64_data().push_back(0);
  sparse_tensor->ref_indices().ref_int64_data().push_back(2);
  sparse_tensor->ref_indices().ref_int64_data().push_back(3);
  sparse_tensor->ref_indices().ref_int64_data().push_back(4);

  EXPECT_EQ(sequence.ref_sparse_tensor_values().size(), 1);
  EXPECT_EQ(sequence.ref_sparse_tensor_values()[0].ref_dims().size(), 2);
  EXPECT_EQ(sequence.ref_sparse_tensor_values()[0].ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sequence.ref_sparse_tensor_values()[0].ref_indices().ref_int64_data().size(), 4);
}

TEST(onnx_proto, SequenceProto_NestedSequences) {
  SequenceProto outer_sequence;
  outer_sequence.set_elem_type(10); // SEQUENCE_TYPE

  // Add a nested sequence
  SequenceProto *inner_sequence = outer_sequence.add_sequence_values();
  inner_sequence->set_elem_type(1); // FLOAT

  // Add a tensor to the nested sequence
  TensorProto *tensor = inner_sequence->add_tensor_values();
  tensor->set_name("inner_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);

  EXPECT_EQ(outer_sequence.ref_sequence_values().size(), 1);
  EXPECT_EQ(outer_sequence.ref_sequence_values()[0].ref_elem_type(), 1);
  EXPECT_EQ(outer_sequence.ref_sequence_values()[0].ref_tensor_values().size(), 1);
  EXPECT_EQ(outer_sequence.ref_sequence_values()[0].ref_tensor_values()[0].ref_name(),
            "inner_tensor");
  EXPECT_EQ(outer_sequence.ref_sequence_values()[0].ref_tensor_values()[0].ref_float_data().size(),
            2);
}

TEST(onnx_proto, SequenceProto_Serialization) {
  SequenceProto sequence1;
  sequence1.set_elem_type(1); // FLOAT

  // Add some tensors
  TensorProto *tensor1 = sequence1.add_tensor_values();
  tensor1->set_name("tensor1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_float_data().push_back(1.0f);

  TensorProto *tensor2 = sequence1.add_tensor_values();
  tensor2->set_name("tensor2");
  tensor2->set_data_type(TensorProto::DataType::FLOAT);
  tensor2->ref_float_data().push_back(2.0f);

  // Serialize
  std::string serialized;
  sequence1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), sequence1.SerializeSize().size());

  // Deserialize
  SequenceProto sequence2;
  sequence2.ParseFromString(serialized);

  // Verify
  EXPECT_EQ(sequence2.ref_elem_type(), 1);
  EXPECT_EQ(sequence2.ref_tensor_values().size(), 2);
  EXPECT_EQ(sequence2.ref_tensor_values()[0].ref_name(), "tensor1");
  EXPECT_EQ(sequence2.ref_tensor_values()[1].ref_name(), "tensor2");
  EXPECT_EQ(sequence2.ref_tensor_values()[0].ref_float_data()[0], 1.0f);
  EXPECT_EQ(sequence2.ref_tensor_values()[1].ref_float_data()[0], 2.0f);
}

TEST(onnx_proto, SequenceProto_CopyFrom) {
  SequenceProto source;
  source.set_elem_type(1); // FLOAT

  TensorProto *tensor = source.add_tensor_values();
  tensor->set_name("source_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(3.14f);

  SequenceProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_elem_type(), 1);
  EXPECT_EQ(target.ref_tensor_values().size(), 1);
  EXPECT_EQ(target.ref_tensor_values()[0].ref_name(), "source_tensor");
  EXPECT_EQ(target.ref_tensor_values()[0].ref_float_data().size(), 1);
  EXPECT_FLOAT_EQ(target.ref_tensor_values()[0].ref_float_data()[0], 3.14f);
}

TEST(onnx_string, SequenceProto) {
  utils::PrintOptions options;

  SequenceProto sequence;
  sequence.set_elem_type(1); // FLOAT

  TensorProto *tensor = sequence.add_tensor_values();
  tensor->set_name("print_test_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_float_data().push_back(1.5f);
  tensor->ref_float_data().push_back(2.5f);

  std::stringstream ss_result;
  sequence.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("elem_type:") != std::string::npos);
  EXPECT_TRUE(serialized.find("1") != std::string::npos); // FLOAT type
  EXPECT_TRUE(serialized.find("tensor_values") != std::string::npos);
  EXPECT_TRUE(serialized.find("print_test_tensor") != std::string::npos);
}

TEST(onnx_proto, SequenceProto_EmptySequence) {
  SequenceProto sequence;

  EXPECT_EQ(sequence.ref_elem_type(), 0);
  EXPECT_EQ(sequence.ref_tensor_values().size(), 0);
  EXPECT_EQ(sequence.ref_sparse_tensor_values().size(), 0);
  EXPECT_EQ(sequence.ref_sequence_values().size(), 0);
  EXPECT_EQ(sequence.ref_map_values().size(), 0);

  // Serialize an empty sequence
  std::string serialized;
  sequence.SerializeToString(serialized);

  // The size should not be zero even for an empty sequence
  // because the metadata is still serialized
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), sequence.SerializeSize().size());

  // Deserialize
  SequenceProto sequence2;
  sequence2.ParseFromString(serialized);

  EXPECT_EQ(sequence2.ref_elem_type(), 0);
  EXPECT_EQ(sequence2.ref_tensor_values().size(), 0);
}

TEST(onnx_proto, SequenceProto_SerializeSize) {
  SequenceProto sequence;
  sequence.set_elem_type(1); // FLOAT

  TensorProto *tensor = sequence.add_tensor_values();
  tensor->set_name("size_test_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(1.0f);

  std::string serialized;
  sequence.SerializeToString(serialized);

  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), sequence.SerializeSize(stream, options).size());
}

TEST(onnx_proto, MapProto_Basic) {
  MapProto map;
  EXPECT_EQ(map.ref_key_type(), 0);
  map.set_key_type(6); // INT32
  EXPECT_EQ(map.ref_key_type(), 6);
  *map.add_keys() = 1;

  // Test with key-value pairs
  SequenceProto &sequence = map.ref_values();
  sequence.set_elem_type(1); // FLOAT

  TensorProto *tensor = sequence.add_tensor_values();
  tensor->set_name("size_test_tensor");
  tensor->set_data_type(6);
  tensor->ref_int32_data().push_back(1);

  EXPECT_EQ(map.ref_keys().size(), 1);
  EXPECT_EQ(map.ref_values().ref_tensor_values()[0].ref_int32_data().size(), 1);
}

TEST(onnx_proto, MapProto_Serialization) {
  // Create a MapProto with key-value pairs
  MapProto map1;
  map1.set_key_type(6); // INT32

  // Add keys
  *map1.add_keys() = 1;
  *map1.add_keys() = 2;
  *map1.add_keys() = 3;

  // Add values (tensors)
  SequenceProto &sequence = map1.ref_values();
  sequence.set_elem_type(1); // FLOAT

  // First tensor
  TensorProto *tensor1 = sequence.add_tensor_values();
  tensor1->set_name("tensor1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_float_data().push_back(10.5f);

  // Second tensor
  TensorProto *tensor2 = sequence.add_tensor_values();
  tensor2->set_name("tensor2");
  tensor2->set_data_type(TensorProto::DataType::FLOAT);
  tensor2->ref_float_data().push_back(20.5f);

  // Third tensor
  TensorProto *tensor3 = sequence.add_tensor_values();
  tensor3->set_name("tensor3");
  tensor3->set_data_type(TensorProto::DataType::FLOAT);
  tensor3->ref_float_data().push_back(30.5f);

  // Verify state before serialization
  EXPECT_EQ(map1.ref_key_type(), 6);
  EXPECT_EQ(map1.ref_keys().size(), 3);
  EXPECT_EQ(map1.ref_keys()[0], 1);
  EXPECT_EQ(map1.ref_keys()[1], 2);
  EXPECT_EQ(map1.ref_keys()[2], 3);
  EXPECT_EQ(map1.ref_values().ref_tensor_values().size(), 3);

  // Serialize
  std::string serialized;
  map1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());

  // Verify que SerializeSize fonctionne correctement
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), map1.SerializeSize(stream, options).size());

  // Deserialize
  MapProto map2;
  map2.ParseFromString(serialized);

  // Verify state after deserialization
  EXPECT_EQ(map2.ref_key_type(), 6);
  EXPECT_EQ(map2.ref_keys().size(), 3);
  EXPECT_EQ(map2.ref_keys()[0], 1);
  EXPECT_EQ(map2.ref_keys()[1], 2);
  EXPECT_EQ(map2.ref_keys()[2], 3);
  EXPECT_EQ(map2.ref_values().ref_tensor_values().size(), 3);
  EXPECT_EQ(map2.ref_values().ref_tensor_values()[0].ref_name(), "tensor1");
  EXPECT_EQ(map2.ref_values().ref_tensor_values()[1].ref_name(), "tensor2");
  EXPECT_EQ(map2.ref_values().ref_tensor_values()[2].ref_name(), "tensor3");
  EXPECT_FLOAT_EQ(map2.ref_values().ref_tensor_values()[0].ref_float_data()[0], 10.5f);
  EXPECT_FLOAT_EQ(map2.ref_values().ref_tensor_values()[1].ref_float_data()[0], 20.5f);
  EXPECT_FLOAT_EQ(map2.ref_values().ref_tensor_values()[2].ref_float_data()[0], 30.5f);
}

TEST(onnx_string, MapProto) {
  utils::PrintOptions options;

  // Create a MapProto to test text printing
  MapProto map;
  map.set_key_type(6); // INT32

  // Add some keys and values
  *map.add_keys() = 42;
  *map.add_keys() = 43;

  SequenceProto &sequence = map.ref_values();
  sequence.set_elem_type(1); // FLOAT

  TensorProto *tensor1 = sequence.add_tensor_values();
  tensor1->set_name("map_value1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_float_data().push_back(3.14f);

  TensorProto *tensor2 = sequence.add_tensor_values();
  tensor2->set_name("map_value2");
  tensor2->set_data_type(TensorProto::DataType::FLOAT);
  tensor2->ref_float_data().push_back(2.71f);

  // Generate the text representation
  std::stringstream ss_result;
  map.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  // Verify that the output contains the main information

  EXPECT_TRUE(serialized.find("key_type:") != std::string::npos);
  EXPECT_TRUE(serialized.find("6") != std::string::npos); // INT32 type
  EXPECT_TRUE(serialized.find("keys:") != std::string::npos);
  EXPECT_TRUE(serialized.find("42") != std::string::npos);
  EXPECT_TRUE(serialized.find("43") != std::string::npos);
  EXPECT_TRUE(serialized.find("values") != std::string::npos);
  EXPECT_TRUE(serialized.find("map_value1") != std::string::npos);
  EXPECT_TRUE(serialized.find("map_value2") != std::string::npos);
}

TEST(onnx_proto, MapProto_CopyFrom) {
  // Create a source MapProto
  MapProto source;
  source.set_key_type(7); // INT64

  *source.add_keys() = 100;
  *source.add_keys() = 200;

  SequenceProto &seq = source.ref_values();
  seq.set_elem_type(1); // FLOAT

  TensorProto *tensor = seq.add_tensor_values();
  tensor->set_name("source_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(9.8f);

  // Copy to a target
  MapProto target;
  target.CopyFrom(source);

  // Verify that the copy is correct
  EXPECT_EQ(target.ref_key_type(), 7);
  EXPECT_EQ(target.ref_keys().size(), 2);
  EXPECT_EQ(target.ref_keys()[0], 100);
  EXPECT_EQ(target.ref_keys()[1], 200);
  EXPECT_EQ(target.ref_values().ref_elem_type(), 1);
  EXPECT_EQ(target.ref_values().ref_tensor_values().size(), 1);
  EXPECT_EQ(target.ref_values().ref_tensor_values()[0].ref_name(), "source_tensor");
  EXPECT_FLOAT_EQ(target.ref_values().ref_tensor_values()[0].ref_float_data()[0], 9.8f);
}

TEST(onnx_proto, OptionalProto_Basic) {
  OptionalProto optional;

  EXPECT_EQ(optional.ref_elem_type(), 0);
  EXPECT_FALSE(optional.has_tensor_value());
  EXPECT_FALSE(optional.has_sparse_tensor_value());
  EXPECT_FALSE(optional.has_sequence_value());
  EXPECT_FALSE(optional.has_map_value());
  EXPECT_FALSE(optional.has_optional_value());

  optional.set_elem_type(1); // FLOAT
  EXPECT_EQ(optional.ref_elem_type(), 1);

  // Add a tensor
  TensorProto *tensor = optional.add_tensor_value();
  tensor->set_name("optional_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_dims().push_back(3);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);

  EXPECT_TRUE(optional.has_tensor_value());
  EXPECT_EQ(optional.ref_tensor_value().ref_name(), "optional_tensor");
  EXPECT_EQ(optional.ref_tensor_value().ref_dims().size(), 2);
  EXPECT_EQ(optional.ref_tensor_value().ref_float_data().size(), 2);
}

TEST(onnx_proto, OptionalProto_SparseTensorValue) {
  OptionalProto optional;
  optional.set_elem_type(1); // FLOAT

  // Add a sparse tensor
  SparseTensorProto *sparse_tensor = optional.add_sparse_tensor_value();
  sparse_tensor->ref_dims().push_back(5);
  sparse_tensor->ref_dims().push_back(5);

  sparse_tensor->ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse_tensor->ref_values().ref_float_data().push_back(1.5f);
  sparse_tensor->ref_values().ref_float_data().push_back(2.5f);

  sparse_tensor->ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse_tensor->ref_indices().ref_int64_data().push_back(0);
  sparse_tensor->ref_indices().ref_int64_data().push_back(2);

  EXPECT_TRUE(optional.has_sparse_tensor_value());
  EXPECT_EQ(optional.ref_sparse_tensor_value().ref_dims().size(), 2);
  EXPECT_EQ(optional.ref_sparse_tensor_value().ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(optional.ref_sparse_tensor_value().ref_indices().ref_int64_data().size(), 2);
}

TEST(onnx_proto, OptionalProto_SequenceValue) {
  OptionalProto optional;
  optional.set_elem_type(10); // SEQUENCE_TYPE

  // Add a sequence
  SequenceProto *sequence = optional.add_sequence_value();
  sequence->set_elem_type(1); // FLOAT

  // Add a tensor to the sequence
  TensorProto *tensor = sequence->add_tensor_values();
  tensor->set_name("tensor_in_sequence");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);

  EXPECT_TRUE(optional.has_sequence_value());
  EXPECT_EQ(optional.ref_sequence_value().ref_elem_type(), 1);
  EXPECT_EQ(optional.ref_sequence_value().ref_tensor_values().size(), 1);
  EXPECT_EQ(optional.ref_sequence_value().ref_tensor_values()[0].ref_name(), "tensor_in_sequence");
  EXPECT_EQ(optional.ref_sequence_value().ref_tensor_values()[0].ref_float_data().size(), 2);
}

TEST(onnx_proto, OptionalProto_MapValue) {
  OptionalProto optional;
  optional.set_elem_type(11); // MAP_TYPE

  // Add a map
  MapProto *map = optional.add_map_value();
  map->set_key_type(6); // INT32

  // Add keys
  *map->add_keys() = 1;
  *map->add_keys() = 2;

  // Add values
  SequenceProto &sequence = map->ref_values();
  sequence.set_elem_type(1); // FLOAT

  TensorProto *tensor1 = sequence.add_tensor_values();
  tensor1->set_name("map_value1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_float_data().push_back(3.14f);

  TensorProto *tensor2 = sequence.add_tensor_values();
  tensor2->set_name("map_value2");
  tensor2->set_data_type(TensorProto::DataType::FLOAT);
  tensor2->ref_float_data().push_back(2.71f);

  EXPECT_TRUE(optional.has_map_value());
  EXPECT_EQ(optional.ref_map_value().ref_key_type(), 6);
  EXPECT_EQ(optional.ref_map_value().ref_keys().size(), 2);
  EXPECT_EQ(optional.ref_map_value().ref_values().ref_tensor_values().size(), 2);
}

TEST(onnx_proto, OptionalProto_NestedOptionalValue) {
  OptionalProto outer_optional;
  outer_optional.set_elem_type(12); // OPTIONAL_TYPE

  // Add a nested optional
  OptionalProto *inner_optional = outer_optional.add_optional_value();
  inner_optional->set_elem_type(1); // FLOAT

  // Add a tensor to the nested optional
  TensorProto *tensor = inner_optional->add_tensor_value();
  tensor->set_name("inner_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(3.14f);

  EXPECT_TRUE(outer_optional.has_optional_value());
  EXPECT_EQ(outer_optional.ref_optional_value().ref_elem_type(), 1);
  EXPECT_TRUE(outer_optional.ref_optional_value().has_tensor_value());
  EXPECT_EQ(outer_optional.ref_optional_value().ref_tensor_value().ref_name(), "inner_tensor");
  EXPECT_EQ(outer_optional.ref_optional_value().ref_tensor_value().ref_float_data().size(), 1);
  EXPECT_FLOAT_EQ(outer_optional.ref_optional_value().ref_tensor_value().ref_float_data()[0],
                  3.14f);
}

TEST(onnx_proto, OptionalProto_Serialization) {
  OptionalProto optional1;
  optional1.set_elem_type(1); // FLOAT

  // Add a tensor
  TensorProto *tensor = optional1.add_tensor_value();
  tensor->set_name("serialized_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_dims().push_back(2);
  tensor->ref_float_data().push_back(1.0f);
  tensor->ref_float_data().push_back(2.0f);
  tensor->ref_float_data().push_back(3.0f);
  tensor->ref_float_data().push_back(4.0f);

  // Serialize
  std::string serialized;
  optional1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), optional1.SerializeSize().size());

  // Deserialize
  OptionalProto optional2;
  optional2.ParseFromString(serialized);

  // Verify
  EXPECT_EQ(optional2.ref_elem_type(), 1);
  EXPECT_TRUE(optional2.has_tensor_value());
  EXPECT_EQ(optional2.ref_tensor_value().ref_name(), "serialized_tensor");
  EXPECT_EQ(optional2.ref_tensor_value().ref_dims().size(), 2);
  EXPECT_EQ(optional2.ref_tensor_value().ref_float_data().size(), 4);
  EXPECT_FLOAT_EQ(optional2.ref_tensor_value().ref_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(optional2.ref_tensor_value().ref_float_data()[1], 2.0f);
  EXPECT_FLOAT_EQ(optional2.ref_tensor_value().ref_float_data()[2], 3.0f);
  EXPECT_FLOAT_EQ(optional2.ref_tensor_value().ref_float_data()[3], 4.0f);
}

TEST(onnx_proto, OptionalProto_CopyFrom) {
  OptionalProto source;
  source.set_elem_type(1); // FLOAT

  TensorProto *tensor = source.add_tensor_value();
  tensor->set_name("source_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(3.14f);

  OptionalProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_elem_type(), 1);
  EXPECT_TRUE(target.has_tensor_value());
  EXPECT_EQ(target.ref_tensor_value().ref_name(), "source_tensor");
  EXPECT_EQ(target.ref_tensor_value().ref_float_data().size(), 1);
  EXPECT_FLOAT_EQ(target.ref_tensor_value().ref_float_data()[0], 3.14f);
}

TEST(onnx_string, OptionalProto) {
  utils::PrintOptions options;

  OptionalProto optional;
  optional.set_elem_type(1); // FLOAT

  TensorProto *tensor = optional.add_tensor_value();
  tensor->set_name("print_test_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_dims().push_back(2);
  tensor->ref_float_data().push_back(1.5f);
  tensor->ref_float_data().push_back(2.5f);

  std::stringstream ss_result;
  optional.PrintToStringStream(ss_result, options);
  std::string serialized = ss_result.str();
  ASSERT_FALSE(serialized.empty());

  EXPECT_TRUE(serialized.find("elem_type:") != std::string::npos);
  EXPECT_TRUE(serialized.find("1") != std::string::npos); // FLOAT type
  EXPECT_TRUE(serialized.find("tensor_value") != std::string::npos);
  EXPECT_TRUE(serialized.find("print_test_tensor") != std::string::npos);
}

TEST(onnx_proto, OptionalProto_EmptyOptional) {
  OptionalProto optional;

  EXPECT_EQ(optional.ref_elem_type(), 0);
  EXPECT_FALSE(optional.has_tensor_value());
  EXPECT_FALSE(optional.has_sparse_tensor_value());
  EXPECT_FALSE(optional.has_sequence_value());
  EXPECT_FALSE(optional.has_map_value());
  EXPECT_FALSE(optional.has_optional_value());

  // Serialize un optional vide
  std::string serialized;
  optional.SerializeToString(serialized);

  // The size should not be zero even for an empty optional
  // because the metadata is still serialized
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), optional.SerializeSize().size());

  // Deserialize
  OptionalProto optional2;
  optional2.ParseFromString(serialized);

  EXPECT_EQ(optional2.ref_elem_type(), 0);
  EXPECT_FALSE(optional2.has_tensor_value());
  EXPECT_FALSE(optional2.has_sparse_tensor_value());
  EXPECT_FALSE(optional2.has_sequence_value());
  EXPECT_FALSE(optional2.has_map_value());
  EXPECT_FALSE(optional2.has_optional_value());
}

TEST(onnx_proto, OptionalProto_SerializeSize) {
  OptionalProto optional;
  optional.set_elem_type(1); // FLOAT

  TensorProto *tensor = optional.add_tensor_value();
  tensor->set_name("size_test_tensor");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(1.0f);

  std::string serialized;
  optional.SerializeToString(serialized);

  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), optional.SerializeSize(stream, options).size());
}

TEST(onnx_proto, OptionalProto_MultipleValueTypes) {
  // Test to verify behavior when multiple value types are added
  // Normally, only the last value added should be valid

  OptionalProto optional;
  optional.set_elem_type(1); // FLOAT

  // Add a tensor
  TensorProto *tensor = optional.add_tensor_value();
  tensor->set_name("tensor_value");
  tensor->set_data_type(TensorProto::DataType::FLOAT);
  tensor->ref_float_data().push_back(1.0f);

  EXPECT_TRUE(optional.has_tensor_value());

  // Add a sequence
  SequenceProto *sequence = optional.add_sequence_value();
  sequence->set_elem_type(1); // FLOAT

  // Now, the optional should have a sequence and no longer have a tensor
  EXPECT_TRUE(optional.has_sequence_value());
  // EXPECT_FALSE(optional.has_tensor_value());

  // Add a map
  MapProto *map = optional.add_map_value();
  map->set_key_type(6); // INT32

  // Now, the optional should have a map and no longer have a sequence
  EXPECT_TRUE(optional.has_map_value());
  // EXPECT_FALSE(optional.has_sequence_value());

  // Serialize and deserialize to verify that only the last value is kept
  std::string serialized;
  optional.SerializeToString(serialized);

  OptionalProto optional2;
  optional2.ParseFromString(serialized);

  EXPECT_TRUE(optional2.has_map_value());
  // EXPECT_FALSE(optional2.has_sequence_value());
  // EXPECT_FALSE(optional2.has_tensor_value());
}

// Security: a length value > INT64_MAX in a length-delimited record must be
// rejected. Previously, CanRead() cast the uint64_t length to int64_t to do
// pos_ + len <= size_; a malicious length encoded as a 10-byte varint with the
// high bit set became a negative int64_t and bypassed the bounds check.
TEST(onnx_stream_security, StringStreamCanReadRejectsHugeLength) {
  std::vector<uint8_t> data{0x00, 0x01, 0x02, 0x03};
  utils::StringStream stream(data.data(), data.size());
  // Plausible bound: anything strictly larger than the underlying buffer.
  EXPECT_THROW(stream.CanRead(static_cast<uint64_t>(data.size()) + 1, "oob"), std::exception);
  // Length above INT64_MAX (would previously cast to negative and pass).
  EXPECT_THROW(stream.CanRead(static_cast<uint64_t>(INT64_MAX) + 1, "hi"), std::exception);
  EXPECT_THROW(stream.CanRead(UINT64_MAX, "max"), std::exception);
  // Sanity: a valid request still succeeds.
  EXPECT_NO_THROW(stream.CanRead(static_cast<uint64_t>(data.size()), "ok"));
}

TEST(onnx_stream_security, StringStreamLimitToNextRejectsHugeLength) {
  std::vector<uint8_t> data{0x00, 0x01, 0x02, 0x03};
  utils::StringStream stream(data.data(), data.size());
  EXPECT_THROW(stream.LimitToNext(static_cast<uint64_t>(INT64_MAX) + 1), std::exception);
  EXPECT_THROW(stream.LimitToNext(UINT64_MAX), std::exception);
}

TEST(onnx_proto, ParseFromZeroCopyStream_StringStream) {
  // Build a small ModelProto and serialize it.
  ModelProto model;
  model.set_ir_version(7);
  GraphProto *graph = model.add_graph();
  graph->set_name("zero_copy_graph");
  std::string serialized;
  model.SerializeToString(serialized);

  // Parse it back through ParseFromZeroCopyStream using a StringStream
  // (an in-memory zero-copy BinaryStream).
  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  EXPECT_TRUE(parsed.ParseFromZeroCopyStream(&stream));

  EXPECT_EQ(parsed.ir_version(), 7);
  ASSERT_TRUE(parsed.has_graph());
  EXPECT_EQ(parsed.graph().ref_name(), "zero_copy_graph");
}

TEST(onnx_proto, ParseFromZeroCopyStream_WithOptions) {
  TensorProto tensor;
  tensor.set_name("zc_tensor");
  tensor.set_data_type(static_cast<int32_t>(TensorProto::FLOAT));
  tensor.add_dims(2);
  tensor.add_dims(3);
  for (int i = 0; i < 6; ++i) {
    tensor.add_float_data(static_cast<float>(i));
  }
  std::string serialized;
  tensor.SerializeToString(serialized);

  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  TensorProto parsed;
  ParseOptions opts;
  EXPECT_TRUE(parsed.ParseFromZeroCopyStream(&stream, opts));

  EXPECT_EQ(parsed.ref_name(), "zc_tensor");
  EXPECT_EQ(parsed.data_type(), static_cast<int32_t>(TensorProto::FLOAT));
  ASSERT_EQ(parsed.float_data().size(), 6u);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(parsed.float_data()[i], static_cast<float>(i));
  }
}

TEST(onnx_proto, ParseFromZeroCopyStream_NullStreamThrows) {
  ModelProto model;
  utils::BinaryStream *null_stream = nullptr;
  EXPECT_THROW(model.ParseFromZeroCopyStream(null_stream), std::exception);
  ParseOptions opts;
  EXPECT_THROW(model.ParseFromZeroCopyStream(null_stream, opts), std::exception);
}

TEST(onnx_proto, SerializeFormat_OrtFlatbuffersParseFromZeroCopyStreamThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_zero_copy_parse");
  std::string serialized;
  model.SerializeToString(serialized);

  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  EXPECT_THROW(parsed.ParseFromZeroCopyStream(&stream, popts), std::exception);
}

TEST(onnx_proto, OrtFlatbuffersParseFromZeroCopyStreamZeroRecursionDepthThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_zero_copy_bad_depth");
  std::string serialized;
  model.SerializeToString(serialized);

  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_recursion_depth = 0;
  EXPECT_THROW(parsed.ParseFromZeroCopyStream(&stream, popts), std::runtime_error);
}

TEST(onnx_proto, OrtFlatbuffersParseFromZeroCopyStreamNegativeMaxTensorSizeBytesThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_zero_copy_bad_tensor_limit");
  std::string serialized;
  model.SerializeToString(serialized);

  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_tensor_size_bytes = -1;
  EXPECT_THROW(parsed.ParseFromZeroCopyStream(&stream, popts), std::runtime_error);
}

// Tests for SerializeFormat option.
TEST(onnx_proto, SerializeFormat_DefaultIsOnnx) {
  ParseOptions popts;
  SerializeOptions sopts;
  EXPECT_EQ(popts.format, SerializeFormat::kOnnx);
  EXPECT_EQ(sopts.format, SerializeFormat::kOnnx);
}

TEST(onnx_proto, SerializeFormat_OrtFlatbuffersSerializeToStringThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_serialize");
  SerializeOptions sopts;
  sopts.format = SerializeFormat::kOrtFlatbuffers;
  std::string out;
  EXPECT_THROW(model.SerializeToString(out, sopts), std::exception);
}

TEST(onnx_proto, SerializeFormat_OrtFlatbuffersParseFromStringThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_parse");
  std::string serialized;
  model.SerializeToString(serialized);

  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::exception);
}

TEST(onnx_proto, SerializeFormat_OrtFlatbuffersSerializeModelProtoToStreamThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_stream");
  SerializeOptions sopts;
  sopts.format = SerializeFormat::kOrtFlatbuffers;
  utils::StringWriteStream stream;
  EXPECT_THROW(SerializeModelProtoToStream(model, stream, sopts), std::exception);
}

TEST(onnx_proto, SerializeFormat_OrtFlatbuffersParseModelProtoFromStreamThrows) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_ort_pstream");
  std::string serialized;
  model.SerializeToString(serialized);
  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  EXPECT_THROW(ParseModelProtoFromStream(parsed, stream, popts), std::exception);
}

TEST(onnx_proto, SerializeFormat_OnnxRoundTripWorks) {
  // The default kOnnx path must remain unchanged.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_onnx_rt");
  SerializeOptions sopts;
  sopts.format = SerializeFormat::kOnnx;
  std::string serialized;
  model.SerializeToString(serialized, sopts);

  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOnnx;
  parsed.ParseFromString(serialized, popts);
  EXPECT_TRUE(parsed.has_graph());
  EXPECT_EQ(parsed.ref_graph().ref_name(), "g_onnx_rt");
}

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

// Wraps payload as a length-delimited (wire type 2) field with the given number.
std::string WrapLengthDelimitedField(int field_number, const std::string &payload) {
  uint64_t tag = (static_cast<uint64_t>(field_number) << 3) | 2;
  std::string out = EncodeVarint(tag);
  out += EncodeVarint(payload.size());
  out += payload;
  return out;
}

// Builds the wire bytes of a TypeProto nested `levels` deep through the
// self-recursive TypeProto.sequence_type (field 4) -> Sequence.elem_type
// (field 1) -> TypeProto chain. Each level adds two sub-message parses.
std::string BuildNestedTypeProto(int levels) {
  std::string inner; // innermost empty TypeProto
  for (int i = 0; i < levels; ++i) {
    std::string sequence = WrapLengthDelimitedField(1, inner); // Sequence.elem_type
    inner = WrapLengthDelimitedField(4, sequence);             // TypeProto.sequence_type
  }
  return inner;
}
} // namespace

TEST(onnx_proto, ParserRecursionLimitRejectsDeeplyNestedMessages) {
  // Each level adds two sub-message parses; 200 levels reaches depth 400, well
  // beyond the default recursion limit of 50, and must be rejected rather than
  // overflowing the stack / exhausting memory. The guard fires at depth
  // limit + 1, so the parser never recurses past the default limit.
  std::string deep = BuildNestedTypeProto(200);
  TypeProto parsed;
  ParseOptions popts;
  EXPECT_EQ(popts.max_recursion_depth, 50);
  EXPECT_THROW(parsed.ParseFromString(deep, popts), std::exception);
  // The recursion counter must be fully unwound even after a rejected parse so
  // the options object can be safely reused.
  EXPECT_EQ(popts._recursion_depth, 0);
}

TEST(onnx_proto, ParserRecursionLimitAcceptsShallowNesting) {
  // 10 levels reaches depth 20, comfortably within the default limit of 50.
  std::string shallow = BuildNestedTypeProto(10);
  TypeProto parsed;
  ParseOptions popts;
  EXPECT_NO_THROW(parsed.ParseFromString(shallow, popts));
  EXPECT_TRUE(parsed.has_sequence_type());
  // The recursion counter must return to 0 once parsing completes.
  EXPECT_EQ(popts._recursion_depth, 0);
}

TEST(onnx_proto, ParserRecursionLimitIsConfigurable) {
  // Lowering the limit rejects messages that the default would accept.
  std::string nested = BuildNestedTypeProto(10); // depth 20
  TypeProto parsed;
  ParseOptions popts;
  popts.max_recursion_depth = 5;
  EXPECT_THROW(parsed.ParseFromString(nested, popts), std::exception);
}

TEST(onnx_proto, OrtFlatbuffersParseFromStringZeroRecursionDepthThrows) {
  // max_recursion_depth must be > 0 for the ORT flatbuffer path.
  // The recursion-OOM guard fires before the "not implemented" stub so that
  // once the real parser lands the protection is already wired up.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_depth0");
  std::string serialized;
  model.SerializeToString(serialized);
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_recursion_depth = 0;
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::runtime_error);
}

TEST(onnx_proto, OrtFlatbuffersParseFromStringNegativeRecursionDepthThrows) {
  // A negative max_recursion_depth must also be rejected.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_depth_neg");
  std::string serialized;
  model.SerializeToString(serialized);
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_recursion_depth = -1;
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::runtime_error);
}

TEST(onnx_proto, MaxTensorSizeBytesDefaultIsZero) {
  // Default must be 0 (no limit) for backward compatibility.
  ParseOptions opts;
  EXPECT_EQ(opts.max_tensor_size_bytes, 0);
}

TEST(onnx_proto, MaxTensorSizeBytesRawDataThrows) {
  // Parsing a TensorProto whose raw_data exceeds the configured limit must
  // raise an error before any allocation is attempted.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(5);
  std::string raw(20, '\x01'); // 5 floats = 20 bytes
  for (char c : raw)
    tp.ref_raw_data().push_back(static_cast<uint8_t>(c));
  std::string serialized;
  SerializeOptions sopts;
  tp.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.max_tensor_size_bytes = 10; // Smaller than the 20-byte raw_data payload.
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::runtime_error);
}

TEST(onnx_proto, MaxTensorSizeBytesRawDataExactLimitAllowed) {
  // Parsing a TensorProto whose raw_data equals the limit exactly must succeed.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(5);
  for (int i = 0; i < 20; ++i)
    tp.ref_raw_data().push_back(0);
  std::string serialized;
  SerializeOptions sopts;
  tp.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.max_tensor_size_bytes = 20; // Exactly the raw_data size — must pass.
  EXPECT_NO_THROW(parsed.ParseFromString(serialized, popts));
  EXPECT_EQ(parsed.ref_raw_data().size(), 20u);
}

TEST(onnx_proto, MaxTensorSizeBytesZeroMeansNoLimit) {
  // max_tensor_size_bytes == 0 disables the limit entirely.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(100);
  for (int i = 0; i < 400; ++i)
    tp.ref_raw_data().push_back(0);
  std::string serialized;
  SerializeOptions sopts;
  tp.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.max_tensor_size_bytes = 0; // No limit.
  EXPECT_NO_THROW(parsed.ParseFromString(serialized, popts));
}

TEST(onnx_proto, MaxSerializedSizeBytesDefaultIsZero) {
  // Default must be 0 (no limit) for backward compatibility.
  SerializeOptions opts;
  EXPECT_EQ(opts.max_serialized_size_bytes, 0);
}

TEST(onnx_proto, MaxSerializedSizeBytesTensorSerializeToStringReturnsFalse) {
  // Serializing a tensor whose output exceeds the configured cap must fail early.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(5);
  std::string raw(20, '\x01'); // 5 floats = 20 bytes
  for (char c : raw)
    tp.ref_raw_data().push_back(static_cast<uint8_t>(c));
  std::string serialized;
  SerializeOptions sopts;
  sopts.max_serialized_size_bytes = 10;
  EXPECT_FALSE(tp.SerializeToString(serialized, sopts));
}

TEST(onnx_proto, MaxSerializedSizeBytesTensorExactLimitAllowed) {
  // A cap equal to the computed serialized size must be accepted.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(5);
  std::string raw(20, '\x01'); // 5 floats = 20 bytes
  for (char c : raw)
    tp.ref_raw_data().push_back(static_cast<uint8_t>(c));
  utils::StringWriteStream stream;
  SerializeOptions size_opts;
  SerializeSizeResult total_size = tp.SerializeSize(stream, size_opts);
  SerializeOptions sopts;
  sopts.max_serialized_size_bytes = total_size.size();
  std::string serialized;
  EXPECT_TRUE(tp.SerializeToString(serialized, sopts));
}

TEST(onnx_proto, MaxSerializedSizeBytesZeroMeansNoLimit) {
  // max_serialized_size_bytes == 0 disables the limit entirely.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(100);
  for (int i = 0; i < 400; ++i)
    tp.ref_raw_data().push_back(0);
  std::string serialized;
  SerializeOptions sopts;
  sopts.max_serialized_size_bytes = 0;
  EXPECT_TRUE(tp.SerializeToString(serialized, sopts));
}

TEST(onnx_proto, NegativeMaxSerializedSizeBytesThrows) {
  // A negative cap must be rejected.
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(1);
  for (int i = 0; i < 4; ++i)
    tp.ref_raw_data().push_back(0);
  std::string serialized;
  SerializeOptions sopts;
  sopts.max_serialized_size_bytes = -1;
  EXPECT_THROW(tp.SerializeToString(serialized, sopts), std::runtime_error);
}

TEST(onnx_proto, MaxSerializedSizeBytesSerializeModelProtoToStreamReturnsFalse) {
  // File/stream serialization path must enforce the same size cap.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_serialized_size");
  TensorProto *initializer = graph->add_initializer();
  initializer->set_name("W");
  initializer->set_data_type(TensorProto::FLOAT);
  initializer->add_dims(5);
  for (int i = 0; i < 20; ++i)
    initializer->ref_raw_data().push_back(1);
  SerializeOptions sopts;
  sopts.max_serialized_size_bytes = 10;
  utils::StringWriteStream stream;
  EXPECT_FALSE(SerializeModelProtoToStream(model, stream, sopts));
}

TEST(onnx_proto, MaxTensorSizeBytesPackedFloatDataThrows) {
  // The limit also applies to packed float_data (non-raw_data path).
  TensorProto tp;
  tp.set_data_type(static_cast<TensorProto::DataType>(1)); // FLOAT
  tp.add_dims(5);
  for (int i = 0; i < 5; ++i)
    tp.ref_float_data().push_back(static_cast<float>(i));
  std::string serialized;
  SerializeOptions sopts;
  tp.SerializeToString(serialized, sopts);

  TensorProto parsed;
  ParseOptions popts;
  popts.max_tensor_size_bytes = 4; // Each float is 4 bytes; 5 floats = 20 bytes total.
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::runtime_error);
}

TEST(onnx_proto, OrtFlatbuffersNegativeMaxTensorSizeBytesThrows) {
  // A negative max_tensor_size_bytes must be rejected for the ORT flatbuffer path.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_tensor_neg");
  std::string serialized;
  model.SerializeToString(serialized);
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_tensor_size_bytes = -1;
  EXPECT_THROW(parsed.ParseFromString(serialized, popts), std::runtime_error);
}

TEST(onnx_proto, OrtFlatbuffersParseModelProtoFromStreamNegativeMaxTensorSizeBytesThrows) {
  // ParseModelProtoFromStream also enforces max_tensor_size_bytes >= 0.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_stream_tensor_neg");
  std::string serialized;
  model.SerializeToString(serialized);
  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_tensor_size_bytes = -1;
  EXPECT_THROW(ParseModelProtoFromStream(parsed, stream, popts), std::runtime_error);
}

TEST(onnx_proto, OrtFlatbuffersParseModelProtoFromStreamZeroRecursionDepthThrows) {
  // ParseModelProtoFromStream also enforces max_recursion_depth > 0.
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g_stream_depth0");
  std::string serialized;
  model.SerializeToString(serialized);
  utils::StringStream stream(reinterpret_cast<const uint8_t *>(serialized.data()),
                             static_cast<int64_t>(serialized.size()));
  ModelProto parsed;
  ParseOptions popts;
  popts.format = SerializeFormat::kOrtFlatbuffers;
  popts.max_recursion_depth = 0;
  EXPECT_THROW(ParseModelProtoFromStream(parsed, stream, popts), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Additional coverage for serialization helpers and proto repeated fields.
// ---------------------------------------------------------------------------

TEST(onnx_string, String_LessThan) {
  utils::String abc("abc", 3);
  utils::String abd("abd", 3);
  utils::String ab("ab", 2);
  utils::RefString abc_ref("abc", 3);
  utils::RefString abd_ref("abd", 3);
  std::string abd_std("abd");
  std::string abc_std("abc");

  // operator<(const String &)
  EXPECT_TRUE(abc < abd);
  EXPECT_FALSE(abd < abc);
  EXPECT_TRUE(ab < abc); // shorter prefix sorts first
  EXPECT_FALSE(abc < abc);

  // compared against a RefString via its owning std::string
  EXPECT_TRUE(abc < std::string(abd_ref));
  EXPECT_FALSE(abd < std::string(abc_ref));
  EXPECT_FALSE(abc < std::string(abc_ref));

  // operator<(const std::string &)
  EXPECT_TRUE(abc < abd_std);
  EXPECT_FALSE(abc < abc_std);

  // operator<(const char *)
  EXPECT_TRUE(abc < "abd");
  EXPECT_FALSE(abc < "abc");
  EXPECT_TRUE(ab < "abc");
  EXPECT_FALSE(abc < "ab");
}

TEST(onnx_string, String_LessThanEdgeCases) {
  utils::String empty;
  utils::String a("a", 1);

  EXPECT_TRUE(empty < "a");
  EXPECT_FALSE(empty < "");
  // exhausting *this without exhausting other
  EXPECT_FALSE(a < "");
}

TEST(onnx_string, String_GreaterThan) {
  utils::String abc("abc", 3);
  utils::String abd("abd", 3);
  utils::String ab("ab", 2);
  utils::RefString abc_ref("abc", 3);
  utils::RefString abd_ref("abd", 3);
  std::string abc_std("abc");

  // operator>(const String &)
  EXPECT_TRUE(abd > abc);
  EXPECT_FALSE(abc > abd);
  EXPECT_TRUE(abc > ab); // longer string with shared prefix sorts later
  EXPECT_FALSE(abc > abc);

  // compared against a RefString via its owning std::string
  EXPECT_TRUE(abd > std::string(abc_ref));
  EXPECT_FALSE(abc > std::string(abd_ref));
  EXPECT_FALSE(abc > std::string(abc_ref));

  // operator>(const std::string &)
  EXPECT_TRUE(abd > abc_std);
  EXPECT_FALSE(abc > abc_std);

  // operator>(const char *)
  EXPECT_TRUE(abd > "abc");
  EXPECT_FALSE(abc > "abd");
  EXPECT_TRUE(abc > "ab");
  EXPECT_FALSE(abc > "abc");
}

TEST(onnx_string, String_GreaterThanEdgeCases) {
  utils::String empty;
  utils::String a("a", 1);

  EXPECT_FALSE(empty > "");
  EXPECT_FALSE(empty > "anything");
  // *this longer than other once other is exhausted
  EXPECT_TRUE(a > "");
}

TEST(onnx_string, OptionalString_Ordering) {
  utils::OptionalString unset;
  utils::OptionalString abc("abc");
  utils::OptionalString abd("abd");

  // absent sorts before any present value
  EXPECT_TRUE(unset < abc);
  EXPECT_FALSE(abc < unset);
  EXPECT_TRUE(unset <= abc);
  EXPECT_TRUE(unset <= unset);
  EXPECT_TRUE(abc > unset);
  EXPECT_FALSE(unset > abc);
  EXPECT_TRUE(abc >= unset);
  EXPECT_TRUE(unset >= unset);

  // present vs present
  EXPECT_TRUE(abc < abd);
  EXPECT_FALSE(abd < abc);
  EXPECT_TRUE(abc <= abd);
  EXPECT_TRUE(abc <= abc);
  EXPECT_TRUE(abd > abc);
  EXPECT_TRUE(abd >= abc);
  EXPECT_TRUE(abc >= abc);

  // != coverage
  EXPECT_TRUE(abc != abd);
  EXPECT_TRUE(abc != unset);
  EXPECT_FALSE(abc != abc);

  // against std::string
  std::string abd_std("abd");
  std::string abc_std("abc");
  EXPECT_TRUE(abc < abd_std);
  EXPECT_FALSE(abc < abc_std);
  EXPECT_TRUE(abc <= abc_std);
  EXPECT_TRUE(abd > abc_std);
  EXPECT_TRUE(abd >= abc_std);
  EXPECT_TRUE(unset < abc_std); // unset before any std::string
  EXPECT_TRUE(unset <= abc_std);
  EXPECT_FALSE(unset > abc_std);
  EXPECT_FALSE(unset >= abc_std);

  // against const char *
  EXPECT_TRUE(abc < "abd");
  EXPECT_FALSE(abc < "abc");
  EXPECT_TRUE(abc <= "abc");
  EXPECT_TRUE(abd > "abc");
  EXPECT_TRUE(abd >= "abc");
  EXPECT_TRUE(unset < "abc");
  EXPECT_FALSE(unset > "abc");
  // nullptr behaves like absent
  EXPECT_FALSE(unset < static_cast<const char *>(nullptr));
  EXPECT_TRUE(unset <= static_cast<const char *>(nullptr));
  EXPECT_FALSE(unset > static_cast<const char *>(nullptr));
  EXPECT_TRUE(unset >= static_cast<const char *>(nullptr));
  EXPECT_TRUE(abc > static_cast<const char *>(nullptr));
  EXPECT_TRUE(abc >= static_cast<const char *>(nullptr));
  EXPECT_FALSE(abc < static_cast<const char *>(nullptr));

  // concatenation (operator+)
  EXPECT_EQ(abc + "d", "abcd");
  EXPECT_EQ("x" + abc, "xabc");
  EXPECT_EQ(abc + abd, "abcabd");
}

TEST(onnx_stream, TwoFilesWriteStream_WriteRawBytesInSecondStream) {
  const std::string model_path = "test_write_second_stream.onnx";
  const std::string weights_path = "test_write_second_stream.onnx.data";
  const std::string extra_location = "test_write_second_stream_extra.bin";

  std::vector<uint8_t> default_bytes = {1, 2, 3, 4, 5};
  std::vector<uint8_t> extra_bytes = {9, 8, 7};

  {
    utils::TwoFilesWriteStream stream(model_path, weights_path);

    // Default-location overload writes to the primary weights file.
    stream.write_raw_bytes_in_second_stream(default_bytes.data(),
                                            static_cast<utils::offset_t>(default_bytes.size()));
    EXPECT_EQ(stream.weights_size(), static_cast<int64_t>(default_bytes.size()));

    // A zero-length write is a no-op and does not change the size.
    stream.write_raw_bytes_in_second_stream(default_bytes.data(), 0);
    EXPECT_EQ(stream.weights_size(), static_cast<int64_t>(default_bytes.size()));

    // Named-location overload routes bytes to a separate weights file.
    stream.write_raw_bytes_in_second_stream(
        extra_bytes.data(), static_cast<utils::offset_t>(extra_bytes.size()), extra_location);
    EXPECT_EQ(stream.weights_size_for_location(extra_location),
              static_cast<int64_t>(extra_bytes.size()));
    // The default-location size is unaffected by the named-location write.
    EXPECT_EQ(stream.weights_size(extra_location), static_cast<int64_t>(extra_bytes.size()));

    stream.FlushMainToFile();
  }

  // The default weights file holds exactly the default bytes.
  std::ifstream f_default(weights_path, std::ios::binary);
  ASSERT_TRUE(f_default.is_open());
  std::vector<uint8_t> read_default((std::istreambuf_iterator<char>(f_default)),
                                    std::istreambuf_iterator<char>());
  EXPECT_EQ(read_default, default_bytes);

  // The extra weights file holds exactly the named-location bytes.
  std::ifstream f_extra(extra_location, std::ios::binary);
  ASSERT_TRUE(f_extra.is_open());
  std::vector<uint8_t> read_extra((std::istreambuf_iterator<char>(f_extra)),
                                  std::istreambuf_iterator<char>());
  EXPECT_EQ(read_extra, extra_bytes);

  std::remove(model_path.c_str());
  std::remove(weights_path.c_str());
  std::remove(extra_location.c_str());
}

TEST(onnx_proto, TensorProto_SegmentSerialization) {
  TensorProto tensor1;
  tensor1.set_name("segmented");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_segment().set_begin(5);
  tensor1.ref_segment().set_end(10);

  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), tensor1.SerializeSize().size());

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  EXPECT_EQ(tensor2.ref_name(), "segmented");
  EXPECT_TRUE(tensor2.has_segment());
  EXPECT_EQ(tensor2.ref_segment().ref_begin(), 5);
  EXPECT_EQ(tensor2.ref_segment().ref_end(), 10);
}

TEST(onnx_proto, ValueInfoProto_MetadataPropsSerialization) {
  ValueInfoProto value_info1;
  value_info1.set_name("with_metadata");

  StringStringEntryProto *meta1 = value_info1.add_metadata_props();
  meta1->set_key("author");
  meta1->set_value("alice");
  StringStringEntryProto *meta2 = value_info1.add_metadata_props();
  meta2->set_key("unit");
  meta2->set_value("ms");

  std::string serialized;
  value_info1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), value_info1.SerializeSize().size());

  ValueInfoProto value_info2;
  value_info2.ParseFromString(serialized);

  EXPECT_EQ(value_info2.ref_name(), "with_metadata");
  ASSERT_EQ(value_info2.ref_metadata_props().size(), 2);
  EXPECT_EQ(value_info2.ref_metadata_props()[0].ref_key(), "author");
  EXPECT_EQ(value_info2.ref_metadata_props()[0].ref_value(), "alice");
  EXPECT_EQ(value_info2.ref_metadata_props()[1].ref_key(), "unit");
  EXPECT_EQ(value_info2.ref_metadata_props()[1].ref_value(), "ms");
}

TEST(onnx_proto, AttributeProto_TypeProtosSerialization) {
  AttributeProto attribute1;
  attribute1.set_name("types");
  attribute1.set_type(AttributeProto::AttributeType::TYPE_PROTOS);

  TypeProto *type1 = attribute1.add_type_protos();
  type1->add_tensor_type()->set_elem_type(1); // FLOAT
  TypeProto *type2 = attribute1.add_type_protos();
  type2->add_tensor_type()->set_elem_type(7); // INT64

  std::string serialized;
  attribute1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), attribute1.SerializeSize().size());

  AttributeProto attribute2;
  attribute2.ParseFromString(serialized);

  EXPECT_EQ(attribute2.ref_name(), "types");
  EXPECT_EQ(attribute2.ref_type(), AttributeProto::AttributeType::TYPE_PROTOS);
  ASSERT_EQ(attribute2.ref_type_protos().size(), 2);
  EXPECT_EQ(attribute2.ref_type_protos()[0].ref_tensor_type().ref_elem_type(), 1);
  EXPECT_EQ(attribute2.ref_type_protos()[1].ref_tensor_type().ref_elem_type(), 7);
}

TEST(onnx_proto, NodeProto_MetadataPropsSerialization) {
  NodeProto node1;
  node1.set_name("node_with_metadata");
  node1.set_op_type("Identity");

  StringStringEntryProto *meta1 = node1.add_metadata_props();
  meta1->set_key("namespace");
  meta1->set_value("layer1");
  StringStringEntryProto *meta2 = node1.add_metadata_props();
  meta2->set_key("color");
  meta2->set_value("blue");

  std::string serialized;
  node1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node1.SerializeSize().size());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_EQ(node2.ref_name(), "node_with_metadata");
  ASSERT_EQ(node2.ref_metadata_props().size(), 2);
  EXPECT_EQ(node2.ref_metadata_props()[0].ref_key(), "namespace");
  EXPECT_EQ(node2.ref_metadata_props()[0].ref_value(), "layer1");
  EXPECT_EQ(node2.ref_metadata_props()[1].ref_key(), "color");
  EXPECT_EQ(node2.ref_metadata_props()[1].ref_value(), "blue");
}

TEST(onnx_proto, NodeProto_DeviceConfigurationsSerialization) {
  NodeProto node1;
  node1.set_name("node_with_device_config");
  node1.set_op_type("MatMul");

  NodeDeviceConfigurationProto *config = node1.add_device_configurations();
  config->set_configuration_id("config_a");
  config->set_pipeline_stage(2);
  ShardingSpecProto *spec = config->add_sharding_spec();
  spec->set_tensor_name("X");
  *spec->add_device() = 0;
  *spec->add_device() = 1;

  std::string serialized;
  node1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node1.SerializeSize().size());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_EQ(node2.ref_name(), "node_with_device_config");
  ASSERT_EQ(node2.ref_device_configurations().size(), 1);
  EXPECT_EQ(node2.ref_device_configurations()[0].ref_configuration_id(), "config_a");
  EXPECT_TRUE(node2.ref_device_configurations()[0].has_pipeline_stage());
  EXPECT_EQ(node2.ref_device_configurations()[0].ref_pipeline_stage(), 2);
  ASSERT_EQ(node2.ref_device_configurations()[0].ref_sharding_spec().size(), 1);
  EXPECT_EQ(node2.ref_device_configurations()[0].ref_sharding_spec()[0].ref_tensor_name(), "X");
  ASSERT_EQ(node2.ref_device_configurations()[0].ref_sharding_spec()[0].ref_device().size(), 2);
  EXPECT_EQ(node2.ref_device_configurations()[0].ref_sharding_spec()[0].ref_device()[0], 0);
  EXPECT_EQ(node2.ref_device_configurations()[0].ref_sharding_spec()[0].ref_device()[1], 1);
}

TEST(onnx_proto, GraphProto_MetadataPropsSerialization) {
  GraphProto graph1;
  graph1.set_name("graph_with_metadata");

  StringStringEntryProto *meta1 = graph1.add_metadata_props();
  meta1->set_key("framework");
  meta1->set_value("onnx-light");
  StringStringEntryProto *meta2 = graph1.add_metadata_props();
  meta2->set_key("stage");
  meta2->set_value("test");

  std::string serialized;
  graph1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), graph1.SerializeSize().size());

  GraphProto graph2;
  graph2.ParseFromString(serialized);

  EXPECT_EQ(graph2.ref_name(), "graph_with_metadata");
  ASSERT_EQ(graph2.ref_metadata_props().size(), 2);
  EXPECT_EQ(graph2.ref_metadata_props()[0].ref_key(), "framework");
  EXPECT_EQ(graph2.ref_metadata_props()[0].ref_value(), "onnx-light");
  EXPECT_EQ(graph2.ref_metadata_props()[1].ref_key(), "stage");
  EXPECT_EQ(graph2.ref_metadata_props()[1].ref_value(), "test");
}

TEST(onnx_proto, GraphProto_QuantizationAnnotationSerialization) {
  GraphProto graph1;
  graph1.set_name("graph_with_quantization");

  TensorAnnotation *annotation = graph1.add_quantization_annotation();
  annotation->set_tensor_name("a");
  StringStringEntryProto *scale = annotation->add_quant_parameter_tensor_names();
  scale->set_key("SCALE_TENSOR");
  scale->set_value("a_scale");
  StringStringEntryProto *zero_point = annotation->add_quant_parameter_tensor_names();
  zero_point->set_key("ZERO_POINT_TENSOR");
  zero_point->set_value("a_zero_point");

  std::string serialized;
  graph1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), graph1.SerializeSize().size());

  GraphProto graph2;
  graph2.ParseFromString(serialized);

  EXPECT_EQ(graph2.ref_name(), "graph_with_quantization");
  ASSERT_EQ(graph2.ref_quantization_annotation().size(), 1);
  EXPECT_EQ(graph2.ref_quantization_annotation()[0].ref_tensor_name(), "a");
  ASSERT_EQ(graph2.ref_quantization_annotation()[0].ref_quant_parameter_tensor_names().size(), 2);
  EXPECT_EQ(graph2.ref_quantization_annotation()[0].ref_quant_parameter_tensor_names()[0].ref_key(),
            "SCALE_TENSOR");
  EXPECT_EQ(
      graph2.ref_quantization_annotation()[0].ref_quant_parameter_tensor_names()[0].ref_value(),
      "a_scale");
  EXPECT_EQ(graph2.ref_quantization_annotation()[0].ref_quant_parameter_tensor_names()[1].ref_key(),
            "ZERO_POINT_TENSOR");
  EXPECT_EQ(
      graph2.ref_quantization_annotation()[0].ref_quant_parameter_tensor_names()[1].ref_value(),
      "a_zero_point");
}

TEST(onnx_proto, FunctionProto_MetadataPropsSerialization) {
  FunctionProto function1;
  function1.set_name("function_with_metadata");
  function1.set_domain("ai.test");

  StringStringEntryProto *meta1 = function1.add_metadata_props();
  meta1->set_key("author");
  meta1->set_value("bob");
  StringStringEntryProto *meta2 = function1.add_metadata_props();
  meta2->set_key("version");
  meta2->set_value("1");

  std::string serialized;
  function1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), function1.SerializeSize().size());

  FunctionProto function2;
  function2.ParseFromString(serialized);

  EXPECT_EQ(function2.ref_name(), "function_with_metadata");
  EXPECT_EQ(function2.ref_domain(), "ai.test");
  ASSERT_EQ(function2.ref_metadata_props().size(), 2);
  EXPECT_EQ(function2.ref_metadata_props()[0].ref_key(), "author");
  EXPECT_EQ(function2.ref_metadata_props()[0].ref_value(), "bob");
  EXPECT_EQ(function2.ref_metadata_props()[1].ref_key(), "version");
  EXPECT_EQ(function2.ref_metadata_props()[1].ref_value(), "1");
}

TEST(onnx_proto, ModelProto_ConfigurationSerialization) {
  ModelProto model1;
  model1.set_ir_version(10);
  GraphProto *graph = model1.add_graph();
  graph->set_name("graph_with_configuration");

  DeviceConfigurationProto *config1 = model1.add_configuration();
  config1->set_name("two_devices");
  config1->set_num_devices(2);
  *config1->add_device() = "gpu0";
  *config1->add_device() = "gpu1";

  DeviceConfigurationProto *config2 = model1.add_configuration();
  config2->set_name("single_device");
  config2->set_num_devices(1);
  *config2->add_device() = "cpu";

  std::string serialized;
  model1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), model1.SerializeSize().size());

  ModelProto model2;
  model2.ParseFromString(serialized);

  ASSERT_EQ(model2.ref_configuration().size(), 2);
  EXPECT_EQ(model2.ref_configuration()[0].ref_name(), "two_devices");
  EXPECT_EQ(model2.ref_configuration()[0].ref_num_devices(), 2);
  ASSERT_EQ(model2.ref_configuration()[0].ref_device().size(), 2);
  EXPECT_EQ(model2.ref_configuration()[0].ref_device()[0], "gpu0");
  EXPECT_EQ(model2.ref_configuration()[0].ref_device()[1], "gpu1");
  EXPECT_EQ(model2.ref_configuration()[1].ref_name(), "single_device");
  EXPECT_EQ(model2.ref_configuration()[1].ref_num_devices(), 1);
  ASSERT_EQ(model2.ref_configuration()[1].ref_device().size(), 1);
  EXPECT_EQ(model2.ref_configuration()[1].ref_device()[0], "cpu");
}

TEST(onnx_proto, ModelProto_SerializeToStringWithOptions) {
  ModelProto model1;
  model1.set_ir_version(9);
  model1.set_producer_name("options_producer");
  GraphProto *graph = model1.add_graph();
  graph->set_name("options_graph");
  NodeProto *node = graph->add_node();
  node->set_name("identity");
  node->set_op_type("Identity");

  // Default overload and the SerializeOptions overload must agree.
  std::string serialized_default;
  model1.SerializeToString(serialized_default);

  SerializeOptions options;
  std::string serialized_with_options;
  model1.SerializeToString(serialized_with_options, options);

  EXPECT_FALSE(serialized_with_options.empty());
  EXPECT_EQ(serialized_default, serialized_with_options);

  ModelProto model2;
  model2.ParseFromString(serialized_with_options);
  EXPECT_EQ(model2.ref_ir_version(), 9);
  EXPECT_EQ(model2.ref_producer_name(), "options_producer");
  ASSERT_TRUE(model2.has_graph());
  EXPECT_EQ(model2.ref_graph().ref_name(), "options_graph");
  ASSERT_EQ(model2.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model2.ref_graph().ref_node()[0].ref_op_type(), "Identity");
}

TEST(onnx_proto, SequenceProto_ParseFromStream) {
  SequenceProto sequence1;
  sequence1.set_elem_type(1); // FLOAT

  TensorProto *tensor1 = sequence1.add_tensor_values();
  tensor1->set_name("seq_tensor1");
  tensor1->set_data_type(TensorProto::DataType::FLOAT);
  tensor1->ref_float_data().push_back(1.5f);

  TensorProto *tensor2 = sequence1.add_tensor_values();
  tensor2->set_name("seq_tensor2");
  tensor2->set_data_type(TensorProto::DataType::FLOAT);
  tensor2->ref_float_data().push_back(2.5f);

  // Serialize into an in-memory stream and parse it back via ParseFromStream.
  utils::StringWriteStream write_stream;
  SerializeOptions serialize_options;
  SerializeSizeResult total_size = sequence1.SerializeSize(write_stream, serialize_options);
  write_stream.pre_allocate(total_size.size());
  sequence1.SerializeToStream(write_stream, serialize_options);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  ParseOptions parse_options;
  SequenceProto sequence2;
  EXPECT_TRUE(sequence2.ParseFromStream(read_stream, parse_options));

  EXPECT_EQ(sequence2.ref_elem_type(), 1);
  ASSERT_EQ(sequence2.ref_tensor_values().size(), 2);
  EXPECT_EQ(sequence2.ref_tensor_values()[0].ref_name(), "seq_tensor1");
  EXPECT_EQ(sequence2.ref_tensor_values()[1].ref_name(), "seq_tensor2");
  ASSERT_EQ(sequence2.ref_tensor_values()[0].ref_float_data().size(), 1);
  EXPECT_FLOAT_EQ(sequence2.ref_tensor_values()[0].ref_float_data()[0], 1.5f);
  ASSERT_EQ(sequence2.ref_tensor_values()[1].ref_float_data().size(), 1);
  EXPECT_FLOAT_EQ(sequence2.ref_tensor_values()[1].ref_float_data()[0], 2.5f);
}

// ---------------------------------------------------------------------------
// ByteSpan::assign_with_deleter and TensorProto::set_raw_data_with_deleter
// ---------------------------------------------------------------------------

TEST(onnx_proto, ByteSpan_AssignWithDeleter_CallsDeleterOnDestruction) {
  // Verify that the deleter is called exactly once when the last ByteSpan that
  // holds the owner token is destroyed.
  bool deleter_called = false;

  std::vector<uint8_t> buf = {1, 2, 3, 4};
  {
    utils::ByteSpan span;
    span.assign_with_deleter(buf.data(), buf.size(),
                             [&deleter_called]() { deleter_called = true; });
    EXPECT_TRUE(span.is_borrowed());
    EXPECT_EQ(span.size(), buf.size());
    // Use a const reference to call the read-only data() overload on a borrowed span.
    const utils::ByteSpan &cspan = span;
    EXPECT_EQ(cspan.data(), buf.data());
    EXPECT_FALSE(deleter_called);
  } // span is destroyed here → owner token refcount reaches 0 → deleter fires
  EXPECT_TRUE(deleter_called);
}

TEST(onnx_proto, ByteSpan_AssignWithDeleter_DeleterCalledOnceAfterCopy) {
  // Copying a ByteSpan with a deleter increments the shared owner refcount.
  // The deleter must fire exactly once when both copies are gone.
  int call_count = 0;

  std::vector<uint8_t> buf = {10, 20, 30};
  {
    utils::ByteSpan span1;
    span1.assign_with_deleter(buf.data(), buf.size(), [&call_count]() { ++call_count; });
    {
      utils::ByteSpan span2(span1); // copy — shares the owner token
      EXPECT_TRUE(span2.is_borrowed());
      EXPECT_EQ(span2.size(), buf.size());
      EXPECT_EQ(call_count, 0);
    } // span2 destroyed; refcount goes from 2→1, deleter NOT called yet
    EXPECT_EQ(call_count, 0);
  } // span1 destroyed; refcount goes from 1→0, deleter called exactly once
  EXPECT_EQ(call_count, 1);
}

TEST(onnx_proto, ByteSpan_AssignWithDeleter_ClearReleasesOwnerToken) {
  // clear() must release the owner token, triggering the deleter.
  bool deleter_called = false;

  std::vector<uint8_t> buf = {5, 6, 7};
  utils::ByteSpan span;
  span.assign_with_deleter(buf.data(), buf.size(), [&deleter_called]() { deleter_called = true; });
  EXPECT_FALSE(deleter_called);
  span.clear();
  EXPECT_TRUE(deleter_called);
  EXPECT_TRUE(span.empty());
}

TEST(onnx_proto, TensorProto_SetRawDataWithDeleter_DeletedOnDestruction) {
  // TensorProto::set_raw_data_with_deleter should attach the deleter to the
  // tensor's raw_data ByteSpan so it fires when the tensor is destroyed.
  bool deleter_called = false;

  std::vector<uint8_t> raw = {0xAA, 0xBB, 0xCC, 0xDD};
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT8);
    tensor.ref_dims().push_back(4);
    tensor.set_raw_data_with_deleter(raw.data(), raw.size(),
                                     [&deleter_called]() { deleter_called = true; });

    // Use a const reference to call the read-only data() overload on a borrowed span.
    const utils::ByteSpan &raw_span = tensor.ref_raw_data();
    EXPECT_TRUE(raw_span.is_borrowed());
    EXPECT_EQ(raw_span.size(), raw.size());
    EXPECT_EQ(raw_span.data(), raw.data());
    EXPECT_TRUE(tensor.is_raw_data());
    EXPECT_FALSE(deleter_called);
  } // tensor destroyed here → deleter fires
  EXPECT_TRUE(deleter_called);
}

TEST(onnx_proto, TensorProto_SetRawDataWithDeleter_NoOpDeleter) {
  // A no-op deleter (lambda that does nothing) should work without errors.
  std::vector<uint8_t> raw = {1, 2, 3, 4};
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT8);
    tensor.ref_dims().push_back(4);
    tensor.set_raw_data_with_deleter(raw.data(), raw.size(), []() {});

    EXPECT_TRUE(tensor.ref_raw_data().is_borrowed());
    EXPECT_EQ(tensor.ref_raw_data().size(), raw.size());
    // Use a const reference to call the read-only data() overload on a borrowed span.
    const utils::ByteSpan &raw_span = tensor.ref_raw_data();
    EXPECT_EQ(raw_span.data()[0], uint8_t{1});
    EXPECT_EQ(raw_span.data()[3], uint8_t{4});
  }
  // No crash expected.
}

// ---------------------------------------------------------------------------
// ParseOptions::raw_data_callback and TensorProto::attach_raw_data_deleter
// ---------------------------------------------------------------------------

TEST(onnx_proto, ByteSpan_AttachDeleter_KeepsDataAndFiresOnDestruction) {
  // Attaches a deleter while keeping the current bytes in place (owned mode here) and calls the
  // deleter exactly once when the span is destroyed.
  bool deleter_called = false;
  {
    utils::ByteSpan span;
    span.resize(4);
    span.data()[0] = 7;
    span.data()[3] = 9;
    span.attach_deleter([&deleter_called]() { deleter_called = true; });
    EXPECT_FALSE(span.is_borrowed());
    const utils::ByteSpan &cspan = span;
    EXPECT_EQ(cspan.size(), 4u);
    EXPECT_EQ(cspan.data()[0], uint8_t{7});
    EXPECT_EQ(cspan.data()[3], uint8_t{9});
    EXPECT_FALSE(deleter_called);
  } // span destroyed → deleter fires
  EXPECT_TRUE(deleter_called);
}

TEST(onnx_proto, TensorProto_AttachRawDataDeleter_FiresOnDestruction) {
  bool deleter_called = false;
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT8);
    tensor.ref_dims().push_back(4);
    tensor.ref_raw_data().resize(4);
    tensor.attach_raw_data_deleter([&deleter_called]() { deleter_called = true; });
    EXPECT_TRUE(tensor.is_raw_data());
    EXPECT_FALSE(deleter_called);
  }
  EXPECT_TRUE(deleter_called);
}

TEST(onnx_proto, ParseOptions_RawDataCallback_InvokedAndDeleterAttached) {
  // Builds a tensor with raw_data, serializes it, then parses it back with a callback that
  // returns a deleter. The callback must be invoked for the tensor and the returned deleter
  // must fire when the parsed tensor is destroyed.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  TensorProto tensor1;
  tensor1.set_name("weights");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(static_cast<int64_t>(data.size()));
  tensor1.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor1.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::string serialized;
  tensor1.SerializeToString(serialized);

  int callback_calls = 0;
  bool deleter_called = false;
  std::string seen_name;
  size_t seen_size = 0;

  ParseOptions options;
  options.raw_data_callback = [&](TensorProto &t) -> std::function<void()> {
    ++callback_calls;
    seen_name = t.ref_name();
    seen_size = t.ref_raw_data().size();
    return [&deleter_called]() { deleter_called = true; };
  };

  {
    TensorProto tensor2;
    tensor2.ParseFromString(serialized, options);
    EXPECT_EQ(callback_calls, 1);
    EXPECT_EQ(seen_name, "weights");
    EXPECT_EQ(seen_size, data.size() * sizeof(float));
    EXPECT_EQ(tensor2.ref_raw_data().size(), data.size() * sizeof(float));
    EXPECT_FALSE(deleter_called);
  } // tensor2 destroyed → attached deleter fires
  EXPECT_TRUE(deleter_called);
}

TEST(onnx_proto, ParseOptions_RawDataCallback_EmptyReturnLeavesOwnershipUnchanged) {
  // Tests that a callback returning an empty std::function does not attach any deleter and
  // leaves the parsed tensor's data intact.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorProto tensor1;
  tensor1.set_name("w");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(static_cast<int64_t>(data.size()));
  tensor1.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor1.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::string serialized;
  tensor1.SerializeToString(serialized);

  int callback_calls = 0;
  ParseOptions options;
  options.raw_data_callback = [&](TensorProto &) -> std::function<void()> {
    ++callback_calls;
    return {}; // leave ownership unchanged
  };

  TensorProto tensor2;
  tensor2.ParseFromString(serialized, options);
  EXPECT_EQ(callback_calls, 1);
  EXPECT_EQ(tensor2.ref_raw_data().size(), data.size() * sizeof(float));
  const float *raw = reinterpret_cast<const float *>(tensor2.ref_raw_data().data());
  EXPECT_EQ(raw[0], 1.0f);
  EXPECT_EQ(raw[3], 4.0f);
}

TEST(onnx_proto, ParseOptions_RawDataCallback_NotInvokedWithoutRawData) {
  // Verifies that a tensor without raw_data (data stored in float_data) does not trigger the
  // callback.
  TensorProto tensor1;
  tensor1.set_name("no_raw");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);

  std::string serialized;
  tensor1.SerializeToString(serialized);

  int callback_calls = 0;
  ParseOptions options;
  options.raw_data_callback = [&](TensorProto &) -> std::function<void()> {
    ++callback_calls;
    return {};
  };

  TensorProto tensor2;
  tensor2.ParseFromString(serialized, options);
  EXPECT_EQ(callback_calls, 0);
}

TEST(onnx_proto, ParseFromIstream_ModelProto) {
  // Build a minimal model and serialize it to a string buffer.
  ModelProto model;
  model.set_ir_version(8);
  model.set_producer_name("test_producer");
  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  std::string serialized;
  model.SerializeToString(serialized);

  // Parse back via ParseFromIstream using std::istringstream.
  std::istringstream iss(serialized, std::ios::binary);
  ModelProto parsed;
  EXPECT_TRUE(parsed.ParseFromIstream(&iss));
  EXPECT_TRUE(parsed.has_ir_version());
  EXPECT_EQ(parsed.ref_ir_version(), 8);
  EXPECT_TRUE(parsed.has_producer_name());
  EXPECT_EQ(parsed.ref_producer_name(), "test_producer");
  EXPECT_TRUE(parsed.has_graph());
  EXPECT_EQ(parsed.ref_graph().ref_name(), "test_graph");
}

TEST(onnx_proto, ParseFromIstream_NullReturnsEnforceFailure) {
  ModelProto model;
  EXPECT_THROW(model.ParseFromIstream(nullptr), std::runtime_error);
}

TEST(onnx_proto, ParseFromIstream_TensorProto) {
  // Verify that ParseFromIstream works for proto types other than ModelProto.
  TensorProto tensor;
  tensor.set_name("my_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);

  std::string serialized;
  tensor.SerializeToString(serialized);

  std::istringstream iss(serialized, std::ios::binary);
  TensorProto parsed;
  EXPECT_TRUE(parsed.ParseFromIstream(&iss));
  EXPECT_EQ(parsed.ref_name(), "my_tensor");
  EXPECT_EQ(parsed.ref_dims().size(), 1u);
  EXPECT_EQ(parsed.ref_dims()[0], 2);
  EXPECT_EQ(parsed.ref_float_data().size(), 2u);
  EXPECT_EQ(parsed.ref_float_data()[0], 1.0f);
  EXPECT_EQ(parsed.ref_float_data()[1], 2.0f);
}

// ---------- RepeatedProtoField move-from-vector overloads ----------

TEST(RepeatedProtoField, MoveConstructFromVector) {
  std::vector<NodeProto> nodes(2);
  nodes[0].set_op_type("Add");
  nodes[0].add_input("a");
  nodes[1].set_op_type("Mul");
  nodes[1].add_input("b");

  utils::RepeatedProtoField<NodeProto> field(std::move(nodes));

  ASSERT_EQ(field.size(), 2u);
  EXPECT_EQ(field[0].ref_op_type(), "Add");
  EXPECT_EQ(field[0].ref_input()[0], "a");
  EXPECT_EQ(field[1].ref_op_type(), "Mul");
  EXPECT_EQ(field[1].ref_input()[0], "b");
  // The source vector should have been drained.
  EXPECT_TRUE(nodes.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(RepeatedProtoField, ExtendFromMovedVector) {
  utils::RepeatedProtoField<NodeProto> field;
  NodeProto seed;
  seed.set_op_type("Neg");
  field.push_back(seed);

  std::vector<NodeProto> extras(2);
  extras[0].set_op_type("Exp");
  extras[1].set_op_type("Log");

  field.extend(std::move(extras));

  ASSERT_EQ(field.size(), 3u);
  EXPECT_EQ(field[0].ref_op_type(), "Neg");
  EXPECT_EQ(field[1].ref_op_type(), "Exp");
  EXPECT_EQ(field[2].ref_op_type(), "Log");
  EXPECT_TRUE(extras.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(RepeatedProtoField, MoveConstructFromTemporaryVector) {
  // Implicit conversion from a temporary vector — the rvalue overload
  // should be selected so no deep copy happens.
  auto make = []() {
    std::vector<NodeProto> v(1);
    v[0].set_op_type("Relu");
    return v;
  };
  utils::RepeatedProtoField<NodeProto> field(make());
  ASSERT_EQ(field.size(), 1u);
  EXPECT_EQ(field[0].ref_op_type(), "Relu");
}
