// Unit tests for the custom value representation implemented by PR02 of the
// "custom, quantized and persistent values" plan: the StructTypeProto /
// EncodedValueProto wire messages, the field-1000 extension branches, and the
// verification, field-location, and payload-access helpers.
#include "encoded_value_test_helpers.h"
#include "onnx_verify.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

/** Measures a TypeProto through the catalogue's public fixed-layout API. */
bool FixedTypeBits(const StructTypeCatalogue &catalogue, const TypeProto &type, uint64_t &bits) {
  StructTypeProto wrapper;
  wrapper.ref_array().set_dimension(uint64_t(1));
  wrapper.ref_array().set_element_type(type);
  return catalogue.FixedBitSize(wrapper, bits);
}

/** Splits @p path on the first dot and returns the leading segment. */
std::string_view SplitPath(std::string_view path, std::string_view &rest) {
  const size_t dot = path.find('.');
  if (dot == std::string_view::npos) {
    rest = std::string_view();
    return path;
  }
  rest = path.substr(dot + 1);
  return path.substr(0, dot);
}

bool LocateInStruct(const StructTypeCatalogue &catalogue, const StructTypeProto &type,
                    std::string_view path, uint64_t base_bits, EncodedFieldRef &out);

/** Describes a fixed-width leaf reachable through a TypeProto. */
bool LocateInType(const StructTypeCatalogue &catalogue, const TypeProto &type,
                  std::string_view path, uint64_t base_bits, EncodedFieldRef &out) {
  if (type.value_case() == TypeProto::kStructType) {
    return LocateInStruct(catalogue, type.ref_struct_type(), path, base_bits, out);
  }
  if (!path.empty() || type.value_case() != TypeProto::kTensorType) {
    return false;
  }
  const auto &tensor = type.ref_tensor_type();
  const uint32_t width = FixedBitWidth(tensor.elem_type());
  uint64_t bits = 0;
  if (width == 0 || !FixedTypeBits(catalogue, type, bits)) {
    return false;
  }
  out.bit_offset = base_bits;
  out.bit_stride = width;
  out.bit_width = width;
  out.count = bits / width;
  out.elem_type = tensor.elem_type();
  return true;
}

/** Walks @p path inside @p type, accumulating bit offsets in declaration order. */
bool LocateInStruct(const StructTypeCatalogue &catalogue, const StructTypeProto &type,
                    std::string_view path, uint64_t base_bits, EncodedFieldRef &out) {
  const StructTypeProto &resolved = catalogue.Resolve(type);
  switch (resolved.kind_case()) {
  case StructTypeProto::kStructure: {
    std::string_view rest;
    const std::string_view head = SplitPath(path, rest);
    if (head.empty()) {
      return false;
    }
    uint64_t offset = base_bits;
    for (const auto &field : resolved.ref_structure().ref_field()) {
      if (field.name().sv() == head) {
        if (field.content_case() != StructTypeProto::Structure::Field::kType) {
          return false;
        }
        return LocateInType(catalogue, field.ref_type(), rest, offset, out);
      }
      if (field.content_case() == StructTypeProto::Structure::Field::kConstant) {
        continue;
      }
      uint64_t field_bits = 0;
      if (!FixedTypeBits(catalogue, field.ref_type(), field_bits) ||
          offset > std::numeric_limits<uint64_t>::max() - field_bits) {
        return false;
      }
      offset += field_bits;
    }
    return false;
  }
  case StructTypeProto::kBitPacking: {
    std::string_view rest;
    const std::string_view head = SplitPath(path, rest);
    if (head.empty() || !rest.empty()) {
      return false;
    }
    const auto &packing = resolved.ref_bit_packing();
    uint64_t bits = 0;
    if (packing.ref_dimension() == 0 || !catalogue.FixedBitSize(resolved, bits)) {
      return false;
    }
    const uint64_t group = bits / packing.ref_dimension();
    uint64_t offset = base_bits;
    for (const auto &component : packing.ref_component()) {
      if (component.name().sv() == head) {
        out.bit_offset = offset;
        out.bit_stride = group;
        out.bit_width = component.ref_bit_width();
        out.count = packing.ref_dimension();
        out.elem_type = TensorProto::UNDEFINED;
        return true;
      }
      if (offset > std::numeric_limits<uint64_t>::max() - component.ref_bit_width()) {
        return false;
      }
      offset += component.ref_bit_width();
    }
    return false;
  }
  case StructTypeProto::kArray: {
    if (!path.empty() || !resolved.ref_array().has_element_type()) {
      return false;
    }
    EncodedFieldRef element;
    if (!LocateInType(catalogue, resolved.ref_array().ref_element_type(), std::string_view(),
                      base_bits, element)) {
      return false;
    }
    const uint64_t dimension = resolved.ref_array().ref_dimension();
    if (element.count != 0 && dimension > std::numeric_limits<uint64_t>::max() / element.count) {
      return false;
    }
    out = element;
    out.count *= dimension;
    return true;
  }
  default:
    return false;
  }
}

bool FindEncodedField(const StructTypeCatalogue &catalogue, const StructTypeProto &root,
                      std::string_view path, EncodedFieldRef &out) {
  out = EncodedFieldRef();
  return LocateInStruct(catalogue, root, path, 0, out);
}

constexpr uint64_t kInt4Block = 2001;
constexpr uint64_t kLinearInt4Fixed = 1003;
constexpr uint64_t kLinearInt4Parameters = 1002;
constexpr uint64_t kCodebookBlock = 1101;
constexpr uint64_t kScaledCodebookBlock = 1102;
constexpr uint64_t kInt8Block = 1201;
constexpr uint64_t kKeyValueCache = 1301;

/** Encodes a value as a base-128 varint (protobuf wire format). */
std::string EncodeVarint(uint64_t value) {
  std::string out;
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
  return out;
}

/** Builds a protobuf field tag (field_number << 3 | wire_type) as a varint. */
std::string FieldTag(int field_number, int wire_type) {
  return EncodeVarint((static_cast<uint64_t>(field_number) << 3) |
                      static_cast<uint64_t>(wire_type));
}

/** Builds a tensor type with concrete dimensions. */
TypeProto MakeTensorType(TensorProto::DataType elem_type, const std::vector<int64_t> &dims) {
  TypeProto type;
  TypeProto::Tensor &tensor = type.ref_tensor_type();
  tensor.set_elem_type(elem_type);
  TensorShapeProto &shape = tensor.ref_shape();
  for (const int64_t dim : dims) {
    shape.add_dim()->set_dim_value(dim);
  }
  return type;
}

/** Builds a tensor type whose single dimension is symbolic. */
TypeProto MakeDynamicTensorType(TensorProto::DataType elem_type, const std::string &dim_param) {
  TypeProto type;
  TypeProto::Tensor &tensor = type.ref_tensor_type();
  tensor.set_elem_type(elem_type);
  tensor.ref_shape().add_dim()->set_dim_param(dim_param);
  return type;
}

/** Builds a struct type describing a tight array of scalar elements. */
TypeProto MakeArrayType(TensorProto::DataType elem_type, uint64_t dimension) {
  TypeProto type;
  StructTypeProto::Array &array = type.ref_struct_type().ref_array();
  array.set_element_type(MakeTensorType(elem_type, {}));
  array.set_dimension(dimension);
  return type;
}

/** Builds a struct type referencing a catalogue identity. */
TypeProto MakeTypeRef(uint64_t type_id) {
  TypeProto type;
  type.ref_struct_type().set_type_ref(type_id);
  return type;
}

/** Adds a typed field to a structure and returns it. */
StructTypeProto::Structure::Field *AddField(StructTypeProto::Structure &structure, const char *name,
                                            const TypeProto &type) {
  StructTypeProto::Structure::Field *field = structure.add_field();
  field->set_name(name);
  field->set_type(type);
  return field;
}

/** Adds a constant field to a structure and returns it. */
StructTypeProto::Structure::Field *AddConstantField(StructTypeProto::Structure &structure,
                                                    const char *name, const TensorProto &value) {
  StructTypeProto::Structure::Field *field = structure.add_field();
  field->set_name(name);
  field->set_constant(value);
  return field;
}

/** Builds a scalar FLOAT tensor constant. */
TensorProto MakeFloatConstant(const std::vector<float> &values, const std::vector<int64_t> &dims) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  for (const int64_t dim : dims) {
    tensor.ref_dims().push_back(dim);
  }
  for (const float value : values) {
    tensor.ref_float_data().push_back(value);
  }
  return tensor;
}

/** Builds a scalar INT64 tensor constant. */
TensorProto MakeInt64Constant(int64_t value) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::INT64);
  tensor.ref_int64_data().push_back(value);
  return tensor;
}

/** Appends a little-endian 32-bit float to a byte payload. */
void AppendFloat(std::string &payload, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; ++i) {
    payload.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
  }
}

/** Appends a little-endian 64-bit signed integer to a byte payload. */
void AppendInt64(std::string &payload, int64_t value) {
  const uint64_t bits = static_cast<uint64_t>(value);
  for (int i = 0; i < 8; ++i) {
    payload.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
  }
}

/** Appends @p count nibble-packed INT4 codes produced by @p code. */
template <typename Fn> void AppendNibbles(std::string &payload, size_t count, Fn code) {
  for (size_t i = 0; i < count; i += 2) {
    const uint8_t low = static_cast<uint8_t>(code(i) & 0x0F);
    const uint8_t high = i + 1 < count ? static_cast<uint8_t>(code(i + 1) & 0x0F) : uint8_t(0);
    payload.push_back(static_cast<char>(low | (high << 4)));
  }
}

/** Wraps @p payload as a length-delimited field on the wire. */
std::string LengthDelimited(int field_number, const std::string &payload) {
  return FieldTag(field_number, 2) + EncodeVarint(payload.size()) + payload;
}

/** Encodes a varint field on the wire. */
std::string VarintField(int field_number, uint64_t value) {
  return FieldTag(field_number, 0) + EncodeVarint(value);
}

/** Serializes @p proto and returns its bytes. */
template <typename T> std::string Bytes(const T &proto) {
  std::string out;
  EXPECT_TRUE(proto.SerializeToString(out));
  return out;
}

/** Runs @p action and returns the message of the std::runtime_error it raises, or "". */
template <typename Fn> std::string RuntimeErrorMessage(Fn action) {
  try {
    action();
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return std::string();
}

/** Runs @p action and returns the message of the std::invalid_argument it raises, or "". */
template <typename Fn> std::string InvalidArgumentMessage(Fn action) {
  try {
    action();
  } catch (const std::invalid_argument &error) {
    return error.what();
  }
  return std::string();
}

/** Builds a minimal but structurally valid model with an Identity graph. */
ModelProto MakeModel() {
  ModelProto model;
  model.add_opset("", 18);
  GraphProto &graph = *model.add_graph();
  graph.set_name("g");
  ValueInfoProto *input = graph.add_input();
  input->set_name("x");
  input->ref_type() = MakeTensorType(TensorProto::FLOAT, {1});
  ValueInfoProto *output = graph.add_output();
  output->set_name("y");
  output->ref_type() = MakeTensorType(TensorProto::FLOAT, {1});
  NodeProto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  return model;
}

/** Declares "Int4Block": 32 INT4 codes followed by one FLOAT scale (20 bytes). */
StructTypeProto *DeclareInt4Block(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kInt4Block);
  declaration->set_name("Int4Block");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  AddField(structure, "codes", MakeArrayType(TensorProto::INT4, 32));
  AddField(structure, "scale", MakeTensorType(TensorProto::FLOAT, {}));
  return declaration;
}

/** Declares "LINEAR_INT4_128_FIXED_PARAMETERS": 128 INT4 codes and two constants. */
StructTypeProto *DeclareLinearInt4Fixed(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kLinearInt4Fixed);
  declaration->set_name("LINEAR_INT4_128_FIXED_PARAMETERS");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  AddField(structure, "values", MakeArrayType(TensorProto::INT4, 128));
  AddConstantField(structure, "scale", MakeFloatConstant({0.25f}, {}));
  AddConstantField(structure, "zero_point", MakeInt64Constant(-2));
  return declaration;
}

/** Declares "LINEAR_INT4_128_WITH_PARAMETERS": 128 INT4 codes, a scale and a zero point. */
StructTypeProto *DeclareLinearInt4Parameters(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kLinearInt4Parameters);
  declaration->set_name("LINEAR_INT4_128_WITH_PARAMETERS");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  AddField(structure, "values", MakeArrayType(TensorProto::INT4, 128));
  AddField(structure, "scale", MakeTensorType(TensorProto::FLOAT, {}));
  AddField(structure, "zero_point", MakeTensorType(TensorProto::INT64, {}));
  return declaration;
}

/** Declares "CODEBOOK2_BLOCK_32": 32 two-bit indices plus a constant four-entry codebook. */
StructTypeProto *DeclareCodebookBlock(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kCodebookBlock);
  declaration->set_name("CODEBOOK2_BLOCK_32");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  TypeProto codes;
  StructTypeProto::BitPacking &packing = codes.ref_struct_type().ref_bit_packing();
  StructTypeProto::BitPacking::Component *index = packing.add_component();
  index->set_name("index");
  index->set_bit_width(2);
  packing.set_dimension(32);
  AddField(structure, "codes", codes);
  AddConstantField(structure, "codebook", MakeFloatConstant({-1.0f, -0.25f, 0.25f, 1.0f}, {4}));
  return declaration;
}

/** Declares "SCALED_CODEBOOK2_BLOCK_32": the codebook block plus a per-block FLOAT scale. */
StructTypeProto *DeclareScaledCodebookBlock(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kScaledCodebookBlock);
  declaration->set_name("SCALED_CODEBOOK2_BLOCK_32");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  AddField(structure, "quantized", MakeTypeRef(kCodebookBlock));
  AddField(structure, "scale", MakeTensorType(TensorProto::FLOAT, {}));
  return declaration;
}

/** Declares an INT8 key/value block: 16 codes plus a FLOAT scale (20 bytes). */
StructTypeProto *DeclareInt8Block(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kInt8Block);
  declaration->set_name("Int8Block16");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  AddField(structure, "codes", MakeArrayType(TensorProto::INT8, 16));
  AddField(structure, "scale", MakeTensorType(TensorProto::FLOAT, {}));
  return declaration;
}

/** Declares a dynamic KV cache: a sequence of blocks, a dynamic key tensor and a length. */
StructTypeProto *DeclareKeyValueCache(ModelProto &model) {
  StructTypeProto *declaration = model.add_struct_types();
  declaration->set_type_id(kKeyValueCache);
  declaration->set_name("KVCache");
  StructTypeProto::Structure &structure = declaration->ref_structure();
  TypeProto blocks;
  blocks.ref_sequence_type().set_elem_type(MakeTypeRef(kInt4Block));
  AddField(structure, "blocks", blocks);
  AddField(structure, "keys", MakeDynamicTensorType(TensorProto::FLOAT, "past_sequence"));
  AddField(structure, "length", MakeTensorType(TensorProto::INT64, {}));
  return declaration;
}

/** Builds one Int4Block payload of @p records records with increasing codes. */
std::string MakeInt4BlockPayload(size_t records) {
  std::string payload;
  for (size_t record = 0; record < records; ++record) {
    AppendNibbles(payload, 32, [record](size_t i) { return (record + i) & 0x0F; });
    AppendFloat(payload, 0.5f * static_cast<float>(record + 1));
  }
  return payload;
}

/** Builds one SCALED_CODEBOOK2_BLOCK_32 payload with indices 0,1,2,3 repeated and a scale. */
std::string MakeScaledCodebookPayload(size_t records, float scale) {
  std::string payload;
  for (size_t record = 0; record < records; ++record) {
    for (int i = 0; i < 8; ++i) {
      payload.push_back(static_cast<char>(0xE4));
    }
    AppendFloat(payload, scale);
  }
  return payload;
}

/** Appends an external_data entry to an encoded value. */
void AddExternalEntry(EncodedValueProto &value, const char *key, const std::string &content) {
  StringStringEntryProto *entry = value.add_external_data();
  entry->set_key(key);
  entry->set_value(content);
}

/** Adds an encoded initializer referencing a catalogue identity. */
EncodedValueProto *AddEncodedInitializer(GraphProto &graph, const char *name, uint64_t type_id,
                                         const std::string &payload) {
  EncodedValueProto *value = graph.add_encoded_initializer();
  value->set_name(name);
  value->ref_struct_type().set_type_ref(type_id);
  value->set_raw_data(payload);
  return value;
}

} // namespace

// ---------------------------------------------------------------------------
// Wire format: round-trip, field 1000 and text printing.
// ---------------------------------------------------------------------------

TEST(custom_values, WireRoundTripModel) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  DeclareCodebookBlock(model);
  StructTypeProto *scaled = DeclareScaledCodebookBlock(model);
  scaled->ref_decoder().set_name("DecodeScaledCodebookBlocks");
  scaled->set_doc_string("composition by reference");
  StringStringEntryProto *metadata = scaled->add_metadata_props();
  metadata->set_key("vendor");
  metadata->set_value("onnx-light");

  GraphProto &graph = model.ref_graph();
  EncodedValueProto *blocks =
      AddEncodedInitializer(graph, "blocks", kInt4Block, MakeInt4BlockPayload(3));
  blocks->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {96});
  blocks->set_doc_string("three blocks");
  AddEncodedInitializer(graph, "codebook_blocks", kScaledCodebookBlock,
                        MakeScaledCodebookPayload(2, 2.0f));

  std::string serialized;
  ASSERT_TRUE(model.SerializeToString(serialized));

  ModelProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  ASSERT_EQ(parsed.ref_struct_types().size(), 3u);
  EXPECT_EQ(parsed.ref_struct_types()[0].ref_type_id(), kInt4Block);
  EXPECT_EQ(parsed.ref_struct_types()[0].name(), "Int4Block");
  EXPECT_EQ(parsed.ref_struct_types()[0].kind_case(), StructTypeProto::kStructure);
  EXPECT_EQ(parsed.ref_struct_types()[1].ref_type_id(), kCodebookBlock);
  EXPECT_EQ(parsed.ref_struct_types()[1]
                .ref_structure()
                .ref_field()[0]
                .ref_type()
                .ref_struct_type()
                .ref_bit_packing()
                .ref_dimension(),
            32u);
  ASSERT_EQ(parsed.ref_struct_types()[2].ref_type_id(), kScaledCodebookBlock);
  EXPECT_TRUE(parsed.ref_struct_types()[2].has_decoder());
  EXPECT_EQ(parsed.ref_struct_types()[2].ref_decoder().name(), "DecodeScaledCodebookBlocks");
  ASSERT_EQ(parsed.ref_struct_types()[2].ref_metadata_props().size(), 1u);
  EXPECT_EQ(parsed.ref_struct_types()[2].ref_metadata_props()[0].value(), "onnx-light");

  const GraphProto &parsed_graph = parsed.ref_graph();
  ASSERT_EQ(parsed_graph.ref_encoded_initializer().size(), 2u);
  const EncodedValueProto &parsed_blocks = parsed_graph.ref_encoded_initializer()[0];
  EXPECT_EQ(parsed_blocks.name(), "blocks");
  EXPECT_EQ(parsed_blocks.layout_case(), EncodedValueProto::kStructType);
  EXPECT_EQ(parsed_blocks.ref_struct_type().ref_type_ref(), kInt4Block);
  EXPECT_EQ(parsed_blocks.ref_raw_data().size(), 60u);
  EXPECT_EQ(parsed_blocks.ref_logical_type().value_case(), TypeProto::kTensorType);
  EXPECT_EQ(parsed_blocks.ref_raw_data(), blocks->ref_raw_data());
  EXPECT_NO_THROW(VerifyModel(parsed));
}

TEST(custom_values, WireRoundTripStructTypeAlone) {
  StructTypeProto declaration;
  declaration.set_type_id(kInt4Block);
  declaration.set_name("Int4Block");
  StructTypeProto::Structure &structure = declaration.ref_structure();
  AddField(structure, "codes", MakeArrayType(TensorProto::INT4, 32));
  AddConstantField(structure, "scale", MakeFloatConstant({0.25f}, {}));

  std::string serialized;
  ASSERT_TRUE(declaration.SerializeToString(serialized));
  StructTypeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  ASSERT_EQ(parsed.ref_structure().ref_field().size(), 2u);
  EXPECT_EQ(parsed.ref_structure().ref_field()[0].content_case(),
            StructTypeProto::Structure::Field::kType);
  EXPECT_EQ(parsed.ref_structure()
                .ref_field()[0]
                .ref_type()
                .ref_struct_type()
                .ref_array()
                .ref_dimension(),
            32u);
  EXPECT_EQ(parsed.ref_structure().ref_field()[1].content_case(),
            StructTypeProto::Structure::Field::kConstant);
  EXPECT_EQ(parsed.ref_structure().ref_field()[1].ref_constant().ref_float_data()[0], 0.25f);

  StructTypeProto copied;
  copied.CopyFrom(parsed);
  EXPECT_EQ(copied.ref_type_id(), kInt4Block);
  EXPECT_EQ(copied.ref_structure().ref_field().size(), 2u);
}

TEST(custom_values, ExtensionFieldNumbersAre1000) {
  const std::string tag = FieldTag(1000, 2);

  TypeProto type = MakeTypeRef(kInt4Block);
  std::string type_bytes;
  ASSERT_TRUE(type.SerializeToString(type_bytes));
  EXPECT_NE(type_bytes.find(tag), std::string::npos);
  TypeProto parsed_type;
  ASSERT_TRUE(parsed_type.ParseFromString(type_bytes));
  EXPECT_EQ(parsed_type.value_case(), TypeProto::kStructType);
  EXPECT_EQ(parsed_type.ref_struct_type().ref_type_ref(), kInt4Block);

  GraphProto graph;
  graph.set_name("g");
  AddEncodedInitializer(graph, "w", kInt4Block, MakeInt4BlockPayload(1));
  std::string graph_bytes;
  ASSERT_TRUE(graph.SerializeToString(graph_bytes));
  EXPECT_NE(graph_bytes.find(tag), std::string::npos);

  ModelProto model;
  DeclareInt4Block(model);
  std::string model_bytes;
  ASSERT_TRUE(model.SerializeToString(model_bytes));
  EXPECT_NE(model_bytes.find(tag), std::string::npos);
}

TEST(custom_values, PrintsStructuredFields) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  AddEncodedInitializer(model.ref_graph(), "w", kInt4Block, MakeInt4BlockPayload(1));

  utils::PrintOptions options;
  std::stringstream stream;
  model.PrintToStringStream(stream, options);
  const std::string text = stream.str();
  EXPECT_NE(text.find("struct_types"), std::string::npos);
  EXPECT_NE(text.find("Int4Block"), std::string::npos);
  EXPECT_NE(text.find("encoded_initializer"), std::string::npos);
  EXPECT_NE(text.find("type_ref"), std::string::npos);

  std::stringstream packing_stream;
  StructTypeProto::BitPacking::Component component;
  component.set_name("index");
  component.set_bit_width(2);
  component.PrintToStringStream(packing_stream, options);
  EXPECT_NE(packing_stream.str().find("bit_width"), std::string::npos);
  EXPECT_NE(packing_stream.str().find("2"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Catalogue resolution and size arithmetic.
// ---------------------------------------------------------------------------

TEST(custom_values, CatalogueResolvesUniqueIdentities) {
  ModelProto model = MakeModel();
  DeclareCodebookBlock(model);
  DeclareScaledCodebookBlock(model);
  DeclareInt4Block(model);

  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(model));
  EXPECT_EQ(catalogue.size(), 3u);
  ASSERT_NE(catalogue.Find(kCodebookBlock), nullptr);
  EXPECT_EQ(catalogue.Find(kCodebookBlock)->name(), "CODEBOOK2_BLOCK_32");
  EXPECT_EQ(catalogue.Find(4242), nullptr);

  // Reordering the catalogue does not change identity.
  ModelProto reordered = MakeModel();
  DeclareInt4Block(reordered);
  DeclareScaledCodebookBlock(reordered);
  DeclareCodebookBlock(reordered);
  StructTypeCatalogue other;
  ASSERT_NO_THROW(other.Build(reordered));
  uint64_t bits = 0;
  uint64_t other_bits = 0;
  ASSERT_TRUE(catalogue.FixedBitSize(*catalogue.Find(kScaledCodebookBlock), bits));
  ASSERT_TRUE(other.FixedBitSize(*other.Find(kScaledCodebookBlock), other_bits));
  EXPECT_EQ(bits, other_bits);
  EXPECT_EQ(bits, 96u);
}

TEST(custom_values, FixedSizesFollowTheDocumentedRules) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  DeclareLinearInt4Fixed(model);
  DeclareLinearInt4Parameters(model);
  DeclareCodebookBlock(model);
  DeclareScaledCodebookBlock(model);
  DeclareInt8Block(model);
  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(model));

  const struct {
    uint64_t type_id;
    uint64_t bytes;
  } expected[] = {
      {kInt4Block, 20},    {kLinearInt4Fixed, 64},     {kLinearInt4Parameters, 76},
      {kCodebookBlock, 8}, {kScaledCodebookBlock, 12}, {kInt8Block, 20},
  };
  for (const auto &entry : expected) {
    uint64_t bytes = 0;
    std::string reason;
    EXPECT_TRUE(catalogue.IsByteEncodable(*catalogue.Find(entry.type_id), bytes, &reason))
        << entry.type_id << ": " << reason;
    EXPECT_EQ(bytes, entry.bytes) << entry.type_id;
  }
}

TEST(custom_values, DynamicStructIsAValidTypeButNotAnEncodedLayout) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  DeclareKeyValueCache(model);

  // The cache type is a legal struct declaration and a legal value type.
  ValueInfoProto *cache_input = model.ref_graph().add_input();
  cache_input->set_name("past");
  cache_input->ref_type() = MakeTypeRef(kKeyValueCache);
  NodeProto *node = model.ref_graph().add_node();
  node->set_op_type("Identity");
  node->add_input("past");
  node->add_output("present");
  ValueInfoProto *cache_output = model.ref_graph().add_output();
  cache_output->set_name("present");
  cache_output->ref_type() = MakeTypeRef(kKeyValueCache);
  EXPECT_NO_THROW(VerifyModel(model));

  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(model));
  uint64_t bits = 0;
  std::string reason;
  EXPECT_FALSE(catalogue.FixedBitSize(*catalogue.Find(kKeyValueCache), bits, &reason));
  EXPECT_NE(reason.find("sequence"), std::string::npos) << reason;

  // The same type cannot select a flat byte-encoded layout.
  AddEncodedInitializer(model.ref_graph(), "cache_bytes", kKeyValueCache, std::string(20, '\0'));
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(custom_values, RejectsNonByteRootsOverflowAndDynamicFields) {
  ModelProto model = MakeModel();
  StructTypeProto *odd = model.add_struct_types();
  odd->set_type_id(7001);
  odd->set_name("FiveThreeBitCodes");
  StructTypeProto::BitPacking &packing = odd->ref_bit_packing();
  StructTypeProto::BitPacking::Component *component = packing.add_component();
  component->set_name("code");
  component->set_bit_width(3);
  packing.set_dimension(5);

  StructTypeProto *overflow = model.add_struct_types();
  overflow->set_type_id(7002);
  overflow->set_name("Overflowing");
  StructTypeProto::Array &array = overflow->ref_array();
  array.set_element_type(MakeTensorType(TensorProto::INT64, {}));
  array.set_dimension(~uint64_t(0));

  StructTypeProto *dynamic = model.add_struct_types();
  dynamic->set_type_id(7003);
  dynamic->set_name("DynamicRow");
  AddField(dynamic->ref_structure(), "row", MakeDynamicTensorType(TensorProto::FLOAT, "sequence"));

  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(model));

  uint64_t bits = 0;
  std::string reason;
  ASSERT_TRUE(catalogue.FixedBitSize(*catalogue.Find(7001), bits));
  EXPECT_EQ(bits, 15u);
  uint64_t bytes = 0;
  EXPECT_FALSE(catalogue.IsByteEncodable(*catalogue.Find(7001), bytes, &reason));
  EXPECT_NE(reason.find("divisible by eight"), std::string::npos) << reason;

  EXPECT_FALSE(catalogue.FixedBitSize(*catalogue.Find(7002), bits, &reason));
  EXPECT_NE(reason.find("overflow"), std::string::npos) << reason;

  EXPECT_FALSE(catalogue.FixedBitSize(*catalogue.Find(7003), bits, &reason));
  EXPECT_NE(reason.find("concrete dimensions"), std::string::npos) << reason;

  for (const uint64_t type_id : {uint64_t(7001), uint64_t(7002), uint64_t(7003)}) {
    ModelProto rejected = model;
    AddEncodedInitializer(rejected.ref_graph(), "bad", type_id, std::string(8, '\0'));
    EXPECT_THROW(VerifyModel(rejected), std::invalid_argument) << type_id;
  }
}

// ---------------------------------------------------------------------------
// Malformed declarations.
// ---------------------------------------------------------------------------

TEST(custom_values, RejectsMalformedIdentities) {
  ModelProto missing = MakeModel();
  DeclareInt4Block(missing)->clear_type_id();
  EXPECT_THROW(VerifyModel(missing), std::invalid_argument);

  ModelProto zero = MakeModel();
  DeclareInt4Block(zero)->set_type_id(uint64_t(0));
  EXPECT_THROW(VerifyModel(zero), std::invalid_argument);

  ModelProto duplicate = MakeModel();
  DeclareInt4Block(duplicate);
  DeclareInt8Block(duplicate)->set_type_id(kInt4Block);
  EXPECT_THROW(VerifyModel(duplicate), std::invalid_argument);

  ModelProto bare_reference = MakeModel();
  DeclareInt4Block(bare_reference);
  StructTypeProto *alias = bare_reference.add_struct_types();
  alias->set_type_id(kInt8Block);
  alias->set_type_ref(kInt4Block);
  EXPECT_THROW(VerifyModel(bare_reference), std::invalid_argument);

  ModelProto inline_identity = MakeModel();
  EncodedValueProto *value =
      AddEncodedInitializer(inline_identity.ref_graph(), "w", kInt4Block, std::string());
  value->ref_struct_type().Clear();
  StructTypeProto::Structure &structure = value->ref_struct_type().ref_structure();
  AddField(structure, "codes", MakeArrayType(TensorProto::INT8, 4));
  value->ref_struct_type().set_type_id(uint64_t(99));
  EXPECT_THROW(VerifyModel(inline_identity), std::invalid_argument);
}

TEST(custom_values, RejectsUnresolvedReferencesAndCycles) {
  ModelProto unresolved = MakeModel();
  DeclareInt4Block(unresolved);
  AddEncodedInitializer(unresolved.ref_graph(), "w", 999999, std::string(20, '\0'));
  EXPECT_THROW(VerifyModel(unresolved), std::invalid_argument);

  ModelProto self_cycle = MakeModel();
  StructTypeProto *recursive = self_cycle.add_struct_types();
  recursive->set_type_id(uint64_t(5001));
  AddField(recursive->ref_structure(), "self", MakeTypeRef(5001));
  EXPECT_THROW(VerifyModel(self_cycle), std::invalid_argument);

  ModelProto mutual_cycle = MakeModel();
  StructTypeProto *first = mutual_cycle.add_struct_types();
  first->set_type_id(uint64_t(5002));
  AddField(first->ref_structure(), "next", MakeTypeRef(5003));
  StructTypeProto *second = mutual_cycle.add_struct_types();
  second->set_type_id(uint64_t(5003));
  AddField(second->ref_structure(), "back", MakeTypeRef(5002));
  EXPECT_THROW(VerifyModel(mutual_cycle), std::invalid_argument);

  ModelProto decorated_reference = MakeModel();
  DeclareInt4Block(decorated_reference);
  EncodedValueProto *value = decorated_reference.ref_graph().add_encoded_initializer();
  value->set_name("w");
  value->ref_struct_type().set_type_ref(kInt4Block);
  value->ref_struct_type().set_name("not allowed on a reference");
  value->set_raw_data(MakeInt4BlockPayload(1));
  EXPECT_THROW(VerifyModel(decorated_reference), std::invalid_argument);
}

TEST(custom_values, RejectsMalformedDeclarationsAndDuplicateNames) {
  ModelProto empty_structure = MakeModel();
  StructTypeProto *empty = empty_structure.add_struct_types();
  empty->set_type_id(uint64_t(6001));
  empty->add_structure();
  EXPECT_THROW(VerifyModel(empty_structure), std::invalid_argument);

  ModelProto duplicate_fields = MakeModel();
  StructTypeProto *duplicated = duplicate_fields.add_struct_types();
  duplicated->set_type_id(uint64_t(6002));
  AddField(duplicated->ref_structure(), "codes", MakeArrayType(TensorProto::INT8, 2));
  AddField(duplicated->ref_structure(), "codes", MakeArrayType(TensorProto::INT8, 2));
  EXPECT_THROW(VerifyModel(duplicate_fields), std::invalid_argument);

  ModelProto zero_width = MakeModel();
  StructTypeProto *packing = zero_width.add_struct_types();
  packing->set_type_id(uint64_t(6003));
  StructTypeProto::BitPacking::Component *component = packing->ref_bit_packing().add_component();
  component->set_name("index");
  packing->ref_bit_packing().set_dimension(8);
  EXPECT_THROW(VerifyModel(zero_width), std::invalid_argument);

  ModelProto unconstrained = MakeModel();
  unconstrained.add_struct_types()->set_type_id(uint64_t(6004));
  EXPECT_THROW(VerifyModel(unconstrained), std::invalid_argument);

  ModelProto duplicate_names = MakeModel();
  DeclareInt4Block(duplicate_names);
  GraphProto &graph = duplicate_names.ref_graph();
  TensorProto *dense = graph.add_initializer();
  dense->set_name("shared");
  dense->set_data_type(TensorProto::FLOAT);
  dense->ref_dims().push_back(1);
  dense->ref_float_data().push_back(1.0f);
  AddEncodedInitializer(graph, "shared", kInt4Block, MakeInt4BlockPayload(1));
  EXPECT_THROW(VerifyModel(duplicate_names), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Payload geometry: one element type, many payload lengths.
// ---------------------------------------------------------------------------

TEST(custom_values, OneTypeManyPayloadLengths) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  GraphProto &graph = model.ref_graph();
  EncodedValueProto *small =
      AddEncodedInitializer(graph, "small", kInt4Block, MakeInt4BlockPayload(128));
  small->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {4096});
  EncodedValueProto *large =
      AddEncodedInitializer(graph, "large", kInt4Block, MakeInt4BlockPayload(256));
  large->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {8192});
  EncodedValueProto *empty = AddEncodedInitializer(graph, "empty", kInt4Block, std::string());
  ASSERT_NE(empty, nullptr);

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  EXPECT_EQ(small->ref_raw_data().size(), 2560u);
  EXPECT_EQ(large->ref_raw_data().size(), 5120u);
  EXPECT_EQ(catalogue.ValidateEncodedValue(*small).record_count, 128u);
  EXPECT_EQ(catalogue.ValidateEncodedValue(*large).record_count, 256u);
  EXPECT_EQ(catalogue.ValidateEncodedValue(*empty).record_count, 0u);
  EXPECT_EQ(catalogue.ValidateEncodedValue(*small).element_bits, 160u);

  // A partial record is rejected.
  EncodedValueProto partial;
  partial.set_name("partial");
  partial.ref_struct_type().set_type_ref(kInt4Block);
  partial.set_raw_data(std::string(30, '\0'));
  EXPECT_THROW(catalogue.ValidateEncodedValue(partial), std::invalid_argument);
}

TEST(custom_values, ReadsConstantAndPerValueParameters) {
  ModelProto model = MakeModel();
  DeclareLinearInt4Fixed(model);
  DeclareLinearInt4Parameters(model);
  GraphProto &graph = model.ref_graph();

  std::string fixed_payload;
  AppendNibbles(fixed_payload, 128, [](size_t i) { return i % 16; });
  ASSERT_EQ(fixed_payload.size(), 64u);
  EncodedValueProto *weight_a =
      AddEncodedInitializer(graph, "weight_a", kLinearInt4Fixed, fixed_payload);
  weight_a->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {128});

  std::string parameter_payload = fixed_payload;
  AppendFloat(parameter_payload, 0.25f);
  AppendInt64(parameter_payload, -2);
  ASSERT_EQ(parameter_payload.size(), 76u);
  EncodedValueProto *weight_b =
      AddEncodedInitializer(graph, "weight_b", kLinearInt4Parameters, parameter_payload);
  weight_b->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {128});

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  // The shared constants live in the declaration and contribute no payload bytes.
  const StructTypeProto *fixed = catalogue.Find(kLinearInt4Fixed);
  ASSERT_NE(fixed, nullptr);
  const StructTypeProto::Structure::Field &scale_constant = fixed->ref_structure().ref_field()[1];
  EXPECT_EQ(scale_constant.content_case(), StructTypeProto::Structure::Field::kConstant);
  EXPECT_EQ(scale_constant.ref_constant().ref_float_data()[0], 0.25f);
  EXPECT_EQ(fixed->ref_structure().ref_field()[2].ref_constant().ref_int64_data()[0], -2);

  EncodedValueView fixed_view(catalogue, *weight_a);
  EXPECT_EQ(fixed_view.record_count(), 1u);
  EXPECT_EQ(fixed_view.record_bits(), 512u);
  EncodedFieldRef values;
  ASSERT_TRUE(FindEncodedField(catalogue, *fixed, "values", values));
  EXPECT_EQ(values.count, 128u);
  EXPECT_EQ(values.bit_width, 4u);
  EXPECT_EQ(values.bit_offset, 0u);
  EXPECT_EQ(fixed_view.ReadElement(values, 0, 0), 0u);
  EXPECT_EQ(fixed_view.ReadElement(values, 0, 7), 7u);
  EXPECT_EQ(fixed_view.ReadSignedElement(values, 0, 9), -7);
  // (code - (-2)) * 0.25 for code 3.
  EXPECT_FLOAT_EQ((static_cast<float>(fixed_view.ReadSignedElement(values, 0, 3)) + 2.0f) * 0.25f,
                  1.25f);

  // Per-value parameters share the layout and the identity.
  const StructTypeProto *parameters = catalogue.Find(kLinearInt4Parameters);
  ASSERT_NE(parameters, nullptr);
  EncodedValueView parameter_view(catalogue, *weight_b);
  EXPECT_EQ(parameter_view.record_bits(), 608u);
  EncodedFieldRef scale;
  EncodedFieldRef zero_point;
  ASSERT_TRUE(FindEncodedField(catalogue, *parameters, "scale", scale));
  ASSERT_TRUE(FindEncodedField(catalogue, *parameters, "zero_point", zero_point));
  EXPECT_EQ(scale.bit_offset, 512u);
  EXPECT_EQ(zero_point.bit_offset, 544u);
  EXPECT_EQ(zero_point.bit_offset % 8, 0u);
  EXPECT_EQ(zero_point.bit_offset / 8, 68u);
  EXPECT_FLOAT_EQ(parameter_view.ReadFloatElement(scale, 0, 0), 0.25f);
  EXPECT_EQ(parameter_view.ReadSignedElement(zero_point, 0, 0), -2);

  // A constant field is not physical: it is read from the declaration instead.
  EncodedFieldRef unused;
  EXPECT_FALSE(FindEncodedField(catalogue, *fixed, "scale", unused));
  EXPECT_FALSE(FindEncodedField(catalogue, *parameters, "missing", unused));
}

TEST(custom_values, DecodesCodebookComposition) {
  ModelProto model = MakeModel();
  DeclareCodebookBlock(model);
  DeclareScaledCodebookBlock(model);
  GraphProto &graph = model.ref_graph();
  EncodedValueProto *one =
      AddEncodedInitializer(graph, "one", kScaledCodebookBlock, MakeScaledCodebookPayload(1, 2.0f));
  one->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {32});
  EncodedValueProto *many = AddEncodedInitializer(graph, "many", kScaledCodebookBlock,
                                                  MakeScaledCodebookPayload(128, 2.0f));
  many->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {4096});

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  EXPECT_EQ(one->ref_raw_data().size(), 12u);
  EXPECT_EQ(many->ref_raw_data().size(), 1536u);
  EncodedValueView one_view(catalogue, *one);
  EncodedValueView many_view(catalogue, *many);
  EXPECT_EQ(one_view.record_count(), 1u);
  EXPECT_EQ(many_view.record_count(), 128u);

  const StructTypeProto *root = catalogue.Find(kScaledCodebookBlock);
  ASSERT_NE(root, nullptr);
  EncodedFieldRef indices;
  EncodedFieldRef scale;
  ASSERT_TRUE(FindEncodedField(catalogue, *root, "quantized.codes.index", indices));
  ASSERT_TRUE(FindEncodedField(catalogue, *root, "scale", scale));
  EXPECT_EQ(indices.bit_offset, 0u);
  EXPECT_EQ(indices.bit_stride, 2u);
  EXPECT_EQ(indices.bit_width, 2u);
  EXPECT_EQ(indices.count, 32u);
  EXPECT_EQ(indices.elem_type, TensorProto::UNDEFINED);
  EXPECT_EQ(scale.bit_offset, 64u);

  // The codebook lives once in the subtype and contributes no payload bytes.
  const StructTypeProto *subtype = catalogue.Find(kCodebookBlock);
  ASSERT_NE(subtype, nullptr);
  const TensorProto &table = subtype->ref_structure().ref_field()[1].ref_constant();
  ASSERT_EQ(table.ref_float_data().size(), 4u);

  const float block_scale = one_view.ReadFloatElement(scale, 0, 0);
  EXPECT_FLOAT_EQ(block_scale, 2.0f);
  const float expected[4] = {-2.0f, -0.5f, 0.5f, 2.0f};
  for (uint64_t i = 0; i < 32; ++i) {
    const uint64_t index = one_view.ReadElement(indices, 0, i);
    ASSERT_LT(index, 4u);
    EXPECT_EQ(index, i % 4);
    EXPECT_FLOAT_EQ(block_scale * table.ref_float_data()[index], expected[i % 4]);
  }
  EXPECT_EQ(many_view.ReadElement(indices, 127, 31), 3u);
}

TEST(custom_values, HeterogeneousKeyValueDeclarations) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  DeclareInt8Block(model);
  DeclareCodebookBlock(model);
  DeclareScaledCodebookBlock(model);
  DeclareKeyValueCache(model);
  GraphProto &graph = model.ref_graph();

  AddEncodedInitializer(graph, "past_key_int4", kInt4Block, MakeInt4BlockPayload(4));
  std::string int8_payload;
  for (int record = 0; record < 4; ++record) {
    for (int i = 0; i < 16; ++i) {
      int8_payload.push_back(static_cast<char>(record * 16 + i));
    }
    AppendFloat(int8_payload, 0.125f);
  }
  AddEncodedInitializer(graph, "past_value_int8", kInt8Block, int8_payload);
  AddEncodedInitializer(graph, "past_value_codebook", kScaledCodebookBlock,
                        MakeScaledCodebookPayload(4, 0.5f));

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  const GraphProto &verified = model.ref_graph();
  ASSERT_EQ(verified.ref_encoded_initializer().size(), 3u);
  const uint64_t expected_records[] = {4, 4, 4};
  const uint64_t expected_bits[] = {160, 160, 96};
  for (size_t i = 0; i < 3; ++i) {
    const EncodedValueLayout layout =
        catalogue.ValidateEncodedValue(verified.ref_encoded_initializer()[i]);
    EXPECT_EQ(layout.record_count, expected_records[i]) << i;
    EXPECT_EQ(layout.element_bits, expected_bits[i]) << i;
  }

  // The cache declaration itself stays a valid dynamic type.
  uint64_t bits = 0;
  EXPECT_FALSE(catalogue.FixedBitSize(*catalogue.Find(kKeyValueCache), bits));
  EXPECT_NO_THROW(catalogue.ValidateType(MakeTypeRef(kKeyValueCache)));

  EncodedValueView int8_view(catalogue, verified.ref_encoded_initializer()[1]);
  EncodedFieldRef codes;
  ASSERT_TRUE(FindEncodedField(catalogue, *catalogue.Find(kInt8Block), "codes", codes));
  EXPECT_EQ(codes.count, 16u);
  EXPECT_EQ(codes.bit_width, 8u);
  EXPECT_EQ(int8_view.ReadSignedElement(codes, 2, 3), 35);
}

// ---------------------------------------------------------------------------
// The built-in affine layout.
// ---------------------------------------------------------------------------

TEST(custom_values, AffineInt8PerTensor) {
  ModelProto model = MakeModel();
  EncodedValueProto *value = model.ref_graph().add_encoded_initializer();
  value->set_name("affine_int8");
  AffineLayoutProto &affine = value->ref_affine();
  affine.set_storage_type(TensorProto::INT8);
  affine.set_scale(MakeFloatConstant({0.125f}, {}));
  TensorProto zero_point;
  zero_point.set_data_type(TensorProto::INT8);
  zero_point.ref_int32_data().push_back(-3);
  affine.set_zero_point(zero_point);
  value->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {2, 2});
  value->set_raw_data(std::string("\x01\x02\x03\x04", 4));

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  const EncodedValueLayout layout = catalogue.ValidateEncodedValue(*value);
  EXPECT_EQ(layout.record_count, 4u);
  EXPECT_EQ(layout.element_bits, 8u);
  EXPECT_EQ(layout.payload_bytes, 4u);
  ASSERT_NE(layout.affine, nullptr);
  EXPECT_EQ(layout.root, nullptr);

  // A payload that does not match the logical element count is rejected.
  value->set_raw_data(std::string("\x01\x02\x03", 3));
  EXPECT_THROW(catalogue.ValidateEncodedValue(*value), std::invalid_argument);

  // A non-floating scale is rejected.
  value->set_raw_data(std::string("\x01\x02\x03\x04", 4));
  affine.set_scale(MakeInt64Constant(1));
  EXPECT_THROW(catalogue.ValidateEncodedValue(*value), std::invalid_argument);
}

TEST(custom_values, AffineInt4OddElementCountPadsTheLastNibble) {
  ModelProto model = MakeModel();
  EncodedValueProto *value = model.ref_graph().add_encoded_initializer();
  value->set_name("affine_int4");
  AffineLayoutProto &affine = value->ref_affine();
  affine.set_storage_type(TensorProto::INT4);
  affine.set_scale(MakeFloatConstant({0.5f}, {}));
  value->ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {5});
  value->set_raw_data(std::string("\x21\x43\x05", 3));

  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  const EncodedValueLayout layout = catalogue.ValidateEncodedValue(*value);
  EXPECT_EQ(layout.record_count, 5u);
  EXPECT_EQ(layout.element_bits, 4u);
  EXPECT_EQ(layout.payload_bytes, 3u);
  EXPECT_TRUE(layout.content_verified);
  EXPECT_NO_THROW(VerifyModel(model));

  // A non-zero unused final high nibble is rejected.
  value->set_raw_data(std::string("\x21\x43\x75", 3));
  EXPECT_THROW(catalogue.ValidateEncodedValue(*value), std::invalid_argument);

  // block_size without an axis is rejected.
  value->set_raw_data(std::string("\x21\x43\x05", 3));
  affine.set_block_size(uint64_t(32));
  EXPECT_THROW(catalogue.ValidateEncodedValue(*value), std::invalid_argument);

  // A storage type outside the frozen set is rejected.
  affine.clear_block_size();
  affine.set_storage_type(TensorProto::FLOAT);
  EXPECT_THROW(catalogue.ValidateEncodedValue(*value), std::invalid_argument);
}

TEST(custom_values, AffineParameterShapesFollowQuantizeLinear) {
  ModelProto model = MakeModel();
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  // Per-tensor quantization needs a scalar scale.
  EncodedValueProto per_tensor;
  per_tensor.set_name("per_tensor");
  per_tensor.ref_affine().set_storage_type(TensorProto::INT8);
  per_tensor.ref_affine().set_scale(MakeFloatConstant({0.5f}, {}));
  per_tensor.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {2, 4});
  per_tensor.set_raw_data(std::string(8, '\x01'));
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(per_tensor));
  per_tensor.ref_affine().set_scale(MakeFloatConstant({0.5f}, {1}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_tensor), std::invalid_argument);
  per_tensor.ref_affine().set_scale(MakeFloatConstant({0.5f, 0.25f}, {}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_tensor), std::invalid_argument);
  per_tensor.ref_affine().set_scale(MakeFloatConstant({0.5f, 0.25f}, {2}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_tensor), std::invalid_argument);

  // Per-axis quantization needs a one-dimensional scale of the axis length.
  EncodedValueProto per_axis;
  per_axis.set_name("per_axis");
  per_axis.ref_affine().set_storage_type(TensorProto::INT8);
  per_axis.ref_affine().set_scale(MakeFloatConstant({0.5f, 0.25f}, {2}));
  per_axis.ref_affine().set_axis(0);
  per_axis.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {2, 4});
  per_axis.set_raw_data(std::string(8, '\x01'));
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(per_axis));
  // A negative axis normalizes to the same axis.
  per_axis.ref_affine().set_axis(-2);
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(per_axis));
  // Axis -1 selects the length-4 axis, so the length-2 scale no longer matches.
  per_axis.ref_affine().set_axis(-1);
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_axis), std::invalid_argument);
  per_axis.ref_affine().set_axis(0);
  per_axis.ref_affine().set_scale(MakeFloatConstant({0.5f, 0.25f, 0.125f}, {3}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_axis), std::invalid_argument);
  per_axis.ref_affine().set_scale(MakeFloatConstant({0.5f, 0.25f}, {1, 2}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_axis), std::invalid_argument);
  per_axis.ref_affine().set_axis(2);
  EXPECT_THROW(catalogue.ValidateEncodedValue(per_axis), std::invalid_argument);

  // Blocked quantization keeps the logical rank with ceil(dim / block_size) blocks.
  EncodedValueProto blocked;
  blocked.set_name("blocked");
  blocked.ref_affine().set_storage_type(TensorProto::INT4);
  blocked.ref_affine().set_axis(1);
  blocked.ref_affine().set_block_size(uint64_t(4));
  blocked.ref_affine().set_scale(MakeFloatConstant({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}));
  blocked.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {2, 7});
  blocked.set_raw_data(std::string(7, '\x11'));
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(blocked));
  // A scalar scale no longer satisfies the blocked geometry.
  blocked.ref_affine().set_scale(MakeFloatConstant({1.0f}, {}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(blocked), std::invalid_argument);
  // The block count must be ceil(7 / 4) == 2, not 7.
  blocked.ref_affine().set_scale(MakeFloatConstant(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
      {2, 7}));
  EXPECT_THROW(catalogue.ValidateEncodedValue(blocked), std::invalid_argument);

  // zero_point must have exactly the scale shape.
  blocked.ref_affine().set_scale(MakeFloatConstant({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}));
  TensorProto zero_point;
  zero_point.set_data_type(TensorProto::INT4);
  zero_point.ref_dims().push_back(2);
  zero_point.ref_dims().push_back(2);
  zero_point.ref_int32_data().push_back(0);
  blocked.ref_affine().set_zero_point(zero_point);
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(blocked));
  zero_point.ref_dims().clear();
  zero_point.ref_dims().push_back(4);
  blocked.ref_affine().set_zero_point(zero_point);
  EXPECT_THROW(catalogue.ValidateEncodedValue(blocked), std::invalid_argument);
}

TEST(custom_values, AffineCodeByteCountRejectsCeilOverflow) {
  ModelProto model = MakeModel();
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  // 4 * 4611686018427387903 == UINT64_MAX - 3, so a naive ceil of (bits + 7) / 8
  // wraps to zero and would accept an empty payload for an enormous tensor.
  EncodedValueProto value;
  value.set_name("huge_int4");
  value.ref_affine().set_storage_type(TensorProto::INT4);
  value.ref_affine().set_scale(MakeFloatConstant({1.0f}, {}));
  value.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {4611686018427387903LL});
  value.set_raw_data(std::string());
  EXPECT_THROW(catalogue.ValidateEncodedValue(value), std::invalid_argument);
  EXPECT_NE(InvalidArgumentMessage([&]() {
              catalogue.ValidateEncodedValue(value);
            }).find("2305843009213693952"),
            std::string::npos);

  // A logical element count that overflows the bit product is rejected outright.
  EncodedValueProto overflowing;
  overflowing.set_name("overflowing");
  overflowing.ref_affine().set_storage_type(TensorProto::INT8);
  overflowing.ref_affine().set_scale(MakeFloatConstant({1.0f}, {}));
  overflowing.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {4611686018427387904LL, 8LL});
  overflowing.set_raw_data(std::string());
  EXPECT_THROW(catalogue.ValidateEncodedValue(overflowing), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Payload location, ownership and bounds-checked access.
// ---------------------------------------------------------------------------

TEST(custom_values, ExternalPayloadNeedsLocationAndExplicitLength) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  EncodedValueProto *value = model.ref_graph().add_encoded_initializer();
  value->set_name("external_blocks");
  value->ref_struct_type().set_type_ref(kInt4Block);
  value->set_data_location(TensorProto::EXTERNAL);
  AddExternalEntry(*value, "location", "weights_0.data");
  AddExternalEntry(*value, "offset", "4096");
  AddExternalEntry(*value, "length", "2560");

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  const EncodedValueLayout layout = catalogue.ValidateEncodedValue(*value);
  EXPECT_TRUE(layout.external);
  EXPECT_EQ(layout.payload_bytes, 2560u);
  EXPECT_EQ(layout.record_count, 128u);

  // Reading an external payload is refused until it is loaded.
  EncodedValueView view(catalogue, *value);
  EXPECT_EQ(view.payload_size(), 0u);
  EXPECT_THROW(view.ReadBits(0, 0, 8), std::invalid_argument);

  // A missing explicit length is rejected.
  EncodedValueProto without_length;
  without_length.set_name("no_length");
  without_length.ref_struct_type().set_type_ref(kInt4Block);
  without_length.set_data_location(TensorProto::EXTERNAL);
  AddExternalEntry(without_length, "location", "weights_0.data");
  EXPECT_THROW(catalogue.ValidateEncodedValue(without_length), std::invalid_argument);

  // Inline and external payloads are mutually exclusive.
  EncodedValueProto both;
  both.set_name("both");
  both.ref_struct_type().set_type_ref(kInt4Block);
  both.set_data_location(TensorProto::EXTERNAL);
  AddExternalEntry(both, "location", "weights_0.data");
  AddExternalEntry(both, "length", "20");
  both.set_raw_data(MakeInt4BlockPayload(1));
  EXPECT_THROW(catalogue.ValidateEncodedValue(both), std::invalid_argument);

  // An external length that is not a whole number of records is rejected.
  EncodedValueProto partial;
  partial.set_name("partial");
  partial.ref_struct_type().set_type_ref(kInt4Block);
  partial.set_data_location(TensorProto::EXTERNAL);
  AddExternalEntry(partial, "location", "weights_0.data");
  AddExternalEntry(partial, "length", "25");
  EXPECT_THROW(catalogue.ValidateEncodedValue(partial), std::invalid_argument);
}

TEST(custom_values, BorrowedPayloadKeepsItsOwnerAlive) {
  const std::string payload = MakeInt4BlockPayload(2);
  auto buffer = std::make_shared<std::vector<uint8_t>>(payload.begin(), payload.end());
  bool released = false;

  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  std::unique_ptr<EncodedValueProto> copy;
  {
    EncodedValueProto value;
    value.set_name("borrowed");
    value.ref_struct_type().set_type_ref(kInt4Block);
    value.set_raw_data_with_deleter(buffer->data(), buffer->size(),
                                    [buffer, &released]() { released = true; });
    const EncodedValueProto &borrowed = value;
    EXPECT_TRUE(borrowed.ref_raw_data().is_borrowed());
    EXPECT_EQ(borrowed.ref_raw_data().data(), buffer->data());

    copy = std::make_unique<EncodedValueProto>(value);
  }
  ASSERT_NE(copy, nullptr);
  EXPECT_FALSE(released);
  const EncodedValueProto &copied = *copy;
  EXPECT_TRUE(copied.ref_raw_data().is_borrowed());
  EXPECT_EQ(copied.ref_raw_data().data(), buffer->data());

  EncodedValueView view(catalogue, *copy);
  EXPECT_EQ(view.record_count(), 2u);
  EncodedFieldRef scale;
  ASSERT_TRUE(FindEncodedField(catalogue, *catalogue.Find(kInt4Block), "scale", scale));
  EXPECT_FLOAT_EQ(view.ReadFloatElement(scale, 1, 0), 1.0f);

  copy.reset();
  EXPECT_TRUE(released);

  // A serialized borrowed payload round-trips into an owned one.
  EncodedValueProto owned;
  owned.set_name("owned");
  owned.ref_struct_type().set_type_ref(kInt4Block);
  owned.set_raw_data(payload);
  std::string serialized;
  ASSERT_TRUE(owned.SerializeToString(serialized));
  EncodedValueProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));
  const EncodedValueProto &read_back = parsed;
  EXPECT_FALSE(read_back.ref_raw_data().is_borrowed());
  EXPECT_EQ(read_back.ref_raw_data().size(), payload.size());
  EXPECT_EQ(catalogue.ValidateEncodedValue(parsed).record_count, 2u);
}

TEST(custom_values, ParseRespectsTheTensorAllocationLimit) {
  EncodedValueProto value;
  value.set_name("big");
  value.ref_struct_type().set_type_ref(kInt4Block);
  value.set_raw_data(std::string(4096, 'x'));
  std::string serialized;
  ASSERT_TRUE(value.SerializeToString(serialized));

  ParseOptions options;
  options.max_tensor_size_bytes = 64;
  EncodedValueProto parsed;
  EXPECT_THROW(parsed.ParseFromString(serialized, options), std::exception);
}

TEST(custom_values, RecordAccessIsBoundsChecked) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  EncodedValueProto *value =
      AddEncodedInitializer(model.ref_graph(), "w", kInt4Block, MakeInt4BlockPayload(2));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  EncodedValueView view(catalogue, *value);

  EXPECT_EQ(view.record_count(), 2u);
  EXPECT_EQ(view.record_bits(), 160u);
  EXPECT_NO_THROW(view.ReadBits(1, 128, 32));

  EXPECT_THROW(view.ReadBits(2, 0, 4), std::invalid_argument);
  EXPECT_THROW(view.ReadBits(0, 157, 4), std::invalid_argument);
  EXPECT_THROW(view.ReadBits(0, 0, 0), std::invalid_argument);
  EXPECT_THROW(view.ReadBits(0, 0, 65), std::invalid_argument);

  EncodedFieldRef codes;
  ASSERT_TRUE(FindEncodedField(catalogue, *catalogue.Find(kInt4Block), "codes", codes));
  EXPECT_NO_THROW(view.ReadElement(codes, 0, 31));
  EXPECT_THROW(view.ReadElement(codes, 0, 32), std::invalid_argument);
  EXPECT_THROW(view.ReadElement(codes, 5, 0), std::invalid_argument);
  EXPECT_THROW(view.ReadFloatElement(codes, 0, 0), std::invalid_argument);
}

TEST(custom_values, InlineLayoutsAndConstantOnlyStructs) {
  ModelProto model = MakeModel();
  GraphProto &graph = model.ref_graph();

  // A concrete inline declaration is a valid encoded layout and owns no identity.
  EncodedValueProto *inline_value = graph.add_encoded_initializer();
  inline_value->set_name("inline_blocks");
  StructTypeProto &declared = inline_value->ref_struct_type();
  AddField(declared.ref_structure(), "codes", MakeArrayType(TensorProto::INT8, 4));
  AddField(declared.ref_structure(), "scale", MakeTensorType(TensorProto::FLOAT, {}));
  std::string payload;
  for (int record = 0; record < 3; ++record) {
    for (int i = 0; i < 4; ++i) {
      payload.push_back(static_cast<char>(i));
    }
    AppendFloat(payload, 1.0f);
  }
  inline_value->set_raw_data(payload);

  // A constant-only group contributes zero bytes to the enclosing record.
  StructTypeProto *constants = model.add_struct_types();
  constants->set_type_id(uint64_t(8001));
  constants->set_name("ConstantsOnly");
  AddConstantField(constants->ref_structure(), "table", MakeFloatConstant({1.0f, 2.0f}, {2}));
  StructTypeProto *wrapper = model.add_struct_types();
  wrapper->set_type_id(uint64_t(8002));
  wrapper->set_name("WrappedConstants");
  AddField(wrapper->ref_structure(), "shared", MakeTypeRef(8001));
  AddField(wrapper->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
  AddEncodedInitializer(graph, "wrapped", 8002, std::string(4, '\x07'));

  ASSERT_NO_THROW(VerifyModel(model));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  const EncodedValueLayout inline_layout = catalogue.ValidateEncodedValue(*inline_value);
  EXPECT_EQ(inline_layout.element_bits, 64u);
  EXPECT_EQ(inline_layout.record_count, 3u);
  EXPECT_EQ(inline_layout.root, &declared);

  uint64_t bits = 0;
  ASSERT_TRUE(catalogue.FixedBitSize(*catalogue.Find(8001), bits));
  EXPECT_EQ(bits, 0u);
  uint64_t bytes = 0;
  std::string reason;
  EXPECT_FALSE(catalogue.IsByteEncodable(*catalogue.Find(8001), bytes, &reason));
  EXPECT_NE(reason.find("strictly positive"), std::string::npos) << reason;
  ASSERT_TRUE(catalogue.IsByteEncodable(*catalogue.Find(8002), bytes));
  EXPECT_EQ(bytes, 1u);
  EXPECT_EQ(catalogue.ValidateEncodedValue(graph.ref_encoded_initializer()[1]).record_count, 4u);

  // A constant-only struct cannot root an encoded payload.
  ModelProto rejected = model;
  AddEncodedInitializer(rejected.ref_graph(), "constants_only", 8001, std::string());
  EXPECT_THROW(VerifyModel(rejected), std::invalid_argument);
}

TEST(custom_values, EncodedInitializersFeedTheGraphScope) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  GraphProto &graph = model.ref_graph();
  AddEncodedInitializer(graph, "packed_weight", kInt4Block, MakeInt4BlockPayload(1));
  NodeProto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("packed_weight");
  node->add_output("z");
  EXPECT_NO_THROW(VerifyModel(model));

  // Without a layout the value is invalid.
  EncodedValueProto *broken = graph.add_encoded_initializer();
  broken->set_name("broken");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);

  // An unnamed encoded initializer is invalid as well.
  broken->ref_struct_type().set_type_ref(kInt4Block);
  broken->clear_name();
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Oneof semantics: setters clear siblings and the last alternative on the wire wins.
// ---------------------------------------------------------------------------

TEST(custom_values, OneofSettersClearSiblings) {
  TypeProto type;
  type.ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  ASSERT_EQ(type.value_case(), TypeProto::kTensorType);
  type.ref_struct_type().set_type_ref(kInt4Block);
  EXPECT_EQ(type.value_case(), TypeProto::kStructType);
  EXPECT_FALSE(type.has_tensor_type());
  EXPECT_FALSE(type.tensor_type_optional().has_value());
  // Mutable access to another alternative also switches the oneof.
  type.ref_sequence_type().set_elem_type(MakeTensorType(TensorProto::FLOAT, {}));
  EXPECT_EQ(type.value_case(), TypeProto::kSequenceType);
  EXPECT_FALSE(type.has_struct_type());
  type.clear_type();
  EXPECT_EQ(type.value_case(), TypeProto::kUndefined);
  EXPECT_FALSE(type.has_type());
  EXPECT_FALSE(type.sequence_type_optional().has_value());

  StructTypeProto declaration;
  declaration.ref_structure().add_field()->set_name("codes");
  ASSERT_EQ(declaration.kind_case(), StructTypeProto::kStructure);
  declaration.set_type_ref(kInt4Block);
  EXPECT_EQ(declaration.kind_case(), StructTypeProto::kTypeRef);
  EXPECT_FALSE(declaration.has_structure());
  declaration.ref_bit_packing().set_dimension(4);
  EXPECT_EQ(declaration.kind_case(), StructTypeProto::kBitPacking);
  EXPECT_FALSE(declaration.has_type_ref());
  declaration.ref_array().set_dimension(2);
  EXPECT_EQ(declaration.kind_case(), StructTypeProto::kArray);
  EXPECT_FALSE(declaration.has_bit_packing());

  StructTypeProto::Structure::Field field;
  field.set_name("scale");
  field.set_type(MakeTensorType(TensorProto::FLOAT, {}));
  ASSERT_EQ(field.content_case(), StructTypeProto::Structure::Field::kType);
  field.set_constant(MakeFloatConstant({0.25f}, {}));
  EXPECT_EQ(field.content_case(), StructTypeProto::Structure::Field::kConstant);
  EXPECT_FALSE(field.has_type());

  EncodedValueProto value;
  value.ref_affine().set_storage_type(TensorProto::INT8);
  ASSERT_EQ(value.layout_case(), EncodedValueProto::kAffine);
  value.ref_struct_type().set_type_ref(kInt4Block);
  EXPECT_EQ(value.layout_case(), EncodedValueProto::kStructType);
  EXPECT_FALSE(value.has_affine());
  value.ref_affine().set_storage_type(TensorProto::INT8);
  EXPECT_EQ(value.layout_case(), EncodedValueProto::kAffine);
  EXPECT_FALSE(value.has_struct_type());

  OptionalProto optional;
  optional.ref_tensor_value().set_data_type(TensorProto::FLOAT);
  ASSERT_TRUE(optional.has_tensor_value());
  optional.ref_sequence_value().set_elem_type(SequenceProto::TENSOR);
  EXPECT_TRUE(optional.has_sequence_value());
  EXPECT_FALSE(optional.has_tensor_value());
  EXPECT_FALSE(optional.tensor_value_optional().has_value());

  TypeProto moved_from;
  moved_from.ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  TypeProto moved_to(std::move(moved_from));
  EXPECT_TRUE(moved_to.has_tensor_type());
  EXPECT_FALSE(moved_from.has_tensor_type());
  EXPECT_EQ(moved_from.value_case(), TypeProto::kUndefined);
}

TEST(custom_values, OneofLastAlternativeOnTheWireWins) {
  TypeProto::Tensor tensor;
  tensor.set_elem_type(TensorProto::FLOAT);
  StructTypeProto reference;
  reference.set_type_ref(kInt4Block);

  // TypeProto: tensor_type (1) then struct_type (1000).
  TypeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(LengthDelimited(1, Bytes(tensor)) +
                                     LengthDelimited(1000, Bytes(reference))));
  EXPECT_EQ(parsed.value_case(), TypeProto::kStructType);
  EXPECT_FALSE(parsed.has_tensor_type());

  // ...and the reverse order.
  TypeProto reversed;
  ASSERT_TRUE(reversed.ParseFromString(LengthDelimited(1000, Bytes(reference)) +
                                       LengthDelimited(1, Bytes(tensor))));
  EXPECT_EQ(reversed.value_case(), TypeProto::kTensorType);
  EXPECT_FALSE(reversed.has_struct_type());

  TensorProto tensor_value;
  tensor_value.set_data_type(TensorProto::FLOAT);
  SequenceProto sequence_value;
  sequence_value.set_elem_type(SequenceProto::TENSOR);
  OptionalProto optional;
  ASSERT_TRUE(optional.ParseFromString(LengthDelimited(3, Bytes(tensor_value)) +
                                       LengthDelimited(5, Bytes(sequence_value))));
  EXPECT_TRUE(optional.has_sequence_value());
  EXPECT_FALSE(optional.has_tensor_value());

  // A tag whose wire type does not match is skipped and leaves the oneof alone.
  TypeProto skipped;
  ASSERT_TRUE(skipped.ParseFromString(LengthDelimited(1000, Bytes(reference)) + VarintField(1, 7)));
  EXPECT_EQ(skipped.value_case(), TypeProto::kStructType);
  EXPECT_FALSE(skipped.has_tensor_type());

  // StructTypeProto: structure (2) then type_ref (4, varint) and the reverse.
  StructTypeProto::Structure structure;
  structure.add_field()->set_name("codes");
  StructTypeProto ref_last;
  ASSERT_TRUE(
      ref_last.ParseFromString(LengthDelimited(2, Bytes(structure)) + VarintField(4, kInt4Block)));
  EXPECT_EQ(ref_last.kind_case(), StructTypeProto::kTypeRef);
  EXPECT_FALSE(ref_last.has_structure());
  StructTypeProto structure_last;
  ASSERT_TRUE(structure_last.ParseFromString(VarintField(4, kInt4Block) +
                                             LengthDelimited(2, Bytes(structure))));
  EXPECT_EQ(structure_last.kind_case(), StructTypeProto::kStructure);
  EXPECT_FALSE(structure_last.has_type_ref());

  // A type_ref tag with the wrong wire type is skipped.
  StructTypeProto ignored_ref;
  ASSERT_TRUE(ignored_ref.ParseFromString(LengthDelimited(2, Bytes(structure)) +
                                          LengthDelimited(4, std::string("\x01", 1))));
  EXPECT_EQ(ignored_ref.kind_case(), StructTypeProto::kStructure);

  // Structure::Field: type (2) then constant (4).
  TypeProto field_type = MakeTensorType(TensorProto::FLOAT, {});
  TensorProto constant = MakeFloatConstant({0.25f}, {});
  StructTypeProto::Structure::Field field;
  ASSERT_TRUE(field.ParseFromString(LengthDelimited(2, Bytes(field_type)) +
                                    LengthDelimited(4, Bytes(constant))));
  EXPECT_EQ(field.content_case(), StructTypeProto::Structure::Field::kConstant);
  EXPECT_FALSE(field.has_type());

  // EncodedValueProto: affine (1) then struct_type (2).
  AffineLayoutProto affine;
  affine.set_storage_type(TensorProto::INT8);
  EncodedValueProto value;
  ASSERT_TRUE(value.ParseFromString(LengthDelimited(1, Bytes(affine)) +
                                    LengthDelimited(2, Bytes(reference))));
  EXPECT_EQ(value.layout_case(), EncodedValueProto::kStructType);
  EXPECT_FALSE(value.has_affine());
  EncodedValueProto affine_last;
  ASSERT_TRUE(affine_last.ParseFromString(LengthDelimited(2, Bytes(reference)) +
                                          LengthDelimited(1, Bytes(affine))));
  EXPECT_EQ(affine_last.layout_case(), EncodedValueProto::kAffine);
  EXPECT_FALSE(affine_last.has_struct_type());
}

TEST(custom_values, RepeatedOneofMessageAlternativeMerges) {
  TypeProto::Tensor element;
  element.set_elem_type(TensorProto::FLOAT);
  TypeProto::Tensor shape;
  shape.ref_shape().add_dim()->set_dim_value(3);
  TypeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(LengthDelimited(1, Bytes(element)) +
                                     LengthDelimited(1, Bytes(shape))));
  ASSERT_TRUE(parsed.has_tensor_type());
  EXPECT_EQ(parsed.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_TRUE(parsed.tensor_type().has_shape());
  ASSERT_EQ(parsed.tensor_type().shape().dim_size(), 1);
  EXPECT_EQ(parsed.tensor_type().shape().dim(0).dim_value(), 3);
}

TEST(custom_values, SerializedSizeMismatchNamesTheField) {
  ModelProto model = MakeModel();
  utils::StringWriteStream stream;
  SerializeOptions options;
  const SerializeSizeResult total = model.SerializeSize(stream, options);
  ASSERT_GT(total.proto_size, 0);
  stream.pre_allocate(total.size());

  // Poison the cached size of the graph sub-message: the write pass must notice
  // the mismatch and name the enclosing field number (ModelProto.graph == 7).
  SerializeSizeResult poisoned;
  poisoned.proto_size = 1;
  stream.CacheSize(reinterpret_cast<const void *>(&model.ref_graph()), poisoned);
  const std::string message =
      RuntimeErrorMessage([&]() { model.SerializeToStream(stream, options); });
  EXPECT_NE(message.find("Serialized size"), std::string::npos) << message;
  EXPECT_NE(message.find("field 7"), std::string::npos) << message;
}

// ---------------------------------------------------------------------------
// Recursive validation through every TypeProto branch, attributes and functions.
// ---------------------------------------------------------------------------

TEST(custom_values, ValidatesNestedStructReferencesEverywhere) {
  const auto make_model = [](const TypeProto &carrier) {
    ModelProto model = MakeModel();
    DeclareInt4Block(model);
    ValueInfoProto *info = model.ref_graph().add_value_info();
    info->set_name("y");
    info->ref_type() = carrier;
    return model;
  };

  // A reference nested in a sequence, a map value or an optional is resolved.
  TypeProto sequence;
  sequence.ref_sequence_type().set_elem_type(MakeTypeRef(kInt4Block));
  TypeProto map;
  map.ref_map_type().set_key_type(TensorProto::INT64);
  map.ref_map_type().set_value_type(MakeTypeRef(kInt4Block));
  TypeProto optional_type;
  optional_type.ref_optional_type().set_elem_type(MakeTypeRef(kInt4Block));
  for (const TypeProto &carrier : {sequence, map, optional_type}) {
    EXPECT_NO_THROW(VerifyModel(make_model(carrier)));
  }

  // The same carriers with an undeclared reference are rejected.
  TypeProto bad_sequence;
  bad_sequence.ref_sequence_type().set_elem_type(MakeTypeRef(987654));
  TypeProto bad_map;
  bad_map.ref_map_type().set_key_type(TensorProto::STRING);
  bad_map.ref_map_type().set_value_type(MakeTypeRef(987654));
  TypeProto bad_optional;
  bad_optional.ref_optional_type().set_elem_type(MakeTypeRef(987654));
  for (const TypeProto &carrier : {bad_sequence, bad_map, bad_optional}) {
    EXPECT_THROW(VerifyModel(make_model(carrier)), std::invalid_argument);
  }
}

TEST(custom_values, ValidatesTypedAttributesFunctionsAndSubgraphs) {
  // A TYPE_PROTO attribute carrying an undeclared reference is rejected.
  ModelProto attribute_model = MakeModel();
  DeclareInt4Block(attribute_model);
  NodeProto *node = attribute_model.ref_graph().add_node();
  node->set_op_type("Cast");
  node->add_input("y");
  node->add_output("z");
  AttributeProto *typed = node->add_attribute();
  typed->set_name("to_type");
  typed->set_type(AttributeProto::TYPE_PROTO);
  typed->ref_tp() = MakeTypeRef(kInt4Block);
  EXPECT_NO_THROW(VerifyModel(attribute_model));
  typed->ref_tp() = MakeTypeRef(424242);
  EXPECT_THROW(VerifyModel(attribute_model), std::invalid_argument);

  // ...and the repeated TYPE_PROTOS form as well.
  typed->set_type(AttributeProto::TYPE_PROTOS);
  typed->clear_tp();
  typed->add_type_protos(MakeTypeRef(kInt4Block));
  EXPECT_NO_THROW(VerifyModel(attribute_model));
  typed->add_type_protos(MakeTypeRef(424242));
  EXPECT_THROW(VerifyModel(attribute_model), std::invalid_argument);

  // A model-local function's value_info is validated too.
  ModelProto function_model = MakeModel();
  DeclareInt4Block(function_model);
  FunctionProto *function = function_model.add_functions();
  function->set_name("Decode");
  function->set_domain("local");
  function->add_input("a");
  function->add_output("b");
  function->add_node("Identity", {"a"}, {"b"});
  ValueInfoProto *info = function->add_value_info();
  info->set_name("b");
  info->ref_type() = MakeTypeRef(kInt4Block);
  EXPECT_NO_THROW(VerifyModel(function_model));
  info->ref_type() = MakeTypeRef(424242);
  EXPECT_THROW(VerifyModel(function_model), std::invalid_argument);

  // A nested subgraph carries the catalogue as well.
  ModelProto subgraph_model = MakeModel();
  DeclareInt4Block(subgraph_model);
  NodeProto *branch = subgraph_model.ref_graph().add_node();
  branch->set_op_type("If");
  branch->add_input("y");
  branch->add_output("w");
  AttributeProto *body = branch->add_attribute();
  body->set_name("then_branch");
  body->set_type(AttributeProto::GRAPH);
  GraphProto &then_branch = body->ref_g();
  then_branch.set_name("then");
  ValueInfoProto *output = then_branch.add_output();
  output->set_name("packed");
  AddEncodedInitializer(then_branch, "packed", kInt4Block, MakeInt4BlockPayload(1));
  EXPECT_NO_THROW(VerifyModel(subgraph_model));
  then_branch.ref_encoded_initializer()[0].ref_struct_type().set_type_ref(424242);
  EXPECT_THROW(VerifyModel(subgraph_model), std::invalid_argument);
}

TEST(custom_values, RejectsInvalidMapKeyTypes) {
  ModelProto model = MakeModel();
  ValueInfoProto *info = model.ref_graph().add_value_info();
  info->set_name("dictionary");
  TypeProto &map = info->ref_type();
  map.ref_map_type().set_value_type(MakeTensorType(TensorProto::FLOAT, {}));

  // The default key_type (-1) is not a valid ONNX map key.
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  map.ref_map_type().set_key_type(TensorProto::FLOAT);
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  map.ref_map_type().set_key_type(TensorProto::INT64);
  EXPECT_NO_THROW(VerifyModel(model));
  map.ref_map_type().set_key_type(TensorProto::STRING);
  EXPECT_NO_THROW(VerifyModel(model));

  // The rule also applies to a map nested inside a struct field.
  ModelProto nested = MakeModel();
  StructTypeProto *declaration = nested.add_struct_types();
  declaration->set_type_id(uint64_t(9001));
  TypeProto bad_map;
  bad_map.ref_map_type().set_key_type(TensorProto::DOUBLE);
  bad_map.ref_map_type().set_value_type(MakeTensorType(TensorProto::FLOAT, {}));
  AddField(declaration->ref_structure(), "lookup", bad_map);
  EXPECT_THROW(VerifyModel(nested), std::invalid_argument);

  // Required tensor metadata is enforced below arrays and other containers.
  ModelProto missing_element_type = MakeModel();
  StructTypeProto *array = missing_element_type.add_struct_types();
  array->set_type_id(uint64_t(9002));
  array->ref_array().set_dimension(uint64_t(1));
  TypeProto incomplete_tensor;
  incomplete_tensor.ref_tensor_type().ref_shape();
  array->ref_array().set_element_type(incomplete_tensor);
  EXPECT_THROW(VerifyModel(missing_element_type), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Format constants must match their declared shape exactly.
// ---------------------------------------------------------------------------

TEST(custom_values, ConstantsMustMatchTheirDeclaredShape) {
  const auto make_model = [](const TensorProto &constant) {
    ModelProto model = MakeModel();
    StructTypeProto *declaration = model.add_struct_types();
    declaration->set_type_id(uint64_t(9100));
    declaration->set_name("WithConstant");
    AddField(declaration->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
    AddConstantField(declaration->ref_structure(), "table", constant);
    return model;
  };

  EXPECT_NO_THROW(VerifyModel(make_model(MakeFloatConstant({1.0f, 2.0f, 3.0f, 4.0f}, {4}))));
  // Too few values for the declared shape.
  EXPECT_THROW(VerifyModel(make_model(MakeFloatConstant({1.0f, 2.0f, 3.0f}, {4}))),
               std::invalid_argument);
  // Too many values: VerifyTensor tolerates the padding, the format constant does not.
  EXPECT_THROW(VerifyModel(make_model(MakeFloatConstant({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {4}))),
               std::invalid_argument);

  // Packed 4-bit constants use ceil(elements / 8) int32 entries.
  TensorProto packed;
  packed.set_data_type(TensorProto::INT4);
  packed.ref_dims().push_back(9);
  packed.ref_int32_data().push_back(0);
  packed.ref_int32_data().push_back(1);
  EXPECT_NO_THROW(VerifyModel(make_model(packed)));
  packed.ref_int32_data().push_back(2);
  EXPECT_THROW(VerifyModel(make_model(packed)), std::invalid_argument);

  // A raw_data constant needs exactly ceil(elements * bit_width / 8) bytes.
  TensorProto raw;
  raw.set_data_type(TensorProto::INT4);
  raw.ref_dims().push_back(5);
  raw.set_raw_data(std::string(3, '\x00'));
  EXPECT_NO_THROW(VerifyModel(make_model(raw)));
  raw.set_raw_data(std::string(4, '\x00'));
  EXPECT_THROW(VerifyModel(make_model(raw)), std::invalid_argument);

  // A constant whose dimensions overflow uint64 is rejected.
  TensorProto overflowing;
  overflowing.set_data_type(TensorProto::INT8);
  overflowing.ref_dims().push_back(4611686018427387904LL);
  overflowing.ref_dims().push_back(8LL);
  overflowing.set_raw_data(std::string(1, '\x00'));
  EXPECT_THROW(VerifyModel(make_model(overflowing)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// External metadata hardening.
// ---------------------------------------------------------------------------

TEST(custom_values, ExternalMetadataRejectsAmbiguousAndUnsafeEntries) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  StructTypeCatalogue catalogue;
  catalogue.Build(model);

  const auto make_value = [](const std::vector<std::pair<std::string, std::string>> &entries) {
    EncodedValueProto value;
    value.set_name("external");
    value.ref_struct_type().set_type_ref(kInt4Block);
    value.set_data_location(TensorProto::EXTERNAL);
    for (const auto &entry : entries) {
      AddExternalEntry(value, entry.first.c_str(), entry.second);
    }
    return value;
  };

  const EncodedValueProto good =
      make_value({{"location", "weights_0.data"}, {"offset", "4096"}, {"length", "2560"}});
  const EncodedValueLayout layout = catalogue.ValidateEncodedValue(good);
  EXPECT_EQ(layout.record_count, 128u);
  // Only the metadata was checked: the backing file was never opened.
  EXPECT_TRUE(layout.external);
  EXPECT_FALSE(layout.content_verified);

  // Duplicate keys make the metadata ambiguous.
  EXPECT_THROW(catalogue.ValidateEncodedValue(
                   make_value({{"location", "a.data"}, {"length", "20"}, {"length", "40"}})),
               std::invalid_argument);
  EXPECT_THROW(catalogue.ValidateEncodedValue(
                   make_value({{"location", "a.data"}, {"location", "b.data"}, {"length", "20"}})),
               std::invalid_argument);

  // Unsafe locations are rejected.
  for (const char *location : {"../secrets.bin", "/etc/passwd", "a/../../b.data", ".."}) {
    EXPECT_THROW(
        catalogue.ValidateEncodedValue(make_value({{"location", location}, {"length", "20"}})),
        std::invalid_argument)
        << location;
  }
  // A location inside a subdirectory stays acceptable.
  EXPECT_NO_THROW(catalogue.ValidateEncodedValue(
      make_value({{"location", "weights/w0.data"}, {"length", "20"}})));

  // offset + length must stay inside the addressable file range.
  EXPECT_THROW(catalogue.ValidateEncodedValue(make_value(
                   {{"location", "a.data"}, {"offset", "18446744073709551615"}, {"length", "20"}})),
               std::invalid_argument);
  EXPECT_THROW(catalogue.ValidateEncodedValue(make_value(
                   {{"location", "a.data"}, {"offset", "9223372036854775800"}, {"length", "20"}})),
               std::invalid_argument);
  EXPECT_THROW(catalogue.ValidateEncodedValue(
                   make_value({{"location", "a.data"}, {"offset", "-1"}, {"length", "20"}})),
               std::invalid_argument);
  EXPECT_THROW(catalogue.ValidateEncodedValue(
                   make_value({{"location", "a.data"}, {"length", "not-a-number"}})),
               std::invalid_argument);

  // An external affine payload cannot check its nibble padding, so the layout
  // is reported as metadata-only.
  EncodedValueProto affine_value;
  affine_value.set_name("external_affine");
  affine_value.ref_affine().set_storage_type(TensorProto::INT4);
  affine_value.ref_affine().set_scale(MakeFloatConstant({0.5f}, {}));
  affine_value.ref_logical_type() = MakeTensorType(TensorProto::FLOAT, {5});
  affine_value.set_data_location(TensorProto::EXTERNAL);
  AddExternalEntry(affine_value, "location", "codes.data");
  AddExternalEntry(affine_value, "length", "3");
  const EncodedValueLayout affine_layout = catalogue.ValidateEncodedValue(affine_value);
  EXPECT_EQ(affine_layout.record_count, 5u);
  EXPECT_FALSE(affine_layout.content_verified);
}

// ---------------------------------------------------------------------------
// Traversal cost and depth.
// ---------------------------------------------------------------------------

TEST(custom_values, CatalogueTraversalIsMemoizedOverSharedSubtypes) {
  // Each level references the previous one twice, so a walk without
  // memoization visits 2^levels nodes and never terminates in practice.
  constexpr uint64_t kLevels = 30;
  ModelProto model = MakeModel();
  StructTypeProto *leaf = model.add_struct_types();
  leaf->set_type_id(uint64_t(10000));
  AddField(leaf->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
  for (uint64_t level = 1; level <= kLevels; ++level) {
    StructTypeProto *node = model.add_struct_types();
    node->set_type_id(10000 + level);
    AddField(node->ref_structure(), "left", MakeTypeRef(10000 + level - 1));
    AddField(node->ref_structure(), "right", MakeTypeRef(10000 + level - 1));
  }

  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(model));
  uint64_t bits = 0;
  ASSERT_TRUE(catalogue.FixedBitSize(*catalogue.Find(10000 + kLevels), bits));
  EXPECT_EQ(bits, uint64_t(8) << kLevels);
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(custom_values, CatalogueTraversalRejectsChainsDeeperThanTheLimit) {
  const uint64_t levels = kMaxStructTypeDepth + 8;

  // Declared root first, so validation walks the whole chain and stops at the limit.
  ModelProto root_first = MakeModel();
  for (uint64_t level = 0; level < levels; ++level) {
    StructTypeProto *node = root_first.add_struct_types();
    node->set_type_id(20000 + level);
    if (level + 1 == levels) {
      AddField(node->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
    } else {
      AddField(node->ref_structure(), "next", MakeTypeRef(20000 + level + 1));
    }
  }
  EXPECT_THROW(VerifyModel(root_first), std::invalid_argument);

  // Declared leaf first, the catalogue builds; measuring the root still walks the
  // whole chain from scratch and is rejected by the same limit.
  ModelProto leaf_first = MakeModel();
  for (uint64_t level = 0; level < levels; ++level) {
    StructTypeProto *node = leaf_first.add_struct_types();
    node->set_type_id(20000 + level);
    if (level == 0) {
      AddField(node->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
    } else {
      AddField(node->ref_structure(), "next", MakeTypeRef(20000 + level - 1));
    }
  }
  StructTypeCatalogue catalogue;
  ASSERT_NO_THROW(catalogue.Build(leaf_first));
  uint64_t bits = 0;
  std::string reason;
  EXPECT_FALSE(catalogue.FixedBitSize(*catalogue.Find(20000 + levels - 1), bits, &reason));
  EXPECT_NE(reason.find("deep"), std::string::npos) << reason;

  // A chain just within the limit stays valid.
  ModelProto shallow = MakeModel();
  for (uint64_t level = 0; level < 8; ++level) {
    StructTypeProto *node = shallow.add_struct_types();
    node->set_type_id(30000 + level);
    if (level + 1 == 8) {
      AddField(node->ref_structure(), "code", MakeTensorType(TensorProto::INT8, {}));
    } else {
      AddField(node->ref_structure(), "next", MakeTypeRef(30000 + level + 1));
    }
  }
  EXPECT_NO_THROW(VerifyModel(shallow));
}

// ---------------------------------------------------------------------------
// Overflow-safe record access.
// ---------------------------------------------------------------------------

TEST(custom_values, ReadAccessRejectsOverflowingFieldRefs) {
  ModelProto model = MakeModel();
  DeclareInt4Block(model);
  EncodedValueProto *value =
      AddEncodedInitializer(model.ref_graph(), "w", kInt4Block, MakeInt4BlockPayload(2));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  EncodedValueView view(catalogue, *value);
  ASSERT_EQ(view.record_bits(), 160u);

  constexpr uint64_t kMax = ~uint64_t(0);
  EncodedFieldRef huge_offset;
  huge_offset.bit_offset = kMax;
  huge_offset.bit_stride = 4;
  huge_offset.bit_width = 4;
  huge_offset.count = 32;
  EXPECT_THROW(view.ReadElement(huge_offset, 0, 0), std::invalid_argument);

  EncodedFieldRef huge_stride;
  huge_stride.bit_offset = 0;
  huge_stride.bit_stride = kMax;
  huge_stride.bit_width = 4;
  huge_stride.count = 32;
  EXPECT_NO_THROW(view.ReadElement(huge_stride, 0, 0));
  EXPECT_THROW(view.ReadElement(huge_stride, 0, 1), std::invalid_argument);

  EncodedFieldRef huge_count;
  huge_count.bit_offset = 0;
  huge_count.bit_stride = 4;
  huge_count.bit_width = 4;
  huge_count.count = kMax;
  EXPECT_THROW(view.ReadElement(huge_count, 0, kMax - 1), std::invalid_argument);
  EXPECT_THROW(view.ReadElement(huge_count, 0, kMax / 4), std::invalid_argument);

  // Direct bit reads reject offsets that would wrap when added to the width.
  EXPECT_THROW(view.ReadBits(0, kMax, 8), std::invalid_argument);
  EXPECT_THROW(view.ReadBits(kMax, 0, 8), std::invalid_argument);
  EXPECT_THROW(view.ReadBits(0, 157, 64), std::invalid_argument);
}
