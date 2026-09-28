#include "onnx.h"
#include "onnx_ort_flatbuffers.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace {

#include "onnx_ort_input_schemas.inc"

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("ORT serialization: " + message);
}

template <typename T> std::string Text(const T &value) {
  return value.empty() ? std::string() : std::string(value.data(), value.size());
}

// Implements only the FlatBuffers wire primitives required by ort.fbs. Forward
// construction keeps all references positive, and explicit little-endian stores
// avoid host byte-order and unaligned-access assumptions. Field slots are eight
// bytes wide; vtables describe their actual locations and scalar widths.
class FlatBuffer {
public:
  struct Table {
    size_t position;
    size_t vtable;
  };

  FlatBuffer(int64_t limit, int64_t alignment)
      : limit_(limit > 0 ? std::min<uint64_t>(limit, INT32_MAX) : INT32_MAX),
        alignment_(static_cast<size_t>(std::max<int64_t>(alignment, 8))) {
    Allocate(8);
    std::memcpy(data_.data() + 4, "ORTM", 4);
  }

  size_t Allocate(size_t count) {
    if (count > limit_ - data_.size())
      throw std::length_error("ORT serialization exceeds the FlatBuffers or configured size limit");
    const size_t position = data_.size();
    data_.resize(position + count, '\0');
    return position;
  }

  void Align(size_t alignment, size_t prefix = 0) {
    Allocate((alignment - (data_.size() + prefix) % alignment) % alignment);
  }

  void AlignData() { Align(alignment_, 4); }

  template <typename T> void Put(size_t position, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      if constexpr (sizeof(T) == 4)
        Put(position, std::bit_cast<uint32_t>(value));
      else
        Put(position, std::bit_cast<uint64_t>(value));
    } else {
      using U = std::make_unsigned_t<T>;
      U bits = static_cast<U>(value);
      for (size_t i = 0; i < sizeof(T); ++i)
        data_[position + i] = static_cast<char>((bits >> (8 * i)) & 255);
    }
  }

  Table MakeTable(uint16_t fields) {
    Align(2);
    const size_t vtable = Allocate(4 + 2 * fields);
    Put<uint16_t>(vtable, 4 + 2 * fields);
    Put<uint16_t>(vtable + 2, 8 + 8 * fields);
    Align(8);
    const size_t position = Allocate(8 + 8 * fields);
    Put<int32_t>(position, static_cast<int32_t>(position - vtable));
    return {position, vtable};
  }

  template <typename T> void Scalar(Table table, size_t field, T value) {
    Put<uint16_t>(table.vtable + 4 + 2 * field, static_cast<uint16_t>(8 + field * 8));
    Put(table.position + 8 + field * 8, value);
  }

  void Offset(size_t position, size_t target) {
    Require(target > position, "invalid forward FlatBuffer reference");
    Put<uint32_t>(position, static_cast<uint32_t>(target - position));
  }

  void Reference(Table table, size_t field, size_t target) {
    Scalar<uint32_t>(table, field,
                     static_cast<uint32_t>(target - (table.position + 8 + field * 8)));
  }

  size_t String(std::string_view value) {
    Align(4);
    const size_t position = Allocate(4);
    Put<uint32_t>(position, static_cast<uint32_t>(value.size()));
    const size_t payload = Allocate(value.size() + 1);
    if (!value.empty())
      std::memcpy(data_.data() + payload, value.data(), value.size());
    return position;
  }

  template <typename T> void String(Table table, size_t field, const T &value) {
    Reference(
        table, field,
        String(value.empty() ? std::string_view() : std::string_view(value.data(), value.size())));
  }

  template <typename T, typename Container> size_t Scalars(const Container &values) {
    Align(std::max<size_t>(4, sizeof(T)), 4);
    const size_t position = Allocate(4);
    Put<uint32_t>(position, static_cast<uint32_t>(values.size()));
    const size_t payload = Allocate(values.size() * sizeof(T));
    size_t i = 0;
    for (const auto value : values)
      Put<T>(payload + sizeof(T) * i++, static_cast<T>(value));
    return position;
  }

  template <typename Container, typename Function>
  size_t Objects(const Container &values, Function write) {
    Align(4);
    const size_t position = Allocate(4);
    Put<uint32_t>(position, static_cast<uint32_t>(values.size()));
    const size_t payload = Allocate(values.size() * 4);
    size_t i = 0;
    for (const auto &value : values)
      Offset(payload + 4 * i++, write(value));
    return position;
  }

  template <typename Container> size_t Strings(const Container &values) {
    return Objects(values, [&](const auto &value) {
      return String(std::string_view(value.data(), value.size()));
    });
  }

  size_t Bytes(const void *data, size_t size) {
    AlignData();
    const size_t position = Allocate(4);
    Put<uint32_t>(position, static_cast<uint32_t>(size));
    const size_t payload = Allocate(size);
    if (size)
      std::memcpy(data_.data() + payload, data, size);
    return position;
  }

  std::string Finish(size_t root) {
    Offset(0, root);
    Align(8);
    return std::move(data_);
  }

private:
  size_t limit_;
  size_t alignment_;
  std::string data_;
};

class OrtWriter {
public:
  explicit OrtWriter(const SerializeOptions &options)
      : buffer_(options.max_serialized_size_bytes, options.alignment) {}

  std::string Model(const ModelProto &model) {
    Require(model.has_graph(), "model has no graph");
    Require(model.has_ir_version() && model.ir_version() > 0, "model has no valid IR version");
    Require(model.functions().empty(), "model-local functions are unsupported");
    Require(model.configuration().empty(), "device configurations are unsupported");
    Require(model.struct_types().empty(), "structured types are unsupported");
    Require(!model.opset_import().empty(), "model has no opset imports");
    std::map<std::string, int64_t> domains;
    for (const auto &opset : model.opset_import()) {
      Require(opset.version() > 0 && opset.version() <= INT32_MAX, "invalid opset version");
      const auto domain = NormalizeDomain(Text(opset.domain()));
      Require(domains.emplace(domain, opset.version()).second, "duplicate opset domain");
      if (domain.empty())
        onnx_opset_ = opset.version();
    }
    domains_ = std::move(domains);
    const auto session = buffer_.MakeTable(4);
    // Version 4 is deliberately used: full ORT upgrades its kernel metadata
    // after resolving schemas. No kernel hashes or optimized EP state are saved.
    buffer_.Reference(session, 0, buffer_.String("4"));
    const auto result = buffer_.MakeTable(10);
    buffer_.Reference(session, 1, result.position);
    buffer_.Scalar<int64_t>(result, 0, model.ir_version());
    buffer_.Reference(result, 1, buffer_.Objects(model.opset_import(), [&](const auto &opset) {
      auto item = buffer_.MakeTable(2);
      buffer_.Reference(item, 0, buffer_.String(NormalizeDomain(Text(opset.domain()))));
      buffer_.Scalar<int64_t>(item, 1, opset.version());
      return item.position;
    }));
    buffer_.String(result, 2, model.producer_name());
    buffer_.String(result, 3, model.producer_version());
    buffer_.String(result, 4, model.domain());
    buffer_.Scalar<int64_t>(result, 5, model.has_model_version() ? model.model_version() : 0);
    buffer_.String(result, 6, model.doc_string());
    buffer_.Reference(result, 7, Graph(model.graph(), {}, 0));
    buffer_.String(result, 8, model.graph().doc_string());
    buffer_.Reference(result, 9, buffer_.Objects(model.metadata_props(), [&](const auto &entry) {
      auto item = buffer_.MakeTable(2);
      buffer_.String(item, 0, entry.key());
      buffer_.String(item, 1, entry.value());
      return item.position;
    }));
    return buffer_.Finish(session.position);
  }

private:
  using Types = std::map<std::string, TypeProto>;

  static std::string NormalizeDomain(const std::string &domain) {
    return domain == "ai.onnx" ? "" : domain;
  }

  std::vector<int32_t> InputArgCounts(const NodeProto &node) const {
    const auto domain = NormalizeDomain(Text(node.domain()));
    const auto name = Text(node.op_type());
    const auto version = domains_.at(domain);
    const auto key = std::pair{std::string_view(domain), std::string_view(name)};
    auto it = std::lower_bound(std::begin(kOrtInputSchemas), std::end(kOrtInputSchemas), key,
                               [](const OrtInputSchema &schema, const auto &value) {
                                 return std::pair{std::string_view(schema.Domain()),
                                                  std::string_view(schema.Name())} < value;
                               });
    const OrtInputSchema *schema = nullptr;
    for (; it != std::end(kOrtInputSchemas) && it->Domain() == key.first &&
           it->Name() == key.second && it->since_version <= version;
         ++it)
      schema = it;
    Require(schema != nullptr, "operator input schema is unavailable for '" + domain + "::" + name +
                                   "' at opset " + std::to_string(version));
    Require(node.input().size() <= INT32_MAX, "node input count exceeds FlatBuffers limit");
    Require(node.input().size() >= static_cast<size_t>(schema->min_input) &&
                node.input().size() <= static_cast<size_t>(schema->max_input),
            "input count does not match the schema for '" + name + "'");
    std::vector<int32_t> counts;
    size_t index = 0;
    for (const char kind : std::string_view(schema->Inputs())) {
      const auto count = kind == 'V' ? node.input().size() - index
                                     : static_cast<size_t>(index < node.input().size());
      Require(kind != 'S' || (count == 1 && !node.input()[index].empty()),
              "missing required input for '" + name + "'");
      Require(kind != 'V' || count >= static_cast<size_t>(schema->min_variadic_arity),
              "missing variadic inputs for '" + name + "'");
      counts.push_back(static_cast<int32_t>(count));
      index += count;
    }
    Require(index == node.input().size(), "too many inputs for '" + name + "'");
    return counts;
  }

  size_t Shape(const TensorShapeProto &shape) {
    auto result = buffer_.MakeTable(1);
    buffer_.Reference(result, 0, buffer_.Objects(shape.dim(), [&](const auto &dim) {
      auto item = buffer_.MakeTable(2);
      auto value = buffer_.MakeTable(3);
      buffer_.Reference(item, 0, value.position);
      buffer_.Scalar<int8_t>(value, 0, dim.has_dim_value() ? 1 : dim.has_dim_param() ? 2 : 0);
      if (dim.has_dim_value()) {
        Require(dim.dim_value() >= 0, "negative shape dimension");
        buffer_.Scalar<int64_t>(value, 1, dim.dim_value());
      }
      if (dim.has_dim_param())
        buffer_.String(value, 2, dim.dim_param());
      buffer_.String(item, 1, dim.denotation());
      return item.position;
    }));
    return result.position;
  }

  size_t Type(const TypeProto &type, size_t depth = 0) {
    Require(depth < 100, "type nesting exceeds 100");
    auto result = buffer_.MakeTable(3);
    buffer_.String(result, 0, type.denotation());
    if (type.has_tensor_type()) {
      const auto element = static_cast<int>(type.tensor_type().elem_type());
      Require(element > 0 && element <= 20 && element != 14 && element != 15,
              "unsupported or missing tensor element type " + std::to_string(element));
      buffer_.Scalar<uint8_t>(result, 1, 1);
      auto tensor = buffer_.MakeTable(2);
      buffer_.Reference(result, 2, tensor.position);
      buffer_.Scalar<int32_t>(tensor, 0, static_cast<int32_t>(type.tensor_type().elem_type()));
      if (type.tensor_type().has_shape())
        buffer_.Reference(tensor, 1, Shape(type.tensor_type().shape()));
    } else if (type.has_sequence_type()) {
      Require(type.sequence_type().has_elem_type(), "sequence has no element type");
      buffer_.Scalar<uint8_t>(result, 1, 2);
      auto sequence = buffer_.MakeTable(1);
      buffer_.Reference(result, 2, sequence.position);
      buffer_.Reference(sequence, 0, Type(type.sequence_type().elem_type(), depth + 1));
    } else if (type.has_map_type()) {
      Require(type.map_type().has_value_type(), "map has no value type");
      buffer_.Scalar<uint8_t>(result, 1, 3);
      auto map = buffer_.MakeTable(2);
      buffer_.Reference(result, 2, map.position);
      buffer_.Scalar<int32_t>(map, 0, static_cast<int32_t>(type.map_type().key_type()));
      buffer_.Reference(map, 1, Type(type.map_type().value_type(), depth + 1));
    } else {
      Require(false, "optional, sparse, opaque, structured, or missing value type is unsupported");
    }
    return result.position;
  }

  static TypeProto TensorType(const TensorProto &tensor) {
    TypeProto result;
    auto *type = result.mutable_tensor_type();
    type->set_elem_type(tensor.data_type());
    auto *shape = type->mutable_shape();
    for (int64_t dim : tensor.dims())
      shape->add_dim()->set_dim_value(dim);
    return result;
  }

  static TypeProto ElementType(int type) {
    TypeProto result;
    result.mutable_tensor_type()->set_elem_type(static_cast<TensorProto::DataType>(type));
    return result;
  }

  std::vector<TypeProto> OutputTypes(const NodeProto &node, const Types &types) const {
    std::vector<TypeProto> result(node.output().size());
    if (!NormalizeDomain(Text(node.domain())).empty())
      return result;
    const auto op = Text(node.op_type());
    const auto input_type = [&](size_t index) {
      if (index >= node.input().size())
        return TypeProto();
      auto it = types.find(Text(node.input()[index]));
      if (it == types.end())
        return TypeProto();
      TypeProto type = it->second;
      if (type.has_tensor_type())
        type.mutable_tensor_type()->clear_shape();
      return type;
    };
    const auto attr = [&](const char *name) -> const AttributeProto * {
      for (const auto &value : node.attribute())
        if (value.name() == name)
          return &value;
      return nullptr;
    };
    // These operators have schema-invariant output element types. Shape
    // inference remains ORT's responsibility. Other operators require explicit
    // value_info rather than guessing a type and changing executable semantics.
    static const std::set<std::string> same_type = {"Abs",
                                                    "Acos",
                                                    "Acosh",
                                                    "Add",
                                                    "Asin",
                                                    "Asinh",
                                                    "Atan",
                                                    "Atanh",
                                                    "AveragePool",
                                                    "BitShift",
                                                    "BitwiseAnd",
                                                    "BitwiseNot",
                                                    "BitwiseOr",
                                                    "BitwiseXor",
                                                    "Ceil",
                                                    "Clip",
                                                    "Compress",
                                                    "Concat",
                                                    "Conv",
                                                    "ConvTranspose",
                                                    "Cos",
                                                    "Cosh",
                                                    "CumSum",
                                                    "DepthToSpace",
                                                    "Det",
                                                    "Div",
                                                    "Dropout",
                                                    "Einsum",
                                                    "Elu",
                                                    "Erf",
                                                    "Exp",
                                                    "Expand",
                                                    "Flatten",
                                                    "Floor",
                                                    "Gather",
                                                    "GatherElements",
                                                    "GatherND",
                                                    "Gemm",
                                                    "GlobalAveragePool",
                                                    "GlobalLpPool",
                                                    "GlobalMaxPool",
                                                    "HardSigmoid",
                                                    "HardSwish",
                                                    "Identity",
                                                    "InstanceNormalization",
                                                    "LRN",
                                                    "LeakyRelu",
                                                    "Log",
                                                    "LogSoftmax",
                                                    "LpNormalization",
                                                    "LpPool",
                                                    "MatMul",
                                                    "Max",
                                                    "MaxPool",
                                                    "MaxUnpool",
                                                    "Mean",
                                                    "Min",
                                                    "Mod",
                                                    "Mul",
                                                    "Neg",
                                                    "PRelu",
                                                    "Pad",
                                                    "Pow",
                                                    "Reciprocal",
                                                    "ReduceL1",
                                                    "ReduceL2",
                                                    "ReduceLogSum",
                                                    "ReduceLogSumExp",
                                                    "ReduceMax",
                                                    "ReduceMean",
                                                    "ReduceMin",
                                                    "ReduceProd",
                                                    "ReduceSum",
                                                    "ReduceSumSquare",
                                                    "Relu",
                                                    "Reshape",
                                                    "Resize",
                                                    "ReverseSequence",
                                                    "Round",
                                                    "Scatter",
                                                    "ScatterElements",
                                                    "ScatterND",
                                                    "Selu",
                                                    "Shrink",
                                                    "Sigmoid",
                                                    "Sign",
                                                    "Sin",
                                                    "Sinh",
                                                    "Slice",
                                                    "Softmax",
                                                    "Softplus",
                                                    "Softsign",
                                                    "SpaceToDepth",
                                                    "Split",
                                                    "Sqrt",
                                                    "Squeeze",
                                                    "Sub",
                                                    "Sum",
                                                    "Tan",
                                                    "Tanh",
                                                    "ThresholdedRelu",
                                                    "Tile",
                                                    "TopK",
                                                    "Transpose",
                                                    "Trilu",
                                                    "Unsqueeze",
                                                    "Upsample"};
    static const std::set<std::string> boolean = {"And",   "Equal", "Greater", "GreaterOrEqual",
                                                  "IsInf", "IsNaN", "Less",    "LessOrEqual",
                                                  "Not",   "Or",    "Xor"};
    if (same_type.count(op)) {
      std::fill(result.begin(), result.end(), input_type(0));
      if (result.size() > 1 && (op == "TopK" || op == "MaxPool"))
        result[1] = ElementType(TensorProto::INT64);
      if (result.size() > 1 && op == "Dropout" && onnx_opset_ >= 10)
        result[1] = ElementType(TensorProto::BOOL);
    } else if (boolean.count(op)) {
      std::fill(result.begin(), result.end(), ElementType(TensorProto::BOOL));
    } else if (op == "Shape" || op == "Size" || op == "NonZero" || op == "ArgMax" ||
               op == "ArgMin") {
      std::fill(result.begin(), result.end(), ElementType(TensorProto::INT64));
    } else if (op == "Cast") {
      const auto *to = attr("to");
      Require(to && to->has_i(), "Cast requires its 'to' attribute");
      std::fill(result.begin(), result.end(), ElementType(static_cast<int>(to->i())));
    } else if (op == "CastLike" || op == "Where" || op == "OneHot") {
      std::fill(result.begin(), result.end(), input_type(op == "OneHot" ? 2 : 1));
    } else if (op == "BatchNormalization") {
      for (size_t i = 0; i < result.size(); ++i)
        result[i] = input_type(i == 0 ? 0 : i % 2 ? 3 : 4);
    } else if (op == "Constant" || op == "ConstantOfShape") {
      TypeProto type;
      const auto *value = attr("value");
      if (value && value->has_t())
        type = TensorType(value->t());
      else if (op == "ConstantOfShape" || attr("value_float") || attr("value_floats"))
        type = ElementType(TensorProto::FLOAT);
      else if (attr("value_int") || attr("value_ints"))
        type = ElementType(TensorProto::INT64);
      else if (attr("value_string") || attr("value_strings"))
        type = ElementType(TensorProto::STRING);
      std::fill(result.begin(), result.end(), type);
    } else if (op == "If" || op == "Loop" || op == "Scan") {
      const auto *body = attr(op == "If" ? "then_branch" : "body");
      if (body && body->has_g()) {
        const size_t offset = op == "Loop" ? 1 : 0;
        for (size_t i = 0; i < result.size() && i + offset < body->g().output().size(); ++i)
          if (body->g().output()[i + offset].has_type())
            result[i] = body->g().output()[i + offset].type();
      }
    }
    return result;
  }

  size_t Tensor(const TensorProto &tensor) {
    Require(tensor.metadata_props().empty(), "tensor metadata_props are unsupported");
    Require(!tensor.has_segment(), "segmented tensors are unsupported");
    EXT_ENFORCE(!tensor.has_data_location() || tensor.data_location() != TensorProto::EXTERNAL ||
                    tensor.has_raw_data(),
                "ORT serialization: external tensor payload must be loaded before serialization");
    const int type = static_cast<int>(tensor.data_type());
    Require(type > 0 && type <= 20 && type != 14 && type != 15,
            "unsupported tensor data type " + std::to_string(type));
    const bool has_typed_data = !tensor.float_data().empty() || !tensor.double_data().empty() ||
                                !tensor.int32_data().empty() || !tensor.int64_data().empty() ||
                                !tensor.uint64_data().empty() || !tensor.string_data().empty();
    Require(!tensor.has_raw_data() || !has_typed_data, "tensor has both raw_data and typed data");
    Require(tensor.float_data().empty() || type == 1, "float_data does not match tensor type");
    Require(tensor.double_data().empty() || type == 11, "double_data does not match tensor type");
    Require(tensor.int64_data().empty() || type == 7, "int64_data does not match tensor type");
    Require(tensor.uint64_data().empty() || type == 12 || type == 13,
            "uint64_data does not match tensor type");
    Require(tensor.string_data().empty() || type == 8, "string_data does not match tensor type");
    Require(tensor.int32_data().empty() ||
                (type != 1 && type != 7 && type != 8 && type != 11 && type != 12 && type != 13),
            "int32_data does not match tensor type");
    size_t count = 1;
    for (int64_t dim : tensor.dims()) {
      Require(dim >= 0, "negative initializer dimension");
      Require(dim == 0 || count <= static_cast<size_t>(INT32_MAX) / static_cast<uint64_t>(dim),
              "initializer element count exceeds FlatBuffers limit");
      count *= static_cast<size_t>(dim);
    }
    auto result = buffer_.MakeTable(7);
    buffer_.String(result, 0, tensor.name());
    buffer_.String(result, 1, tensor.doc_string());
    buffer_.Reference(result, 2, buffer_.Scalars<int64_t>(tensor.dims()));
    buffer_.Scalar<int32_t>(result, 3, type);
    if (type == 8) {
      Require(!tensor.has_raw_data(), "string tensors cannot contain raw_data");
      Require(tensor.string_data().size() == count, "string tensor element count mismatch");
      buffer_.Reference(result, 5, buffer_.Strings(tensor.string_data()));
      return result.position;
    }
    static constexpr size_t widths[] = {0, 4, 1, 1, 2,  2, 4, 8, 0, 1, 2,
                                        8, 4, 8, 8, 16, 2, 1, 1, 1, 1};
    Require(count <= static_cast<size_t>(INT32_MAX) / widths[type],
            "tensor byte size exceeds FlatBuffers limit");
    const size_t bytes = count * widths[type];
    EXT_ENFORCE(!tensor.has_data_location() || tensor.data_location() != TensorProto::EXTERNAL ||
                    tensor.raw_data().size() == bytes,
                "ORT serialization: external tensor payload must be loaded completely before "
                "serialization (expected ",
                bytes, " bytes, found ", tensor.raw_data().size(), ")");
    if (tensor.has_raw_data()) {
      Require(tensor.raw_data().size() == bytes, "raw tensor byte size mismatch");
      buffer_.Reference(result, 4, buffer_.Bytes(tensor.raw_data().data(), bytes));
    } else {
      size_t data = 0;
      switch (type) {
      case 1:
      case 14:
        Require(tensor.float_data().size() == count * (type == 14 ? 2 : 1),
                "float tensor element count mismatch");
        data = NumericBytes<float>(tensor.float_data());
        break;
      case 11:
      case 15:
        Require(tensor.double_data().size() == count * (type == 15 ? 2 : 1),
                "double tensor element count mismatch");
        data = NumericBytes<double>(tensor.double_data());
        break;
      case 7:
        Require(tensor.int64_data().size() == count, "int64 tensor element count mismatch");
        data = NumericBytes<int64_t>(tensor.int64_data());
        break;
      case 12:
      case 13:
        Require(tensor.uint64_data().size() == count, "unsigned tensor element count mismatch");
        data = type == 12 ? NumericBytes<uint32_t>(tensor.uint64_data())
                          : NumericBytes<uint64_t>(tensor.uint64_data());
        break;
      default:
        Require(tensor.int32_data().size() == count, "integer tensor element count mismatch");
        data = widths[type] == 1   ? NumericBytes<uint8_t>(tensor.int32_data())
               : widths[type] == 2 ? NumericBytes<uint16_t>(tensor.int32_data())
                                   : NumericBytes<int32_t>(tensor.int32_data());
      }
      buffer_.Reference(result, 4, data);
    }
    return result.position;
  }

  template <typename T, typename Container> size_t NumericBytes(const Container &values) {
    buffer_.AlignData();
    size_t result = buffer_.Allocate(4);
    buffer_.Put<uint32_t>(result, static_cast<uint32_t>(values.size() * sizeof(T)));
    size_t payload = buffer_.Allocate(values.size() * sizeof(T));
    size_t index = 0;
    for (auto value : values)
      buffer_.Put<T>(payload + sizeof(T) * index++, static_cast<T>(value));
    return result;
  }

  static std::set<std::string> Captures(const GraphProto &graph, size_t depth) {
    Require(depth < 100, "graph nesting exceeds 100");
    std::set<std::string> local, used;
    for (const auto &value : graph.input())
      local.insert(Text(value.name()));
    for (const auto &value : graph.initializer())
      local.insert(Text(value.name()));
    for (const auto &node : graph.node()) {
      for (const auto &name : node.output())
        local.insert(Text(name));
      for (const auto &name : node.input())
        used.insert(Text(name));
      for (const auto &attr : node.attribute()) {
        if (attr.has_g()) {
          auto nested = Captures(attr.g(), depth + 1);
          used.insert(nested.begin(), nested.end());
        }
      }
    }
    for (const auto &value : graph.output())
      used.insert(Text(value.name()));
    used.erase("");
    for (const auto &name : local)
      used.erase(name);
    return used;
  }

  size_t Attribute(const AttributeProto &attr, const Types &types, size_t depth) {
    Require(attr.ref_attr_name().empty(), "reference attributes require function expansion");
    Require(!attr.name().empty(), "attribute name is missing");
    const int type = static_cast<int>(attr.type());
    Require(type >= 1 && type <= 10, "unsupported attribute type " + std::to_string(type));
    Require(type != 10, "GRAPHS attributes are unsupported by the ORT loader");
    auto result = buffer_.MakeTable(13);
    buffer_.String(result, 0, attr.name());
    buffer_.String(result, 1, attr.doc_string());
    buffer_.Scalar<int32_t>(result, 2, type);
    switch (type) {
    case 1:
      Require(attr.has_f(), "FLOAT attribute has no value");
      buffer_.Scalar<float>(result, 3, attr.f());
      break;
    case 2:
      Require(attr.has_i(), "INT attribute has no value");
      buffer_.Scalar<int64_t>(result, 4, attr.i());
      break;
    case 3:
      buffer_.String(result, 5, attr.s());
      break;
    case 4:
      Require(attr.has_t(), "TENSOR attribute has no value");
      buffer_.Reference(result, 6, Tensor(attr.t()));
      break;
    case 5:
      Require(attr.has_g(), "GRAPH attribute has no value");
      buffer_.Reference(result, 7, Graph(attr.g(), types, depth + 1));
      break;
    case 6:
      buffer_.Reference(result, 8, buffer_.Scalars<float>(attr.floats()));
      break;
    case 7:
      buffer_.Reference(result, 9, buffer_.Scalars<int64_t>(attr.ints()));
      break;
    case 8:
      buffer_.Reference(result, 10, buffer_.Strings(attr.strings()));
      break;
    case 9:
      buffer_.Reference(result, 11, buffer_.Objects(attr.tensors(), [&](const auto &tensor) {
        return Tensor(tensor);
      }));
      break;
    }
    return result.position;
  }

  size_t Graph(const GraphProto &graph, const Types &outer, size_t depth) {
    Require(graph.persistent_bindings().empty(), "persistent bindings are unsupported");
    Require(depth < 100, "graph nesting exceeds 100");
    Require(graph.metadata_props().empty(), "graph metadata_props are unsupported");
    for (const auto &node : graph.node())
      Require(node.metadata_props().empty(), "node metadata_props are unsupported");
    const auto is_constant = [](const NodeProto &node) {
      return NormalizeDomain(Text(node.domain())).empty() && node.op_type() == "Constant";
    };
    if (std::any_of(graph.node().begin(), graph.node().end(), is_constant)) {
      // ORT has no executable Constant kernel: its ONNX loader lifts these to
      // initializers before saving ORT format. Reproduces that normalization.
      GraphProto normalized;
      normalized.CopyFrom(graph);
      normalized.clr_node();
      for (const auto &node : graph.node()) {
        if (!is_constant(node)) {
          normalized.add_node()->CopyFrom(node);
          continue;
        }
        Require(node.device_configurations().empty() && node.overload().empty(),
                "configured or overloaded Constant nodes are unsupported");
        Require(node.input().empty() && node.output().size() == 1 && node.attribute().size() == 1,
                "Constant must have one output, one value attribute and no inputs");
        const auto &attr = node.attribute()[0];
        Require(attr.ref_attr_name().empty(), "Constant reference attributes are unsupported");
        TensorProto *tensor = normalized.add_initializer();
        if (attr.name() == "value" && attr.type() == AttributeProto::TENSOR && attr.has_t()) {
          tensor->CopyFrom(attr.t());
        } else if (attr.name() == "value_float" && attr.type() == AttributeProto::FLOAT &&
                   attr.has_f()) {
          tensor->set_data_type(TensorProto::FLOAT);
          tensor->add_float_data(attr.f());
        } else if (attr.name() == "value_int" && attr.type() == AttributeProto::INT &&
                   attr.has_i()) {
          tensor->set_data_type(TensorProto::INT64);
          tensor->add_int64_data(attr.i());
        } else if (attr.name() == "value_string" && attr.type() == AttributeProto::STRING) {
          tensor->set_data_type(TensorProto::STRING);
          tensor->add_string_data(Text(attr.s()));
        } else if (attr.name() == "value_floats" && attr.type() == AttributeProto::FLOATS) {
          tensor->set_data_type(TensorProto::FLOAT);
          tensor->add_dims(static_cast<int64_t>(attr.floats().size()));
          for (float value : attr.floats())
            tensor->add_float_data(value);
        } else if (attr.name() == "value_ints" && attr.type() == AttributeProto::INTS) {
          tensor->set_data_type(TensorProto::INT64);
          tensor->add_dims(static_cast<int64_t>(attr.ints().size()));
          for (int64_t value : attr.ints())
            tensor->add_int64_data(value);
        } else if (attr.name() == "value_strings" && attr.type() == AttributeProto::STRINGS) {
          tensor->set_data_type(TensorProto::STRING);
          tensor->add_dims(static_cast<int64_t>(attr.strings().size()));
          for (const auto &value : attr.strings())
            tensor->add_string_data(value);
        } else {
          Require(false, "unsupported or invalid Constant value attribute");
        }
        tensor->set_name(node.output()[0]);
      }
      return Graph(normalized, outer, depth);
    }
    Require(graph.encoded_initializer().empty(), "encoded initializers are unsupported");
    Require(graph.paged_cache_initializer().empty(), "paged cache initializers are unsupported");
    Require(graph.sparse_initializer().empty(), "sparse initializers are unsupported");
    Require(graph.quantization_annotation().empty(), "quantization annotations are unsupported");
    Types types;
    std::map<std::string, const ValueInfoProto *> infos;
    std::set<std::string> available;
    auto add_info = [&](const auto &values) {
      for (const auto &value : values) {
        Require(value.metadata_props().empty(), "value_info metadata_props are unsupported");
        const auto name = Text(value.name());
        Require(!name.empty(), "graph value has an empty name");
        infos[name] = &value;
        if (value.has_type())
          types[name] = value.type();
      }
    };
    add_info(graph.value_info());
    add_info(graph.output());
    add_info(graph.input());
    for (const auto &value : graph.input())
      available.insert(Text(value.name()));
    std::set<std::string> initializer_names;
    for (const auto &tensor : graph.initializer()) {
      const auto name = Text(tensor.name());
      Require(!name.empty() && initializer_names.insert(name).second,
              "empty or duplicate initializer name");
      auto declared = types.find(name);
      if (declared == types.end()) {
        types[name] = TensorType(tensor);
      } else {
        Require(declared->second.has_tensor_type(),
                "initializer '" + name + "' has a non-tensor declared type");
        const auto &type = declared->second.tensor_type();
        Require(type.elem_type() == tensor.data_type(),
                "initializer '" + name + "' does not match its declared element type");
        if (type.has_shape()) {
          Require(type.shape().dim().size() == tensor.dims().size(),
                  "initializer '" + name + "' does not match its declared rank");
          for (size_t i = 0; i < tensor.dims().size(); ++i) {
            const auto &dim = type.shape().dim()[i];
            Require(!dim.has_dim_value() || dim.dim_value() == tensor.dims()[i],
                    "initializer '" + name + "' does not match its declared dimension");
          }
        }
        // Initializers can be overridable graph inputs: their default payload
        // must not narrow the caller's symbolic, unknown, or unspecified shape.
      }
      available.insert(name);
    }
    for (const auto &value : graph.input())
      Require(types.count(Text(value.name())),
              "graph input '" + Text(value.name()) + "' requires explicit type information");
    for (const auto &name : Captures(graph, depth)) {
      auto it = outer.find(name);
      Require(it != outer.end(), "unresolved graph input '" + name + "'");
      types[name] = it->second;
      available.insert(name);
    }
    struct Edge {
      uint32_t node;
      int32_t source;
      int32_t destination;
    };
    std::map<std::string, std::pair<uint32_t, int32_t>> producers;
    std::vector<std::vector<Edge>> inputs(graph.node().size()), outputs(graph.node().size());
    std::vector<std::vector<std::string>> implicit(graph.node().size());
    uint32_t index = 0;
    for (const auto &node : graph.node()) {
      Require(node.device_configurations().empty(), "node device configurations are unsupported");
      Require(node.overload().empty(), "node overloads require function expansion");
      Require(!node.op_type().empty(), "node has no operator type");
      Require(domains_.count(NormalizeDomain(Text(node.domain()))) != 0,
              "node domain has no opset import");
      std::set<std::string> captures, attribute_names;
      for (const auto &attr : node.attribute()) {
        Require(attribute_names.insert(Text(attr.name())).second, "duplicate attribute name");
        if (attr.has_g()) {
          auto names = Captures(attr.g(), depth + 1);
          captures.insert(names.begin(), names.end());
        }
      }
      implicit[index].assign(captures.begin(), captures.end());
      std::vector<std::string> arguments;
      for (const auto &name : node.input())
        arguments.push_back(Text(name));
      arguments.insert(arguments.end(), captures.begin(), captures.end());
      int32_t argument = 0;
      for (const auto &name : arguments) {
        Require(name.empty() || available.count(name),
                "input '" + name + "' is unresolved or nodes are not topologically sorted");
        Require(name.empty() || types.count(name),
                "missing type for '" + name + "'; populate graph value_info before serialization");
        auto producer = producers.find(name);
        if (producer != producers.end()) {
          inputs[index].push_back({producer->second.first, producer->second.second, argument});
          outputs[producer->second.first].push_back({index, producer->second.second, argument});
        }
        ++argument;
      }
      const auto inferred = OutputTypes(node, types);
      int32_t output = 0;
      for (const auto &value : node.output()) {
        const auto name = Text(value);
        if (!name.empty()) {
          Require(available.insert(name).second, "duplicate node output '" + name + "'");
          producers[name] = {index, output};
          if (!types.count(name)) {
            Require(inferred[output].has_tensor_type() || inferred[output].has_sequence_type() ||
                        inferred[output].has_map_type(),
                    "cannot infer type for '" + name + "' produced by " + Text(node.op_type()) +
                        "; populate graph value_info before serialization");
            types[name] = inferred[output];
          }
        }
        ++output;
      }
      ++index;
    }
    for (const auto &value : graph.output())
      Require(available.count(Text(value.name())), "unresolved graph output");
    // Empty names represent omitted optional node arguments and have no type.
    types[""] = TypeProto();
    auto result = buffer_.MakeTable(9);
    buffer_.Reference(result, 0, buffer_.Objects(graph.initializer(), [&](const auto &tensor) {
      return Tensor(tensor);
    }));
    buffer_.Reference(result, 1, buffer_.Objects(types, [&](const auto &entry) {
      auto value = buffer_.MakeTable(3);
      buffer_.Reference(value, 0, buffer_.String(entry.first));
      auto info = infos.find(entry.first);
      if (info != infos.end())
        buffer_.String(value, 1, info->second->doc_string());
      if (!entry.first.empty())
        buffer_.Reference(value, 2, Type(entry.second));
      return value.position;
    }));
    index = 0;
    Types scope = outer;
    scope.insert(types.begin(), types.end());
    for (const auto &entry : types)
      scope[entry.first] = entry.second;
    buffer_.Reference(result, 2, buffer_.Objects(graph.node(), [&](const auto &node) {
      auto value = buffer_.MakeTable(13);
      buffer_.String(value, 0, node.name());
      buffer_.String(value, 1, node.doc_string());
      buffer_.Reference(value, 2, buffer_.String(NormalizeDomain(Text(node.domain()))));
      buffer_.Scalar<int32_t>(value, 3, -1);
      buffer_.Scalar<uint32_t>(value, 4, index);
      buffer_.String(value, 5, node.op_type());
      buffer_.Reference(value, 8, buffer_.Strings(node.input()));
      buffer_.Reference(value, 9, buffer_.Strings(node.output()));
      buffer_.Reference(value, 10, buffer_.Objects(node.attribute(), [&](const auto &attr) {
        return Attribute(attr, scope, depth);
      }));
      buffer_.Reference(value, 11, buffer_.Scalars<int32_t>(InputArgCounts(node)));
      buffer_.Reference(value, 12, buffer_.Strings(implicit[index++]));
      return value.position;
    }));
    buffer_.Scalar<uint32_t>(result, 3, index);
    std::vector<uint32_t> indices(index);
    for (uint32_t i = 0; i < index; ++i)
      indices[i] = i;
    auto edges = [&](const std::vector<Edge> &values) {
      buffer_.Align(4);
      size_t position = buffer_.Allocate(4);
      buffer_.Put<uint32_t>(position, static_cast<uint32_t>(values.size()));
      for (const auto &edge : values) {
        size_t item = buffer_.Allocate(12);
        buffer_.Put<uint32_t>(item, edge.node);
        buffer_.Put<int32_t>(item + 4, edge.source);
        buffer_.Put<int32_t>(item + 8, edge.destination);
      }
      return position;
    };
    buffer_.Reference(result, 4, buffer_.Objects(indices, [&](uint32_t i) {
      auto value = buffer_.MakeTable(3);
      buffer_.Scalar<uint32_t>(value, 0, i);
      buffer_.Reference(value, 1, edges(inputs[i]));
      buffer_.Reference(value, 2, edges(outputs[i]));
      return value.position;
    }));
    auto names = [&](const auto &values) {
      return buffer_.Objects(values,
                             [&](const auto &value) { return buffer_.String(Text(value.name())); });
    };
    buffer_.Reference(result, 5, names(graph.input()));
    buffer_.Reference(result, 6, names(graph.output()));
    return result.position;
  }

  FlatBuffer buffer_;
  std::map<std::string, int64_t> domains_;
  int64_t onnx_opset_ = 0;
};

} // namespace

std::string SerializeModelToOrtFlatbuffers(const ModelProto &model,
                                           const SerializeOptions &options) {
  EXT_ENFORCE(options.max_serialized_size_bytes >= 0,
              "ORT serialization: max_serialized_size_bytes must be non-negative");
  EXT_ENFORCE(!options.skip_raw_data,
              "ORT serialization: skip_raw_data cannot produce an executable ORT model");
  EXT_ENFORCE(options.alignment >= 0 &&
                  (options.alignment == 0 || (options.alignment & (options.alignment - 1)) == 0),
              "ORT serialization: alignment must be zero or a positive power of two");
  EXT_ENFORCE(options.alignment <= INT32_MAX,
              "ORT serialization: alignment exceeds the FlatBuffers size limit");
  if (options.raw_data_callback || options.node_callback) {
    ModelProto copy;
    copy.CopyFrom(model);
    auto restore = ApplySerializeRawDataCallback(copy, options);
    return OrtWriter(options).Model(copy);
  }
  return OrtWriter(options).Model(model);
}

bool SerializeModelToOrtFlatbuffers(const ModelProto &model, std::string &output,
                                    const SerializeOptions &options) {
  output.clear();
  try {
    output = SerializeModelToOrtFlatbuffers(model, options);
    return true;
  } catch (const std::length_error &) {
    if (options.max_serialized_size_bytes > 0)
      return false;
    throw;
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
