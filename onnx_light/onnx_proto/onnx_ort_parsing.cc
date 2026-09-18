#include "onnx.h"
#include "onnx_ort_flatbuffers.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace {

void OrtCheck(bool condition, const char *message) {
  EXT_ENFORCE(condition, "ORT parsing: ", message);
}

class OrtReader {
public:
  OrtReader(std::span<const uint8_t> bytes, ParseOptions &options)
      : bytes_(bytes), options_(options), expansion_left_(bytes.size() * uint64_t{64}) {}

  ModelProto ReadModel() {
    Bounds(0, 8);
    OrtCheck(std::memcmp(bytes_.data() + 4, "ORTM", 4) == 0, "missing ORTM file identifier");
    size_t root = Offset(0);
    OrtCheck(root >= 8, "root overlaps the file header");
    Table session(*this, root, 4);
    const auto version = session.Text(0, true);
    OrtCheck(version == "4" || version == "5" || version == "6", "unsupported ORT format version");
    if (auto state = session.Object(2))
      SessionState(state);
    if (auto resolver = session.Object(3))
      KernelResolver(resolver);
    Table source(*this, session.Object(1, true), 10);
    ModelProto model;
    const int64_t ir = source.Scalar<int64_t>(0);
    OrtCheck(ir > 0, "invalid model IR version");
    model.set_ir_version(ir);
    auto imports = source.Vector(1, 4, true);
    OrtCheck(imports.count != 0, "missing opset imports");
    for (size_t i = 0; i < imports.count; ++i) {
      Table opset(*this, imports.Object(i), 2);
      auto domain = Copy(opset.Text(0));
      if (domain == "ai.onnx")
        domain.clear();
      const int64_t version_number = opset.Scalar<int64_t>(1);
      OrtCheck(version_number > 0 && version_number <= INT32_MAX, "invalid opset version");
      OrtCheck(opsets_.emplace(domain, version_number).second, "duplicate opset domain");
      auto *entry = model.add_opset_import();
      entry->set_domain(domain);
      entry->set_version(version_number);
    }
    model.set_producer_name(Copy(source.Text(2)));
    model.set_producer_version(Copy(source.Text(3)));
    model.set_domain(Copy(source.Text(4)));
    model.set_model_version(source.Scalar<int64_t>(5));
    model.set_doc_string(Copy(source.Text(6)));
    Graph(source.Object(7, true), *model.mutable_graph(), {});
    model.mutable_graph()->set_doc_string(Copy(source.Text(8)));
    auto metadata = source.Vector(9, 4);
    std::set<std::string> keys;
    for (size_t i = 0; i < metadata.count; ++i) {
      Table item(*this, metadata.Object(i), 2);
      auto key = Copy(item.Text(0, true));
      OrtCheck(keys.insert(key).second, "duplicate model metadata key");
      auto *entry = model.add_metadata_props();
      entry->set_key(key);
      entry->set_value(Copy(item.Text(1, true)));
    }
    return model;
  }

private:
  void Bounds(size_t position, size_t count) const {
    OrtCheck(position <= bytes_.size() && count <= bytes_.size() - position,
             "truncated or out-of-bounds FlatBuffer");
  }

  template <typename T> T Read(size_t position) const {
    Bounds(position, sizeof(T));
    if constexpr (std::is_floating_point_v<T>) {
      if constexpr (sizeof(T) == 4)
        return std::bit_cast<T>(Read<uint32_t>(position));
      else
        return std::bit_cast<T>(Read<uint64_t>(position));
    } else {
      using U = std::make_unsigned_t<T>;
      U value = 0;
      for (size_t i = 0; i < sizeof(T); ++i)
        value |= static_cast<U>(bytes_[position + i]) << (8 * i);
      if constexpr (std::is_signed_v<T>)
        return std::bit_cast<T>(value);
      else
        return value;
    }
  }

  size_t Offset(size_t position) const {
    const uint32_t offset = Read<uint32_t>(position);
    OrtCheck(offset != 0 && offset <= bytes_.size() - position, "invalid relative offset");
    const size_t target = position + offset;
    Bounds(target, 4);
    OrtCheck(target % 4 == 0, "unaligned object offset");
    return target;
  }

  void TensorPayloadLimit(uint64_t bytes) const {
    EXT_ENFORCE_LIMIT(options_.max_tensor_size_bytes == 0 ||
                          bytes <= static_cast<uint64_t>(options_.max_tensor_size_bytes),
                      "ORT parsing: allocation of ", bytes,
                      " bytes exceeds max_tensor_size_bytes=", options_.max_tensor_size_bytes);
  }

  // Bounds both materialized data and repeated validation of shared wire objects.
  void Expansion(uint64_t bytes) {
    EXT_ENFORCE_LIMIT(bytes <= expansion_left_,
                      "ORT parsing: excessive expansion of shared FlatBuffer objects");
    expansion_left_ -= bytes;
  }

  std::string Copy(std::string_view value) {
    Expansion(value.size());
    return std::string(value);
  }

  struct VectorView {
    OrtReader &reader;
    size_t data;
    size_t count;
    size_t width;

    size_t At(size_t index) const {
      OrtCheck(index < count, "vector index is out of range");
      return data + index * width;
    }

    size_t Object(size_t index) const {
      const size_t target = reader.Offset(At(index));
      OrtCheck(target >= data + count * width, "object overlaps its offset vector");
      return target;
    }
  };

  std::string_view String(size_t position) const {
    const uint32_t count = Read<uint32_t>(position);
    Bounds(position + 4, static_cast<size_t>(count) + 1);
    OrtCheck(bytes_[position + 4 + count] == 0, "string has no null terminator");
    return {reinterpret_cast<const char *>(bytes_.data() + position + 4), count};
  }

  class Table {
  public:
    Table(OrtReader &reader, size_t position, size_t fields)
        : reader_(reader), position_(position) {
      reader.Bounds(position, 4);
      OrtCheck(position % 4 == 0, "unaligned table");
      const int32_t delta = reader.Read<int32_t>(position);
      const int64_t vtable = static_cast<int64_t>(position) - delta;
      OrtCheck(delta != 0 && vtable >= 0 && static_cast<uint64_t>(vtable) < reader.bytes_.size(),
               "invalid signed vtable offset");
      vtable_ = static_cast<size_t>(vtable);
      OrtCheck(vtable_ % 2 == 0, "unaligned vtable");
      vtable_size_ = reader.Read<uint16_t>(vtable_);
      object_size_ = reader.Read<uint16_t>(vtable_ + 2);
      OrtCheck(vtable_size_ >= 4 && vtable_size_ % 2 == 0 && object_size_ >= 4,
               "invalid vtable or table size");
      reader.Bounds(vtable_, vtable_size_);
      reader.Bounds(position_, object_size_);
      OrtCheck(vtable_ + vtable_size_ <= position_ || vtable_ >= position_ + object_size_,
               "vtable overlaps its table");
      reader.Expansion(16 + vtable_size_);
      for (size_t entry = 4; entry < vtable_size_; entry += 2) {
        const auto offset = reader.Read<uint16_t>(vtable_ + entry);
        OrtCheck(offset == 0 || (offset >= 4 && offset < object_size_),
                 "field offset exceeds its table");
        OrtCheck((entry - 4) / 2 < fields || offset == 0, "unsupported FlatBuffer schema field");
      }
      EXT_ENFORCE_LIMIT(reader.depth_ < reader.options_.max_recursion_depth,
                        "ORT parsing: max_recursion_depth exceeded");
      OrtCheck(!reader.active_.count(position), "cyclic FlatBuffer table references");
      reader.active_.insert(position);
      ++reader.depth_;
    }

    ~Table() {
      --reader_.depth_;
      reader_.active_.erase(position_);
    }
    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;

    size_t Field(size_t field, size_t width) const {
      if (field >= (vtable_size_ - 4) / 2)
        return 0;
      const uint16_t offset = reader_.Read<uint16_t>(vtable_ + 4 + 2 * field);
      if (!offset)
        return 0;
      OrtCheck(width <= object_size_ - offset, "field extends past its table");
      const size_t position = position_ + offset;
      OrtCheck(position % std::min<size_t>(width, 8) == 0, "unaligned scalar field");
      return position;
    }

    template <typename T> T Scalar(size_t field, T fallback = T{}) const {
      const size_t position = Field(field, sizeof(T));
      return position ? reader_.Read<T>(position) : fallback;
    }

    size_t Object(size_t field, bool required = false) const {
      const size_t position = Field(field, 4);
      OrtCheck(position != 0 || !required, "missing required table field");
      if (!position)
        return 0;
      const size_t target = reader_.Offset(position);
      OrtCheck(target >= position_ + object_size_, "object overlaps its containing table");
      return target;
    }

    std::string_view Text(size_t field, bool required = false) const {
      const size_t target = Object(field, required);
      return target ? reader_.String(target) : std::string_view();
    }

    VectorView Vector(size_t field, size_t width, bool required = false,
                      size_t alignment = 0) const {
      const size_t target = Object(field, required);
      if (!target)
        return {reader_, 0, 0, width};
      const uint32_t count = reader_.Read<uint32_t>(target);
      const size_t data = target + 4;
      OrtCheck(width != 0 && count <= (reader_.bytes_.size() - data) / width,
               "truncated or overflowing vector");
      OrtCheck(data % (alignment ? alignment : std::min<size_t>(width, 8)) == 0,
               "unaligned vector elements");
      return {reader_, data, count, width};
    }

  private:
    OrtReader &reader_;
    size_t position_;
    size_t vtable_;
    size_t vtable_size_;
    size_t object_size_;
  };

  void Shape(size_t position, TensorShapeProto &shape) {
    Table table(*this, position, 1);
    auto dims = table.Vector(0, 4);
    Expansion(dims.count * uint64_t{sizeof(TensorShapeProto::Dimension)});
    for (size_t i = 0; i < dims.count; ++i) {
      Table dim(*this, dims.Object(i), 2);
      auto *output = shape.add_dim();
      output->set_denotation(Copy(dim.Text(1)));
      if (auto value_position = dim.Object(0)) {
        Table value(*this, value_position, 3);
        switch (value.Scalar<int8_t>(0)) {
        case 0:
          break;
        case 1: {
          const auto size = value.Scalar<int64_t>(1);
          OrtCheck(size >= 0, "negative tensor dimension");
          output->set_dim_value(size);
          break;
        }
        case 2:
          output->set_dim_param(Copy(value.Text(2, true)));
          break;
        default:
          OrtCheck(false, "invalid dimension kind");
        }
      }
    }
  }

  static void DataType(int type) {
    OrtCheck(type >= 1 && type <= 20, "unsupported tensor element type");
  }

  void Type(size_t position, TypeProto &type) {
    Table table(*this, position, 3);
    type.set_denotation(Copy(table.Text(0)));
    const size_t value = table.Object(2, true);
    switch (table.Scalar<uint8_t>(1)) {
    case 1: {
      Table tensor(*this, value, 2);
      const int element = tensor.Scalar<int32_t>(0);
      DataType(element);
      auto *output = type.mutable_tensor_type();
      output->set_elem_type(static_cast<TensorProto::DataType>(element));
      if (auto shape = tensor.Object(1))
        Shape(shape, *output->mutable_shape());
      break;
    }
    case 2: {
      Table sequence(*this, value, 1);
      Type(sequence.Object(0, true), *type.mutable_sequence_type()->mutable_elem_type());
      break;
    }
    case 3: {
      Table map(*this, value, 2);
      const int key = map.Scalar<int32_t>(0);
      OrtCheck(key == 8 || (key >= 2 && key <= 7) || key == 12 || key == 13,
               "invalid map key type");
      auto *output = type.mutable_map_type();
      output->set_key_type(static_cast<TensorProto::DataType>(key));
      Type(map.Object(1, true), *output->mutable_value_type());
      break;
    }
    default:
      OrtCheck(false, "unsupported type union (optional, sparse and opaque types are unavailable)");
    }
  }

  void ValueInfo(size_t position, ValueInfoProto &value) {
    Table table(*this, position, 3);
    value.set_name(Copy(table.Text(0, true)));
    value.set_doc_string(Copy(table.Text(1)));
    if (auto type = table.Object(2))
      Type(type, *value.mutable_type());
    else
      OrtCheck(value.name().empty(), "named graph value has no type");
  }

  void Tensor(size_t position, TensorProto &tensor) {
    Table table(*this, position, 7);
    OrtCheck(table.Scalar<int64_t>(6, -1) == -1,
             "external ORT tensor offsets and companion .ort.data files are unsupported");
    const int type = table.Scalar<int32_t>(3);
    DataType(type);
    auto dims = table.Vector(2, 8, true);
    Expansion(dims.count * uint64_t{8});
    uint64_t elements = 1;
    bool zero = false;
    for (size_t i = 0; i < dims.count; ++i) {
      const auto dim = Read<int64_t>(dims.At(i));
      OrtCheck(dim >= 0, "negative initializer dimension");
      zero |= dim == 0;
    }
    if (zero)
      elements = 0;
    else
      for (size_t i = 0; i < dims.count; ++i) {
        const uint64_t dim = static_cast<uint64_t>(Read<int64_t>(dims.At(i)));
        OrtCheck(elements <= std::numeric_limits<uint64_t>::max() / dim,
                 "tensor element count overflow");
        elements *= dim;
      }
    auto raw = table.Vector(4, 1);
    auto strings = table.Vector(5, 4);
    TensorPayloadLimit(raw.count);
    if (type == TensorProto::STRING) {
      OrtCheck(raw.count == 0 && strings.count == elements, "string tensor payload size mismatch");
      Expansion(strings.count * uint64_t{sizeof(utils::String)});
      uint64_t total = 0;
      for (size_t i = 0; i < strings.count; ++i) {
        const auto value = String(strings.Object(i));
        OrtCheck(value.size() <= std::numeric_limits<uint64_t>::max() - total,
                 "string tensor size overflow");
        total += value.size();
      }
      TensorPayloadLimit(total);
      Expansion(total);
    } else {
      static constexpr uint64_t widths[] = {0, 4, 1, 1, 2,  2, 4, 8, 0, 1, 2,
                                            8, 4, 8, 8, 16, 2, 1, 1, 1, 1};
      OrtCheck(elements <= std::numeric_limits<uint64_t>::max() / widths[type],
               "tensor byte size overflow");
      const uint64_t expected = elements * widths[type];
      TensorPayloadLimit(expected);
      OrtCheck(strings.count == 0 && raw.count == expected, "numeric tensor payload size mismatch");
      if (!options_.skip_raw_data) {
        const uint64_t padding = expected && options_.alignment > 1 ? options_.alignment - 1 : 0;
        OrtCheck(padding <= std::numeric_limits<size_t>::max() - expected,
                 "aligned tensor allocation overflow");
        Expansion(expected + padding);
      }
    }
    // All dimensions and payload extents are checked before allocating fields.
    tensor.set_name(Copy(table.Text(0)));
    tensor.set_doc_string(Copy(table.Text(1)));
    tensor.set_data_type(static_cast<TensorProto::DataType>(type));
    for (size_t i = 0; i < dims.count; ++i)
      tensor.add_dims(Read<int64_t>(dims.At(i)));
    if (type == TensorProto::STRING) {
      for (size_t i = 0; i < strings.count; ++i)
        tensor.add_string_data(std::string(String(strings.Object(i))));
    } else if (!options_.skip_raw_data) {
      auto &data = tensor.ref_raw_data();
      data.resize_aligned(raw.count, static_cast<size_t>(options_.alignment));
      if (raw.count)
        std::memcpy(data.data(), bytes_.data() + raw.data, raw.count);
    }
  }

  void SparseTensor(size_t position, SparseTensorProto &tensor) {
    Table table(*this, position, 3);
    auto dims = table.Vector(2, 8, true);
    Expansion(dims.count * uint64_t{8});
    for (size_t i = 0; i < dims.count; ++i) {
      const auto dim = Read<int64_t>(dims.At(i));
      OrtCheck(dim >= 0, "negative sparse tensor dimension");
      tensor.add_dims(dim);
    }
    Tensor(table.Object(0, true), *tensor.mutable_values());
    Tensor(table.Object(1, true), *tensor.mutable_indices());
    const auto &values = tensor.values();
    const auto &indices = tensor.indices();
    OrtCheck(values.dims().size() == 1 && indices.data_type() == TensorProto::INT64,
             "unsupported sparse tensor values or indices");
    OrtCheck(
        (indices.dims().size() == 1 || indices.dims().size() == 2) &&
            indices.dims()[0] == values.dims()[0] &&
            (indices.dims().size() != 2 || indices.dims()[1] == static_cast<int64_t>(dims.count)),
        "inconsistent sparse tensor index shape");
    uint64_t dense_size = 1;
    if (std::find(tensor.dims().begin(), tensor.dims().end(), 0) != tensor.dims().end()) {
      dense_size = 0;
    } else {
      for (int64_t dimension : tensor.dims()) {
        OrtCheck(dense_size <=
                     std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension),
                 "sparse tensor element count overflow");
        dense_size *= static_cast<uint64_t>(dimension);
      }
    }
    const uint64_t nnz = static_cast<uint64_t>(values.dims()[0]);
    OrtCheck(nnz <= dense_size, "too many sparse tensor values");
    // Validates source indices even when skip_raw_data omits their owned copy.
    Table indices_table(*this, table.Object(1, true), 7);
    auto raw = indices_table.Vector(4, 1);
    Expansion(raw.count);
    if (indices.dims().size() == 1) {
      int64_t previous = -1;
      for (uint64_t i = 0; i < nnz; ++i) {
        const auto index = Read<int64_t>(raw.data + static_cast<size_t>(i) * 8);
        OrtCheck(index >= 0 && static_cast<uint64_t>(index) < dense_size && index > previous,
                 "sparse tensor indices are out of bounds, duplicated or unsorted");
        previous = index;
      }
    } else {
      for (uint64_t row = 0; row < nnz; ++row) {
        bool greater = row == 0;
        bool equal = true;
        for (size_t axis = 0; axis < dims.count; ++axis) {
          const size_t offset = (static_cast<size_t>(row) * dims.count + axis) * 8;
          const auto index = Read<int64_t>(raw.data + offset);
          OrtCheck(index >= 0 && index < tensor.dims()[axis],
                   "sparse tensor coordinate is out of bounds");
          if (row && equal) {
            const auto previous = Read<int64_t>(raw.data + offset - dims.count * 8);
            if (index != previous) {
              greater = index > previous;
              equal = false;
            }
          }
        }
        OrtCheck(greater, "sparse tensor coordinates are duplicated or unsorted");
      }
    }
  }

  void Attribute(size_t position, AttributeProto &attribute, const std::set<std::string> &scope) {
    Table table(*this, position, 13);
    auto name = table.Text(0, true);
    OrtCheck(!name.empty(), "empty attribute name");
    attribute.set_name(Copy(name));
    attribute.set_doc_string(Copy(table.Text(1)));
    const int type = table.Scalar<int32_t>(2);
    OrtCheck(type >= 1 && type <= 10, "unsupported attribute type");
    attribute.set_type(static_cast<AttributeProto::AttributeType>(type));
    switch (type) {
    case 1:
      attribute.set_f(table.Scalar<float>(3));
      break;
    case 2:
      attribute.set_i(table.Scalar<int64_t>(4));
      break;
    case 3:
      attribute.set_s(Copy(table.Text(5, true)));
      break;
    case 4:
      Tensor(table.Object(6, true), *attribute.mutable_t());
      break;
    case 5:
      Graph(table.Object(7, true), *attribute.mutable_g(), scope);
      break;
    case 6: {
      auto values = table.Vector(8, 4, true);
      Expansion(values.count * uint64_t{4});
      for (size_t i = 0; i < values.count; ++i)
        attribute.add_floats(Read<float>(values.At(i)));
      break;
    }
    case 7: {
      auto values = table.Vector(9, 8, true);
      Expansion(values.count * uint64_t{8});
      for (size_t i = 0; i < values.count; ++i)
        attribute.add_ints(Read<int64_t>(values.At(i)));
      break;
    }
    case 8: {
      auto values = table.Vector(10, 4, true);
      Expansion(values.count * uint64_t{sizeof(utils::String)});
      uint64_t total = 0;
      for (size_t i = 0; i < values.count; ++i)
        total += String(values.Object(i)).size();
      Expansion(total);
      for (size_t i = 0; i < values.count; ++i)
        attribute.add_strings(std::string(String(values.Object(i))));
      break;
    }
    case 9: {
      auto values = table.Vector(11, 4, true);
      Expansion(values.count * uint64_t{sizeof(TensorProto)});
      for (size_t i = 0; i < values.count; ++i)
        Tensor(values.Object(i), *attribute.add_tensors());
      break;
    }
    case 10: {
      auto values = table.Vector(12, 4, true);
      Expansion(values.count * uint64_t{sizeof(GraphProto)});
      for (size_t i = 0; i < values.count; ++i)
        Graph(values.Object(i), *attribute.add_graphs(), scope);
      break;
    }
    }
  }

  std::vector<std::string> Names(const Table &table, size_t field, bool required = false) {
    auto values = table.Vector(field, 4, required);
    Expansion(values.count * uint64_t{sizeof(std::string)});
    std::vector<std::string> result;
    result.reserve(values.count);
    for (size_t i = 0; i < values.count; ++i)
      result.push_back(Copy(String(values.Object(i))));
    return result;
  }

  struct NodeInfo {
    uint32_t index;
    std::vector<std::string> implicit;
  };

  NodeInfo Node(size_t position, NodeProto &node, const std::set<std::string> &scope) {
    Table table(*this, position, 13);
    OrtCheck(table.Scalar<int32_t>(6) == 0, "runtime-fused nodes cannot be represented as ONNX");
    node.set_name(Copy(table.Text(0)));
    node.set_doc_string(Copy(table.Text(1)));
    auto domain = Copy(table.Text(2));
    if (domain == "ai.onnx")
      domain.clear();
    auto opset = opsets_.find(domain);
    OrtCheck(opset != opsets_.end(), "node domain has no opset import");
    const int32_t since = table.Scalar<int32_t>(3);
    OrtCheck(since == -1 || (since > 0 && since <= opset->second), "invalid node schema version");
    node.set_domain(domain);
    auto op = table.Text(5, true);
    OrtCheck(!op.empty(), "empty operator type");
    node.set_op_type(Copy(op));
    table.Text(7); // Execution-provider placement is not executable ONNX data.
    auto inputs = Names(table, 8, true);
    auto outputs = Names(table, 9, true);
    for (const auto &input : inputs)
      node.add_input(input);
    for (const auto &output : outputs)
      node.add_output(output);
    auto attributes = table.Vector(10, 4);
    Expansion(attributes.count * uint64_t{sizeof(AttributeProto)});
    std::set<std::string> names;
    for (size_t i = 0; i < attributes.count; ++i) {
      auto *attribute = node.add_attribute();
      Attribute(attributes.Object(i), *attribute, scope);
      OrtCheck(names.insert(std::string(attribute->name())).second, "duplicate attribute name");
    }
    auto counts = table.Vector(11, 4, true);
    Expansion(counts.count * uint64_t{4});
    uint64_t total = 0;
    for (size_t i = 0; i < counts.count; ++i) {
      const auto count = Read<int32_t>(counts.At(i));
      OrtCheck(count >= 0, "negative node input argument count");
      total += static_cast<uint32_t>(count);
    }
    OrtCheck(total == inputs.size(), "input argument counts do not match node inputs");
    return {table.Scalar<uint32_t>(4), Names(table, 12, true)};
  }

  static void Captures(const GraphProto &graph, std::set<std::string> &result) {
    std::set<std::string> local;
    for (const auto &input : graph.input())
      local.insert(std::string(input.name()));
    for (const auto &tensor : graph.initializer())
      local.insert(std::string(tensor.name()));
    for (const auto &tensor : graph.sparse_initializer())
      local.insert(std::string(tensor.values().name()));
    std::set<std::string> used;
    for (const auto &node : graph.node()) {
      for (const auto &name : node.input())
        used.insert(name);
      for (const auto &name : node.output())
        local.insert(name);
      for (const auto &attribute : node.attribute()) {
        if (attribute.has_g())
          Captures(attribute.g(), used);
        for (const auto &nested : attribute.graphs())
          Captures(nested, used);
      }
    }
    for (const auto &output : graph.output())
      used.insert(std::string(output.name()));
    for (const auto &name : used)
      if (!name.empty() && !local.count(name))
        result.insert(name);
  }

  void Graph(size_t position, GraphProto &graph, const std::set<std::string> &outer) {
    Table table(*this, position, 9);
    graph.set_name("ort_graph_" + std::to_string(graph_count_++));
    if (auto optimizations = table.Object(8)) {
      Table saved(*this, optimizations, 1);
      OrtCheck(saved.Vector(0, 4).count == 0, "saved runtime optimizations are unsupported");
    }
    auto infos = table.Vector(1, 4);
    Expansion(infos.count * uint64_t{sizeof(ValueInfoProto)});
    std::map<std::string, ValueInfoProto> values;
    for (size_t i = 0; i < infos.count; ++i) {
      ValueInfoProto info;
      ValueInfo(infos.Object(i), info);
      const std::string name = info.name();
      OrtCheck(values.emplace(name, std::move(info)).second, "duplicate node argument name");
    }
    const auto input_names = Names(table, 5);
    const auto output_names = Names(table, 6);
    std::set<std::string> io_names, available, initializer_names;
    for (const auto &name : input_names) {
      OrtCheck(!name.empty() && available.insert(name).second, "empty or duplicate graph input");
      OrtCheck(values.count(name), "graph input has no node argument");
      graph.add_input()->CopyFrom(values.at(name));
      io_names.insert(name);
    }
    std::set<std::string> output_set;
    for (const auto &name : output_names) {
      OrtCheck(!name.empty() && output_set.insert(name).second, "empty or duplicate graph output");
      OrtCheck(values.count(name), "graph output has no node argument");
      graph.add_output()->CopyFrom(values.at(name));
      io_names.insert(name);
    }
    auto initializers = table.Vector(0, 4);
    Expansion(initializers.count * uint64_t{sizeof(TensorProto)});
    for (size_t i = 0; i < initializers.count; ++i) {
      auto *tensor = graph.add_initializer();
      Tensor(initializers.Object(i), *tensor);
      const std::string name = tensor->name();
      OrtCheck(!name.empty() && initializer_names.insert(name).second,
               "empty or duplicate initializer name");
      available.insert(name);
    }
    auto sparse = table.Vector(7, 4);
    Expansion(sparse.count * uint64_t{sizeof(SparseTensorProto)});
    for (size_t i = 0; i < sparse.count; ++i) {
      auto *tensor = graph.add_sparse_initializer();
      SparseTensor(sparse.Object(i), *tensor);
      const std::string name = tensor->values().name();
      OrtCheck(!name.empty() && initializer_names.insert(name).second,
               "empty or duplicate sparse initializer name");
      available.insert(name);
    }
    const auto validate_initializer = [&](const std::string &name, TensorProto::DataType element,
                                          const auto &dims) {
      auto declared = values.find(name);
      if (declared == values.end())
        return;
      const auto &type = declared->second.type();
      OrtCheck(type.has_tensor_type() && type.tensor_type().elem_type() == element,
               "initializer type conflicts with its node argument");
      if (type.tensor_type().has_shape()) {
        const auto &shape = type.tensor_type().shape().dim();
        OrtCheck(shape.size() == dims.size(), "initializer rank conflicts with its node argument");
        for (size_t i = 0; i < dims.size(); ++i)
          OrtCheck(!shape[i].has_dim_value() || shape[i].dim_value() == dims[i],
                   "initializer shape conflicts with its node argument");
      }
    };
    for (const auto &tensor : graph.initializer())
      validate_initializer(std::string(tensor.name()), tensor.data_type(), tensor.dims());
    for (const auto &tensor : graph.sparse_initializer())
      validate_initializer(std::string(tensor.values().name()), tensor.values().data_type(),
                           tensor.dims());
    for (auto &[name, value] : values)
      if (!name.empty() && !io_names.count(name))
        *graph.add_value_info() = std::move(value);
    auto nodes = table.Vector(2, 4);
    Expansion(nodes.count * uint64_t{sizeof(NodeProto) + sizeof(NodeInfo)});
    const uint32_t max_index = table.Scalar<uint32_t>(3);
    std::map<std::string, size_t> producers;
    std::set<std::string> scope = outer;
    scope.insert(available.begin(), available.end());
    for (size_t i = 0; i < nodes.count; ++i) {
      Table node(*this, nodes.Object(i), 13);
      for (const auto &output : Names(node, 9, true)) {
        if (output.empty())
          continue;
        OrtCheck(!available.count(output) && producers.emplace(output, i).second,
                 "duplicate node output");
        scope.insert(output);
      }
    }
    std::vector<NodeInfo> metadata;
    metadata.reserve(nodes.count);
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < nodes.count; ++i) {
      metadata.push_back(Node(nodes.Object(i), *graph.add_node(), scope));
      OrtCheck(metadata.back().index < max_index &&
                   indices.emplace(metadata.back().index, i).second,
               "duplicate or out-of-range node index");
      for (const auto &name : graph.node()[i].input())
        OrtCheck(values.count(name) || name.empty(), "node input has no node argument");
      for (const auto &name : graph.node()[i].output())
        OrtCheck(values.count(name) || name.empty(), "node output has no node argument");
    }
    auto edges = table.Vector(4, 4);
    std::set<uint32_t> edge_nodes;
    for (size_t i = 0; i < edges.count; ++i) {
      Table edge(*this, edges.Object(i), 3);
      const uint32_t index = edge.Scalar<uint32_t>(0);
      OrtCheck(indices.count(index) && edge_nodes.insert(index).second,
               "duplicate or unknown edge owner");
      for (size_t direction = 1; direction <= 2; ++direction) {
        auto ends = edge.Vector(direction, 12, false, 4);
        using EdgeKey = std::tuple<uint32_t, int32_t, int32_t>;
        Expansion(ends.count * uint64_t{sizeof(EdgeKey) + 4 * sizeof(void *)});
        std::set<EdgeKey> endpoints;
        for (size_t j = 0; j < ends.count; ++j) {
          const size_t endpoint = ends.At(j);
          const uint32_t other = Read<uint32_t>(endpoint);
          const int32_t source_arg = Read<int32_t>(endpoint + 4);
          const int32_t destination_arg = Read<int32_t>(endpoint + 8);
          OrtCheck(endpoints.emplace(other, source_arg, destination_arg).second,
                   "duplicate edge endpoint");
          OrtCheck(indices.count(other) && source_arg >= 0 && destination_arg >= 0,
                   "unknown edge node or unsupported control edge");
          const size_t source = indices.at(direction == 1 ? other : index);
          const size_t destination = indices.at(direction == 1 ? index : other);
          const auto &source_node = graph.node()[source];
          const auto &destination_node = graph.node()[destination];
          OrtCheck(static_cast<size_t>(source_arg) < source_node.output().size() &&
                       static_cast<size_t>(destination_arg) <
                           destination_node.input().size() + metadata[destination].implicit.size(),
                   "edge argument index is out of range");
          const auto expected =
              static_cast<size_t>(destination_arg) < destination_node.input().size()
                  ? std::string(destination_node.input()[destination_arg])
                  : metadata[destination]
                        .implicit[destination_arg - destination_node.input().size()];
          OrtCheck(source_node.output()[source_arg] == expected,
                   "edge disagrees with node arguments");
        }
      }
    }
    std::vector<std::set<size_t>> dependencies(nodes.count), consumers(nodes.count);
    for (size_t i = 0; i < nodes.count; ++i) {
      std::set<std::string> inputs(metadata[i].implicit.begin(), metadata[i].implicit.end());
      const auto &node = graph.node()[i];
      inputs.insert(node.input().begin(), node.input().end());
      for (const auto &attribute : node.attribute()) {
        if (attribute.has_g())
          Captures(attribute.g(), inputs);
        for (const auto &nested : attribute.graphs())
          Captures(nested, inputs);
      }
      for (const auto &name : inputs) {
        if (name.empty())
          continue;
        auto producer = producers.find(name);
        if (producer != producers.end()) {
          dependencies[i].insert(producer->second);
          consumers[producer->second].insert(i);
        } else {
          OrtCheck(available.count(name) || outer.count(name), "unresolved graph input or capture");
        }
      }
    }
    for (const auto &name : output_names)
      OrtCheck(available.count(name) || producers.count(name) || outer.count(name),
               "unresolved graph output");
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < nodes.count; ++i)
      if (dependencies[i].empty())
        ready.push(i);
    std::vector<size_t> order;
    order.reserve(nodes.count);
    while (!ready.empty()) {
      const size_t index = ready.top();
      ready.pop();
      order.push_back(index);
      for (size_t consumer : consumers[index]) {
        dependencies[consumer].erase(index);
        if (dependencies[consumer].empty())
          ready.push(consumer);
      }
    }
    OrtCheck(order.size() == nodes.count, "cyclic graph dependencies");
    auto unsorted = std::move(graph.ref_node());
    graph.clr_node();
    for (size_t i : order)
      *graph.add_node() = std::move(unsorted[i]);
  }

  void KernelResolver(size_t position) {
    Table resolver(*this, position, 1);
    auto operators = resolver.Vector(0, 4);
    for (size_t i = 0; i < operators.count; ++i) {
      Table op(*this, operators.Object(i), 2);
      op.Text(0, true);
      auto types = op.Vector(1, 4);
      for (size_t j = 0; j < types.count; ++j) {
        Table type(*this, types.Object(j), 2);
        type.Text(0, true);
        auto args = type.Vector(1, 4);
        for (size_t k = 0; k < args.count; ++k) {
          Table arg(*this, args.Object(k), 2);
          const auto kind = arg.Scalar<int8_t>(0);
          OrtCheck(kind == 0 || kind == 1, "invalid kernel argument kind");
          arg.Scalar<uint32_t>(1);
        }
      }
    }
  }

  void SessionState(size_t position) {
    Table state(*this, position, 2);
    if (auto kernels = state.Object(0)) {
      Table table(*this, kernels, 2);
      OrtCheck(table.Vector(0, 4).count == table.Vector(1, 8).count,
               "inconsistent deprecated kernel metadata");
    }
    auto subgraphs = state.Vector(1, 4);
    for (size_t i = 0; i < subgraphs.count; ++i) {
      Table subgraph(*this, subgraphs.Object(i), 2);
      subgraph.Text(0, true);
      SessionState(subgraph.Object(1, true));
    }
  }

  std::span<const uint8_t> bytes_;
  ParseOptions &options_;
  uint64_t expansion_left_;
  int32_t depth_ = 0;
  size_t graph_count_ = 0;
  std::set<size_t> active_;
  std::map<std::string, int64_t> opsets_;
};

void OrtCallbacks(GraphProto &graph, ParseOptions &options) {
  const auto tensor = [&](TensorProto &value) {
    if (options.raw_data_callback && value.has_raw_data()) {
      auto deleter = options.raw_data_callback(value, &graph);
      if (deleter)
        value.ref_raw_data().attach_deleter(std::move(deleter));
    }
  };
  for (auto &value : graph.ref_initializer())
    tensor(value);
  for (auto &value : graph.ref_sparse_initializer()) {
    tensor(value.ref_values());
    tensor(value.ref_indices());
  }
  for (auto &node : graph.ref_node())
    for (auto &attribute : node.ref_attribute()) {
      if (attribute.has_t())
        tensor(attribute.ref_t());
      for (auto &value : attribute.ref_tensors())
        tensor(value);
      if (attribute.has_g())
        OrtCallbacks(attribute.ref_g(), options);
      for (auto &nested : attribute.ref_graphs())
        OrtCallbacks(nested, options);
    }
  if (options.node_callback)
    for (auto &node : graph.ref_node())
      options.node_callback(node, graph);
}

} // namespace

void ParseModelFromOrtFlatbuffers(ModelProto &model, utils::BinaryStream &stream,
                                  ParseOptions &options) {
  EXT_ENFORCE(options.max_recursion_depth > 0, "ORT parsing: max_recursion_depth must be positive");
  EXT_ENFORCE(options.max_tensor_size_bytes >= 0,
              "ORT parsing: max_tensor_size_bytes must be non-negative");
  EXT_ENFORCE(options.alignment >= 0 && options.alignment <= INT32_MAX &&
                  (options.alignment == 0 || (options.alignment & (options.alignment - 1)) == 0),
              "ORT parsing: alignment must be zero or a supported positive power of two");
  EXT_ENFORCE(!stream.ExternalWeights(), "ORT parsing: external weights streams are unsupported");
  EXT_ENFORCE(options.tiny_external_data_threshold < 0 && options.io_trace == nullptr &&
                  !options.io_storage_kind.has_value(),
              "ORT parsing: external-data I/O policy options are unsupported");
  std::string storage;
  std::span<const uint8_t> bytes;
  const int64_t start = stream.tell();
  const int64_t size = stream.size();
  OrtCheck(start >= 0 && size >= start, "invalid stream extent");
  if (size == INT64_MAX) {
    const void *block = nullptr;
    int count = 0;
    while (stream.Next(&block, &count)) {
      OrtCheck(count > 0 && block != nullptr, "invalid stream block");
      OrtCheck(static_cast<uint64_t>(count) <= static_cast<uint64_t>(INT32_MAX) - storage.size(),
               "FlatBuffer exceeds the signed 32-bit format limit");
      storage.append(static_cast<const char *>(block), static_cast<size_t>(count));
    }
    if (const auto *fd = dynamic_cast<const utils::FdReadStream *>(&stream))
      OrtCheck(!fd->failed(), "failed to read the ORT stream");
    bytes = {reinterpret_cast<const uint8_t *>(storage.data()), storage.size()};
  } else {
    const uint64_t remaining = static_cast<uint64_t>(size - start);
    OrtCheck(remaining >= 8 && remaining <= INT32_MAX,
             "truncated FlatBuffer or signed 32-bit format size limit exceeded");
    stream.CanRead(remaining, "ORT parsing input");
    if (stream.CanNoCopy()) {
      bytes = {stream.read_bytes(static_cast<int64_t>(remaining)), static_cast<size_t>(remaining)};
    } else {
      storage.resize(static_cast<size_t>(remaining));
      stream.read_bytes(static_cast<int64_t>(remaining),
                        reinterpret_cast<uint8_t *>(storage.data()));
      bytes = {reinterpret_cast<const uint8_t *>(storage.data()), storage.size()};
    }
  }
  ModelProto decoded = OrtReader(bytes, options).ReadModel();
  OrtCallbacks(decoded.ref_graph(), options);
  model = std::move(decoded);
}

} // namespace ONNX_LIGHT_NAMESPACE
