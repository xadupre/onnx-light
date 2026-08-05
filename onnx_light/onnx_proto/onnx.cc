#include "onnx.h"
#include "blake3/blake3_hash.h"
#include "onnx_helper.h"
#include "stream_class.hpp"
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

// Parses external-data numeric metadata (offset/size) without creating a temporary std::string.
int64_t ParseInt64Fast(const utils::OptionalString &value) {
  int64_t out = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto parsed = std::from_chars(begin, end, out);
  EXT_ENFORCE(parsed.ec == std::errc() && parsed.ptr == end, "Unable to parse int64 from string ",
              ::ONNX_LIGHT_NAMESPACE::utils::quote_string((value).sv(), true), ".");
  return out;
}

bool TryParseInt64(const utils::OptionalString &value, int64_t &out) {
  const char *begin = value.data();
  const char *end = begin + value.size();
  auto parsed = std::from_chars(begin, end, out);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

std::string BaseDirFromStream(utils::BinaryStream &stream) {
  const auto parent_dir = [](const std::string &path) {
    return std::filesystem::path(path).parent_path().string();
  };
  if (auto *file_stream = dynamic_cast<utils::FileStream *>(&stream)) {
    return parent_dir(file_stream->file_path());
  }
  if (auto *mmap_stream = dynamic_cast<utils::MmapFileStream *>(&stream)) {
    return parent_dir(mmap_stream->file_path());
  }
  return "";
}

// Sets external_data metadata for one tensor using location/offset/length fields.
void SetTensorExternalMetadata(TensorProto &tensor, const std::string &location, int64_t offset) {
  tensor.clr_external_data();
  tensor.ref_data_location() = TensorProto::DataLocation::EXTERNAL;
  StringStringEntryProto *loc = tensor.add_external_data();
  loc->set_key("location");
  loc->set_value(location);
  StringStringEntryProto *off = tensor.add_external_data();
  off->set_key("offset");
  off->set_value(onnx_light_helpers::MakeString(offset));
  StringStringEntryProto *length = tensor.add_external_data();
  length->set_key("length");
  length->set_value(onnx_light_helpers::MakeString(tensor.raw_data_.size()));
}

// Rounds a file offset up to the nearest alignment boundary (no-op when alignment <= 1 or offset ==
// 0).
static inline int64_t align_up_offset(int64_t offset, int64_t alignment) {
  if (alignment <= 1 || offset <= 0)
    return offset;
  return ((offset + alignment - 1) / alignment) * alignment;
}

// Assigns external-data locations and offsets so each generated weights file stays under max size.
// When alignment > 0 each tensor's offset within its file is rounded up to the next multiple of
// alignment, matching the padding that SerializeToStream will write before the raw bytes.
std::vector<std::string> AssignExternalDataChunks(ModelProto &model, size_t threshold,
                                                  size_t max_external_file_size,
                                                  const std::string &external_file_prefix,
                                                  int64_t alignment = 0) {
  EXT_ENFORCE(max_external_file_size > 0, "max_external_file_size must be > 0.");
  onnx_light_helpers::ValidateAlignmentOption(alignment, "SerializeOptions.alignment");
  std::vector<std::string> locations;
  if (!model.has_graph()) {
    return locations;
  }
  size_t file_index = 0;
  int64_t file_offset = 0;
  std::string current_location = external_file_prefix + "_0.data";
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (!it->has_raw_data() || it->raw_data_.size() < threshold) {
      continue;
    }
    const size_t tensor_size = it->raw_data_.size();
    EXT_ENFORCE(tensor_size <= max_external_file_size, "Tensor raw_data is too large (",
                tensor_size, ") for max_external_file_size=", max_external_file_size, " name='",
                it->ref_name(), "'.");
    // Align the current offset within the file before checking the size cap.
    int64_t aligned_offset = align_up_offset(file_offset, alignment);
    if (aligned_offset > 0 && aligned_offset + static_cast<int64_t>(tensor_size) >
                                  static_cast<int64_t>(max_external_file_size)) {
      ++file_index;
      current_location = external_file_prefix + "_" + std::to_string(file_index) + ".data";
      file_offset = 0;
      aligned_offset = 0;
    }
    if (locations.empty() || locations.back() != current_location) {
      locations.push_back(current_location);
    }
    SetTensorExternalMetadata(*it, current_location, aligned_offset);
    file_offset = aligned_offset + static_cast<int64_t>(tensor_size);
  }
  return locations;
}

// In-memory stream that keeps the main proto and all external payloads in buffers.
class MemoryExternalWriteStream : public utils::BinaryWriteStream {
public:
  explicit MemoryExternalWriteStream(std::string default_location = "default.data")
      : default_location_(std::move(default_location)) {}

  void write_raw_bytes(const uint8_t *data, utils::offset_t n_bytes) override {
    if (n_bytes <= 0) {
      return;
    }
    const size_t start = main_buffer_.size();
    main_buffer_.resize(start + static_cast<size_t>(n_bytes));
    std::copy(data, data + n_bytes, main_buffer_.begin() + start);
  }

  int64_t size() const override { return static_cast<int64_t>(main_buffer_.size()); }

  const uint8_t *data() const override {
    return main_buffer_.empty() ? nullptr : main_buffer_.data();
  }

  bool ExternalWeights() const override { return true; }

  void write_raw_bytes_in_second_stream(const uint8_t *data, utils::offset_t n_bytes) override {
    write_raw_bytes_in_second_stream(data, n_bytes, default_location_);
  }

  void write_raw_bytes_in_second_stream(const uint8_t *data, utils::offset_t n_bytes,
                                        const std::string &location) override {
    if (n_bytes <= 0) {
      return;
    }
    auto &buffer = external_buffers_[location];
    const size_t start = buffer.size();
    buffer.resize(start + static_cast<size_t>(n_bytes));
    std::copy(data, data + n_bytes, buffer.begin() + start);
  }

  int64_t weights_size() const override { return weights_size_for_location(default_location_); }

  int64_t weights_size_for_location(const std::string &location) const override {
    auto it = external_buffers_.find(location);
    if (it == external_buffers_.end()) {
      return 0;
    }
    return static_cast<int64_t>(it->second.size());
  }

  void CopyOutputsTo(std::string &main,
                     std::unordered_map<std::string, std::string> &external) const {
    if (main_buffer_.empty()) {
      main.clear();
    } else {
      main.assign(reinterpret_cast<const char *>(main_buffer_.data()), main_buffer_.size());
    }
    external.clear();
    for (const auto &kv : external_buffers_) {
      if (kv.second.empty()) {
        external[kv.first] = "";
      } else {
        external[kv.first].assign(reinterpret_cast<const char *>(kv.second.data()),
                                  kv.second.size());
      }
    }
  }

  void pre_allocate_main(size_t n_bytes) { main_buffer_.reserve(n_bytes); }

private:
  std::string default_location_;
  std::vector<uint8_t> main_buffer_;
  std::unordered_map<std::string, std::vector<uint8_t>> external_buffers_;
};

// Mixes ``value`` into the running hash ``seed`` (boost-style mixing).
inline void HashCombine(uint64_t &seed, uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

} // namespace

// StringStringEntryProto

IMPLEMENT_PROTO(StringStringEntryProto)
SerializeSizeResult StringStringEntryProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                          SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, key)
  SIZE_FIELD(size, options, stream, value)
  return size;
}
void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                               SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key)
  WRITE_FIELD(options, stream, value)
}
bool StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, StringStringEntryProto) //
  READ_FIELD(options, stream, key)                    //
  READ_FIELD(options, stream, value)                  //
  READ_END(options, stream, StringStringEntryProto)   //  // NOLINT
  return true;
}
void StringStringEntryProto::PrintToStringStream(std::stringstream &ss,
                                                 utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value));
}

// TensorAnnotation
IMPLEMENT_PROTO(TensorAnnotation)
SerializeSizeResult TensorAnnotation::SerializeSize(utils::BinaryWriteStream &stream,
                                                    SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, tensor_name)
  SIZE_REPEATED_FIELD(size, options, stream, quant_parameter_tensor_names)
  return size;
}
void TensorAnnotation::SerializeToStream(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  WRITE_FIELD(options, stream, tensor_name)
  WRITE_REPEATED_FIELD(options, stream, quant_parameter_tensor_names)
}
bool TensorAnnotation::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TensorAnnotation)                      //
  READ_FIELD(options, stream, tensor_name)                           //
  READ_REPEATED_FIELD(options, stream, quant_parameter_tensor_names) //
  READ_END(options, stream, TensorAnnotation)                        //  // NOLINT
  return true;
}
void TensorAnnotation::PrintToStringStream(std::stringstream &ss,
                                           utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(tensor_name),
                                 NAME_EXIST_VALUE(quant_parameter_tensor_names));
}

// IntIntListEntryProto

IMPLEMENT_PROTO(IntIntListEntryProto)
SerializeSizeResult IntIntListEntryProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                        SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, key)
  SIZE_REPEATED_FIELD(size, options, stream, value)
  return size;
}
void IntIntListEntryProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key)
  WRITE_REPEATED_FIELD(options, stream, value)
}
bool IntIntListEntryProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, IntIntListEntryProto) //
  READ_FIELD(options, stream, key)                  //
  READ_REPEATED_FIELD(options, stream, value)       //
  READ_END(options, stream, IntIntListEntryProto)   //
  return true;
}
void IntIntListEntryProto::PrintToStringStream(std::stringstream &ss,
                                               utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value));
}

// DeviceConfigurationProto

IMPLEMENT_PROTO(DeviceConfigurationProto)
SerializeSizeResult DeviceConfigurationProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                            SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, num_devices)
  SIZE_REPEATED_FIELD(size, options, stream, device)
  return size;
}
void DeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, num_devices)
  WRITE_REPEATED_FIELD(options, stream, device)
}
bool DeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, DeviceConfigurationProto) //
  READ_FIELD(options, stream, name)                     //
  READ_FIELD(options, stream, num_devices)              //
  READ_REPEATED_FIELD(options, stream, device)          //
  READ_END(options, stream, DeviceConfigurationProto)   //
  return true;
}
void DeviceConfigurationProto::PrintToStringStream(std::stringstream &ss,
                                                   utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(num_devices),
                                 NAME_EXIST_VALUE(device));
}

// SimpleShardedDimProto

IMPLEMENT_PROTO(SimpleShardedDimProto)
SerializeSizeResult SimpleShardedDimProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                         SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, dim_value)
  SIZE_FIELD(size, options, stream, dim_param)
  SIZE_FIELD(size, options, stream, num_shards)
  return size;
}
void SimpleShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                              SerializeOptions &options) const {
  WRITE_FIELD(options, stream, dim_value)
  WRITE_FIELD(options, stream, dim_param)
  WRITE_FIELD(options, stream, num_shards)
}
bool SimpleShardedDimProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, SimpleShardedDimProto) //
  READ_FIELD(options, stream, dim_value)             //
  READ_FIELD(options, stream, dim_param)             //
  READ_FIELD(options, stream, num_shards)            //
  READ_END(options, stream, SimpleShardedDimProto)   //
  return true;
}
void SimpleShardedDimProto::PrintToStringStream(std::stringstream &ss,
                                                utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(dim_value),
                                 NAME_EXIST_VALUE(dim_param), NAME_EXIST_VALUE(num_shards));
}

// ShardedDimProto

IMPLEMENT_PROTO(ShardedDimProto)
SerializeSizeResult ShardedDimProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                   SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, axis)
  SIZE_REPEATED_FIELD(size, options, stream, simple_sharding)
  return size;
}
void ShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                        SerializeOptions &options) const {
  WRITE_FIELD(options, stream, axis)
  WRITE_REPEATED_FIELD(options, stream, simple_sharding)
}

bool ShardedDimProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, ShardedDimProto)          //
  READ_FIELD(options, stream, axis)                     //
  READ_REPEATED_FIELD(options, stream, simple_sharding) //
  READ_END(options, stream, ShardedDimProto)            //
  return true;
}
void ShardedDimProto::PrintToStringStream(std::stringstream &ss,
                                          utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(axis),
                                 NAME_EXIST_VALUE(simple_sharding));
}

// ShardingSpecProto

IMPLEMENT_PROTO(ShardingSpecProto)
SerializeSizeResult ShardingSpecProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, tensor_name)
  SIZE_REPEATED_FIELD(size, options, stream, device)
  SIZE_REPEATED_FIELD(size, options, stream, index_to_device_group_map)
  SIZE_REPEATED_FIELD(size, options, stream, sharded_dim)
  return size;
}
void ShardingSpecProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, tensor_name)
  WRITE_REPEATED_FIELD(options, stream, device)
  WRITE_REPEATED_FIELD(options, stream, index_to_device_group_map)
  WRITE_REPEATED_FIELD(options, stream, sharded_dim)
}
bool ShardingSpecProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, ShardingSpecProto)                  //
  READ_FIELD(options, stream, tensor_name)                        //
  READ_REPEATED_FIELD(options, stream, device)                    //
  READ_REPEATED_FIELD(options, stream, index_to_device_group_map) //
  READ_REPEATED_FIELD(options, stream, sharded_dim)               //
  READ_END(options, stream, ShardingSpecProto)                    //  // NOLINT
  return true;
}
void ShardingSpecProto::PrintToStringStream(std::stringstream &ss,
                                            utils::PrintOptions &options) const {
  write_proto_into_vector_string(
      ss, options, NAME_EXIST_VALUE(tensor_name), NAME_EXIST_VALUE(device),
      NAME_EXIST_VALUE(index_to_device_group_map), NAME_EXIST_VALUE(sharded_dim));
}

// NodeDeviceConfigurationProto

IMPLEMENT_PROTO(NodeDeviceConfigurationProto)
SerializeSizeResult NodeDeviceConfigurationProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                                SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, configuration_id)
  SIZE_REPEATED_FIELD(size, options, stream, sharding_spec)
  SIZE_FIELD(size, options, stream, pipeline_stage)
  return size;
}
void NodeDeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  WRITE_FIELD(options, stream, configuration_id)
  WRITE_REPEATED_FIELD(options, stream, sharding_spec)
  WRITE_FIELD(options, stream, pipeline_stage)
}
bool NodeDeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream,
                                                   ParseOptions &options) {
  READ_BEGIN(options, stream, NodeDeviceConfigurationProto) //
  READ_FIELD(options, stream, configuration_id)             //
  READ_REPEATED_FIELD(options, stream, sharding_spec)       //
  READ_FIELD(options, stream, pipeline_stage)               //
  READ_END(options, stream, NodeDeviceConfigurationProto)   //
  return true;
}
void NodeDeviceConfigurationProto::PrintToStringStream(std::stringstream &ss,
                                                       utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(configuration_id),
                                 NAME_EXIST_VALUE(sharding_spec), NAME_EXIST_VALUE(pipeline_stage));
}

// OperatorSetIdProto

IMPLEMENT_PROTO(OperatorSetIdProto)
SerializeSizeResult OperatorSetIdProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                      SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD_EMPTY(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, version)
  return size;
}
void OperatorSetIdProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                           SerializeOptions &options) const {
  WRITE_FIELD_EMPTY(options, stream, domain)
  WRITE_FIELD(options, stream, version)
}
bool OperatorSetIdProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, OperatorSetIdProto) //
  READ_FIELD(options, stream, domain)             //
  READ_FIELD(options, stream, version)            //
  READ_END(options, stream, OperatorSetIdProto)   //
  return true;
}
void OperatorSetIdProto::PrintToStringStream(std::stringstream &ss,
                                             utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(version));
}

// TensorShapeProto::Dimension

IMPLEMENT_PROTO(TensorShapeProto::Dimension)
SerializeSizeResult TensorShapeProto::Dimension::SerializeSize(utils::BinaryWriteStream &stream,
                                                               SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, dim_value)
  SIZE_FIELD(size, options, stream, dim_param)
  SIZE_FIELD(size, options, stream, denotation)
  return size;
}
void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream,
                                                    SerializeOptions &options) const {
  WRITE_FIELD(options, stream, dim_value)
  WRITE_FIELD(options, stream, dim_param)
  WRITE_FIELD(options, stream, denotation)
}
bool TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream,
                                                  ParseOptions &options) {
  READ_BEGIN(options, stream, TensorShapeProto::Dimension) //
  READ_FIELD(options, stream, dim_value)                   //
  READ_FIELD(options, stream, dim_param)                   //
  READ_FIELD(options, stream, denotation)                  //
  READ_END(options, stream, TensorShapeProto::Dimension)   //
  return true;
}
void TensorShapeProto::Dimension::PrintToStringStream(std::stringstream &ss,
                                                      utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(dim_value),
                                 NAME_EXIST_VALUE(dim_param), NAME_EXIST_VALUE(denotation));
}

// TensorShapeProto

IMPLEMENT_PROTO(TensorShapeProto)
SerializeSizeResult TensorShapeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                    SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_REPEATED_FIELD(size, options, stream, dim)
  return size;
}
void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, dim)
}
bool TensorShapeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TensorShapeProto) //
  READ_REPEATED_FIELD(options, stream, dim)     //
  READ_END(options, stream, TensorShapeProto)   //
  return true;
}
void TensorShapeProto::PrintToStringStream(std::stringstream &ss,
                                           utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(dim));
}

// TensorProto::Segment

IMPLEMENT_PROTO(TensorProto::Segment)
SerializeSizeResult TensorProto::Segment::SerializeSize(utils::BinaryWriteStream &stream,
                                                        SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, begin)
  SIZE_FIELD(size, options, stream, end)
  return size;
}
void TensorProto::Segment::SerializeToStream(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  WRITE_FIELD(options, stream, begin)
  WRITE_FIELD(options, stream, end)
}
bool TensorProto::Segment::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TensorProto::Segment) //
  READ_FIELD(options, stream, begin)                //
  READ_FIELD(options, stream, end)                  //
  READ_END(options, stream, TensorProto::Segment)   //
  return true;
}
void TensorProto::Segment::PrintToStringStream(std::stringstream &ss,
                                               utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(begin), NAME_EXIST_VALUE(end));
}

// TensorProto

IMPLEMENT_PROTO(TensorProto)
SerializeSizeResult TensorProto::SerializeSize(utils::BinaryWriteStream &stream,
                                               SerializeOptions &options) const {
  SerializeSizeResult size;
  const bool write_external_raw_data = options.use_external_data_location && has_data_location() &&
                                       ref_data_location() == DataLocation::EXTERNAL &&
                                       stream.ExternalWeights();
  SIZE_REPEATED_FIELD(size, options, stream, dims)
  SIZE_ENUM_FIELD(size, options, stream, data_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, segment)
  SIZE_ENUM_FIELD(size, options, stream, data_location)
  SIZE_FIELD_NULL(size, options, stream, name)
  if (has_raw_data()) {
    if (!write_external_raw_data) {
      size += size_field_limit(stream, order_raw_data(), raw_data_, options);
    } else {
      size.add_tensor_data_size(static_cast<int64_t>(raw_data_.size()), options.raw_data_threshold);
    }
  }
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, external_data)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, double_data)
  SIZE_REPEATED_FIELD(size, options, stream, float_data)
  SIZE_REPEATED_FIELD(size, options, stream, int32_data)
  SIZE_REPEATED_FIELD(size, options, stream, int64_data)
  SIZE_REPEATED_FIELD(size, options, stream, uint64_data)
  SIZE_REPEATED_FIELD(size, options, stream, string_data)
  return size;
}
void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                    SerializeOptions &options) const {
  // Validation for external data.
  bool write_external_raw_data = false;
  const utils::OptionalString *external_location = nullptr;
  if (options.use_external_data_location && has_data_location() &&
      ref_data_location() == DataLocation::EXTERNAL && stream.ExternalWeights()) {
    bool has_location = false;
    bool has_size = false;
    bool has_offset = false;
    int64_t expected_offset = -1;
    for (size_t i = 0; i < ref_external_data().size(); ++i) {
      const StringStringEntryProto &entry = ref_external_data()[i];
      if (entry.ref_key() == "location") {
        EXT_ENFORCE(!entry.ref_value().empty(), "External data location must not be empty.");
        {
          std::filesystem::path loc_path(std::string(entry.ref_value()));
          auto normalized = loc_path.lexically_normal();
          auto parent = normalized.parent_path();
          EXT_ENFORCE(parent.empty() || parent == std::filesystem::path("."),
                      "External data location must be a filename with no folder, name='",
                      ref_name(), "', location='", entry.ref_value(), "'");
        }
        external_location = &entry.ref_value();
        has_location = true;
      } else if (entry.ref_key() == "size" || entry.ref_key() == "length") {
        int64_t size = entry.ref_value().toint64();
        if (size != static_cast<int64_t>(raw_data_.size())) {
          if (raw_data_.size() == 0) {
            EXT_THROW(
                "Tensor '", ref_name(),
                "' is marked EXTERNAL but its raw_data is empty while serializing external data. "
                "This usually means the model was loaded with load_external_data=False. "
                "Reload it with load_external_data=True, call load_external_data_for_model(), "
                "or use save_model_with_shared_external_data() to reuse the existing weights "
                "file.");
          }
          EXT_THROW("Size mismatch ", size, " != ", static_cast<int64_t>(raw_data_.size()),
                    " name='", ref_name(), "'");
        }
        has_size = true;
      } else if (entry.ref_key() == "offset") {
        expected_offset = entry.ref_value().toint64();
        has_offset = true;
      }
    }
    EXT_ENFORCE(has_location && has_size && has_offset,
                "External data is not fully specified. 'location', 'size', and 'offset' "
                "must be present in external_data, name='",
                ref_name(), "'");
    const int64_t current_offset =
        stream.weights_size_for_location(std::string(*(external_location)));
    // Write alignment padding before the tensor data if the expected offset is ahead of the
    // current write position.  This happens when PopulateExternalData (or
    // AssignExternalDataChunks) has rounded the offset up to an alignment boundary.
    // The padding is emitted from a small static zero buffer in fixed-size chunks rather than
    // a single allocation proportional to `padding`. This mirrors the mitigation in upstream
    // PR https://github.com/onnx/onnx/pull/8055 and prevents a crafted external-data offset
    // from triggering an unbounded allocation during serialization.
    if (expected_offset > current_offset) {
      const int64_t padding = expected_offset - current_offset;
      static constexpr size_t CHUNK = 128;
      static const uint8_t zeros[CHUNK] = {};
      for (int64_t written = 0; written < padding;) {
        const int64_t to_write = std::min(padding - written, static_cast<int64_t>(CHUNK));
        stream.write_raw_bytes_in_second_stream(zeros, to_write, *(external_location));
        written += to_write;
      }
    }
    EXT_ENFORCE(expected_offset == stream.weights_size_for_location(*(external_location)),
                "Offset mismatch ", expected_offset,
                " != ", stream.weights_size_for_location(*(external_location)), " name='",
                ref_name(), "'");
    // TODO Checks sparse initializer as well.
    write_external_raw_data = true;
  }
  WRITE_REPEATED_FIELD(options, stream, dims)
  WRITE_ENUM_FIELD(options, stream, data_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, segment)
  WRITE_ENUM_FIELD(options, stream, data_location)
  WRITE_FIELD_NULL(options, stream, name)
  if (has_raw_data()) {
    if (write_external_raw_data) {
      stream.write_raw_bytes_in_second_stream(raw_data_.data(),
                                              static_cast<utils::offset_t>(raw_data_.size()),
                                              std::string(*(external_location)));
    } else {
      write_field_limit(stream, order_raw_data(), raw_data_, options);
    }
  }
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, external_data)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, double_data)
  WRITE_REPEATED_FIELD(options, stream, float_data)
  WRITE_REPEATED_FIELD(options, stream, int32_data)
  WRITE_REPEATED_FIELD(options, stream, int64_data)
  WRITE_REPEATED_FIELD(options, stream, uint64_data)
  WRITE_REPEATED_FIELD(options, stream, string_data)
}
bool TensorProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TensorProto)                 //
  READ_REPEATED_FIELD(options, stream, dims)               //
  READ_ENUM_FIELD(options, stream, data_type)              //
  READ_OPTIONAL_PROTO_FIELD(options, stream, segment)      //
  READ_OPTIONAL_ENUM_FIELD(options, stream, data_location) //
  READ_FIELD(options, stream, name)                        //
  READ_FIELD(options, stream, doc_string)                  //
  else if (static_cast<int>(field_number.field_number) == order_raw_data()) {
    read_field_limit_parallel_nc(stream, static_cast<int>(field_number.wire_type), raw_data_,
                                 "raw_data", options);
    raw_data_was_set_ = true;
  } //
  READ_REPEATED_FIELD(options, stream, external_data)  //
  READ_REPEATED_FIELD(options, stream, metadata_props) //
  READ_REPEATED_FIELD(options, stream, double_data)    //
  READ_REPEATED_FIELD(options, stream, float_data)     //
  READ_REPEATED_FIELD(options, stream, int32_data)     //
  READ_REPEATED_FIELD(options, stream, int64_data)     //
  READ_REPEATED_FIELD(options, stream, uint64_data)    //
  READ_REPEATED_FIELD(options, stream, string_data)    //
  READ_END(options, stream, TensorProto)               //
                                         // After the reading, we need to check the data location.
  if (has_data_location() && ref_data_location() == DataLocation::EXTERNAL &&
      stream.ExternalWeights()) {
    utils::TwoFilesStream &two_stream = dynamic_cast<utils::TwoFilesStream &>(stream);
    offset_t offset = -1; // two_stream.second_tell();
    int64_t size = -1;
    std::string location;
    auto &external_data = ref_external_data();

    if (external_data.size() >= 3 && external_data[0].ref_key() == "location") {
      EXT_ENFORCE(!external_data[0].ref_value().empty(),
                  "External data location must not be empty.");
      {
        std::filesystem::path loc_path(std::string(external_data[0].ref_value()));
        auto normalized = loc_path.lexically_normal();
        auto parent = normalized.parent_path();
        EXT_ENFORCE(parent.empty() || parent == std::filesystem::path("."),
                    "External data location must be a filename with no folder, name='", ref_name(),
                    "', location='", std::string(external_data[0].ref_value()), "'");
      }
      location = std::string(external_data[0].ref_value());
      const StringStringEntryProto &entry1 = external_data[1];
      const StringStringEntryProto &entry2 = external_data[2];
      if (entry1.ref_key() == "offset" &&
          (entry2.ref_key() == "size" || entry2.ref_key() == "length")) {
        offset = ParseInt64Fast(entry1.ref_value());
        size = ParseInt64Fast(entry2.ref_value());
      } else if ((entry1.ref_key() == "size" || entry1.ref_key() == "length") &&
                 entry2.ref_key() == "offset") {
        size = ParseInt64Fast(entry1.ref_value());
        offset = ParseInt64Fast(entry2.ref_value());
      }
    }

    if (offset < 0 || size <= 0) {
      for (size_t i = 0; i < external_data.size(); ++i) {
        const StringStringEntryProto &entry = external_data[i];
        if (entry.ref_key() == "location") {
          EXT_ENFORCE(!entry.ref_value().empty(), "External data location must not be empty.");
          {
            std::filesystem::path loc_path(std::string(entry.ref_value()));
            auto normalized = loc_path.lexically_normal();
            auto parent = normalized.parent_path();
            EXT_ENFORCE(parent.empty() || parent == std::filesystem::path("."),
                        "External data location must be a filename with no folder, name='",
                        ref_name(), "', location='", entry.ref_value(), "'");
          }
          location = entry.ref_value();
          // Should check the value with the location of the second stream?
        } else if (entry.ref_key() == "length" || entry.ref_key() == "size") {
          size = ParseInt64Fast(entry.ref_value());
        } else if (entry.ref_key() == "offset") {
          offset = ParseInt64Fast(entry.ref_value());
        }
      }
    }
    EXT_ENFORCE(offset >= 0 && size > 0, "External data offset and size must be specified, name='",
                ref_name(), "'");
    if (options.max_tensor_size_bytes > 0 &&
        static_cast<int64_t>(size) > options.max_tensor_size_bytes) {
      EXT_THROW("TensorProto::ParseFromStream (external data): tensor '", ref_name(), "' requests ",
                size, " bytes which exceeds ParseOptions::max_tensor_size_bytes=",
                options.max_tensor_size_bytes,
                ". Increase max_tensor_size_bytes or set it to 0 to disable the limit.");
    }
    onnx_light_helpers::ValidateAlignmentOption(options.alignment, "ParseOptions.alignment");
    if (options.alignment > 1 && offset % options.alignment != 0) {
      std::ostringstream oss;
      oss << "Serialized external-data offset " << offset << " for tensor '" << ref_name()
          << "' (location '" << location
          << "') is incompatible with ParseOptions.alignment=" << options.alignment << ".";
      if (options.no_copy) {
        EXT_THROW(oss.str(), " no_copy=true forbids automatic realignment.");
      }
      std::cerr << "Warning: " << oss.str() << " Realigning tensor bytes into an aligned buffer."
                << std::endl;
    }
    if (options.no_copy) {
      std::shared_ptr<void> owner;
      const uint8_t *ptr = two_stream.borrow_weights_bytes(
          location, offset, size, static_cast<size_t>(std::max<int64_t>(options.alignment, 0)),
          owner);
      ref_raw_data().assign_borrowed(ptr, static_cast<size_t>(size), owner);
    } else {
      two_stream.set_active_weights_location(location);
      if (options.alignment > 1) {
        ref_raw_data().resize_aligned(static_cast<size_t>(size),
                                      static_cast<size_t>(options.alignment));
      } else {
        ref_raw_data().resize(size);
      }
      if (options.is_parallel()) {
        two_stream.set_active_weights_location(location);
        utils::DelayedBlock block;
        block.size = size;
        block.data = ref_raw_data().data();
        block.offset = offset;
        block.stream_id = 1; // The second stream is the weights stream.
        two_stream.ReadDelayedBlock(block);
      } else {
        two_stream.read_bytes_from_weights_stream(size, ref_raw_data().data(), offset);
      }
    }
  }
  // After raw_data (inline or external) has been resolved, gives the caller a chance to take
  // custom ownership of the tensor data and register a matching deleter.
  if (options.raw_data_callback && has_raw_data()) {
    std::function<void()> deleter = options.raw_data_callback(*this);
    if (deleter) {
      ref_raw_data().attach_deleter(std::move(deleter));
    }
  }
  if (options.tiny_external_data_threshold >= 0 && has_data_location() &&
      ref_data_location() == DataLocation::EXTERNAL && !stream.ExternalWeights()) {
    int64_t length = -1;
    for (const StringStringEntryProto &entry : ref_external_data()) {
      if ((entry.ref_key() == "length" || entry.ref_key() == "size") &&
          TryParseInt64(entry.ref_value(), length)) {
        break;
      }
    }
    if (length >= 0 && length < options.tiny_external_data_threshold) {
      const std::string base_dir = BaseDirFromStream(stream);
      if (!base_dir.empty()) {
        LoadExternalData(base_dir);
        ref_data_location() = DataLocation::DEFAULT;
        clr_external_data();
      }
    }
  }
  return true;
}
void TensorProto::LoadExternalData(const std::string &base_dir) {
  EXT_ENFORCE(has_data_location() && ref_data_location() == DataLocation::EXTERNAL,
              "TensorProto::LoadExternalData requires data_location == EXTERNAL, name='",
              ref_name(), "'.");
  std::string location;
  int64_t offset = 0;
  int64_t length = -1;
  for (const StringStringEntryProto &entry : ref_external_data()) {
    const utils::OptionalString &key = entry.ref_key();
    if (key == "location") {
      location = entry.ref_value();
    } else if (key == "offset") {
      offset = ParseInt64Fast(entry.ref_value());
    } else if (key == "length" || key == "size") {
      length = ParseInt64Fast(entry.ref_value());
    }
  }
  EXT_ENFORCE(!location.empty(),
              "TensorProto::LoadExternalData missing 'location' entry in external_data, name='",
              ref_name(), "'.");
  // Validate that location does not escape the base directory (path traversal).
  std::filesystem::path loc_path(location);
  std::filesystem::path loc_normal = loc_path.lexically_normal();
  EXT_ENFORCE(!loc_normal.has_root_path() && !loc_normal.empty() && *loc_normal.begin() != "..",
              "TensorProto::LoadExternalData: location '", location,
              "' must be a relative path that does not escape the base directory.");
  std::filesystem::path data_path =
      base_dir.empty() ? loc_normal : std::filesystem::path(base_dir) / loc_normal;
  // Reject symlinks: external data must be a regular file, never a symbolic
  // link, otherwise a malicious model could read arbitrary files on disk.
  EXT_ENFORCE(!std::filesystem::is_symlink(data_path),
              "TensorProto::LoadExternalData: external data file '", data_path.string(),
              "' is a symbolic link, which is not allowed, for tensor '", ref_name(), "'.");
  // Verify canonical containment to catch symlinks in any parent component
  // that would resolve outside the base directory.
  if (!base_dir.empty()) {
    std::error_code ec;
    std::filesystem::path canonical_data = std::filesystem::weakly_canonical(data_path, ec);
    EXT_ENFORCE(!ec, "TensorProto::LoadExternalData: external data path '", data_path.string(),
                "' could not be canonicalized: ", ec.message());
    std::filesystem::path canonical_base =
        std::filesystem::weakly_canonical(std::filesystem::path(base_dir), ec);
    EXT_ENFORCE(!ec, "TensorProto::LoadExternalData: base directory '", base_dir,
                "' could not be canonicalized: ", ec.message());
    std::filesystem::path::string_type base_str = canonical_base.native();
    if (!base_str.empty() && base_str.back() != std::filesystem::path::preferred_separator) {
      base_str += std::filesystem::path::preferred_separator;
    }
    EXT_ENFORCE(canonical_data.native().find(base_str) == 0 || canonical_data == canonical_base,
                "TensorProto::LoadExternalData: external data '", data_path.string(),
                "' resolves outside the base directory '", base_dir, "' for tensor '", ref_name(),
                "'.");
  }
  // Reject hardlinks: a hardlink to a sensitive file placed inside the model
  // directory would pass the path-traversal and symlink checks above, but
  // would allow reading arbitrary files on disk.
  {
    std::error_code ec_hc;
    auto hc = std::filesystem::hard_link_count(data_path, ec_hc);
    EXT_ENFORCE(!ec_hc, "TensorProto::LoadExternalData: could not determine hard link count for '",
                data_path.string(), "': ", ec_hc.message());
    EXT_ENFORCE(hc <= 1, "TensorProto::LoadExternalData: external data file '", data_path.string(),
                "' has multiple hard links (", static_cast<int64_t>(hc),
                "), which is not allowed for tensor '", ref_name(), "'.");
  }
  std::ifstream file(data_path, std::ios::binary);
  EXT_ENFORCE(file.is_open(), "TensorProto::LoadExternalData unable to open external data file '",
              data_path.string(), "' for tensor '", ref_name(), "'.");
  if (offset > 0) {
    file.seekg(offset, std::ios::beg);
    EXT_ENFORCE(file.good(), "TensorProto::LoadExternalData unable to seek to offset ", offset,
                " in '", data_path.string(), "'.");
  }
  if (length < 0) {
    file.seekg(0, std::ios::end);
    std::streampos file_end = file.tellg();
    EXT_ENFORCE(file_end != std::streampos(-1) && static_cast<std::streamoff>(file_end) >= 0,
                "TensorProto::LoadExternalData unable to determine size of '", data_path.string(),
                "' (tellg failed).");
    const int64_t effective_offset = offset > 0 ? offset : 0;
    EXT_ENFORCE(static_cast<int64_t>(file_end) >= effective_offset,
                "TensorProto::LoadExternalData offset ", effective_offset, " is past end of file '",
                data_path.string(), "' (size=", static_cast<int64_t>(file_end), ").");
    file.seekg(effective_offset, std::ios::beg);
    length = static_cast<int64_t>(file_end) - effective_offset;
  }
  EXT_ENFORCE(length >= 0, "TensorProto::LoadExternalData negative length=", length, " for '",
              data_path.string(), "'.");
  ref_raw_data().resize(static_cast<size_t>(length));
  if (length > 0) {
    file.read(reinterpret_cast<char *>(ref_raw_data().data()),
              static_cast<std::streamsize>(length));
    EXT_ENFORCE(file.gcount() == static_cast<std::streamsize>(length),
                "TensorProto::LoadExternalData short read from '", data_path.string(),
                "': expected ", length, " bytes, got ", static_cast<int64_t>(file.gcount()), ".");
  }
}
int64_t TensorProto::ContentHash(bool include_content) const {
  if (!include_content) {
    // Cheap metadata-only hash: element type, shape, data location and the
    // size of every payload field. Two tensors sharing this hash are candidates
    // for an exact byte comparison, nothing more.
    uint64_t seed = static_cast<uint64_t>(std::hash<int>()(static_cast<int>(data_type())));
    for (int64_t dim : dims().values()) {
      HashCombine(seed, dim);
    }
    const int location = has_data_location() ? static_cast<int>(data_location()) : 0;
    HashCombine(seed, static_cast<uint64_t>(location));
    HashCombine(seed, static_cast<uint64_t>(raw_data().size()));
    HashCombine(seed, static_cast<uint64_t>(float_data().size()));
    HashCombine(seed, static_cast<uint64_t>(int32_data().size()));
    HashCombine(seed, static_cast<uint64_t>(int64_data().size()));
    HashCombine(seed, static_cast<uint64_t>(double_data().size()));
    HashCombine(seed, static_cast<uint64_t>(uint64_data().size()));
    HashCombine(seed, static_cast<uint64_t>(string_data().size()));
    HashCombine(seed, static_cast<uint64_t>(external_data().size()));
    return static_cast<int64_t>(seed);
  }

  // Content-sensitive hash: BLAKE3 over the element type, shape, data location
  // and every payload byte. Large payloads are hashed in parallel through the
  // BLAKE3 tree (see onnx_proto/blake3), so the result is stable regardless of
  // the number of threads used.
  utils::Blake3Hasher hasher;
  auto absorb_size = [&hasher](uint64_t value) { hasher.Update(&value, sizeof(value)); };

  absorb_size(static_cast<uint64_t>(data_type()));
  for (int64_t dim : dims().values()) {
    absorb_size(dim);
  }
  absorb_size(static_cast<uint64_t>(has_data_location() ? static_cast<int>(data_location()) : 0));

  // Prefix each field with its element count so that, for example, an empty
  // float_data followed by populated int32_data cannot collide with the reverse.
  auto absorb_packed = [&hasher, &absorb_size](const auto &field) {
    absorb_size(static_cast<uint64_t>(field.size()));
    hasher.Update(field.data(), field.size() * sizeof(*field.data()));
  };

  absorb_size(static_cast<uint64_t>(raw_data().size()));
  hasher.Update(raw_data().data(), raw_data().size());
  absorb_packed(float_data());
  absorb_packed(int32_data());
  absorb_packed(int64_data());
  absorb_packed(double_data());
  absorb_packed(uint64_data());

  absorb_size(static_cast<uint64_t>(string_data().size()));
  for (const utils::String &value : string_data().values()) {
    absorb_size(static_cast<uint64_t>(value.size()));
    hasher.Update(value.data(), value.size());
  }

  absorb_size(static_cast<uint64_t>(external_data().size()));
  for (std::size_t i = 0; i < external_data().size(); ++i) {
    const std::string_view key = external_data()[i].key().sv();
    const std::string_view value = external_data()[i].value().sv();
    absorb_size(static_cast<uint64_t>(key.size()));
    hasher.Update(key.data(), key.size());
    absorb_size(static_cast<uint64_t>(value.size()));
    hasher.Update(value.data(), value.size());
  }
  return hasher.Finalize64();
}
void TensorProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(
      ss, options, NAME_EXIST_VALUE(dims), NAME_EXIST_VALUE(data_type),
      NAME_EXIST_VALUE(data_location), NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(segment),
      NAME_EXIST_VALUE(raw_data), NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(external_data),
      NAME_EXIST_VALUE(metadata_props), NAME_EXIST_VALUE(double_data), NAME_EXIST_VALUE(float_data),
      NAME_EXIST_VALUE(int32_data), NAME_EXIST_VALUE(int64_data), NAME_EXIST_VALUE(uint64_data),
      NAME_EXIST_VALUE(string_data));
}

// SparseTensorProto

IMPLEMENT_PROTO(SparseTensorProto)
SerializeSizeResult SparseTensorProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, values)
  SIZE_FIELD(size, options, stream, indices)
  SIZE_REPEATED_FIELD(size, options, stream, dims)
  return size;
}
void SparseTensorProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, values)
  WRITE_FIELD(options, stream, indices)
  WRITE_REPEATED_FIELD(options, stream, dims)
}
bool SparseTensorProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, SparseTensorProto) //
  READ_FIELD(options, stream, values)            //
  READ_FIELD(options, stream, indices)           //
  READ_REPEATED_FIELD(options, stream, dims)     //
  READ_END(options, stream, SparseTensorProto)   //
  return true;
}
void SparseTensorProto::PrintToStringStream(std::stringstream &ss,
                                            utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(values), NAME_EXIST_VALUE(indices),
                                 NAME_EXIST_VALUE(dims));
}

// TypeProto::Tensor

IMPLEMENT_PROTO(TypeProto::Tensor)
SerializeSizeResult TypeProto::Tensor::SerializeSize(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_ENUM_FIELD(size, options, stream, elem_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, shape)
  return size;
}
void TypeProto::Tensor::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_ENUM_FIELD(options, stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, shape)
}
bool TypeProto::Tensor::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::Tensor)       //
  READ_OPTIONAL_ENUM_FIELD(options, stream, elem_type) //
  READ_OPTIONAL_PROTO_FIELD(options, stream, shape)    //
  READ_END(options, stream, TypeProto::Tensor)         //
  return true;
}
void TypeProto::Tensor::PrintToStringStream(std::stringstream &ss,
                                            utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::SparseTensor

IMPLEMENT_PROTO(TypeProto::SparseTensor)
SerializeSizeResult TypeProto::SparseTensor::SerializeSize(utils::BinaryWriteStream &stream,
                                                           SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_ENUM_FIELD(size, options, stream, elem_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, shape)
  return size;
}
void TypeProto::SparseTensor::SerializeToStream(utils::BinaryWriteStream &stream,
                                                SerializeOptions &options) const {
  WRITE_ENUM_FIELD(options, stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, shape)
}
bool TypeProto::SparseTensor::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::SparseTensor) //
  READ_OPTIONAL_ENUM_FIELD(options, stream, elem_type) //
  READ_OPTIONAL_PROTO_FIELD(options, stream, shape)    //
  READ_END(options, stream, TypeProto::SparseTensor)   //
  return true;
}
void TypeProto::SparseTensor::PrintToStringStream(std::stringstream &ss,
                                                  utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::Sequence

IMPLEMENT_PROTO(TypeProto::Sequence)
SerializeSizeResult TypeProto::Sequence::SerializeSize(utils::BinaryWriteStream &stream,
                                                       SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, elem_type)
  return size;
}
void TypeProto::Sequence::SerializeToStream(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, elem_type)
}
bool TypeProto::Sequence::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::Sequence)      //
  READ_OPTIONAL_PROTO_FIELD(options, stream, elem_type) //
  READ_END(options, stream, TypeProto::Sequence)        //
  return true;
}
void TypeProto::Sequence::PrintToStringStream(std::stringstream &ss,
                                              utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(elem_type));
}

//  TypeProto::Map

IMPLEMENT_PROTO(TypeProto::Map)
SerializeSizeResult TypeProto::Map::SerializeSize(utils::BinaryWriteStream &stream,
                                                  SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, key_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, value_type)
  return size;
}
void TypeProto::Map::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, value_type)
}
bool TypeProto::Map::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::Map)            //
  READ_FIELD(options, stream, key_type)                  //
  READ_OPTIONAL_PROTO_FIELD(options, stream, value_type) //
  READ_END(options, stream, TypeProto::Map)              //
  return true;
}
void TypeProto::Map::PrintToStringStream(std::stringstream &ss,
                                         utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(key_type),
                                 NAME_EXIST_VALUE(value_type));
}

// TypeProto::Opaque

IMPLEMENT_PROTO(TypeProto::Opaque)
SerializeSizeResult TypeProto::Opaque::SerializeSize(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, name)
  return size;
}
void TypeProto::Opaque::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, domain)
  WRITE_FIELD(options, stream, name)
}
bool TypeProto::Opaque::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::Opaque) //
  READ_FIELD(options, stream, domain)            //
  READ_FIELD(options, stream, name)              //
  READ_END(options, stream, TypeProto::Opaque)   //
  return true;
}
void TypeProto::Opaque::PrintToStringStream(std::stringstream &ss,
                                            utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(name));
}

// TypeProto::Optional

IMPLEMENT_PROTO(TypeProto::Optional)
SerializeSizeResult TypeProto::Optional::SerializeSize(utils::BinaryWriteStream &stream,
                                                       SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, elem_type)
  return size;
}
void TypeProto::Optional::SerializeToStream(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, elem_type)
}
bool TypeProto::Optional::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto::Optional)      //
  READ_OPTIONAL_PROTO_FIELD(options, stream, elem_type) //
  READ_END(options, stream, TypeProto::Optional)        //
  return true;
}
void TypeProto::Optional::PrintToStringStream(std::stringstream &ss,
                                              utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(elem_type));
}

// TypeProto

IMPLEMENT_PROTO(TypeProto)
SerializeSizeResult TypeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, tensor_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sequence_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, map_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, opaque_type)
  SIZE_FIELD(size, options, stream, denotation)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sparse_tensor_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, optional_type)
  return size;
}
void TypeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                  SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sequence_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, map_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, opaque_type)
  WRITE_FIELD(options, stream, denotation)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, optional_type)
}
bool TypeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TypeProto)                         //
  READ_OPTIONAL_PROTO_FIELD(options, stream, tensor_type)        //
  READ_OPTIONAL_PROTO_FIELD(options, stream, sequence_type)      //
  READ_OPTIONAL_PROTO_FIELD(options, stream, map_type)           //
  READ_OPTIONAL_PROTO_FIELD(options, stream, opaque_type)        //
  READ_FIELD(options, stream, denotation)                        //
  READ_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_type) //
  READ_OPTIONAL_PROTO_FIELD(options, stream, optional_type)      //
  READ_END(options, stream, TypeProto)                           //
  return true;
}
void TypeProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(
      ss, options, NAME_EXIST_VALUE(tensor_type), NAME_EXIST_VALUE(sequence_type),
      NAME_EXIST_VALUE(map_type), NAME_EXIST_VALUE(opaque_type), NAME_EXIST_VALUE(denotation),
      NAME_EXIST_VALUE(sparse_tensor_type), NAME_EXIST_VALUE(optional_type));
}

// ValueInfoProto

IMPLEMENT_PROTO(ValueInfoProto)
SerializeSizeResult ValueInfoProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                  SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, type)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void ValueInfoProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, type)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}

bool ValueInfoProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, ValueInfoProto)          //
  READ_FIELD(options, stream, name)                    //
  READ_OPTIONAL_PROTO_FIELD(options, stream, type)     //
  READ_FIELD(options, stream, doc_string)              //
  READ_REPEATED_FIELD(options, stream, metadata_props) //
  READ_END(options, stream, ValueInfoProto)            //
  return true;
}
void ValueInfoProto::PrintToStringStream(std::stringstream &ss,
                                         utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(type),
                                 NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(metadata_props));
}

// AttributeProto

IMPLEMENT_PROTO(AttributeProto)
SerializeSizeResult AttributeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                  SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, ref_attr_name)
  SIZE_ENUM_FIELD(size, options, stream, type)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_FIELD(size, options, stream, f)
  SIZE_FIELD(size, options, stream, i)
  SIZE_FIELD_NULL(size, options, stream, s)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, t)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sparse_tensor)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, g)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, tp)
  SIZE_REPEATED_FIELD(size, options, stream, floats)
  SIZE_REPEATED_FIELD(size, options, stream, ints)
  SIZE_REPEATED_FIELD(size, options, stream, strings)
  SIZE_REPEATED_FIELD(size, options, stream, tensors)
  SIZE_REPEATED_FIELD(size, options, stream, sparse_tensors)
  SIZE_REPEATED_FIELD(size, options, stream, graphs)
  SIZE_REPEATED_FIELD(size, options, stream, type_protos)
  return size;
}
void AttributeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, ref_attr_name)
  WRITE_ENUM_FIELD(options, stream, type)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_FIELD(options, stream, f)
  WRITE_FIELD(options, stream, i)
  WRITE_FIELD_NULL(options, stream, s)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, t)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, g)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, tp)
  WRITE_REPEATED_FIELD(options, stream, floats)
  WRITE_REPEATED_FIELD(options, stream, ints)
  WRITE_REPEATED_FIELD(options, stream, strings)
  WRITE_REPEATED_FIELD(options, stream, tensors)
  WRITE_REPEATED_FIELD(options, stream, sparse_tensors)
  WRITE_REPEATED_FIELD(options, stream, graphs)
  WRITE_REPEATED_FIELD(options, stream, type_protos)
}
bool AttributeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, AttributeProto)               //
  READ_FIELD(options, stream, name)                         //
  READ_FIELD(options, stream, ref_attr_name)                //
  READ_ENUM_FIELD(options, stream, type)                    //
  READ_FIELD(options, stream, doc_string)                   //
  READ_FIELD(options, stream, f)                            //
  READ_FIELD(options, stream, i)                            //
  READ_FIELD(options, stream, s)                            //
  READ_OPTIONAL_PROTO_FIELD(options, stream, t)             //
  READ_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor) //
  READ_OPTIONAL_PROTO_FIELD(options, stream, g)             //
  READ_OPTIONAL_PROTO_FIELD(options, stream, tp)            //
  READ_REPEATED_FIELD(options, stream, floats)              //
  READ_REPEATED_FIELD(options, stream, ints)                //
  READ_REPEATED_FIELD(options, stream, strings)             //
  READ_REPEATED_FIELD(options, stream, tensors)             //
  READ_REPEATED_FIELD(options, stream, sparse_tensors)      //
  READ_REPEATED_FIELD(options, stream, graphs)              //
  READ_REPEATED_FIELD(options, stream, type_protos)         //
  READ_END(options, stream, AttributeProto)                 //
  return true;
}
void AttributeProto::PrintToStringStream(std::stringstream &ss,
                                         utils::PrintOptions &options) const {
  switch (type_) {
  case AttributeType::UNDEFINED:
    ss << "{ " << name_ << ": UNDEFINED }";
    return;
  case AttributeType::FLOAT:
    ss << "{ " << std::string(name_) << ": " << (has_f() ? MakeString(*f_) : "?") << " }";
    return;
  case AttributeType::INT:
    ss << "{ " << std::string(name_) << ": " << (has_i() ? MakeString(*i_) : "?") << " }";
    return;
  case AttributeType::STRING:
    ss << "{ " << name_ << ": " << s_ << " }";
    return;
  case AttributeType::FLOATS:
    ss << "{ " << name_ << ": " << write_as_string(options, floats_) << " }";
    return;
  case AttributeType::INTS:
    ss << "{ " << name_ << ": " << write_as_string(options, ints_) << " }";
    return;
  case AttributeType::STRINGS:
    ss << "{ " << name_ << ": " << write_as_string(options, strings_) << " }";
    return;
  default:
    write_proto_into_vector_string(
        ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(ref_attr_name),
        NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(type), NAME_EXIST_VALUE(f),
        NAME_EXIST_VALUE(i), NAME_EXIST_VALUE(s), NAME_EXIST_VALUE(t),
        NAME_EXIST_VALUE(sparse_tensor), NAME_EXIST_VALUE(g), NAME_EXIST_VALUE(floats),
        NAME_EXIST_VALUE(ints), NAME_EXIST_VALUE(strings), NAME_EXIST_VALUE(tensors),
        NAME_EXIST_VALUE(sparse_tensors), NAME_EXIST_VALUE(graphs), NAME_EXIST_VALUE(tp),
        NAME_EXIST_VALUE(type_protos));
  }
}

// NodeProto

IMPLEMENT_PROTO(NodeProto)
SerializeSizeResult NodeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, op_type)
  SIZE_REPEATED_FIELD(size, options, stream, attribute)
  SIZE_FIELD_NULL(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, overload)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, device_configurations)
  return size;
}
void NodeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                  SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, op_type)
  WRITE_REPEATED_FIELD(options, stream, attribute)
  WRITE_FIELD_NULL(options, stream, domain)
  WRITE_FIELD(options, stream, overload)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, device_configurations)
}
bool NodeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, NodeProto)                      //
  READ_REPEATED_FIELD(options, stream, input)                 //
  READ_REPEATED_FIELD(options, stream, output)                //
  READ_FIELD(options, stream, name)                           //
  READ_FIELD(options, stream, op_type)                        //
  READ_REPEATED_FIELD(options, stream, attribute)             //
  READ_FIELD(options, stream, domain)                         //
  READ_FIELD(options, stream, overload)                       //
  READ_FIELD(options, stream, doc_string)                     //
  READ_REPEATED_FIELD(options, stream, metadata_props)        //
  READ_REPEATED_FIELD(options, stream, device_configurations) //
  READ_END(options, stream, NodeProto)                        //
  return true;
}
void NodeProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(
      ss, options, NAME_EXIST_VALUE(input), NAME_EXIST_VALUE(output), NAME_EXIST_VALUE(name),
      NAME_EXIST_VALUE(op_type), NAME_EXIST_VALUE(attribute), NAME_EXIST_VALUE(domain),
      NAME_EXIST_VALUE(overload), NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(metadata_props),
      NAME_EXIST_VALUE(device_configurations));
}

// GraphProto

IMPLEMENT_PROTO(GraphProto)
SerializeSizeResult GraphProto::SerializeSize(utils::BinaryWriteStream &stream,
                                              SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_REPEATED_FIELD(size, options, stream, node)
  SIZE_FIELD(size, options, stream, name)
  SIZE_REPEATED_FIELD(size, options, stream, initializer)
  SIZE_REPEATED_FIELD(size, options, stream, sparse_initializer)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_REPEATED_FIELD(size, options, stream, value_info)
  SIZE_REPEATED_FIELD(size, options, stream, quantization_annotation)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void GraphProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                   SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, node)
  WRITE_FIELD(options, stream, name)
  WRITE_REPEATED_FIELD(options, stream, initializer)
  WRITE_REPEATED_FIELD(options, stream, sparse_initializer)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_REPEATED_FIELD(options, stream, value_info)
  WRITE_REPEATED_FIELD(options, stream, quantization_annotation)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}
bool GraphProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, GraphProto)                       //
  READ_REPEATED_FIELD(options, stream, node)                    //
  READ_FIELD(options, stream, name)                             //
  READ_REPEATED_FIELD(options, stream, initializer)             //
  READ_REPEATED_FIELD(options, stream, sparse_initializer)      //
  READ_FIELD(options, stream, doc_string)                       //
  READ_REPEATED_FIELD(options, stream, input)                   //
  READ_REPEATED_FIELD(options, stream, output)                  //
  READ_REPEATED_FIELD(options, stream, value_info)              //
  READ_REPEATED_FIELD(options, stream, quantization_annotation) //
  READ_REPEATED_FIELD(options, stream, metadata_props)          //
  READ_END(options, stream, GraphProto)                         //  // NOLINT
  return true;
}
void GraphProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(
      ss, options, NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(input),
      NAME_EXIST_VALUE(output), NAME_EXIST_VALUE(metadata_props), NAME_EXIST_VALUE(node),
      NAME_EXIST_VALUE(initializer), NAME_EXIST_VALUE(sparse_initializer),
      NAME_EXIST_VALUE(value_info), NAME_EXIST_VALUE(quantization_annotation));
}

// FunctionProto

IMPLEMENT_PROTO(FunctionProto)
SerializeSizeResult FunctionProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_REPEATED_FIELD(size, options, stream, attribute)
  SIZE_REPEATED_FIELD(size, options, stream, attribute_proto)
  SIZE_REPEATED_FIELD(size, options, stream, node)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, opset_import)
  SIZE_FIELD_NULL(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, overload)
  SIZE_REPEATED_FIELD(size, options, stream, value_info)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void FunctionProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                      SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_REPEATED_FIELD(options, stream, attribute)
  WRITE_REPEATED_FIELD(options, stream, attribute_proto)
  WRITE_REPEATED_FIELD(options, stream, node)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, opset_import)
  WRITE_FIELD_NULL(options, stream, domain)
  WRITE_FIELD(options, stream, overload)
  WRITE_REPEATED_FIELD(options, stream, value_info)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}
bool FunctionProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, FunctionProto)            //
  READ_FIELD(options, stream, name)                     //
  READ_REPEATED_FIELD(options, stream, input)           //
  READ_REPEATED_FIELD(options, stream, output)          //
  READ_REPEATED_FIELD(options, stream, attribute)       //
  READ_REPEATED_FIELD(options, stream, attribute_proto) //
  READ_REPEATED_FIELD(options, stream, node)            //
  READ_FIELD(options, stream, doc_string)               //
  READ_REPEATED_FIELD(options, stream, opset_import)    //
  READ_FIELD(options, stream, domain)                   //
  READ_FIELD(options, stream, overload)                 //
  READ_REPEATED_FIELD(options, stream, value_info)      //
  READ_REPEATED_FIELD(options, stream, metadata_props)  //
  READ_END(options, stream, FunctionProto)              //  // NOLINT
  return true;
}
void FunctionProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(domain),
                                 NAME_EXIST_VALUE(overload), NAME_EXIST_VALUE(doc_string),
                                 NAME_EXIST_VALUE(input), NAME_EXIST_VALUE(output),
                                 NAME_EXIST_VALUE(opset_import), NAME_EXIST_VALUE(attribute),
                                 NAME_EXIST_VALUE(attribute_proto), NAME_EXIST_VALUE(node),
                                 NAME_EXIST_VALUE(value_info), NAME_EXIST_VALUE(metadata_props));
}

// ModelProto

IMPLEMENT_PROTO(ModelProto)
SerializeSizeResult ModelProto::SerializeSize(utils::BinaryWriteStream &stream,
                                              SerializeOptions &options) const {
  if (options.raw_data_callback) {
    ModelProto copy;
    copy.CopyFrom(*this);
    ApplySerializeRawDataCallback(copy, options);
    SerializeOptions local_opts = options;
    local_opts.raw_data_callback = {};
    return copy.SerializeSize(stream, local_opts);
  }
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, ir_version)
  SIZE_REPEATED_FIELD(size, options, stream, opset_import)
  SIZE_FIELD(size, options, stream, producer_name)
  SIZE_FIELD(size, options, stream, producer_version)
  SIZE_FIELD(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, model_version)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, graph)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, functions)
  SIZE_REPEATED_FIELD(size, options, stream, configuration)
  return size;
}
void ModelProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                   SerializeOptions &options) const {
  WRITE_FIELD(options, stream, ir_version)
  WRITE_REPEATED_FIELD(options, stream, opset_import)
  WRITE_FIELD(options, stream, producer_name)
  WRITE_FIELD(options, stream, producer_version)
  WRITE_FIELD(options, stream, domain)
  WRITE_FIELD(options, stream, model_version)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, graph)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, functions)
  WRITE_REPEATED_FIELD(options, stream, configuration)
}
bool ModelProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, ModelProto)              //
  READ_FIELD(options, stream, ir_version)              //
  READ_REPEATED_FIELD(options, stream, opset_import)   //
  READ_FIELD(options, stream, producer_name)           //
  READ_FIELD(options, stream, producer_version)        //
  READ_FIELD(options, stream, domain)                  //
  READ_FIELD(options, stream, model_version)           //
  READ_FIELD(options, stream, doc_string)              //
  READ_OPTIONAL_PROTO_FIELD(options, stream, graph)    //
  READ_REPEATED_FIELD(options, stream, metadata_props) //
  READ_REPEATED_FIELD(options, stream, functions)      //
  READ_REPEATED_FIELD(options, stream, configuration)  //
  READ_END(options, stream, ModelProto)                //  // NOLINT
  return true;
}
void ModelProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(ir_version),
                                 NAME_EXIST_VALUE(opset_import), NAME_EXIST_VALUE(producer_name),
                                 NAME_EXIST_VALUE(producer_version), NAME_EXIST_VALUE(domain),
                                 NAME_EXIST_VALUE(model_version), NAME_EXIST_VALUE(doc_string),
                                 NAME_EXIST_VALUE(graph), NAME_EXIST_VALUE(metadata_props),
                                 NAME_EXIST_VALUE(functions), NAME_EXIST_VALUE(configuration));
}

bool ModelProto::SerializeToString(std::string &out,
                                   std::unordered_map<std::string, std::string> &external_files,
                                   size_t max_external_file_size,
                                   const std::string &external_file_prefix) const {
  SerializeOptions opts;
  return SerializeToString(out, external_files, max_external_file_size, external_file_prefix, opts);
}

bool ModelProto::SerializeToString(std::string &out,
                                   std::unordered_map<std::string, std::string> &external_files,
                                   size_t max_external_file_size,
                                   const std::string &external_file_prefix,
                                   const SerializeOptions &opts) const {
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "ModelProto::SerializeToString: SerializeFormat::kOrtFlatbuffers is not "
              "implemented yet. Use SerializeFormat::kOnnx for now.");
  ModelProto copy;
  copy.CopyFrom(*this);
  SerializeOptions local_opts = opts;
  if (local_opts.raw_data_callback) {
    ApplySerializeRawDataCallback(copy, local_opts);
    local_opts.raw_data_callback = {};
  }
  local_opts.num_threads = 1;
  local_opts.use_external_data_location = true;
  AssignExternalDataChunks(copy, static_cast<size_t>(local_opts.raw_data_threshold),
                           max_external_file_size, external_file_prefix, local_opts.alignment);
  MemoryExternalWriteStream stream;
  SerializeSizeResult total_size = copy.SerializeSize(stream, local_opts);
  if (!EnforceMaxSerializedSize(total_size, local_opts, "ModelProto::SerializeToString")) {
    out.clear();
    external_files.clear();
    return false;
  }
  stream.pre_allocate_main(static_cast<size_t>(total_size.proto_size));
  copy.SerializeToStream(stream, local_opts);
  stream.CopyOutputsTo(out, external_files);
  return true;
}

// SequenceProto

IMPLEMENT_PROTO(SequenceProto)
SerializeSizeResult SequenceProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_ENUM_FIELD(size, options, stream, elem_type)
  SIZE_REPEATED_FIELD(size, options, stream, tensor_values)
  SIZE_REPEATED_FIELD(size, options, stream, sparse_tensor_values)
  SIZE_REPEATED_FIELD(size, options, stream, sequence_values)
  SIZE_REPEATED_FIELD(size, options, stream, map_values)
  SIZE_REPEATED_FIELD(size, options, stream, optional_values)
  return size;
}
void SequenceProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                      SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_ENUM_FIELD(options, stream, elem_type)
  WRITE_REPEATED_FIELD(options, stream, tensor_values)
  WRITE_REPEATED_FIELD(options, stream, sparse_tensor_values)
  WRITE_REPEATED_FIELD(options, stream, sequence_values)
  WRITE_REPEATED_FIELD(options, stream, map_values)
  WRITE_REPEATED_FIELD(options, stream, optional_values)
}
bool SequenceProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, SequenceProto)                 //
  READ_FIELD(options, stream, name)                          //
  READ_ENUM_FIELD(options, stream, elem_type)                //
  READ_REPEATED_FIELD(options, stream, tensor_values)        //
  READ_REPEATED_FIELD(options, stream, sparse_tensor_values) //
  READ_REPEATED_FIELD(options, stream, sequence_values)      //
  READ_REPEATED_FIELD(options, stream, map_values)           //
  READ_REPEATED_FIELD(options, stream, optional_values)      //
  READ_END(options, stream, SequenceProto)                   // NOLINT
  return true;
}
void SequenceProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(elem_type),
                                 NAME_EXIST_VALUE(tensor_values),
                                 NAME_EXIST_VALUE(sparse_tensor_values),
                                 NAME_EXIST_VALUE(sequence_values), NAME_EXIST_VALUE(map_values),
                                 NAME_EXIST_VALUE(optional_values));
}

// MapProto

IMPLEMENT_PROTO(MapProto)
SerializeSizeResult MapProto::SerializeSize(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_ENUM_FIELD(size, options, stream, key_type)
  SIZE_REPEATED_FIELD(size, options, stream, keys)
  SIZE_REPEATED_FIELD(size, options, stream, string_keys)
  SIZE_FIELD(size, options, stream, values)
  return size;
}
void MapProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                 SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_ENUM_FIELD(options, stream, key_type)
  WRITE_REPEATED_FIELD(options, stream, keys)
  WRITE_REPEATED_FIELD(options, stream, string_keys)
  WRITE_FIELD(options, stream, values)
}
bool MapProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, MapProto)             //
  READ_FIELD(options, stream, name)                 //
  READ_ENUM_FIELD(options, stream, key_type)        //
  READ_REPEATED_FIELD(options, stream, keys)        //
  READ_REPEATED_FIELD(options, stream, string_keys) //
  READ_FIELD(options, stream, values)               //
  READ_END(options, stream, MapProto)               // NOLINT
  return true;
}
void MapProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(key_type),
                                 NAME_EXIST_VALUE(keys), NAME_EXIST_VALUE(string_keys),
                                 NAME_EXIST_VALUE(values));
}

// OptionalProto

IMPLEMENT_PROTO(OptionalProto)
SerializeSizeResult OptionalProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  SerializeSizeResult size;
  SIZE_FIELD(size, options, stream, name)
  SIZE_ENUM_FIELD(size, options, stream, elem_type)
  SIZE_FIELD(size, options, stream, tensor_value)
  SIZE_FIELD(size, options, stream, sparse_tensor_value)
  SIZE_FIELD(size, options, stream, sequence_value)
  SIZE_FIELD(size, options, stream, map_value)
  SIZE_FIELD(size, options, stream, optional_value)
  return size;
}
void OptionalProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                      SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)           //
  WRITE_ENUM_FIELD(options, stream, elem_type) //
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, tensor_value)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_value)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sequence_value)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, map_value)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, optional_value)
}
bool OptionalProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, OptionalProto)                      //
  READ_FIELD(options, stream, name)                               //
  READ_ENUM_FIELD(options, stream, elem_type)                     //
  READ_OPTIONAL_PROTO_FIELD(options, stream, tensor_value)        //
  READ_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_value) //
  READ_OPTIONAL_PROTO_FIELD(options, stream, sequence_value)      //
  READ_OPTIONAL_PROTO_FIELD(options, stream, map_value)           //
  READ_OPTIONAL_PROTO_FIELD(options, stream, optional_value)      //
  READ_END(options, stream, OptionalProto)                        // NOLINT
  return true;
}
void OptionalProto::PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const {
  write_proto_into_vector_string(ss, options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(elem_type),
                                 NAME_EXIST_VALUE(tensor_value),
                                 NAME_EXIST_VALUE(sparse_tensor_value),
                                 NAME_EXIST_VALUE(sequence_value), NAME_EXIST_VALUE(map_value),
                                 NAME_EXIST_VALUE(optional_value));
}

// Convenience builder methods for proto classes.
// These keep the proto-native overloads (replace-by-name for attributes,
// idempotent-by-key for metadata, ``(domain, version)`` for opsets).

namespace {

// Sets metadata property *key* to *value* on a metadata_props field, updating
// an existing entry with the same key in place, and returns a reference to it.
template <typename RepeatedT>
StringStringEntryProto &SetOrAddMetadataEntry(RepeatedT &metadata_props, const std::string &key,
                                              const std::string &value) {
  for (size_t i = 0; i < metadata_props.size(); ++i) {
    if (std::string(metadata_props[i].ref_key()) == key) {
      metadata_props[i].set_value(value);
      return metadata_props[i];
    }
  }
  StringStringEntryProto entry;
  entry.set_key(key);
  entry.set_value(value);
  metadata_props.push_back(entry);
  return metadata_props.back();
}

// Appends an OperatorSetIdProto ``(domain, version)`` to *opset_import* and
// returns a reference to the new entry.
template <typename RepeatedT>
OperatorSetIdProto &AddOpsetEntry(RepeatedT &opset_import, const std::string &domain,
                                  int64_t version) {
  OperatorSetIdProto opset;
  opset.set_domain(domain);
  opset.set_version(version);
  opset_import.push_back(opset);
  return opset_import.back();
}

} // namespace

AttributeProto &NodeProto::set_attribute(const AttributeProto &attr) {
  const std::string name = attr.ref_name();
  for (size_t i = 0; i < attribute_.size(); ++i) {
    if (std::string(attribute_[i].ref_name()) == name) {
      attribute_[i] = attr;
      return attribute_[i];
    }
  }
  attribute_.push_back(attr);
  return attribute_.back();
}

StringStringEntryProto &NodeProto::add_metadata(const std::string &key, const std::string &value) {
  return SetOrAddMetadataEntry(metadata_props_, key, value);
}

std::string NodeProto::Signature(const std::vector<std::string> &resolved_inputs) const {
  std::ostringstream signature;
  signature << op_type().value() << '\x1f'
            << NormaliseDomain(domain().empty() ? std::string() : domain().value()) << '\x1f';
  for (const std::string &input : resolved_inputs) {
    signature << input << '\x1e';
  }
  signature << '\x1f';
  std::vector<std::string> attributes;
  attributes.reserve(static_cast<std::size_t>(attribute().size()));
  for (const auto &attr : attribute()) {
    attributes.push_back(attr.SerializeAsString());
  }
  std::sort(attributes.begin(), attributes.end());
  for (const std::string &attr : attributes) {
    signature << attr << '\x1d';
  }
  return signature.str();
}

NodeProto &GraphProto::add_node(const std::string &op_type, const std::vector<std::string> &inputs,
                                const std::vector<std::string> &outputs, const std::string &domain,
                                const std::string &name) {
  node_.push_back(MakeNode(op_type.c_str(), inputs, outputs,
                           domain.empty() ? nullptr : domain.c_str(),
                           name.empty() ? nullptr : name.c_str()));
  return node_.back();
}

StringStringEntryProto &GraphProto::add_metadata(const std::string &key, const std::string &value) {
  return SetOrAddMetadataEntry(metadata_props_, key, value);
}

NodeProto &FunctionProto::add_node(const std::string &op_type,
                                   const std::vector<std::string> &inputs,
                                   const std::vector<std::string> &outputs,
                                   const std::string &domain, const std::string &name) {
  node_.push_back(MakeNode(op_type.c_str(), inputs, outputs,
                           domain.empty() ? nullptr : domain.c_str(),
                           name.empty() ? nullptr : name.c_str()));
  return node_.back();
}

OperatorSetIdProto &FunctionProto::add_opset(const std::string &domain, int64_t version) {
  return AddOpsetEntry(opset_import_, domain, version);
}

StringStringEntryProto &FunctionProto::add_metadata(const std::string &key,
                                                    const std::string &value) {
  return SetOrAddMetadataEntry(metadata_props_, key, value);
}

FunctionProto &ModelProto::add_function(const FunctionProto &function) {
  functions_.push_back(function);
  return functions_.back();
}

OperatorSetIdProto &ModelProto::add_opset(const std::string &domain, int64_t version) {
  return AddOpsetEntry(opset_import_, domain, version);
}

StringStringEntryProto &ModelProto::add_metadata(const std::string &key, const std::string &value) {
  return SetOrAddMetadataEntry(metadata_props_, key, value);
}

} // namespace ONNX_LIGHT_NAMESPACE
