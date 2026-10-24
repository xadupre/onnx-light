#include "external_data_internal.h"
#include "onnx.h"
#include "onnx_helper.h"
#include <charconv>
#include <filesystem>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE {

struct DeferredWeightSource {
  std::string base_dir;
  std::string location;
  std::string identity;
  std::string file_key;
  int64_t offset;
  int64_t length;
  int64_t alignment;
};

namespace {

using utils::detail::external_path_from_utf8;
using utils::detail::external_path_to_utf8;
using utils::detail::ExternalDataFileIdentity;
using utils::detail::ExternalDataFileSize;
using utils::detail::OpenExternalDataFile;

std::filesystem::path SourcePath(utils::BinaryStream &stream) {
  if (const auto *file = dynamic_cast<utils::FileStream *>(&stream))
    return std::filesystem::absolute(external_path_from_utf8(file->file_path()));
  if (const auto *file = dynamic_cast<utils::MmapFileStream *>(&stream))
    return std::filesystem::absolute(external_path_from_utf8(file->file_path()));
  EXT_THROW("defer_weights requires an ONNX file-backed stream.");
}

int SourceDescriptor(utils::BinaryStream &stream) {
  if (const auto *file = dynamic_cast<utils::FileStream *>(&stream))
    return *file->external_file();
  return *dynamic_cast<utils::MmapFileStream &>(stream).deferred_file();
}

int64_t ParseOffset(const utils::OptionalString &value) {
  int64_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  EXT_ENFORCE(result.ec == std::errc() && result.ptr == value.data() + value.size() && parsed >= 0,
              "Invalid deferred weight offset/length: '", value, "'.");
  return parsed;
}

void ValidatePayload(const TensorProto &tensor, int64_t length) {
  EXT_ENFORCE(!tensor.has_segment(), "Deferred weights do not support segmented tensors.");
  const uint32_t bits = FixedBitWidth(tensor.data_type());
  EXT_ENFORCE(bits > 0, "Deferred weights require a fixed-width raw tensor type.");
  uint64_t count =
      std::find(tensor.dims().begin(), tensor.dims().end(), 0) == tensor.dims().end() ? 1 : 0;
  for (const int64_t dim : tensor.dims()) {
    EXT_ENFORCE(dim >= 0 && (dim == 0 || count <= std::numeric_limits<uint64_t>::max() /
                                                      static_cast<uint64_t>(dim)),
                "Deferred weight shape is invalid or overflows for '", tensor.str_name(), "'.");
    count *= static_cast<uint64_t>(dim);
  }
  EXT_ENFORCE(count <= (std::numeric_limits<uint64_t>::max() - 7) / bits,
              "Deferred weight byte size overflows for '", tensor.str_name(), "'.");
  const uint64_t expected = (count * bits + 7) / 8;
  EXT_ENFORCE(length >= 0 && static_cast<uint64_t>(length) == expected,
              "Deferred weight payload size mismatch for '", tensor.str_name(), "': ", length,
              " bytes, expected ", expected, ".");
}

TensorProto &FindWeight(const ModelProto &model, const std::string &name) {
  TensorProto *found = nullptr;
  IteratorTensorProto iterator(const_cast<ModelProto *>(&model));
  while (iterator.next()) {
    if (iterator->str_name() == name) {
      EXT_ENFORCE(found == nullptr, "Ambiguous weight name '", name,
                  "': multiple tensors match across model scopes.");
      found = &*iterator;
    }
  }
  EXT_ENFORCE(found != nullptr, "Unknown weight name '", name, "'.");
  return *found;
}

} // namespace

namespace utils::detail {

std::string ValidateDeferredWeightParse(BinaryStream &stream, const ParseOptions &options) {
  if (!options.defer_weights)
    return {};
  EXT_ENFORCE_LIMIT(options.max_tensor_size_bytes >= 0,
                    "defer_weights requires max_tensor_size_bytes >= 0.");
  EXT_ENFORCE(options.format == SerializeFormat::kOnnx && !stream.ExternalWeights() &&
                  !options.no_copy && options.tiny_external_data_threshold < 0,
              "defer_weights requires ONNX input, no explicit external stream, no_copy=false, "
              "and tiny_external_data_threshold < 0.");
  (void)SourcePath(stream);
  if (auto *file = dynamic_cast<FileStream *>(&stream))
    file->PinDeferredSource();
  else
    dynamic_cast<MmapFileStream &>(stream).PinDeferredSource();
  return ExternalDataFileIdentity(SourceDescriptor(stream));
}

void DeferInlineWeight(TensorProto &tensor, BinaryStream &stream, int64_t offset, int64_t length,
                       const ParseOptions &options) {
  const auto path = SourcePath(stream);
  auto source = std::make_shared<DeferredWeightSource>();
  source->base_dir = external_path_to_utf8(path.parent_path());
  source->location = external_path_to_utf8(path.filename());
  source->offset = offset;
  source->length = length;
  source->alignment = options.alignment;
  const auto file = OpenExternalDataFile(source->base_dir, source->location);
  const int64_t file_size = ExternalDataFileSize(*file);
  EXT_ENFORCE(offset >= 0 && length >= 0 && offset <= file_size && length <= file_size - offset,
              "Deferred inline weight range is outside its backing file.");
  source->identity = ExternalDataFileIdentity(*file);
  EXT_ENFORCE(source->identity == ExternalDataFileIdentity(SourceDescriptor(stream)),
              "Deferred model backing file changed while parsing.");
  source->file_key = ExternalDataFileIdentity(*file, false);
  tensor.deferred_weight_source = std::move(source);
  tensor.ref_raw_data() = ByteSpan{};
}

void FinishDeferredWeight(TensorProto &tensor, BinaryStream &stream, const ParseOptions &options) {
  if (tensor.has_data_location() && tensor.data_location() == TensorProto::EXTERNAL) {
    EXT_ENFORCE(!tensor.HasDeferredWeight() && !tensor.has_raw_data(),
                "Deferred external tensors must not also contain inline raw_data.");
    auto source = std::make_shared<DeferredWeightSource>();
    source->base_dir = external_path_to_utf8(SourcePath(stream).parent_path());
    source->offset = 0;
    source->length = -1;
    source->alignment = options.alignment;
    bool has_offset = false, has_length = false, has_location = false;
    for (const auto &entry : tensor.external_data()) {
      EXT_ENFORCE(entry.key() != kExternalDataEncryptionKey,
                  "Deferred loading of encrypted external weights is unsupported.");
      if (entry.key() == "location") {
        EXT_ENFORCE(!has_location, "Duplicate deferred external location.");
        has_location = true;
        source->location = entry.value();
      } else if (entry.key() == "offset") {
        EXT_ENFORCE(!has_offset, "Duplicate deferred external offset.");
        has_offset = true;
        source->offset = ParseOffset(entry.value());
      } else if (entry.key() == "length" || entry.key() == "size") {
        EXT_ENFORCE(!has_length, "Duplicate deferred external length/size.");
        has_length = true;
        source->length = ParseOffset(entry.value());
      }
    }
    EXT_ENFORCE(has_location && IsSafeExternalDataLocation(source->location),
                "Unsafe or missing deferred external weight location.");
    const auto file = OpenExternalDataFile(source->base_dir, source->location);
    const int64_t file_size = ExternalDataFileSize(*file);
    EXT_ENFORCE(source->offset <= file_size, "Deferred external weight offset is out of bounds.");
    if (!has_length)
      source->length = file_size - source->offset;
    EXT_ENFORCE(source->length <= file_size - source->offset,
                "Deferred external weight length is out of bounds.");
    source->identity = ExternalDataFileIdentity(*file);
    source->file_key = ExternalDataFileIdentity(*file, false);
    tensor.deferred_weight_source = std::move(source);
  }
  if (!tensor.HasDeferredWeight() && FixedBitWidth(tensor.data_type()) > 0 &&
      std::find(tensor.dims().begin(), tensor.dims().end(), 0) != tensor.dims().end()) {
    DeferInlineWeight(tensor, stream, stream.tell(), 0, options);
  }
  if (tensor.HasDeferredWeight()) {
    ValidatePayload(tensor, tensor.deferred_weight_source->length);
    EXT_ENFORCE(static_cast<uint64_t>(tensor.deferred_weight_source->length) <=
                    std::numeric_limits<size_t>::max(),
                "Deferred weight exceeds the addressable allocation size.");
    EXT_ENFORCE_LIMIT(options.max_tensor_size_bytes == 0 ||
                          tensor.deferred_weight_source->length <= options.max_tensor_size_bytes,
                      "Deferred weight exceeds max_tensor_size_bytes.");
    onnx_light_helpers::ValidateAlignmentOption(options.alignment, "ParseOptions.alignment");
  }
}

} // namespace utils::detail

bool TensorProto::LoadDeferredWeight() {
  EXT_ENFORCE(HasDeferredWeight(), "Weight '", str_name(), "' has no recoverable backing source.");
  if (has_raw_data())
    return false;
  const auto &source = *deferred_weight_source;
  ValidatePayload(*this, source.length);
  const auto file = OpenExternalDataFile(source.base_dir, source.location);
  EXT_ENFORCE(ExternalDataFileIdentity(*file) == source.identity,
              "Deferred weight backing file changed: ", source.location);
  utils::ByteSpan payload;
  if (source.alignment > 1)
    payload.resize_aligned(static_cast<size_t>(source.length),
                           static_cast<size_t>(source.alignment));
  else
    payload.resize(static_cast<size_t>(source.length));
  utils::detail::ReadExternalDataFile(*file, payload.data(), source.length, source.offset);
  EXT_ENFORCE(ExternalDataFileIdentity(*file) == source.identity,
              "Deferred weight backing file changed while reading: ", source.location);
  raw_data_ = std::move(payload);
  return true;
}

bool TensorProto::UnloadDeferredWeight() {
  EXT_ENFORCE(HasDeferredWeight(), "Weight '", str_name(), "' has no recoverable backing source.");
  const bool loaded = has_raw_data();
  raw_data_ = utils::ByteSpan{};
  return loaded;
}

std::vector<std::string> ModelProto::WeightNames() const {
  std::vector<std::string> names;
  IteratorTensorProto iterator(const_cast<ModelProto *>(this));
  while (iterator.next())
    names.push_back(iterator->str_name());
  return names;
}

bool ModelProto::HasDeferredWeights() const {
  IteratorTensorProto iterator(const_cast<ModelProto *>(this));
  while (iterator.next())
    if (iterator->HasDeferredWeight())
      return true;
  return false;
}

bool ModelProto::IsWeightLoaded(const std::string &name) const {
  const auto &tensor = FindWeight(*this, name);
  return tensor.has_raw_data() || tensor.has_float_data() || tensor.has_double_data() ||
         tensor.has_int32_data() || tensor.has_int64_data() || tensor.has_uint64_data() ||
         tensor.has_string_data();
}

bool ModelProto::LoadWeight(const std::string &name) {
  return FindWeight(*this, name).LoadDeferredWeight();
}

bool ModelProto::UnloadWeight(const std::string &name) {
  return FindWeight(*this, name).UnloadDeferredWeight();
}

size_t ModelProto::LoadWeights() {
  size_t loaded = 0;
  IteratorTensorProto iterator(this);
  while (iterator.next())
    if (iterator->HasDeferredWeight())
      loaded += iterator->LoadDeferredWeight();
  return loaded;
}

size_t ModelProto::UnloadWeights() {
  size_t unloaded = 0;
  IteratorTensorProto iterator(this);
  while (iterator.next())
    if (iterator->HasDeferredWeight())
      unloaded += iterator->UnloadDeferredWeight();
  return unloaded;
}

void ModelProto::ValidateWeightSavePath(const std::string &path) const {
  IteratorTensorProto iterator(const_cast<ModelProto *>(this));
  while (iterator.next()) {
    if (!iterator->HasDeferredWeight())
      continue;
    EXT_ENFORCE(iterator->has_raw_data(), "Cannot serialize unloaded deferred weight '",
                iterator->str_name(), "'. Call LoadWeights() first.");
    if (path.empty())
      continue;
    const auto &source = *iterator->deferred_weight_source;
    const auto backing =
        external_path_from_utf8(source.base_dir) / external_path_from_utf8(source.location);
    const auto target = std::filesystem::absolute(external_path_from_utf8(path));
    EXT_ENFORCE(std::filesystem::weakly_canonical(target) !=
                        std::filesystem::weakly_canonical(backing) &&
                    (!std::filesystem::exists(target) || !std::filesystem::exists(backing) ||
                     !std::filesystem::equivalent(target, backing)),
                "Cannot overwrite a deferred weight backing source: ", path);
  }
}

bool ModelProto::SerializeToFile(const std::string &path, SerializeOptions options) const {
  ValidateWeightSavePath(path);
  EXT_ENFORCE(!HasDeferredWeights() || options.format == SerializeFormat::kOnnx,
              "Models retaining deferred sources support single-file ONNX export only.");
  if (HasDeferredWeights()) {
    const auto destination = std::filesystem::absolute(external_path_from_utf8(path));
    auto file = OpenExternalDataFile(external_path_to_utf8(destination.parent_path()),
                                     external_path_to_utf8(destination.filename()), true);
    ValidateWeightSaveDescriptor(*file);
    utils::FileWriteStream stream(path, std::move(file));
    return SerializeModelProtoToStream(const_cast<ModelProto &>(*this), stream, options);
  }
  utils::FileWriteStream stream(path);
  return SerializeModelProtoToStream(const_cast<ModelProto &>(*this), stream, options);
}

void ModelProto::ValidateWeightSaveDescriptor(int descriptor) const {
  if (!HasDeferredWeights())
    return;
  ValidateWeightSavePath();
  const std::string identity = ExternalDataFileIdentity(descriptor, false);
  IteratorTensorProto iterator(const_cast<ModelProto *>(this));
  while (iterator.next()) {
    if (!iterator->HasDeferredWeight())
      continue;
    const auto &source = *iterator->deferred_weight_source;
    EXT_ENFORCE(identity != source.file_key,
                "Cannot write to a deferred weight backing source descriptor.");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
