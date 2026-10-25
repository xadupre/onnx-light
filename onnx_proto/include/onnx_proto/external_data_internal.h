#pragma once

#include "onnx_light_helpers.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
class TensorProto;
struct ParseOptions;
namespace utils {
class BinaryStream;
}
} // namespace ONNX_LIGHT_NAMESPACE

namespace ONNX_LIGHT_NAMESPACE::utils::detail {

inline constexpr char kExternalDataLocationKey[] = "location";
inline constexpr char kExternalDataOffsetKey[] = "offset";
inline constexpr char kExternalDataLengthKey[] = "length";
inline constexpr char kExternalDataEncryptionKey[] = "onnx_light.encryption";
inline constexpr char kExternalDataDigestKey[] = "onnx_light.encrypted_sha256";
inline constexpr char kExternalDataEncryptionFormat[] = "ONNXCRY3";

// Exported only for the checker in lib_onnx_lib; this header is not installed.
ONNX_LIGHT_PROTO_API std::shared_ptr<int> OpenExternalDataFile(const std::string &base_dir,
                                                               const std::string &relative_path,
                                                               bool write = false);
inline std::string external_path_to_utf8(const std::filesystem::path &value) {
  const auto utf8 = value.u8string();
  return std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size());
}
inline std::filesystem::path external_path_from_utf8(const std::string &value) {
#ifdef __cpp_char8_t
  return std::filesystem::path(reinterpret_cast<const char8_t *>(value.c_str()));
#else
  return std::filesystem::u8path(value);
#endif
}
int64_t ExternalDataFileSize(int descriptor);
std::string ExternalDataFileIdentity(int descriptor, bool include_changes = true);
std::string ValidateDeferredWeightParse(BinaryStream &stream, const ParseOptions &options);
void DeferInlineWeight(TensorProto &tensor, BinaryStream &stream, int64_t offset, int64_t length,
                       const ParseOptions &options);
void FinishDeferredWeight(TensorProto &tensor, BinaryStream &stream, const ParseOptions &options);
void ReadExternalDataFile(int descriptor, void *destination, int64_t size, int64_t offset);
std::shared_ptr<uint8_t> MapExternalDataFile(int descriptor);
std::filesystem::path relative_external_data_path(const std::filesystem::path &candidate_path,
                                                  const std::filesystem::path &base_dir);

} // namespace ONNX_LIGHT_NAMESPACE::utils::detail
