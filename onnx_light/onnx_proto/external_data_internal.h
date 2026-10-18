#pragma once

#include "onnx_light_helpers.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::utils::detail {

// Exported only for the checker in lib_onnx_lib; this header is not installed.
ONNX_LIGHT_PROTO_API std::shared_ptr<int> OpenExternalDataFile(const std::string &base_dir,
                                                               const std::string &relative_path,
                                                               bool write = false);
inline std::string external_path_to_utf8(const std::filesystem::path &value) {
  const auto utf8 = value.u8string();
  return std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size());
}
int64_t ExternalDataFileSize(int descriptor);
void ReadExternalDataFile(int descriptor, void *destination, int64_t size, int64_t offset);
std::shared_ptr<uint8_t> MapExternalDataFile(int descriptor);
std::filesystem::path relative_external_data_path(const std::filesystem::path &candidate_path,
                                                  const std::filesystem::path &base_dir);

} // namespace ONNX_LIGHT_NAMESPACE::utils::detail
