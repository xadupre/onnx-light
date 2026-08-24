// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "path.h"
#include "proto_utils.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Loads a serialized protobuf message from a filesystem path.
 *
 * Opens the file at @p proto_path in binary mode, reads its full contents, and
 * parses them into @p proto with ParseProtoFromBytes().
 *
 * @tparam T Protobuf-like message type accepted by ParseProtoFromBytes().
 * @param proto_path UTF-8 path to the serialized protobuf file.
 * @param proto Destination message that receives the parsed file contents.
 *
 * @throws std::runtime_error Throws when the file cannot be opened, cannot be
 * read, or cannot be parsed as the requested protobuf type.
 */
template <typename T> void LoadProtoFromPath(const std::string &proto_path, T &proto) {
  std::filesystem::path proto_u8_path = utf8_to_path(proto_path);
  std::fstream proto_stream(proto_u8_path, std::ios::in | std::ios::binary);
  if (!proto_stream.is_open()) {
    ONNX_THROW("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto.");
  }
  std::error_code size_error;
  const std::uintmax_t file_size = std::filesystem::file_size(proto_u8_path, size_error);
  std::string data;
  bool read_ok = false;
  if (!size_error) {
    data.resize(file_size);
    proto_stream.read(data.data(), static_cast<std::streamsize>(file_size));
    read_ok = static_cast<std::uintmax_t>(proto_stream.gcount()) == file_size;
  } else {
    data.assign(std::istreambuf_iterator<char>{proto_stream}, std::istreambuf_iterator<char>{});
    read_ok = !proto_stream.bad();
  }
  if (!read_ok) {
    ONNX_THROW("Unable to read proto file: ", proto_path, ". Please check if it is a valid proto.");
  }
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    ONNX_THROW("Unable to parse proto from file: ", proto_path,
               ". Please check if it is a valid protobuf file of proto.");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
