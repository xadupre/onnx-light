// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "path.h"
#include "proto_utils.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace ONNX_NAMESPACE {

template <typename T> void LoadProtoFromPath(const std::string &proto_path, T &proto) {
  std::filesystem::path proto_u8_path = utf8_to_path(proto_path);
  std::fstream proto_stream(proto_u8_path, std::ios::in | std::ios::binary);
  if (!proto_stream.is_open()) {
    ONNX_THROW("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto.");
  }
  std::string data{std::istreambuf_iterator<char>{proto_stream}, std::istreambuf_iterator<char>{}};
  if (proto_stream.bad()) {
    ONNX_THROW("Unable to read proto file: ", proto_path, ". Please check if it is a valid proto.");
  }
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    ONNX_THROW("Unable to parse proto from file: ", proto_path,
               ". Please check if it is a valid protobuf file of proto.");
  }
}

} // namespace ONNX_NAMESPACE
